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
TRAIL_ENGINE_VERSION = "17.0-live-safe-invalidation-path-brain"
CONFIDENCE_MODEL_VERSION = "17.0-data-path-calibrated-v1"

MAIN_ENTRY_MAX_ATR = 1.60
MIN_DISPLACEMENT_ATR = 0.30
ENTRY_LOOKBACK = 24
SWING_LB = 3
POI_LOOKBACK = 96
HTF_POI_LOOKBACK = 120
ENTRY_MIN_RISK_ATR = 0.55
ENTRY_MAX_RISK_ATR = 2.20
ENTRY_MIN_RISK_PCT = 0.08
ENTRY_MAX_RISK_PCT = 3.50
ENTRY_TARGET_RR = 2.60

TRAIL_ARM_R = 0.80
TRAIL_MFE_ARM_R = 0.90
TRAIL_MFE_EXTENDED = 1.50
TRAIL_MFE_DEEP = 2.50
TRAIL_GIVEBACK_WARN = 0.30
TRAIL_GIVEBACK_STRONG = 0.50
TRAIL_GIVEBACK_CRITICAL = 0.70
TRAIL_LOCK_WARN_R = 0.05
TRAIL_LOCK_STRONG_R = 0.20
TRAIL_LOCK_CRITICAL_R = 0.45
TRAIL_MIN_UPDATE_R = 0.05
TRAIL_MAX_CHURN = 5
TRAIL_STRUCT_BUFFER_ATR = 0.34
TRAIL_RETRACE_BUFFER_ATR = 0.30
TRAIL_MIN_MARKET_GAP_ATR = 0.45
TRAIL_REVERSAL_BODY_ATR = 0.85
TRAIL_COUNTER_BODY_ATR = 0.55
TRAIL_VOLUME_EXHAUSTION = 0.72
TRAIL_VOLUME_COUNTER = 1.20
TRAIL_PEAK_LOOKBACK = 48

# V17 research/path intelligence. History is soft evidence only.
HISTORY_MIN_MATCH = 6
HISTORY_MFE_GOOD_R = 1.00
HISTORY_MFE_BAD_R = 0.30
HISTORY_GIVEBACK_BAD = 0.70
TARGET_HIST_MFE_PREFERRED = 0.65
TARGET_HIST_MFE_STRONG = 0.80
SL_BUFFER_ATR = 0.22
SL_MIN_MARKET_GAP_ATR = 0.12
TRAIL_REVALIDATE_GAP_ATR = 0.45
TRAIL_DEEP_GIVEBACK_R = 0.70
TRAIL_REVERSAL_CONFIRM_MIN = 3


def _num(v, default=None):
    try:
        x = float(v)
        return default if not np.isfinite(x) else x
    except Exception:
        return default


def _clip(v, lo=0.0, hi=100.0):
    try:
        return float(np.clip(float(v), lo, hi))
    except Exception:
        return float(lo)


def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def rsi(s, n=14):
    d = s.diff()
    g = d.clip(lower=0).rolling(n).mean()
    l = (-d.clip(upper=0)).rolling(n).mean()
    out = 100 - 100 / (1 + g / l.replace(0, np.nan))
    out = out.mask((l <= 0) & (g > 0), 100.0)
    out = out.mask((g <= 0) & (l > 0), 0.0)
    out = out.fillna(50.0)
    return out


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
    idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    boundary = pd.Timestamp.now(tz="UTC").floor(f"{minutes}min")
    return out.loc[idx < boundary].copy() if len(idx) and idx[-1] >= boundary else out


def build_df(df, interval_minutes=None):
    if df is None or len(df) < 60:
        return None
    x = df.copy()
    if interval_minutes:
        x = _closed_candles(x, interval_minutes)
    if x is None or len(x) < 60:
        return None
    for col in ("open", "high", "low", "close", "volume"):
        if col not in x.columns:
            return None
        x[col] = pd.to_numeric(x[col], errors="coerce")
    x["ema9"] = ema(x["close"], 9)
    x["ema21"] = ema(x["close"], 21)
    x["ema50"] = ema(x["close"], 50)
    x["ema200"] = ema(x["close"], 200) if len(x) >= 200 else ema(x["close"], 50)
    x["rsi"] = rsi(x["close"])
    x["atr"] = atr_fn(x)
    x["vol_sma"] = x["volume"].rolling(20).mean()
    x["range"] = x["high"] - x["low"]
    x["body"] = (x["close"] - x["open"]).abs()
    return x.dropna()


def swing_pts(df, lb=5):
    sh, sl = [], []
    if df is None or len(df) < lb * 2 + 3:
        return sh, sl
    hi, lo = df["high"].to_numpy(dtype=float), df["low"].to_numpy(dtype=float)
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


def _pct_move(df, bars):
    if df is None or len(df) <= bars:
        return 0.0
    a, b = float(df["close"].iloc[-bars - 1]), float(df["close"].iloc[-1])
    return (b - a) / max(abs(a), 1e-12) * 100.0


def _efficiency(df, bars=24):
    if df is None or len(df) <= bars:
        return 0.5
    sub = df.tail(bars)
    net = abs(float(sub["close"].iloc[-1] - sub["close"].iloc[0]))
    gross = float(sub["close"].diff().abs().sum())
    return _clip(net / max(gross, 1e-12), 0.0, 1.0)


def _direction_context(h1, m15, d1, market_context):
    sh1, sl1 = swing_pts(h1, 5)
    sh15, sl15 = swing_pts(m15, 5)
    struct_h1 = _market_structure(h1, sh1, sl1)
    struct_m15 = _market_structure(m15, sh15, sl15)
    d1_bias = "neutral"
    if d1 is not None and len(d1) >= 60:
        shd, sld = swing_pts(d1, 3)
        sd1 = _market_structure(d1, shd, sld)
        bull = sd1 == "bullish" or bool(d1["ema9"].iloc[-1] > d1["ema21"].iloc[-1] > d1["ema50"].iloc[-1])
        bear = sd1 == "bearish" or bool(d1["ema9"].iloc[-1] < d1["ema21"].iloc[-1] < d1["ema50"].iloc[-1])
        d1_bias = "bullish" if bull and not bear else "bearish" if bear and not bull else "neutral"

    ema_bull = bool(h1["ema9"].iloc[-1] > h1["ema21"].iloc[-1] > h1["ema50"].iloc[-1])
    ema_bear = bool(h1["ema9"].iloc[-1] < h1["ema21"].iloc[-1] < h1["ema50"].iloc[-1])
    atr = max(float(m15["atr"].iloc[-1]), float(h1["atr"].iloc[-1]) / 4, float(m15["close"].iloc[-1]) * 0.0025)
    fast = (float(m15["close"].iloc[-1]) - float(m15["close"].iloc[-4])) / atr
    slow = (float(m15["close"].iloc[-1]) - float(m15["close"].iloc[-12])) / atr
    eff = _efficiency(h1, 24)

    bull, bear = 50.0, 50.0
    if d1_bias == "bullish": bull += 18
    if d1_bias == "bearish": bear += 18
    if struct_h1 == "bullish": bull += 22
    if struct_h1 == "bearish": bear += 22
    if ema_bull: bull += 8
    if ema_bear: bear += 8
    if fast > 0: bull += min(8.0, fast * 3.0)
    if fast < 0: bear += min(8.0, -fast * 3.0)
    if slow > 0: bull += min(8.0, slow * 1.6)
    if slow < 0: bear += min(8.0, -slow * 1.6)
    if struct_m15 == "bullish": bull += 5
    if struct_m15 == "bearish": bear += 5

    mc = market_context if isinstance(market_context, dict) else {}
    breadth = _num(mc.get("bullish_breadth_pct"), None)
    if breadth is not None:
        if breadth >= 62: bull += 7
        elif breadth <= 38: bear += 7
    rs = _num(mc.get("relative_strength_1h_pct"), None)
    if rs is not None:
        if rs >= 0.30: bull += 5
        elif rs <= -0.30: bear += 5
    rv = _num(mc.get("relative_volume"), _num(mc.get("avg_relative_volume"), None))
    if rv is not None and rv >= 1.20:
        if fast > 0: bull += 2.5
        elif fast < 0: bear += 2.5

    regime = str(mc.get("market_regime") or mc.get("chart_regime") or "").lower()
    macro = "unknown"
    if "bull" in regime: macro = "bullish"
    elif "bear" in regime: macro = "bearish"
    elif "range" in regime or "compression" in regime: macro = "ranging"
    elif "transition" in regime: macro = "transition"
    if macro == "bullish":
        bull += 4
        bear -= 2
    elif macro == "bearish":
        bear += 4
        bull -= 2
    elif macro == "transition":
        bull -= 3
        bear -= 3

    direction = "bull" if bull >= bear else "bear"
    edge = abs(bull - bear)
    quality = _clip(42 + edge * 1.45 + max(0.0, eff - 0.5) * 18)
    htf_conflict = d1_bias in ("bullish", "bearish") and struct_h1 in ("bullish", "bearish") and d1_bias != struct_h1
    htf_alignment = not htf_conflict and (d1_bias == "neutral" or struct_h1 == "ranging" or d1_bias == struct_h1)
    return {
        "direction": direction,
        "bull": round(bull, 2), "bear": round(bear, 2),
        "direction_quality": round(quality, 2), "edge": round(edge, 2),
        "struct_h1": struct_h1, "struct_m15": struct_m15, "d1_bias": d1_bias,
        "macro_bias": macro, "htf_alignment": htf_alignment, "htf_conflict": htf_conflict,
        "atr": atr, "price": float(m15["close"].iloc[-1]), "efficiency_h1": eff,
        "sh1": sh1, "sl1": sl1, "sh15": sh15, "sl15": sl15,
    }


def score_direction(df_h1, df_m15, df_d1=None, df_btc_h1=None):
    h1, m15 = build_df(df_h1, 60), build_df(df_m15, 15)
    if h1 is None or m15 is None:
        return None
    d1 = build_df(df_d1, 1440) if df_d1 is not None else None
    ctx = _direction_context(h1, m15, d1, {})
    ctx.update({"h1": h1, "m15": m15, "d1": d1})
    return ctx


def _zone_fresh(df, idx, top, bot, direction):
    if idx >= len(df) - 2:
        return True
    sub = df.iloc[idx + 2:]
    return not bool((sub["close"] < bot).any()) if direction == "bull" else not bool((sub["close"] > top).any())


def _find_ob(df, direction, lookback=POI_LOOKBACK):
    sub = df.tail(lookback).reset_index(drop=True)
    base = len(df) - len(sub)
    med_body = max(float(sub["body"].median()), 1e-12)
    out = []
    for i in range(1, len(sub) - 4):
        c, nxt = sub.iloc[i], sub.iloc[i + 1]
        if direction == "bull" and not (c["close"] < c["open"] and nxt["close"] > nxt["open"]):
            continue
        if direction == "bear" and not (c["close"] > c["open"] and nxt["close"] < nxt["open"]):
            continue
        body_atr = float(nxt["body"]) / max(float(nxt["atr"]), 1e-12)
        if float(nxt["body"]) < med_body * 1.15 or body_atr < MIN_DISPLACEMENT_ATR:
            continue
        top, bot = float(max(c["open"], c["close"])), float(min(c["open"], c["close"]))
        idx = base + i
        if not _zone_fresh(df, idx, top, bot, direction):
            continue
        score = 35.0 + min(35.0, body_atr * 18.0)
        if i + 2 < len(sub):
            c2 = sub.iloc[i + 2]
            if direction == "bull" and c2["low"] > c["high"]: score += 15
            if direction == "bear" and c2["high"] < c["low"]: score += 15
        age = len(df) - 1 - idx
        score += max(0.0, 18.0 - age * 0.45)
        out.append({"top": top, "bot": bot, "mid": (top + bot) / 2, "idx": idx, "score": _clip(score), "kind": "ob", "age": age, "body_atr": round(body_atr, 3)})
    out.sort(key=lambda z: (-z["score"], -z["idx"]))
    return out[:6]


def _find_fvg(df, direction, lookback=POI_LOOKBACK):
    sub = df.tail(lookback).reset_index(drop=True)
    base = len(df) - len(sub)
    out = []
    for i in range(1, len(sub) - 1):
        a, c = sub.iloc[i - 1], sub.iloc[i + 1]
        if direction == "bull" and c["low"] > a["high"]:
            top, bot = float(c["low"]), float(a["high"])
        elif direction == "bear" and c["high"] < a["low"]:
            top, bot = float(a["low"]), float(c["high"])
        else:
            continue
        idx = base + i + 1
        if not _zone_fresh(df, idx, top, bot, direction):
            continue
        width_atr = (top - bot) / max(float(df["atr"].iloc[-1]), 1e-12)
        age = len(df) - 1 - idx
        score = _clip(28 + min(40, width_atr * 22) + max(0, 18 - age * 0.40))
        out.append({"top": top, "bot": bot, "mid": (top + bot) / 2, "idx": idx, "score": score, "kind": "fvg", "age": age, "width_atr": round(width_atr, 3)})
    out.sort(key=lambda z: (-z["score"], -z["idx"]))
    return out[:6]


def _sweep(df, direction, sh, sl):
    atr = max(float(df["atr"].iloc[-1]), 1e-12)
    if direction == "bull" and sl:
        level = float(df["low"].iloc[sl[-1]])
        lo, close = float(df["low"].iloc[-1]), float(df["close"].iloc[-1])
        if lo < level and close > level:
            return {"detected": True, "type": "sell_side_reclaim", "level": level, "strength": _clip(48 + (level - lo) / atr * 28)}
    if direction == "bear" and sh:
        level = float(df["high"].iloc[sh[-1]])
        hi, close = float(df["high"].iloc[-1]), float(df["close"].iloc[-1])
        if hi > level and close < level:
            return {"detected": True, "type": "buy_side_reclaim", "level": level, "strength": _clip(48 + (hi - level) / atr * 28)}
    return {"detected": False, "type": "none", "level": None, "strength": 0.0}


def _displacement(df, direction):
    if len(df) < 8:
        return {"confirmed": False, "body_atr": 0.0, "range_atr": 0.0, "break_strength": 0.0, "follow_through": 0.0}
    a = df.iloc[-1]
    atr = max(float(a["atr"]), 1e-12)
    body_atr = float(a["body"]) / atr
    range_atr = float(a["range"]) / atr
    prior = df.iloc[-6:-1]
    if direction == "bull":
        broke = float(a["close"]) > float(prior["high"].max())
        aligned = float(a["close"]) > float(a["open"])
    else:
        broke = float(a["close"]) < float(prior["low"].min())
        aligned = float(a["close"]) < float(a["open"])
    rv = float(a["volume"]) / max(float(a["vol_sma"]), 1e-12)
    follow = _clip((body_atr - 0.25) * 35 + (rv - 0.8) * 22 + (range_atr - 0.5) * 10, 0, 100)
    confirmed = aligned and broke and body_atr >= MIN_DISPLACEMENT_ATR
    strength = _clip(body_atr * 35 + max(0, rv - 0.9) * 24 + (20 if broke else 0))
    return {"confirmed": bool(confirmed), "body_atr": round(body_atr, 3), "range_atr": round(range_atr, 3), "break_strength": round(strength, 2), "follow_through": round(follow, 2), "relative_volume": round(rv, 3)}


def _pullback(df, direction, atr):
    sub = df.tail(14)
    gross = float(sub["close"].diff().abs().sum())
    net = float(sub["close"].iloc[-1] - sub["close"].iloc[0])
    if direction == "bear": net = -net
    eff = abs(net) / max(gross, 1e-12)
    if direction == "bull":
        adverse = max(0.0, float(sub["high"].iloc[0] - sub["low"].min()))
    else:
        adverse = max(0.0, float(sub["high"].max() - sub["low"].iloc[0]))
    depth = adverse / max(atr, 1e-12)
    score = 54.0
    if 0.20 <= depth <= 1.70: score += 16
    elif depth > 2.10: score -= 24
    if eff >= 0.50: score += 18
    elif eff < 0.22: score -= 18
    aligned_last = (direction == "bull" and sub["close"].iloc[-1] > sub["close"].iloc[-2]) or (direction == "bear" and sub["close"].iloc[-1] < sub["close"].iloc[-2])
    if aligned_last: score += 7
    return {"score": _clip(score), "state": "clean" if score >= 74 else "mixed" if score >= 48 else "damaged", "efficiency": round(eff, 3), "depth_atr": round(depth, 3)}


def _location(df, direction, price):
    look = df.tail(ENTRY_LOOKBACK)
    lo, hi = float(look["low"].min()), float(look["high"].max())
    width = max(hi - lo, 1e-12)
    pos = (price - lo) / width
    if direction == "bull":
        score = 85 if pos <= 0.50 else 72 if pos <= 0.68 else 48 if pos <= 0.82 else 28
    else:
        score = 85 if pos >= 0.50 else 72 if pos >= 0.32 else 48 if pos >= 0.18 else 28
    return {"score": float(score), "range_position": round(pos, 4), "range_low": lo, "range_high": hi, "state": "preferred" if score >= 75 else "acceptable" if score >= 48 else "late"}


def _context_quality(mc, direction):
    if not isinstance(mc, dict) or not mc:
        return {"score": 50.0, "conflicts": ["cross_market_context_unavailable"], "reasons": [], "available": False}
    score, reasons, conflicts = 50.0, [], []
    breadth = _num(mc.get("bullish_breadth_pct"), None)
    if breadth is not None:
        aligned = breadth >= 60 if direction == "bull" else breadth <= 40
        opposed = breadth <= 40 if direction == "bull" else breadth >= 60
        if aligned: score += 14; reasons.append("breadth_aligned")
        elif opposed: score -= 18; conflicts.append("breadth_opposed")
    rs = _num(mc.get("relative_strength_1h_pct"), None)
    if rs is not None:
        if (direction == "bull" and rs >= 0.30) or (direction == "bear" and rs <= -0.30): score += 12; reasons.append("relative_strength_aligned")
        elif (direction == "bull" and rs <= -0.30) or (direction == "bear" and rs >= 0.30): score -= 13; conflicts.append("relative_strength_opposed")
    rs4 = _num(mc.get("relative_strength_4h_pct"), None)
    if rs4 is not None:
        if (direction == "bull" and rs4 >= 0.50) or (direction == "bear" and rs4 <= -0.50): score += 8; reasons.append("relative_strength_4h_aligned")
        elif (direction == "bull" and rs4 <= -0.50) or (direction == "bear" and rs4 >= 0.50): score -= 8; conflicts.append("relative_strength_4h_opposed")
    rv = _num(mc.get("relative_volume"), _num(mc.get("avg_relative_volume"), None))
    if rv is not None:
        if rv >= 1.25: score += 7; reasons.append("participation_expanded")
        elif rv <= 0.55: score -= 6; conflicts.append("participation_thin")
    regime = str(mc.get("market_regime") or mc.get("chart_regime") or "").lower()
    if "transition" in regime: score -= 8; conflicts.append("transition_regime")
    elif "trend" in regime or "expansion" in regime: score += 4; reasons.append("trend_regime")
    elif "range" in regime or "compression" in regime: score -= 2; reasons.append("range_regime")
    return {"score": _clip(score), "conflicts": conflicts, "reasons": reasons, "available": True}


def _poi_quality(zone, htf_overlap, displacement, sweep, pullback):
    base = float(zone.get("score", 25.0))
    kind = zone.get("kind", "market")
    quality = 0.55 * base + 45.0
    quality = min(90.0, quality)
    if htf_overlap: quality += 8
    if kind == "fvg" and not displacement["confirmed"]: quality -= 12
    if kind == "ob" and displacement["confirmed"]: quality += 5
    if sweep["detected"]: quality += 5
    if pullback["state"] == "damaged": quality -= 12
    age = float(zone.get("age", 999))
    if age > 45: quality -= 8
    return _clip(quality)


def _geometry(m15, h1, direction, entry, atr):
    """Build structural invalidation SL with bounded downside.

    Prefer the NEAREST structural anchor that still gives the setup enough room.
    The old implementation used the farthest valid swing (min for BUY/max for
    SELL), which could make initial risk unnecessarily large. V17 treats SL as
    a true thesis-invalidation boundary plus a volatility buffer.
    """
    sh, sl = swing_pts(m15, SWING_LB)
    hsh, hsl = swing_pts(h1, 5)
    if direction == "bull":
        anchors = []
        if sl:
            anchors.append((float(m15["low"].iloc[sl[-1]]), "m15_swing"))
        if hsl:
            anchors.append((float(h1["low"].iloc[hsl[-1]]), "h1_swing"))
        valid = sorted([(x, lbl) for x, lbl in anchors if x < entry], key=lambda z: entry - z[0])
        if not valid:
            return None
        # Pick the closest anchor that still clears the minimum risk.
        selected = None
        for anchor, label in valid:
            risk0 = entry - (anchor - atr * SL_BUFFER_ATR)
            if risk0 >= atr * ENTRY_MIN_RISK_ATR:
                selected = (anchor, label)
                break
        if selected is None:
            selected = valid[-1]
        anchor, anchor_label = selected
        stop = anchor - atr * SL_BUFFER_ATR
        # Never allow a stop to sit materially above/at the entry.
        stop = min(stop, entry - atr * SL_MIN_MARKET_GAP_ATR)
    else:
        anchors = []
        if sh:
            anchors.append((float(m15["high"].iloc[sh[-1]]), "m15_swing"))
        if hsh:
            anchors.append((float(h1["high"].iloc[hsh[-1]]), "h1_swing"))
        valid = sorted([(x, lbl) for x, lbl in anchors if x > entry], key=lambda z: z[0] - entry)
        if not valid:
            return None
        selected = None
        for anchor, label in valid:
            risk0 = (anchor + atr * SL_BUFFER_ATR) - entry
            if risk0 >= atr * ENTRY_MIN_RISK_ATR:
                selected = (anchor, label)
                break
        if selected is None:
            selected = valid[-1]
        anchor, anchor_label = selected
        stop = anchor + atr * SL_BUFFER_ATR
        stop = max(stop, entry + atr * SL_MIN_MARKET_GAP_ATR)

    risk = (entry - stop) if direction == "bull" else (stop - entry)
    if risk <= 0:
        return None
    risk_atr = risk / max(atr, 1e-12)
    risk_pct = risk / max(entry, 1e-12) * 100.0
    if not (ENTRY_MIN_RISK_ATR <= risk_atr <= ENTRY_MAX_RISK_ATR and ENTRY_MIN_RISK_PCT <= risk_pct <= ENTRY_MAX_RISK_PCT):
        return None
    return float(stop), float(risk), float(risk_atr), float(risk_pct), sh, sl, hsh, hsl, anchor_label


def _targets(m15, h1, direction, entry, risk):
    vals = []
    if direction == "bull":
        sh, _ = swing_pts(h1, 5)
        for i in reversed(sh[-10:]):
            x = float(h1["high"].iloc[i])
            if x > entry: vals.append((x, "h1_swing"))
        vals += [(entry + risk * (1 + FIB_EXT_1), "fib_1.272"), (entry + risk * (1 + FIB_EXT_2), "fib_1.618")]
    else:
        _, sl = swing_pts(h1, 5)
        for i in reversed(sl[-10:]):
            x = float(h1["low"].iloc[i])
            if x < entry: vals.append((x, "h1_swing"))
        vals += [(entry - risk * (1 + FIB_EXT_1), "fib_1.272"), (entry - risk * (1 + FIB_EXT_2), "fib_1.618")]
    valid = []
    for x, label in vals:
        rr = ((x - entry) if direction == "bull" else (entry - x)) / max(risk, 1e-12)
        if rr >= MIN_RR:
            valid.append((x, label, rr))
    if not valid: return None
    valid.sort(key=lambda z: abs(z[2] - ENTRY_TARGET_RR))
    tp, label, rr = valid[0]
    return float(tp), label, float(rr)


def _archetype(direction, zone, sweep, displacement, pullback, htf_overlap, market_regime):
    k = zone.get("kind", "market")
    if sweep["detected"] and displacement["confirmed"]:
        return "LIQUIDITY_SWEEP_RECLAIM"
    if htf_overlap and k == "ob" and pullback["state"] == "clean" and displacement["confirmed"]:
        return "HTF_OB_PULLBACK_CONTINUATION"
    if k == "fvg" and displacement["confirmed"]:
        return "FVG_DISPLACEMENT_RETEST"
    if "range" in str(market_regime).lower() and sweep["detected"]:
        return "RANGE_LIQUIDITY_REVERSAL"
    if displacement["confirmed"] and pullback["state"] == "clean":
        return "STRUCTURE_CONTINUATION"
    return "CONTEXTUAL_PULLBACK"


def _candidate_quality(ctx, zone, htf_overlap, sweep, displacement, pullback, location, context, geom, rr, archetype, data_quality):
    stop, risk, risk_atr, risk_pct = geom[:4]
    poi = _poi_quality(zone, htf_overlap, displacement, sweep, pullback)
    direction_q = float(ctx["direction_quality"])
    htf_q = 90.0 if ctx["htf_alignment"] else 42.0 if ctx["htf_conflict"] else 60.0
    trigger_q = 46.0
    if displacement["confirmed"]: trigger_q += 34
    else: trigger_q += min(14, displacement["break_strength"] * 0.18)
    if sweep["detected"]: trigger_q += min(10, sweep["strength"] * 0.10)
    if pullback["state"] == "clean": trigger_q += 8
    elif pullback["state"] == "damaged": trigger_q -= 22
    trigger_q = _clip(trigger_q)

    risk_q = 76.0
    if risk_atr < 0.70: risk_q -= 18
    elif 0.85 <= risk_atr <= 1.60: risk_q += 10
    elif risk_atr > 1.90: risk_q -= 14
    if risk_pct < 0.10: risk_q -= 8
    risk_q = _clip(risk_q)

    reward_q = _clip(52 + min(28, max(0, rr - 2.0) * 13))
    if rr > 5.5: reward_q -= 4

    comps = {
        "htf": htf_q,
        "direction": direction_q,
        "setup": trigger_q,
        "poi": poi,
        "location": float(location["score"]),
        "risk": risk_q,
        "reward": reward_q,
        "context": float(context["score"]),
    }
    weights = {"htf": 0.16, "direction": 0.15, "setup": 0.20, "poi": 0.15, "location": 0.10, "risk": 0.10, "reward": 0.06, "context": 0.08}
    quality = sum(comps[k] * weights[k] for k in comps)

    contradictions = []
    if ctx["htf_conflict"]: contradictions.append("d1_h1_conflict")
    if location["state"] == "late": contradictions.append("late_entry_location")
    if pullback["state"] == "damaged": contradictions.append("damaged_pullback")
    if not displacement["confirmed"]: contradictions.append("no_fresh_displacement")
    contradictions += list(context.get("conflicts", []))[:4]
    if zone.get("kind") == "fvg" and not displacement["confirmed"]: contradictions.append("fvg_without_confirmation")
    if risk_q < 45: contradictions.append("weak_risk_geometry")

    penalty = min(30.0, len(set(contradictions)) * 4.2)
    quality = _clip(quality - penalty)

    if sweep["detected"] and displacement["confirmed"]:
        quality += 4.0
    if htf_overlap and zone.get("kind") == "ob" and pullback["state"] == "clean":
        quality += 3.0
    if archetype == "CONTEXTUAL_PULLBACK":
        quality -= 4.0
    quality = _clip(quality)

    # History is deliberately soft: it ranks similar path behaviour without
    # turning a small sample into a hard gate.
    hist = globals().get("_ACTIVE_HISTORY_CONTEXT") or {}
    hist_adj, hist_reasons = _history_adjustment(hist)
    quality = _clip(quality + hist_adj)
    contradictions.extend([r for r in hist_reasons if r != "history_mfe_healthy" and r != "history_positive_expectancy_hint"])

    quality *= max(0.82, min(1.0, data_quality / 100.0))
    confidence = int(np.clip(round(20 + quality * 0.76), 25, 94))
    if quality < 44: confidence = min(confidence, 52)
    elif quality < 54: confidence = min(confidence, 59)
    return {
        "quality": round(quality, 2), "confidence": confidence,
        "components": {k: round(v, 2) for k, v in comps.items()},
        "contradictions": list(dict.fromkeys(contradictions)),
        "penalty": round(penalty, 2), "risk_atr": round(risk_atr, 3), "risk_pct": round(risk_pct, 4),
        "poi_quality": round(poi, 2), "trigger_quality": round(trigger_q, 2), "risk_quality": round(risk_q, 2),
        "data_quality": round(data_quality, 2),
    }


def _data_quality(h1, m15, d1, market_context):
    score = 100.0
    if d1 is None: score -= 8
    if not isinstance(market_context, dict) or not market_context: score -= 8
    if len(h1) < 120: score -= 4
    if len(m15) < 100: score -= 4
    return _clip(score)


def _history_context(history, direction, entry_label, archetype):
    """Turn closed-trade history into soft, path-aware evidence.

    Never hard-veto from history; small samples only adjust ranking/confidence.
    """
    rows = [r for r in (history or []) if isinstance(r, dict)]
    side = "BUY" if direction == "bull" else "SELL"
    matched = [r for r in rows if str(r.get("decision", "")).upper() == side and str(r.get("entry_label", "")) == entry_label]
    if len(matched) < HISTORY_MIN_MATCH:
        matched = [r for r in rows if str(r.get("decision", "")).upper() == side]
    if len(matched) < HISTORY_MIN_MATCH:
        matched = rows

    def f(r, k, d=0.0):
        x = _num(r.get(k), d)
        return float(x if x is not None else d)

    final_r = [f(r, "final_r") for r in matched]
    mfe = [f(r, "mfe_r") for r in matched]
    give = []
    for r in matched:
        m = max(0.0, f(r, "mfe_r"))
        c = f(r, "current_r", f(r, "final_r"))
        give.append(max(0.0, m-c) / max(m, 0.25) if m > 0 else 0.0)
    med_mfe = float(np.median(mfe)) if mfe else 0.0
    med_final = float(np.median(final_r)) if final_r else 0.0
    immediate_fail = float(np.mean([x < HISTORY_MFE_BAD_R and y < 0 for x, y in zip(mfe, final_r)])) if matched else 0.0
    deep_give = float(np.mean([x >= HISTORY_GIVEBACK_BAD for x in give])) if matched else 0.0
    return {
        "samples": len(rows),
        "matched": len(matched),
        "win_rate": round(float(np.mean([x > 0 for x in final_r])) if final_r else 0.0, 3),
        "median_final_r": round(med_final, 3),
        "median_mfe_r": round(med_mfe, 3),
        "immediate_failure_rate": round(immediate_fail, 3),
        "deep_giveback_rate": round(deep_give, 3),
        "archetype": archetype,
        "usage": "soft_path_evidence",
    }

def _history_adjustment(hist):
    if not isinstance(hist, dict) or hist.get("matched", 0) < HISTORY_MIN_MATCH:
        return 0.0, ["history_insufficient"]
    adj = 0.0; reasons = []
    med_mfe = float(hist.get("median_mfe_r", 0.0) or 0.0)
    fail = float(hist.get("immediate_failure_rate", 0.0) or 0.0)
    final = float(hist.get("median_final_r", 0.0) or 0.0)
    if med_mfe >= HISTORY_MFE_GOOD_R:
        adj += 4.0; reasons.append("history_mfe_healthy")
    if fail >= 0.55:
        adj -= 5.0; reasons.append("history_immediate_failure_risk")
    if final > 0.15:
        adj += 2.0; reasons.append("history_positive_expectancy_hint")
    elif final < -0.20:
        adj -= 2.0; reasons.append("history_negative_expectancy_hint")
    return float(np.clip(adj, -7.0, 6.0)), reasons

def _target_history_factor(hist, rr):
    if not isinstance(hist, dict) or hist.get("matched", 0) < HISTORY_MIN_MATCH:
        return 0.0, "history_insufficient"
    med_mfe = float(hist.get("median_mfe_r", 0.0) or 0.0)
    if med_mfe <= 0:
        return 0.0, "history_no_mfe"
    reach = med_mfe / max(rr, 1e-9)
    if reach >= TARGET_HIST_MFE_STRONG:
        return 5.0, "historical_mfe_supports_target"
    if reach >= TARGET_HIST_MFE_PREFERRED:
        return 2.5, "historical_mfe_partially_supports_target"
    return -4.0, "historical_mfe_below_target_distance"


def full_analyze(df_h1, df_m15, df_d1=None, symbol=None, df_btc_h1=None, trade_history=None, market_context=None):
    try:
        h1, m15 = build_df(df_h1, 60), build_df(df_m15, 15)
        if h1 is None or m15 is None:
            return None
        d1 = build_df(df_d1, 1440) if df_d1 is not None else None
        ctx = _direction_context(h1, m15, d1, market_context)
        direction, price, atr = ctx["direction"], ctx["price"], ctx["atr"]
        sweep = _sweep(m15, direction, ctx["sh15"], ctx["sl15"])
        displacement = _displacement(m15, direction)
        pullback = _pullback(m15, direction, atr)
        location = _location(m15, direction, price)
        context = _context_quality(market_context, direction)
        data_quality = _data_quality(h1, m15, d1, market_context)
        regime = str((market_context or {}).get("market_regime") or (market_context or {}).get("chart_regime") or ctx["macro_bias"])

        zones = _find_ob(m15, direction) + _find_fvg(m15, direction)
        zones += _find_ob(h1, direction, HTF_POI_LOOKBACK) + _find_fvg(h1, direction, HTF_POI_LOOKBACK)
        zones = [z for z in zones if abs(price - float(z["mid"])) <= atr * MAIN_ENTRY_MAX_ATR]
        zones.sort(key=lambda z: (-float(z.get("score", 0)), -int(z.get("idx", 0))))
        if not zones:
            zones = [{"mid": price, "top": price, "bot": price, "score": 24.0, "kind": "market", "idx": len(m15) - 1, "age": 0}]

        htf_zones = _find_ob(h1, direction, HTF_POI_LOOKBACK) + _find_fvg(h1, direction, HTF_POI_LOOKBACK)
        evaluated = []
        for zone in zones[:10]:
            entry = float(zone["mid"])
            geo = _geometry(m15, h1, direction, entry, atr)
            if geo is None:
                continue
            tp = _targets(m15, h1, direction, entry, geo[1])
            if tp is None:
                continue
            tp_price, tp_label, rr = tp
            overlap = any(float(z["bot"]) <= entry <= float(z["top"]) for z in htf_zones)
            archetype = _archetype(direction, zone, sweep, displacement, pullback, overlap, regime)
            hist = _history_context(trade_history, direction, zone.get("kind", "market"), archetype)
            globals()["_ACTIVE_HISTORY_CONTEXT"] = hist
            q = _candidate_quality(ctx, zone, overlap, sweep, displacement, pullback, location, context, geo, rr, archetype, data_quality)
            timing_penalty = 0.0
            age = float(zone.get("age", 999))
            if age > 36: timing_penalty = 5.0
            if abs(price - entry) > atr * 1.25: timing_penalty += 5.0
            hist_target_adj, hist_target_reason = _target_history_factor(hist, rr)
            execution_score = q["quality"] * 0.93 + min(rr, 5.5) * 1.8 + q["poi_quality"] * 0.025 + hist_target_adj - timing_penalty
            evaluated.append({
                "entry": entry, "entry_label": zone.get("kind", "market"), "sl": geo[0], "risk": geo[1],
                "sl_anchor": geo[8] if len(geo) > 8 else None,
                "risk_atr": geo[2], "risk_pct": geo[3], "tp": tp_price, "tp_label": tp_label, "rr": rr,
                "q": q, "archetype": archetype, "history": hist, "history_target_reason": hist_target_reason, "execution_score": execution_score,
                "htf_overlap": overlap, "zone": zone,
            })
        if not evaluated:
            return None

        evaluated.sort(key=lambda x: x["execution_score"], reverse=True)
        best = evaluated[0]
        quality = best["q"]["quality"]
        confidence = int(best["q"]["confidence"])
        if context.get("conflicts") and not displacement["confirmed"]:
            confidence = max(25, confidence - min(8, len(context["conflicts"]) * 2))
        if ctx["htf_conflict"] and best["archetype"] not in {"LIQUIDITY_SWEEP_RECLAIM", "RANGE_LIQUIDITY_REVERSAL"}:
            confidence = min(confidence, 58)
        if best["entry_label"] == "market":
            confidence = min(confidence, 55)
        band = "ELITE" if confidence >= 80 else "STRONG" if confidence >= 68 else "QUALIFIED" if confidence >= 55 else "WEAK"
        decision = "BUY" if direction == "bull" else "SELL"
        evidence_for = [
            f"D1 bias: {ctx['d1_bias']}",
            f"H1 structure: {ctx['struct_h1']}",
            f"M15 structure: {ctx['struct_m15']}",
            f"setup: {best['archetype']}",
            f"POI: {best['entry_label']}",
        ]
        if displacement["confirmed"]: evidence_for.append(f"fresh displacement {displacement['body_atr']:.2f} ATR")
        if sweep["detected"]: evidence_for.append(f"liquidity reclaim {sweep['strength']:.0f}")
        evidence_against = list(best["q"]["contradictions"])
        globals()["_ACTIVE_HISTORY_CONTEXT"] = {}
        return {
            "symbol": symbol, "decision": decision, "confidence": confidence,
            "direction_confidence": int(round(ctx["direction_quality"])),
            "setup_quality": int(round(quality)),
            "trade_quality": int(round(quality)),
            "confidence_band": band, "confidence_model": CONFIDENCE_MODEL_VERSION,
            "confidence_is_probability": False,
            "confidence_uncertainty": int(round(100 - data_quality + len(evidence_against) * 5)),
            "confidence_diagnostics": {
                "components": best["q"]["components"],
                "contradictions": evidence_against,
                "penalty": best["q"]["penalty"],
                "archetype": best["archetype"],
                "history": best["history"],
                "market_context": context,
                "data_quality": data_quality,
                "candidates_evaluated": len(evaluated),
                "ranking_score": round(best["execution_score"], 3),
            },
            "market_thesis": {
                "direction": decision,
                "archetype": best["archetype"],
                "market_regime": regime,
                "evidence_for": evidence_for,
                "evidence_against": evidence_against,
                "invalidation": "protected structural swing breaks against the thesis",
            },
            "entry_location_score": int(round(location["score"])),
            "entry_location_state": location["state"],
            "entry_range_position": location["range_position"],
            "entry_zone_low": location["range_low"], "entry_zone_high": location["range_high"],
            "trend_strength": {"score": ctx["direction_quality"], "state": ctx["struct_h1"], "efficiency": ctx["efficiency_h1"]},
            "pullback_quality": pullback,
            "liquidity_context": {"sweep": sweep},
            "poi_quality": best["q"]["poi_quality"],
            "poi_state": "fresh" if best["q"]["poi_quality"] >= 75 else "usable" if best["q"]["poi_quality"] >= 55 else "weak",
            "market_regime": regime,
            "entry": round(best["entry"], 10), "price": round(price, 10), "entry_label": best["entry_label"],
            "sl": round(best["sl"], 10), "initial_sl": round(best["sl"], 10), "initial_risk": round(best["risk"], 10),
            "tp": round(best["tp"], 10), "rr": round(best["rr"], 3), "tp_label": best["tp_label"], "atr": round(atr, 10),
            "risk_atr": round(best["risk_atr"], 3), "risk_pct": round(best["risk_pct"], 4),
            "risk_quality": best["q"]["risk_quality"],
            "rsi": round(float(m15["rsi"].iloc[-1]), 2), "m15_rsi": round(float(m15["rsi"].iloc[-1]), 2),
            "m15_rsi_slope": round(float(m15["rsi"].iloc[-1] - m15["rsi"].iloc[-2]), 3),
            "m15_relative_volume": round(float(m15["volume"].iloc[-1] / max(m15["vol_sma"].iloc[-1], 1e-12)), 3),
            "struct_h1": ctx["struct_h1"], "d1_bias": ctx["d1_bias"],
            "htf_bias": ctx["d1_bias"] if ctx["d1_bias"] != "neutral" else ctx["struct_h1"],
            "h1_bias": ctx["struct_h1"],
            "entry_confirmation": displacement,
            "selected_sweep": sweep["detected"],
            "trigger_count": int(displacement["confirmed"]) + int(sweep["detected"]),
            "data_quality": data_quality,
            "evidence_for": evidence_for,
            "evidence_against": evidence_against,
            "archetype": best["archetype"],
            "reasoning_engine": TRAIL_ENGINE_VERSION,
            "tp_sl_reason": f"{best['archetype']} | {decision} | entry={best['entry']:.8g} | SL={best['sl']:.8g} | TP={best['tp']:.8g} | RR={best['rr']:.2f} | quality={quality:.1f} | conf={confidence}%",
        }
    except Exception as exc:
        log.exception("[full_analyze] %s: %s", symbol or "?", exc)
        return None


def _current_price(df, state):
    p = _num(state.get("current_price"), None)
    if p is not None and p > 0:
        return p
    if df is not None and not df.empty:
        return float(df["close"].iloc[-1])
    return None


def _path_metrics(df, state, direction, entry, risk, price):
    cur = (price - entry) / risk if direction == "bull" else (entry - price) / risk
    stored_mfe = _num(state.get("mfe_r"), 0.0) or 0.0
    stored_mae = _num(state.get("mae_r"), 0.0) or 0.0
    sub = df.tail(TRAIL_PEAK_LOOKBACK)
    peak = float(sub["high"].max()) if direction == "bull" else float(sub["low"].min())
    peak_r = (peak - entry) / risk if direction == "bull" else (entry - peak) / risk
    mfe = max(0.0, stored_mfe, peak_r)
    give = max(0.0, mfe - cur)
    ratio = give / max(mfe, 0.25) if mfe > 0 else 0.0
    return {"current_r": cur, "mfe_r": mfe, "mae_r": stored_mae, "giveback_r": give, "giveback_ratio": ratio}


def _reversal_state(df, direction, current_r, giveback_ratio):
    a, b = df.iloc[-1], df.iloc[-2]
    atr = max(float(a["atr"]), 1e-12)
    body = float(a["body"]) / atr
    counter = (a["close"] < a["open"]) if direction == "bull" else (a["close"] > a["open"])
    counter2 = (b["close"] < b["open"]) if direction == "bull" else (b["close"] > b["open"])
    rv = float(a["volume"]) / max(float(a["vol_sma"]), 1e-12)
    if counter and counter2 and body >= TRAIL_REVERSAL_BODY_ATR and giveback_ratio >= TRAIL_GIVEBACK_STRONG:
        return "REVERSAL_CONFIRMED", 3, rv
    if counter and body >= TRAIL_COUNTER_BODY_ATR and giveback_ratio >= TRAIL_GIVEBACK_WARN:
        return "WEAKENING", 2, rv
    if giveback_ratio >= TRAIL_GIVEBACK_STRONG or current_r < 0.5:
        return "CAUTION", 1, rv
    return "HEALTHY", 0, rv


def _structural_trail(df, direction, price, atr):
    sub = df.tail(STRUCT_TRAIL_LOOKBACK)
    sh, sl = swing_pts(sub, STRUCT_TRAIL_LB)
    if direction == "bull" and sl:
        return float(sub["low"].iloc[sl[-1]]) - atr * max(TRAIL_STRUCT_BUFFER_ATR, 0.40)
    if direction == "bear" and sh:
        return float(sub["high"].iloc[sh[-1]]) + atr * max(TRAIL_STRUCT_BUFFER_ATR, 0.40)
    return None


def _retracement_trail(df, direction, entry, risk, price, path):
    atr = max(float(df["atr"].iloc[-1]), 1e-12)
    if path["mfe_r"] < TRAIL_MFE_ARM_R or path["giveback_ratio"] < TRAIL_GIVEBACK_WARN:
        return None
    if direction == "bull":
        peak = float(df["high"].tail(TRAIL_PEAK_LOOKBACK).max())
        keep = 0.45 if path["giveback_ratio"] < TRAIL_GIVEBACK_STRONG else 0.35
        raw = peak - (peak - entry) * keep
        safe_max = price - atr * TRAIL_REVALIDATE_GAP_ATR
        return min(raw, safe_max)
    peak = float(df["low"].tail(TRAIL_PEAK_LOOKBACK).min())
    keep = 0.45 if path["giveback_ratio"] < TRAIL_GIVEBACK_STRONG else 0.35
    raw = peak + (entry - peak) * keep
    safe_min = price + atr * TRAIL_REVALIDATE_GAP_ATR
    return max(raw, safe_min)


def _liquidity_trail(df, direction, price, atr):
    sub = df.tail(30)
    sh, sl = swing_pts(sub, 3)
    if direction == "bull" and sl:
        return float(sub["low"].iloc[sl[-1]]) - atr * 0.20
    if direction == "bear" and sh:
        return float(sub["high"].iloc[sh[-1]]) + atr * 0.20
    return None


def _trail_candidates(m15, state, direction, entry, current_sl, price, path, state_name, weakness):
    atr = max(float(m15["atr"].iloc[-1]), 1e-12)
    candidates = []
    struct = _structural_trail(m15, direction, price, atr)
    if struct is not None:
        candidates.append((float(struct), "structure"))
    retrace = _retracement_trail(m15, direction, entry, abs(entry - float(state.get("initial_sl") or state.get("signal", {}).get("sl") or entry)), price, path)
    if retrace is not None:
        candidates.append((float(retrace), "retracement"))
    if path["mfe_r"] >= TRAIL_MFE_EXTENDED:
        lock_r = TRAIL_LOCK_STRONG_R if path["giveback_ratio"] >= TRAIL_GIVEBACK_WARN else TRAIL_LOCK_WARN_R
        if path["giveback_ratio"] >= TRAIL_GIVEBACK_CRITICAL and path["mfe_r"] >= TRAIL_MFE_DEEP:
            lock_r = TRAIL_LOCK_CRITICAL_R
        lock = entry + abs(entry - float(state.get("initial_sl") or state.get("signal", {}).get("sl") or entry)) * lock_r if direction == "bull" else entry - abs(entry - float(state.get("initial_sl") or state.get("signal", {}).get("sl") or entry)) * lock_r
        candidates.append((float(lock), "path_protection"))
    liq = _liquidity_trail(m15, direction, price, atr)
    if liq is not None and (state_name in {"WEAKENING", "REVERSAL_CONFIRMED"} or path["giveback_ratio"] >= TRAIL_GIVEBACK_STRONG):
        candidates.append((float(liq), "liquidity_structure"))
    return candidates, atr


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
        state_name, weakness, rv = _reversal_state(m15, direction, path["current_r"], path["giveback_ratio"])
        reasons = [f"mfe={path['mfe_r']:.2f}R", f"giveback={path['giveback_ratio']:.0%}"]
        if weakness: reasons.append(f"reversal_signal={weakness}")
        if rv <= TRAIL_VOLUME_EXHAUSTION: reasons.append("volume_exhaustion")
        if rv >= TRAIL_VOLUME_COUNTER and weakness: reasons.append("counter_volume")

        if path["current_r"] < TRAIL_ARM_R and path["mfe_r"] < TRAIL_MFE_ARM_R:
            return {"action": "HOLD", "state": "DEVELOPING", "profit_r": round(path["current_r"], 3), "lifecycle": path, "weakness_score": weakness, "relative_volume": round(rv, 3), "reason": reasons + ["trail_not_mature"], "reasoning_engine": TRAIL_ENGINE_VERSION}

        candidates, atr = _trail_candidates(m15, state, direction, entry, current_sl, price, path, state_name, weakness)
        valid = []
        market_gap = atr * max(TRAIL_MIN_MARKET_GAP_ATR, TRAIL_REVALIDATE_GAP_ATR)
        for cand, source in candidates:
            if direction == "bull":
                if cand >= price - market_gap: cand = price - market_gap
                if cand <= current_sl or cand >= price: continue
            else:
                if cand <= price + market_gap: cand = price + market_gap
                if cand >= current_sl or cand <= price: continue
            improvement_r = ((cand - current_sl) if direction == "bull" else (current_sl - cand)) / risk
            if improvement_r < 0.0: continue
            valid.append((cand, source, improvement_r))
        if not valid:
            action = "PROTECT" if path["giveback_ratio"] >= TRAIL_GIVEBACK_WARN or weakness >= 2 else "HOLD"
            return {"action": action, "state": state_name, "profit_r": round(path["current_r"], 3), "lifecycle": path, "weakness_score": weakness, "relative_volume": round(rv, 3), "reason": reasons + ["no_safe_trail_candidate"], "reasoning_engine": TRAIL_ENGINE_VERSION}

        if direction == "bull":
            best = max(valid, key=lambda x: (x[0], x[2]))
        else:
            best = min(valid, key=lambda x: (x[0], -x[2]))
        cand, source, improvement_r = best
        count = int(state.get("trail_update_count", 0) or 0)
        if count >= TRAIL_MAX_CHURN and improvement_r < TRAIL_MIN_UPDATE_R:
            return {"action": "PROTECT", "state": state_name, "profit_r": round(path["current_r"], 3), "lifecycle": path, "weakness_score": weakness, "relative_volume": round(rv, 3), "reason": reasons + ["anti_churn_small_improvement"], "reasoning_engine": TRAIL_ENGINE_VERSION}

        lock_r = ((cand - entry) if direction == "bull" else (entry - cand)) / risk
        reasons.append(f"source={source}")
        if path["mfe_r"] >= TRAIL_MFE_EXTENDED: reasons.append("extended_winner_path")
        if path["giveback_ratio"] >= TRAIL_GIVEBACK_STRONG: reasons.append("deep_giveback")
        return {
            "action": "TRAIL", "state": state_name, "sl": round(float(cand), 10),
            "profit_r": round(path["current_r"], 3), "locked_r": round(float(lock_r), 3),
            "trail_source": source, "candidate_type": source,
            "weakness_score": weakness, "relative_volume": round(rv, 3), "lifecycle": path,
            "reversal_diagnostics": {"state": state_name, "score": weakness, "relative_volume": round(rv, 3)},
            "trail_geometry": {"market_price": round(price, 10), "atr": round(atr, 10), "min_market_gap_atr": TRAIL_MIN_MARKET_GAP_ATR, "improvement_r": round(improvement_r, 3)},
            "reason": reasons, "reasoning_engine": TRAIL_ENGINE_VERSION,
        }
    except Exception as exc:
        log.exception("[manage_position] %s: %s", symbol or "?", exc)
        return {"action": "PROTECT", "state": "ERROR", "reason": [f"management_exception:{type(exc).__name__}"], "reasoning_engine": TRAIL_ENGINE_VERSION}
