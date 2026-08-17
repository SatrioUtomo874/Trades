"""
strategy_logic.py 1.2
====================
Research-built strategy layer for the supplied Binance M1 dataset and the
SMC/ICT transcript corpus (combined.txt).

IMPORTANT DESIGN:
- This file is the strategy only. It does NOT make API requests.
- main.py remains the data/API/execution layer.
- Decision order is strictly: ENTRY -> STRUCTURAL SL -> LIQUIDITY TP -> TRAIL.
- A liquidity sweep is an information event, not an automatic entry.
- RSI and volume are early/confirmation evidence, never standalone triggers.
- Confidence is a 0..100 quality estimate, not a guarantee of probability.
- RR is constrained by main.py at execution time: target selection prefers 2R..4R.

The raw M1 dataset was inspected at M15 aggregation for pattern research.
The research found that a bare sweep is noisy on BTC/ETH/XRP/SOL; therefore the
strategy gives substantially more weight to the post-sweep reaction:
reclaim + displacement/structure response + location. ZEC was materially more
volatile, so all geometry is volatility-normalized instead of using fixed
percentage distances.
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple, List, Dict
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Public contract expected by main.py
# -----------------------------------------------------------------------------
STRATEGY_VERSION = "1.2"
MIN_RR = 2.0
MAX_RR = 4.0
TRAIL_R_LADDER = []
STRUCT_TRAIL_LB = 3
STRUCT_TRAIL_BUF_PCT = 0.0015
STRUCT_TRAIL_LOOKBACK = 80
FIB_EXT_1 = 0.272
FIB_EXT_2 = 0.618

# No API calls are made here. main.py supplies all market data.

# Empirical/research geometry. Percentages are only used as small buffers;
# actual invalidation remains structural.
SWEEP_ATR_BUFFER = 0.18
TRAIL_ATR_BUFFER = 0.18
ENTRY_MAX_ATR_DISTANCE = 1.50


def _ensure_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    out = df.copy()
    for c in ("open", "high", "low", "close", "volume"):
        if c not in out.columns:
            out[c] = np.nan
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out[["open", "high", "low", "close", "volume"]].dropna()


def _closed_candles(df: pd.DataFrame, interval_minutes: int) -> pd.DataFrame:
    # Offline historical frames are already closed. For a live frame, remove
    # the currently forming candle. This avoids repainting.
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
    if idx[-1] < boundary:
        return out
    return out.loc[idx < boundary].copy()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    d = s.diff()
    gain = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    loss = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def atr_fn(df: pd.DataFrame, n: int = 14) -> pd.Series:
    prev = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev).abs(),
        (df["low"] - prev).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def build_df(df: pd.DataFrame, interval_minutes: Optional[int] = None) -> Optional[pd.DataFrame]:
    out = _ensure_ohlcv(df)
    if interval_minutes is not None:
        out = _closed_candles(out, interval_minutes)
    if len(out) < 60:
        return None
    out = out.copy()
    out["ema9"] = ema(out["close"], 9)
    out["ema21"] = ema(out["close"], 21)
    out["ema50"] = ema(out["close"], 50)
    out["ema200"] = ema(out["close"], 200) if len(out) >= 200 else ema(out["close"], 50)
    out["rsi"] = rsi(out["close"])
    out["atr"] = atr_fn(out)
    out["vol_sma"] = out["volume"].rolling(20).mean()
    out["vol_ratio"] = out["volume"] / out["vol_sma"].replace(0, np.nan)
    out["buy_volume_ratio"] = np.nan
    out = out.replace([np.inf, -np.inf], np.nan)
    required = ["open", "high", "low", "close", "volume", "ema9", "ema21", "ema50", "ema200", "rsi", "atr", "vol_sma", "vol_ratio"]
    out = out.dropna(subset=required)
    return out if not out.empty else None


def swing_pts(df: pd.DataFrame, lb: int = 5):
    df = _ensure_ohlcv(df)
    hi, lo = df["high"].to_numpy(), df["low"].to_numpy()
    sh, sl = [], []
    for i in range(lb, len(df) - lb):
        if hi[i] >= np.max(hi[i-lb:i+lb+1]):
            sh.append(i)
        if lo[i] <= np.min(lo[i-lb:i+lb+1]):
            sl.append(i)
    return sh, sl


_raw_swing_pts = swing_pts


def _structure(df: pd.DataFrame, lb: int = 3) -> str:
    sh, sl = swing_pts(df, lb)
    if len(sh) < 2 or len(sl) < 2:
        return "ranging"
    hh = df.high.iloc[sh[-1]] > df.high.iloc[sh[-2]]
    hl = df.low.iloc[sl[-1]] > df.low.iloc[sl[-2]]
    lh = df.high.iloc[sh[-1]] < df.high.iloc[sh[-2]]
    ll = df.low.iloc[sl[-1]] < df.low.iloc[sl[-2]]
    if hh and hl:
        return "bullish"
    if lh and ll:
        return "bearish"
    return "ranging"


mkt_struct = _structure


def _last_confirmed_swing(df: pd.DataFrame, direction: str, lb: int = 3):
    sh, sl = swing_pts(df, lb)
    if direction == "bull":
        return (float(df.low.iloc[sl[-1]]), sl[-1]) if sl else (None, None)
    return (float(df.high.iloc[sh[-1]]), sh[-1]) if sh else (None, None)


def detect_equal_highs_lows(df: pd.DataFrame, kind: str = "high", lb: int = 80, tol: float = 0.0018):
    sub = _ensure_ohlcv(df).iloc[-lb:]
    vals = sub["high" if kind == "high" else "low"].to_numpy(float)
    if len(vals) < 4:
        return []
    clusters = []
    for i in range(len(vals)):
        for j in range(i + 2, len(vals)):
            base = max(abs(vals[i]), 1e-12)
            if abs(vals[j] - vals[i]) / base <= tol:
                clusters.append((vals[i] + vals[j]) / 2)
    if not clusters:
        return []
    clusters.sort()
    out = []
    for x in clusters:
        if not out or abs(x - out[-1]) / max(abs(x), 1e-12) > tol:
            out.append(float(x))
        else:
            out[-1] = (out[-1] + x) / 2
    return out[-8:]


def detect_liquidity_sweep(df: pd.DataFrame, sh: list, sl: list, direction: str) -> dict:
    """Sweep = meaningful swing/equal-liquidity breach + reclaim on close."""
    out = {"type": "none", "level": None, "extreme": None, "strength": 0}
    if df is None or len(df) < 10:
        return out
    last = df.iloc[-1]
    if direction == "bull":
        levels = []
        if sl:
            levels.append((float(df.low.iloc[sl[-1]]), "swing"))
        levels += [(float(x), "equal") for x in detect_equal_highs_lows(df, "low")]
        for level, kind in sorted(levels, key=lambda x: abs(float(last.close) - x[0])):
            if float(last.low) < level and float(last.close) > level:
                depth = (level - float(last.low)) / max(float(last.atr), 1e-12) if "atr" in df else 0.0
                return {"type": "sweep", "level": level, "extreme": float(last.low),
                        "strength": int(min(3, max(1, round(depth + (1 if kind == "equal" else 0))))),
                        "kind": kind}
    else:
        levels = []
        if sh:
            levels.append((float(df.high.iloc[sh[-1]]), "swing"))
        levels += [(float(x), "equal") for x in detect_equal_highs_lows(df, "high")]
        for level, kind in sorted(levels, key=lambda x: abs(float(last.close) - x[0])):
            if float(last.high) > level and float(last.close) < level:
                depth = (float(last.high) - level) / max(float(last.atr), 1e-12) if "atr" in df else 0.0
                return {"type": "sweep", "level": level, "extreme": float(last.high),
                        "strength": int(min(3, max(1, round(depth + (1 if kind == "equal" else 0))))),
                        "kind": kind}
    return out


def detect_inducement(df: pd.DataFrame, direction: str, lb: int = 40) -> dict:
    sub = _ensure_ohlcv(df).iloc[-lb:].reset_index(drop=True)
    if len(sub) < 12:
        return {"found": False, "swept": False, "level": None}
    sh, sl = swing_pts(sub, 2)
    if direction == "bull" and sl:
        level = float(sub.low.iloc[sl[-1]])
        after = sub.iloc[sl[-1] + 1:]
        swept = bool((after.low < level).any() and float(after.close.iloc[-1]) > level)
        return {"found": True, "swept": swept, "level": level}
    if direction == "bear" and sh:
        level = float(sub.high.iloc[sh[-1]])
        after = sub.iloc[sh[-1] + 1:]
        swept = bool((after.high > level).any() and float(after.close.iloc[-1]) < level)
        return {"found": True, "swept": swept, "level": level}
    return {"found": False, "swept": False, "level": None}


def detect_fvg(df: pd.DataFrame, direction: str, lb: int = 60) -> list:
    sub = _ensure_ohlcv(df).iloc[-lb:]
    out = []
    for i in range(2, len(sub)):
        a, c = sub.iloc[i-2], sub.iloc[i]
        if direction == "bull" and c.low > a.high:
            out.append({"top": float(c.low), "bot": float(a.high), "mid": float((c.low+a.high)/2), "idx": len(df)-len(sub)+i})
        elif direction == "bear" and c.high < a.low:
            out.append({"top": float(a.low), "bot": float(c.high), "mid": float((a.low+c.high)/2), "idx": len(df)-len(sub)+i})
    # Fresh means not fully invalidated by a later close.
    fresh=[]
    for z in out:
        later=_ensure_ohlcv(df).iloc[z["idx"]+1:]
        if later.empty:
            fresh.append(z); continue
        if direction == "bull" and not (later.close < z["bot"]).any(): fresh.append(z)
        if direction == "bear" and not (later.close > z["top"]).any(): fresh.append(z)
    return fresh[-5:]


def detect_order_block(df: pd.DataFrame, direction: str, lb: int = 80, sh=None, sl=None) -> list:
    sub=_ensure_ohlcv(df).iloc[-lb:]
    if len(sub)<8: return []
    avg_body=float((sub.close-sub.open).abs().rolling(20).mean().iloc[-1] or 0)
    out=[]
    for i in range(2,len(sub)-2):
        c=sub.iloc[i]; n=sub.iloc[i+1]
        body=abs(float(n.close-n.open))
        if direction=="bull" and c.close<c.open and n.close>n.open and body>=max(avg_body*1.15,1e-12):
            out.append((body,{"top":float(max(c.open,c.close)),"bot":float(min(c.open,c.close)),"mid":float((c.open+c.close)/2),"idx":len(df)-len(sub)+i}))
        elif direction=="bear" and c.close>c.open and n.close<n.open and body>=max(avg_body*1.15,1e-12):
            out.append((body,{"top":float(max(c.open,c.close)),"bot":float(min(c.open,c.close)),"mid":float((c.open+c.close)/2),"idx":len(df)-len(sub)+i}))
    out.sort(key=lambda x:x[0],reverse=True)
    return [z for _,z in out[:5]]


def _displacement(df: pd.DataFrame, direction: str, lookback: int = 4) -> dict:
    if len(df)<lookback+2: return {"confirmed":False,"body_atr":0.0}
    last=df.iloc[-1]; prior=df.iloc[-lookback-1:-1]
    body=abs(float(last.close-last.open)); atr=max(float(last.atr),1e-12)
    body_atr=body/atr
    if direction=="bull": ok=last.close>last.open and last.close>prior.high.max() and body_atr>=0.45
    else: ok=last.close<last.open and last.close<prior.low.min() and body_atr>=0.45
    return {"confirmed":bool(ok),"body_atr":round(body_atr,3)}


def _momentum_state(df: pd.DataFrame, direction: str) -> dict:
    x=df.iloc[-1]; prev=df.iloc[-4]
    r=float(x.rsi); dr=r-float(prev.rsi)
    vr=float(x.vol_ratio) if np.isfinite(x.vol_ratio) else 1.0
    if direction=="bull":
        r_ok=42<=r<=72 and dr>0
        vol_ok=vr>=1.05 or (vr>=0.8 and x.close>x.open)
    else:
        r_ok=28<=r<=58 and dr<0
        vol_ok=vr>=1.05 or (vr>=0.8 and x.close<x.open)
    return {"rsi":r,"rsi_delta":dr,"vol_ratio":vr,"rsi_ok":bool(r_ok),"volume_ok":bool(vol_ok)}


def detect_choch(df: pd.DataFrame, sh: list, sl: list) -> dict:
    out={"bullish_choch":False,"bearish_choch":False}
    if len(sh)<2 or len(sl)<2: return out
    close=float(df.close.iloc[-1]); lh=float(df.high.iloc[sh[-1]]); ll=float(df.low.iloc[sl[-1]])
    prev_h=float(df.high.iloc[sh[-2]]); prev_l=float(df.low.iloc[sl[-2]])
    struct=_structure(df,3)
    if struct=="bearish" and close>lh: out["bullish_choch"]=True
    if struct=="bullish" and close<ll: out["bearish_choch"]=True
    if lh>prev_h and ll>prev_l and close>lh: out["bullish_choch"]=True
    if lh<prev_h and ll<prev_l and close<ll: out["bearish_choch"]=True
    return out


def detect_bos(df: pd.DataFrame, sh: list, sl: list) -> dict:
    out={"bullish_bos":False,"bearish_bos":False}
    if not sh or not sl: return out
    close=float(df.close.iloc[-1]); ph=float(df.high.iloc[sh[-1]]); pl=float(df.low.iloc[sl[-1]])
    prev=float(df.close.iloc[-2])
    out["bullish_bos"]=close>ph and prev<=ph
    out["bearish_bos"]=close<pl and prev>=pl
    return out


def detect_cisd(df: pd.DataFrame, lb:int=8) -> dict:
    out={"bullish_cisd":False,"bearish_cisd":False}
    if len(df)<lb+2: return out
    s=df.iloc[-lb:]
    o=s.open.to_numpy(); c=s.close.to_numpy()
    if c[-1]>o[-1]:
        run=0
        for j in range(len(c)-2,-1,-1):
            if c[j]<o[j]: run+=1
            else: break
        if run>=2:
            k=len(c)-1-run; mid=(o[k]+c[k])/2
            out["bullish_cisd"]=c[-1]>mid
    elif c[-1]<o[-1]:
        run=0
        for j in range(len(c)-2,-1,-1):
            if c[j]>o[j]: run+=1
            else: break
        if run>=2:
            k=len(c)-1-run; mid=(o[k]+c[k])/2
            out["bearish_cisd"]=c[-1]<mid
    return out


def _find_entry(m15: pd.DataFrame, h1: pd.DataFrame, direction: str, score: dict) -> Optional[dict]:
    up=direction=="bull"; price=float(m15.close.iloc[-1]); atr=float(m15.atr.iloc[-1])
    sweep=score["sweep"]
    # First choice: FVG created by displacement after sweep / structure response.
    fvgs=detect_fvg(m15,direction,70)
    obs=detect_order_block(m15,direction,80,score.get("sh15"),score.get("sl15"))
    candidates=[]
    for z in fvgs:
        e=z["mid"]
        if (up and e<=price and e>=price-ENTRY_MAX_ATR_DISTANCE*atr) or (not up and e>=price and e<=price+ENTRY_MAX_ATR_DISTANCE*atr):
            q=50
            if sweep["type"]=="sweep": q+=20
            if score["displacement"]["confirmed"]: q+=15
            if score["inducement"]["swept"]: q+=5
            candidates.append((q,e,z["bot"] if up else z["top"],"fvg"))
    for z in obs:
        e=z["mid"]
        if (up and e<=price and e>=price-ENTRY_MAX_ATR_DISTANCE*atr) or (not up and e>=price and e<=price+ENTRY_MAX_ATR_DISTANCE*atr):
            q=42
            if sweep["type"]=="sweep": q+=18
            if score["displacement"]["confirmed"]: q+=15
            if score["inducement"]["swept"]: q+=5
            candidates.append((q,e,z["bot"] if up else z["top"],"ob"))
    # If a sweep just occurred, reclaim entry can be the level itself when no
    # clean imbalance exists. This is still a limit/retest concept, not chase.
    if sweep["type"]=="sweep":
        e=float(sweep["level"])
        if (up and e<=price and price-e<=ENTRY_MAX_ATR_DISTANCE*atr) or (not up and e>=price and e-price<=ENTRY_MAX_ATR_DISTANCE*atr):
            candidates.append((55,e,float(sweep["extreme"]),"sweep_reclaim"))
    if not candidates:
        # Continuation setup: fresh POI aligned with H1, no synthetic market entry.
        for z in obs+fvgs:
            e=float(z.get("mid",(z["top"]+z["bot"])/2))
            if (up and e<=price) or (not up and e>=price):
                candidates.append((25,e,float(z["bot"] if up else z["top"]),"poi"))
    if not candidates: return None
    candidates.sort(key=lambda x:x[0],reverse=True)
    q,e,invalid,label=candidates[0]
    return {"entry":float(e),"invalid":float(invalid),"label":label,"quality":q}


def _compute_sl(m15,h1,direction,entry,atr,sweep,invalid=None):
    up=direction=="bull"
    structural=[]
    sh15,sl15=swing_pts(m15,3); sh1,sl1=swing_pts(h1,3)
    if up:
        if sweep.get("type")=="sweep" and sweep.get("extreme") is not None: structural.append(float(sweep["extreme"]))
        if invalid is not None and invalid<entry: structural.append(float(invalid))
        if sl15: structural.append(float(m15.low.iloc[sl15[-1]]))
        if sl1: structural.append(float(h1.low.iloc[sl1[-1]]))
        if not structural: return entry-atr,atr,"atr_fallback"
        anchor=min(x for x in structural if x<entry)
        sl=anchor-atr*SWEEP_ATR_BUFFER
    else:
        if sweep.get("type")=="sweep" and sweep.get("extreme") is not None: structural.append(float(sweep["extreme"]))
        if invalid is not None and invalid>entry: structural.append(float(invalid))
        if sh15: structural.append(float(m15.high.iloc[sh15[-1]]))
        if sh1: structural.append(float(h1.high.iloc[sh1[-1]]))
        if not structural: return entry+atr,atr,"atr_fallback"
        anchor=max(x for x in structural if x>entry)
        sl=anchor+atr*SWEEP_ATR_BUFFER
    return sl,abs(entry-sl),"structural"


def _target_pool(h1,m15,direction,entry,atr):
    up=direction=="bull"; sgn=1 if up else -1; pool=[]
    sh1,sl1=swing_pts(h1,3)
    swing=[float(h1.high.iloc[i]) for i in sh1] if up else [float(h1.low.iloc[i]) for i in sl1]
    for v in swing[-6:]:
        if sgn*(v-entry)>atr*0.35: pool.append((v,"H1_swing",1))
    eq=detect_equal_highs_lows(h1,"high" if up else "low",120)
    for v in eq:
        if sgn*(v-entry)>atr*0.35: pool.append((v,"H1_equal_liquidity",0))
    # Opposite-side H1 OB/FVG are external-ish targets; keep lower priority than
    # obvious swing/equal liquidity.
    opp="bear" if up else "bull"
    for z in detect_order_block(h1,opp,100,sh1,sl1):
        v=float(z["bot"] if up else z["top"])
        if sgn*(v-entry)>atr*0.35: pool.append((v,"H1_OPP_OB",2))
    for z in detect_fvg(h1,opp,100):
        v=float(z["mid"])
        if sgn*(v-entry)>atr*0.35: pool.append((v,"H1_OPP_FVG",3))
    # Older swings are deliberately included so a close first target <2R does
    # not automatically kill the setup.
    pool.sort(key=lambda x:(abs(x[0]-entry),x[2]))
    return pool


def _select_tp(pool,entry,risk,direction):
    if risk<=0: return None,None,None
    sgn=1 if direction=="bull" else -1
    valid=[]
    for v,label,tier in pool:
        rr=sgn*(float(v)-entry)/risk
        if rr>0: valid.append((rr,float(v),label,tier))
    valid.sort(key=lambda x:x[0])
    for rr,v,label,tier in valid:
        if rr>=MIN_RR:
            if rr<=MAX_RR: return round(v,8),label,round(rr,2)
            return round(entry+sgn*risk*MAX_RR,8),label+"_capped",MAX_RR
    # No genuine >=2R target: return the furthest REAL target and let main.py's
    # MIN_RR gate reject execution. We do not manufacture a 2R target.
    if valid:
        rr,v,label,_=valid[-1]
        return round(v,8),label,round(rr,2)
    return None,None,None


def _confidence(direction, h1, m15, d1, score, entry, sl, tp, rr):
    up=direction=="bull"; points=0
    # Context 0..22
    h1s=score["h1_struct"]; d1s=score["d1_struct"]
    if (up and h1s=="bullish") or ((not up) and h1s=="bearish"): points+=10
    if (up and d1s=="bullish") or ((not up) and d1s=="bearish"): points+=7
    if h1s=="ranging": points+=2
    # Liquidity event 0..22
    sw=score["sweep"]
    if sw["type"]=="sweep": points+=12+min(3,sw.get("strength",0))*2
    if score["inducement"].get("swept"): points+=4
    # Entry 0..24
    if score["displacement"]["confirmed"]: points+=10
    if score["choch"]: points+=6
    if score["cisd"]: points+=4
    if score["entry_label"]=="fvg": points+=4
    elif score["entry_label"]=="ob": points+=3
    # Momentum 0..14
    if score["momentum"]["rsi_ok"]: points+=7
    if score["momentum"]["volume_ok"]: points+=7
    # Geometry 0..18
    if rr>=2: points+=8
    if rr>=3: points+=4
    if score["risk_atr"]<=3.0: points+=3
    if score["risk_atr"]<=2.0: points+=3
    # Contradiction penalties
    mstruct=_structure(m15,3)
    if (up and mstruct=="bearish") or ((not up) and mstruct=="bullish"): points-=8
    if (up and float(m15.rsi.iloc[-1])>78) or ((not up) and float(m15.rsi.iloc[-1])<22): points-=6
    return int(max(0,min(100,points)))


def score_direction(df_h1, df_m15, df_d1=None, df_btc_h1=None):
    h1=build_df(df_h1,60); m15=build_df(df_m15,15)
    if h1 is None or h1.empty or m15 is None or m15.empty: return None
    d1=build_df(df_d1,1440) if df_d1 is not None and len(df_d1)>=20 else None
    if d1 is None:
        d1=h1.resample("1D").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna()
        d1=build_df(d1,None)
    sh15,sl15=swing_pts(m15,3); sh1,sl1=swing_pts(h1,3)
    h1_struct=_structure(h1,3); d1_struct=_structure(d1,2) if d1 is not None else "ranging"
    if h1.empty or len(h1) < 1: return None
    ema_bull=h1.ema9.iloc[-1]>h1.ema21.iloc[-1]>h1.ema50.iloc[-1]
    ema_bear=h1.ema9.iloc[-1]<h1.ema21.iloc[-1]<h1.ema50.iloc[-1]
    bull= (12 if ema_bull else 0)+(10 if h1_struct=="bullish" else 0)+(6 if d1_struct=="bullish" else 0)
    bear= (12 if ema_bear else 0)+(10 if h1_struct=="bearish" else 0)+(6 if d1_struct=="bearish" else 0)
    liq_b=detect_liquidity_sweep(m15,sh15,sl15,"bull"); liq_s=detect_liquidity_sweep(m15,sh15,sl15,"bear")
    # Sweep is evidence, not an automatic direction gate.
    if liq_b["type"]=="sweep": bull+=18+liq_b.get("strength",0)*2
    if liq_s["type"]=="sweep": bear+=18+liq_s.get("strength",0)*2
    choch=detect_choch(m15,sh15,sl15); bos=detect_bos(m15,sh15,sl15); cisd=detect_cisd(m15)
    if choch["bullish_choch"]: bull+=10
    if choch["bearish_choch"]: bear+=10
    if cisd["bullish_cisd"]: bull+=7
    if cisd["bearish_cisd"]: bear+=7
    if bos["bullish_bos"]: bull+=5
    if bos["bearish_bos"]: bear+=5
    # If HTF is clear, favor it; otherwise choose stronger current evidence.
    if h1_struct=="bullish" and bull>=bear: direction="bull"
    elif h1_struct=="bearish" and bear>=bull: direction="bear"
    else: direction="bull" if bull>=bear else "bear"
    sweep=liq_b if direction=="bull" else liq_s
    induce=detect_inducement(m15,direction)
    disp=_displacement(m15,direction)
    mom=_momentum_state(m15,direction)
    return {"direction":direction,"bull_score":bull,"bear_score":bear,"price":float(m15.close.iloc[-1]),
            "atr":float(m15.atr.iloc[-1]),"h1_struct":h1_struct,"d1_struct":d1_struct,
            "sh15":sh15,"sl15":sl15,"sh1":sh1,"sl1":sl1,"sweep":sweep,"inducement":induce,
            "displacement":disp,"momentum":mom,"choch":bool(choch["bullish_choch"] if direction=="bull" else choch["bearish_choch"]),
            "cisd":bool(cisd["bullish_cisd"] if direction=="bull" else cisd["bearish_cisd"]),
            "bos":bos}


def full_analyze(df_h1: pd.DataFrame, df_m15: pd.DataFrame, df_d1: Optional[pd.DataFrame]=None,
                 symbol: Optional[str]=None, df_btc_h1: Optional[pd.DataFrame]=None) -> Optional[dict]:
    try:
        score=score_direction(df_h1,df_m15,df_d1,df_btc_h1)
        if score is None: return None
        direction=score["direction"]; h1=build_df(df_h1,60); m15=build_df(df_m15,15)
        if h1 is None or h1.empty or m15 is None or m15.empty: return None
        cand=_find_entry(m15,h1,direction,score)
        if cand is None: return None
        sl,risk,sl_reason=_compute_sl(m15,h1,direction,cand["entry"],score["atr"],score["sweep"],cand.get("invalid"))
        if (direction=="bull" and sl>=cand["entry"]) or (direction=="bear" and sl<=cand["entry"]): return None
        tp,tp_label,rr=_select_tp(_target_pool(h1,m15,direction,cand["entry"],score["atr"]),cand["entry"],risk,direction)
        if tp is None: return None
        score["entry_label"]=cand["label"]; score["risk_atr"]=risk/max(score["atr"],1e-12)
        conf=_confidence(direction,h1,m15,df_d1,score,cand["entry"],sl,tp,rr or 0)
        # Keep a low-confidence setup observable; main.py decides whether it is
        # executable through MIN_CONFIDENCE.
        return {
            "symbol":symbol,"original_dir":direction,"decision":"BUY" if direction=="bull" else "SELL",
            "confidence":conf,"price":score["price"],"entry":round(cand["entry"],8),"entry_label":cand["label"],
            "sl":round(sl,8),"tp":round(tp,8),"rr":float(rr),"atr":round(score["atr"],8),
            "rsi":round(float(m15.rsi.iloc[-1]),2),"struct_h1":score["h1_struct"],"struct_m15":_structure(m15,3),
            "d1_bias":score["d1_struct"],"h1_bias":score["h1_struct"],"htf_bias":score["h1_struct"],
            "choch_m15":detect_choch(m15,score["sh15"],score["sl15"]),
            "choch_h1":detect_choch(h1,score["sh1"],score["sl1"]),"cisd_m15":detect_cisd(m15),
            "failed_retest":{"failed_retest_buy":False,"failed_retest_sell":False},
            "selected_sweep":score["sweep"]["type"]=="sweep","sweep":score["sweep"],
            "inducement":score["inducement"],"entry_confirmation":score["displacement"],
            "momentum":score["momentum"],"trigger_count":int(score["displacement"]["confirmed"])+int(score["choch"])+int(score["cisd"]),
            "tp_sl_reason":f"Entry@{cand['entry']:.8g}({cand['label']}) | SL@{sl:.8g}({sl_reason}) | TP@{tp:.8g}({tp_label}) | RR={rr:.2f}",
        }
    except Exception as e:
        if symbol: log.exception("full_analyze %s failed",symbol)
        return None


def get_best_signal(candidates: list) -> Optional[dict]:
    if not candidates: return None
    return max(candidates,key=lambda x:(int(x.get("confidence",0)),float(x.get("rr",0))))


def validate_and_adjust_geometry(entry: float, sl: float, tp: float, current_price: float,
                                 atr: float, direction: str) -> Optional[dict]:
    up=direction=="bull"
    if up and not (sl<entry<tp): return None
    if not up and not (tp<entry<sl): return None
    if (up and current_price<=sl) or ((not up) and current_price>=sl): return None
    rr=abs(tp-entry)/max(abs(entry-sl),1e-12)
    return {"entry":entry,"sl":sl,"tp":tp,"rr":round(rr,2),"adjusted":False}


def strategy_trailing_stop(df_m15: pd.DataFrame, entry: float, current_sl: float,
                           direction: str, risk: float, current_price: float,
                           tp: float, position=None) -> dict:
    """Structural trail only. No profit ladder / BE ladder."""
    df=build_df(df_m15,15)
    if df is None or len(df)<20: return {"candidate":None,"reason":"data"}
    sh,sl=swing_pts(df,STRUCT_TRAIL_LB)
    atr=float(df.atr.iloc[-1])
    if direction=="bull" and sl:
        anchor=float(df.low.iloc[sl[-1]])
        candidate=anchor-atr*TRAIL_ATR_BUFFER
        if candidate>current_sl and candidate<current_price: return {"candidate":round(candidate,8),"reason":"new_HL"}
    if direction=="bear" and sh:
        anchor=float(df.high.iloc[sh[-1]])
        candidate=anchor+atr*TRAIL_ATR_BUFFER
        if candidate<current_sl and candidate>current_price: return {"candidate":round(candidate,8),"reason":"new_LH"}
    return {"candidate":None,"reason":"structure_not_advanced"}



def manage_position(state: dict, df_m15: pd.DataFrame, df_h1: Optional[pd.DataFrame] = None,
                    df_d1: Optional[pd.DataFrame] = None, symbol: Optional[str] = None) -> Optional[dict]:
    """Single position-management contract used by main.py.

    Strategy owns trailing decisions. The engine only executes the returned SL/close
    instruction. Trail is structural invalidation, not a profit-lock ladder.
    """
    try:
        if not isinstance(state, dict) or df_m15 is None or df_m15.empty:
            return None
        signal = state.get("signal") if isinstance(state.get("signal"), dict) else state
        direction = str(signal.get("decision") or state.get("direction") or "").upper()
        if direction not in ("BUY", "SELL"):
            return None
        entry = float(signal.get("entry", state.get("entry")))
        current_sl = float(state.get("current_sl", signal.get("sl")))
        tp = signal.get("tp", state.get("tp"))
        tp = float(tp) if tp is not None else entry
        current_price = state.get("current_price", state.get("price"))
        if current_price is None:
            current_price = float(df_m15["close"].iloc[-1])
        else:
            current_price = float(current_price)
        atr = float(build_df(df_m15, 15)["atr"].iloc[-1]) if build_df(df_m15, 15) is not None else 0.0
        risk = abs(entry - float(signal.get("sl", current_sl)))
        out = strategy_trailing_stop(
            df_m15=df_m15, entry=entry, current_sl=current_sl,
            direction="bull" if direction == "BUY" else "bear",
            risk=risk, current_price=current_price, tp=tp, position=state
        )
        cand = out.get("candidate") if isinstance(out, dict) else None
        if cand is None:
            return None
        if direction == "BUY" and not (cand > current_sl and cand < current_price):
            return None
        if direction == "SELL" and not (cand < current_sl and cand > current_price):
            return None
        return {"sl": round(float(cand), 8), "reason": "trail", "trail_reason": out.get("reason", "structure")}
    except Exception:
        if symbol:
            log.debug("manage_position %s failed", symbol, exc_info=True)
        return None

def select_best_signal(candidates: list) -> Optional[dict]:
    """Compatibility helper; multi-signal main.py does not use this.

    Kept so older callers can hot-swap this strategy without a missing symbol.
    """
    if not candidates:
        return None
    return max(candidates, key=lambda x: (float(x.get("confidence", 0)), float(x.get("rr", 0))))


def analyze_setup(*args, **kwargs):
    return full_analyze(*args, **kwargs)
