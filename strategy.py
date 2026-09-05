"""
strategy.py — Adaptive SMC/ICT Strategy vNext

No network / no API. Main.py owns all market I/O and passes snapshots here.
The engine produces explainable setup + monitoring diagnostics and keeps
parameter mutation behind apply_update().
"""
from __future__ import annotations
import math, time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple
try:
    import numpy as np
except ImportError:
    np = None

STRATEGY_NAME = "adaptive-smc-ict"
STRATEGY_SCHEMA = 2
CONFIDENCE_WEIGHTS = {
    "structure": 18.0, "liquidity": 14.0, "entry_quality": 14.0,
    "risk_reward": 12.0, "momentum": 9.0, "volatility": 6.0,
    "btc_context": 10.0, "regime": 6.0, "session": 3.0,
    "freshness": 4.0, "execution_viability": 4.0,
}
DEFAULT_PARAMS = {
    "ACTIVE_THRESHOLD": 0.0,
    "swing_left": 2, "swing_right": 2, "equal_level_tol_atr": 0.15,
    "displacement_atr_mult": 1.5, "min_rr": 1.2, "sweep_lookback": 40,
    "structure_lookback": 120, "momentum_lookback": 10, "atr_period": 14,
    "vol_regime_lookback": 100, "trend_lookback": 30, "btc_corr_lookback": 50,
    "sl_atr_buffer": 0.25, "min_price_distance_ticks": 2,
    "entry_retracement_fib": 0.618, "entry_min_offset_atr": 0.25,
    "setup_max_age_bars": 8, "fvg_max_age_bars": 6,
    "tp_liquidity_buffer_atr": 0.15, "trail_min_profit_r": 0.8,
    "trail_weakness_threshold": 3, "trail_min_step_r": 0.15,
}
_REQUIRED=("t","o","h","l","c","v")

def _sf(x:Any,d:float=0.0)->float:
    try:
        v=float(x); return v if math.isfinite(v) else d
    except (TypeError,ValueError): return d

def _cl(c): return [_sf(x.get("c")) for x in c]
def _hi(c): return [_sf(x.get("h")) for x in c]
def _lo(c): return [_sf(x.get("l")) for x in c]

def validate_candles(candles:Sequence[Dict[str,Any]], min_len:int=1)->Tuple[bool,str]:
    if candles is None or len(candles)<min_len: return False,"INSUFFICIENT_CANDLES"
    prev=None
    for i,c in enumerate(candles):
        if not isinstance(c,dict) or any(k not in c for k in _REQUIRED): return False,f"MALFORMED_CANDLE_{i}"
        vals=[_sf(c.get(k),float("nan")) for k in _REQUIRED]
        if any(not math.isfinite(v) for v in vals): return False,f"NON_FINITE_{i}"
        t,o,h,l,cl,v=vals
        if t<=0 or min(o,h,l,cl)<=0 or v<0 or h<max(o,cl) or l>min(o,cl) or h<l: return False,f"INVALID_OHLC_{i}"
        if prev is not None and t<=prev: return False,f"TIMESTAMP_NOT_ASCENDING_{i}"
        prev=t
    return True,"OK"

def _confirmed(c):
    if c and c[-1].get("confirm",True) is False: return list(c[:-1])
    return list(c)

def true_range(c):
    out=[]; prev=None
    for x in c:
        h,l,cl=_sf(x["h"]),_sf(x["l"]),_sf(x["c"])
        out.append(h-l if prev is None else max(h-l,abs(h-prev),abs(l-prev))); prev=cl
    return out

def atr_series(c,period=14):
    tr=true_range(c)
    if not tr:return []
    period=max(1,int(period))
    if len(tr)<period:return [sum(tr)/len(tr)]*len(tr)
    run=sum(tr[:period])/period; out=[run]*period
    for v in tr[period:]: run=(run*(period-1)+v)/period; out.append(run)
    return out

def linreg_slope(v):
    n=len(v)
    if n<3:return 0.0,0.0
    if np is not None:
        x=np.arange(n,dtype=float); y=np.asarray(v,dtype=float)
        slope,_=np.polyfit(x,y,1); yhat=slope*x+(y.mean()-slope*x.mean())
        ssres=float(np.sum((y-yhat)**2)); sst=float(np.sum((y-y.mean())**2)) or 1e-12
        return float(slope), max(0.0,min(1.0,1.0-ssres/sst))
    xm=(n-1)/2; ym=sum(v)/n; den=sum((i-xm)**2 for i in range(n)) or 1e-12
    slope=sum((i-xm)*(y-ym) for i,y in enumerate(v))/den
    return slope,0.0

def returns(v): return [0.0 if v[i-1]==0 else (v[i]-v[i-1])/v[i-1] for i in range(1,len(v))]
def corr(a,b):
    n=min(len(a),len(b))
    if n<5:return 0.0
    a,b=list(a[-n:]),list(b[-n:]); ma,mb=sum(a)/n,sum(b)/n
    va=sum((x-ma)**2 for x in a); vb=sum((x-mb)**2 for x in b)
    if va<=0 or vb<=0:return 0.0
    return sum((a[i]-ma)*(b[i]-mb) for i in range(n))/math.sqrt(va*vb)

def swing_points(c,left=2,right=2):
    h,l=_hi(c),_lo(c); n=len(c); out=[]; left=max(1,int(left)); right=max(1,int(right))
    for i in range(left,n-right):
        wh=h[i-left:i+right+1]; wl=l[i-left:i+right+1]
        if h[i]==max(wh) and wh.count(h[i])==1:out.append((i,h[i],"H"))
        if l[i]==min(wl) and wl.count(l[i])==1:out.append((i,l[i],"L"))
    return out

def equal_levels(swings,atr_val,tol):
    band=max(atr_val*tol,1e-12)
    def cluster(vals):
        vals=sorted(vals); out=[]; i=0
        while i<len(vals):
            g=[vals[i]]; j=i
            while j+1<len(vals) and vals[j+1]-vals[i]<=band:j+=1; g.append(vals[j])
            if len(g)>=2:out.append(sum(g)/len(g))
            i=j+1
        return out
    return {"equal_highs":cluster([p for _,p,t in swings if t=="H"]),"equal_lows":cluster([p for _,p,t in swings if t=="L"])}

def detect_sweep(c,lookback):
    if len(c)<8:return None
    w=c[-max(5,min(len(c)-2,int(lookback))):-1]; last=c[-1]
    ph=max(_hi(w)); pl=min(_lo(w))
    if last["h"]>ph and last["c"]<ph:
        wick=last["h"]-max(last["o"],last["c"]); return {"type":"BEARISH_SWEEP","level":ph,"wick":wick}
    if last["l"]<pl and last["c"]>pl:
        wick=min(last["o"],last["c"])-last["l"]; return {"type":"BULLISH_SWEEP","level":pl,"wick":wick}
    return None

def detect_displacement(c,atr,mult):
    if not c or atr<=0:return None
    x=c[-1]; body=abs(x["c"]-x["o"])
    if body>=atr*mult:return {"direction":"BUY" if x["c"]>x["o"] else "SELL","body":body,"strength":body/atr}
    return None

def detect_fvg(c,max_age=6):
    if len(c)<3:return None
    start=max(0,len(c)-max(1,int(max_age))-3)
    for i in range(len(c)-3,start-1,-1):
        a,b,d=c[i],c[i+1],c[i+2]
        if a["h"]<d["l"]:return {"type":"BULLISH_FVG","top":d["l"],"bottom":a["h"],"index":i+2,"age":len(c)-(i+2)}
        if a["l"]>d["h"]:return {"type":"BEARISH_FVG","top":a["l"],"bottom":d["h"],"index":i+2,"age":len(c)-(i+2)}
    return None

def classify_regime(c,params):
    if not c:return "SIDEWAYS"
    closes=_cl(c)[-params["trend_lookback"]:]; atrs=atr_series(c,params["atr_period"])
    if len(closes)<5:return "SIDEWAYS"
    slope,r2=linreg_slope(closes); avg=max(sum(closes)/len(closes),1e-12); atr=sum(atrs[-len(closes):])/max(1,len(atrs[-len(closes):]))
    vol=(atr/avg)*100; ns=(slope*len(closes))/avg
    if vol>3:return "HIGH_VOLATILITY"
    if vol<0.15:return "LOW_VOLATILITY"
    if ns>0.02 and r2>0.25:return "BULLISH_TREND"
    if ns<-0.02 and r2>0.25:return "BEARISH_TREND"
    return "SIDEWAYS"

def classify_session(ts_ms):
    h=time.gmtime(_sf(ts_ms,time.time()*1000)/1000).tm_hour
    return "ASIA" if h<7 else "LONDON" if h<13 else "NEWYORK" if h<21 else "OFF_HOURS"

def _nearest_target(levels,entry,direction):
    if direction=="BUY":
        xs=[x for x in levels if x>entry]; return min(xs) if xs else None
    xs=[x for x in levels if x<entry]; return max(xs) if xs else None

@dataclass
class Setup:
    pair:str; direction:str; entry:float; tp:float; sl:float; confidence:float
    reason:List[str]; components:Dict[str,float]; setup_type:str; regime:str; session:str
    atr:float; timestamp:float; strategy_version:str; threshold_passed:bool=True
    reference_levels:Dict[str,Any]=field(default_factory=dict)
    diagnostics:Dict[str,Any]=field(default_factory=dict)
    def to_dict(self):
        return {"pair":self.pair,"direction":self.direction,"entry":self.entry,"tp":self.tp,"sl":self.sl,
                "confidence":round(self.confidence,2),"reason":list(self.reason),"components":dict(self.components),
                "setup_type":self.setup_type,"regime":self.regime,"session":self.session,"atr":self.atr,
                "timestamp":self.timestamp,"strategy_version":self.strategy_version,"threshold_passed":self.threshold_passed,
                "reference_levels":self.reference_levels,"diagnostics":self.diagnostics}

def validate_geometry(direction,entry,sl,tp,tick_size=0.0,atr_val=0.0):
    if any(not math.isfinite(_sf(x,float("nan"))) or _sf(x)<=0 for x in (entry,sl,tp)):return False,"INVALID_PRICE"
    if direction=="BUY" and not (sl<entry<tp):return False,"BUY_REQUIRES_SL_LT_ENTRY_LT_TP"
    if direction=="SELL" and not (tp<entry<sl):return False,"SELL_REQUIRES_TP_LT_ENTRY_LT_SL"
    if tick_size and min(abs(entry-sl),abs(tp-entry))<tick_size:return False,"DISTANCE_TOO_SMALL"
    return True,"OK"

class Strategy:
    def __init__(self,params:Optional[Dict[str,Any]]=None):
        self.version="2.00"; self.params=dict(DEFAULT_PARAMS); self.params.update(params or {})
        self.version_history=[{"version":self.version,"timestamp":time.time(),"reason":"INITIAL_V2","old_params":None,"new_params":dict(self.params),"evidence":None}]
    def get_active_threshold(self):return float(self.params.get("ACTIVE_THRESHOLD",0.0))
    def apply_update(self,new_params,reason,evidence=None):
        old=dict(self.params); self.params.update(new_params)
        major,minor=self.version.split("."); self.version=f"{major}.{int(minor)+1:02d}"
        rec={"version":self.version,"timestamp":time.time(),"reason":reason,"old_params":old,"new_params":dict(self.params),"evidence":evidence}
        self.version_history.append(rec); return rec
    def rollback(self):
        if len(self.version_history)<2:return None
        self.version_history.pop(); prev=self.version_history[-1]; self.params=dict(prev["new_params"]); self.version=prev["version"]; return prev
    def export_state(self):return {"schema":STRATEGY_SCHEMA,"version":self.version,"params":dict(self.params),"version_history":list(self.version_history)}
    def load_state(self,state):
        if not state:return
        self.version=state.get("version",self.version); self.params.update(state.get("params",{}))
        if isinstance(state.get("version_history"),list) and state["version_history"]:self.version_history=list(state["version_history"])

    def analyze_with_diagnostics(self,symbol,candles,btc_candles=None,market_context=None,enforce_threshold=False):
        p=self.params; c=list(_confirmed(candles)); min_len=max(p["structure_lookback"],p["vol_regime_lookback"],p["atr_period"])+5
        ok,why=validate_candles(c,min_len)
        if not ok:return None,{"status":"NO_DATA","reason":why}
        atrs=atr_series(c,p["atr_period"]); atr=atrs[-1]
        if atr<=0:return None,{"status":"INVALID_VOLATILITY","reason":"ATR_ZERO"}
        closes=_cl(c); last=closes[-1]
        struct=c[-p["structure_lookback"]:]; swings=swing_points(struct,p["swing_left"],p["swing_right"])
        highs=[x for x in swings if x[2]=="H"]; lows=[x for x in swings if x[2]=="L"]
        slope,r2=linreg_slope(closes[-p["trend_lookback"]:]); trend="BUY" if slope>0 else "SELL" if slope<0 else "NEUTRAL"
        last_high=highs[-1][1] if highs else None; last_low=lows[-1][1] if lows else None
        last_high_i=highs[-1][0] if highs else -999; last_low_i=lows[-1][0] if lows else -999
        bos="BOS_UP" if last_high is not None and last>last_high else "BOS_DOWN" if last_low is not None and last<last_low else None
        setup_age_bars=max(0, len(struct)-1-(last_high_i if bos=="BOS_UP" else last_low_i if bos=="BOS_DOWN" else len(struct)-1))
        if not bos:return None,{"status":"NO_VALID_ENTRY_CANDIDATE","reason":"NO_STRUCTURE_BREAK","trend":trend,"atr":atr}
        direction="BUY" if bos=="BOS_UP" else "SELL"; reasons=[f"{bos}",f"trend={trend}"]
        levels=equal_levels(swings,atr,p["equal_level_tol_atr"]); sweep=detect_sweep(c,p["sweep_lookback"]); disp=detect_displacement(c,atr,p["displacement_atr_mult"]); fvg=detect_fvg(c,p["fvg_max_age_bars"])
        liq=0.0
        if sweep and ((direction=="BUY" and sweep["type"]=="BULLISH_SWEEP") or (direction=="SELL" and sweep["type"]=="BEARISH_SWEEP")):
            liq+=CONFIDENCE_WEIGHTS["liquidity"]*0.75; reasons.append("directional liquidity sweep")
        pool=levels["equal_highs"] if direction=="BUY" else levels["equal_lows"]
        if pool: liq+=CONFIDENCE_WEIGHTS["liquidity"]*0.25; reasons.append("equal liquidity pool")
        iq=0.0; parts=["SMC_BOS"]
        if disp and disp["direction"]==direction: iq+=CONFIDENCE_WEIGHTS["entry_quality"]*.55; parts.append("DISPLACEMENT"); reasons.append("directional displacement")
        if fvg and ((direction=="BUY" and fvg["type"]=="BULLISH_FVG") or (direction=="SELL" and fvg["type"]=="BEARISH_FVG")):
            iq+=CONFIDENCE_WEIGHTS["entry_quality"]*.45; parts.append("FVG"); reasons.append("fresh directional FVG")
        fib=p["entry_retracement_fib"]; minoff=atr*p["entry_min_offset_atr"]
        if direction=="BUY":
            leg_low=last_low if last_low is not None else last-2*atr; rng=max(last-leg_low,atr*1e-6); entry=last-rng*fib; entry=min(entry,last-minoff); entry=max(entry,leg_low+0.05*atr)
            sl=min(leg_low,entry-0.5*atr)-atr*p["sl_atr_buffer"]; target=_nearest_target(levels["equal_highs"],entry,direction); tp=target if target else entry+2*abs(entry-sl)
        else:
            leg_high=last_high if last_high is not None else last+2*atr; rng=max(leg_high-last,atr*1e-6); entry=last+rng*fib; entry=max(entry,last+minoff); entry=min(entry,leg_high-0.05*atr)
            sl=max(leg_high,entry+0.5*atr)+atr*p["sl_atr_buffer"]; target=_nearest_target(levels["equal_lows"],entry,direction); tp=target if target else entry-2*abs(sl-entry)
        geom,greason=validate_geometry(direction,entry,sl,tp,atr_val=atr)
        if not geom:return None,{"status":"INVALID_GEOMETRY","reason":greason,"direction":direction}
        risk=abs(entry-sl); reward=abs(tp-entry); rr=reward/risk if risk else 0
        if rr<p["min_rr"]:return None,{"status":"LOW_EXPECTED_VALUE","reason":f"RR {rr:.2f}<min {p['min_rr']:.2f}","rr":rr}
        rrscore=CONFIDENCE_WEIGHTS["risk_reward"]*min(1,rr/max(2,p["min_rr"]*1.5))
        mlb=p["momentum_lookback"]; roc=(last-closes[-mlb-1])/closes[-mlb-1] if len(closes)>mlb and closes[-mlb-1] else 0
        malign=(direction=="BUY" and roc>0) or (direction=="SELL" and roc<0); mscore=CONFIDENCE_WEIGHTS["momentum"]*min(1,abs(roc)*20) if malign else 0
        vr=atrs[-p["vol_regime_lookback"]:] if len(atrs)>=p["vol_regime_lookback"] else atrs; vrank=sum(x<=atr for x in vr)/max(1,len(vr)); vscore=CONFIDENCE_WEIGHTS["volatility"]*(1-abs(vrank-.5)*2)
        btcdiag={"available":bool(btc_candles)}; btcscore=0.0
        if btc_candles:
            bc=list(_confirmed(btc_candles)); bcl=_cl(bc); br=returns(bcl[-p["btc_corr_lookback"]:]); sr=returns(closes[-p["btc_corr_lookback"]:]); cr=corr(sr,br); bslope,_=linreg_slope(bcl[-p["trend_lookback"]:]); bdir="BUY" if bslope>0 else "SELL" if bslope<0 else "NEUTRAL"; btcscore=CONFIDENCE_WEIGHTS["btc_context"]*(.75 if bdir==direction else .2); btcscore*=min(1,.4+abs(cr)); btcdiag.update({"trend":bdir,"correlation":cr,"aligned":bdir==direction})
        regime=classify_regime(btc_candles if btc_candles else c,p); rscore=CONFIDENCE_WEIGHTS["regime"] if ((regime=="BULLISH_TREND" and direction=="BUY") or (regime=="BEARISH_TREND" and direction=="SELL")) else CONFIDENCE_WEIGHTS["regime"]*.35 if regime=="SIDEWAYS" else 0
        session=classify_session(c[-1]["t"]); sscore=CONFIDENCE_WEIGHTS["session"] if session in ("LONDON","NEWYORK") else CONFIDENCE_WEIGHTS["session"]*.3
        setup_age=setup_age_bars if bos else 999; freshness=max(0.0,1-(setup_age/max(1,p["setup_max_age_bars"]))); fscore=CONFIDENCE_WEIGHTS["freshness"]*freshness
        current_distance=abs(last-entry)/max(atr,1e-12); fill_viability=max(0.0,1.0-min(1.0,current_distance/4.0)); escore=CONFIDENCE_WEIGHTS["execution_viability"]*fill_viability
        structure_score=CONFIDENCE_WEIGHTS["structure"]*min(1,(0.45+.35*r2)*(1 if trend==direction else .45))
        confirm=sum(bool(x) for x in (sweep,disp,fvg,pool)); confscore=0
        components={"structure":structure_score,"liquidity":min(liq,CONFIDENCE_WEIGHTS["liquidity"]),"entry_quality":min(iq,CONFIDENCE_WEIGHTS["entry_quality"]),"risk_reward":rrscore,"momentum":mscore,"volatility":max(0,vscore),"btc_context":btcscore,"regime":rscore,"session":sscore,"freshness":fscore,"execution_viability":escore}
        confidence=max(0,min(100,sum(components.values())))
        reasons.append(f"entry pullback {fib:.3f} retracement; current_distance={current_distance:.2f} ATR")
        diagnostics={"status":"VALID_LOW_CONF","data_age_bars":setup_age,"setup_age_bars":setup_age,"data_points":len(c),"atr":atr,"structure":{"bos":bos,"trend":trend,"r2":r2,"swing_count":len(swings)},"liquidity":{"sweep":sweep,"pools":levels},"entry":{"current_price":last,"distance_atr":current_distance,"fill_probability_proxy":fill_viability},"tp":{"target_liquidity":target,"rr":rr,"risk":risk,"reward":reward},"sl":{"structural_level":last_low if direction=="BUY" else last_high,"atr_buffer":p["sl_atr_buffer"]},"btc":btcdiag,"regime":regime,"session":session,"confidence_components":components}
        passed=confidence>=self.get_active_threshold(); diagnostics["status"]="VALID_HIGH_CONF" if passed else "VALID_LOW_CONF"
        setup=Setup(symbol,direction,entry,tp,sl,confidence,reasons,components,"+".join(parts),regime,session,atr,c[-1]["t"],self.version,passed,{"bos":bos,"equal_highs":levels["equal_highs"][-5:],"equal_lows":levels["equal_lows"][-5:],"sweep":sweep,"fvg":fvg,"rr":rr},diagnostics)
        if enforce_threshold and not passed:return None,diagnostics
        return setup,diagnostics

    def analyze(self,symbol,candles,btc_candles=None,enforce_threshold=True):
        setup,_=self.analyze_with_diagnostics(symbol,candles,btc_candles,None,enforce_threshold); return setup

    def monitor_position(self,position,candles,btc_candles=None,market_context=None):
        c=list(_confirmed(candles)); p=self.params
        if len(c)<p["atr_period"]+5:return {"action":"HOLD","new_sl":None,"reason":["insufficient data"],"weakness_score":0,"engine":"none"}
        ok,why=validate_candles(c,p["atr_period"]+5)
        if not ok:return {"action":"HOLD","new_sl":None,"reason":[why],"weakness_score":0,"engine":"validation"}
        atr=atr_series(c,p["atr_period"])[-1]; direction=position["direction"]; entry=_sf(position.get("fill_price",position.get("entry"))); current_sl=_sf(position.get("sl")); tp=_sf(position.get("tp")); price=_cl(c)[-1]
        initrisk=abs(entry-_sf(position.get("initial_sl",current_sl))) or atr; profit_r=((price-entry)/initrisk if direction=="BUY" else (entry-price)/initrisk)
        weakness=0; reasons=[]; swings=swing_points(c[-p["structure_lookback"]:],p["swing_left"],p["swing_right"])
        recent_cl=_cl(c[-p["momentum_lookback"]-1:]); slope,_=linreg_slope(recent_cl); aligned=(direction=="BUY" and slope>0) or (direction=="SELL" and slope<0)
        if not aligned:weakness+=1; reasons.append("structure/momentum weakening")
        last=c[-1]; opp=(direction=="BUY" and last["c"]<last["o"]) or (direction=="SELL" and last["c"]>last["o"])
        if opp:weakness+=1; reasons.append("opposite candle")
        fill_ms=_sf(position.get("fill_time",0)); path=[x for x in c if fill_ms<=0 or _sf(x.get("t"))>=fill_ms] or c[-20:]
        peak=max(_hi(path)) if direction=="BUY" else min(_lo(path)); giveback=((peak-price)/atr if direction=="BUY" else (price-peak)/atr)
        if giveback>.5:weakness+=1; reasons.append("giveback")
        if giveback>1.2:weakness+=1; reasons.append("deep giveback")
        # BTC conflict can increase caution but never force trail alone
        btc_conflict=False
        if btc_candles:
            bs,_=linreg_slope(_cl(btc_candles)[-p["trend_lookback"]:]); bdir="BUY" if bs>0 else "SELL" if bs<0 else "NEUTRAL"; btc_conflict=(bdir not in ("NEUTRAL",direction)); btc_conflict and (reasons.append("BTC context conflicts"))
            if btc_conflict:weakness+=1
        candidate=None; trail_reason=[]
        if profit_r>=p["trail_min_profit_r"] and weakness>=p["trail_weakness_threshold"]:
            rsw=swing_points(path,p["swing_left"],p["swing_right"]) if len(path)>=p["swing_left"]+p["swing_right"]+3 else []
            if direction=="BUY":
                lows=[v for _,v,t in rsw if t=="L"]; structural=max(lows[-3:]) if lows else price-atr*p["sl_atr_buffer"]; candidate=structural-atr*p["sl_atr_buffer"]*.5; candidate=min(candidate,price-atr*.05)
                if candidate<=current_sl:candidate=None
            else:
                highs=[v for _,v,t in rsw if t=="H"]; structural=min(highs[-3:]) if highs else price+atr*p["sl_atr_buffer"]; candidate=structural+atr*p["sl_atr_buffer"]*.5; candidate=max(candidate,price+atr*.05)
                if candidate>=current_sl:candidate=None
            if candidate is not None:
                step_r=abs(candidate-entry)/initrisk
                if step_r>=p["trail_min_step_r"]:trail_reason=["profit threshold","structure checkpoint","weakness confirmed"]; return {"action":"TRAIL","new_sl":candidate,"reason":reasons+trail_reason,"weakness_score":weakness,"engine":"structure+momentum","profit_r":profit_r,"giveback_atr":giveback,"trail_step_r":step_r,"btc_conflict":btc_conflict}
        return {"action":"HOLD","new_sl":None,"reason":reasons or ["structure intact"],"weakness_score":weakness,"engine":"structure+momentum","profit_r":profit_r,"giveback_atr":giveback,"btc_conflict":btc_conflict}

def new_default_strategy():return Strategy()
