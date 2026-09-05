"""learn.py — Adaptive Learning Brain vNext.
No exchange/API access. It only processes data/events supplied by main.py.
Ollama is optional local advisory only; its output can never directly change strategy.
"""
from __future__ import annotations
import hashlib, json, logging, math, os, shutil, subprocess, time
from statistics import median, pstdev
from threading import RLock
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
try:
    import requests
except ImportError:
    requests=None

logger=logging.getLogger("learn")
SCHEMA_VERSION=6
HALF_LIFE_DAYS=21.0
OUTCOMES=("TP","INITIAL_SL","TRAIL","BE","TIMEOUT")
ECONOMIC=("TP","INITIAL_SL","TRAIL","BE")

def _log(event, **fields):
    text="[LEARN] "+event
    if fields:text += " | "+" | ".join(f"{k}={v}" for k,v in fields.items())
    logger.info(text)

def sf(x,d=0.0):
    try:v=float(x); return v if math.isfinite(v) else d
    except (TypeError,ValueError):return d

def si(x,d=0):
    try:return int(x)
    except (TypeError,ValueError):return d

def weight(ts,now=None,half_life=HALF_LIFE_DAYS):
    now=time.time() if now is None else now
    age=max(0,(now-sf(ts,now))/86400.0); return 0.5**(age/max(.1,half_life))

def bucket(c):
    c=max(0,min(100,sf(c)))
    lo=int(c//10)*10
    if c<40:return f"0-{int(c)//10*10+9}"
    if c>=90:return "90-100"
    return f"{lo}-{lo+9}"

def atomic_write(path,data):
    tmp=path+".tmp"
    with open(tmp,"w",encoding="utf-8") as f:
        json.dump(data,f,indent=2,ensure_ascii=False,allow_nan=False,default=str); f.flush(); os.fsync(f.fileno())
    os.replace(tmp,path)

def validate_json_safe(obj):
    def walk(v):
        if isinstance(v,float) and not math.isfinite(v):return False
        if isinstance(v,dict):return all(walk(k) and walk(x) for k,x in v.items())
        if isinstance(v,list):return all(walk(x) for x in v)
        return True
    return walk(obj)

class LearnEngine:
    def __init__(self,checkpoint_path="state/learn_checkpoint.json",backup_path=None,ollama_url=None,ollama_api_key=None,git_enabled=False,git_repo_dir=None):
        self.checkpoint_path=checkpoint_path; self.backup_path=backup_path or checkpoint_path+".backup"
        self.ollama_url=ollama_url or os.environ.get("OLLAMA_URL",""); self.ollama_api_key=ollama_api_key or os.environ.get("OLLAMA_API_KEY","")
        self.git_enabled=bool(git_enabled); self.git_repo_dir=git_repo_dir or "."; self._lock=RLock(); self._schema_version=SCHEMA_VERSION
        self.raw_events=[]; self.trade_history=[]; self.scan_summaries=[]; self.candidate_history=[]; self.shadow_history=[]
        self.market_history=[]; self.position_diagnostics=[]; self.feature_cache={}; self.calibration_cache={}; self.frequency_cache={}; self.quality_cache={}; self.exit_cache={}; self.btc_cache={}
        self.threshold_history=[]; self.strategy_change_log=[]; self.decision_history=[]; self.pending_challenger=None; self.last_audit_report={}; self.strategy_state={}; self.current_strategy_version=None
        self.trades_since_last_change=0; self.last_change_ts=0.; self.last_audit_ts=0.; self.last_autosave_ts=0.
        os.makedirs(os.path.dirname(self.checkpoint_path) or ".",exist_ok=True)

    def _export_state(self):
        with self._lock:
            data={"schema_version":self._schema_version,"saved_at":time.time(),"trade_history":self.trade_history[-8000:],"scan_summaries":self.scan_summaries[-5000:],"candidate_history":self.candidate_history[-30000:],"shadow_history":self.shadow_history[-30000:],"market_history":self.market_history[-5000:],"position_diagnostics":self.position_diagnostics[-8000:],"feature_cache":self.feature_cache,"calibration_cache":self.calibration_cache,"frequency_cache":self.frequency_cache,"quality_cache":self.quality_cache,"exit_cache":self.exit_cache,"btc_cache":self.btc_cache,"threshold_history":self.threshold_history[-2000:],"strategy_change_log":self.strategy_change_log[-2000:],"decision_history":self.decision_history[-3000:],"pending_challenger":self.pending_challenger,"last_audit_report":self.last_audit_report,"strategy_state":self.strategy_state,"current_strategy_version":self.current_strategy_version,"trades_since_last_change":self.trades_since_last_change,"last_change_ts":self.last_change_ts,"last_audit_ts":self.last_audit_ts}
            if not validate_json_safe(data):raise ValueError("non-finite value in learn state")
            return data
    def _restore(self,d):
        if not isinstance(d,dict):raise ValueError("checkpoint bukan object")
        for k,default in [("trade_history",[]),("scan_summaries",[]),("candidate_history",[]),("shadow_history",[]),("market_history",[]),("position_diagnostics",[]),("threshold_history",[]),("strategy_change_log",[]),("decision_history",[])]:setattr(self,k,list(d.get(k,default)))
        for k in ("feature_cache","calibration_cache","frequency_cache","quality_cache","exit_cache","btc_cache","last_audit_report","strategy_state") : setattr(self,k,dict(d.get(k,{})))
        self.pending_challenger=d.get("pending_challenger"); self.current_strategy_version=d.get("current_strategy_version"); self.trades_since_last_change=si(d.get("trades_since_last_change")); self.last_change_ts=sf(d.get("last_change_ts")); self.last_audit_ts=sf(d.get("last_audit_ts"))
    def load(self):
        _log("LOAD START")
        with self._lock:
            for path,label in ((self.checkpoint_path,"primary"),(self.backup_path,"backup")):
                if not os.path.exists(path):continue
                try:
                    with open(path,encoding="utf-8") as f:d=json.load(f)
                    self._restore(d); _log("LOAD OK",source=label,trades=len(self.trade_history),candidates=len(self.candidate_history),shadow=len(self.shadow_history)); return label
                except Exception as e:logger.warning("learn checkpoint %s invalid: %s",label,e)
        _log("LOAD EMPTY"); return "empty"
    def save_checkpoint(self):
        _log("SAVE START",trades=len(self.trade_history),events=len(self.raw_events))
        try:
            with self._lock:
                data=self._export_state()
                if os.path.exists(self.checkpoint_path):shutil.copyfile(self.checkpoint_path,self.backup_path)
                atomic_write(self.checkpoint_path,data); self.last_autosave_ts=time.time()
            _log("SAVE OK",path=self.checkpoint_path,backup=self.backup_path)
            return True
        except Exception as e:logger.error("learn save failed: %s",e); return False
    def autosave(self):
        _log("AUTOSAVE START")
        ok=self.save_checkpoint()
        if ok and self.git_enabled:self._git_mirror()
        _log("AUTOSAVE DONE",local=ok,git=self.git_enabled)
    def _git_mirror(self):
        try:
            rel=os.path.relpath(os.path.abspath(self.checkpoint_path),os.path.abspath(self.git_repo_dir))
            r=subprocess.run(["git","add","--",rel],cwd=self.git_repo_dir,capture_output=True,timeout=5); _=r
            subprocess.run(["git","commit","-m",f"autosave learn {time.strftime('%Y-%m-%d %H:%M:%S')}"],cwd=self.git_repo_dir,capture_output=True,timeout=5)
            subprocess.run(["git","push"],cwd=self.git_repo_dir,capture_output=True,timeout=15)
        except Exception as e:logger.warning("learn git mirror failed: %s",e)
    def _event(self,kind,payload):
        e=dict(payload); e["kind"]=kind; e.setdefault("timestamp",time.time()); self.raw_events.append(e)
        if len(self.raw_events)>20000:del self.raw_events[:-15000]
    def record_scan_context(self,summary):
        _log("SCAN CONTEXT",processed=summary.get("processed"),candidate=summary.get("candidate"),eligible=summary.get("eligible"),breadth_buy=summary.get("breadth_buy"),regime=summary.get("regime"))
        with self._lock:self.scan_summaries.append(dict(summary)); self._event("SCAN_SUMMARY",summary)
    def record_market_snapshot(self,snapshot):
        with self._lock:
            row=dict(snapshot); row.setdefault("timestamp",time.time()); self.market_history.append(row); self._event("MARKET_SNAPSHOT",row)
    def record_scan_summary(self,summary):self.record_scan_context(summary)
    def record_scan_candidate(self,setup,eligible,threshold,reason=""):
        row=dict(setup); row.update({"eligible":bool(eligible),"threshold":sf(threshold),"eligibility_reason":reason,"timestamp":time.time()})
        _log("CANDIDATE",coin=row.get("pair"),direction=row.get("direction"),confidence=round(sf(row.get("confidence")),1),eligible=eligible,reason=reason)
        with self._lock:self.candidate_history.append(row); self._event("CANDIDATE",row)
    def record_shadow_outcome(self,candidate,outcome,pnl_r=0.,**extra):
        row=dict(candidate); row.update(extra); row.update({"kind":"SHADOW_OUTCOME","outcome":outcome if outcome in OUTCOMES else "TIMEOUT","pnl_r":sf(pnl_r),"timestamp":time.time()})
        _log("SHADOW OUTCOME",coin=row.get("pair"),outcome=row.get("outcome"),pnl_r=round(sf(pnl_r),3))
        with self._lock:self.shadow_history.append(row); self._event("SHADOW_OUTCOME",row)
    def record_trade_outcome(self,setup,outcome,close_info):
        row={"pair":setup.get("pair"),"direction":setup.get("direction"),"confidence":sf(setup.get("confidence")),"bucket":bucket(setup.get("confidence")),"setup_type":setup.get("setup_type","UNKNOWN"),"regime":setup.get("regime","UNKNOWN"),"session":setup.get("session","UNKNOWN"),"components":dict(setup.get("components",{})),"diagnostics":dict(setup.get("diagnostics",{})),"reference_levels":dict(setup.get("reference_levels",{})),"strategy_version":setup.get("strategy_version"),"outcome":outcome if outcome in OUTCOMES else "BE","pnl_pct":sf(close_info.get("pnl_pct")),"pnl_r":sf(close_info.get("pnl_r")),"trail_count":si(close_info.get("trail_count")),"entry_time":setup.get("timestamp"),"close_time":close_info.get("close_time",time.time()*1000),"timestamp":time.time(),"close_info":dict(close_info)}
        _log("TRADE OUTCOME",coin=row["pair"],outcome=row["outcome"],pnl_r=round(row["pnl_r"],3),trail=row["trail_count"])
        with self._lock:self.trade_history.append(row); self.position_diagnostics.append(row); self.trades_since_last_change+=1; self._event("TRADE_OUTCOME",row); self._update_feature_cache_locked()
    def set_strategy_state(self,state):
        with self._lock:self.strategy_state=dict(state or {}); self.current_strategy_version=self.strategy_state.get("version",self.current_strategy_version); self._event("STRATEGY_STATE",self.strategy_state)
    @staticmethod
    def _economic(rows):return [r for r in rows if r.get("outcome") in ECONOMIC]
    def weighted_stats(self,rows,half_life=HALF_LIFE_DAYS):
        xs=self._economic(rows); now=time.time()
        if not xs:return {"n":0,"effective_n":0.,"win_rate":0.,"expectancy":0.,"profit_factor":0.,"median_r":0.,"avg_win":0.,"avg_loss":0.}
        ww=[(sf(r.get("pnl_r")),weight(r.get("timestamp"),now,half_life)) for r in xs]; tw=sum(w for _,w in ww) or 1e-9; wins=[(p,w) for p,w in ww if p>0]; losses=[(p,w) for p,w in ww if p<0]
        gw=sum(p*w for p,w in wins); gl=abs(sum(p*w for p,w in losses)); wr=sum(w for p,w in wins)/sum(w for p,w in ww if p!=0 or True)
        return {"n":len(xs),"effective_n":round(tw,2),"win_rate":round(wr*100,2),"expectancy":round(sum(p*w for p,w in ww)/tw,4),"profit_factor":round(gw/gl,3) if gl else (999. if gw else 0.),"median_r":round(median([p for p,_ in ww]),4),"avg_win":round(gw/(sum(w for _,w in wins) or 1e-9),4),"avg_loss":round(-gl/(sum(w for _,w in losses) or 1e-9),4)}
    def _update_feature_cache_locked(self):
        rows=self.trade_history; self.feature_cache={"overall":self.weighted_stats(rows),"regime":self._group_stats(rows,"regime"),"session":self._group_stats(rows,"session"),"direction":self._group_stats(rows,"direction"),"setup_type":self._group_stats(rows,"setup_type")}
    def _group_stats(self,rows,key):
        groups={}
        for r in rows:groups.setdefault(str(r.get(key,"UNKNOWN")),[]).append(r)
        return {k:self.weighted_stats(v) for k,v in groups.items()}
    def confidence_calibration(self):
        with self._lock:
            groups={}
            for r in self.trade_history:groups.setdefault(bucket(r.get("confidence")),[]).append(r)
            self.calibration_cache={k:self.weighted_stats(v) for k,v in groups.items()}; return dict(self.calibration_cache)
    def frequency_diagnosis(self,window=100):
        with self._lock:
            scans=self.scan_summaries[-window:]
            if not scans:return {"status":"NO_DATA","n_scans":0}
            n=len(scans); cand=sum(sf(x.get("candidate")) for x in scans); elig=sum(sf(x.get("eligible")) for x in scans); proc=sum(sf(x.get("processed")) for x in scans)
            avg_c=cand/n; avg_e=elig/n; avg_p=proc/n; reject=sum(sf((x.get("rejects") or {}).get("BELOW_ACTIVE_THRESHOLD")) for x in scans)
            statuses="HEALTHY" if avg_c>=2 and avg_e>=.5 else "LOW_FREQUENCY" if avg_c<2 else "THRESHOLD_SUPPRESSING" if reject/max(cand+reject,1)>.5 else "HEALTHY"
            out={"status":statuses,"n_scans":n,"avg_processed":round(avg_p,2),"avg_candidate":round(avg_c,2),"avg_eligible":round(avg_e,2),"threshold_rejected":int(reject),"candidate_rate_per_scan":round(avg_c,3),"eligible_rate_per_scan":round(avg_e,3)}; self.frequency_cache=out; return out
    def shadow_performance_below(self,threshold):
        rows=[r for r in self.shadow_history if sf(r.get("confidence"))<threshold and r.get("outcome") in ECONOMIC]; return self.weighted_stats(rows,14.0)
    def exit_attribution(self):
        with self._lock:
            out={}
            for kind in OUTCOMES:
                rows=[r for r in self.trade_history if r.get("outcome")==kind]
                st=self.weighted_stats(rows); maes=[]; mfes=[]; trail_gain=[]; ambiguities=0
                for r in rows:
                    ci=r.get("close_info",{}) or {}
                    path=ci.get("path_candles") or []
                    setup=dict(r)
                    replay=self.replay_fixed_levels(setup,path,ci.get("trail_history") or []) if path else None
                    no_trail=self.replay_fixed_levels(setup,path,[]) if path else None
                    if replay:
                        maes.append(replay["mae_r"]); mfes.append(replay["mfe_r"]); ambiguities += int(replay["ambiguous"])
                        if no_trail: trail_gain.append(replay["pnl_r"]-no_trail["pnl_r"])
                    else:
                        maes.append(sf(ci.get("mae_r"))); mfes.append(sf(ci.get("mfe_r")))
                out[kind]={"stats":st,"count":len(rows),"avg_mae_r":round(sum(maes)/len(maes),3) if maes else 0.,"avg_mfe_r":round(sum(mfes)/len(mfes),3) if mfes else 0.,"trail_counterfactual_delta_r":round(sum(trail_gain)/len(trail_gain),3) if trail_gain else 0.,"ambiguous_paths":ambiguities}
            self.exit_cache=out; return out

    def drawdown(self):
        eq=0.; peak=0.; maxdd=0.; curve=[]
        for r in sorted(self._economic(self.trade_history),key=lambda x:sf(x.get("timestamp"))):
            eq+=sf(r.get("pnl_r")); peak=max(peak,eq); dd=peak-eq; maxdd=max(maxdd,dd); curve.append(eq)
        return {"max_drawdown_r":round(maxdd,4),"equity_r":round(eq,4),"curve_points":len(curve)}
    def direction_breadth(self,window=100):
        rows=self.candidate_history[-window:]; buy=sum(1 for r in rows if r.get("direction")=="BUY"); sell=sum(1 for r in rows if r.get("direction")=="SELL"); n=buy+sell
        return {"buy":buy,"sell":sell,"total":n,"buy_pct":round(100*buy/n,2) if n else 0,"sell_pct":round(100*sell/n,2) if n else 0}
    def freshness_diagnosis(self):
        rows=self.candidate_history[-5000:]
        ages=[]
        for r in rows:
            d=(r.get("diagnostics") or {})
            ages.append(sf(d.get("data_age_bars", d.get("setup_age_bars", 0))))
        stale=sum(1 for x in ages if x>8)
        return {"n":len(ages),"stale":stale,"stale_pct":round(100*stale/len(ages),2) if ages else 0.,"median_age_bars":round(median(ages),2) if ages else 0.}

    def _quality_quantity_matrix(self,freq):
        q=self.weighted_stats(self.trade_history); exp=q["expectancy"]; pf=q["profit_factor"]; dd=self.drawdown()["max_drawdown_r"]; f=freq.get("avg_candidate",0)
        quality="HIGH" if q["n"]>=20 and exp>0.15 and pf>=1.2 and dd<5 else "LOW" if q["n"]>=20 and (exp<0 or pf<1) else "UNKNOWN"
        quantity="HIGH" if f>=5 else "MEDIUM" if f>=2 else "LOW"
        decision="KEEP" if quality=="HIGH" and quantity in ("HIGH","MEDIUM") else "IMPROVE_FREQUENCY" if quality=="HIGH" and quantity=="LOW" else "IMPROVE_QUALITY" if quality=="LOW" and quantity in ("HIGH","MEDIUM") else "INVESTIGATE_MODEL"
        return {"quality":quality,"quantity":quantity,"decision":decision,"expectancy":exp,"profit_factor":pf,"drawdown_r":dd,"candidate_rate":f}
    def counterfactual_threshold(self,rows,threshold):
        selected=[r for r in rows if sf(r.get("confidence"))>=threshold and r.get("outcome") in ECONOMIC]
        b=self.weighted_stats(rows); c=self.weighted_stats(selected); return {"baseline":b,"challenger":c,"selected_n":c["n"],"delta_expectancy":round(c["expectancy"]-b["expectancy"],4)}
    def _chronological_split(self,rows,frac=.25):
        o=sorted(rows,key=lambda r:sf(r.get("timestamp"))); cut=max(1,int(len(o)*(1-frac))); return o[:cut],o[cut:]
    def _ollama_critique(self,context):
        if not self.ollama_url or requests is None:return None
        _log("OLLAMA START")
        prompt=("You are a statistical trading critic. Do not make a trading decision. "
                "Find blind spots, contradictions, confounds, data freshness issues, and quality-vs-frequency tradeoffs. "
                "Return at most 6 concise observations.\n"+json.dumps(context,ensure_ascii=False,default=str)[:12000])
        try:
            headers={"Content-Type":"application/json"};
            if self.ollama_api_key:headers["Authorization"]="Bearer "+self.ollama_api_key
            r=requests.post(self.ollama_url.rstrip("/")+"/api/generate",headers=headers,json={"model":os.environ.get("OLLAMA_MODEL","llama3"),"prompt":prompt,"stream":False},timeout=20)
            if r.status_code==200:
                ans=str((r.json() or {}).get("response","")).strip()[:4000]; _log("OLLAMA DONE",chars=len(ans)); return ans
        except Exception as e:logger.warning("ollama critic unavailable: %s",e)
        return None
    def evaluate_current_version_degradation(self):
        current=self.current_strategy_version
        rows=[r for r in self.trade_history if r.get("strategy_version")==current]
        prior=[r for r in self.trade_history if r.get("strategy_version")!=current]
        a=self.weighted_stats(rows); b=self.weighted_stats(prior)
        status="DEGRADED" if a["n"]>=15 and b["n"]>=15 and a["expectancy"]<b["expectancy"]-0.30 else "OK"
        return {"status":status,"current":a,"baseline":b}
    def _proposal(self,strategy_engine,freq,cal):
        current=strategy_engine.get_active_threshold(); shadow=self.shadow_performance_below(current)
        if freq.get("status") in ("LOW_FREQUENCY","THRESHOLD_SUPPRESSING") and shadow.get("n",0)>=20 and shadow.get("expectancy",0)>0.10 and current>0:
            return round(max(0,current-3),1),{"type":"LOWER_THRESHOLD","shadow":shadow,"frequency":freq}
        usable=sorted([(k,v) for k,v in cal.items() if si(v.get("n"))>=20],key=lambda x:int(x[0].split("-")[0]))
        low=[x for x in usable if int(x[0].split("-")[0])<=current+10 and x[1].get("expectancy",0)<-.05]; high=[x for x in usable if int(x[0].split("-")[0])>current and x[1].get("expectancy",0)>.05]
        if low and high:
            new=round(min(95,current+min(5,max(1,int(high[0][0].split("-")[0])-current))),1); return new,{"type":"RAISE_THRESHOLD","low":low,"high":high,"frequency":freq}
        return None
    def audit(self,strategy_engine):
        started=time.time(); _log("AUDIT START",trades=len(self.trade_history),candidates=len(self.candidate_history),shadow=len(self.shadow_history),strategy=getattr(strategy_engine,"version","?"))
        with self._lock:
            freq=self.frequency_diagnosis(); cal=self.confidence_calibration(); self._update_feature_cache_locked(); quality=self._quality_quantity_matrix(freq); exit_stats=self.exit_attribution(); dd=self.drawdown(); breadth=self.direction_breadth(); freshness=self.freshness_diagnosis(); degradation=self.evaluate_current_version_degradation()
            report={"timestamp":time.time(),"strategy_version":getattr(strategy_engine,"version",None),"total_trades":len(self.trade_history),"frequency":freq,"calibration":cal,"quality_quantity":quality,"quality":self.feature_cache,"exit_attribution":exit_stats,"drawdown":dd,"direction_breadth":breadth,"freshness":freshness,"version_health":degradation,"action":"NO_ACTION"}
            _log("DATA QUALITY DONE",valid_events=len(self.raw_events)); _log("QUALITY DONE",expectancy=quality.get("expectancy"),pf=quality.get("profit_factor"),dd=dd.get("max_drawdown_r")); _log("FREQUENCY DONE",status=freq.get("status"),candidate_rate=freq.get("avg_candidate"),eligible_rate=freq.get("avg_eligible")); _log("EXIT ATTRIBUTION DONE",tp=exit_stats.get("TP",{}).get("count"),sl=exit_stats.get("INITIAL_SL",{}).get("count"),trail=exit_stats.get("TRAIL",{}).get("count"),trail_delta=exit_stats.get("TRAIL",{}).get("trail_counterfactual_delta_r")); _log("FRESHNESS DONE",stale_pct=freshness.get("stale_pct"),median_age=freshness.get("median_age_bars"))
            ollama=self._ollama_critique({"quality_quantity":quality,"frequency":freq,"drawdown":dd,"exit_attribution":exit_stats,"breadth":breadth,"freshness":freshness,"calibration":cal,"version_health":degradation}); report["ollama_critique"]=ollama
            if degradation.get("status")=="DEGRADED":report["action"]="ROLLBACK_RECOMMENDED"; report["reason"]="current strategy materially degraded"
            else:
                prop=self._proposal(strategy_engine,freq,cal)
                if prop:
                    new,evidence=prop; current=strategy_engine.get_active_threshold(); training,holdout=self._chronological_split(self.trade_history,.25); train=self.counterfactual_threshold(training,new); hs=[r for r in holdout if sf(r.get("confidence"))>=new and r.get("outcome") in ECONOMIC]; hb=[r for r in holdout if r.get("outcome") in ECONOMIC]; baseline=self.weighted_stats(hb); ch=self.weighted_stats(hs); delta=ch["expectancy"]-baseline["expectancy"] if baseline["n"] else 0; hold_ok=ch["n"]>=20 and delta>=-.10 and ch["profit_factor"]>=max(.9,baseline["profit_factor"]*.9)
                    evidence.update({"counterfactual_train":train,"holdout":{"baseline":baseline,"challenger":ch,"delta_expectancy":delta},"ollama":ollama})
                    _log("CHALLENGER",old=current,new=new,train_delta=train.get("delta_expectancy"),holdout=hold_ok)
                    if len(self.trade_history)<40:report["reason"]="sample belum cukup"
                    elif not hold_ok:report["action"]="DEFERRED"; report["reason"]="holdout gate failed"
                    else:
                        rec=strategy_engine.apply_update({"ACTIVE_THRESHOLD":new},reason=evidence.get("type","LEARNED_UPDATE"),evidence=evidence); self.threshold_history.append({"timestamp":time.time(),"old":current,"new":new,"evidence":evidence}); self.strategy_change_log.append(rec); self.decision_history.append({"type":"ACCEPTED","proposal":{"ACTIVE_THRESHOLD":new},"evidence":evidence}); self.trades_since_last_change=0; self.last_change_ts=time.time(); self.current_strategy_version=rec.get("version"); report.update({"action":"APPLIED","old_threshold":current,"new_threshold":new,"strategy_version":rec.get("version"),"evidence":evidence,"reason":"quality+frequency+holdout gates passed"})
                else:report["reason"]="no evidence-based parameter change"
            self.last_audit_report=report; self.last_audit_ts=time.time(); _log("AUDIT DECISION",action=report["action"],reason=report.get("reason"),elapsed=round(time.time()-started,2)); return report
    @staticmethod
    def confidence_interval(rate: float, n: int, z: float = 1.96) -> Dict[str, float]:
        """Wilson interval for a proportion; transparent uncertainty estimate."""
        if n <= 0:
            return {"low": 0.0, "high": 0.0}
        p = max(0.0, min(1.0, rate)); den = 1.0 + z*z/n
        center = (p + z*z/(2*n)) / den
        half = z * math.sqrt((p*(1-p) + z*z/(4*n))/n) / den
        return {"low": round(max(0.0, center-half), 4), "high": round(min(1.0, center+half), 4)}

    def aging_report(self) -> Dict[str, Any]:
        with self._lock:
            now = time.time()
            ages=[]
            for row in self.candidate_history:
                ts=sf(row.get("timestamp"), now); ages.append(max(0.0,(now-ts)/86400.0))
            if not ages:
                return {"count":0,"median_days":0.0,"recent_pct":0.0,"stale_pct":0.0}
            recent=sum(1 for a in ages if a<=3)/len(ages)
            stale=sum(1 for a in ages if a>21)/len(ages)
            return {"count":len(ages),"median_days":round(median(ages),3),"recent_pct":round(recent*100,2),"stale_pct":round(stale*100,2)}

    def risk_adjusted_quality(self) -> Dict[str, Any]:
        with self._lock:
            rows=[r for r in self.trade_history if r.get("outcome") in ECONOMIC]
            rs=[sf(r.get("pnl_r")) for r in rows]
            if not rs:return {"n":0,"median_r":0.0,"downside_deviation":0.0,"max_drawdown_r":0.0}
            downside=[min(0.0,x) for x in rs]
            downside_dev=(sum(x*x for x in downside)/len(downside))**0.5
            equity=peak=dd=0.0
            for x in rs:
                equity+=x; peak=max(peak,equity); dd=max(dd,peak-equity)
            return {"n":len(rs),"median_r":round(median(rs),4),"downside_deviation":round(downside_dev,4),"max_drawdown_r":round(dd,4)}

    def direction_breadth(self, window_scans: int = 100) -> Dict[str, Any]:
        with self._lock:
            rows=self.candidate_history[-max(1,window_scans):]
            buy=sum(1 for r in rows if r.get("direction")=="BUY")
            sell=sum(1 for r in rows if r.get("direction")=="SELL")
            n=buy+sell
            return {"buy":buy,"sell":sell,"total":n,"buy_pct":round(100*buy/max(1,n),2),"sell_pct":round(100*sell/max(1,n),2),"dominant":"BUY" if buy>sell else "SELL" if sell>buy else "NEUTRAL"}

    def exit_attribution_v2(self) -> Dict[str, Any]:
        with self._lock:
            out={}
            for outcome in OUTCOMES:
                rows=[r for r in self.trade_history if r.get("outcome")==outcome]
                rs=[sf(r.get("pnl_r")) for r in rows]
                out[outcome]={"count":len(rows),"avg_r":round(sum(rs)/len(rs),4) if rs else 0.0,"median_r":round(median(rs),4) if rs else 0.0,
                              "avg_mae":round(_mean([sf((r.get("close_info") or {}).get("mae_r")) for r in rows]),4),
                              "avg_mfe":round(_mean([sf((r.get("close_info") or {}).get("mfe_r")) for r in rows]),4)}
            trails=[r for r in self.trade_history if r.get("outcome")=="TRAIL"]
            deltas=[]; saved=[]
            for r in trails:
                ci=r.get("close_info") or {}; path=ci.get("path_candles") or []; th=ci.get("trail_history") or []
                if path and th:
                    try:
                        cf=self.trail_counterfactual(r,path,th); deltas.append(sf(cf.get("delta_r"))); saved.append(cf)
                    except Exception: pass
            out.setdefault("TRAIL",{}).update({"counterfactual_n":len(deltas),"avg_delta_vs_no_trail":round(sum(deltas)/len(deltas),4) if deltas else 0.0})
            return out

    def quality_quantity_matrix_v2(self, frequency: Optional[Dict[str, Any]]=None) -> Dict[str, Any]:
        with self._lock:
            st=self.weighted_stats(self.trade_history)
            freq=frequency or self.frequency_diagnosis()
            quality_score=max(0.0,min(100.0, 40*min(1.0,max(0.0,st.get("expectancy",0)+0.5)/1.0)+20*min(1.0,st.get("profit_factor",0)/2)+20*min(1.0,st.get("win_rate",0)/70)+20*max(0.0,1.0-self.risk_adjusted_quality().get("max_drawdown_r",0)/5)))
            freq_score=max(0.0,min(100.0, 50*min(1.0,freq.get("avg_candidate",0)/5)+50*min(1.0,freq.get("avg_eligible",0)/2)))
            if quality_score>=65 and freq_score>=55: decision="KEEP"
            elif quality_score>=65 and freq_score<55: decision="RELAX_LOWEST_VALUE_GATE"
            elif quality_score<45 and freq_score>=55: decision="TIGHTEN_OR_REMODEL"
            elif quality_score<45 and freq_score<55: decision="REGIME_OR_MODEL_PROBLEM"
            else: decision="CONTINUE_LEARNING"
            return {"quality_score":round(quality_score,2),"frequency_score":round(freq_score,2),"decision":decision,"expectancy":st.get("expectancy",0),"profit_factor":st.get("profit_factor",0)}

    def replay_threshold_and_exit_grid(self, setup: Dict[str, Any], path_candles: Sequence[Dict[str, Any]], thresholds=(0,30,40,50,60,70), rr_multipliers=(0.8,1.0,1.2)) -> Dict[str, Any]:
        """Offline scenario grid; no API. Uses observed path candles only."""
        base=self.replay_fixed_levels(setup,path_candles,[])
        trials=[]
        entry=sf(setup.get("fill_price",setup.get("entry"))); initial_sl=sf(setup.get("initial_sl",setup.get("sl")))
        base_risk=abs(entry-initial_sl) or 1e-9
        for thr in thresholds:
            for mult in rr_multipliers:
                trial_setup=dict(setup); trial_setup["tp"] = entry + base_risk*mult*max(1.0, 1.0 if setup.get("direction")=="BUY" else -1.0)
                if setup.get("direction")=="SELL": trial_setup["tp"]=entry-base_risk*mult
                rep=self.replay_fixed_levels(trial_setup,path_candles,[])
                trials.append({"threshold":thr,"rr_multiple":mult,"result":rep})
        return {"baseline":base,"grid":trials}

    def replay_fixed_levels(self, setup: Dict[str, Any], path_candles: Sequence[Dict[str, Any]], trail_levels: Optional[Sequence[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Offline path replay. No API. Tests whether fixed TP/SL or recorded trail checkpoints would fire first."""
        entry=sf(setup.get("fill_price", setup.get("entry"))); tp=sf(setup.get("tp")); sl=sf(setup.get("sl")); direction=setup.get("direction","BUY")
        risk=abs(entry-sf(setup.get("initial_sl", sl))) or 1e-9; mae=0.; mfe=0.; outcome="UNRESOLVED"; exit_price=None; exit_t=None
        trails=list(trail_levels or []); active_sl=sl
        for c in sorted(path_candles or [], key=lambda x: sf(x.get("t"))):
            h,l=sf(c.get("h")),sf(c.get("l")); cl=sf(c.get("c")); t=sf(c.get("t"))
            adverse=((entry-l)/risk if direction=="BUY" else (h-entry)/risk); favorable=((h-entry)/risk if direction=="BUY" else (entry-l)/risk)
            mae=max(mae,max(0.,adverse)); mfe=max(mfe,max(0.,favorable))
            for tr in trails:
                tv=sf(tr.get("new_sl")); tt=sf(tr.get("timestamp"))
                if tt and t>=tt:
                    if direction=="BUY": active_sl=max(active_sl,tv)
                    else: active_sl=min(active_sl,tv)
            hit_tp=(h>=tp if direction=="BUY" else l<=tp); hit_sl=(l<=active_sl if direction=="BUY" else h>=active_sl)
            if hit_tp and hit_sl:
                # OHLC cannot establish intrabar order; mark ambiguous instead of fabricating certainty.
                outcome="AMBIGUOUS_INTRABAR"; exit_t=t; break
            if hit_tp: outcome="TP"; exit_price=tp; exit_t=t; break
            if hit_sl: outcome="TRAIL" if trails and active_sl!=sl else "INITIAL_SL"; exit_price=active_sl; exit_t=t; break
        pnl_r=((exit_price-entry)/risk if direction=="BUY" else (entry-exit_price)/risk) if exit_price is not None else 0.0
        return {"outcome":outcome,"exit_price":exit_price,"exit_time":exit_t,"mae_r":round(mae,4),"mfe_r":round(mfe,4),"pnl_r":round(pnl_r,4),"ambiguous":outcome=="AMBIGUOUS_INTRABAR"}

    def trail_counterfactual(self, setup: Dict[str, Any], path_candles: Sequence[Dict[str, Any]], trail_levels: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        base=self.replay_fixed_levels(setup,path_candles,[])
        trailed=self.replay_fixed_levels(setup,path_candles,trail_levels)
        return {"no_trail":base,"with_trail":trailed,"delta_r":round(trailed["pnl_r"]-base["pnl_r"],4)}

    def overall_stats(self):
        with self._lock:return {"stats":self.weighted_stats(self.trade_history),"counts":{o:sum(1 for r in self.trade_history if r.get("outcome")==o) for o in OUTCOMES},"last_trades":self.trade_history[-5:],"drawdown":self.drawdown(),"quality":self.feature_cache,"frequency":self.frequency_cache,"calibration":self.calibration_cache}
    def get_last_audit_report(self):return dict(self.last_audit_report)
    def should_run_audit(self,interval_seconds=300):return time.time()-self.last_audit_ts>=max(1,interval_seconds)
    def export_memory_summary(self):return {"schema_version":self._schema_version,"trades":len(self.trade_history),"candidates":len(self.candidate_history),"shadow":len(self.shadow_history),"scans":len(self.scan_summaries),"market_snapshots":len(self.market_history),"strategy_version":self.current_strategy_version,"quality":self.feature_cache.get("overall"),"frequency":self.frequency_cache,"exit_attribution":self.exit_cache,"last_audit":self.last_audit_report}
