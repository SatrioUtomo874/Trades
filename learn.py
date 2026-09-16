"""
learn.py — mathematical/statistical auditor of strategy.py.

learn.py NEVER touches Bybit/Binance/Telegram. It only observes what
main.py reports and answers questions about strategy performance.
No Ollama, no LLM calls — pure statistics.

Public interface:
    record_candidate(candidate: dict)
    record_trade(trade: dict)
    record_timeout(event: dict)
    record_scan(summary: dict)
    audit() -> dict
    maybe_calibrate(current_threshold: float) -> dict
    get_stats_summary() -> dict
    load_checkpoint() / save_checkpoint()
    get_checkpoint_bytes_for_sync() -> bytes | None   # used by main.py for optional GitHub autosave
"""

from __future__ import annotations

import json
import math
import os
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

LEARN_VERSION = "1.0.0"

BASE_DIR = Path(__file__).resolve().parent
STATE_DIR = BASE_DIR / "state"
CKPT_PATH = STATE_DIR / "learn_checkpoint.json"
BACKUP_PATH = STATE_DIR / "learn_checkpoint.json.backup"

MAX_TRADES = 4000
MAX_CANDIDATES = 6000
MAX_TIMEOUTS = 2000
MAX_SCANS = 1000
MAX_AUDIT_HISTORY = 500

MIN_SAMPLES_CALIBRATE = 50
MIN_SAMPLES_FEATURE = 30
MIN_SAMPLES_SESSION = 20

REQUIRED_KEYS = ("strategy_version", "threshold", "trades", "candidates", "timeouts", "scans", "audit_history")

_DEFAULT_STATE: dict[str, Any] = {
    "strategy_version": "1.0.0",
    "threshold": 0.0,
    "trades": [],
    "candidates": [],
    "timeouts": [],
    "scans": [],
    "audit_history": [],
    "last_saved": None,
}

_state: dict[str, Any] = json.loads(json.dumps(_DEFAULT_STATE))
_dirty = False


# ----------------------------------------------------------------------
# checkpoint persistence
# ----------------------------------------------------------------------

def _validate(data: Any) -> bool:
    if not isinstance(data, dict):
        return False
    return all(k in data for k in REQUIRED_KEYS)


def _atomic_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp_learn_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            try:
                os.remove(tmp_name)
            except OSError:
                pass


def save_checkpoint() -> bool:
    global _dirty
    payload = dict(_state)
    payload["last_saved"] = int(time.time())
    try:
        _atomic_write(CKPT_PATH, payload)
        _atomic_write(BACKUP_PATH, payload)
        _dirty = False
        return True
    except OSError:
        return False


def load_checkpoint() -> dict:
    """Load primary checkpoint; fall back to backup if invalid. Always returns a report."""
    global _state
    report = {"restored": False, "source": None, "reason": None}

    for path, label in ((CKPT_PATH, "primary"), (BACKUP_PATH, "backup")):
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            report["reason"] = f"{label} unreadable: {exc}"
            continue
        if not _validate(data):
            report["reason"] = f"{label} missing required keys"
            continue
        merged = json.loads(json.dumps(_DEFAULT_STATE))
        merged.update(data)
        _state = merged
        report["restored"] = True
        report["source"] = label
        report["reason"] = None
        return report

    report["reason"] = report["reason"] or "no checkpoint found"
    return report


def get_checkpoint_bytes_for_sync() -> Optional[bytes]:
    """learn.py never calls external APIs itself; main.py may push this to GitHub."""
    if not CKPT_PATH.exists():
        return None
    try:
        return CKPT_PATH.read_bytes()
    except OSError:
        return None


def _cap(lst: list, max_len: int) -> list:
    if len(lst) > max_len:
        del lst[: len(lst) - max_len]
    return lst


# ----------------------------------------------------------------------
# recording
# ----------------------------------------------------------------------

def record_candidate(candidate: dict) -> None:
    global _dirty
    entry = dict(candidate)
    entry.setdefault("recorded_at", int(time.time() * 1000))
    _state["candidates"].append(entry)
    _cap(_state["candidates"], MAX_CANDIDATES)
    _dirty = True


def record_trade(trade: dict) -> None:
    """
    trade: {symbol, direction, mode('real'|'sim'), entry, exit_price, result('tp'|'initial_sl'|'trail'),
            r_multiple, pnl_pct, pnl_usd, confidence, session, market_regime, setup_type,
            feature_scores, opened_at, closed_at}
    """
    global _dirty
    entry = dict(trade)
    entry.setdefault("closed_at", int(time.time() * 1000))
    _state["trades"].append(entry)
    _cap(_state["trades"], MAX_TRADES)
    _dirty = True


def record_timeout(event: dict) -> None:
    """Only AUTOMATIC timeouts. Manual /timeout cleanup must never be recorded here."""
    global _dirty
    entry = dict(event)
    entry.setdefault("recorded_at", int(time.time() * 1000))
    _state["timeouts"].append(entry)
    _cap(_state["timeouts"], MAX_TIMEOUTS)
    _dirty = True


def record_scan(summary: dict) -> None:
    global _dirty
    entry = dict(summary)
    entry.setdefault("recorded_at", int(time.time() * 1000))
    _state["scans"].append(entry)
    _cap(_state["scans"], MAX_SCANS)
    _dirty = True


def maybe_autosave(min_interval_sec: int = 120) -> bool:
    """Call periodically from main.py's background loop; saves only if dirty and due."""
    if not _dirty:
        return False
    last = _state.get("last_saved") or 0
    if time.time() - last < min_interval_sec:
        return False
    return save_checkpoint()


# ----------------------------------------------------------------------
# statistics helpers
# ----------------------------------------------------------------------

def _closed_trades(mode: Optional[str] = None, since_ms: Optional[int] = None) -> list[dict]:
    trades = _state["trades"]
    out = []
    for t in trades:
        if mode and t.get("mode") != mode:
            continue
        if since_ms and t.get("closed_at", 0) < since_ms:
            continue
        out.append(t)
    return out


def _f(x: Any) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def _win_loss_stats(trades: list[dict]) -> dict:
    if not trades:
        return {
            "count": 0, "wins": 0, "losses": 0, "breakeven": 0, "win_rate": None,
            "expectancy_r": None, "avg_r": None, "median_r": None,
            "profit_factor": None, "avg_win_pct": None, "avg_loss_pct": None,
        }
    wins = [t for t in trades if _f(t.get("pnl_pct")) > 0]
    losses = [t for t in trades if _f(t.get("pnl_pct")) < 0]
    breakeven = [t for t in trades if _f(t.get("pnl_pct")) == 0]
    r_values = [_f(t.get("r_multiple")) for t in trades if t.get("r_multiple") is not None]

    gross_profit = sum(_f(t.get("pnl_usd")) for t in wins)
    gross_loss = abs(sum(_f(t.get("pnl_usd")) for t in losses))
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (float("inf") if gross_profit > 0 else None)

    return {
        "count": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "breakeven": len(breakeven),
        "win_rate": round(len(wins) / len(trades) * 100, 2) if trades else None,
        "expectancy_r": round(statistics.fmean(r_values), 4) if r_values else None,
        "avg_r": round(statistics.fmean(r_values), 4) if r_values else None,
        "median_r": round(statistics.median(r_values), 4) if r_values else None,
        "profit_factor": round(profit_factor, 3) if isinstance(profit_factor, float) and math.isfinite(profit_factor) else profit_factor,
        "avg_win_pct": round(statistics.fmean([_f(t.get("pnl_pct")) for t in wins]), 4) if wins else None,
        "avg_loss_pct": round(statistics.fmean([_f(t.get("pnl_pct")) for t in losses]), 4) if losses else None,
    }


def _drawdown(trades: list[dict], anchor_balance: float = 10.0) -> dict:
    balance = anchor_balance
    peak = anchor_balance
    max_dd_pct = 0.0
    max_dd_abs = 0.0
    for t in sorted(trades, key=lambda x: x.get("closed_at", 0)):
        balance += _f(t.get("pnl_usd"))
        peak = max(peak, balance)
        dd_abs = peak - balance
        dd_pct = (dd_abs / peak * 100) if peak > 0 else 0.0
        max_dd_abs = max(max_dd_abs, dd_abs)
        max_dd_pct = max(max_dd_pct, dd_pct)
    return {"max_drawdown_pct": round(max_dd_pct, 3), "max_drawdown_abs": round(max_dd_abs, 4), "ending_balance": round(balance, 4)}


def _calibration_buckets(trades: list[dict]) -> dict:
    buckets = {"<50": [], "50-64": [], "65-79": [], "80+": []}
    for t in trades:
        conf = _f(t.get("confidence"))
        if conf < 50:
            key = "<50"
        elif conf < 65:
            key = "50-64"
        elif conf < 80:
            key = "65-79"
        else:
            key = "80+"
        buckets[key].append(t)
    out = {}
    for key, items in buckets.items():
        stats = _win_loss_stats(items)
        out[key] = {"count": stats["count"], "win_rate": stats["win_rate"], "expectancy_r": stats["expectancy_r"]}
    return out


def _breakdown_by(trades: list[dict], field: str) -> dict:
    groups: dict[str, list[dict]] = {}
    for t in trades:
        key = str(t.get(field) or "unknown")
        groups.setdefault(key, []).append(t)
    out = {}
    for key, items in groups.items():
        stats = _win_loss_stats(items)
        out[key] = {"count": stats["count"], "win_rate": stats["win_rate"], "expectancy_r": stats["expectancy_r"]}
    return out


def _rolling_windows(trades: list[dict]) -> dict:
    now = int(time.time() * 1000)
    windows = {
        "24h": 24 * 3600 * 1000,
        "3d": 3 * 24 * 3600 * 1000,
        "7d": 7 * 24 * 3600 * 1000,
        "14d": 14 * 24 * 3600 * 1000,
        "30d": 30 * 24 * 3600 * 1000,
    }
    out = {}
    for label, span in windows.items():
        subset = [t for t in trades if t.get("closed_at", 0) >= now - span]
        out[label] = _win_loss_stats(subset)
    return out


# ----------------------------------------------------------------------
# audit
# ----------------------------------------------------------------------

def audit() -> dict:
    real_trades = _closed_trades(mode="real")
    sim_trades = _closed_trades(mode="sim")
    all_trades = real_trades + sim_trades

    result = {
        "strategy_version": _state.get("strategy_version"),
        "generated_at": int(time.time() * 1000),
        "overall": _win_loss_stats(all_trades),
        "real": _win_loss_stats(real_trades),
        "simulation": _win_loss_stats(sim_trades),
        "drawdown_real": _drawdown(real_trades),
        "drawdown_sim": _drawdown(sim_trades),
        "rolling": _rolling_windows(all_trades),
        "by_session": _breakdown_by(all_trades, "session"),
        "by_regime": _breakdown_by(all_trades, "market_regime"),
        "by_setup_type": _breakdown_by(all_trades, "setup_type"),
        "by_direction": _breakdown_by(all_trades, "direction"),
        "by_result": _breakdown_by(all_trades, "result"),
        "confidence_calibration": _calibration_buckets(all_trades),
        "sample_size": len(all_trades),
        "candidate_count": len(_state["candidates"]),
        "rejected_low_confidence": len([c for c in _state["candidates"] if c.get("rejected_reason") == "BELOW_ACTIVE_THRESHOLD"]),
        "timeout_count": len(_state["timeouts"]),
    }
    _state["audit_history"].append({"at": result["generated_at"], "sample_size": result["sample_size"],
                                     "win_rate": result["overall"]["win_rate"], "expectancy_r": result["overall"]["expectancy_r"]})
    _cap(_state["audit_history"], MAX_AUDIT_HISTORY)
    return result


# ----------------------------------------------------------------------
# calibration (evidence-gated)
# ----------------------------------------------------------------------

def maybe_calibrate(current_threshold: float) -> dict:
    """
    Evidence-based threshold recommendation. Never acts on a single trade.
    Returns {"action": "none"|"adjust", "new_threshold": float|None, "reason": str}
    """
    trades = _closed_trades()
    if len(trades) < MIN_SAMPLES_CALIBRATE:
        return {
            "action": "none",
            "new_threshold": None,
            "reason": f"insufficient sample ({len(trades)}/{MIN_SAMPLES_CALIBRATE}) — observing only",
        }

    buckets = _calibration_buckets(trades)
    viable = {k: v for k, v in buckets.items() if v["count"] >= MIN_SAMPLES_FEATURE and v["expectancy_r"] is not None}
    if not viable:
        return {
            "action": "none",
            "new_threshold": None,
            "reason": "no confidence bucket has enough samples yet for a robust comparison",
        }

    best_bucket = max(viable.items(), key=lambda kv: kv[1]["expectancy_r"])
    bucket_floor = {"<50": 0.0, "50-64": 50.0, "65-79": 65.0, "80+": 80.0}
    suggested = bucket_floor[best_bucket[0]]

    current_bucket_expectancy = None
    for label, floor in sorted(bucket_floor.items(), key=lambda kv: kv[1]):
        if current_threshold >= floor:
            current_bucket_expectancy = buckets.get(label, {}).get("expectancy_r")

    if suggested <= current_threshold:
        return {
            "action": "none",
            "new_threshold": None,
            "reason": f"current threshold ({current_threshold}%) already at/above best-performing bucket floor ({suggested}%)",
        }

    improvement = None
    if current_bucket_expectancy is not None and best_bucket[1]["expectancy_r"] is not None:
        improvement = best_bucket[1]["expectancy_r"] - current_bucket_expectancy

    if improvement is not None and improvement <= 0.02:
        return {
            "action": "none",
            "new_threshold": None,
            "reason": "expectancy improvement too small to justify a threshold change (avoiding overfit to noise)",
        }

    return {
        "action": "adjust",
        "new_threshold": suggested,
        "reason": (
            f"confidence bucket >= {suggested}% shows expectancy {best_bucket[1]['expectancy_r']}R "
            f"over {best_bucket[1]['count']} trades, vs current threshold {current_threshold}%"
        ),
        "sample_size": best_bucket[1]["count"],
    }


def get_threshold() -> float:
    return float(_state.get("threshold", 0.0))


def apply_calibration(new_threshold: float, reason: str) -> None:
    global _dirty
    old = _state.get("threshold")
    _state["threshold"] = new_threshold
    _state["audit_history"].append({
        "at": int(time.time() * 1000),
        "event": "threshold_change",
        "old_threshold": old,
        "new_threshold": new_threshold,
        "reason": reason,
    })
    _cap(_state["audit_history"], MAX_AUDIT_HISTORY)
    _dirty = True


def record_strategy_version(version: str, reason: str, metrics_before: Optional[dict] = None) -> None:
    global _dirty
    old = _state.get("strategy_version")
    _state["strategy_version"] = version
    _state["audit_history"].append({
        "at": int(time.time() * 1000),
        "event": "strategy_version_change",
        "old_version": old,
        "new_version": version,
        "reason": reason,
        "metrics_before": metrics_before,
    })
    _cap(_state["audit_history"], MAX_AUDIT_HISTORY)
    _dirty = True


# ----------------------------------------------------------------------
# /stats summary
# ----------------------------------------------------------------------

def get_stats_summary(mode: str = "sim") -> dict:
    trades = _closed_trades(mode=mode)
    stats = _win_loss_stats(trades)
    tp = len([t for t in trades if t.get("result") == "tp"])
    isl = len([t for t in trades if t.get("result") == "initial_sl"])
    trail = len([t for t in trades if t.get("result") == "trail"])
    last5 = sorted(trades, key=lambda t: t.get("closed_at", 0))[-5:]
    avg_conf = statistics.fmean([_f(t.get("confidence")) for t in trades]) if trades else None
    return {
        "count": stats["count"],
        "wins": stats["wins"],
        "losses": stats["losses"],
        "breakeven": stats["breakeven"],
        "win_rate": stats["win_rate"],
        "tp_count": tp,
        "trail_count": trail,
        "sl_count": isl,
        "tp_pct": round(tp / stats["count"] * 100, 1) if stats["count"] else None,
        "trail_pct": round(trail / stats["count"] * 100, 1) if stats["count"] else None,
        "sl_pct": round(isl / stats["count"] * 100, 1) if stats["count"] else None,
        "avg_closed_confidence": round(avg_conf, 1) if avg_conf is not None else None,
        "last_5": last5,
        "drawdown": _drawdown(trades),
        "threshold": _state.get("threshold"),
        "strategy_version": _state.get("strategy_version"),
    }
