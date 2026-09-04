from __future__ import annotations

import json
import math
import os
import threading
import time
from collections import deque
from pathlib import Path

STATE_FILE = Path(__file__).resolve().parent / "learn_state.json"
LEARN_VERSION = "L1.0"
CONFIDENCE_BASE = 65.0
CONFIDENCE_MIN = 55.0
CONFIDENCE_MAX = 82.0
FREQUENCY_TARGET_LOW = 0.05
FREQUENCY_TARGET_HIGH = 0.18
RECENCY_HALF_LIFE_HOURS = 72.0
MAX_HISTORY = 5000

_LOCK = threading.RLock()
_HISTORY = deque(maxlen=MAX_HISTORY)
_SCAN_HISTORY = deque(maxlen=80)
_PROTECTION_EVENTS = deque(maxlen=500)
_THRESHOLD = CONFIDENCE_BASE
_WORKER = None
_STOP = threading.Event()
_WAKE = threading.Event()


def _safe_float(value, default=0.0):
    try:
        value = float(value)
        return value if math.isfinite(value) else default
    except Exception:
        return default


def _now():
    return time.time()


def _recency_weight(timestamp):
    age_hours = max(0.0, (_now() - _safe_float(timestamp, _now())) / 3600.0)
    return math.exp(-math.log(2.0) * age_hours / RECENCY_HALF_LIFE_HOURS)


def get_confidence_threshold():
    with _LOCK:
        return float(_THRESHOLD)


def set_confidence_threshold(value):
    global _THRESHOLD
    with _LOCK:
        _THRESHOLD = max(CONFIDENCE_MIN, min(CONFIDENCE_MAX, _safe_float(value, CONFIDENCE_BASE)))
    return get_confidence_threshold()


def evaluate_candidate(symbol, candidate):
    """Use history as a gate adjustment; it never invents a signal."""
    with _LOCK:
        similar = [
            row for row in list(_HISTORY)[-1000:]
            if row.get("kind") == "outcome"
            and str(row.get("symbol")) == str(symbol)
        ]

    weighted_wins = 0.0
    weighted_total = 0.0
    for row in similar:
        weight = _recency_weight(row.get("timestamp"))
        weighted_total += weight
        if _safe_float(row.get("realized_r"), 0.0) > 0:
            weighted_wins += weight

    win_rate = weighted_wins / weighted_total if weighted_total else None

    threshold = get_confidence_threshold()
    if win_rate is not None and weighted_total >= 5:
        if win_rate < 0.40:
            threshold = min(CONFIDENCE_MAX, threshold + 2.0)
        elif win_rate > 0.65:
            threshold = max(CONFIDENCE_MIN, threshold - 2.0)

    return {
        "threshold": round(threshold, 2),
        "symbol_history": len(similar),
        "recent_win_rate": round(win_rate * 100, 2) if win_rate is not None else None,
    }


def record_candidate(packet):
    candidate = packet.get("candidate") or {}
    row = {
        "kind": "candidate",
        "timestamp": _now(),
        "symbol": packet.get("symbol"),
        "side": candidate.get("side"),
        "confidence": candidate.get("confidence"),
        "rr": candidate.get("rr"),
        "strategy_version": candidate.get("strategy_version"),
        "reasons": list(candidate.get("reasons") or []),
    }
    with _LOCK:
        _HISTORY.append(row)
    return row


def record_trade_outcome(trade, outcome=None):
    outcome = outcome or {}
    row = {
        "kind": "outcome",
        "timestamp": _now(),
        "symbol": trade.get("symbol") if isinstance(trade, dict) else None,
        "side": trade.get("side") if isinstance(trade, dict) else None,
        "confidence": trade.get("confidence", 0) if isinstance(trade, dict) else 0,
        "rr": trade.get("rr", 0) if isinstance(trade, dict) else 0,
        "result": outcome.get("result", trade.get("result") if isinstance(trade, dict) else None),
        "realized_r": _safe_float(outcome.get("realized_r", trade.get("realized_r", 0) if isinstance(trade, dict) else 0)),
        "pnl": _safe_float(outcome.get("pnl", trade.get("pnl", 0) if isinstance(trade, dict) else 0)),
        "strategy_version": trade.get("strategy_version") if isinstance(trade, dict) else None,
    }
    with _LOCK:
        _HISTORY.append(row)
    return row


def record_protection_event(event):
    row = dict(event or {})
    row["timestamp"] = row.get("timestamp", _now())
    with _LOCK:
        _PROTECTION_EVENTS.append(row)
    return row


def record_scan_summary(summary):
    row = dict(summary or {})
    row["timestamp"] = row.get("timestamp", _now())
    with _LOCK:
        _SCAN_HISTORY.append(row)
    return row


def get_stats():
    with _LOCK:
        outcomes = [dict(x) for x in _HISTORY if x.get("kind") == "outcome"]
        candidates = [dict(x) for x in _HISTORY if x.get("kind") == "candidate"]
        protections = list(_PROTECTION_EVENTS)
        scans = list(_SCAN_HISTORY)

    tp = sum(1 for x in outcomes if str(x.get("result", "")).upper() == "TP")
    sl = sum(1 for x in outcomes if str(x.get("result", "")).upper() == "SL")
    trail = sum(1 for x in outcomes if str(x.get("result", "")).upper() == "TRAIL")
    total = len(outcomes)
    wins = tp + trail
    avg_r = sum(_safe_float(x.get("realized_r")) for x in outcomes) / total if total else 0.0

    return {
        "learning_version": LEARN_VERSION,
        "total_outcomes": total,
        "candidates": len(candidates),
        "tp": tp,
        "sl": sl,
        "trail": trail,
        "win_rate": wins / total * 100 if total else 0.0,
        "avg_r": avg_r,
        "confidence_threshold": get_confidence_threshold(),
        "protection_events": len(protections),
        "scan_cycles": len(scans),
    }


def get_status():
    stats = get_stats()
    with _LOCK:
        worker_alive = bool(_WORKER and _WORKER.is_alive())
        history_size = len(_HISTORY)
    return {
        **stats,
        "worker_alive": worker_alive,
        "history_size": history_size,
    }


def _adapt_threshold_from_frequency():
    global _THRESHOLD
    with _LOCK:
        rows = list(_SCAN_HISTORY)
        if not rows:
            return
        analyzed = sum(max(1, int(x.get("analyzed_symbols", 0) or 0)) for x in rows)
        eligible = sum(int(x.get("eligible_count", 0) or 0) for x in rows)

    rate = eligible / analyzed if analyzed else 0.0

    with _LOCK:
        if rate < FREQUENCY_TARGET_LOW:
            _THRESHOLD = max(CONFIDENCE_MIN, _THRESHOLD - 2.0)
        elif rate > FREQUENCY_TARGET_HIGH:
            _THRESHOLD = min(CONFIDENCE_MAX, _THRESHOLD + 2.0)


def _worker_loop():
    while not _STOP.is_set():
        _adapt_threshold_from_frequency()
        _WAKE.wait(30.0)
        _WAKE.clear()


def full_command(action="status"):
    action = str(action or "status").strip().lower()

    if action in {"on", "/full on", "full on"}:
        return adaptive_agent_start()

    if action in {"off", "/full off", "full off"}:
        return adaptive_agent_stop()

    if action in {"reset", "/full reset", "full reset"}:
        adaptive_agent_stop()
        reset()
        return {"ok": True, "message": "learning reset"}

    if action in {"status", "/full", "full"}:
        return get_status()

    return get_status()


def adaptive_agent_start():
    global _WORKER
    with _LOCK:
        if _WORKER is not None and _WORKER.is_alive():
            return {"ok": True, "message": "FULL already ON", "status": get_status()}
        _STOP.clear()
        _WORKER = threading.Thread(target=_worker_loop, name="learning-worker", daemon=True)
        _WORKER.start()
    return {"ok": True, "message": "FULL ON", "status": get_status()}


def adaptive_agent_stop():
    _STOP.set()
    _WAKE.set()
    return {"ok": True, "message": "FULL OFF", "status": get_status()}


def reset():
    global _THRESHOLD
    with _LOCK:
        _HISTORY.clear()
        _SCAN_HISTORY.clear()
        _PROTECTION_EVENTS.clear()
        _THRESHOLD = CONFIDENCE_BASE
    save_state()


def export_state():
    with _LOCK:
        return {
            "schema": "learn_state_v1",
            "version": LEARN_VERSION,
            "saved_at": _now(),
            "threshold": _THRESHOLD,
            "history": list(_HISTORY),
            "scan_history": list(_SCAN_HISTORY),
            "protection_events": list(_PROTECTION_EVENTS),
        }


def import_state(data):
    global _THRESHOLD
    if not isinstance(data, dict) or data.get("schema") != "learn_state_v1":
        raise ValueError("Format learn_state.json tidak dikenali")

    with _LOCK:
        _HISTORY.clear()
        _HISTORY.extend(data.get("history") or [])
        _SCAN_HISTORY.clear()
        _SCAN_HISTORY.extend(data.get("scan_history") or [])
        _PROTECTION_EVENTS.clear()
        _PROTECTION_EVENTS.extend(data.get("protection_events") or [])
        _THRESHOLD = max(CONFIDENCE_MIN, min(CONFIDENCE_MAX, _safe_float(data.get("threshold"), CONFIDENCE_BASE)))

    return {"ok": True, "status": get_status()}


def save_state(path=None):
    target = Path(path) if path else STATE_FILE
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(export_state(), ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, target)
    return {"ok": True, "path": str(target)}


def open_state(path=None):
    target = Path(path) if path else STATE_FILE
    if not target.exists():
        return {"ok": False, "message": f"File tidak ditemukan: {target}"}
    data = json.loads(target.read_text(encoding="utf-8"))
    result = import_state(data)
    result["path"] = str(target)
    return result


save = save_state
open = open_state

try:
    open_state()
except Exception:
    pass

__all__ = [
    "LEARN_VERSION",
    "get_confidence_threshold",
    "set_confidence_threshold",
    "evaluate_candidate",
    "record_candidate",
    "record_trade_outcome",
    "record_protection_event",
    "record_scan_summary",
    "get_stats",
    "get_status",
    "full_command",
    "adaptive_agent_start",
    "adaptive_agent_stop",
    "reset",
    "export_state",
    "import_state",
    "save_state",
    "open_state",
    "save",
    "open",
]
