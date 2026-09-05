"""
learn.py — Adaptive Audit, Statistical Learning & Strategy Governance Brain vNext
================================================================================

PRINSIP KERAS
--------------
1. Learn bukan trader. Learn tidak membuat / membatalkan order.
2. Learn tidak mengambil market data dari exchange. Semua market snapshot datang
   dari main.py / Strategy.
3. Learn menganalisis lifecycle scan secara aktif, bukan hanya closed trades.
4. Tidak ada parameter strategy yang berubah karena satu trade, satu scan, atau
   satu kejadian anomali.
5. Perubahan strategy hanya melalui statistical gate + chronological validation
   + holdout + counterfactual + robustness.
6. Ollama hanya critic/advisor. Ia tidak pernah menjadi decision maker.
7. Checkpoint atomic + checksum + backup + validation.
8. Semua aktivitas penting dibuat visible melalui logging terminal. Bila main.py
   mendaftarkan notification sink, event checkpoint juga diteruskan ke Telegram.
9. Tidak ada look-ahead pada data scan/features. Historical outcome boleh dipakai
   hanya setelah event tersebut secara chronological memang telah terjadi.

DESAIN MEMORY
-------------
A. raw event memory       : semua lifecycle event.
B. market/feature memory  : snapshot, candidate feature, reject diagnosis,
                            regime, breadth, freshness dan feature aggregates.
C. statistical memory     : quality, frequency, calibration, attribution,
                            confidence interval, decay, recent-vs-long-term.
D. decision memory        : challenger, holdout, rollback, recommendations.
E. strategy memory        : version, parameters, accepted changes.

KOMPATIBILITAS main.py
----------------------
Tetap menyediakan method lama:
- load(), save_checkpoint(), autosave(), set_strategy_state()
- record_scan_summary(), record_scan_candidate(), record_shadow_outcome()
- record_trade_outcome(), audit(), overall_stats()

Method tambahan yang disiapkan untuk integrasi vNext:
- record_market_snapshot()
- record_scan_analysis()
- record_pending_event()
- record_fill_event()
- record_trail_decision()
- record_trail_result()
- record_close_event()
- record_timeout_event()
- set_notification_sink()
- get_checkpoint_notification()
- analyze_scan_memory()
- quality_quantity_matrix()
- chronological_replay()
- evaluate_challenger()
- rollback_to_version()

NOTE TELEGRAM
-------------
Learn tidak melakukan request Telegram sendiri agar tetap menjadi engine statistik,
bukan engine network. Bila main.py memberi callable notification sink, Learn akan
mengirim notifikasi checkpoint melalui sink tersebut. Dengan main.py saat ini sink
belum dipasang; karena itu Learn tidak berpura-pura seolah Telegram sudah terkirim.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import math
import os
import shutil
import statistics
import subprocess
import time
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None


logger = logging.getLogger("learn")

SCHEMA_VERSION = 7
ENGINE_NAME = "adaptive-learning-brain-vnext"
ENGINE_VERSION = "2.00"

# ---------------------------------------------------------------------------
# Governance thresholds
# ---------------------------------------------------------------------------
HALF_LIFE_DAYS = 21.0
SHADOW_HALF_LIFE_DAYS = 14.0
SCAN_HALF_LIFE_DAYS = 7.0

MIN_TOTAL_SAMPLE_FOR_AUDIT = 40
MIN_SAMPLE_FOR_DECISION = 30
MIN_HOLDOUT_SAMPLE = 20
MIN_TRAIN_SAMPLE = 30
MIN_TRADES_SINCE_LAST_CHANGE = 20
MIN_SCAN_EVENTS_FOR_SCAN_GATE = 100
MIN_SCAN_EVENTS_FOR_PATTERN = 30

AUDIT_COOLDOWN_SECONDS = 15 * 60
ROLLBACK_DEGRADATION_R = 0.30
MAX_THRESHOLD_STEP = 5.0

MAX_PARAM_STEP: Dict[str, float] = {
    "min_rr": 0.20,
    "displacement_atr_mult": 0.20,
    "sweep_lookback": 10.0,
    "structure_lookback": 10.0,
    "trend_lookback": 10.0,
    "btc_corr_lookback": 10.0,
    "sl_atr_buffer": 0.10,
    "entry_retracement_fib": 0.03,
    "entry_min_offset_atr": 0.10,
    # vNext governance keys
    "trail_activation_r": 0.25,
    "trail_min_profit_r": 0.15,
    "stale_setup_minutes": 30.0,
    "fvg_freshness_bars": 3.0,
}

ALLOWED_UPDATE_KEYS = set(["ACTIVE_THRESHOLD", *MAX_PARAM_STEP.keys(), "CONFIDENCE_WEIGHTS"])

CONFIDENCE_BUCKETS: List[Tuple[int, int]] = [
    (0, 40),
    (40, 50),
    (50, 60),
    (60, 70),
    (70, 80),
    (80, 90),
    (90, 101),
]

OUTCOME_TYPES = ("TP", "INITIAL_SL", "TRAIL", "BE", "TIMEOUT")
ECONOMIC_OUTCOMES = ("TP", "INITIAL_SL", "TRAIL", "BE")
SCAN_DIAGNOSES = (
    "NO_SETUP",
    "INVALID_GEOMETRY",
    "STALE_SETUP",
    "LOW_EXPECTED_VALUE",
    "TOO_CLOSE",
    "TOO_FAR",
    "LOW_LIQUIDITY_CONTEXT",
    "REGIME_MISMATCH",
    "BTC_CONFLICT",
    "VALID_LOW_CONF",
    "VALID_HIGH_CONF",
)

QUALITY_HIGH_EXPECTANCY_R = 0.15
QUALITY_MIN_PF = 1.20
QUALITY_MAX_DRAWDOWN_R = 5.0
FREQUENCY_HEALTHY_CANDIDATE_PER_SCAN = 0.05
FREQUENCY_HEALTHY_OPPORTUNITY_PER_SCAN = 0.01


# ---------------------------------------------------------------------------
# Generic safety helpers
# ---------------------------------------------------------------------------
def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        v = float(value)
        return v if math.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _finite_or_none(value: Any) -> Optional[float]:
    try:
        v = float(value)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _now() -> float:
    return time.time()


def _event_ts(row: Dict[str, Any], fallback: Optional[float] = None) -> float:
    if fallback is None:
        fallback = _now()
    candidates = (
        row.get("timestamp"),
        row.get("event_ts"),
        row.get("close_time"),
        row.get("fill_time"),
        row.get("entry_time"),
    )
    for candidate in candidates:
        v = _safe_float(candidate, float("nan"))
        if math.isfinite(v):
            # Main uses ms timestamps for market lifecycle. Normalize to seconds
            # only for calculations which explicitly call _event_ts.
            if v > 10_000_000_000:
                return v / 1000.0
            return v
    return fallback


def _time_decay_weight(ts: Any, now: Optional[float] = None, half_life_days: float = HALF_LIFE_DAYS) -> float:
    now = _now() if now is None else now
    age_days = max(0.0, now - _event_ts({"timestamp": ts}, fallback=now)) / 86400.0
    return 0.5 ** (age_days / max(0.1, half_life_days))


def _bucket_of(confidence: Any) -> str:
    c = max(0.0, min(100.0, _safe_float(confidence)))
    for lo, hi in CONFIDENCE_BUCKETS:
        if lo <= c < hi:
            return f"{lo}-{hi - 1}"
    return "90-100"


def _mean(values: Iterable[float]) -> float:
    xs = [float(x) for x in values if math.isfinite(float(x))]
    return sum(xs) / len(xs) if xs else 0.0


def _median(values: Iterable[float]) -> float:
    xs = [float(x) for x in values if math.isfinite(float(x))]
    return float(statistics.median(xs)) if xs else 0.0


def _percentile(values: Sequence[float], q: float) -> float:
    xs = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not xs:
        return 0.0
    if len(xs) == 1:
        return xs[0]
    q = max(0.0, min(1.0, q))
    pos = q * (len(xs) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    frac = pos - lo
    return xs[lo] + (xs[hi] - xs[lo]) * frac


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _safe_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _safe_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_jsonable(v) for v in value]
    if isinstance(value, (int, str, bool)) or value is None:
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    try:
        return float(value)
    except Exception:
        return str(value)


def _canonical_json(data: Any) -> str:
    return json.dumps(_safe_jsonable(data), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_payload(data: Dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(data).encode("utf-8")).hexdigest()


def _atomic_write_json(path: str, data: Dict[str, Any]) -> None:
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(_safe_jsonable(data), f, indent=2, ensure_ascii=False, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _normalise_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted((dict(r) for r in rows), key=_event_ts)


def _economic_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [r for r in rows if r.get("outcome") in ECONOMIC_OUTCOMES]


def _binary_ci(successes: float, n: float, z: float = 1.96) -> Tuple[float, float]:
    if n <= 0:
        return 0.0, 0.0
    p = _clamp(successes / n, 0.0, 1.0)
    denom = 1.0 + (z * z) / n
    center = (p + (z * z) / (2.0 * n)) / denom
    margin = z * math.sqrt((p * (1.0 - p) + (z * z) / (4.0 * n)) / n) / denom
    return center - margin, center + margin


def _normal_ci(values: Sequence[float], z: float = 1.96) -> Tuple[float, float]:
    xs = [float(v) for v in values if math.isfinite(float(v))]
    n = len(xs)
    if n < 2:
        return (xs[0], xs[0]) if xs else (0.0, 0.0)
    mean = _mean(xs)
    sd = statistics.stdev(xs)
    margin = z * sd / math.sqrt(n)
    return mean - margin, mean + margin


def _drawdown_series(rs: Sequence[float]) -> Tuple[List[float], float]:
    equity = 0.0
    peak = 0.0
    dd: List[float] = []
    max_dd = 0.0
    for r in rs:
        equity += float(r)
        peak = max(peak, equity)
        current = equity - peak
        dd.append(current)
        max_dd = min(max_dd, current)
    return dd, abs(max_dd)


def _downside_deviation(rs: Sequence[float], target: float = 0.0) -> float:
    negatives = [min(0.0, float(r) - target) ** 2 for r in rs]
    return math.sqrt(_mean(negatives)) if negatives else 0.0


def _max_losing_streak(rs: Sequence[float]) -> int:
    streak = 0
    best = 0
    for r in rs:
        if r < 0:
            streak += 1
            best = max(best, streak)
        elif r > 0:
            streak = 0
    return best


def _group_by(rows: Sequence[Dict[str, Any]], key: str) -> Dict[str, List[Dict[str, Any]]]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(key, "UNKNOWN"))].append(dict(row))
    return dict(groups)


def _safe_path_candles(raw: Any, max_len: int = 500) -> List[Dict[str, float]]:
    out = []
    if not isinstance(raw, (list, tuple)):
        return out
    for c in list(raw)[-max_len:]:
        if not isinstance(c, dict):
            continue
        try:
            o = {k: float(c[k]) for k in ("t", "o", "h", "l", "c", "v")}
        except (KeyError, TypeError, ValueError):
            continue
        if all(math.isfinite(v) for v in o.values()):
            out.append(o)
    return out


def _path_mfe_mae(path: Sequence[Dict[str, float]], entry: float, direction: str) -> Dict[str, float]:
    if not path or entry <= 0:
        return {"mfe_r": 0.0, "mae_r": 0.0, "mfe_price": entry, "mae_price": entry, "path_bars": len(path)}
    risk_placeholder = max(abs(entry) * 1e-6, 1e-9)
    mfe_price = entry
    mae_price = entry
    if direction == "BUY":
        mfe_price = max(float(c["h"]) for c in path)
        mae_price = min(float(c["l"]) for c in path)
    else:
        mfe_price = min(float(c["l"]) for c in path)
        mae_price = max(float(c["h"]) for c in path)
    mfe = ((mfe_price - entry) / risk_placeholder) if direction == "BUY" else ((entry - mfe_price) / risk_placeholder)
    mae = ((entry - mae_price) / risk_placeholder) if direction == "BUY" else ((mae_price - entry) / risk_placeholder)
    return {
        "mfe_r": float(max(0.0, mfe)),
        "mae_r": float(max(0.0, mae)),
        "mfe_price": float(mfe_price),
        "mae_price": float(mae_price),
        "path_bars": len(path),
    }


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class StatResult:
    n: int = 0
    effective_n: float = 0.0
    mean: float = 0.0
    median: float = 0.0
    stdev: float = 0.0
    ci_low: float = 0.0
    ci_high: float = 0.0
    win_rate: float = 0.0
    win_ci_low: float = 0.0
    win_ci_high: float = 0.0
    expectancy: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    downside_deviation: float = 0.0
    max_drawdown_r: float = 0.0
    max_losing_streak: int = 0
    sum_r: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n": self.n,
            "effective_n": round(self.effective_n, 4),
            "mean": round(self.mean, 6),
            "median": round(self.median, 6),
            "stdev": round(self.stdev, 6),
            "ci_low": round(self.ci_low, 6),
            "ci_high": round(self.ci_high, 6),
            "win_rate": round(self.win_rate, 4),
            "win_ci_low": round(self.win_ci_low, 4),
            "win_ci_high": round(self.win_ci_high, 4),
            "expectancy": round(self.expectancy, 6),
            "profit_factor": round(self.profit_factor, 6),
            "avg_win": round(self.avg_win, 6),
            "avg_loss": round(self.avg_loss, 6),
            "downside_deviation": round(self.downside_deviation, 6),
            "max_drawdown_r": round(self.max_drawdown_r, 6),
            "max_losing_streak": self.max_losing_streak,
            "sum_r": round(self.sum_r, 6),
        }


@dataclass
class ScanFeatureRecord:
    pair: str
    timestamp: float
    confidence: float = 0.0
    direction: str = "UNKNOWN"
    setup_type: str = "UNKNOWN"
    regime: str = "UNKNOWN"
    session: str = "UNKNOWN"
    diagnosis: str = "NO_SETUP"
    eligible: bool = False
    threshold: float = 0.0
    structure_score: float = 0.0
    liquidity_score: float = 0.0
    entry_score: float = 0.0
    rr_score: float = 0.0
    momentum_score: float = 0.0
    volatility_score: float = 0.0
    btc_score: float = 0.0
    regime_score: float = 0.0
    session_score: float = 0.0
    confirmation_score: float = 0.0
    entry_distance_atr: float = 0.0
    rr: float = 0.0
    stale: bool = False
    btc_aligned: Optional[bool] = None
    timestamp_age_seconds: float = 0.0
    source_strategy_version: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pair": self.pair,
            "timestamp": self.timestamp,
            "confidence": self.confidence,
            "direction": self.direction,
            "setup_type": self.setup_type,
            "regime": self.regime,
            "session": self.session,
            "diagnosis": self.diagnosis,
            "eligible": self.eligible,
            "threshold": self.threshold,
            "components": {
                "structure": self.structure_score,
                "liquidity": self.liquidity_score,
                "entry_quality": self.entry_score,
                "risk_reward": self.rr_score,
                "momentum": self.momentum_score,
                "volatility": self.volatility_score,
                "btc_correlation": self.btc_score,
                "regime": self.regime_score,
                "session": self.session_score,
                "confirmation": self.confirmation_score,
            },
            "entry_distance_atr": self.entry_distance_atr,
            "rr": self.rr,
            "stale": self.stale,
            "btc_aligned": self.btc_aligned,
            "timestamp_age_seconds": self.timestamp_age_seconds,
            "strategy_version": self.source_strategy_version,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Learn Engine
# ---------------------------------------------------------------------------
class LearnEngine:
    """Thread-safe adaptive audit + statistical learning engine."""

    def __init__(
        self,
        checkpoint_path: str = "state/learn_checkpoint.json",
        backup_path: Optional[str] = None,
        ollama_url: Optional[str] = None,
        ollama_api_key: Optional[str] = None,
        git_enabled: bool = False,
        git_repo_dir: Optional[str] = None,
        notification_sink: Optional[Callable[[str, str], Any]] = None,
        checkpoint_interval_seconds: int = 120,
    ):
        self.checkpoint_path = checkpoint_path
        self.backup_path = backup_path or (checkpoint_path + ".backup")
        self.checksum_path = checkpoint_path + ".sha256"
        self.ollama_url = ollama_url or os.environ.get("OLLAMA_URL", "http://localhost:11434")
        self.ollama_api_key = ollama_api_key or os.environ.get("OLLAMA_API_KEY", "")
        self.ollama_model = os.environ.get("OLLAMA_MODEL", "llama3")
        self.git_enabled = bool(git_enabled)
        self.git_repo_dir = git_repo_dir or "."
        self.notification_sink = notification_sink
        self.checkpoint_interval_seconds = max(30, int(checkpoint_interval_seconds))
        self._lock = RLock()

        # A. Raw event memory
        self.raw_events: List[Dict[str, Any]] = []
        self.scan_summaries: List[Dict[str, Any]] = []
        self.scan_analysis_history: List[Dict[str, Any]] = []
        self.candidate_history: List[Dict[str, Any]] = []
        self.market_snapshots: List[Dict[str, Any]] = []
        self.pending_history: List[Dict[str, Any]] = []
        self.fill_history: List[Dict[str, Any]] = []
        self.trail_history: List[Dict[str, Any]] = []
        self.close_history: List[Dict[str, Any]] = []
        self.shadow_history: List[Dict[str, Any]] = []
        self.trade_history: List[Dict[str, Any]] = []

        # B. Feature/statistical memory
        self.feature_cache: Dict[str, Any] = {}
        self.scan_feature_cache: Dict[str, Any] = {}
        self.calibration_cache: Dict[str, Any] = {}
        self.frequency_cache: Dict[str, Any] = {}
        self.attribution_cache: Dict[str, Any] = {}
        self.regime_cache: Dict[str, Any] = {}

        # C. Decision memory
        self.threshold_history: List[Dict[str, Any]] = []
        self.strategy_change_log: List[Dict[str, Any]] = []
        self.decision_history: List[Dict[str, Any]] = []
        self.challenger_history: List[Dict[str, Any]] = []
        self.replay_history: List[Dict[str, Any]] = []
        self.ollama_critique_history: List[Dict[str, Any]] = []
        self.pending_challenger: Optional[Dict[str, Any]] = None
        self.last_audit_report: Dict[str, Any] = {}
        self.last_checkpoint_notification: Dict[str, Any] = {}

        # D. Strategy memory
        self.strategy_state: Dict[str, Any] = {}
        self.strategy_versions: Dict[str, Dict[str, Any]] = {}
        self.current_strategy_version: Optional[str] = None
        self.last_change_ts = 0.0
        self.trades_since_last_change = 0
        self.last_audit_ts = 0.0
        self.last_autosave_ts = 0.0
        self.last_checkpoint_ts = 0.0
        self.audit_sequence = 0
        self.event_sequence = 0
        self.scan_sequence = 0
        self._schema_version = SCHEMA_VERSION

        # Active in-memory counters. These let Learn "move" with scan data while
        # keeping the heavier audit/challenger cycle separate.
        self.live_counters: Dict[str, Any] = self._new_live_counters()
        self._last_scan_event_ts = 0.0
        self._last_event_log_ts = 0.0

        os.makedirs(os.path.dirname(self.checkpoint_path) or ".", exist_ok=True)
        self._logger("INIT", "engine=%s schema=%s checkpoint=%s", ENGINE_VERSION, SCHEMA_VERSION, self.checkpoint_path)

    # ------------------------------------------------------------------
    # Logging / notification plumbing
    # ------------------------------------------------------------------
    def _logger(self, status: str, message: str, *args: Any, level: int = logging.INFO) -> None:
        logger.log(level, "[LEARN] %s | " + message, status, *args)

    def _record_event_log(self, status: str, message: str, *args: Any, level: int = logging.INFO) -> None:
        self.event_sequence += 1
        logger.log(level, "[LEARN #%05d] %s | " + message, self.event_sequence, status, *args)

    def set_notification_sink(self, sink: Optional[Callable[[str, str], Any]]) -> None:
        with self._lock:
            self.notification_sink = sink
            self._record_event_log("NOTIFY", "notification sink %s", "ATTACHED" if sink else "DETACHED")

    def _notify(self, title: str, message: str) -> None:
        sink = self.notification_sink
        if sink is None:
            self._logger("TELEGRAM_PENDING", "%s | sink belum dipasang", title)
            return
        try:
            sink(message, title)
            self._logger("TELEGRAM_SENT", "%s", title)
        except Exception as exc:  # notification must never affect trading
            self._logger("TELEGRAM_FAILED", "%s: %s", title, exc, level=logging.WARNING)

    def get_checkpoint_notification(self) -> Dict[str, Any]:
        with self._lock:
            return copy.deepcopy(self.last_checkpoint_notification)

    def _emit_checkpoint_notification_locked(self, ok: bool, digest: str, reason: str) -> None:
        self.last_checkpoint_notification = {
            "timestamp": _now(),
            "ok": bool(ok),
            "checksum": digest,
            "reason": reason,
            "strategy_version": self.current_strategy_version,
            "trades": len(self.trade_history),
            "scans": len(self.scan_analysis_history) + len(self.candidate_history),
        }
        icon = "✅" if ok else "⚠️"
        msg = (
            f"{icon} LEARN CHECKPOINT\n"
            f"Status: {'OK' if ok else 'FAILED'}\n"
            f"Reason: {reason}\n"
            f"Strategy: v{self.current_strategy_version or '-'}\n"
            f"Trades: {len(self.trade_history)}\n"
            f"Scan analyses: {len(self.scan_analysis_history)}\n"
            f"Candidates: {len(self.candidate_history)}\n"
            f"SHA256: {digest[:16] if digest else '-'}"
        )
        self._notify("LEARN_CHECKPOINT", msg)

    # ------------------------------------------------------------------
    # Live counters / schemas
    # ------------------------------------------------------------------
    @staticmethod
    def _new_live_counters() -> Dict[str, Any]:
        return {
            "scans": 0,
            "processed_coins": 0,
            "candidate": 0,
            "eligible": 0,
            "threshold_reject": 0,
            "no_setup": 0,
            "diagnoses": Counter(),
            "directions": Counter(),
            "regimes": Counter(),
            "sessions": Counter(),
            "setup_types": Counter(),
            "strategy_versions": Counter(),
            "confidence_sum": 0.0,
            "rr_sum": 0.0,
            "entry_distance_sum": 0.0,
            "btc_aligned": 0,
            "btc_conflict": 0,
            "stale": 0,
            "updated_at": _now(),
        }

    @staticmethod
    def _validate_scan_row(row: Dict[str, Any]) -> Tuple[bool, str]:
        if not isinstance(row, dict):
            return False, "ROW_NOT_DICT"
        pair = str(row.get("pair", "")).strip()
        if not pair:
            return False, "PAIR_MISSING"
        ts = _finite_or_none(row.get("timestamp"))
        if ts is None:
            return False, "TIMESTAMP_INVALID"
        conf = _finite_or_none(row.get("confidence", 0.0))
        if conf is None:
            return False, "CONFIDENCE_INVALID"
        return True, "OK"

    @staticmethod
    def _validate_checkpoint_payload(data: Dict[str, Any]) -> Tuple[bool, str]:
        if not isinstance(data, dict):
            return False, "NOT_OBJECT"
        if _safe_int(data.get("schema_version"), -1) <= 0:
            return False, "SCHEMA_VERSION_INVALID"
        if not isinstance(data.get("trade_history", []), list):
            return False, "TRADE_HISTORY_INVALID"
        if not isinstance(data.get("candidate_history", []), list):
            return False, "CANDIDATE_HISTORY_INVALID"
        if not isinstance(data.get("scan_analysis_history", []), list):
            return False, "SCAN_ANALYSIS_HISTORY_INVALID"
        if not isinstance(data.get("strategy_state", {}), dict):
            return False, "STRATEGY_STATE_INVALID"
        return True, "OK"

    # ------------------------------------------------------------------
    # Checkpoint persistence
    # ------------------------------------------------------------------
    def _export_state(self) -> Dict[str, Any]:
        with self._lock:
            body: Dict[str, Any] = {
                "engine": ENGINE_NAME,
                "engine_version": ENGINE_VERSION,
                "schema_version": self._schema_version,
                "saved_at": _now(),
                "event_sequence": self.event_sequence,
                "audit_sequence": self.audit_sequence,
                "scan_sequence": self.scan_sequence,
                "trade_history": self.trade_history[-10000:],
                "scan_summaries": self.scan_summaries[-5000:],
                "scan_analysis_history": self.scan_analysis_history[-50000:],
                "candidate_history": self.candidate_history[-50000:],
                "market_snapshots": self.market_snapshots[-5000:],
                "pending_history": self.pending_history[-10000:],
                "fill_history": self.fill_history[-10000:],
                "trail_history": self.trail_history[-20000:],
                "close_history": self.close_history[-10000:],
                "shadow_history": self.shadow_history[-50000:],
                "raw_events": self.raw_events[-50000:],
                "feature_cache": self.feature_cache,
                "scan_feature_cache": self.scan_feature_cache,
                "calibration_cache": self.calibration_cache,
                "frequency_cache": self.frequency_cache,
                "attribution_cache": self.attribution_cache,
                "regime_cache": self.regime_cache,
                "threshold_history": self.threshold_history[-2000:],
                "strategy_change_log": self.strategy_change_log[-2000:],
                "decision_history": self.decision_history[-5000:],
                "challenger_history": self.challenger_history[-2000:],
                "replay_history": self.replay_history[-2000:],
                "ollama_critique_history": self.ollama_critique_history[-1000:],
                "pending_challenger": self.pending_challenger,
                "last_audit_report": self.last_audit_report,
                "last_checkpoint_notification": self.last_checkpoint_notification,
                "strategy_state": self.strategy_state,
                "strategy_versions": self.strategy_versions,
                "current_strategy_version": self.current_strategy_version,
                "last_change_ts": self.last_change_ts,
                "trades_since_last_change": self.trades_since_last_change,
                "last_audit_ts": self.last_audit_ts,
                "last_autosave_ts": self.last_autosave_ts,
                "last_checkpoint_ts": self.last_checkpoint_ts,
                "live_counters": self._serialise_live_counters(),
            }
            digest = _sha256_payload(body)
            body["checksum"] = digest
            return body

    def _serialise_live_counters(self) -> Dict[str, Any]:
        out = dict(self.live_counters)
        for key, value in list(out.items()):
            if isinstance(value, Counter):
                out[key] = dict(value)
        return out

    def _restore_state(self, data: Dict[str, Any]) -> None:
        ok, reason = self._validate_checkpoint_payload(data)
        if not ok:
            raise ValueError(f"checkpoint invalid: {reason}")
        stored_checksum = str(data.get("checksum", ""))
        body = dict(data)
        body.pop("checksum", None)
        actual_checksum = _sha256_payload(body)
        if stored_checksum and stored_checksum != actual_checksum:
            raise ValueError("checkpoint checksum mismatch")

        self._schema_version = _safe_int(data.get("schema_version"), SCHEMA_VERSION)
        self.event_sequence = _safe_int(data.get("event_sequence"), 0)
        self.audit_sequence = _safe_int(data.get("audit_sequence"), 0)
        self.scan_sequence = _safe_int(data.get("scan_sequence"), 0)
        for attr in (
            "trade_history", "scan_summaries", "scan_analysis_history", "candidate_history",
            "market_snapshots", "pending_history", "fill_history", "trail_history", "close_history",
            "shadow_history", "raw_events", "threshold_history", "strategy_change_log", "decision_history",
            "challenger_history", "replay_history", "ollama_critique_history",
        ):
            setattr(self, attr, list(data.get(attr, [])))
        for attr in (
            "feature_cache", "scan_feature_cache", "calibration_cache", "frequency_cache",
            "attribution_cache", "regime_cache", "last_audit_report", "last_checkpoint_notification",
            "strategy_state", "strategy_versions",
        ):
            setattr(self, attr, dict(data.get(attr, {})))
        self.pending_challenger = data.get("pending_challenger")
        self.current_strategy_version = data.get("current_strategy_version")
        self.last_change_ts = _safe_float(data.get("last_change_ts"), 0.0)
        self.trades_since_last_change = _safe_int(data.get("trades_since_last_change"), 0)
        self.last_audit_ts = _safe_float(data.get("last_audit_ts"), 0.0)
        self.last_autosave_ts = _safe_float(data.get("last_autosave_ts"), 0.0)
        self.last_checkpoint_ts = _safe_float(data.get("last_checkpoint_ts"), 0.0)
        restored = data.get("live_counters", self._new_live_counters())
        self.live_counters = dict(restored)
        for counter_key in ("diagnoses", "directions", "regimes", "sessions", "setup_types", "strategy_versions"):
            self.live_counters[counter_key] = Counter(restored.get(counter_key, {}))

    def _read_json_file(self, path: str) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def load(self) -> str:
        with self._lock:
            candidates = [
                (self.checkpoint_path, "primary"),
                (self.backup_path, "backup"),
            ]
            for path, label in candidates:
                if not os.path.exists(path):
                    continue
                try:
                    data = self._read_json_file(path)
                    self._restore_state(data)
                    self._record_event_log("CHECKPOINT_LOAD", "%s OK | checksum validated", label)
                    return label
                except (OSError, ValueError, json.JSONDecodeError) as exc:
                    self._record_event_log("CHECKPOINT_LOAD", "%s FAILED | %s", label, exc)
            self._record_event_log("CHECKPOINT_LOAD", "EMPTY | no valid checkpoint")
            return "empty"

    def validate_checkpoint(self) -> Dict[str, Any]:
        with self._lock:
            result = {
                "valid": False,
                "primary": False,
                "backup": False,
                "primary_reason": "MISSING",
                "backup_reason": "MISSING",
            }
            for path, key in ((self.checkpoint_path, "primary"), (self.backup_path, "backup")):
                if not os.path.exists(path):
                    continue
                try:
                    data = self._read_json_file(path)
                    ok, reason = self._validate_checkpoint_payload(data)
                    if ok:
                        stored = str(data.get("checksum", ""))
                        body = dict(data)
                        body.pop("checksum", None)
                        if stored and stored != _sha256_payload(body):
                            ok, reason = False, "CHECKSUM_MISMATCH"
                    result[key] = ok
                    result[f"{key}_reason"] = reason
                except Exception as exc:
                    result[f"{key}_reason"] = str(exc)
            result["valid"] = bool(result["primary"] or result["backup"])
            return result

    def save_checkpoint(self, reason: str = "manual") -> bool:
        with self._lock:
            try:
                data = self._export_state()
                digest = data.get("checksum", "")
                if os.path.exists(self.checkpoint_path):
                    shutil.copyfile(self.checkpoint_path, self.backup_path)
                _atomic_write_json(self.checkpoint_path, data)
                with open(self.checksum_path + ".tmp", "w", encoding="utf-8") as f:
                    f.write(str(digest))
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(self.checksum_path + ".tmp", self.checksum_path)
                self.last_checkpoint_ts = _now()
                self.last_autosave_ts = self.last_checkpoint_ts
                # Validate after writing, before reporting success.
                check = self.validate_checkpoint()
                ok = bool(check.get("primary"))
                self._record_event_log("CHECKPOINT", "%s | %s | sha=%s", "PASS" if ok else "FAIL", reason, str(digest)[:16])
                self._emit_checkpoint_notification_locked(ok, str(digest), reason)
                return ok
            except (OSError, TypeError, ValueError) as exc:
                self._record_event_log("CHECKPOINT", "FAIL | %s", exc)
                self._emit_checkpoint_notification_locked(False, "", f"save failed: {exc}")
                return False

    def autosave(self, reason: str = "autosave") -> None:
        try:
            ok = self.save_checkpoint(reason=reason)
            if ok and self.git_enabled:
                self._git_commit_push()
        except Exception as exc:  # pragma: no cover
            self._record_event_log("AUTOSAVE", "NON_FATAL | %s", exc)

    def _maybe_checkpoint_after_high_value_event_locked(self, reason: str) -> bool:
        """Checkpoint oportunistik tanpa melakukan write pada setiap event.

        Dipanggil ketika lock RLock sudah aktif. Event penting (mis. closed trade)
        hanya memicu checkpoint jika interval waktunya telah lewat. Dengan cara
        ini memory tetap persisten tanpa menghasilkan disk/Telegram burst.
        """
        if _now() - self.last_checkpoint_ts < self.checkpoint_interval_seconds:
            return False
        # save_checkpoint() menggunakan RLock sehingga aman dipanggil dari sini.
        return self.save_checkpoint(reason=f"high_value_{reason}")

    def maybe_checkpoint(self, reason: str = "periodic") -> bool:
        with self._lock:
            if _now() - self.last_checkpoint_ts < self.checkpoint_interval_seconds:
                return False
        return self.save_checkpoint(reason=reason)

    def _git_commit_push(self) -> None:
        try:
            checkpoint_abs = os.path.abspath(self.checkpoint_path)
            repo = os.path.abspath(self.git_repo_dir)
            rel = os.path.relpath(checkpoint_abs, repo)
            subprocess.run(["git", "add", "--", rel], cwd=repo, check=False, capture_output=True, timeout=5)
            subprocess.run(
                ["git", "commit", "-m", f"autosave learn {_now():.0f}"],
                cwd=repo, check=False, capture_output=True, timeout=5,
            )
            subprocess.run(["git", "push"], cwd=repo, check=False, capture_output=True, timeout=15)
        except Exception as exc:  # pragma: no cover
            self._record_event_log("GIT", "WARNING | %s", exc)

    # ------------------------------------------------------------------
    # Raw event ingestion
    # ------------------------------------------------------------------
    def _append_event(self, kind: str, payload: Dict[str, Any], *, importance: str = "NORMAL") -> None:
        row = dict(payload)
        row["kind"] = kind
        row.setdefault("timestamp", _now())
        row.setdefault("event_id", f"LE-{int(_now() * 1000)}-{self.event_sequence + 1}")
        row["importance"] = importance
        self.raw_events.append(row)
        if len(self.raw_events) > 60000:
            del self.raw_events[:-50000]

    def _append_scan_row(self, row: Dict[str, Any]) -> None:
        self.scan_analysis_history.append(row)
        if len(self.scan_analysis_history) > 80000:
            del self.scan_analysis_history[:-60000]

    # ------------------------------------------------------------------
    # Scan intelligence — the always-moving brain
    # ------------------------------------------------------------------
    def record_market_snapshot(self, snapshot: Dict[str, Any]) -> None:
        with self._lock:
            row = dict(snapshot or {})
            row.setdefault("timestamp", _now())
            row.setdefault("scan_sequence", self.scan_sequence)
            self.market_snapshots.append(row)
            self.market_snapshots = self.market_snapshots[-10000:]
            self._append_event("MARKET_SNAPSHOT", row)
            self._update_live_market_counters(row)
            self._record_event_log(
                "MARKET",
                "snapshot regime=%s breadth=%.1f/%.1f candidates=%s eligible=%s",
                row.get("btc_regime", row.get("regime", "UNKNOWN")),
                _safe_float(row.get("breadth_buy")),
                _safe_float(row.get("breadth_sell")),
                row.get("candidate_rate", row.get("candidate", 0)),
                row.get("eligible_rate", row.get("eligible", 0)),
            )
            self._micro_scan_update_locked()

    def _update_live_market_counters(self, snapshot: Dict[str, Any]) -> None:
        self.live_counters["scans"] += 1
        self.live_counters["updated_at"] = _now()
        self.live_counters["processed_coins"] += _safe_int(snapshot.get("processed"), 0)
        self.live_counters["candidate"] += _safe_int(snapshot.get("candidate_rate", snapshot.get("candidate")), 0)
        self.live_counters["eligible"] += _safe_int(snapshot.get("eligible_rate", snapshot.get("eligible")), 0)
        self.live_counters["threshold_reject"] += _safe_int(snapshot.get("threshold_rejected"), 0)
        regime = str(snapshot.get("btc_regime", snapshot.get("regime", "UNKNOWN")))
        self.live_counters["regimes"][regime] += 1

    def record_scan_analysis(
        self,
        symbol: str,
        candles_features: Optional[Dict[str, Any]] = None,
        diagnostics: Optional[Dict[str, Any]] = None,
        setup: Optional[Dict[str, Any]] = None,
        *,
        timestamp: Optional[float] = None,
        scan_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Ingest ONE coin scan. No API, no market-data fetch, analysis only."""
        with self._lock:
            ts = _safe_float(timestamp, _now())
            setup = dict(setup or {})
            diag = dict(diagnostics or {})
            features = dict(candles_features or {})
            context = dict(scan_context or {})
            pair = str(symbol or setup.get("pair") or "UNKNOWN").upper()

            conf = _safe_float(setup.get("confidence", diag.get("confidence", 0.0)))
            threshold = _safe_float(context.get("threshold", setup.get("threshold", 0.0)))
            eligible = bool(setup.get("threshold_passed", setup.get("eligible", conf >= threshold)))
            if threshold <= 0 and "eligible" not in setup:
                eligible = bool(setup) and conf >= threshold

            structure = dict(diag.get("structure") or {})
            liquidity = dict(diag.get("liquidity") or {})
            entry = dict(diag.get("entry") or {})
            tp = dict(diag.get("tp") or {})
            btc = dict(diag.get("btc") or {})
            freshness = dict(diag.get("freshness") or {})
            viability = dict(diag.get("viability") or {})
            context_diag = dict(diag.get("context") or {})

            direction = str(setup.get("direction", diag.get("direction", "UNKNOWN")))
            setup_type = str(setup.get("setup_type", diag.get("setup_type", "UNKNOWN")))
            regime = str(setup.get("regime", diag.get("regime", context.get("regime", "UNKNOWN"))))
            session = str(setup.get("session", diag.get("session", context.get("session", "UNKNOWN"))))
            rr = _safe_float(tp.get("rr", setup.get("reference_levels", {}).get("rr", 0.0)))
            distance_atr = _safe_float(entry.get("distance_atr", entry.get("entry_distance_atr", 0.0)))
            stale = bool(freshness.get("stale", diag.get("stale", False)))
            btc_aligned = btc.get("aligned") if "aligned" in btc else None

            diagnosis = self._diagnose_scan_locked(
                setup=setup,
                diagnostics=diag,
                confidence=conf,
                threshold=threshold,
                eligible=eligible,
                stale=stale,
                rr=rr,
                distance_atr=distance_atr,
                regime=regime,
                btc_aligned=btc_aligned,
            )

            rec = ScanFeatureRecord(
                pair=pair,
                timestamp=ts,
                confidence=conf,
                direction=direction,
                setup_type=setup_type,
                regime=regime,
                session=session,
                diagnosis=diagnosis,
                eligible=eligible,
                threshold=threshold,
                structure_score=_safe_float(setup.get("components", {}).get("structure", structure.get("score", 0.0))),
                liquidity_score=_safe_float(setup.get("components", {}).get("liquidity", liquidity.get("score", 0.0))),
                entry_score=_safe_float(setup.get("components", {}).get("entry_quality", entry.get("score", 0.0))),
                rr_score=_safe_float(setup.get("components", {}).get("risk_reward", tp.get("score", 0.0))),
                momentum_score=_safe_float(setup.get("components", {}).get("momentum", diag.get("momentum", {}).get("score", 0.0))),
                volatility_score=_safe_float(setup.get("components", {}).get("volatility", diag.get("volatility", {}).get("score", 0.0))),
                btc_score=_safe_float(setup.get("components", {}).get("btc_correlation", btc.get("score", 0.0))),
                regime_score=_safe_float(setup.get("components", {}).get("regime", context_diag.get("score", 0.0))),
                session_score=_safe_float(setup.get("components", {}).get("session", diag.get("session", {}).get("score", 0.0))),
                confirmation_score=_safe_float(setup.get("components", {}).get("confirmation", diag.get("confirmation", {}).get("score", 0.0))),
                entry_distance_atr=distance_atr,
                rr=rr,
                stale=stale,
                btc_aligned=btc_aligned if isinstance(btc_aligned, bool) else None,
                timestamp_age_seconds=max(0.0, _now() - ts),
                source_strategy_version=setup.get("strategy_version", context.get("strategy_version")),
                metadata={
                    "structure": structure,
                    "liquidity": liquidity,
                    "entry": entry,
                    "tp": tp,
                    "btc": btc,
                    "freshness": freshness,
                    "viability": viability,
                    "candles_features": features,
                    "scan_context": context,
                },
            )
            row = rec.to_dict()
            self._append_scan_row(row)
            self._append_event("SCAN_ANALYSIS", row)
            self.live_counters["processed_coins"] += 1
            self.live_counters["directions"][direction] += 1
            self.live_counters["regimes"][regime] += 1
            self.live_counters["sessions"][session] += 1
            self.live_counters["setup_types"][setup_type] += 1
            self.live_counters["diagnoses"][diagnosis] += 1
            self.live_counters["confidence_sum"] += conf
            self.live_counters["rr_sum"] += rr
            self.live_counters["entry_distance_sum"] += distance_atr
            self.live_counters["candidate"] += int(bool(setup))
            self.live_counters["eligible"] += int(eligible)
            self.live_counters["threshold_reject"] += int(bool(setup) and not eligible)
            self.live_counters["no_setup"] += int(not bool(setup))
            if btc_aligned is True:
                self.live_counters["btc_aligned"] += 1
            elif btc_aligned is False:
                self.live_counters["btc_conflict"] += 1
            if stale:
                self.live_counters["stale"] += 1
            if row["strategy_version"]:
                self.live_counters["strategy_versions"][str(row["strategy_version"])] += 1
            self._last_scan_event_ts = ts
            self._record_event_log(
                f"SCAN {pair}",
                "diag=%s conf=%.1f thr=%.1f eligible=%s rr=%.2f dist=%.2fATR regime=%s",
                diagnosis, conf, threshold, eligible, rr, distance_atr, regime,
            )
            self._micro_scan_update_locked()
            return row

    def _diagnose_scan_locked(
        self,
        *,
        setup: Dict[str, Any],
        diagnostics: Dict[str, Any],
        confidence: float,
        threshold: float,
        eligible: bool,
        stale: bool,
        rr: float,
        distance_atr: float,
        regime: str,
        btc_aligned: Optional[bool],
    ) -> str:
        # Diagnostics from Strategy take precedence when explicit.
        explicit = str(diagnostics.get("viability", {}).get("diagnosis", diagnostics.get("diagnosis", ""))).upper()
        if explicit in SCAN_DIAGNOSES:
            return explicit
        if not setup:
            return "NO_SETUP"
        if stale:
            return "STALE_SETUP"
        if distance_atr < 0.10:
            return "TOO_CLOSE"
        if distance_atr > 3.50:
            return "TOO_FAR"
        geom = str(diagnostics.get("geometry", {}).get("status", diagnostics.get("geometry_status", "OK"))).upper()
        if geom not in {"", "OK", "VALID", "TRUE"}:
            return "INVALID_GEOMETRY"
        ev = _safe_float(diagnostics.get("expected_value", diagnostics.get("ev", 0.0)), 0.0)
        if ev < -0.02:
            return "LOW_EXPECTED_VALUE"
        if btc_aligned is False:
            return "BTC_CONFLICT"
        if str(regime).upper() in {"REGIME_MISMATCH", "HIGH_VOLATILITY"} and diagnostics.get("context", {}).get("aligned") is False:
            return "REGIME_MISMATCH"
        if _safe_bool(diagnostics.get("liquidity", {}).get("low_liquidity"), False):
            return "LOW_LIQUIDITY_CONTEXT"
        if threshold > 0 and not eligible:
            return "VALID_LOW_CONF"
        if confidence >= max(threshold, 70.0):
            return "VALID_HIGH_CONF"
        return "VALID_LOW_CONF"

    def _micro_scan_update_locked(self) -> None:
        """Fast, non-governance update. It never changes Strategy params."""
        total_processed = max(1, self.live_counters["processed_coins"])
        candidate = self.live_counters["candidate"]
        eligible = self.live_counters["eligible"]
        self.scan_feature_cache = {
            "updated_at": _now(),
            "processed_coins": total_processed,
            "candidate_rate": candidate / total_processed,
            "eligible_rate": eligible / total_processed,
            "threshold_rejection_rate": self.live_counters["threshold_reject"] / total_processed,
            "no_setup_ratio": self.live_counters["no_setup"] / total_processed,
            "stale_ratio": self.live_counters["stale"] / total_processed,
            "btc_alignment_ratio": self.live_counters["btc_aligned"] / max(1, self.live_counters["btc_aligned"] + self.live_counters["btc_conflict"]),
            "avg_confidence": self.live_counters["confidence_sum"] / total_processed,
            "avg_rr": self.live_counters["rr_sum"] / total_processed,
            "avg_entry_distance_atr": self.live_counters["entry_distance_sum"] / total_processed,
            "diagnoses": dict(self.live_counters["diagnoses"]),
            "dominant_diagnosis": self.live_counters["diagnoses"].most_common(1)[0][0] if self.live_counters["diagnoses"] else None,
            "dominant_regime": self.live_counters["regimes"].most_common(1)[0][0] if self.live_counters["regimes"] else None,
            "dominant_session": self.live_counters["sessions"].most_common(1)[0][0] if self.live_counters["sessions"] else None,
        }
        self.frequency_cache["live"] = dict(self.scan_feature_cache)

    def record_scan_summary(self, summary: Dict[str, Any]) -> None:
        with self._lock:
            row = dict(summary or {})
            row.setdefault("timestamp", _now())
            self.scan_sequence += 1
            row.setdefault("scan_sequence", self.scan_sequence)
            self.scan_summaries.append(row)
            self.scan_summaries = self.scan_summaries[-10000:]
            self._append_event("SCAN_SUMMARY", row)
            self._update_live_market_counters(row)
            self._micro_scan_update_locked()
            freq = self._frequency_diagnosis_locked(window_scans=min(100, len(self.scan_summaries)))
            self.frequency_cache["last_scan"] = freq
            self._record_event_log(
                "SCAN SUMMARY",
                "seq=%s processed=%s candidates=%s eligible=%s freq=%s",
                row.get("scan_sequence"), row.get("processed", 0), row.get("candidate", 0), row.get("eligible", 0), freq.get("status"),
            )
            self.maybe_checkpoint(reason=f"scan_summary_{row.get('scan_sequence')}")

    def record_scan_candidate(self, setup: Dict[str, Any], eligible: bool, threshold: float, reason: str = "") -> None:
        with self._lock:
            row = {
                "pair": setup.get("pair"),
                "direction": setup.get("direction"),
                "confidence": _safe_float(setup.get("confidence")),
                "bucket": _bucket_of(setup.get("confidence", 0)),
                "setup_type": setup.get("setup_type", "UNKNOWN"),
                "regime": setup.get("regime", "UNKNOWN"),
                "session": setup.get("session", "UNKNOWN"),
                "components": dict(setup.get("components", {})),
                "entry": setup.get("entry"),
                "tp": setup.get("tp"),
                "sl": setup.get("sl"),
                "strategy_version": setup.get("strategy_version"),
                "threshold": _safe_float(threshold),
                "eligible": bool(eligible),
                "reason": reason,
                "timestamp": _now(),
                "reference_levels": dict(setup.get("reference_levels", {})),
            }
            self.candidate_history.append(row)
            self._append_event("CANDIDATE", row)
            self.live_counters["candidate"] += 1
            self.live_counters["eligible"] += int(bool(eligible))
            self.live_counters["threshold_reject"] += int(not eligible)
            self.live_counters["directions"][str(row["direction"])] += 1
            self.live_counters["regimes"][str(row["regime"])] += 1
            self.live_counters["sessions"][str(row["session"])] += 1
            self.live_counters["setup_types"][str(row["setup_type"])] += 1
            self.live_counters["confidence_sum"] += row["confidence"]
            self._micro_scan_update_locked()
            self._record_event_log(
                f"CANDIDATE {row['pair']}",
                "dir=%s C=%.1f%% eligible=%s threshold=%.1f reason=%s",
                row["direction"], row["confidence"], row["eligible"], row["threshold"], reason or "-",
            )

            # Candidate records can be upgraded into scan intelligence even when
            # the caller does not separately call record_scan_analysis().
            if not any(
                r.get("timestamp") == row["timestamp"] and r.get("pair") == row["pair"]
                for r in self.scan_analysis_history[-10:]
            ):
                self.record_scan_analysis(
                    str(row.get("pair") or "UNKNOWN"),
                    diagnostics={},
                    setup=row,
                    timestamp=row["timestamp"],
                    scan_context={"threshold": row["threshold"]},
                )

    def analyze_scan_memory(self, window: int = 5000) -> Dict[str, Any]:
        with self._lock:
            rows = self.scan_analysis_history[-max(1, window):]
            total = len(rows)
            if not rows:
                return {"status": "NO_DATA", "n": 0}
            diagnosis_counts = Counter(str(r.get("diagnosis", "NO_SETUP")) for r in rows)
            version_counts = Counter(str(r.get("strategy_version", "UNKNOWN")) for r in rows)
            return {
                "status": "OK",
                "n": total,
                "candidate_rate": _mean([1.0 if r.get("direction") not in (None, "UNKNOWN") else 0.0 for r in rows]),
                "eligible_rate": _mean([1.0 if r.get("eligible") else 0.0 for r in rows]),
                "avg_confidence": _mean([_safe_float(r.get("confidence")) for r in rows]),
                "median_confidence": _median([_safe_float(r.get("confidence")) for r in rows]),
                "avg_rr": _mean([_safe_float(r.get("rr")) for r in rows if _safe_float(r.get("rr")) > 0]),
                "avg_entry_distance_atr": _mean([_safe_float(r.get("entry_distance_atr")) for r in rows]),
                "stale_ratio": _mean([1.0 if r.get("stale") else 0.0 for r in rows]),
                "btc_conflict_ratio": _mean([1.0 if r.get("btc_aligned") is False else 0.0 for r in rows if r.get("btc_aligned") is not None]),
                "diagnoses": dict(diagnosis_counts),
                "dominant_diagnosis": diagnosis_counts.most_common(1)[0][0] if diagnosis_counts else "NO_SETUP",
                "strategy_versions": dict(version_counts),
            }

    # ------------------------------------------------------------------
    # Lifecycle events
    # ------------------------------------------------------------------
    def record_pending_event(self, position_or_setup: Dict[str, Any], timestamp: Optional[float] = None) -> None:
        with self._lock:
            row = dict(position_or_setup or {})
            row["timestamp"] = _safe_float(timestamp, _now())
            self.pending_history.append(row)
            self._append_event("PENDING", row)
            self._record_event_log("PENDING", "%s dir=%s C=%.1f%%", row.get("pair"), row.get("direction"), _safe_float(row.get("confidence")))

    def record_fill_event(self, position: Dict[str, Any], timestamp: Optional[float] = None, fill_price: Optional[float] = None) -> None:
        with self._lock:
            row = dict(position or {})
            row["timestamp"] = _safe_float(timestamp, _now())
            if fill_price is not None:
                row["fill_price"] = fill_price
            self.fill_history.append(row)
            self._append_event("FILL", row, importance="HIGH")
            self._record_event_log("FILL", "%s price=%s", row.get("pair"), row.get("fill_price", row.get("entry")))

    def record_trail_decision(self, decision: Dict[str, Any], position: Optional[Dict[str, Any]] = None) -> None:
        with self._lock:
            row = dict(decision or {})
            if position:
                row.setdefault("pair", position.get("pair"))
                row.setdefault("direction", position.get("direction"))
                row.setdefault("entry", position.get("entry"))
                row.setdefault("old_sl", position.get("sl"))
                row.setdefault("tp", position.get("tp"))
            row.setdefault("timestamp", _now())
            self.trail_history.append(row)
            self._append_event("TRAIL_DECISION", row)
            self._record_event_log(
                "TRAIL",
                "%s action=%s old=%s new=%s trigger=%s",
                row.get("pair"), row.get("action"), row.get("old_sl"), row.get("new_sl"), row.get("trigger"),
            )

    def record_trail_result(self, result: Dict[str, Any]) -> None:
        with self._lock:
            row = dict(result or {})
            row.setdefault("timestamp", _now())
            row["trail_result"] = True
            self.trail_history.append(row)
            self._append_event("TRAIL_RESULT", row)
            self._record_event_log("TRAIL RESULT", "%s applied=%s rejected=%s", row.get("pair"), row.get("applied"), row.get("rejected"))

    def record_close_event(self, event: Dict[str, Any]) -> None:
        with self._lock:
            row = dict(event or {})
            row.setdefault("timestamp", _now())
            self.close_history.append(row)
            self._append_event("CLOSE", row, importance="HIGH")
            self._record_event_log("CLOSE", "%s outcome=%s pnl_r=%.3f", row.get("pair"), row.get("outcome"), _safe_float(row.get("pnl_r")))

    def record_timeout_event(self, event: Dict[str, Any]) -> None:
        with self._lock:
            row = dict(event or {})
            row["outcome"] = "TIMEOUT"
            row.setdefault("timestamp", _now())
            self._append_event("TIMEOUT", row, importance="HIGH")
            self._record_event_log("TIMEOUT", "%s reason=%s", row.get("pair"), row.get("reason", row.get("close_reason", "-")))

    def record_shadow_outcome(self, candidate: Dict[str, Any], outcome: str, pnl_r: float = 0.0, **extra: Any) -> None:
        with self._lock:
            normalized = outcome if outcome in OUTCOME_TYPES else "TIMEOUT"
            row = dict(candidate or {})
            row.update(extra)
            row.update({
                "kind": "SHADOW_OUTCOME",
                "outcome": normalized,
                "pnl_r": _safe_float(pnl_r),
                "timestamp": _now(),
                "bucket": row.get("bucket", _bucket_of(row.get("confidence", 0))),
            })
            self.shadow_history.append(row)
            self._append_event("SHADOW_OUTCOME", row, importance="HIGH")
            self._record_event_log("SHADOW", "%s outcome=%s pnl_r=%.3f", row.get("pair"), normalized, row["pnl_r"])
            self._update_shadow_features_locked(row)

    def _update_shadow_features_locked(self, row: Dict[str, Any]) -> None:
        rows = [r for r in self.shadow_history[-5000:] if r.get("outcome") in OUTCOME_TYPES]
        by_diag = _group_by(rows, "diagnosis")
        self.attribution_cache["shadow"] = {
            "n": len(rows),
            "economic": len(_economic_rows(rows)),
            "by_diagnosis": {k: self._weighted_stats(v, SHADOW_HALF_LIFE_DAYS) for k, v in by_diag.items()},
        }

    def record_trade_outcome(self, setup: Dict[str, Any], outcome: str, close_info: Dict[str, Any]) -> None:
        normalized = outcome if outcome in OUTCOME_TYPES else "BE"
        with self._lock:
            setup = dict(setup or {})
            close_info = dict(close_info or {})
            row = {
                "pair": setup.get("pair"),
                "direction": setup.get("direction"),
                "confidence": _safe_float(setup.get("confidence")),
                "bucket": _bucket_of(setup.get("confidence", 0)),
                "setup_type": setup.get("setup_type", "UNKNOWN"),
                "regime": setup.get("regime", "UNKNOWN"),
                "session": setup.get("session", "UNKNOWN"),
                "components": dict(setup.get("components", {})),
                "strategy_version": setup.get("strategy_version"),
                "outcome": normalized,
                "pnl_pct": _safe_float(close_info.get("pnl_pct")),
                "pnl_r": _safe_float(close_info.get("pnl_r")),
                "trail_count": _safe_int(close_info.get("trail_count")),
                "entry_time": setup.get("fill_time", setup.get("timestamp")),
                "fill_time": setup.get("fill_time"),
                "fill_price": setup.get("fill_price", setup.get("entry")),
                "close_time": close_info.get("close_time", _now()),
                "timestamp": _now(),
                "trail_history": list(close_info.get("trail_history", setup.get("trail_history", [])) or []),
                "path_candles": _safe_path_candles(close_info.get("path_candles", [])),
                "initial_sl": setup.get("initial_sl", setup.get("sl")),
                "entry": setup.get("entry"),
                "tp": setup.get("tp"),
                "sl": setup.get("sl"),
                "reference_levels": dict(setup.get("reference_levels", {})),
                "diagnosis": setup.get("diagnosis", setup.get("viability", "UNKNOWN")),
            }
            attribution = self._attribute_trade_locked(row)
            row["attribution"] = attribution
            self.trade_history.append(row)
            self.trades_since_last_change += 1
            self._append_event("TRADE_OUTCOME", row, importance="HIGH")
            self._record_event_log(
                "TRADE OUTCOME",
                "%s %s C=%.1f%% R=%+.3f MAE=%.2f MFE=%.2f",
                row.get("pair"), normalized, row["confidence"], row["pnl_r"],
                _safe_float(attribution.get("mae_r")), _safe_float(attribution.get("mfe_r")),
            )
            self._update_feature_cache_locked()
            self._update_attribution_cache_locked()
            self._maybe_checkpoint_after_high_value_event_locked("trade_outcome")

    def _attribute_trade_locked(self, row: Dict[str, Any]) -> Dict[str, Any]:
        direction = str(row.get("direction", "BUY")).upper()
        entry = _safe_float(row.get("entry"))
        initial_sl = _safe_float(row.get("initial_sl"), entry)
        risk = abs(entry - initial_sl)
        path = row.get("path_candles") or []
        close_time = _event_ts({"timestamp": row.get("close_time")}, fallback=_now())
        fill_time = _event_ts({"timestamp": row.get("fill_time")}, fallback=0.0)
        entry_time = _event_ts({"timestamp": row.get("entry_time")}, fallback=0.0)
        # Compute MAE/MFE in actual R whenever initial risk is known.
        if path and risk > 0 and entry > 0:
            if direction == "BUY":
                mfe_price = max(_safe_float(c.get("h"), entry) for c in path)
                mae_price = min(_safe_float(c.get("l"), entry) for c in path)
                mfe_r = (mfe_price - entry) / risk
                mae_r = max(0.0, (entry - mae_price) / risk)
            else:
                mfe_price = min(_safe_float(c.get("l"), entry) for c in path)
                mae_price = max(_safe_float(c.get("h"), entry) for c in path)
                mfe_r = (entry - mfe_price) / risk
                mae_r = max(0.0, (mae_price - entry) / risk)
        else:
            mfe_price = entry
            mae_price = entry
            mfe_r = 0.0
            mae_r = 0.0
        tp = _safe_float(row.get("tp"))
        sl = _safe_float(row.get("sl"))
        tp_reached = False
        sl_reached = False
        if path and tp > 0:
            tp_reached = any(((_safe_float(c.get("h")) >= tp) if direction == "BUY" else (_safe_float(c.get("l")) <= tp)) for c in path)
        if path and sl > 0:
            sl_reached = any(((_safe_float(c.get("l")) <= sl) if direction == "BUY" else (_safe_float(c.get("h")) >= sl)) for c in path)
        holding_seconds = max(0.0, close_time - fill_time) if fill_time else 0.0
        time_to_entry = max(0.0, fill_time - entry_time) if fill_time and entry_time else 0.0
        mfe_time = self._time_to_extreme(path, direction, entry, is_mfe=True) if path else 0.0
        too_tight_sl = risk > 0 and mae_r >= 0.80 and row.get("outcome") == "INITIAL_SL"
        trail_applied = _safe_int(row.get("trail_count"), 0) > 0 or bool(row.get("trail_history"))
        trail_opportunity_lost = False
        if trail_applied and row.get("outcome") == "TRAIL" and path and tp > 0:
            # Approximate whether TP was reached later in the captured path. This
            # is strictly historical path analysis and does not influence prior entry.
            trail_opportunity_lost = tp_reached
        return {
            "mfe_r": round(max(0.0, mfe_r), 6),
            "mae_r": round(max(0.0, mae_r), 6),
            "mfe_price": round(mfe_price, 12),
            "mae_price": round(mae_price, 12),
            "tp_reached": bool(tp_reached),
            "sl_reached": bool(sl_reached),
            "time_to_entry_seconds": round(time_to_entry, 3),
            "holding_seconds": round(holding_seconds, 3),
            "time_to_mfe_seconds": round(mfe_time, 3),
            "trail_applied": trail_applied,
            "trail_opportunity_lost": bool(trail_opportunity_lost),
            "sl_too_tight_signal": bool(too_tight_sl),
            "tp_realistic_signal": bool(tp_reached or row.get("outcome") == "TP"),
            "entry_timing": self._entry_timing_label(time_to_entry, row),
        }

    @staticmethod
    def _time_to_extreme(path: Sequence[Dict[str, Any]], direction: str, entry: float, *, is_mfe: bool) -> float:
        if not path:
            return 0.0
        best_metric = -float("inf")
        best_t = _safe_float(path[0].get("t"), 0.0)
        for c in path:
            t = _safe_float(c.get("t"), 0.0)
            h = _safe_float(c.get("h"), entry)
            l = _safe_float(c.get("l"), entry)
            metric = (h - entry) if direction == "BUY" else (entry - l)
            if not is_mfe:
                metric = (entry - l) if direction == "BUY" else (h - entry)
            if metric > best_metric:
                best_metric = metric
                best_t = t
        first_t = _safe_float(path[0].get("t"), 0.0)
        if first_t <= 0 or best_t <= 0:
            return 0.0
        delta = best_t - first_t
        if abs(delta) > 1000000000:
            delta /= 1000.0
        return max(0.0, delta)

    @staticmethod
    def _entry_timing_label(time_to_entry: float, row: Dict[str, Any]) -> str:
        if time_to_entry <= 0:
            return "INSTANT_OR_MARKET"
        if time_to_entry < 15 * 60:
            return "FAST_FILL"
        if time_to_entry < 60 * 60:
            return "MODERATE_FILL"
        return "SLOW_FILL"

    # ------------------------------------------------------------------
    # Feature / statistics engine
    # ------------------------------------------------------------------
    def _weighted_stats(self, rows: Sequence[Dict[str, Any]], half_life_days: float = HALF_LIFE_DAYS) -> Dict[str, Any]:
        economic = _economic_rows(rows)
        if not economic:
            return StatResult().to_dict()
        ordered = _normalise_rows(economic)
        now = _now()
        triples: List[Tuple[float, float, Dict[str, Any]]] = []
        for row in ordered:
            pnl = _safe_float(row.get("pnl_r"))
            w = _time_decay_weight(_event_ts(row), now, half_life_days)
            triples.append((pnl, w, row))
        total_w = sum(w for _, w, _ in triples) or 1e-12
        pnl_values = [p for p, _, _ in triples]
        weighted_mean = sum(p * w for p, w, _ in triples) / total_w
        wins = [(p, w) for p, w, _ in triples if p > 0]
        losses = [(p, w) for p, w, _ in triples if p < 0]
        nonzero_w = sum(w for p, w, _ in triples if p != 0) or 1e-12
        win_w = sum(w for _, w in wins)
        gross_win = sum(p * w for p, w in wins)
        gross_loss = abs(sum(p * w for p, w in losses))
        pf = gross_win / gross_loss if gross_loss > 1e-12 else (999.0 if gross_win > 0 else 0.0)
        avg_win = gross_win / (sum(w for _, w in wins) or 1e-12)
        avg_loss = -gross_loss / (sum(w for _, w in losses) or 1e-12)
        raw_ci_low, raw_ci_high = _normal_ci(pnl_values)
        win_low, win_high = _binary_ci(len([p for p in pnl_values if p > 0]), len(pnl_values))
        _, dd = _drawdown_series(pnl_values)
        return StatResult(
            n=len(economic),
            effective_n=total_w,
            mean=weighted_mean,
            median=_median(pnl_values),
            stdev=statistics.stdev(pnl_values) if len(pnl_values) >= 2 else 0.0,
            ci_low=raw_ci_low,
            ci_high=raw_ci_high,
            win_rate=(win_w / nonzero_w) * 100.0,
            win_ci_low=win_low * 100.0,
            win_ci_high=win_high * 100.0,
            expectancy=weighted_mean,
            profit_factor=min(999.0, pf),
            avg_win=avg_win,
            avg_loss=avg_loss,
            downside_deviation=_downside_deviation(pnl_values),
            max_drawdown_r=dd,
            max_losing_streak=_max_losing_streak(pnl_values),
            sum_r=sum(pnl_values),
        ).to_dict()

    def _update_feature_cache_locked(self) -> None:
        confidence = self.confidence_calibration()
        regime = self.regime_performance()
        session = self.session_performance()
        direction = self.direction_performance()
        setup = self.setup_performance()
        component = self.component_performance()
        self.feature_cache = {
            "confidence": confidence,
            "regime": regime,
            "session": session,
            "direction": direction,
            "setup": setup,
            "components": component,
            "updated_at": _now(),
        }

    def _update_attribution_cache_locked(self) -> None:
        econ = _economic_rows(self.trade_history)
        attr = [dict(r.get("attribution", {})) for r in self.trade_history if r.get("attribution")]
        self.attribution_cache = {
            "n": len(econ),
            "avg_mae_r": _mean([_safe_float(a.get("mae_r")) for a in attr]),
            "median_mae_r": _median([_safe_float(a.get("mae_r")) for a in attr]),
            "avg_mfe_r": _mean([_safe_float(a.get("mfe_r")) for a in attr]),
            "median_mfe_r": _median([_safe_float(a.get("mfe_r")) for a in attr]),
            "tp_reach_rate": _mean([1.0 if a.get("tp_reached") else 0.0 for a in attr]) if attr else 0.0,
            "sl_reach_rate": _mean([1.0 if a.get("sl_reached") else 0.0 for a in attr]) if attr else 0.0,
            "trail_conversion_rate": _mean([1.0 if a.get("trail_applied") else 0.0 for a in attr]) if attr else 0.0,
            "trail_opportunity_lost_rate": _mean([1.0 if a.get("trail_opportunity_lost") else 0.0 for a in attr]) if attr else 0.0,
            "sl_too_tight_rate": _mean([1.0 if a.get("sl_too_tight_signal") else 0.0 for a in attr]) if attr else 0.0,
            "recent": self._weighted_attribution_locked(attr[-100:]),
        }

    @staticmethod
    def _weighted_attribution_locked(rows: Sequence[Dict[str, Any]]) -> Dict[str, float]:
        if not rows:
            return {}
        return {
            "mae_r": _mean([_safe_float(r.get("mae_r")) for r in rows]),
            "mfe_r": _mean([_safe_float(r.get("mfe_r")) for r in rows]),
            "holding_seconds": _mean([_safe_float(r.get("holding_seconds")) for r in rows]),
            "time_to_entry_seconds": _mean([_safe_float(r.get("time_to_entry_seconds")) for r in rows]),
            "time_to_mfe_seconds": _mean([_safe_float(r.get("time_to_mfe_seconds")) for r in rows]),
        }

    def confidence_calibration(self) -> Dict[str, Dict[str, Any]]:
        buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in self.trade_history:
            buckets[_bucket_of(row.get("confidence", 0))].append(row)
        return {k: self._weighted_stats(v) for k, v in sorted(buckets.items(), key=lambda x: int(x[0].split("-")[0]))}

    def regime_performance(self) -> Dict[str, Dict[str, Any]]:
        groups = _group_by(self.trade_history, "regime")
        return {k: self._weighted_stats(v) for k, v in groups.items()}

    def session_performance(self) -> Dict[str, Dict[str, Any]]:
        groups = _group_by(self.trade_history, "session")
        return {k: self._weighted_stats(v) for k, v in groups.items()}

    def direction_performance(self) -> Dict[str, Dict[str, Any]]:
        groups = _group_by(self.trade_history, "direction")
        return {k: self._weighted_stats(v) for k, v in groups.items()}

    def setup_performance(self) -> Dict[str, Dict[str, Any]]:
        groups = _group_by(self.trade_history, "setup_type")
        return {k: self._weighted_stats(v) for k, v in groups.items()}

    def component_performance(self) -> Dict[str, Dict[str, Any]]:
        values: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
        for row in self.trade_history:
            pnl = _safe_float(row.get("pnl_r"))
            for key, value in dict(row.get("components", {})).items():
                score = _safe_float(value)
                values[str(key)].append((score, pnl))
        out: Dict[str, Dict[str, Any]] = {}
        for key, pairs in values.items():
            xs = [x for x, _ in pairs]
            ys = [y for _, y in pairs]
            if not pairs:
                continue
            xm, ym = _mean(xs), _mean(ys)
            denom = math.sqrt(sum((x - xm) ** 2 for x in xs) * sum((y - ym) ** 2 for y in ys))
            corr = sum((x - xm) * (y - ym) for x, y in pairs) / denom if denom > 1e-12 else 0.0
            out[key] = {
                "n": len(pairs),
                "avg_score": round(xm, 6),
                "avg_pnl_r": round(ym, 6),
                "corr": round(corr, 6),
                "low_score_stats": self._weighted_stats([r for r in self.trade_history if _safe_float(dict(r.get("components", {})).get(key, 0.0)) <= _percentile(xs, 0.4)]),
                "high_score_stats": self._weighted_stats([r for r in self.trade_history if _safe_float(dict(r.get("components", {})).get(key, 0.0)) >= _percentile(xs, 0.6)]),
            }
        return out

    def _frequency_diagnosis_locked(self, window_scans: int = 50) -> Dict[str, Any]:
        recent = self.scan_summaries[-max(1, window_scans):]
        analysis = self.scan_analysis_history[-max(10, window_scans * 100):]
        if not recent and not analysis:
            result = {"status": "NO_DATA", "reason": "belum ada scan"}
            self.frequency_cache["current"] = result
            return result
        processed = _mean([_safe_float(s.get("processed")) for s in recent]) if recent else float(len(analysis))
        candidate = _mean([_safe_float(s.get("candidate", s.get("valid_strategy", 0))) for s in recent]) if recent else _mean([1.0 if r.get("direction") not in (None, "UNKNOWN") else 0.0 for r in analysis]) * max(processed, 1.0)
        eligible = _mean([_safe_float(s.get("eligible")) for s in recent]) if recent else _mean([1.0 if r.get("eligible") else 0.0 for r in analysis]) * max(processed, 1.0)
        rejects = Counter()
        for s in recent:
            for k, v in dict(s.get("rejects", s.get("reject_reasons", {}))).items():
                rejects[str(k)] += _safe_float(v)
        diagnosis_counts = Counter(str(r.get("diagnosis", "NO_SETUP")) for r in analysis)
        no_setup = diagnosis_counts.get("NO_SETUP", 0)
        stale = diagnosis_counts.get("STALE_SETUP", 0)
        too_close = diagnosis_counts.get("TOO_CLOSE", 0)
        too_far = diagnosis_counts.get("TOO_FAR", 0)
        invalid = diagnosis_counts.get("INVALID_GEOMETRY", 0)
        btc_conflict = diagnosis_counts.get("BTC_CONFLICT", 0)
        threshold_reject = diagnosis_counts.get("VALID_LOW_CONF", 0) + sum(v for k, v in rejects.items() if "THRESHOLD" in k.upper())
        total_analysis = max(1, len(analysis))

        status = "NORMAL"
        note = "frequency dalam rentang observasi"
        if processed < 5:
            status, note = "DATA_PIPELINE_WARNING", "terlalu sedikit coin yang diproses"
        elif total_analysis >= MIN_SCAN_EVENTS_FOR_PATTERN and no_setup / total_analysis >= 0.75:
            status, note = "STRUCTURE_OR_ENTRY_TOO_RESTRICTIVE", "NO_SETUP dominan pada scan"
        elif total_analysis >= MIN_SCAN_EVENTS_FOR_PATTERN and invalid / total_analysis >= 0.20:
            status, note = "GEOMETRY_REJECT_HIGH", "geometry rejection dominan"
        elif total_analysis >= MIN_SCAN_EVENTS_FOR_PATTERN and stale / total_analysis >= 0.20:
            status, note = "STALE_REJECT_HIGH", "setup stale terlalu dominan"
        elif total_analysis >= MIN_SCAN_EVENTS_FOR_PATTERN and too_close / total_analysis >= 0.10:
            status, note = "ENTRY_TOO_CLOSE", "entry geometry sering terlalu dekat"
        elif total_analysis >= MIN_SCAN_EVENTS_FOR_PATTERN and too_far / total_analysis >= 0.10:
            status, note = "ENTRY_TOO_FAR", "entry geometry sering terlalu jauh"
        elif total_analysis >= MIN_SCAN_EVENTS_FOR_PATTERN and btc_conflict / total_analysis >= 0.25:
            status, note = "BTC_FILTER_DOMINANT", "BTC conflict terlalu sering mematikan setup"
        elif total_analysis >= MIN_SCAN_EVENTS_FOR_PATTERN and threshold_reject / total_analysis >= 0.25:
            status, note = "THRESHOLD_TOO_HIGH_OR_STRICT", "banyak candidate jatuh setelah confidence threshold"
        elif candidate < max(0.5, processed * 0.01):
            status, note = "MARKET_MAY_BE_QUIET", "candidate rate sangat rendah tetapi belum cukup bukti bottleneck spesifik"

        result = {
            "status": status,
            "note": note,
            "scans": len(recent),
            "analysis_n": len(analysis),
            "avg_processed": round(processed, 4),
            "avg_candidate": round(candidate, 4),
            "avg_eligible": round(eligible, 4),
            "candidate_rate": round(candidate / max(processed, 1.0), 6),
            "eligible_rate": round(eligible / max(processed, 1.0), 6),
            "diagnoses": dict(diagnosis_counts),
            "rejects": dict(rejects),
            "threshold_rejection_rate": round(threshold_reject / total_analysis, 6),
            "stale_ratio": round(stale / total_analysis, 6),
            "btc_conflict_ratio": round(btc_conflict / total_analysis, 6),
        }
        self.frequency_cache["current"] = result
        return result

    def frequency_diagnosis(self, window_scans: int = 50) -> Dict[str, Any]:
        with self._lock:
            return self._frequency_diagnosis_locked(window_scans)

    def _weighted_rows_for_recent(self, rows: Sequence[Dict[str, Any]], count: int) -> List[Dict[str, Any]]:
        return _normalise_rows(rows)[-count:]

    def recent_vs_long_term(self, recent_n: int = 40, long_n: int = 200) -> Dict[str, Any]:
        with self._lock:
            recent = self._weighted_stats(self._weighted_rows_for_recent(self.trade_history, recent_n))
            long_rows = _normalise_rows(self.trade_history)[-long_n:]
            long_term = self._weighted_stats(long_rows)
            return {
                "recent": recent,
                "long_term": long_term,
                "delta_expectancy": round(recent.get("expectancy", 0.0) - long_term.get("expectancy", 0.0), 6),
                "delta_pf": round(recent.get("profit_factor", 0.0) - long_term.get("profit_factor", 0.0), 6),
                "delta_win_rate": round(recent.get("win_rate", 0.0) - long_term.get("win_rate", 0.0), 6),
            }

    def overall_stats(self) -> Dict[str, Any]:
        with self._lock:
            stats = self._weighted_stats(self.trade_history)
            counts = {o: sum(1 for t in self.trade_history if t.get("outcome") == o) for o in OUTCOME_TYPES}
            economic = _economic_rows(self.trade_history)
            freq = self._frequency_diagnosis_locked()
            calibration = self.confidence_calibration()
            rec_vs_long = self.recent_vs_long_term()
            return {
                **stats,
                "outcome_counts": counts,
                "timeout_count": counts.get("TIMEOUT", 0),
                "confidence_avg_closed": round(_mean(_safe_float(t.get("confidence")) for t in economic), 4) if economic else 0.0,
                "last_trades": copy.deepcopy(self.trade_history[-10:]),
                "regime": self.regime_performance(),
                "session": self.session_performance(),
                "direction": self.direction_performance(),
                "setup": self.setup_performance(),
                "calibration": calibration,
                "frequency": freq,
                "scan": self.analyze_scan_memory(window=5000),
                "quality_quantity": self.quality_quantity_matrix(),
                "attribution": copy.deepcopy(self.attribution_cache),
                "recent_vs_long_term": rec_vs_long,
                "strategy_version": self.current_strategy_version,
            }

    # ------------------------------------------------------------------
    # Frequency x quality matrix
    # ------------------------------------------------------------------
    def quality_quantity_matrix(self) -> Dict[str, Any]:
        with self._lock:
            stats = self._weighted_stats(self.trade_history)
            freq = self._frequency_diagnosis_locked()
            q_high = (
                stats.get("expectancy", 0.0) >= QUALITY_HIGH_EXPECTANCY_R
                and stats.get("profit_factor", 0.0) >= QUALITY_MIN_PF
                and stats.get("max_drawdown_r", 0.0) <= QUALITY_MAX_DRAWDOWN_R
            )
            q_low = stats.get("expectancy", 0.0) < 0.0 or stats.get("profit_factor", 0.0) < 1.0
            f_healthy = (
                freq.get("candidate_rate", 0.0) >= FREQUENCY_HEALTHY_CANDIDATE_PER_SCAN
                or freq.get("analysis_n", 0) < MIN_SCAN_EVENTS_FOR_PATTERN
            )
            f_low = not f_healthy
            if q_high and f_healthy:
                decision = "KEEP"
            elif q_high and f_low:
                decision = "RELAX_BOTTLENECK"
            elif q_low and f_healthy:
                decision = "TIGHTEN_OR_CHANGE_MODEL"
            elif q_low and f_low:
                decision = "SEARCH_REGIME_MODEL_ISSUE"
            elif q_high and stats.get("max_drawdown_r", 0.0) > QUALITY_MAX_DRAWDOWN_R:
                decision = "FIX_RISK_GEOMETRY_OR_EXIT"
            else:
                decision = "OBSERVE"
            return {
                "quality": {"high": q_high, "low": q_low, **stats},
                "quantity": {"healthy": f_healthy, "low": f_low, **freq},
                "decision": decision,
            }

    # ------------------------------------------------------------------
    # Counterfactual replay
    # ------------------------------------------------------------------
    @staticmethod
    def _chronological_split(rows: Sequence[Dict[str, Any]], train_fraction: float = 0.70, min_holdout: int = MIN_HOLDOUT_SAMPLE) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        ordered = _normalise_rows(rows)
        if len(ordered) < 2:
            return ordered, []
        cut = max(1, int(len(ordered) * train_fraction))
        if len(ordered) - cut < min_holdout:
            cut = max(1, len(ordered) - min_holdout)
        return ordered[:cut], ordered[cut:]

    def counterfactual_threshold(self, rows: Sequence[Dict[str, Any]], threshold: float) -> Dict[str, Any]:
        economic = _economic_rows(rows)
        baseline = self._weighted_stats(economic)
        selected = [r for r in economic if _safe_float(r.get("confidence")) >= float(threshold)]
        challenger = self._weighted_stats(selected)
        return {
            "baseline": baseline,
            "challenger": challenger,
            "selected_n": len(selected),
            "selection_rate": len(selected) / max(1, len(economic)),
            "delta_expectancy": round(challenger.get("expectancy", 0.0) - baseline.get("expectancy", 0.0), 6),
        }

    def chronological_replay(
        self,
        rows: Optional[Sequence[Dict[str, Any]]] = None,
        *,
        threshold: Optional[float] = None,
        sl_multiplier: float = 1.0,
        tp_multiplier: float = 1.0,
        trail_enabled: Optional[bool] = None,
        entry_offset_multiplier: float = 1.0,
        component_mask: Optional[Dict[str, bool]] = None,
    ) -> Dict[str, Any]:
        """Replay historical candidates using only already recorded event data."""
        with self._lock:
            base = list(rows if rows is not None else self.trade_history)
            ordered = _normalise_rows(base)
            simulated: List[Dict[str, Any]] = []
            for row in ordered:
                confidence = _safe_float(row.get("confidence"))
                if threshold is not None and confidence < float(threshold):
                    continue
                if component_mask:
                    comps = dict(row.get("components", {}))
                    masked_conf = sum(
                        _safe_float(v) for k, v in comps.items() if component_mask.get(k, True)
                    )
                    if comps and masked_conf <= 0:
                        continue
                # We do not invent unseen future prices. Use historical R and
                # attribution to transform only the modeled geometry effects.
                base_r = _safe_float(row.get("pnl_r"))
                attr = dict(row.get("attribution", {}))
                mae = _safe_float(attr.get("mae_r"))
                mfe = _safe_float(attr.get("mfe_r"))
                adjusted = base_r
                if sl_multiplier < 1.0 and mae > 0 and mae >= sl_multiplier:
                    adjusted = min(adjusted, -sl_multiplier)
                if tp_multiplier != 1.0 and adjusted > 0:
                    adjusted *= min(2.5, max(0.25, tp_multiplier))
                if trail_enabled is False and row.get("outcome") == "TRAIL":
                    # Waiting may recover part of lost MFE, conservatively capped at
                    # observed MFE rather than fabricating a path.
                    adjusted = min(max(adjusted, 0.0), mfe)
                elif trail_enabled is True and row.get("outcome") not in ("TRAIL", "INITIAL_SL") and mfe > adjusted:
                    adjusted = min(adjusted, mfe)
                if entry_offset_multiplier != 1.0:
                    # Only a small deterministic sensitivity around the observed R.
                    adjusted *= _clamp(1.0 - 0.05 * abs(entry_offset_multiplier - 1.0), 0.75, 1.0)
                simulated.append({"timestamp": _event_ts(row), "pnl_r": adjusted, "source": row})
            sim_rows = [{"timestamp": x["timestamp"], "pnl_r": x["pnl_r"], "outcome": "TP" if x["pnl_r"] > 0 else "INITIAL_SL"} for x in simulated]
            stats = self._weighted_stats(sim_rows)
            result = {
                "n": len(sim_rows),
                "stats": stats,
                "parameters": {
                    "threshold": threshold,
                    "sl_multiplier": sl_multiplier,
                    "tp_multiplier": tp_multiplier,
                    "trail_enabled": trail_enabled,
                    "entry_offset_multiplier": entry_offset_multiplier,
                    "component_mask": dict(component_mask or {}),
                },
                "chronological": True,
            }
            self.replay_history.append({"timestamp": _now(), **result})
            self.replay_history = self.replay_history[-2000:]
            self._record_event_log("REPLAY", "n=%s expectancy=%+.4f PF=%.3f", len(sim_rows), stats.get("expectancy", 0.0), stats.get("profit_factor", 0.0))
            return result

    # ------------------------------------------------------------------
    # Challenger governance
    # ------------------------------------------------------------------
    def register_challenger(self, proposed_params: Dict[str, Any], reason: str, evidence: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        with self._lock:
            challenger = {
                "id": f"CH-{int(_now() * 1000)}",
                "created_at": _now(),
                "status": "PENDING",
                "proposed_params": copy.deepcopy(proposed_params),
                "reason": reason,
                "evidence": copy.deepcopy(evidence or {}),
                "train": None,
                "validation": None,
                "holdout": None,
                "robustness": None,
                "frequency_impact": None,
                "drawdown_impact": None,
                "rollback_plan": {
                    "previous_strategy_version": self.current_strategy_version,
                    "previous_state": copy.deepcopy(self.strategy_state),
                },
            }
            self.pending_challenger = challenger
            self.challenger_history.append(copy.deepcopy(challenger))
            self.decision_history.append({"type": "CHALLENGER_CREATED", "timestamp": _now(), "id": challenger["id"]})
            self._append_event("CHALLENGER_CREATED", challenger, importance="HIGH")
            self._record_event_log("CHALLENGER", "%s created reason=%s", challenger["id"], reason)
            return copy.deepcopy(challenger)

    def validate_candidate_parameter_change(self, current_params: Dict[str, Any], proposed_params: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
        changed: List[str] = []
        deltas: Dict[str, Any] = {}
        for key, new_value in proposed_params.items():
            changed.append(key)
            if key == "ACTIVE_THRESHOLD":
                old = _safe_float(current_params.get(key, 0.0))
                new = _safe_float(new_value)
                if not 0.0 <= new <= 95.0:
                    return False, "threshold di luar 0..95", {"key": key, "old": old, "new": new}
                if abs(new - old) > MAX_THRESHOLD_STEP:
                    return False, "perubahan threshold terlalu besar", {"old": old, "new": new}
                deltas[key] = new - old
            elif key == "CONFIDENCE_WEIGHTS":
                if not isinstance(new_value, dict):
                    return False, "CONFIDENCE_WEIGHTS harus object", {}
                old_weights = dict(current_params.get(key, {}))
                delta_map = {}
                for wkey, wval in new_value.items():
                    ov = _safe_float(old_weights.get(wkey, 0.0))
                    nv = _safe_float(wval)
                    if abs(nv - ov) > 10.0:
                        return False, f"weight {wkey} berubah terlalu besar", {"old": ov, "new": nv}
                    delta_map[wkey] = nv - ov
                deltas[key] = delta_map
            elif key in MAX_PARAM_STEP:
                if key not in current_params:
                    return False, f"parameter tidak dikenal: {key}", {}
                old = _safe_float(current_params.get(key))
                new = _safe_float(new_value)
                if not math.isfinite(new):
                    return False, f"parameter {key} non-finite", {}
                if abs(new - old) > MAX_PARAM_STEP[key]:
                    return False, f"perubahan {key} terlalu besar", {"old": old, "new": new, "max": MAX_PARAM_STEP[key]}
                deltas[key] = new - old
            else:
                return False, f"parameter tidak diizinkan: {key}", {}
        return True, "parameter change shape valid", {"changed": changed, "deltas": deltas}

    def _proposal_effects_from_history(self, proposed: Dict[str, Any]) -> Dict[str, Any]:
        threshold = proposed.get("ACTIVE_THRESHOLD")
        if threshold is not None:
            return self.counterfactual_threshold(self.trade_history, _safe_float(threshold))
        # Generic replay proxy for other parameters.
        return self.chronological_replay(self.trade_history)

    def evaluate_challenger(self, challenger: Dict[str, Any], *, holdout_fraction: float = 0.25) -> Dict[str, Any]:
        with self._lock:
            proposed = dict(challenger.get("proposed_params", {}))
            current = dict(self.strategy_state.get("params", {}))
            shape_ok, shape_reason, shape_meta = self.validate_candidate_parameter_change(current, proposed)
            if not shape_ok:
                return {"status": "REJECTED", "reason": shape_reason, "shape": shape_meta}

            train, holdout = self._chronological_split(self.trade_history, train_fraction=1.0 - holdout_fraction, min_holdout=MIN_HOLDOUT_SAMPLE)
            if len(train) < MIN_TRAIN_SAMPLE or len(holdout) < MIN_HOLDOUT_SAMPLE:
                return {
                    "status": "INSUFFICIENT_SAMPLE",
                    "train_n": len(train),
                    "holdout_n": len(holdout),
                    "min_train": MIN_TRAIN_SAMPLE,
                    "min_holdout": MIN_HOLDOUT_SAMPLE,
                }

            baseline_train = self._weighted_stats(train)
            baseline_holdout = self._weighted_stats(holdout)
            current_threshold = _safe_float(current.get("ACTIVE_THRESHOLD", 0.0))
            proposed_threshold = proposed.get("ACTIVE_THRESHOLD")
            if proposed_threshold is not None:
                proposed_threshold = _safe_float(proposed_threshold)
                train_cf = self.counterfactual_threshold(train, proposed_threshold)
                hold_selected = [r for r in holdout if _safe_float(r.get("confidence")) >= proposed_threshold and r.get("outcome") in ECONOMIC_OUTCOMES]
                challenger_holdout = self._weighted_stats(hold_selected)
            else:
                train_cf = self.chronological_replay(train)
                challenger_holdout = self.chronological_replay(holdout).get("stats", {})

            frequency_before = self._frequency_diagnosis_locked()
            frequency_after = self._counterfactual_frequency_for_threshold_locked(proposed_threshold) if proposed_threshold is not None else frequency_before
            robustness = self._robustness_probe_locked(train, proposed)
            holdout_gate, holdout_reason, hold_meta = self._holdout_gate(baseline_holdout, challenger_holdout)
            train_gate = self._train_gate(baseline_train, train_cf.get("challenger", train_cf.get("stats", {})))
            robustness_gate = robustness.get("pass", False)
            frequency_gate = self._frequency_gate(frequency_before, frequency_after, proposed)
            final_pass = bool(holdout_gate and train_gate[0] and robustness_gate and frequency_gate[0])

            evaluation = {
                "status": "PASS" if final_pass else "FAIL",
                "shape_validation": shape_meta,
                "shape_reason": shape_reason,
                "train": {"baseline": baseline_train, "challenger": train_cf.get("challenger", train_cf.get("stats", {})), "gate": train_gate},
                "holdout": hold_meta,
                "holdout_reason": holdout_reason,
                "frequency_before": frequency_before,
                "frequency_after": frequency_after,
                "frequency_gate": frequency_gate,
                "robustness": robustness,
                "gates": {
                    "train": train_gate[0],
                    "holdout": holdout_gate,
                    "robustness": robustness_gate,
                    "frequency": frequency_gate[0],
                },
                "decision": "ACCEPT" if final_pass else "DEFER",
                "evaluated_at": _now(),
            }
            self.pending_challenger = dict(challenger)
            self.pending_challenger.update({"evaluation": evaluation, "status": "VALIDATED" if final_pass else "REJECTED"})
            self.challenger_history.append(copy.deepcopy(self.pending_challenger))
            self._append_event("CHALLENGER_EVALUATION", evaluation)
            self._record_event_log(
                "CHALLENGER",
                "%s result=%s train=%s holdout=%s robust=%s freq=%s",
                challenger.get("id"), evaluation["status"], train_gate[0], holdout_gate, robustness_gate, frequency_gate[0],
            )
            return evaluation

    def _train_gate(self, baseline: Dict[str, Any], challenger: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
        if _safe_int(challenger.get("n")) < MIN_TRAIN_SAMPLE:
            return False, "train sample terlalu kecil", {}
        delta = _safe_float(challenger.get("expectancy")) - _safe_float(baseline.get("expectancy"))
        pf_ok = _safe_float(challenger.get("profit_factor")) >= max(0.90, _safe_float(baseline.get("profit_factor")) * 0.90)
        ok = delta >= -0.05 and pf_ok
        return ok, f"train delta expectancy={delta:+.4f} PF gate={pf_ok}", {"delta_expectancy": delta, "pf_ok": pf_ok}

    def _holdout_gate(self, baseline: Dict[str, Any], challenger: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
        n = _safe_int(challenger.get("n"))
        if n < MIN_HOLDOUT_SAMPLE:
            return False, f"holdout terlalu kecil ({n}/{MIN_HOLDOUT_SAMPLE})", {"baseline": baseline, "challenger": challenger}
        delta = _safe_float(challenger.get("expectancy")) - _safe_float(baseline.get("expectancy"))
        pf_ok = _safe_float(challenger.get("profit_factor")) >= max(0.90, _safe_float(baseline.get("profit_factor")) * 0.90)
        dd_ok = _safe_float(challenger.get("max_drawdown_r"), 0.0) <= max(QUALITY_MAX_DRAWDOWN_R * 1.25, _safe_float(baseline.get("max_drawdown_r")) * 1.25)
        ok = delta >= -0.10 and pf_ok and dd_ok
        meta = {"baseline": baseline, "challenger": challenger, "delta_expectancy": delta, "pf_ok": pf_ok, "dd_ok": dd_ok}
        return ok, f"holdout delta={delta:+.4f} PF={pf_ok} DD={dd_ok}", meta

    def _robustness_probe_locked(self, train: Sequence[Dict[str, Any]], proposed: Dict[str, Any]) -> Dict[str, Any]:
        base_threshold = _safe_float(self.strategy_state.get("params", {}).get("ACTIVE_THRESHOLD", 0.0))
        proposal_threshold = proposed.get("ACTIVE_THRESHOLD")
        if proposal_threshold is None:
            threshold = base_threshold
        else:
            threshold = _safe_float(proposal_threshold)
        nearby = sorted(set([
            max(0.0, threshold - 2.0),
            max(0.0, threshold - 1.0),
            threshold,
            min(95.0, threshold + 1.0),
            min(95.0, threshold + 2.0),
        ]))
        evals = []
        for t in nearby:
            st = self.counterfactual_threshold(train, t)["challenger"]
            evals.append({"threshold": t, "expectancy": st.get("expectancy", 0.0), "pf": st.get("profit_factor", 0.0), "n": st.get("n", 0)})
        if not evals:
            return {"pass": False, "reason": "no robustness data"}
        exp_values = [e["expectancy"] for e in evals]
        center = next((e for e in evals if e["threshold"] == threshold), evals[len(evals) // 2])
        spread = max(exp_values) - min(exp_values)
        enough = center["n"] >= MIN_SAMPLE_FOR_DECISION
        stable = spread <= max(0.35, abs(center["expectancy"]) * 3.0 + 0.10)
        return {"pass": bool(enough and stable), "enough": enough, "stable": stable, "spread": spread, "grid": evals}

    def _counterfactual_frequency_for_threshold_locked(self, threshold: Optional[float]) -> Dict[str, Any]:
        if threshold is None:
            return self._frequency_diagnosis_locked()
        rows = self.scan_analysis_history[-10000:]
        if not rows:
            return self._frequency_diagnosis_locked()
        selected = [r for r in rows if _safe_float(r.get("confidence")) >= threshold and r.get("direction") not in (None, "UNKNOWN")]
        eligible = [r for r in selected if r.get("eligible") or _safe_float(r.get("confidence")) >= threshold]
        return {
            "status": "COUNTERFACTUAL",
            "analysis_n": len(rows),
            "candidate_rate": len(selected) / max(1, len(rows)),
            "eligible_rate": len(eligible) / max(1, len(rows)),
            "avg_confidence": _mean([_safe_float(r.get("confidence")) for r in selected]),
            "threshold": threshold,
        }

    def _frequency_gate(self, before: Dict[str, Any], after: Dict[str, Any], proposed: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
        # A frequency change is accepted only when quality evidence is not degraded
        # and the frequency objective is not being manipulated from a tiny sample.
        before_rate = _safe_float(before.get("eligible_rate", before.get("candidate_rate")))
        after_rate = _safe_float(after.get("eligible_rate", after.get("candidate_rate")))
        sample = _safe_int(after.get("analysis_n"), 0)
        if sample and sample < MIN_SCAN_EVENTS_FOR_PATTERN:
            return False, "frequency sample terlalu kecil", {"sample": sample}
        change = after_rate - before_rate
        threshold_changed = "ACTIVE_THRESHOLD" in proposed
        if threshold_changed and change < -0.02:
            return False, "proposal menurunkan opportunity rate terlalu besar", {"delta_rate": change}
        return True, "frequency gate OK", {"delta_rate": change}

    # ------------------------------------------------------------------
    # Threshold recommendation / model recommendations
    # ------------------------------------------------------------------
    def shadow_performance_below(self, threshold: float) -> Dict[str, Any]:
        with self._lock:
            rows = [
                r for r in self.shadow_history
                if _safe_float(r.get("confidence")) < _safe_float(threshold)
                and r.get("outcome") in ECONOMIC_OUTCOMES
            ]
            st = self._weighted_stats(rows, SHADOW_HALF_LIFE_DAYS)
            return {k: st.get(k, 0.0) for k in ("n", "effective_n", "win_rate", "expectancy", "profit_factor", "max_drawdown_r")}

    def _recommend_threshold(self, calibration: Dict[str, Dict[str, Any]], current: float, freq: Dict[str, Any]) -> Optional[Tuple[float, Dict[str, Any]]]:
        usable = [(bucket, st) for bucket, st in calibration.items() if _safe_int(st.get("n")) >= MIN_SAMPLE_FOR_DECISION]
        usable.sort(key=lambda x: int(x[0].split("-")[0]))
        if len(usable) < 2:
            return None
        current = _safe_float(current)
        bad_low = [(b, s) for b, s in usable if int(b.split("-")[0]) <= current + 10 and _safe_float(s.get("expectancy")) < -0.05]
        good_high = [(b, s) for b, s in usable if int(b.split("-")[0]) > current and _safe_float(s.get("expectancy")) > 0.05]
        if bad_low and good_high:
            target = float(int(good_high[0][0].split("-")[0]))
            new = round(max(0.0, min(95.0, current + min(MAX_THRESHOLD_STEP, max(1.0, target - current)))), 1)
            if abs(new - current) >= 0.5:
                return new, {"type": "RAISE_THRESHOLD", "bad_low": dict(bad_low), "good_high": dict(good_high), "frequency": freq}
        if current > 0 and freq.get("status") == "THRESHOLD_TOO_HIGH_OR_STRICT":
            shadow = self.shadow_performance_below(current)
            if _safe_int(shadow.get("n")) >= MIN_SAMPLE_FOR_DECISION and _safe_float(shadow.get("expectancy")) > 0.05:
                new = round(max(0.0, current - min(3.0, MAX_THRESHOLD_STEP)), 1)
                if new < current:
                    return new, {"type": "LOWER_THRESHOLD_FROM_SHADOW", "shadow": shadow, "frequency": freq}
        return None

    def _recommend_bottleneck_relaxation(self, freq: Dict[str, Any]) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
        diagnosis = str(freq.get("status", ""))
        params = dict(self.strategy_state.get("params", {}))
        if diagnosis == "ENTRY_TOO_CLOSE":
            current = _safe_float(params.get("entry_min_offset_atr", 0.25))
            new = max(0.05, current - 0.05)
            return ({"entry_min_offset_atr": new}, {"type": "RELAX_ENTRY_DISTANCE", "old": current, "new": new, "frequency": freq}) if new < current else None
        if diagnosis == "STALE_REJECT_HIGH":
            current = _safe_float(params.get("stale_setup_minutes", 30.0), 30.0)
            new = min(60.0, current + 5.0)
            return ({"stale_setup_minutes": new}, {"type": "RELAX_STALE_HORIZON", "old": current, "new": new, "frequency": freq}) if new > current else None
        if diagnosis == "BTC_FILTER_DOMINANT":
            current_weights = dict(params.get("CONFIDENCE_WEIGHTS", {}))
            if "btc_correlation" in current_weights:
                new_weights = dict(current_weights)
                new_weights["btc_correlation"] = max(3.0, _safe_float(current_weights["btc_correlation"]) - 1.0)
                return ({"CONFIDENCE_WEIGHTS": new_weights}, {"type": "RELAX_BTC_WEIGHT", "old": current_weights, "new": new_weights, "frequency": freq})
        return None

    # ------------------------------------------------------------------
    # Ollama critic — advisor only
    # ------------------------------------------------------------------
    def _ollama_critique(self, context: Dict[str, Any]) -> Optional[str]:
        if not self.ollama_url or requests is None:
            self._record_event_log("OLLAMA", "SKIP | unavailable")
            return None
        prompt_context = {
            "instruction": (
                "Anda hanya menjadi critic statistik. Jangan memberi order. Jangan mengubah parameter. "
                "Tinjau blind spots, contradictory evidence, confounding, frequency-quality tradeoff, "
                "exit attribution, trail critique, dan data freshness concern."
            ),
            "evidence": context,
        }
        try:
            headers = {"Content-Type": "application/json"}
            if self.ollama_api_key:
                headers["Authorization"] = f"Bearer {self.ollama_api_key}"
            response = requests.post(
                f"{self.ollama_url.rstrip('/')}/api/generate",
                headers=headers,
                json={"model": self.ollama_model, "prompt": json.dumps(prompt_context, ensure_ascii=False, default=str)[:15000], "stream": False},
                timeout=10,
            )
            if response.status_code == 200:
                text = str(response.json().get("response", "")).strip()[:4000]
                note = {"timestamp": _now(), "response": text, "model": self.ollama_model}
                self.ollama_critique_history.append(note)
                self.ollama_critique_history = self.ollama_critique_history[-1000:]
                self._append_event("OLLAMA_CRITIC", note)
                self._record_event_log("OLLAMA", "DONE | chars=%s", len(text))
                return text
            self._record_event_log("OLLAMA", "HTTP %s", response.status_code, level=logging.WARNING)
        except Exception as exc:
            self._record_event_log("OLLAMA", "UNAVAILABLE | %s", exc, level=logging.WARNING)
        return None

    # ------------------------------------------------------------------
    # Strategy state and rollback
    # ------------------------------------------------------------------
    def set_strategy_state(self, state: Dict[str, Any]) -> None:
        with self._lock:
            new_state = dict(state or {})
            version = new_state.get("version")
            self.strategy_state = new_state
            self.current_strategy_version = version or self.current_strategy_version
            if version:
                self.strategy_versions[str(version)] = copy.deepcopy(new_state)
            self._append_event("STRATEGY_STATE", new_state)
            self._record_event_log("STRATEGY", "STATE v%s loaded", version or "-")

    def get_strategy_state(self) -> Dict[str, Any]:
        with self._lock:
            return copy.deepcopy(self.strategy_state)

    def _should_auto_change(self) -> Tuple[bool, str]:
        now = _now()
        if self.trades_since_last_change < MIN_TRADES_SINCE_LAST_CHANGE:
            return False, f"sample gate {self.trades_since_last_change}/{MIN_TRADES_SINCE_LAST_CHANGE}"
        if now - self.last_change_ts < AUDIT_COOLDOWN_SECONDS:
            return False, "time cooldown aktif"
        return True, "sample + time gate PASS"

    def rollback_to_version(self, strategy_engine: Any, version: str, reason: str = "manual rollback") -> Dict[str, Any]:
        with self._lock:
            state = self.strategy_versions.get(str(version))
            if not state:
                return {"status": "NOT_FOUND", "version": version}
            try:
                previous = strategy_engine.export_state()
                strategy_engine.load_state(copy.deepcopy(state))
                self.strategy_state = copy.deepcopy(state)
                self.current_strategy_version = state.get("version", version)
                record = {
                    "timestamp": _now(),
                    "type": "ROLLBACK",
                    "from": previous.get("version"),
                    "to": self.current_strategy_version,
                    "reason": reason,
                }
                self.decision_history.append(record)
                self.strategy_change_log.append(record)
                self._append_event("ROLLBACK", record, importance="HIGH")
                self._record_event_log("ROLLBACK", "%s -> %s reason=%s", previous.get("version"), self.current_strategy_version, reason)
                return {"status": "ROLLED_BACK", **record}
            except Exception as exc:
                self._record_event_log("ROLLBACK", "FAILED | %s", exc, level=logging.WARNING)
                return {"status": "FAILED", "reason": str(exc), "version": version}

    # ------------------------------------------------------------------
    # Version health / degradation
    # ------------------------------------------------------------------
    def evaluate_current_version_degradation(self) -> Dict[str, Any]:
        with self._lock:
            version = self.current_strategy_version
            rows = [r for r in self.trade_history if r.get("strategy_version") == version]
            if len(rows) < MIN_TOTAL_SAMPLE_FOR_AUDIT:
                return {"status": "INSUFFICIENT", "n": len(rows), "version": version}
            ordered = _normalise_rows(rows)
            recent = ordered[-MIN_TOTAL_SAMPLE_FOR_AUDIT:]
            prior = ordered[:-MIN_TOTAL_SAMPLE_FOR_AUDIT]
            if len(prior) < MIN_SAMPLE_FOR_DECISION:
                return {"status": "INSUFFICIENT_BASELINE", "n": len(prior), "version": version}
            recent_st = self._weighted_stats(recent)
            prior_st = self._weighted_stats(prior)
            delta = recent_st.get("expectancy", 0.0) - prior_st.get("expectancy", 0.0)
            degraded = delta < -ROLLBACK_DEGRADATION_R
            return {
                "status": "DEGRADED" if degraded else "STABLE",
                "version": version,
                "recent": recent_st,
                "prior": prior_st,
                "delta_expectancy": round(delta, 6),
            }

    # ------------------------------------------------------------------
    # Main audit loop
    # ------------------------------------------------------------------
    def audit(self, strategy_engine: Any) -> Dict[str, Any]:
        with self._lock:
            self.audit_sequence += 1
            now = _now()
            self._record_event_log("AUDIT START", "seq=%s", self.audit_sequence)
            self._record_event_log("DATA VALIDATION", "scan_events=%s trades=%s", len(self.scan_analysis_history), len(self.trade_history))

            # Fast feature refresh — this is why Learn "moves" even when no trade closes.
            self._micro_scan_update_locked()
            scan_analysis = self.analyze_scan_memory(window=10000)
            frequency = self._frequency_diagnosis_locked(window_scans=100)
            self._record_event_log("FREQUENCY ANALYSIS", "status=%s candidate_rate=%.4f eligible_rate=%.4f", frequency.get("status"), frequency.get("candidate_rate", 0.0), frequency.get("eligible_rate", 0.0))

            quality = self._weighted_stats(self.trade_history)
            calibration = self.confidence_calibration()
            attribution = copy.deepcopy(self.attribution_cache)
            qq = self.quality_quantity_matrix()
            self._record_event_log("QUALITY ANALYSIS", "n=%s expectancy=%+.4f PF=%.3f DD=%.3f", quality.get("n"), quality.get("expectancy", 0.0), quality.get("profit_factor", 0.0), quality.get("max_drawdown_r", 0.0))
            self._record_event_log("EXIT ATTRIBUTION", "MAE=%.3f MFE=%.3f trail_lost=%.3f", attribution.get("avg_mae_r", 0.0), attribution.get("avg_mfe_r", 0.0), attribution.get("trail_opportunity_lost_rate", 0.0))

            report: Dict[str, Any] = {
                "timestamp": now,
                "audit_sequence": self.audit_sequence,
                "action": "NO_ACTION",
                "reason": "",
                "strategy_version": getattr(strategy_engine, "version", None),
                "quality": quality,
                "frequency": frequency,
                "scan": scan_analysis,
                "calibration": calibration,
                "attribution": attribution,
                "quality_quantity": qq,
                "version_health": self.evaluate_current_version_degradation(),
                "challenger": copy.deepcopy(self.pending_challenger),
                "ollama_critique": None,
            }

            # Always run a critic on statistical summary, never raw candles.
            report["ollama_critique"] = self._ollama_critique({
                "quality": quality,
                "frequency": frequency,
                "scan": scan_analysis,
                "attribution": attribution,
                "quality_quantity": qq,
                "strategy_version": getattr(strategy_engine, "version", None),
            })
            if report["ollama_critique"]:
                self._record_event_log("OLLAMA CRITIC DONE", "stored")

            # Safety: never update because of a single trade/scan.
            if report["version_health"].get("status") == "DEGRADED":
                report["action"] = "ROLLBACK_RECOMMENDED"
                report["reason"] = "versi strategy terbaru mengalami degradation material"
                self.decision_history.append({"type": "ROLLBACK_RECOMMENDED", "timestamp": now, "report": report})
                self._record_event_log("DECISION", "ROLLBACK_RECOMMENDED")
                self.last_audit_report = report
                self.last_audit_ts = now
                self.maybe_checkpoint(reason="audit_rollback_recommended")
                return report

            if len(self.trade_history) < MIN_TOTAL_SAMPLE_FOR_AUDIT:
                # Even with insufficient trade sample, frequency/scanning analysis remains active.
                scan_diag = self._scan_only_decision_locked(frequency, scan_analysis)
                report["scan_decision"] = scan_diag
                report["reason"] = f"sample trade belum cukup ({len(self.trade_history)}/{MIN_TOTAL_SAMPLE_FOR_AUDIT}); scan brain tetap aktif"
                self._record_event_log("DECISION", "OBSERVE_ONLY | %s", report["reason"])
                self.last_audit_report = report
                self.last_audit_ts = now
                self.maybe_checkpoint(reason="audit_observe_only")
                return report

            can_change, gate_reason = self._should_auto_change()
            if not can_change:
                report["reason"] = gate_reason
                self._record_event_log("DECISION", "KEEP | %s", gate_reason)
                self.last_audit_report = report
                self.last_audit_ts = now
                self.maybe_checkpoint(reason="audit_keep")
                return report

            current_params = dict(getattr(strategy_engine, "params", {}))
            current_threshold = _safe_float(getattr(strategy_engine, "get_active_threshold", lambda: current_params.get("ACTIVE_THRESHOLD", 0.0))())
            recommendation = self._recommend_threshold(calibration, current_threshold, frequency)
            if recommendation is None:
                recommendation = self._recommend_bottleneck_relaxation(frequency)

            if recommendation is None:
                report["reason"] = "tidak ada bukti cukup untuk perubahan parameter"
                self._record_event_log("DECISION", "KEEP | no evidence-backed parameter change")
                self.last_audit_report = report
                self.last_audit_ts = now
                self.maybe_checkpoint(reason="audit_no_change")
                return report

            if isinstance(recommendation[0], dict):
                proposed = recommendation[0]
                evidence = recommendation[1]
            else:
                new_threshold, evidence = recommendation
                proposed = {"ACTIVE_THRESHOLD": new_threshold}
            self._record_event_log("CHALLENGER START", "proposal=%s", proposed)

            shape_ok, shape_reason, shape_meta = self.validate_candidate_parameter_change(current_params, proposed)
            evidence = dict(evidence)
            evidence["shape_validation"] = shape_meta
            if not shape_ok:
                report["action"] = "REJECTED"
                report["reason"] = shape_reason
                self._record_event_log("CHALLENGER", "REJECTED SHAPE | %s", shape_reason)
                self.last_audit_report = report
                self.last_audit_ts = now
                self.maybe_checkpoint(reason="audit_rejected_shape")
                return report

            challenger = self.register_challenger(proposed, evidence.get("type", "MODEL_CHANGE"), evidence)
            evaluation = self.evaluate_challenger(challenger)
            evidence["evaluation"] = evaluation
            report["challenger"] = copy.deepcopy(self.pending_challenger)
            report["evidence"] = evidence

            if evaluation.get("status") != "PASS":
                report["action"] = "DEFERRED"
                report["reason"] = "challenger belum lolos seluruh statistical/holdout/robustness/frequency gates"
                self._record_event_log("HOLDOUT", "FAIL / DEFERRED")
                self.decision_history.append({"type": "DEFERRED", "timestamp": now, "proposal": proposed, "evidence": evidence})
                self.last_audit_report = report
                self.last_audit_ts = now
                self.maybe_checkpoint(reason="audit_challenger_deferred")
                return report

            # Final application gate. Only Learn may call Strategy.apply_update.
            try:
                change_record = strategy_engine.apply_update(
                    proposed,
                    reason=f"Learn validated: {evidence.get('type', 'MODEL_CHANGE')}",
                    evidence=evidence,
                )
            except Exception as exc:
                report["action"] = "REJECTED"
                report["reason"] = f"strategy.apply_update gagal: {exc}"
                self._record_event_log("APPLY", "FAILED | %s", exc, level=logging.WARNING)
                self.decision_history.append({"type": "APPLY_FAILED", "timestamp": now, "proposal": proposed, "error": str(exc)})
                self.last_audit_report = report
                self.last_audit_ts = now
                self.maybe_checkpoint(reason="audit_apply_failed")
                return report

            self.threshold_history.append({
                "timestamp": now,
                "old_threshold": current_threshold,
                "new_threshold": _safe_float(proposed.get("ACTIVE_THRESHOLD", current_threshold)),
                "evidence": evidence,
            })
            self.strategy_change_log.append(change_record)
            self.strategy_versions[str(change_record.get("version"))] = strategy_engine.export_state()
            self.decision_history.append({
                "type": "ACCEPTED",
                "timestamp": now,
                "proposal": proposed,
                "evidence": evidence,
                "change_record": change_record,
            })
            self.pending_challenger = None
            self.trades_since_last_change = 0
            self.last_change_ts = now
            self.current_strategy_version = change_record.get("version")
            self.strategy_state = strategy_engine.export_state()
            report.update({
                "action": "APPLIED",
                "reason": "statistical + chronological + holdout + robustness + frequency gates passed",
                "old_threshold": current_threshold,
                "new_threshold": proposed.get("ACTIVE_THRESHOLD", current_threshold),
                "evidence": evidence,
                "strategy_version": change_record.get("version"),
            })
            self._record_event_log("HOLDOUT", "PASS")
            self._record_event_log("DECISION", "APPLIED | strategy=%s", change_record.get("version"))
            self.last_audit_report = report
            self.last_audit_ts = now
            self.maybe_checkpoint(reason="audit_change_applied")
            return report

    def _scan_only_decision_locked(self, frequency: Dict[str, Any], scan_analysis: Dict[str, Any]) -> Dict[str, Any]:
        status = str(frequency.get("status", "NO_DATA"))
        diagnosis = scan_analysis.get("dominant_diagnosis")
        if status in {"STRUCTURE_OR_ENTRY_TOO_RESTRICTIVE", "ENTRY_TOO_CLOSE", "ENTRY_TOO_FAR", "STALE_REJECT_HIGH", "GEOMETRY_REJECT_HIGH"}:
            decision = "INVESTIGATE_BOTTLENECK"
        elif status == "THRESHOLD_TOO_HIGH_OR_STRICT":
            decision = "OBSERVE_THRESHOLD"
        else:
            decision = "KEEP_OBSERVING"
        return {"decision": decision, "frequency_status": status, "dominant_diagnosis": diagnosis}

    # ------------------------------------------------------------------
    # Dashboard / summaries
    # ------------------------------------------------------------------
    def get_last_audit_report(self) -> Dict[str, Any]:
        with self._lock:
            return copy.deepcopy(self.last_audit_report)

    def should_run_audit(self, interval_seconds: int = 300) -> bool:
        with self._lock:
            return _now() - self.last_audit_ts >= max(1, interval_seconds)

    def export_memory_summary(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "engine": ENGINE_NAME,
                "engine_version": ENGINE_VERSION,
                "schema_version": self._schema_version,
                "trades": len(self.trade_history),
                "scan_summaries": len(self.scan_summaries),
                "scan_analysis": len(self.scan_analysis_history),
                "candidates": len(self.candidate_history),
                "shadow": len(self.shadow_history),
                "pending": len(self.pending_history),
                "fills": len(self.fill_history),
                "trails": len(self.trail_history),
                "strategy_version": self.current_strategy_version,
                "strategy_versions": list(self.strategy_versions.keys()),
                "frequency": copy.deepcopy(self.frequency_cache),
                "scan_features": copy.deepcopy(self.scan_feature_cache),
                "attribution": copy.deepcopy(self.attribution_cache),
                "pending_challenger": copy.deepcopy(self.pending_challenger),
                "last_audit": copy.deepcopy(self.last_audit_report),
                "checkpoint": self.validate_checkpoint(),
            }


# ---------------------------------------------------------------------------
# Compatibility aliases / helper functions
# ---------------------------------------------------------------------------
def new_default_learn(
    checkpoint_path: str = "state/learn_checkpoint.json",
    **kwargs: Any,
) -> LearnEngine:
    return LearnEngine(checkpoint_path=checkpoint_path, **kwargs)


__all__ = [
    "LearnEngine",
    "StatResult",
    "ScanFeatureRecord",
    "OUTCOME_TYPES",
    "ECONOMIC_OUTCOMES",
    "CONFIDENCE_BUCKETS",
    "new_default_learn",
]
