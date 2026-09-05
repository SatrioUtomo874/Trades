"""
learn.py — Adaptive Audit & Learning Brain
==========================================

Learn adalah auditor strategy, bukan trader.
Kontrak keras:
- tidak mengambil market data/API sendiri;
- tidak membuat/membatalkan order;
- tidak memblokir keputusan entry yang sudah dibuat main.py;
- tidak mengubah strategy karena satu trade;
- Ollama hanya critic/advisor;
- perubahan strategy wajib lolos statistical gate + holdout/counterfactual gate;
- checkpoint atomic dengan primary + backup.

Memory dibagi menjadi:
1) raw event memory
2) feature/statistical memory
3) decision memory
4) strategy memory

Kompatibilitas main.py dijaga untuk method:
load(), save_checkpoint(), autosave(), set_strategy_state(),
record_scan_summary(), record_scan_candidate(), record_shadow_outcome(),
record_trade_outcome(), audit(), overall_stats().
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import shutil
import time
from threading import RLock
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import quote

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None

logger = logging.getLogger("learn")

SCHEMA_VERSION = 4
HALF_LIFE_DAYS = 21.0
SHADOW_HALF_LIFE_DAYS = 14.0

CONFIDENCE_BUCKETS = [
    (0, 40), (40, 50), (50, 60), (60, 70), (70, 80), (80, 90), (90, 101)
]

OUTCOME_TYPES = ("TP", "INITIAL_SL", "TRAIL", "BE", "TIMEOUT")
ECONOMIC_OUTCOMES = ("TP", "INITIAL_SL", "TRAIL", "BE")

# Conservative gates. These are deliberately stronger than a simple win-rate rule.
MIN_TOTAL_SAMPLE_FOR_AUDIT = 40
MIN_SAMPLE_FOR_DECISION = 30
MIN_HOLDOUT_SAMPLE = 20
MIN_TRADES_SINCE_LAST_CHANGE = 20
MAX_THRESHOLD_STEP = 5.0
MAX_PARAM_STEP = {
    "min_rr": 0.20,
    "displacement_atr_mult": 0.20,
    "sweep_lookback": 10,
    "structure_lookback": 10,
    "trend_lookback": 10,
    "btc_corr_lookback": 10,
    "sl_atr_buffer": 0.10,
    "entry_retracement_fib": 0.03,
    "entry_min_offset_atr": 0.10,
}
AUDIT_COOLDOWN_SECONDS = 15 * 60
ROLLBACK_DEGRADATION_R = 0.30


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


def _time_decay_weight(ts: float, now: Optional[float] = None, half_life_days: float = HALF_LIFE_DAYS) -> float:
    now = time.time() if now is None else now
    age_days = max(0.0, (now - _safe_float(ts, now)) / 86400.0)
    return 0.5 ** (age_days / max(0.1, half_life_days))


def _bucket_of(confidence: float) -> str:
    c = max(0.0, min(100.0, _safe_float(confidence)))
    for lo, hi in CONFIDENCE_BUCKETS:
        if lo <= c < hi:
            return f"{lo}-{hi - 1}"
    return "90-100"


def _atomic_write_json(path: str, data: Dict[str, Any]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _mean(values: Iterable[float]) -> float:
    xs = list(values)
    return sum(xs) / len(xs) if xs else 0.0


class LearnEngine:
    """Thread-safe learning/audit engine with conservative strategy governance."""

    def __init__(
        self,
        checkpoint_path: str = "state/learn_checkpoint.json",
        backup_path: Optional[str] = None,
        ollama_url: Optional[str] = None,
        ollama_api_key: Optional[str] = None,
        git_enabled: bool = False,
        git_repo_dir: Optional[str] = None,
    ):
        self.checkpoint_path = checkpoint_path
        self.backup_path = backup_path or (checkpoint_path + ".backup")
        self.ollama_url = ollama_url or os.environ.get("OLLAMA_URL", "http://localhost:11434")
        self.ollama_api_key = ollama_api_key or os.environ.get("OLLAMA_API_KEY", "")
        self.git_enabled = bool(git_enabled)
        self.git_repo_dir = git_repo_dir or "."
        self.github_token = os.environ.get("GITHUB_TOKEN", "")
        self.github_repo = os.environ.get("REPO_NAME", "")
        self.github_branch = os.environ.get("GITHUB_BRANCH", "main")
        # Stable, visible GitHub mirror. /open deliberately never reads this path.
        self.git_memory_path = os.path.join(self.git_repo_dir, "memory", "learn_autosave.json")
        self.github_memory_path = "memory/learn_autosave.json"
        self.github_memory_backup_path = "memory/learn_autosave.backup.json"
        self._last_github_digest = ""
        self._lock = RLock()

        # A. raw event memory
        self.raw_events: List[Dict[str, Any]] = []
        self.trade_history: List[Dict[str, Any]] = []
        self.scan_summaries: List[Dict[str, Any]] = []
        self.candidate_history: List[Dict[str, Any]] = []
        self.shadow_history: List[Dict[str, Any]] = []

        # B. feature/statistical memory
        self.feature_cache: Dict[str, Any] = {}
        self.calibration_cache: Dict[str, Any] = {}
        self.frequency_cache: Dict[str, Any] = {}

        # C. decision memory
        self.threshold_history: List[Dict[str, Any]] = []
        self.strategy_change_log: List[Dict[str, Any]] = []
        self.decision_history: List[Dict[str, Any]] = []
        self.pending_challenger: Optional[Dict[str, Any]] = None
        self.last_audit_report: Dict[str, Any] = {}

        # D. strategy memory
        self.strategy_state: Dict[str, Any] = {}
        self.current_strategy_version: Optional[str] = None

        self.trades_since_last_change = 0
        self.last_change_ts = 0.0
        self.last_audit_ts = 0.0
        self.last_autosave_ts = 0.0
        self._schema_version = SCHEMA_VERSION
        # Learning can be paused safely for /save and /open.  The condition is
        # deliberately local: it never touches trading state or exchange APIs.
        self._learning_paused = False
        self._active_operation = 0

        os.makedirs(os.path.dirname(self.checkpoint_path) or ".", exist_ok=True)

    def _finite_check(self, value: Any, path: str = "root") -> None:
        """Reject NaN/inf recursively before a learning snapshot is considered valid."""
        if isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError(f"non-finite value at {path}")
            return
        if isinstance(value, dict):
            for k, v in value.items():
                self._finite_check(v, f"{path}.{k}")
        elif isinstance(value, (list, tuple)):
            for i, v in enumerate(value):
                self._finite_check(v, f"{path}[{i}]")

    def _snapshot_digest(self, data: Dict[str, Any]) -> str:
        payload = json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def _validate_snapshot(self, data: Dict[str, Any]) -> Tuple[bool, str]:
        try:
            if not isinstance(data, dict):
                return False, "snapshot bukan object JSON"
            if _safe_int(data.get("schema_version"), -1) != SCHEMA_VERSION:
                return False, f"schema version tidak cocok: {data.get('schema_version')} != {SCHEMA_VERSION}"
            self._finite_check(data)
            # Required containers must exist and have their expected basic types.
            for key in ("trade_history", "scan_summaries", "candidate_history", "shadow_history", "raw_events", "threshold_history", "strategy_change_log", "decision_history"):
                if not isinstance(data.get(key), list):
                    return False, f"field {key} bukan list"
            for key in ("feature_cache", "calibration_cache", "frequency_cache", "strategy_state"):
                if not isinstance(data.get(key, {}), dict):
                    return False, f"field {key} bukan dict"
            digest = data.get("integrity", {}).get("sha256") if isinstance(data.get("integrity"), dict) else None
            if digest:
                body = dict(data)
                body.pop("integrity", None)
                if digest != self._snapshot_digest(body):
                    return False, "integrity SHA256 tidak cocok"
            return True, "OK"
        except (TypeError, ValueError, OverflowError) as e:
            return False, str(e)

    def pause(self) -> None:
        with self._lock:
            self._learning_paused = True
            logger.info("[GLOBAL] [LEARN] PAUSE — safe point requested")

    def resume(self) -> None:
        with self._lock:
            self._learning_paused = False
            logger.info("[GLOBAL] [LEARN] RESUME — brain active")

    def is_paused(self) -> bool:
        with self._lock:
            return self._learning_paused

    def _build_ready_snapshot_locked(self, source: str) -> Dict[str, Any]:
        # Must be called while _lock is held and while no audit is in progress.
        data = self._export_state()
        data["snapshot_kind"] = "READY_BRAIN_MEMORY"
        data["snapshot_source"] = source
        data["snapshot_seq"] = sum(len(x) for x in (self.trade_history, self.candidate_history, self.shadow_history, self.raw_events))
        self._finite_check(data)
        body = dict(data)
        body.pop("integrity", None)
        data["integrity"] = {
            "algorithm": "sha256",
            "sha256": self._snapshot_digest(body),
            "validated_at": time.time(),
        }
        return data

    def _write_ready_file_locked(self, path: str, source: str) -> Tuple[bool, str]:
        data = self._build_ready_snapshot_locked(source)
        ok, reason = self._validate_snapshot(data)
        if not ok:
            return False, reason
        tmp = path + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
            return True, "OK"
        except (OSError, TypeError, ValueError) as e:
            try:
                if os.path.exists(tmp): os.remove(tmp)
            except OSError:
                pass
            return False, str(e)

    def manual_save(self, ready_path: Optional[str] = None, backup_path: Optional[str] = None) -> Dict[str, Any]:
        """Create a validated, atomic, human-invoked brain snapshot at a safe point."""
        ready_path = ready_path or os.path.join(os.path.dirname(self.checkpoint_path) or ".", "learn_memory_ready.json")
        backup_path = backup_path or (ready_path + ".backup")
        with self._lock:
            self._learning_paused = True
            try:
                logger.info("[GLOBAL] [LEARN] SAVE — reaching safe point")
                if os.path.exists(ready_path):
                    try:
                        with open(ready_path, "r", encoding="utf-8") as f:
                            old_data = json.load(f)
                        old_ok, _old_reason = self._validate_snapshot(old_data)
                        if old_ok:
                            shutil.copyfile(ready_path, backup_path)
                        else:
                            logger.warning("[GLOBAL] [LEARN] SAVE — existing ready memory invalid; backup tidak ditimpa")
                    except (OSError, json.JSONDecodeError, TypeError, ValueError) as e:
                        logger.warning("[GLOBAL] [LEARN] SAVE — existing ready memory unreadable; backup tidak ditimpa: %s", e)
                ok, reason = self._write_ready_file_locked(ready_path, "MANUAL_SAVE")
                if not ok:
                    return {"ok": False, "reason": reason, "path": ready_path}
                size = os.path.getsize(ready_path)
                return {"ok": True, "reason": "validated+atomic", "path": ready_path, "bytes": size, "saved_at": time.time()}
            finally:
                self._learning_paused = False
                logger.info("[GLOBAL] [LEARN] SAVE — complete; brain resumed")

    def mirror_ready_to_git(self) -> bool:
        """Optional explicit GitHub mirror of the latest validated local memory."""
        if not self.git_enabled:
            logger.info("[GLOBAL] [LEARN] GIT — disabled (GIT_AUTOSAVE=false)")
            return False
        ready = os.path.join(os.path.dirname(self.checkpoint_path) or ".", "learn_memory_ready.json")
        if not os.path.exists(ready):
            logger.warning("[GLOBAL] [LEARN] GIT — no validated ready memory to mirror")
            return False
        try:
            self._git_commit_push(ready)
            return True
        except Exception:
            return False

    def open_ready_memory(self, ready_path: Optional[str] = None, backup_path: Optional[str] = None) -> Dict[str, Any]:
        """Replace current Learn memory from the validated local /save snapshot.
        GitHub is intentionally not consulted. The brain is paused during the swap.
        """
        ready_path = ready_path or os.path.join(os.path.dirname(self.checkpoint_path) or ".", "learn_memory_ready.json")
        backup_path = backup_path or (ready_path + ".backup")
        with self._lock:
            self._learning_paused = True
            logger.info("[GLOBAL] [LEARN] OPEN — brain paused")
            try:
                candidates = [(ready_path, "ready"), (backup_path, "backup")]
                last_error = "memory tidak ditemukan"
                for path, label in candidates:
                    if not os.path.exists(path):
                        continue
                    try:
                        with open(path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        ok, reason = self._validate_snapshot(data)
                        if not ok:
                            last_error = f"{label}: {reason}"
                            continue
                        self._restore_state(data)
                        self.last_audit_ts = 0.0
                        self.last_autosave_ts = time.time()
                        logger.info("[GLOBAL] [LEARN] OPEN — %s memory restored", label)
                        return {"ok": True, "label": label, "path": path, "reason": "integrity+schema valid", "saved_at": data.get("saved_at"), "strategy_version": self.current_strategy_version}
                    except (OSError, json.JSONDecodeError, ValueError, TypeError) as e:
                        last_error = f"{label}: {e}"
                return {"ok": False, "reason": last_error}
            finally:
                self._learning_paused = False
                logger.info("[GLOBAL] [LEARN] OPEN — brain resumed")

    # ------------------------------------------------------------------
    # checkpoint / persistence
    # ------------------------------------------------------------------
    def _export_state(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "schema_version": self._schema_version,
                "saved_at": time.time(),
                "trade_history": self.trade_history[-5000:],
                "scan_summaries": self.scan_summaries[-3000:],
                "candidate_history": self.candidate_history[-20000:],
                "shadow_history": self.shadow_history[-20000:],
                "raw_events": self.raw_events[-10000:],
                "feature_cache": self.feature_cache,
                "calibration_cache": self.calibration_cache,
                "frequency_cache": self.frequency_cache,
                "threshold_history": self.threshold_history[-1000:],
                "strategy_change_log": self.strategy_change_log[-1000:],
                "decision_history": self.decision_history[-1000:],
                "pending_challenger": self.pending_challenger,
                "last_audit_report": self.last_audit_report,
                "strategy_state": self.strategy_state,
                "current_strategy_version": self.current_strategy_version,
                "trades_since_last_change": self.trades_since_last_change,
                "last_change_ts": self.last_change_ts,
                "last_audit_ts": self.last_audit_ts,
            }

    def _restore_state(self, data: Dict[str, Any]) -> None:
        if not isinstance(data, dict):
            raise ValueError("checkpoint bukan object JSON")
        self._schema_version = _safe_int(data.get("schema_version"), SCHEMA_VERSION)
        self.trade_history = list(data.get("trade_history", []))
        self.scan_summaries = list(data.get("scan_summaries", []))
        self.candidate_history = list(data.get("candidate_history", []))
        self.shadow_history = list(data.get("shadow_history", []))
        self.raw_events = list(data.get("raw_events", []))
        self.feature_cache = dict(data.get("feature_cache", {}))
        self.calibration_cache = dict(data.get("calibration_cache", {}))
        self.frequency_cache = dict(data.get("frequency_cache", {}))
        self.threshold_history = list(data.get("threshold_history", []))
        self.strategy_change_log = list(data.get("strategy_change_log", []))
        self.decision_history = list(data.get("decision_history", []))
        self.pending_challenger = data.get("pending_challenger")
        self.last_audit_report = dict(data.get("last_audit_report", {}))
        self.strategy_state = dict(data.get("strategy_state", {}))
        self.current_strategy_version = data.get("current_strategy_version")
        self.trades_since_last_change = _safe_int(data.get("trades_since_last_change"), 0)
        self.last_change_ts = _safe_float(data.get("last_change_ts"), 0.0)
        self.last_audit_ts = _safe_float(data.get("last_audit_ts"), 0.0)

    def load(self) -> str:
        with self._lock:
            for path, label in ((self.checkpoint_path, "primary"), (self.backup_path, "backup")):
                if not os.path.exists(path):
                    continue
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    self._restore_state(data)
                    logger.info("learn.py: checkpoint %s dimuat", label)
                    return label
                except (OSError, ValueError, json.JSONDecodeError) as e:
                    logger.warning("learn.py: checkpoint %s gagal dibaca: %s", label, e)
            return "empty"

    def save_checkpoint(self) -> bool:
        try:
            with self._lock:
                data = self._export_state()
                if os.path.exists(self.checkpoint_path):
                    shutil.copyfile(self.checkpoint_path, self.backup_path)
                _atomic_write_json(self.checkpoint_path, data)
                self.last_autosave_ts = time.time()
            return True
        except (OSError, TypeError, ValueError) as e:
            logger.error("learn.py: gagal simpan checkpoint: %s", e)
            return False

    def autosave(self) -> None:
        try:
            with self._lock:
                if self._learning_paused:
                    logger.info("[GLOBAL] [LEARN] AUTOSAVE — deferred while paused")
                    return
                ok = self.save_checkpoint()
                # Also maintain a stable local memory image. /open never reads GitHub.
                ready = os.path.join(os.path.dirname(self.checkpoint_path) or ".", "learn_memory_ready.json")
                backup = ready + ".backup"
                if ok:
                    if os.path.exists(ready):
                        try:
                            with open(ready, "r", encoding="utf-8") as f:
                                old_data = json.load(f)
                            old_ok, _old_reason = self._validate_snapshot(old_data)
                            if old_ok:
                                shutil.copyfile(ready, backup)
                            else:
                                logger.warning("[GLOBAL] [LEARN] AUTOSAVE — existing ready memory invalid; backup tidak ditimpa")
                        except (OSError, json.JSONDecodeError, TypeError, ValueError) as e:
                            logger.warning("[GLOBAL] [LEARN] AUTOSAVE — existing ready memory unreadable; backup tidak ditimpa: %s", e)
                    ready_ok, ready_reason = self._write_ready_file_locked(ready, "AUTO_SAVE")
                    logger.info("[GLOBAL] [LEARN] AUTOSAVE — ready=%s reason=%s", ready_ok, ready_reason)
                if ok and self.git_enabled:
                    self._git_commit_push(ready)
        except Exception as e:  # pragma: no cover
            logger.error("learn.py: autosave non-fatal: %s", e)

    def _github_put_file(self, local_path: str, remote_path: str, commit_message: str) -> bool:
        """Publish one validated snapshot through GitHub Contents API.
        This works even when Render only has a tarball checkout and no .git dir.
        """
        if requests is None or not self.github_token or not self.github_repo:
            logger.info("[GLOBAL] [LEARN] GIT — API mirror unavailable (token/repo missing)")
            return False
        try:
            with open(local_path, "rb") as f:
                content = f.read()
            import base64
            encoded_path = "/".join(quote(part, safe="") for part in remote_path.strip("/").split("/"))
        except Exception as e:
            logger.warning("[GLOBAL] [LEARN] GIT — read snapshot gagal: %s", e)
            return False
        try:
            headers = {
                "Authorization": f"Bearer {self.github_token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            }
            base = f"https://api.github.com/repos/{self.github_repo}/contents/{encoded_path}"
            get_resp = requests.get(base, headers=headers, params={"ref": self.github_branch}, timeout=15)
            sha = None
            if get_resp.status_code == 200:
                payload = get_resp.json()
                sha = payload.get("sha")
            elif get_resp.status_code != 404:
                logger.warning("[GLOBAL] [LEARN] GIT — lookup gagal HTTP %s", get_resp.status_code)
                return False
            put_payload = {
                "message": commit_message,
                "content": base64.b64encode(content).decode("ascii"),
                "branch": self.github_branch,
            }
            if sha:
                put_payload["sha"] = sha
            put_resp = requests.put(base, headers=headers, json=put_payload, timeout=20)
            if put_resp.status_code not in (200, 201):
                logger.warning("[GLOBAL] [LEARN] GIT — upload gagal HTTP %s: %s", put_resp.status_code, put_resp.text[:300])
                return False
            return True
        except Exception as e:
            logger.warning("[GLOBAL] [LEARN] GIT — API mirror gagal: %s", e)
            return False

    def _git_commit_push(self, ready_path: Optional[str] = None) -> None:
        """Publish stable memory to GitHub via Contents API; never affects trading."""
        try:
            source = os.path.abspath(ready_path or self.checkpoint_path)
            if not os.path.exists(source):
                logger.warning("[GLOBAL] [LEARN] GIT — source snapshot tidak ada")
                return
            import hashlib
            digest = hashlib.sha256(open(source, "rb").read()).hexdigest()
            if digest == self._last_github_digest:
                logger.info("[GLOBAL] [LEARN] GIT — unchanged, push dilewati")
                return
            ok = self._github_put_file(
                source,
                self.github_memory_path,
                f"autosave learn {time.strftime('%Y-%m-%d %H:%M:%S')}",
            )
            if ok:
                self._last_github_digest = digest
                logger.info("[GLOBAL] [LEARN] GIT — memory mirror pushed: %s", self.github_memory_path)
        except Exception as e:
            logger.warning("learn.py: git autosave gagal: %s", e)

    # ------------------------------------------------------------------
    # event ingestion
    # ------------------------------------------------------------------
    def _append_event(self, kind: str, payload: Dict[str, Any]) -> None:
        event = dict(payload)
        event["kind"] = kind
        event.setdefault("timestamp", time.time())
        self.raw_events.append(event)
        if len(self.raw_events) > 15000:
            del self.raw_events[:-10000]

    def record_scan_summary(self, summary: Dict[str, Any]) -> None:
        with self._lock:
            row = dict(summary)
            row.setdefault("timestamp", time.time())
            self.scan_summaries.append(row)
            self._append_event("SCAN_SUMMARY", row)

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
                "entry": setup.get("entry"), "tp": setup.get("tp"), "sl": setup.get("sl"),
                "strategy_version": setup.get("strategy_version"),
                "threshold": _safe_float(threshold),
                "eligible": bool(eligible),
                "reason": reason,
                "timestamp": time.time(),
            }
            self.candidate_history.append(row)
            self._append_event("CANDIDATE", row)

    def record_shadow_outcome(self, candidate: Dict[str, Any], outcome: str, pnl_r: float = 0.0) -> None:
        with self._lock:
            row = dict(candidate)
            row.update({
                "kind": "SHADOW_OUTCOME",
                "outcome": outcome if outcome in OUTCOME_TYPES else "TIMEOUT",
                "pnl_r": _safe_float(pnl_r),
                "timestamp": time.time(),
            })
            row.setdefault("bucket", _bucket_of(row.get("confidence", 0)))
            self.shadow_history.append(row)
            self._append_event("SHADOW_OUTCOME", row)

    def record_trade_outcome(self, setup: Dict[str, Any], outcome: str, close_info: Dict[str, Any]) -> None:
        normalized = outcome if outcome in OUTCOME_TYPES else "BE"
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
                "strategy_version": setup.get("strategy_version"),
                "outcome": normalized,
                "pnl_pct": _safe_float(close_info.get("pnl_pct")),
                "pnl_r": _safe_float(close_info.get("pnl_r")),
                "trail_count": _safe_int(close_info.get("trail_count")),
                "entry_time": setup.get("timestamp"),
                "close_time": close_info.get("close_time", time.time() * 1000),
                "timestamp": time.time(),
            }
            self.trade_history.append(row)
            self.trades_since_last_change += 1
            self._append_event("TRADE_OUTCOME", row)
            self._update_feature_cache_locked()

    def set_strategy_state(self, state: Dict[str, Any]) -> None:
        with self._lock:
            self.strategy_state = dict(state or {})
            self.current_strategy_version = self.strategy_state.get("version", self.current_strategy_version)
            self._append_event("STRATEGY_STATE", self.strategy_state)

    # ------------------------------------------------------------------
    # feature / statistics
    # ------------------------------------------------------------------
    @staticmethod
    def _economic_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [r for r in rows if r.get("outcome") in ECONOMIC_OUTCOMES]

    def _weighted_stats(self, rows: Sequence[Dict[str, Any]], half_life_days: float = HALF_LIFE_DAYS) -> Dict[str, float]:
        economic = self._economic_rows(rows)
        if not economic:
            return {
                "n": 0, "effective_n": 0.0, "win_rate": 0.0, "expectancy": 0.0,
                "profit_factor": 0.0, "avg_win": 0.0, "avg_loss": 0.0,
                "max_losing_streak": 0,
            }
        now = time.time()
        weighted = []
        wins = []
        losses = []
        for r in economic:
            pnl = _safe_float(r.get("pnl_r"))
            w = _time_decay_weight(_safe_float(r.get("timestamp"), now), now, half_life_days)
            weighted.append((pnl, w, r))
            if pnl > 0:
                wins.append((pnl, w))
            elif pnl < 0:
                losses.append((pnl, w))
        total_w = sum(w for _, w, _ in weighted) or 1e-9
        decision_w = sum(w for pnl, w, _r in weighted if pnl != 0) or 1e-9
        expectancy = sum(pnl * w for pnl, w, _ in weighted) / total_w
        win_w = sum(w for pnl, w in wins)
        gross_win = sum(pnl * w for pnl, w in wins)
        gross_loss = abs(sum(pnl * w for pnl, w in losses))
        pf = gross_win / gross_loss if gross_loss > 1e-9 else (999.0 if gross_win > 0 else 0.0)
        avg_win = gross_win / (sum(w for _, w in wins) or 1e-9)
        avg_loss = -gross_loss / (sum(w for _, w in losses) or 1e-9)

        # Recent chronological losing streak. TIMEOUT/BE are neutral separators.
        streak = 0
        max_streak = 0
        for r in economic:
            pnl = _safe_float(r.get("pnl_r"))
            if pnl < 0:
                streak += 1
                max_streak = max(max_streak, streak)
            elif pnl > 0:
                streak = 0

        return {
            "n": len(economic),
            "effective_n": round(total_w, 2),
            "win_rate": round(win_w / decision_w * 100.0, 2),
            "expectancy": round(expectancy, 4),
            "profit_factor": round(min(pf, 999.0), 3),
            "avg_win": round(avg_win, 4),
            "avg_loss": round(avg_loss, 4),
            "max_losing_streak": max_streak,
        }

    def confidence_calibration(self) -> Dict[str, Dict[str, float]]:
        with self._lock:
            buckets: Dict[str, List[Dict[str, Any]]] = {}
            for t in self.trade_history:
                buckets.setdefault(_bucket_of(t.get("confidence", 0)), []).append(t)
            out = {b: self._weighted_stats(rows) for b, rows in buckets.items()}
            self.calibration_cache = out
            return out

    def regime_performance(self) -> Dict[str, Dict[str, float]]:
        with self._lock:
            groups: Dict[str, List[Dict[str, Any]]] = {}
            for t in self.trade_history:
                groups.setdefault(str(t.get("regime", "UNKNOWN")), []).append(t)
            return {k: self._weighted_stats(v) for k, v in groups.items()}

    def session_performance(self) -> Dict[str, Dict[str, float]]:
        with self._lock:
            groups: Dict[str, List[Dict[str, Any]]] = {}
            for t in self.trade_history:
                groups.setdefault(str(t.get("session", "UNKNOWN")), []).append(t)
            return {k: self._weighted_stats(v) for k, v in groups.items()}

    def setup_performance(self) -> Dict[str, Dict[str, float]]:
        with self._lock:
            groups: Dict[str, List[Dict[str, Any]]] = {}
            for t in self.trade_history:
                groups.setdefault(str(t.get("setup_type", "UNKNOWN")), []).append(t)
            return {k: self._weighted_stats(v) for k, v in groups.items()}

    def component_performance(self) -> Dict[str, Dict[str, float]]:
        """Compare high/low component scores against realized R. This is diagnostic, not causal proof."""
        with self._lock:
            values: Dict[str, List[Tuple[float, float]]] = {}
            for t in self.trade_history:
                pnl = _safe_float(t.get("pnl_r"))
                for key, value in dict(t.get("components", {})).items():
                    values.setdefault(str(key), []).append((_safe_float(value), pnl))
            out: Dict[str, Dict[str, float]] = {}
            for key, pairs in values.items():
                if not pairs:
                    continue
                xs = [x for x, _ in pairs]
                ys = [y for _, y in pairs]
                xm, ym = _mean(xs), _mean(ys)
                denom = math.sqrt(sum((x - xm) ** 2 for x in xs) * sum((y - ym) ** 2 for y in ys))
                corr = sum((x - xm) * (y - ym) for x, y in pairs) / denom if denom > 1e-12 else 0.0
                out[key] = {"n": len(pairs), "avg_score": round(_mean(xs), 4), "avg_pnl_r": round(ym, 4), "corr": round(corr, 4)}
            return out

    def _update_feature_cache_locked(self) -> None:
        self.feature_cache = {
            "confidence": self.confidence_calibration(),
            "regime": self.regime_performance(),
            "session": self.session_performance(),
            "setup": self.setup_performance(),
            "components": self.component_performance(),
            "updated_at": time.time(),
        }

    def overall_stats(self) -> Dict[str, Any]:
        with self._lock:
            stats = self._weighted_stats(self.trade_history)
            counts = {o: sum(1 for t in self.trade_history if t.get("outcome") == o) for o in OUTCOME_TYPES}
            economic = self._economic_rows(self.trade_history)
            stats.update({
                "outcome_counts": counts,
                "timeout_count": counts.get("TIMEOUT", 0),
                "confidence_avg_closed": round(_mean(_safe_float(t.get("confidence")) for t in economic), 2) if economic else 0.0,
                "last_trades": list(self.trade_history[-5:]),
                "regime": self.regime_performance(),
                "session": self.session_performance(),
                "setup": self.setup_performance(),
                "calibration": self.confidence_calibration(),
                "frequency": self.frequency_diagnosis(),
            })
            return stats

    # ------------------------------------------------------------------
    # frequency + shadow
    # ------------------------------------------------------------------
    def frequency_diagnosis(self, window_scans: int = 50) -> Dict[str, Any]:
        with self._lock:
            recent = self.scan_summaries[-max(1, window_scans):]
            if not recent:
                result = {"status": "NO_DATA", "reason": "belum ada scan"}
                self.frequency_cache = result
                return result

            avg_processed = _mean(_safe_float(s.get("processed")) for s in recent)
            avg_candidate = _mean(_safe_float(s.get("candidate")) for s in recent)
            avg_eligible = _mean(_safe_float(s.get("eligible")) for s in recent)
            avg_reject_geometry = _mean(_safe_float(dict(s.get("reject_reasons", {})).get("GEOMETRY_ORDER_INVALID", 0)) for s in recent)
            avg_no_candidate = _mean(_safe_float(dict(s.get("reject_reasons", {})).get("NO_VALID_ENTRY_CANDIDATE", 0)) for s in recent)
            avg_below_threshold = _mean(_safe_float(dict(s.get("reject_reasons", {})).get("BELOW_ACTIVE_THRESHOLD", 0)) for s in recent)

            if avg_processed < 5:
                result = {"status": "DATA_PIPELINE_WARNING", "note": "terlalu sedikit coin yang benar-benar diproses", "avg_processed": round(avg_processed, 2)}
            elif avg_candidate < 1 and avg_no_candidate > 0.5 * max(avg_processed, 1.0):
                result = {"status": "STRATEGY_TOO_RESTRICTIVE", "note": "banyak coin selesai diproses tetapi candidate hampir selalu nol", "avg_processed": round(avg_processed, 2), "avg_candidate": round(avg_candidate, 2), "avg_no_candidate": round(avg_no_candidate, 2)}
            elif avg_candidate >= 1 and avg_eligible < 0.5 and avg_below_threshold > 0:
                result = {"status": "THRESHOLD_TOO_HIGH_OR_STRICT", "note": "candidate ada tetapi eligible terlalu sedikit", "avg_candidate": round(avg_candidate, 2), "avg_eligible": round(avg_eligible, 2), "avg_below_threshold": round(avg_below_threshold, 2)}
            elif avg_reject_geometry > 0.25 * max(avg_candidate + avg_reject_geometry, 1.0):
                result = {"status": "GEOMETRY_WARNING", "note": "terlalu banyak candidate gugur di geometry", "avg_geometry_reject": round(avg_reject_geometry, 2)}
            else:
                result = {"status": "NORMAL", "avg_processed": round(avg_processed, 2), "avg_candidate": round(avg_candidate, 2), "avg_eligible": round(avg_eligible, 2)}
            self.frequency_cache = result
            return result

    def shadow_performance_below(self, threshold: float) -> Dict[str, float]:
        with self._lock:
            rows = [
                r for r in self.shadow_history
                if r.get("kind") == "SHADOW_OUTCOME"
                and _safe_float(r.get("confidence")) < _safe_float(threshold)
                and r.get("outcome") in ECONOMIC_OUTCOMES
            ]
            if not rows:
                return {"n": 0, "effective_n": 0.0, "win_rate": 0.0, "expectancy": 0.0}
            st = self._weighted_stats(rows, SHADOW_HALF_LIFE_DAYS)
            return {k: st.get(k, 0.0) for k in ("n", "effective_n", "win_rate", "expectancy")}

    # ------------------------------------------------------------------
    # challenger / replay / holdout
    # ------------------------------------------------------------------
    def register_challenger(self, proposed_params: Dict[str, Any], reason: str, evidence: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            challenger = {
                "id": f"CH-{int(time.time() * 1000)}",
                "created_at": time.time(),
                "status": "PENDING",
                "proposed_params": dict(proposed_params),
                "reason": reason,
                "evidence": dict(evidence),
            }
            self.pending_challenger = challenger
            self.decision_history.append({"type": "CHALLENGER_CREATED", **challenger})
            return dict(challenger)

    def _chronological_split(self, rows: Sequence[Dict[str, Any]], holdout_fraction: float = 0.25) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        ordered = sorted(rows, key=lambda r: _safe_float(r.get("timestamp")))
        if len(ordered) < 2:
            return list(ordered), []
        cut = max(1, int(len(ordered) * (1.0 - holdout_fraction)))
        if len(ordered) - cut < MIN_HOLDOUT_SAMPLE:
            cut = max(1, len(ordered) - MIN_HOLDOUT_SAMPLE)
        return ordered[:cut], ordered[cut:]

    def counterfactual_threshold(self, rows: Sequence[Dict[str, Any]], threshold: float) -> Dict[str, Any]:
        """Approximate replay using actual realized outcomes: exclude historical candidates below proposed threshold."""
        selected = [r for r in rows if _safe_float(r.get("confidence")) >= threshold and r.get("outcome") in ECONOMIC_OUTCOMES]
        baseline = self._weighted_stats(rows)
        challenger = self._weighted_stats(selected)
        return {
            "baseline": baseline,
            "challenger": challenger,
            "selected_n": challenger["n"],
            "delta_expectancy": round(challenger["expectancy"] - baseline["expectancy"], 4),
        }

    def validate_candidate_parameter_change(self, current_params: Dict[str, Any], proposed_params: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
        """Validate that only known parameters move and no move is excessively large."""
        for key, new_value in proposed_params.items():
            if key == "ACTIVE_THRESHOLD":
                old = _safe_float(current_params.get(key, 0.0))
                new = _safe_float(new_value)
                if not (0.0 <= new <= 95.0):
                    return False, "threshold di luar 0..95", {}
                if abs(new - old) > MAX_THRESHOLD_STEP:
                    return False, "perubahan threshold terlalu besar", {"old": old, "new": new}
            elif key in MAX_PARAM_STEP:
                if key not in current_params:
                    return False, f"parameter tidak dikenal: {key}", {}
                old = _safe_float(current_params.get(key))
                new = _safe_float(new_value)
                if abs(new - old) > MAX_PARAM_STEP[key]:
                    return False, f"perubahan {key} terlalu besar", {"old": old, "new": new}
            else:
                return False, f"parameter tidak diizinkan: {key}", {}
        return True, "parameter change shape valid", {"changed": list(proposed_params.keys())}

    def _holdout_gate(self, baseline: Sequence[Dict[str, Any]], challenger: Sequence[Dict[str, Any]]) -> Tuple[bool, str, Dict[str, Any]]:
        """Use the later chronological slice as holdout. Challenger is represented by counterfactual filtering when possible."""
        if len(challenger) < MIN_HOLDOUT_SAMPLE:
            return False, f"holdout terlalu kecil ({len(challenger)}/{MIN_HOLDOUT_SAMPLE})", {}
        b = self._weighted_stats(baseline)
        c = self._weighted_stats(challenger)
        delta = c["expectancy"] - b["expectancy"]
        # A challenger can be accepted with small neutral degradation only if PF and WR do not collapse.
        ok = delta >= -0.10 and c["profit_factor"] >= max(0.9, b["profit_factor"] * 0.90)
        return ok, (
            f"holdout delta expectancy={delta:.4f}, PF {b['profit_factor']:.3f}->{c['profit_factor']:.3f}"
        ), {"baseline": b, "challenger": c, "delta_expectancy": round(delta, 4)}

    # ------------------------------------------------------------------
    # Ollama critic
    # ------------------------------------------------------------------
    def _ollama_critique(self, context: Dict[str, Any]) -> Optional[str]:
        if not self.ollama_url or requests is None:
            return None
        try:
            prompt = (
                "Anda adalah kritikus statistik trading. Jangan membuat keputusan. "
                "Nilai apakah bukti berikut memiliki blind spot. Maksimal 5 bullet singkat.\n"
                + json.dumps(context, ensure_ascii=False, default=str)[:7000]
            )
            headers = {"Content-Type": "application/json"}
            if self.ollama_api_key:
                headers["Authorization"] = f"Bearer {self.ollama_api_key}"
            resp = requests.post(
                f"{self.ollama_url.rstrip('/')}/api/generate",
                headers=headers,
                json={"model": os.environ.get("OLLAMA_MODEL", "llama3"), "prompt": prompt, "stream": False},
                timeout=10,
            )
            if resp.status_code == 200:
                return str(resp.json().get("response", "")).strip()[:2000]
        except Exception as e:  # optional dependency/service
            logger.debug("Ollama critic unavailable: %s", e)
        return None

    # ------------------------------------------------------------------
    # recommendation
    # ------------------------------------------------------------------
    def _recommend_threshold(self, calibration: Dict[str, Dict[str, float]], current: float, freq: Dict[str, Any]) -> Optional[Tuple[float, Dict[str, Any]]]:
        usable = [(bucket, st) for bucket, st in calibration.items() if _safe_int(st.get("n")) >= MIN_SAMPLE_FOR_DECISION]
        usable.sort(key=lambda x: _safe_int(x[0].split("-")[0]))
        if len(usable) < 2:
            return None

        current = _safe_float(current)
        # Raise: require a genuinely negative lower band and positive higher band.
        bad_low = [(b, s) for b, s in usable if _safe_int(b.split("-")[0]) <= current + 10 and s.get("expectancy", 0) < -0.05]
        good_high = [(b, s) for b, s in usable if _safe_int(b.split("-")[0]) > current and s.get("expectancy", 0) > 0.05]
        if bad_low and good_high:
            target = float(_safe_int(good_high[0][0].split("-")[0]))
            new = round(max(0.0, min(95.0, current + min(MAX_THRESHOLD_STEP, max(1.0, target - current)))), 1)
            if abs(new - current) >= 0.5:
                return new, {
                    "type": "RAISE_THRESHOLD",
                    "bad_low": {b: s for b, s in bad_low},
                    "good_high": {b: s for b, s in good_high},
                    "frequency": freq,
                }

        # Lower only when the threshold is materially suppressing frequency and shadow evidence is positive.
        if current > 0 and freq.get("status") == "THRESHOLD_TOO_HIGH_OR_STRICT":
            shadow = self.shadow_performance_below(current)
            if _safe_int(shadow.get("n")) >= MIN_SAMPLE_FOR_DECISION and _safe_float(shadow.get("expectancy")) > 0.05:
                new = round(max(0.0, current - min(3.0, MAX_THRESHOLD_STEP)), 1)
                if new < current:
                    return new, {"type": "LOWER_THRESHOLD_FROM_SHADOW", "shadow": shadow, "frequency": freq}
        return None

    def _should_auto_change(self) -> Tuple[bool, str]:
        now = time.time()
        if self.trades_since_last_change < MIN_TRADES_SINCE_LAST_CHANGE:
            return False, f"cooldown sample {self.trades_since_last_change}/{MIN_TRADES_SINCE_LAST_CHANGE}"
        if now - self.last_change_ts < AUDIT_COOLDOWN_SECONDS:
            return False, "cooldown waktu masih aktif"
        return True, "gate sample + time cooldown lolos"

    # ------------------------------------------------------------------
    # rollback monitor
    # ------------------------------------------------------------------
    def evaluate_current_version_degradation(self) -> Dict[str, Any]:
        with self._lock:
            version = self.current_strategy_version
            version_rows = [r for r in self.trade_history if r.get("strategy_version") == version]
            if len(version_rows) < MIN_TOTAL_SAMPLE_FOR_AUDIT:
                return {"status": "INSUFFICIENT", "n": len(version_rows), "version": version}
            recent = version_rows[-MIN_TOTAL_SAMPLE_FOR_AUDIT:]
            prior = version_rows[:-MIN_TOTAL_SAMPLE_FOR_AUDIT]
            if len(prior) < MIN_SAMPLE_FOR_DECISION:
                return {"status": "INSUFFICIENT_BASELINE", "n": len(prior), "version": version}
            recent_st = self._weighted_stats(recent)
            prior_st = self._weighted_stats(prior)
            degraded = recent_st["expectancy"] < prior_st["expectancy"] - ROLLBACK_DEGRADATION_R
            return {
                "status": "DEGRADED" if degraded else "STABLE",
                "version": version,
                "recent": recent_st,
                "prior": prior_st,
                "delta_expectancy": round(recent_st["expectancy"] - prior_st["expectancy"], 4),
            }

    # ------------------------------------------------------------------
    # main audit
    # ------------------------------------------------------------------
    def audit(self, strategy_engine: Any) -> Dict[str, Any]:
        with self._lock:
            now = time.time()
            freq = self.frequency_diagnosis()
            calibration = self.confidence_calibration()
            total = len(self.trade_history)
            report: Dict[str, Any] = {
                "timestamp": now,
                "total_trades": total,
                "action": "NO_ACTION",
                "frequency": freq,
                "calibration": calibration,
                "strategy_version": getattr(strategy_engine, "version", None),
            }

            # Frequency diagnosis always runs, even before minimum trade sample.
            if freq.get("status") == "DATA_PIPELINE_WARNING":
                report["reason"] = "data pipeline warning; no automatic strategy update"
                self.last_audit_report = report
                self.last_audit_ts = now
                return report

            # Auto rollback signal is advisory unless strategy supports rollback explicitly.
            degradation = self.evaluate_current_version_degradation()
            report["version_health"] = degradation
            if degradation.get("status") == "DEGRADED":
                report["action"] = "ROLLBACK_RECOMMENDED"
                report["reason"] = "versi strategy terbaru mengalami degradation material"
                self.decision_history.append({"type": "ROLLBACK_RECOMMENDED", "report": report})
                self.last_audit_report = report
                self.last_audit_ts = now
                return report

            if total < MIN_TOTAL_SAMPLE_FOR_AUDIT:
                report["reason"] = f"sample belum cukup ({total}/{MIN_TOTAL_SAMPLE_FOR_AUDIT})"
                self.last_audit_report = report
                self.last_audit_ts = now
                return report

            ok, gate_reason = self._should_auto_change()
            if not ok:
                report["reason"] = gate_reason
                self.last_audit_report = report
                self.last_audit_ts = now
                return report

            current_params = dict(getattr(strategy_engine, "params", {}))
            current_threshold = _safe_float(getattr(strategy_engine, "get_active_threshold", lambda: current_params.get("ACTIVE_THRESHOLD", 0.0))())
            recommendation = self._recommend_threshold(calibration, current_threshold, freq)
            if not recommendation:
                report["reason"] = "tidak ada bukti cukup untuk perubahan parameter"
                self.last_audit_report = report
                self.last_audit_ts = now
                return report

            new_threshold, evidence = recommendation
            proposed = {"ACTIVE_THRESHOLD": new_threshold}
            shape_ok, shape_reason, shape_meta = self.validate_candidate_parameter_change(current_params, proposed)
            evidence["shape_validation"] = shape_meta
            if not shape_ok:
                report["action"] = "REJECTED"
                report["reason"] = shape_reason
                self.last_audit_report = report
                self.last_audit_ts = now
                return report

            # Counterfactual training/holdout gate.
            training, holdout = self._chronological_split(self.trade_history, 0.25)
            train_eval = self.counterfactual_threshold(training, new_threshold)
            holdout_selected = [r for r in holdout if _safe_float(r.get("confidence")) >= new_threshold and r.get("outcome") in ECONOMIC_OUTCOMES]
            baseline_holdout = [r for r in holdout if r.get("outcome") in ECONOMIC_OUTCOMES]
            hold_ok, hold_reason, hold_meta = self._holdout_gate(baseline_holdout, holdout_selected)
            evidence["counterfactual_train"] = train_eval
            evidence["holdout"] = hold_meta
            evidence["holdout_reason"] = hold_reason
            evidence["ollama_critique"] = self._ollama_critique({
                "current_threshold": current_threshold,
                "proposed_threshold": new_threshold,
                "evidence": evidence,
            })

            if not hold_ok:
                report["action"] = "DEFERRED"
                report["reason"] = hold_reason
                self.decision_history.append({"type": "DEFERRED", "timestamp": now, "proposal": proposed, "evidence": evidence})
                self.last_audit_report = report
                self.last_audit_ts = now
                return report

            # Create challenger first; strategy is untouched until all gates pass.
            self.register_challenger(proposed, recommendation[1].get("type", "THRESHOLD_CALIBRATION"), evidence)

            try:
                change_record = strategy_engine.apply_update(
                    proposed,
                    reason=f"Learn validated threshold change: {evidence.get('type', 'CALIBRATION')}",
                    evidence=evidence,
                )
            except Exception as e:
                report["action"] = "REJECTED"
                report["reason"] = f"strategy.apply_update gagal: {e}"
                self.decision_history.append({"type": "APPLY_FAILED", "timestamp": now, "proposal": proposed, "error": str(e)})
                self.last_audit_report = report
                self.last_audit_ts = now
                return report

            self.threshold_history.append({
                "timestamp": now,
                "old_threshold": current_threshold,
                "new_threshold": new_threshold,
                "evidence": evidence,
            })
            self.strategy_change_log.append(change_record)
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
            report.update({
                "action": "APPLIED",
                "reason": "statistical + counterfactual + holdout gate passed",
                "old_threshold": current_threshold,
                "new_threshold": new_threshold,
                "evidence": evidence,
                "strategy_version": change_record.get("version"),
            })
            self.last_audit_report = report
            self.last_audit_ts = now
            return report

    # ------------------------------------------------------------------
    # utility APIs for future main.py integration
    # ------------------------------------------------------------------
    def get_last_audit_report(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self.last_audit_report)

    def should_run_audit(self, interval_seconds: int = 300) -> bool:
        with self._lock:
            return time.time() - self.last_audit_ts >= max(1, interval_seconds)

    def export_memory_summary(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "schema_version": self._schema_version,
                "trades": len(self.trade_history),
                "candidates": len(self.candidate_history),
                "shadow": len(self.shadow_history),
                "scan_summaries": len(self.scan_summaries),
                "strategy_version": self.current_strategy_version,
                "frequency": dict(self.frequency_cache),
                "pending_challenger": dict(self.pending_challenger) if isinstance(self.pending_challenger, dict) else None,
                "last_audit": dict(self.last_audit_report),
                "learning_paused": self._learning_paused,
                "ready_memory_path": os.path.join(os.path.dirname(self.checkpoint_path) or ".", "learn_memory_ready.json"),
                "ready_memory_exists": os.path.exists(os.path.join(os.path.dirname(self.checkpoint_path) or ".", "learn_memory_ready.json")),
            }
