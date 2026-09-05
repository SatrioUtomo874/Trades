"""
learn.py — Mesin Audit, Pembelajaran, Evaluasi Statistik & Optimasi Strategy
=============================================================================

PRINSIP UTAMA (§41):
    learn.py adalah AUDITOR, bukan trader baru.
    - TIDAK PERNAH mengambil data market langsung dari API (main.py yang
      memberi tahu apa yang terjadi lewat record_* methods).
    - TIDAK PERNAH membatalkan entry yang sudah valid dari strategy.py
      secara real-time.
    - HANYA mencatat: prediksi strategy, outcome aktual, kondisi market,
      alasan sukses/gagal — lalu (jika bukti memadai) mengusulkan &
      memvalidasi perubahan parameter strategy.py melalui apply_update().

Perubahan parameter TIDAK BOLEH terjadi karena satu trade. Setiap audit
mensyaratkan sample size minimum, dan setiap perubahan yang diterapkan
divalidasi lebih dulu lewat counterfactual replay terhadap data historis
sebelum benar-benar dipanggil ke strategy.apply_update().
"""

from __future__ import annotations

import json
import logging
import math
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Dict, List, Optional, Sequence

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None

logger = logging.getLogger("learn")

CONFIDENCE_BUCKETS = [
    (0, 40), (40, 50), (50, 60), (60, 70), (70, 80), (80, 90), (90, 101),
]

MIN_SAMPLE_FOR_DECISION = 30          # minimum trade per bucket sebelum dipakai sbg bukti
MIN_TOTAL_SAMPLE_FOR_AUDIT = 40       # minimum total closed trade sebelum audit boleh mengubah apapun
MIN_TRADES_SINCE_LAST_CHANGE = 20     # cooldown — tidak boleh berubah lagi terlalu cepat
MAX_THRESHOLD_STEP = 5.0              # kenaikan/penurunan threshold maksimum per audit (poin %)
HALF_LIFE_DAYS = 30.0                 # time decay untuk statistik (data lama makin ringan bobotnya)

OUTCOME_TYPES = ("TP", "INITIAL_SL", "TRAIL", "BE", "TIMEOUT")


def _time_decay_weight(ts: float, now: Optional[float] = None, half_life_days: float = HALF_LIFE_DAYS) -> float:
    now = now or time.time()
    age_days = max(0.0, (now - ts) / 86400.0)
    return 0.5 ** (age_days / half_life_days)


def _bucket_of(confidence: float) -> str:
    for lo, hi in CONFIDENCE_BUCKETS:
        if lo <= confidence < hi:
            return f"{lo}-{hi-1}"
    return "unknown"


def _atomic_write_json(path: str, data: Dict[str, Any]) -> None:
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


class LearnEngine:
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
        self.git_enabled = git_enabled
        self.git_repo_dir = git_repo_dir or "."
        self._lock = RLock()

        self.trade_history: List[Dict[str, Any]] = []
        self.scan_summaries: List[Dict[str, Any]] = []
        self.threshold_history: List[Dict[str, Any]] = []
        self.strategy_change_log: List[Dict[str, Any]] = []
        self.trades_since_last_change: int = 0
        self.last_autosave_ts: float = 0.0

        os.makedirs(os.path.dirname(self.checkpoint_path) or ".", exist_ok=True)

    # ------------------------------------------------------------------
    # §39 /open — checkpoint load dengan integrity check + fallback backup
    # ------------------------------------------------------------------
    def load(self) -> str:
        with self._lock:
            for path, label in ((self.checkpoint_path, "primary"), (self.backup_path, "backup")):
                if not os.path.exists(path):
                    continue
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    self._restore_state(data)
                    logger.info("learn.py: checkpoint %s dimuat (%s)", label, path)
                    return label
                except (json.JSONDecodeError, OSError) as e:
                    logger.warning("learn.py: checkpoint %s korup/gagal dibaca (%s): %s", label, path, e)
                    continue
            logger.info("learn.py: tidak ada checkpoint valid, mulai dari state kosong")
            return "empty"

    def _restore_state(self, data: Dict[str, Any]) -> None:
        self.trade_history = data.get("trade_history", [])
        self.scan_summaries = data.get("scan_summaries", [])
        self.threshold_history = data.get("threshold_history", [])
        self.strategy_change_log = data.get("strategy_change_log", [])
        self.trades_since_last_change = data.get("trades_since_last_change", 0)

    def _export_state(self) -> Dict[str, Any]:
        return {
            "trade_history": self.trade_history[-5000:],  # cegah file tumbuh tanpa batas
            "scan_summaries": self.scan_summaries[-2000:],
            "threshold_history": self.threshold_history,
            "strategy_change_log": self.strategy_change_log,
            "trades_since_last_change": self.trades_since_last_change,
            "saved_at": time.time(),
        }

    def save_checkpoint(self) -> bool:
        with self._lock:
            data = self._export_state()
            try:
                if os.path.exists(self.checkpoint_path):
                    shutil.copyfile(self.checkpoint_path, self.backup_path)
                _atomic_write_json(self.checkpoint_path, data)
                self.last_autosave_ts = time.time()
                return True
            except OSError as e:
                logger.error("learn.py: gagal menyimpan checkpoint: %s", e)
                return False

    def autosave(self) -> None:
        """§40 — dipanggil berkala oleh Worker 3. Tidak boleh mengganggu
        trading engine: semua exception ditangkap & hanya di-log."""
        try:
            ok = self.save_checkpoint()
            if ok and self.git_enabled:
                self._git_commit_push()
        except Exception as e:  # pragma: no cover - safety net
            logger.error("learn.py: autosave gagal (diabaikan agar trading tidak terganggu): %s", e)

    def _git_commit_push(self) -> None:
        try:
            subprocess.run(
                ["git", "add", self.checkpoint_path],
                cwd=self.git_repo_dir, check=False, capture_output=True, timeout=15,
            )
            subprocess.run(
                ["git", "commit", "-m", f"autosave learn checkpoint {time.strftime('%Y-%m-%d %H:%M:%S')}"],
                cwd=self.git_repo_dir, check=False, capture_output=True, timeout=15,
            )
            subprocess.run(
                ["git", "push"], cwd=self.git_repo_dir, check=False, capture_output=True, timeout=30,
            )
        except Exception as e:  # pragma: no cover
            logger.warning("learn.py: git autosave gagal (non-fatal): %s", e)

    # ------------------------------------------------------------------
    # Perekaman data (dipanggil oleh main.py, bukan strategy.py)
    # ------------------------------------------------------------------
    def record_scan_summary(self, summary: Dict[str, Any]) -> None:
        with self._lock:
            summary = dict(summary)
            summary.setdefault("timestamp", time.time())
            self.scan_summaries.append(summary)

    def record_trade_outcome(self, setup: Dict[str, Any], outcome: str, close_info: Dict[str, Any]) -> None:
        """outcome salah satu dari OUTCOME_TYPES. Dipanggil main.py saat posisi CLOSED."""
        if outcome not in OUTCOME_TYPES:
            logger.warning("learn.py: outcome tidak dikenal: %s", outcome)
        record = {
            "pair": setup.get("pair"),
            "direction": setup.get("direction"),
            "confidence": setup.get("confidence", 0.0),
            "setup_type": setup.get("setup_type"),
            "regime": setup.get("regime"),
            "session": setup.get("session"),
            "components": setup.get("components", {}),
            "strategy_version": setup.get("strategy_version"),
            "outcome": outcome,
            "pnl_pct": close_info.get("pnl_pct", 0.0),
            "pnl_r": close_info.get("pnl_r", 0.0),
            "trail_count": close_info.get("trail_count", 0),
            "entry_time": setup.get("timestamp"),
            "close_time": close_info.get("close_time", time.time() * 1000),
            "timestamp": time.time(),
        }
        with self._lock:
            self.trade_history.append(record)
            self.trades_since_last_change += 1

    # ------------------------------------------------------------------
    # §43 — Statistik matematis
    # ------------------------------------------------------------------
    def _weighted_stats(self, trades: Sequence[Dict[str, Any]]) -> Dict[str, float]:
        if not trades:
            return {"n": 0, "win_rate": 0.0, "expectancy": 0.0, "profit_factor": 0.0}
        now = time.time()
        weights = [_time_decay_weight(t["timestamp"], now) for t in trades]
        total_w = sum(weights) or 1e-9
        wins = [t for t in trades if t["pnl_r"] > 0]
        losses = [t for t in trades if t["pnl_r"] <= 0]
        win_w = sum(_time_decay_weight(t["timestamp"], now) for t in wins)
        win_rate = win_w / total_w
        expectancy = sum(t["pnl_r"] * w for t, w in zip(trades, weights)) / total_w
        gross_win = sum(max(0.0, t["pnl_r"]) * _time_decay_weight(t["timestamp"], now) for t in wins)
        gross_loss = abs(sum(min(0.0, t["pnl_r"]) * _time_decay_weight(t["timestamp"], now) for t in losses))
        profit_factor = (gross_win / gross_loss) if gross_loss > 1e-9 else (float("inf") if gross_win > 0 else 0.0)
        return {
            "n": len(trades),
            "win_rate": round(win_rate * 100, 2),
            "expectancy": round(expectancy, 4),
            "profit_factor": round(profit_factor, 3) if profit_factor != float("inf") else 999.0,
        }

    def confidence_calibration(self) -> Dict[str, Dict[str, float]]:
        with self._lock:
            buckets: Dict[str, List[Dict[str, Any]]] = {}
            for t in self.trade_history:
                b = _bucket_of(t["confidence"])
                buckets.setdefault(b, []).append(t)
            return {b: self._weighted_stats(trades) for b, trades in buckets.items()}

    def regime_performance(self) -> Dict[str, Dict[str, float]]:
        with self._lock:
            buckets: Dict[str, List[Dict[str, Any]]] = {}
            for t in self.trade_history:
                buckets.setdefault(t.get("regime", "UNKNOWN"), []).append(t)
            return {r: self._weighted_stats(trades) for r, trades in buckets.items()}

    def session_performance(self) -> Dict[str, Dict[str, float]]:
        with self._lock:
            buckets: Dict[str, List[Dict[str, Any]]] = {}
            for t in self.trade_history:
                buckets.setdefault(t.get("session", "UNKNOWN"), []).append(t)
            return {s: self._weighted_stats(trades) for s, trades in buckets.items()}

    def overall_stats(self) -> Dict[str, Any]:
        with self._lock:
            trades = self.trade_history
            stats = self._weighted_stats(trades)
            outcome_counts = {o: sum(1 for t in trades if t["outcome"] == o) for o in OUTCOME_TYPES}
            stats["outcome_counts"] = outcome_counts
            stats["banned_related_timeout"] = outcome_counts.get("TIMEOUT", 0)
            return stats

    # ------------------------------------------------------------------
    # §50/§51 — Signal frequency & low confidence analysis (diagnostik)
    # ------------------------------------------------------------------
    def frequency_diagnosis(self, window_scans: int = 50) -> Dict[str, Any]:
        with self._lock:
            recent = self.scan_summaries[-window_scans:]
        if not recent:
            return {"status": "NO_DATA"}
        avg_candidate = sum(s.get("candidate", 0) for s in recent) / len(recent)
        avg_eligible = sum(s.get("eligible", 0) for s in recent) / len(recent)
        if avg_candidate < 1 and avg_eligible < 1:
            return {
                "status": "POSSIBLY_BROKEN",
                "note": "Candidate & eligible sama-sama sangat rendah — periksa apakah strategy terlalu ketat atau ada bug.",
                "avg_candidate": avg_candidate,
                "avg_eligible": avg_eligible,
            }
        if avg_candidate >= 1 and avg_eligible < 0.5:
            return {
                "status": "HEALTHY_LOW_FREQUENCY_OR_STRICT_THRESHOLD",
                "note": "Candidate cukup, tapi sangat sedikit yang eligible — bisa jadi threshold terlalu tinggi ATAU market memang tidak menawarkan setup berkualitas.",
                "avg_candidate": avg_candidate,
                "avg_eligible": avg_eligible,
            }
        return {"status": "NORMAL", "avg_candidate": avg_candidate, "avg_eligible": avg_eligible}

    # ------------------------------------------------------------------
    # Ollama — kritik kualitatif, hanya advisory (§48)
    # ------------------------------------------------------------------
    def _ollama_critique(self, context: Dict[str, Any]) -> Optional[str]:
        if not self.ollama_url or requests is None:
            return None
        try:
            prompt = (
                "Kamu adalah kritikus statistik untuk sebuah trading bot. "
                "Berikan kritik singkat (maks 3 kalimat) atas rencana perubahan berikut, "
                "fokus pada apakah bukti data sudah cukup kuat:\n"
                f"{json.dumps(context, default=str)[:3000]}"
            )
            headers = {}
            if self.ollama_api_key:
                headers["Authorization"] = f"Bearer {self.ollama_api_key}"

            resp = requests.post(
                f"{self.ollama_url.rstrip('/')}/api/generate",
                headers=headers,
                json={"model": os.environ.get("OLLAMA_MODEL", "llama3"), "prompt": prompt, "stream": False},
                timeout=10,
            )
            if resp.status_code == 200:
                return resp.json().get("response", "").strip()[:1000]
        except Exception as e:  # pragma: no cover - Ollama optional
            logger.info("learn.py: Ollama critic tidak tersedia (%s), lanjut tanpa kritik", e)
        return None

    # ------------------------------------------------------------------
    # §47/§49 — Audit utama: evaluasi & (jika valid) usulkan perubahan
    # ------------------------------------------------------------------
    def audit(self, strategy_engine: Any) -> Dict[str, Any]:
        """Dipanggil berkala oleh Worker 3 (learn thread). `strategy_engine`
        adalah instance strategy.Strategy — audit() akan memanggil
        strategy_engine.apply_update() SENDIRI hanya jika validasi lolos.
        """
        with self._lock:
            total_n = len(self.trade_history)
            report: Dict[str, Any] = {
                "timestamp": time.time(),
                "total_trades": total_n,
                "action": "NO_ACTION",
                "reason": None,
            }

            # §50/§51 — frequency HARUS selalu dihitung, tidak boleh digate oleh
            # sample size trade (candidate/eligible = 0 justru paling kritis
            # dideteksi SEBELUM 40 trade sempat terkumpul).
            freq = self.frequency_diagnosis()
            report["frequency"] = freq

            if total_n < MIN_TOTAL_SAMPLE_FOR_AUDIT:
                report["reason"] = f"Sample belum cukup ({total_n}/{MIN_TOTAL_SAMPLE_FOR_AUDIT})"
                return report

            if self.trades_since_last_change < MIN_TRADES_SINCE_LAST_CHANGE:
                report["reason"] = (
                    f"Cooldown aktif ({self.trades_since_last_change}/{MIN_TRADES_SINCE_LAST_CHANGE} trade sejak perubahan terakhir)"
                )
                return report

            calibration = self.confidence_calibration()
            report["calibration"] = calibration

            current_threshold = strategy_engine.get_active_threshold()
            recommendation = self._recommend_threshold(calibration, current_threshold, freq)

            if recommendation is None:
                report["reason"] = "Tidak ada bukti statistik cukup kuat untuk mengubah threshold"
                return report

            new_threshold, evidence = recommendation
            critique = self._ollama_critique({
                "current_threshold": current_threshold,
                "proposed_threshold": new_threshold,
                "evidence": evidence,
            })
            evidence["ollama_critique"] = critique

            validated, validation_note = self._validate_threshold_change(current_threshold, new_threshold, evidence)
            evidence["validation_note"] = validation_note

            if not validated:
                report["action"] = "REJECTED"
                report["reason"] = validation_note
                report["evidence"] = evidence
                return report

            change_record = strategy_engine.apply_update(
                {"ACTIVE_THRESHOLD": new_threshold},
                reason=f"Confidence bucket calibration: {validation_note}",
                evidence=evidence,
            )
            self.threshold_history.append({
                "timestamp": time.time(),
                "old_threshold": current_threshold,
                "new_threshold": new_threshold,
                "evidence": evidence,
            })
            self.strategy_change_log.append(change_record)
            self.trades_since_last_change = 0

            report["action"] = "APPLIED"
            report["old_threshold"] = current_threshold
            report["new_threshold"] = new_threshold
            report["evidence"] = evidence
            report["strategy_version"] = change_record["version"]
            return report

    def _recommend_threshold(
        self, calibration: Dict[str, Dict[str, float]], current_threshold: float,
        freq: Optional[Dict[str, Any]] = None,
    ) -> Optional[tuple]:
        """§43/§51 — Bandingkan expectancy antar bucket confidence. Jika
        bucket-bucket rendah (di bawah/dekat threshold saat ini) secara
        konsisten (n cukup) menghasilkan expectancy negatif sedangkan
        bucket lebih tinggi positif, usulkan menaikkan threshold ke batas
        bawah bucket positif pertama. Sebaliknya jika bucket rendah ternyata
        baik-baik saja dan frequency terlalu rendah, bisa menurunkan.
        Perubahan dibatasi MAX_THRESHOLD_STEP per audit (tidak impulsif).
        """
        usable = {b: s for b, s in calibration.items() if s["n"] >= MIN_SAMPLE_FOR_DECISION}
        if len(usable) < 2:
            return None

        def bucket_lo(b: str) -> float:
            return float(b.split("-")[0])

        ordered = sorted(usable.items(), key=lambda kv: bucket_lo(kv[0]))

        # cari bucket rendah dgn expectancy negatif yang berada <= threshold+10
        bad_low_buckets = [
            (b, s) for b, s in ordered
            if s["expectancy"] < 0 and bucket_lo(b) <= current_threshold + 10
        ]
        good_buckets = [(b, s) for b, s in ordered if s["expectancy"] > 0]

        if bad_low_buckets and good_buckets:
            # naikkan threshold ke batas bawah bucket baik pertama yg lebih tinggi dari bucket buruk
            worst = max(bucket_lo(b) for b, _ in bad_low_buckets)
            candidates = [bucket_lo(b) + 1 for b, _ in good_buckets if bucket_lo(b) > worst - 10]
            if not candidates:
                return None
            target = min(candidates)
            step = max(-MAX_THRESHOLD_STEP, min(MAX_THRESHOLD_STEP, target - current_threshold))
            new_threshold = round(max(0.0, min(95.0, current_threshold + step)), 1)
            if abs(new_threshold - current_threshold) < 0.5:
                return None
            evidence = {
                "type": "RAISE_THRESHOLD_LOW_CONF_DEGRADATION",
                "bad_buckets": {b: s for b, s in bad_low_buckets},
                "good_buckets": {b: s for b, s in good_buckets},
            }
            return new_threshold, evidence

        # --- §51 — exploratory lowering ------------------------------------
        # PENTING: begitu ACTIVE_THRESHOLD > 0, tidak ada lagi trade yang
        # dieksekusi pada confidence di bawah threshold, sehingga TIDAK ADA
        # data outcome untuk memvalidasi apakah menurunkan threshold aman
        # secara statistik. Karena itu, ini BUKAN keputusan berbasis bukti
        # penuh seperti jalur menaikkan di atas — hanya langkah eksplorasi
        # kecil (dibatasi & ditandai jelas sbg EXPLORATORY) supaya learn.py
        # kembali mendapat data di pita confidence yang sempat tertutup,
        # dan tetap harus lolos _validate_threshold_change sebelum diterapkan.
        if (
            freq
            and freq.get("status") == "HEALTHY_LOW_FREQUENCY_OR_STRICT_THRESHOLD"
            and current_threshold > 0
            and not bad_low_buckets
        ):
            new_threshold = round(max(0.0, current_threshold - min(MAX_THRESHOLD_STEP, 3.0)), 1)
            if abs(new_threshold - current_threshold) < 0.5:
                return None
            evidence = {
                "type": "EXPLORATORY_LOWER_THRESHOLD_LOW_FREQUENCY",
                "note": (
                    "Tidak ada bukti bucket rendah buruk (karena memang belum ada data outcome "
                    "di bawah threshold saat ini) — ini langkah eksplorasi kecil untuk mengumpulkan "
                    "data lagi, BUKAN kesimpulan statistik penuh. Pantau ketat siklus berikutnya."
                ),
                "frequency": freq,
            }
            return new_threshold, evidence

        return None

    def _validate_threshold_change(
        self, old_threshold: float, new_threshold: float, evidence: Dict[str, Any]
    ) -> tuple:
        """Counterfactual replay sederhana: hitung ulang expectancy gabungan
        seandainya semua trade historis dengan confidence < new_threshold
        DIABAIKAN, bandingkan dengan expectancy historis apa adanya. Hanya
        valid jika expectancy hasil filter lebih baik ATAU sama dengan
        sebelumnya (tidak boleh menerapkan perubahan yang memperburuk).

        Catatan untuk kasus EXPLORATORY_LOWER: karena hampir semua trade
        historis sudah punya confidence >= current_threshold (threshold lama
        lebih tinggi dari usulan baru), filter tidak akan banyak mengubah
        himpunan trade — validasi ini pada dasarnya "lolos otomatis" untuk
        kasus tersebut. Itu diharapkan & bukan bug: keamanan penurunan
        threshold baru benar-benar teruji lewat data LIVE pada siklus
        berikutnya, bukan lewat replay historis (lihat evidence.note)."""
        with self._lock:
            all_trades = list(self.trade_history)
        if not all_trades:
            return False, "Tidak ada data historis untuk validasi"

        baseline = self._weighted_stats(all_trades)
        filtered = [t for t in all_trades if t["confidence"] >= new_threshold]
        if len(filtered) < MIN_SAMPLE_FOR_DECISION:
            return False, f"Setelah filter, sample tersisa terlalu kecil ({len(filtered)})"
        after = self._weighted_stats(filtered)

        if after["expectancy"] >= baseline["expectancy"]:
            return True, (
                f"Expectancy membaik: {baseline['expectancy']} -> {after['expectancy']} "
                f"(n={baseline['n']} -> {after['n']})"
            )
        return False, (
            f"Expectancy tidak membaik setelah counterfactual replay "
            f"({baseline['expectancy']} -> {after['expectancy']}), perubahan ditolak"
        )
