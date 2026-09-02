from __future__ import annotations

"""
SMCAutoTrade learn_v3.py

Research / learning brain.

Responsibilities:
- Persist compact learning memory in SQLite.
- Learn passively from market events, signals, and trade outcomes.
- Compute pair/global features with pure Python.
- Measure quality, frequency, regime performance, and feature/outcome relationships.
- Generate hypotheses deterministically first; optionally ask Ollama for additional reasoning.
- Run evidence-based policy experiments against observed historical signals/trades.
- Create versioned candidate strategy files from the active strategy source.
- Promote only when a candidate beats the baseline on configurable gates.
- Save checkpoints locally and optionally push learning artifacts to GitHub.
- Restore state through /open.

Ollama is a reasoning assistant, never the primary judge.
No exchange or Telegram polling is performed here.
"""

import base64
import gzip
import hashlib
import itertools
import json
import logging
import math
import os
import re
import shutil
import sqlite3
import statistics
import threading
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger("learn")

VERSION = "3.0"
BASE_DIR = Path(__file__).resolve().parent
LEARNING_DIR = Path(os.getenv("LEARNING_DIR", str(BASE_DIR / "learning"))).resolve()
DB_FILE = Path(os.getenv("LEARNING_DB", str(LEARNING_DIR / "brain.sqlite3"))).resolve()
STATE_FILE = Path(os.getenv("LEARNING_STATE", str(LEARNING_DIR / "state.json"))).resolve()
CONSTITUTION_FILE = Path(os.getenv("LEARNING_CONSTITUTION", str(LEARNING_DIR / "constitution.json"))).resolve()
REPORT_DIR = Path(os.getenv("LEARNING_REPORT_DIR", str(LEARNING_DIR / "reports"))).resolve()
STRATEGY_DIR = Path(os.getenv("LEARNING_STRATEGY_DIR", str(LEARNING_DIR / "strategies"))).resolve()

OLLAMA_API_KEY = (os.getenv("OLLAMA_API_KEY") or "").strip()
OLLAMA_MODEL = (os.getenv("OLLAMA_MODEL") or "gpt-oss:20b").strip()
OLLAMA_URL = (os.getenv("OLLAMA_URL") or ("https://ollama.com/api/chat" if OLLAMA_API_KEY else "http://localhost:11434/api/chat")).strip()
OLLAMA_TIMEOUT = max(10, int(os.getenv("OLLAMA_TIMEOUT", "90")))

GITHUB_TOKEN = (os.getenv("GITHUB_TOKEN") or "").strip()
REPO_NAME = (os.getenv("REPO_NAME") or "").strip()
GITHUB_BRANCH = (os.getenv("GITHUB_BRANCH") or "main").strip()
GITHUB_API = "https://api.github.com"
GITHUB_PUSH_ENABLED = str(os.getenv("LEARNING_GITHUB_PUSH", "1")).strip().lower() not in {"0", "false", "no"}

MIN_SAMPLE = max(20, int(os.getenv("LEARNING_MIN_SAMPLE", "50")))
MIN_IMPROVEMENT_PCT = max(0.0, float(os.getenv("LEARNING_MIN_IMPROVEMENT_PCT", "3.0")))
MAX_FREQUENCY_DROP_PCT = max(0.0, float(os.getenv("LEARNING_MAX_FREQUENCY_DROP_PCT", "20.0")))
MAX_DRAWDOWN_WORSEN_PCT = max(0.0, float(os.getenv("LEARNING_MAX_DRAWDOWN_WORSEN_PCT", "20.0")))
CHECKPOINT_SECONDS = max(60, int(os.getenv("LEARNING_CHECKPOINT_SECONDS", "900")))
MAX_WORKERS = max(1, min(5, int(os.getenv("LEARNING_MAX_WORKERS", "5"))))
SILENCE_CHECK_SECONDS = max(300, int(os.getenv("LEARNING_SILENCE_CHECK_SECONDS", "900")))
ZERO_SIGNAL_MINUTES = max(15, int(os.getenv("LEARNING_ZERO_SIGNAL_MINUTES", "60")))
SILENCE_COOLDOWN_SECONDS = max(900, int(os.getenv("LEARNING_SILENCE_COOLDOWN_SECONDS", "3600")))

API: Any = None
CONTEXT: dict[str, Any] = {}
STRATEGY_MODULE: Any = None
STRATEGY_PATH: Path | None = None
LOCK = threading.RLock()
FULL_RUNNING = False
FULL_EXECUTOR = ThreadPoolExecutor(max_workers=MAX_WORKERS, thread_name_prefix="learn-worker")
LAST_FULL_RESULT: dict[str, Any] = {}
LAST_CHECKPOINT = 0.0
GLOBAL_CONTEXT: dict[str, Any] = {}
SILENCE_THREAD: threading.Thread | None = None
SILENCE_STOP = threading.Event()
LAST_SILENCE_AUDIT = 0.0

DEFAULT_CONSTITUTION = {
    "rr_min": 2.0,
    "rr_max": 4.0,
    "frequency_priority": True,
    "max_frequency_drop_pct": MAX_FREQUENCY_DROP_PCT,
    "min_improvement_pct": MIN_IMPROVEMENT_PCT,
    "min_sample": MIN_SAMPLE,
    "live_execution_requires_mode_on": True,
    "strategy_can_change_rr_limits": False,
}


@dataclass
class PairFeatures:
    symbol: str
    timestamp: int
    price_change_1h: float
    price_change_4h: float
    directional_efficiency_1h: float
    directional_efficiency_4h: float
    atr_pct: float
    atr_percentile: float
    relative_volume: float
    range_expansion: float
    regime: str
    directional_bias: str
    structure_label: str
    d1_bias: str
    btc_relative_strength_4h: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------- utilities ----------------
def _now_ms() -> int:
    return int(time.time() * 1000)


def _ensure_dirs() -> None:
    for p in (LEARNING_DIR, REPORT_DIR, STRATEGY_DIR):
        p.mkdir(parents=True, exist_ok=True)


def _notify(text: str) -> None:
    fn = CONTEXT.get("send_message") if isinstance(CONTEXT, dict) else None
    chat_id = CONTEXT.get("chat_id") if isinstance(CONTEXT, dict) else None
    if callable(fn) and chat_id is not None:
        try:
            fn(chat_id, text)
        except Exception:
            log.exception("[LEARN] telegram notification failed")


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def _db() -> sqlite3.Connection:
    _ensure_dirs()
    conn = sqlite3.connect(DB_FILE, timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def _init_db() -> None:
    with _db() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS observations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts INTEGER NOT NULL,
                symbol TEXT NOT NULL,
                features_json TEXT NOT NULL,
                UNIQUE(symbol, ts)
            );
            CREATE INDEX IF NOT EXISTS idx_obs_symbol_ts ON observations(symbol, ts);
            CREATE TABLE IF NOT EXISTS signals (
                signal_id TEXT PRIMARY KEY,
                ts INTEGER NOT NULL,
                symbol TEXT NOT NULL,
                direction TEXT,
                model TEXT,
                state TEXT,
                decision TEXT,
                score REAL,
                rr REAL,
                frequency_per_day REAL,
                payload_json TEXT NOT NULL,
                features_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_signals_symbol_ts ON signals(symbol, ts);
            CREATE TABLE IF NOT EXISTS trades (
                trade_id TEXT PRIMARY KEY,
                ts INTEGER NOT NULL,
                symbol TEXT NOT NULL,
                direction TEXT,
                status TEXT,
                outcome TEXT,
                pnl REAL,
                r_multiple REAL,
                mode TEXT,
                payload_json TEXT NOT NULL,
                features_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_trades_ts ON trades(ts);
            CREATE TABLE IF NOT EXISTS hypotheses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_ts INTEGER NOT NULL,
                title TEXT NOT NULL,
                source TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'OPEN'
            );
            CREATE TABLE IF NOT EXISTS experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_ts INTEGER NOT NULL,
                hypothesis_id INTEGER,
                baseline_json TEXT NOT NULL,
                candidate_json TEXT NOT NULL,
                decision TEXT NOT NULL,
                report_json TEXT NOT NULL
            );
            """
        )


def _get_meta(key: str, default: Any = None) -> Any:
    with _db() as conn:
        row = conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    if not row:
        return default
    try:
        return json.loads(row["value"])
    except Exception:
        return row["value"]


def _set_meta(key: str, value: Any) -> None:
    with _db() as conn:
        conn.execute(
            "INSERT INTO meta(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, _json(value)),
        )


def _save_state() -> None:
    global LAST_CHECKPOINT
    with LOCK:
        state = {
            "version": VERSION,
            "saved_at": time.time(),
            "active_strategy": str(STRATEGY_PATH) if STRATEGY_PATH else _get_meta("active_strategy", ""),
            "last_full": LAST_FULL_RESULT,
            "global_context": GLOBAL_CONTEXT,
        }
    _ensure_dirs()
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(STATE_FILE)
    LAST_CHECKPOINT = time.time()


def _load_state() -> dict[str, Any]:
    if not STATE_FILE.exists():
        return {}
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        log.exception("[LEARN] state read failed")
        return {}


def _load_constitution() -> dict[str, Any]:
    if not CONSTITUTION_FILE.exists():
        _ensure_dirs()
        CONSTITUTION_FILE.write_text(json.dumps(DEFAULT_CONSTITUTION, indent=2), encoding="utf-8")
        return dict(DEFAULT_CONSTITUTION)
    try:
        data = json.loads(CONSTITUTION_FILE.read_text(encoding="utf-8"))
        base = dict(DEFAULT_CONSTITUTION)
        base.update(data if isinstance(data, dict) else {})
        return base
    except Exception:
        log.exception("[LEARN] constitution read failed; using defaults")
        return dict(DEFAULT_CONSTITUTION)


# ---------------- feature engine ----------------
def _ema(values: list[float], period: int) -> float | None:
    if len(values) < period:
        return None
    a = 2.0 / (period + 1)
    e = sum(values[:period]) / period
    for x in values[period:]:
        e = a * x + (1 - a) * e
    return e


def _atr(candles: list[dict[str, Any]], period: int = 14) -> float:
    if len(candles) < period + 1:
        return 0.0
    trs = []
    for i in range(1, len(candles)):
        c, p = candles[i], candles[i - 1]
        trs.append(max(c["high"] - c["low"], abs(c["high"] - p["close"]), abs(c["low"] - p["close"])))
    return sum(trs[-period:]) / period


def _efficiency(candles: list[dict[str, Any]], bars: int) -> float:
    if len(candles) < bars + 1:
        return 0.0
    segment = candles[-bars - 1 :]
    net = abs(segment[-1]["close"] - segment[0]["close"])
    path = sum(abs(segment[i]["close"] - segment[i - 1]["close"]) for i in range(1, len(segment)))
    return net / path if path > 0 else 0.0


def _aggregate(candles: list[dict[str, Any]], minutes: int) -> list[dict[str, Any]]:
    if not candles:
        return []
    bucket = minutes * 60_000
    groups: dict[int, list[dict[str, Any]]] = {}
    for c in candles:
        key = (int(c["timestamp"]) // bucket) * bucket
        groups.setdefault(key, []).append(c)
    out = []
    for ts, group in sorted(groups.items()):
        g = sorted(group, key=lambda x: x["timestamp"])
        out.append({
            "timestamp": ts,
            "open": float(g[0]["open"]),
            "high": max(float(x["high"]) for x in g),
            "low": min(float(x["low"]) for x in g),
            "close": float(g[-1]["close"]),
            "volume": sum(float(x.get("volume", 0)) for x in g),
            "confirmed": all(bool(x.get("confirmed", True)) for x in g),
        })
    return out


def _structure_label(h1: list[dict[str, Any]]) -> str:
    if len(h1) < 10:
        return "UNKNOWN"
    look = h1[-8:]
    hi = max(c["high"] for c in look)
    lo = min(c["low"] for c in look)
    mid = (hi + lo) / 2
    close = look[-1]["close"]
    highs_up = look[-1]["high"] >= look[0]["high"]
    lows_up = look[-1]["low"] >= look[0]["low"]
    highs_down = look[-1]["high"] <= look[0]["high"]
    lows_down = look[-1]["low"] <= look[0]["low"]
    if highs_up and lows_up and close > mid:
        return "HH_HL"
    if highs_down and lows_down and close < mid:
        return "LH_LL"
    return "RANGE"


def _regime(return4h: float, eff4h: float, atr_pct: float, range_expansion: float) -> str:
    if eff4h >= 0.65 and abs(return4h) >= max(0.008, atr_pct * 1.25):
        return "TRENDING"
    if range_expansion >= 1.6:
        return "EXPANSION"
    if eff4h <= 0.25 and range_expansion <= 0.8:
        return "RANGE"
    return "TRANSITION"


def compute_pair_features(symbol: str) -> PairFeatures | None:
    if API is None:
        return None
    c15 = API.get_candles(symbol, "15", 700)
    if len(c15) < 40:
        return None
    c1 = API.get_candles(symbol, "1", 100) or []
    latest = c15[-1]
    closes = [float(c["close"]) for c in c15]
    price = closes[-1]
    ret1h = price / closes[-5] - 1 if len(closes) >= 5 else 0.0
    ret4h = price / closes[-17] - 1 if len(closes) >= 17 else 0.0
    e1 = _efficiency(c15, 4)
    e4 = _efficiency(c15, 16)
    atr = _atr(c15, 14)
    atr_pct = atr / price if price else 0.0
    atr_series: list[float] = []
    for i in range(30, len(c15) + 1):
        x = c15[:i]
        a = _atr(x, 14)
        if a:
            atr_series.append(a / x[-1]["close"])
    atr_pctile = 0.5
    if atr_series:
        below = sum(1 for x in atr_series if x <= atr_pct)
        atr_pctile = below / len(atr_series)
    vols = [float(c.get("volume", 0.0)) for c in c15]
    base_vol = statistics.fmean(vols[-21:-1]) if len(vols) >= 21 else statistics.fmean(vols[:-1] or [1.0])
    rvol = vols[-1] / base_vol if base_vol > 0 else 1.0
    ranges = [float(c["high"]) - float(c["low"]) for c in c15]
    base_range = statistics.fmean(ranges[-21:-1]) if len(ranges) >= 21 else statistics.fmean(ranges[:-1] or [1.0])
    range_exp = ranges[-1] / base_range if base_range > 0 else 1.0
    h1 = _aggregate(c15, 60)
    d1 = _aggregate(c15, 1440)
    ema9 = _ema([float(c["close"]) for c in h1], 9)
    ema20 = _ema([float(c["close"]) for c in h1], 20)
    if ema9 is not None and ema20 is not None:
        bias = "BULLISH" if ema9 > ema20 else "BEARISH" if ema9 < ema20 else "NEUTRAL"
    else:
        bias = "NEUTRAL"
    d1_bias = "NEUTRAL"
    if len(d1) >= 2:
        d1c = [float(c["close"]) for c in d1]
        d1_bias = "BULLISH" if d1c[-1] > d1c[-2] else "BEARISH" if d1c[-1] < d1c[-2] else "NEUTRAL"
    structure = _structure_label(h1)
    regime = _regime(ret4h, e4, atr_pct, range_exp)
    btc_rel = 0.0
    try:
        if symbol != "BTCUSDT":
            btc = API.get_candles("BTCUSDT", "15", 20)
            if len(btc) >= 17:
                btc_rel = ret4h - (float(btc[-1]["close"]) / float(btc[-17]["close"]) - 1)
    except Exception:
        pass
    return PairFeatures(
        symbol=symbol,
        timestamp=int(latest["timestamp"]),
        price_change_1h=ret1h,
        price_change_4h=ret4h,
        directional_efficiency_1h=e1,
        directional_efficiency_4h=e4,
        atr_pct=atr_pct,
        atr_percentile=atr_pctile,
        relative_volume=rvol,
        range_expansion=range_exp,
        regime=regime,
        directional_bias=bias,
        structure_label=structure,
        d1_bias=d1_bias,
        btc_relative_strength_4h=btc_rel,
    )


def _save_observation(f: PairFeatures) -> None:
    with _db() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO observations(ts,symbol,features_json) VALUES(?,?,?)",
            (f.timestamp, f.symbol, _json(f.to_dict())),
        )


# ---------------- global context ----------------
def build_global_context() -> dict[str, Any]:
    if API is None:
        return {}
    symbols = API.get_symbols()
    features: list[PairFeatures] = []
    for symbol in symbols:
        try:
            f = compute_pair_features(symbol)
            if f:
                features.append(f)
        except Exception:
            log.exception("[GLOBAL] feature failed %s", symbol)
    if not features:
        return {}
    bullish = sum(f.directional_bias == "BULLISH" for f in features)
    bearish = sum(f.directional_bias == "BEARISH" for f in features)
    neutral = len(features) - bullish - bearish
    breadth = bullish / len(features)
    alt = [f for f in features if f.symbol != "BTCUSDT"]
    alt_breadth = (sum(f.directional_bias == "BULLISH" for f in alt) / len(alt)) if alt else breadth
    btc = next((f for f in features if f.symbol == "BTCUSDT"), None)
    median_rvol = statistics.median(f.relative_volume for f in features)
    median_atr_pct = statistics.median(f.atr_pct for f in features)
    median_eff = statistics.median(f.directional_efficiency_4h for f in features)
    expansion_share = sum(f.range_expansion >= 1.5 for f in features) / len(features)
    transition_share = sum(f.regime == "TRANSITION" for f in features) / len(features)
    prior = GLOBAL_CONTEXT.copy()
    prior_breadth = float(prior.get("breadth", breadth))
    delta = breadth - prior_breadth

    if btc and btc.directional_bias == "BULLISH" and alt_breadth < 0.45:
        market_label = "BTC_LED"
    elif breadth >= 0.60:
        market_label = "BROAD_BULLISH"
    elif breadth <= 0.40:
        market_label = "BROAD_BEARISH"
    else:
        market_label = "MIXED"
    if transition_share >= 0.45:
        regime = "TRANSITION"
    elif expansion_share >= 0.35:
        regime = "EXPANSION"
    elif median_eff >= 0.55 and abs(breadth - 0.5) >= 0.12:
        regime = "TRENDING"
    else:
        regime = "RANGE"

    context = {
        "timestamp": _now_ms(),
        "symbols": len(features),
        "bullish": bullish,
        "bearish": bearish,
        "neutral": neutral,
        "breadth": breadth,
        "alt_breadth": alt_breadth,
        "breadth_delta": delta,
        "median_rvol": median_rvol,
        "median_atr_pct": median_atr_pct,
        "median_efficiency_4h": median_eff,
        "expansion_share": expansion_share,
        "transition_share": transition_share,
        "market_label": market_label,
        "regime": regime,
        "btc": btc.to_dict() if btc else None,
    }
    GLOBAL_CONTEXT.update(context)
    return context


# ---------------- event ingestion ----------------
def _feature_for_signal(symbol: str) -> dict[str, Any]:
    try:
        f = compute_pair_features(symbol)
        if f:
            return f.to_dict()
    except Exception:
        log.exception("[LEARN] signal feature failed %s", symbol)
    return {}


def on_market_event(event: dict[str, Any], strategy: Any = None) -> None:
    if not API or event.get("type") != "candle":
        return
    candle = event.get("candle") or {}
    symbol = str(event.get("symbol") or "").upper()
    tf = str(event.get("timeframe") or "")
    if not candle.get("confirmed") or tf != "15" or not symbol:
        return
    try:
        f = compute_pair_features(symbol)
        if f:
            _save_observation(f)
        if time.time() - LAST_CHECKPOINT >= CHECKPOINT_SECONDS:
            _save_state()
    except Exception:
        log.exception("[LEARN] market event ingestion failed")


def on_signal(signal: dict[str, Any]) -> None:
    if not signal:
        return
    sid = str(signal.get("id") or hashlib.sha1(_json(signal).encode()).hexdigest()[:16])
    symbol = str(signal.get("symbol") or "").upper()
    features = _feature_for_signal(symbol)
    with _db() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO signals
            (signal_id,ts,symbol,direction,model,state,decision,score,rr,frequency_per_day,payload_json,features_json)
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                sid, _now_ms(), symbol, signal.get("direction"), signal.get("model"), signal.get("state"),
                signal.get("decision"), signal.get("score"), signal.get("rr"), signal.get("frequency_per_day"),
                _json(signal), _json(features),
            ),
        )
    log.info("[LEARN] signal ingested %s %s", symbol, sid)


def on_trade_event(trade_event: dict[str, Any]) -> None:
    if not trade_event:
        return
    tid = str(trade_event.get("trade_id") or trade_event.get("id") or hashlib.sha1(_json(trade_event).encode()).hexdigest()[:16])
    symbol = str(trade_event.get("symbol") or "").upper()
    # Prefer the feature snapshot captured at signal time. This prevents the
    # learning engine from accidentally conditioning an outcome on post-entry
    # / post-exit information (look-ahead leakage).
    features: dict[str, Any] = {}
    signal_id = str(trade_event.get("signal_id") or "")
    if signal_id:
        try:
            with _db() as conn:
                row = conn.execute("SELECT features_json FROM signals WHERE signal_id=?", (signal_id,)).fetchone()
            if row:
                features = json.loads(row["features_json"] or "{}")
        except Exception:
            log.exception("[LEARN] signal feature lookup failed %s", signal_id)
    if not features:
        features = _feature_for_signal(symbol)
    with _db() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO trades
            (trade_id,ts,symbol,direction,status,outcome,pnl,r_multiple,mode,payload_json,features_json)
            VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
            (
                tid, int(trade_event.get("ts") or _now_ms()), symbol, trade_event.get("direction"), trade_event.get("status"),
                trade_event.get("outcome"), trade_event.get("pnl"), trade_event.get("r_multiple"), trade_event.get("mode"),
                _json(trade_event), _json(features),
            ),
        )
    if trade_event.get("status") == "CLOSED":
        _save_state()


# ---------------- signal-silence auditor ----------------
def _latest_signal_ts() -> int | None:
    with _db() as conn:
        row = conn.execute("SELECT MAX(ts) ts FROM signals").fetchone()
    return int(row["ts"]) if row and row["ts"] is not None else None

def _recent_signal_count(hours: float = 24.0) -> int:
    cutoff = _now_ms() - int(hours * 3_600_000)
    with _db() as conn:
        row = conn.execute("SELECT COUNT(*) n FROM signals WHERE ts >= ?", (cutoff,)).fetchone()
    return int(row["n"] or 0)

def diagnose_signal_silence(force: bool = False) -> dict[str, Any]:
    global LAST_SILENCE_AUDIT
    now = time.time()
    if not force and now - LAST_SILENCE_AUDIT < SILENCE_COOLDOWN_SECONDS:
        return {"skipped": True}
    LAST_SILENCE_AUDIT = now
    latest = _latest_signal_ts()
    silence_min = ((now * 1000 - latest) / 60_000.0) if latest else float("inf")
    count24 = _recent_signal_count(24.0)
    snap: dict[str, Any] = {}
    try:
        if STRATEGY_MODULE and hasattr(STRATEGY_MODULE, "get_learning_snapshot"):
            snap = STRATEGY_MODULE.get_learning_snapshot() or {}
    except Exception:
        log.exception("[SILENCE] strategy snapshot failed")
    active = snap.get("active_setups") or []
    waiting = sum(1 for x in active if x.get("state") == "WAITING_CONFIRMATION")
    pending = sum(1 for x in active if x.get("state") == "PENDING_LIMIT")
    watching = sum(1 for x in active if x.get("state") in {"WATCHING", "IN_ZONE"})
    freq_vals = [float(x.get("frequency_per_day")) for x in active if isinstance(x.get("frequency_per_day"), (int, float))]
    median_freq = statistics.median(freq_vals) if freq_vals else 0.0
    reasons = []
    if waiting: reasons.append(f"{waiting} setup stuck at confirmation")
    if watching and not waiting and not pending: reasons.append(f"{watching} setup not reaching confirmation/POI")
    if pending: reasons.append(f"{pending} confirmed setup pending execution")
    if median_freq <= 0: reasons.append("strategy opportunity-frequency estimate is zero")
    if not active: reasons.append("no active setup candidates")
    result = {"ts": _now_ms(), "silence_minutes": silence_min, "signals_24h": count24, "active": len(active), "waiting_confirmation": waiting, "pending_limit": pending, "watching_or_in_zone": watching, "median_setup_frequency_per_day": median_freq, "reasons": reasons, "market_context": GLOBAL_CONTEXT}
    with _db() as conn:
        conn.execute("INSERT INTO hypotheses(created_ts,title,source,payload_json,status) VALUES(?,?,?,?,?)", (_now_ms(), "ZERO_SIGNAL_SILENCE_AUDIT", "automatic", _json(result), "OPEN"))
    log.warning("[SILENCE AUDIT] %s", result)
    return result

def _silence_worker() -> None:
    while not SILENCE_STOP.wait(SILENCE_CHECK_SECONDS):
        try:
            latest = _latest_signal_ts()
            silence_min = ((time.time() * 1000 - latest) / 60_000.0) if latest else float("inf")
            if silence_min >= ZERO_SIGNAL_MINUTES:
                diagnose_signal_silence()
        except Exception:
            log.exception("[SILENCE] automatic audit failed")

def _start_silence_monitor() -> None:
    global SILENCE_THREAD
    SILENCE_STOP.clear()
    if SILENCE_THREAD and SILENCE_THREAD.is_alive(): return
    SILENCE_THREAD = threading.Thread(target=_silence_worker, name="learning-silence-auditor", daemon=True)
    SILENCE_THREAD.start()

def _stop_silence_monitor() -> None:
    SILENCE_STOP.set()

# ---------------- statistics ----------------
def _load_closed_rows() -> list[sqlite3.Row]:
    with _db() as conn:
        return list(conn.execute("SELECT * FROM trades WHERE status='CLOSED' ORDER BY ts"))


def _baseline_metrics(rows: list[sqlite3.Row]) -> dict[str, Any]:
    rs = [float(r["r_multiple"]) for r in rows if r["r_multiple"] is not None]
    pnl = [float(r["pnl"] or 0) for r in rows]
    wins = sum(x > 0 for x in rs)
    running = 0.0
    peak = 0.0
    max_dd = 0.0
    for x in rs:
        running += x
        peak = max(peak, running)
        max_dd = max(max_dd, peak - running)
    return {
        "sample": len(rs),
        "wins": wins,
        "losses": len(rs) - wins,
        "win_rate": wins / len(rs) if rs else 0.0,
        "expectancy_r": statistics.fmean(rs) if rs else 0.0,
        "net_r": sum(rs),
        "net_pnl": sum(pnl),
        "max_drawdown_r": max_dd,
        "frequency_per_day": (len(rs) / max(1.0, (rows[-1]["ts"] - rows[0]["ts"]) / 86_400_000)) if len(rows) >= 2 else 0.0,
    }


def _signal_trade_join() -> list[dict[str, Any]]:
    with _db() as conn:
        rows = list(conn.execute("SELECT * FROM trades WHERE status='CLOSED' ORDER BY ts"))
    joined: list[dict[str, Any]] = []
    for r in rows:
        payload = {}
        feats = {}
        try:
            payload = json.loads(r["payload_json"])
            feats = json.loads(r["features_json"])
        except Exception:
            pass
        joined.append({**dict(r), "payload": payload, "features": feats})
    return joined


def _group_expectancy(rows: list[dict[str, Any]], key_fn, min_sample: int = 10) -> dict[str, Any]:
    groups: dict[str, list[float]] = {}
    for r in rows:
        key = str(key_fn(r))
        if r.get("r_multiple") is None:
            continue
        groups.setdefault(key, []).append(float(r["r_multiple"]))
    out = {}
    for k, vals in groups.items():
        if len(vals) >= min_sample:
            out[k] = {
                "sample": len(vals),
                "expectancy_r": statistics.fmean(vals),
                "win_rate": sum(v > 0 for v in vals) / len(vals),
            }
    return out


# ---------------- hypotheses / experiments ----------------
def generate_hypotheses() -> list[dict[str, Any]]:
    rows = _signal_trade_join()
    if len(rows) < MIN_SAMPLE:
        return [{"title": "INSUFFICIENT_SAMPLE", "source": "python", "reason": f"Need {MIN_SAMPLE}, have {len(rows)}"}]

    hyps: list[dict[str, Any]] = []
    by_regime = _group_expectancy(rows, lambda r: r.get("features", {}).get("regime", "UNKNOWN"))
    for regime, metrics in by_regime.items():
        if metrics["expectancy_r"] < 0:
            hyps.append({
                "title": f"REGIME_FILTER_{regime}",
                "source": "python",
                "change_type": "regime_penalty",
                "regime": regime,
                "reason": metrics,
            })

    for field_name, label in (
        ("relative_volume", "RVOL"),
        ("directional_efficiency_4h", "EFF4H"),
        ("atr_percentile", "ATR_PCTL"),
        ("range_expansion", "RANGE_EXP"),
    ):
        high: list[float] = []
        low: list[float] = []
        values = [r.get("features", {}).get(field_name) for r in rows]
        values = [float(v) for v in values if isinstance(v, (int, float))]
        if len(values) < MIN_SAMPLE:
            continue
        q25, q75 = statistics.quantiles(values, n=4, method="inclusive")[0], statistics.quantiles(values, n=4, method="inclusive")[2]
        for r in rows:
            v = r.get("features", {}).get(field_name)
            rr = r.get("r_multiple")
            if not isinstance(v, (int, float)) or rr is None:
                continue
            if float(v) >= q75:
                high.append(float(rr))
            elif float(v) <= q25:
                low.append(float(rr))
        if len(high) >= 10 and len(low) >= 10:
            diff = statistics.fmean(high) - statistics.fmean(low)
            if abs(diff) >= 0.08:
                hyps.append({
                    "title": f"{label}_REGIME_EFFECT",
                    "source": "python",
                    "change_type": "feature_filter",
                    "feature": field_name,
                    "low_mean_r": statistics.fmean(low),
                    "high_mean_r": statistics.fmean(high),
                    "delta_r": diff,
                    "q25": q25,
                    "q75": q75,
                })

    by_model = _group_expectancy(rows, lambda r: r.get("payload", {}).get("model", "UNKNOWN"))
    for model, m in by_model.items():
        if m["sample"] >= MIN_SAMPLE and m["expectancy_r"] < -0.05:
            hyps.append({
                "title": f"MODEL_WEAK_{model}",
                "source": "python",
                "change_type": "model_penalty",
                "model": model,
                "metrics": m,
            })
    return hyps


def _ollama_reason(hypotheses: list[dict[str, Any]], baseline: dict[str, Any], global_ctx: dict[str, Any]) -> list[dict[str, Any]]:
    if not OLLAMA_API_KEY and "ollama.com" in OLLAMA_URL:
        return []
    prompt = {
        "task": "Propose conservative, testable strategy experiments. Never declare a change valid. Return JSON only.",
        "baseline": baseline,
        "global_context": global_ctx,
        "observed_hypotheses": hypotheses[:20],
        "constitution": _load_constitution(),
        "required_output": {"hypotheses": [{"title": "", "change_type": "", "feature": "", "threshold": 0, "reason": ""}]},
    }
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": "You are a trading research assistant. Evidence first. Return JSON only. Do not give live trading instructions. Never declare a strategy change safe; only propose testable hypotheses."},
            {"role": "user", "content": _json(prompt)},
        ],
        "stream": False,
        "options": {"temperature": 0.1},
    }
    req = urllib.request.Request(OLLAMA_URL, data=_json(payload).encode(), method="POST")
    req.add_header("Content-Type", "application/json")
    if OLLAMA_API_KEY:
        req.add_header("Authorization", f"Bearer {OLLAMA_API_KEY}")
    try:
        with urllib.request.urlopen(req, timeout=OLLAMA_TIMEOUT) as resp:
            body = json.loads(resp.read().decode())
        content = ((body.get("message") or {}).get("content") or "").strip()
        parsed = json.loads(content)
        vals = parsed.get("hypotheses") or []
        return [dict(x, source="ollama") for x in vals if isinstance(x, dict)]
    except Exception:
        log.exception("[OLLAMA] reasoning failed; continuing with Python hypotheses")
        return []


def _experiment_filter(rows: list[dict[str, Any]], field: str, threshold: float, direction: str = ">=") -> list[dict[str, Any]]:
    out = []
    for r in rows:
        v = r.get("features", {}).get(field)
        if not isinstance(v, (int, float)):
            continue
        ok = float(v) >= threshold if direction == ">=" else float(v) <= threshold
        if ok:
            out.append(r)
    return out



def _walk_forward(rows: list[dict[str, Any]], candidate_selector) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda r: int(r.get("ts") or 0))
    if len(ordered) < max(MIN_SAMPLE, 40):
        return {"pass": False, "folds": [], "reason": "insufficient chronological sample"}
    fold_size = max(10, len(ordered) // 5)
    folds = []
    candidate_wins = 0
    total = 0
    for i in range(0, len(ordered) - fold_size + 1, fold_size):
        test = ordered[i:i + fold_size]
        if len(test) < 10:
            continue
        base_m = _simulate_rows(test)
        cand_m = _simulate_rows(candidate_selector(test))
        total += 1
        if cand_m["expectancy_r"] > base_m["expectancy_r"]:
            candidate_wins += 1
        folds.append({"index": total, "baseline": base_m, "candidate": cand_m})
        if total >= 5:
            break
    return {
        "pass": total >= 3 and candidate_wins >= math.ceil(total * 0.5),
        "folds": folds,
        "candidate_winning_folds": candidate_wins,
        "total_folds": total,
    }


def _oos_test(rows: list[dict[str, Any]], candidate_selector) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda r: int(r.get("ts") or 0))
    if len(ordered) < max(MIN_SAMPLE, 50):
        return {"pass": False, "reason": "insufficient OOS sample"}
    cut = max(10, int(len(ordered) * 0.8))
    oos = ordered[cut:]
    base = _simulate_rows(oos)
    cand = _simulate_rows(candidate_selector(oos))
    return {
        "pass": cand["sample"] >= max(10, len(oos) // 2) and cand["expectancy_r"] >= base["expectancy_r"],
        "sample": len(oos),
        "baseline": base,
        "candidate": cand,
    }


def _robustness_test(rows: list[dict[str, Any]], candidate_selector) -> dict[str, Any]:
    selected = candidate_selector(rows)
    if not selected:
        return {"pass": False, "reason": "candidate selects zero rows"}
    # Conservative execution friction stress: subtract a fixed 0.05R from winners
    # and add 0.05R to losers to approximate extra slippage/fees.
    stressed = []
    for r in selected:
        rr = float(r.get("r_multiple") or 0.0)
        stressed.append({"ts": r.get("ts"), "r_multiple": rr - (0.05 if rr >= 0 else -0.05)})
    m = _simulate_rows(stressed)
    return {"pass": m["expectancy_r"] > 0, "stressed": m}

def _simulate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    vals = [float(r["r_multiple"]) for r in rows if r.get("r_multiple") is not None]
    if not vals:
        return {"sample": 0, "expectancy_r": 0.0, "win_rate": 0.0, "net_r": 0.0, "max_drawdown_r": 0.0, "frequency_per_day": 0.0}
    running = peak = dd = 0.0
    for x in vals:
        running += x
        peak = max(peak, running)
        dd = max(dd, peak - running)
    duration_days = 1.0
    ts = [int(r.get("ts") or 0) for r in rows]
    if len(ts) >= 2 and max(ts) > min(ts):
        duration_days = max(1.0, (max(ts) - min(ts)) / 86_400_000)
    return {
        "sample": len(vals),
        "expectancy_r": statistics.fmean(vals),
        "win_rate": sum(x > 0 for x in vals) / len(vals),
        "net_r": sum(vals),
        "max_drawdown_r": dd,
        "frequency_per_day": len(vals) / duration_days,
    }


def evaluate_hypothesis(h: dict[str, Any], rows: list[dict[str, Any]], baseline: dict[str, Any]) -> dict[str, Any]:
    change_type = h.get("change_type")

    def selector(data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if change_type == "feature_filter" and h.get("feature") and isinstance(h.get("q75"), (int, float)):
            return _experiment_filter(data, str(h["feature"]), float(h["q75"]), ">=")
        if change_type == "regime_penalty" and h.get("regime"):
            return [r for r in data if r.get("features", {}).get("regime") != h.get("regime")]
        if change_type == "model_penalty" and h.get("model"):
            return [r for r in data if r.get("payload", {}).get("model") != h.get("model")]
        return list(data)

    candidate_rows = selector(rows)
    candidate = _simulate_rows(candidate_rows)
    sample_ok = candidate["sample"] >= max(10, min(MIN_SAMPLE, baseline["sample"]))
    improvement = ((candidate["expectancy_r"] - baseline["expectancy_r"]) / abs(baseline["expectancy_r"]) * 100) if baseline["expectancy_r"] else 0.0
    freq_drop = ((baseline["frequency_per_day"] - candidate["frequency_per_day"]) / baseline["frequency_per_day"] * 100) if baseline["frequency_per_day"] else 0.0
    dd_worse = ((candidate["max_drawdown_r"] - baseline["max_drawdown_r"]) / max(abs(baseline["max_drawdown_r"]), 1e-9) * 100) if baseline["max_drawdown_r"] else 0.0
    wf = _walk_forward(rows, selector)
    oos = _oos_test(rows, selector)
    robust = _robustness_test(rows, selector)

    decision = "REJECT"
    if (
        sample_ok
        and improvement >= MIN_IMPROVEMENT_PCT
        and freq_drop <= MAX_FREQUENCY_DROP_PCT
        and dd_worse <= MAX_DRAWDOWN_WORSEN_PCT
        and wf.get("pass")
        and oos.get("pass")
        and robust.get("pass")
    ):
        decision = "PROMOTE_CANDIDATE"

    return {
        "hypothesis": h,
        "baseline": baseline,
        "candidate": candidate,
        "sample_ok": sample_ok,
        "improvement_pct": improvement,
        "frequency_drop_pct": freq_drop,
        "drawdown_worsen_pct": dd_worse,
        "walk_forward": wf,
        "out_of_sample": oos,
        "robustness": robust,
        "decision": decision,
    }


# ---------------- strategy candidate generation ----------------
def _next_strategy_version(current: Path) -> int:
    versions = []
    for p in STRATEGY_DIR.glob("strategy_v*.py"):
        m = re.search(r"strategy_v(\d+)", p.name)
        if m:
            versions.append(int(m.group(1)))
    m = re.search(r"strategy_v(\d+)", current.name)
    if m:
        versions.append(int(m.group(1)))
    return (max(versions) + 1) if versions else 1


def _current_policy() -> dict[str, Any]:
    return {
        "min_score": 58,
        "transition_penalty": 0,
        "rvol_min": 0.0,
        "efficiency_min": 0.0,
    }


def _read_policy_from_source(source: str) -> dict[str, Any]:
    policy = _current_policy()
    for key, default in policy.items():
        m = re.search(rf"[\"']{re.escape(key)}[\"']\s*:\s*([-+]?\d+(?:\.\d+)?)", source)
        if m:
            try:
                policy[key] = float(m.group(1)) if "." in m.group(1) else int(m.group(1))
            except ValueError:
                pass
    return policy


def create_candidate_strategy(hypothesis_result: dict[str, Any], base_path: Path) -> Path | None:
    if not base_path.exists():
        return None
    source = base_path.read_text(encoding="utf-8")
    version = _next_strategy_version(base_path)
    candidate = STRATEGY_DIR / f"strategy_v{version}_candidate.py"

    policy = _current_policy()
    hr = hypothesis_result.get("hypothesis") or {}
    if hr.get("change_type") == "feature_filter":
        field = hr.get("feature")
        q75 = hr.get("q75")
        if field == "relative_volume" and isinstance(q75, (int, float)):
            policy["rvol_min"] = round(float(q75), 4)
        if field == "directional_efficiency_4h" and isinstance(q75, (int, float)):
            policy["efficiency_min"] = round(float(q75), 4)
    elif hr.get("change_type") == "regime_penalty" and hr.get("regime") == "TRANSITION":
        policy["transition_penalty"] = 8

    marker = "# LEARNED_POLICY_V1 = "
    block = marker + _json(policy) + "\n"
    if marker in source:
        source = re.sub(r"# LEARNED_POLICY_V1 = .*\n", block, source, count=1)
    else:
        source = source.replace("from __future__ import annotations\n", "from __future__ import annotations\n\n" + block, 1)

    source = source.replace("strategy_v5", f"strategy_v{version}", 1)
    source = source.replace('MIN_SCORE = int(os.getenv("STRAT_V5_MIN_SCORE", "58"))', f'MIN_SCORE = int(os.getenv("STRAT_V5_MIN_SCORE", "{policy["min_score"]}"))')
    source = source.replace('SL_ATR_PAD = float(os.getenv("STRAT_V5_SL_ATR_PAD", "0.20"))', 'SL_ATR_PAD = float(os.getenv("STRAT_V5_SL_ATR_PAD", "0.20"))')
    candidate.write_text(source, encoding="utf-8")
    return candidate


def _promote_candidate(candidate: Path, report: dict[str, Any], base_path: Path) -> Path:
    version_match = re.search(r"strategy_v(\d+)_candidate", candidate.name)
    version = version_match.group(1) if version_match else str(_next_strategy_version(base_path))
    promoted = BASE_DIR / f"strategy_v{version}.py"
    shutil.copy2(candidate, promoted)
    manifest = promoted.with_suffix(".json")
    manifest.write_text(json.dumps({"parent": str(base_path), "report": report, "promoted_at": time.time()}, indent=2, ensure_ascii=False), encoding="utf-8")
    _set_meta("active_strategy", str(promoted))
    return promoted


# ---------------- github persistence ----------------
def _github_get_json(path: str) -> dict[str, Any]:
    return _github_request(path, method="GET")

def _github_file_blob(repo_path: str) -> bytes | None:
    if not GITHUB_TOKEN or not REPO_NAME:
        return None
    try:
        ref = _github_get_json(f"/repos/{REPO_NAME}/git/ref/heads/{urllib.parse.quote(GITHUB_BRANCH)}")
        head = ((ref.get("object") or {}).get("sha") or "")
        commit = _github_get_json(f"/repos/{REPO_NAME}/git/commits/{head}")
        tree_sha = ((commit.get("tree") or {}).get("sha") or "")
        tree = _github_get_json(f"/repos/{REPO_NAME}/git/trees/{tree_sha}?recursive=1")
        target = repo_path.strip("/")
        item = next((x for x in tree.get("tree", []) if x.get("path") == target and x.get("type") == "blob"), None)
        if not item:
            return None
        blob = _github_get_json(f"/repos/{REPO_NAME}/git/blobs/{item['sha']}")
        if blob.get("encoding") != "base64":
            return None
        return base64.b64decode(blob.get("content", ""))
    except Exception:
        log.exception("[GITHUB] restore failed %s", repo_path)
        return None

def _github_put_large_file(local: Path, repo_path: str, message: str) -> None:
    if not GITHUB_PUSH_ENABLED or not GITHUB_TOKEN or not REPO_NAME:
        return
    data = local.read_bytes()
    blob = _github_request(f"/repos/{REPO_NAME}/git/blobs", method="POST", payload={"content": base64.b64encode(data).decode(), "encoding": "base64"})
    ref = _github_get_json(f"/repos/{REPO_NAME}/git/ref/heads/{urllib.parse.quote(GITHUB_BRANCH)}")
    head = ((ref.get("object") or {}).get("sha") or "")
    commit0 = _github_get_json(f"/repos/{REPO_NAME}/git/commits/{head}")
    base_tree = ((commit0.get("tree") or {}).get("sha") or "")
    tree = _github_request(f"/repos/{REPO_NAME}/git/trees", method="POST", payload={"base_tree": base_tree, "tree": [{"path": repo_path, "mode": "100644", "type": "blob", "sha": blob.get("sha")}]})
    commit = _github_request(f"/repos/{REPO_NAME}/git/commits", method="POST", payload={"message": message, "tree": tree.get("sha"), "parents": [head]})
    _github_request(f"/repos/{REPO_NAME}/git/refs/heads/{urllib.parse.quote(GITHUB_BRANCH)}", method="PATCH", payload={"sha": commit.get("sha")})

def _github_request(path: str, method: str = "GET", payload: dict[str, Any] | None = None) -> dict[str, Any]:
    if not GITHUB_TOKEN or not REPO_NAME:
        raise RuntimeError("GITHUB_TOKEN/REPO_NAME tidak tersedia")
    url = f"{GITHUB_API}{path}"
    data = _json(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", f"Bearer {GITHUB_TOKEN}")
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("X-GitHub-Api-Version", "2022-11-28")
    if data:
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())


def _github_put_file(local: Path, repo_path: str, message: str) -> None:
    if not GITHUB_PUSH_ENABLED:
        return
    if not GITHUB_TOKEN or not REPO_NAME:
        log.warning("[GITHUB] push skipped: missing credentials/repo")
        return
    api_path = f"/repos/{REPO_NAME}/contents/{urllib.parse.quote(repo_path)}?ref={urllib.parse.quote(GITHUB_BRANCH)}"
    sha = None
    try:
        existing = _github_request(api_path)
        sha = existing.get("sha")
    except Exception:
        pass
    content = base64.b64encode(local.read_bytes()).decode()
    payload = {"message": message, "content": content, "branch": GITHUB_BRANCH}
    if sha:
        payload["sha"] = sha
    _github_request(f"/repos/{REPO_NAME}/contents/{urllib.parse.quote(repo_path)}", method="PUT", payload=payload)


def push_checkpoint(report: dict[str, Any] | None = None) -> None:
    _ensure_dirs()
    _save_state()
    report = report or LAST_FULL_RESULT
    checkpoint = REPORT_DIR / f"checkpoint_{int(time.time())}.json"
    checkpoint.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    for local, repo in (
        (STATE_FILE, "learning/state.json"),
        (CONSTITUTION_FILE, "learning/constitution.json"),
        (checkpoint, f"learning/reports/{checkpoint.name}"),
    ):
        try:
            _github_put_file(local, repo, f"learn: save {local.name}")
        except Exception:
            log.exception("[GITHUB] push failed %s", local)
    try:
        db_gz = LEARNING_DIR / "brain.sqlite3.gz"
        with DB_FILE.open("rb") as src, gzip.open(db_gz, "wb", compresslevel=6) as dst:
            dst.write(src.read())
        _github_put_large_file(db_gz, "learning/brain.sqlite3.gz", "learn: backup brain memory")
    except Exception:
        log.exception("[GITHUB] sqlite backup failed")


def push_strategy(path: Path) -> None:
    try:
        _github_put_file(path, f"learning/strategies/{path.name}", f"learn: promote {path.name}")
        # Also push root strategy version so main's all-python sync can see it.
        _github_put_file(path, path.name, f"learn: promote {path.name}")
    except Exception:
        log.exception("[GITHUB] strategy push failed %s", path)


# ---------------- public lifecycle ----------------
def initialize(api: Any, context: dict[str, Any]) -> None:
    global API, CONTEXT, LAST_FULL_RESULT, GLOBAL_CONTEXT
    API = api
    CONTEXT = dict(context)
    _init_db()
    state = _load_state()
    LAST_FULL_RESULT = dict(state.get("last_full") or {})
    GLOBAL_CONTEXT.update(state.get("global_context") or {})
    _ensure_dirs()
    _load_constitution()
    _start_silence_monitor()
    log.info("[LEARN V3] initialized | db=%s | ollama=%s | github_push=%s", DB_FILE, OLLAMA_MODEL, GITHUB_PUSH_ENABLED)


def on_strategy_loaded(module: Any, path: str | Path) -> None:
    global STRATEGY_MODULE, STRATEGY_PATH
    STRATEGY_MODULE = module
    STRATEGY_PATH = Path(path).resolve()
    _set_meta("active_strategy", str(STRATEGY_PATH))
    _save_state()
    log.info("[LEARN] active strategy=%s", STRATEGY_PATH)


def _restore_from_github() -> bool:
    restored = False
    gz = _github_file_blob("learning/brain.sqlite3.gz")
    if gz:
        try:
            temp_gz = LEARNING_DIR / "brain.restore.sqlite3.gz"
            temp_db = LEARNING_DIR / "brain.restore.sqlite3"
            temp_gz.write_bytes(gz)
            with gzip.open(temp_gz, "rb") as src, temp_db.open("wb") as dst:
                dst.write(src.read())
            temp_db.replace(DB_FILE)
            temp_gz.unlink(missing_ok=True)
            restored = True
        except Exception:
            log.exception("[GITHUB] brain memory restore failed")
    for repo_path, local in (("learning/state.json", STATE_FILE), ("learning/constitution.json", CONSTITUTION_FILE)):
        data = _github_file_blob(repo_path)
        if data:
            try:
                local.parent.mkdir(parents=True, exist_ok=True)
                local.write_bytes(data)
                restored = True
            except Exception:
                log.exception("[GITHUB] restore failed %s", repo_path)
    if restored:
        _init_db()
    return restored

def open_memory() -> str:
    # Pause the monitor while replacing the SQLite file from GitHub.
    _stop_silence_monitor()
    restored_github = _restore_from_github() if (GITHUB_TOKEN and REPO_NAME) else False
    state = _load_state()
    global GLOBAL_CONTEXT, LAST_FULL_RESULT
    GLOBAL_CONTEXT = dict(state.get("global_context") or GLOBAL_CONTEXT)
    LAST_FULL_RESULT = dict(state.get("last_full") or LAST_FULL_RESULT)
    active = state.get("active_strategy") or _get_meta("active_strategy", "-")
    with _db() as conn:
        obs = conn.execute("SELECT COUNT(*) n FROM observations").fetchone()["n"]
        signals = conn.execute("SELECT COUNT(*) n FROM signals").fetchone()["n"]
        trades = conn.execute("SELECT COUNT(*) n FROM trades WHERE status='CLOSED'").fetchone()["n"]
        hyps = conn.execute("SELECT COUNT(*) n FROM hypotheses").fetchone()["n"]
        exps = conn.execute("SELECT COUNT(*) n FROM experiments").fetchone()["n"]
    source = "GITHUB + LOCAL" if restored_github else "LOCAL"
    _start_silence_monitor()
    return (
        "🧠 LEARNING MEMORY OPENED\n\n"
        f"Source: {source}\n"
        f"Active strategy: {Path(str(active)).name}\n"
        f"Observations: {obs}\n"
        f"Signals learned: {signals}\n"
        f"Closed trades learned: {trades}\n"
        f"Hypotheses: {hyps}\n"
        f"Experiments: {exps}\n"
        "State: RESTORED"
    )

def status() -> str:
    with _db() as conn:
        obs = conn.execute("SELECT COUNT(*) n FROM observations").fetchone()["n"]
        signals = conn.execute("SELECT COUNT(*) n FROM signals").fetchone()["n"]
        trades = conn.execute("SELECT COUNT(*) n FROM trades WHERE status='CLOSED'").fetchone()["n"]
        hyps = conn.execute("SELECT COUNT(*) n FROM hypotheses WHERE status='OPEN'").fetchone()["n"]
        exps = conn.execute("SELECT COUNT(*) n FROM experiments").fetchone()["n"]
    active = Path(str(_get_meta("active_strategy", STRATEGY_PATH or "-"))).name
    return (
        "🧠 LEARN STATUS\n"
        f"Version: {VERSION}\n"
        f"Memory observations: {obs}\n"
        f"Signals: {signals}\n"
        f"Closed trades: {trades}\n"
        f"Open hypotheses: {hyps}\n"
        f"Experiments: {exps}\n"
        f"Active strategy: {active}\n"
        f"Full cycle: {'RUNNING' if FULL_RUNNING else 'IDLE'}\n"
        f"Market regime: {GLOBAL_CONTEXT.get('regime', '-')}\n"
        f"Market label: {GLOBAL_CONTEXT.get('market_label', '-')}"
    )


def full_cycle_background() -> str:
    global FULL_RUNNING
    with LOCK:
        if FULL_RUNNING:
            return "ℹ️ /full sedang berjalan."
        FULL_RUNNING = True
    FULL_EXECUTOR.submit(_full_worker)
    return "🧠 FULL LEARNING STARTED\nResearch berjalan di background. Lihat terminal untuk progress; gunakan /learn untuk status."


def _full_worker() -> None:
    global FULL_RUNNING, LAST_FULL_RESULT
    started = time.time()
    try:
        log.info("[FULL] cycle START")
        global_ctx = build_global_context()
        rows = _signal_trade_join()
        baseline = _baseline_metrics(_load_closed_rows())
        log.info("[FULL] baseline sample=%d exp=%.4fR freq=%.2f/d dd=%.2fR", baseline["sample"], baseline["expectancy_r"], baseline["frequency_per_day"], baseline["max_drawdown_r"])

        hypotheses = generate_hypotheses()
        llm_h = _ollama_reason(hypotheses, baseline, global_ctx)
        combined = hypotheses + llm_h

        # Persist hypotheses.
        with _db() as conn:
            for h in combined:
                conn.execute(
                    "INSERT INTO hypotheses(created_ts,title,source,payload_json) VALUES(?,?,?,?)",
                    (_now_ms(), str(h.get("title") or "UNTITLED"), str(h.get("source") or "python"), _json(h)),
                )

        results = []
        for h in combined:
            try:
                result = evaluate_hypothesis(h, rows, baseline)
                results.append(result)
                with _db() as conn:
                    conn.execute(
                        "INSERT INTO experiments(created_ts,hypothesis_id,baseline_json,candidate_json,decision,report_json) VALUES(?,?,?,?,?,?)",
                        (_now_ms(), None, _json(baseline), _json(result.get("candidate")), result["decision"], _json(result)),
                    )
                log.info("[FULL] experiment %s → %s improvement=%.2f%% freq_drop=%.2f%%", h.get("title"), result["decision"], result["improvement_pct"], result["frequency_drop_pct"])
            except Exception:
                log.exception("[FULL] experiment failed %s", h.get("title"))

        promotable = [r for r in results if r.get("decision") == "PROMOTE_CANDIDATE"]
        best = max(promotable, key=lambda r: (r["improvement_pct"], -r["frequency_drop_pct"])) if promotable else None
        candidate_path = None
        promoted_path = None
        decision = "NO_CHANGE"
        if best and STRATEGY_PATH:
            candidate_path = create_candidate_strategy(best, STRATEGY_PATH)
            if candidate_path:
                decision = "CANDIDATE_CREATED"
                # Static validation of generated source.
                compile(candidate_path.read_text(encoding="utf-8"), str(candidate_path), "exec")
                promoted_path = _promote_candidate(candidate_path, best, STRATEGY_PATH)
                push_strategy(promoted_path)
                decision = "PROMOTED"
                log.info("[FULL] promoted %s", promoted_path)
                _notify(
                    "✅ LEARNING PROMOTED STRATEGY\n"
                    f"New strategy: {promoted_path.name}\n"
                    "Validation: retrospective evidence gate passed.\n"
                    "Reloading strategy runtime..."
                )
                reset_cb = CONTEXT.get("reset_strategy") if isinstance(CONTEXT, dict) else None
                if callable(reset_cb):
                    try:
                        reset_result = reset_cb()
                        log.info("[FULL] strategy reset callback: %s", reset_result)
                    except Exception:
                        log.exception("[FULL] promoted strategy reset failed")
                        _notify("🚨 STRATEGY AUTO-RELOAD FAILED\nPromoted version was saved; use /reset after reviewing terminal.")

        LAST_FULL_RESULT = {
            "started_at": started,
            "finished_at": time.time(),
            "duration_s": time.time() - started,
            "baseline": baseline,
            "global_context": global_ctx,
            "hypotheses": combined,
            "experiments": results,
            "promotable": len(promotable),
            "candidate_strategy": str(candidate_path) if candidate_path else None,
            "promoted_strategy": str(promoted_path) if promoted_path else None,
            "decision": decision,
        }
        report = REPORT_DIR / f"full_{time.strftime('%Y%m%d_%H%M%S')}.json"
        report.write_text(json.dumps(LAST_FULL_RESULT, ensure_ascii=False, indent=2), encoding="utf-8")
        _save_state()
        push_checkpoint(LAST_FULL_RESULT)
        _notify(
            "🧠 FULL LEARNING COMPLETE\n\n"
            f"Decision: {decision}\n"
            f"Hypotheses: {len(combined)}\n"
            f"Experiments: {len(results)}\n"
            f"Promotable: {len(promotable)}\n"
            f"Duration: {time.time() - started:.1f}s"
        )
        log.info("[FULL] cycle DONE decision=%s duration=%.1fs", decision, time.time() - started)
    except Exception as exc:
        log.exception("[FULL] cycle failed")
        LAST_FULL_RESULT = {"decision": "ERROR", "error": f"{type(exc).__name__}: {exc}", "finished_at": time.time()}
        _save_state()
        _notify(f"🚨 FULL LEARNING ERROR\n{type(exc).__name__}: {exc}")
    finally:
        FULL_RUNNING = False


def handle_command(text: str) -> str | None:
    parts = text.split()
    cmd = parts[0].lower() if parts else ""
    if cmd == "/open":
        return open_memory()
    if cmd == "/silence":
        result = diagnose_signal_silence(force=True)
        return "🧠 SIGNAL SILENCE AUDIT\n" + json.dumps(result, ensure_ascii=False, indent=2, default=str)[:3600]
    if cmd == "/learn":
        return status()
    if cmd == "/save":
        push_checkpoint(LAST_FULL_RESULT)
        return "✅ Learning checkpoint saved locally and pushed to GitHub."
    if cmd == "/full":
        return full_cycle_background()
    if cmd == "/learningreport":
        if not LAST_FULL_RESULT:
            return "📭 Belum ada /full report."
        return json.dumps(LAST_FULL_RESULT, ensure_ascii=False)[:3900]
    return None


def shutdown() -> None:
    global FULL_RUNNING
    FULL_RUNNING = False
    _stop_silence_monitor()
    try:
        _save_state()
    except Exception:
        log.exception("[LEARN] shutdown save failed")


def get_global_context() -> dict[str, Any]:
    return dict(GLOBAL_CONTEXT)
