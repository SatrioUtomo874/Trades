"""
strategy_logic.py — OTAK v2 (Revisi Berbasis Transkrip RUANG TRADER)
=====================================================================
Dibangun ulang dari 30 transkrip video SMC/ICT (channel RUANG TRADER) dengan
penekanan pada:
  • Entry yang presisi (OB di zona diskon/premium, Liquidity Sweep, ChoCH, OTE)
  • SL struktural yang anti‑Liquidity Sweep (buffer + level M15/H1)
  • TP dinamis: jika RR < 2.0, cari target lebih jauh (cap 4.0)
  • Confidence global tunggal (tanpa bias sesi)
  • Trail Ladder sebagai update SL struktural, BUKAN profit‑taker paksa

Kompatibel dengan main.py:
  - full_analyze(df_h1, df_m15, df_d1=None, symbol=None) → dict | None
  - score_direction(df_h1, df_m15, df_d1=None) → dict | None
  - swing_pts(df, lb) → (sh, sl)
  - TRAIL_R_LADDER, STRUCT_TRAIL_LB, STRUCT_TRAIL_BUF_PCT, STRUCT_TRAIL_LOOKBACK
  - MIN_RR, MAX_RR, FIB_EXT_1, FIB_EXT_2
"""

import logging
from typing import Optional, Tuple
import pandas as pd
import numpy as np

log = logging.getLogger(__name__)

# =============================================================================
# KONFIGURASI — Diimpor langsung oleh main.py
# =============================================================================

MIN_RR   = 2.0
MAX_RR   = 4.0

# Trail Ladder: (min_R_profit_untuk_aktif, fraksi_lock_dari_risk)
# Digunakan sebagai update SL struktural, bukan profit‑taker.
TRAIL_R_LADDER = [
    (0.5, 0.00),   # break-even
    (1.0, 0.30),   # lock 0.3 R
    (1.5, 0.50),   # lock 0.5 R
    (2.0, 0.65),   # lock 0.65 R
    (2.8, 0.80),   # lock 0.8 R
    (3.5, 0.85),   # lock 0.85 R
]

# Trailing struktural M15
STRUCT_TRAIL_LB       = 3      # lookback swing_pts saat trailing
STRUCT_TRAIL_BUF_PCT  = 0.002  # buffer 0.2% di luar swing agar tidak kena LS biasa
STRUCT_TRAIL_LOOKBACK = 60     # candle M15 untuk cari swing trailing

# Fibonacci extension TP (level 1.272 dan 1.618 dari impulse leg)
FIB_EXT_1 = 0.272   # 127.2%
FIB_EXT_2 = 0.618   # 161.8%


# =============================================================================
# UTILITY — Indikator teknikal
# =============================================================================

def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    d = s.diff()
    g = d.clip(lower=0).rolling(n).mean()
    lo = (-d.clip(upper=0)).rolling(n).mean()
    rs = g / lo.replace(0, np.nan)
    return 100 - 100 / (1 + rs)

def atr_fn(df: pd.DataFrame, n: int = 14) -> pd.Series:
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"] - df["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def build_df(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Tambahkan EMA, RSI, ATR, volume SMA ke DataFrame OHLCV."""
    if df is None or len(df) < 60:
        return None
    df = df.copy()
    df["ema9"] = ema(df["close"], 9)
    df["ema21"] = ema(df["close"], 21)
    df["ema50"] = ema(df["close"], 50)
    df["ema200"] = ema(df["close"], 200) if len(df) >= 200 else ema(df["close"], 50)
    df["rsi"] = rsi(df["close"])
    df["atr"] = atr_fn(df)
    df["vol_sma"] = df["volume"].rolling(20).mean()
    return df.dropna()

def swing_pts(df: pd.DataFrame, lb: int = 5):
    """Swing high & low — dipakai main.py untuk trailing."""
    sh, sl = [], []
    high = df["high"].values
    low = df["low"].values
    n = len(high)
    for i in range(lb, n - lb):
        window_h = high[max(0, i - lb): i + lb + 1]
        window_l = low[max(0, i - lb): i + lb + 1]
        if high[i] == window_h.max():
            sh.append(i)
        if low[i] == window_l.min():
            sl.append(i)
    return sh, sl

def _market_structure(df: pd.DataFrame, sh: list, sl: list) -> str:
    """HH+HL = bullish · LH+LL = bearish · else ranging."""
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

mkt_struct = _market_structure  # alias


# =============================================================================
# SMC / ICT DETECTORS (berdasarkan transkrip)
# =============================================================================

def is_zone_fresh(df: pd.DataFrame, top: float, bot: float,
                  formed_idx: int, end_idx: Optional[int] = None) -> bool:
    """True jika zona (OB/FVG) belum pernah ditembus setelah terbentuk."""
    if formed_idx is None or formed_idx + 2 >= len(df):
        return True
    start = formed_idx + 2
    end = end_idx if end_idx is not None else len(df) - 1
    if start >= end:
        return True
    sub = df.iloc[start:end]
    touched = ((sub["low"] <= top) & (sub["high"] >= bot)).any()
    return not bool(touched)

def fib_position(price: float, swing_low: float, swing_high: float) -> float:
    """Posisi harga dalam range 0–1, 0 = swing_low, 1 = swing_high."""
    rng = swing_high - swing_low
    if rng <= 0:
        return 0.5
    return max(0.0, min(1.0, (price - swing_low) / rng))

def is_in_ote(price: float, swing_low: float, swing_high: float,
              direction: str) -> bool:
    """
    OTE (Optimal Trade Entry) — zona 61.8%–78.6% retracement.
    Bull: fib_position antara 0.214 dan 0.382
    Bear: fib_position antara 0.618 dan 0.786
    """
    if swing_high <= swing_low:
        return False
    r = fib_position(price, swing_low, swing_high)
    if direction == "bull":
        return 0.214 <= r <= 0.382
    else:
        return 0.618 <= r <= 0.786

def detect_fvg(df: pd.DataFrame, direction: str, lb: int = 50) -> list:
    """
    Fair Value Gap (imbalance) — 3‑candle pattern.
    Hanya return zona yang FRESH.
    """
    sub = df.iloc[-lb:]
    base = len(df) - len(sub)
    out = []
    for i in range(len(sub) - 2):
        c0, c2 = sub.iloc[i], sub.iloc[i + 2]
        gap = None
        if direction == "bull" and c2["low"] > c0["high"]:
            gap = {"top": float(c2["low"]), "bot": float(c0["high"])}
        elif direction == "bear" and c2["high"] < c0["low"]:
            gap = {"top": float(c0["low"]), "bot": float(c2["high"])}
        if gap:
            gap["mid"] = (gap["top"] + gap["bot"]) / 2
            gap["idx"] = base + i + 2
            gap["is_fresh"] = is_zone_fresh(df, gap["top"], gap["bot"], gap["idx"])
            out.append(gap)
    fresh = [f for f in out if f["is_fresh"]]
    return fresh[-3:] if fresh else []

def detect_order_block(df: pd.DataFrame, direction: str, lb: int = 60,
                       sh: Optional[list] = None, sl: Optional[list] = None) -> list:
    """
    Order Block berkualitas (transkrip 1, 14, 26).
    Filter:
      - zona fresh
      - Fibonacci diskon (bull ≤ 0.618) / premium (bear ≥ 0.382)
      - kualitas: impulse kuat, FVG setelahnya, BOS, recency.
    Return list max 3, urut kualitas tertinggi.
    """
    is_demand = direction == "bull"
    sub = df.iloc[-lb:]
    base = len(df) - len(sub)
    avg_body = (sub["close"] - sub["open"]).abs().mean() or 1e-8

    # Fibonacci context
    fib_sh = float(df["high"].iloc[sh[-1]]) if (sh and len(sh) > 0) else None
    fib_sl = float(df["low"].iloc[sl[-1]]) if (sl and len(sl) > 0) else None

    # BOS global
    has_bos_global = False
    if sh and sl:
        if is_demand and len(sh) >= 2:
            has_bos_global = float(df["high"].iloc[-1]) > float(df["high"].iloc[sh[-2]])
        elif not is_demand and len(sl) >= 2:
            has_bos_global = float(df["low"].iloc[-1]) < float(df["low"].iloc[sl[-2]])

    zones = []
    for i in range(1, len(sub) - 3):
        c, nx = sub.iloc[i], sub.iloc[i + 1]
        # Pola trigger + impulse
        if is_demand:
            if not (c["close"] < c["open"] and nx["close"] > nx["open"]):
                continue
        else:
            if not (c["close"] > c["open"] and nx["close"] < nx["open"]):
                continue

        impulse_body = abs(nx["close"] - nx["open"])
        if impulse_body < avg_body * 1.2:
            continue

        ob_top = float(max(c["open"], c["close"]))
        ob_bot = float(min(c["open"], c["close"]))
        df_idx = base + i

        if not is_zone_fresh(df, ob_top, ob_bot, df_idx):
            continue

        q = 0
        if impulse_body >= avg_body * 1.5:
            q += 1
        if impulse_body >= avg_body * 2.5:
            q += 1

        # FVG setelah OB
        if i + 2 < len(sub):
            c2 = sub.iloc[i + 2]
            if is_demand and c2["low"] > c["high"]:
                q += 1
            elif not is_demand and c2["high"] < c["low"]:
                q += 1

        if has_bos_global:
            q += 1

        # Fibonacci diskon/premium (kunci dari transkrip 1)
        if fib_sh is not None and fib_sl is not None:
            ob_mid = (ob_top + ob_bot) / 2
            fib_r = fib_position(ob_mid, fib_sl, fib_sh)
            if is_demand and fib_r <= 0.618:
                q += 1
            elif not is_demand and fib_r >= 0.382:
                q += 1

        if df_idx >= len(df) - 20:
            q += 1

        if q >= 2:
            zones.append({
                "top": ob_top,
                "bot": ob_bot,
                "mid": (ob_top + ob_bot) / 2,
                "idx": df_idx,
                "quality": q,
                "has_bos": has_bos_global,
            })

    zones.sort(key=lambda z: (-z["quality"], -z["idx"]))
    return zones[:3]

def detect_choch(df: pd.DataFrame, sh: list, sl: list) -> dict:
    """Change of Character (transkrip 17, 19, 20)."""
    result = {"bullish_choch": False, "bearish_choch": False}
    if len(sh) < 2 or len(sl) < 2:
        return result
    close = float(df["close"].iloc[-1])
    prev_high = float(df["high"].iloc[sh[-2]])
    last_high = float(df["high"].iloc[sh[-1]])
    prev_low = float(df["low"].iloc[sl[-2]])
    last_low = float(df["low"].iloc[sl[-1]])
    struct = _market_structure(df, sh, sl)

    if struct == "bearish" and close > prev_low:
        result["bullish_choch"] = True
    if struct == "bullish" and close < prev_high:
        result["bearish_choch"] = True
    # Raw ChoCH
    if last_high > prev_high and last_low > prev_low and close > prev_low:
        result["bullish_choch"] = True
    if last_high < prev_high and last_low < prev_low and close < prev_low:
        result["bearish_choch"] = True
    return result

def detect_bos(df: pd.DataFrame, sh: list, sl: list) -> dict:
    """Break of Structure (transkrip 17)."""
    result = {"bullish_bos": False, "bearish_bos": False}
    if len(sh) < 2 or len(sl) < 2:
        return result
    if float(df["high"].iloc[-1]) > float(df["high"].iloc[sh[-2]]):
        result["bullish_bos"] = True
    if float(df["low"].iloc[-1]) < float(df["low"].iloc[sl[-2]]):
        result["bearish_bos"] = True
    return result

def detect_cisd(df: pd.DataFrame, lb: int = 8) -> dict:
    """Change In State of Delivery (transkrip 29)."""
    result = {"bullish_cisd": False, "bearish_cisd": False}
    if len(df) < lb + 1:
        return result
    sub = df.iloc[-lb:]
    opens = sub["open"].values
    closes = sub["close"].values
    n = len(closes)
    if n < 4:
        return result
    last_bull = closes[-1] > opens[-1]
    last_bear = closes[-1] < opens[-1]

    if last_bull:
        bear_run = 0
        for j in range(n - 2, -1, -1):
            if closes[j] < opens[j]:
                bear_run += 1
            else:
                break
        if bear_run >= 3:
            first_idx = n - 1 - bear_run
            if first_idx >= 0:
                bear_mid = (opens[first_idx] + closes[first_idx]) / 2
                if closes[-1] > bear_mid:
                    result["bullish_cisd"] = True
    elif last_bear:
        bull_run = 0
        for j in range(n - 2, -1, -1):
            if closes[j] > opens[j]:
                bull_run += 1
            else:
                break
        if bull_run >= 3:
            first_idx = n - 1 - bull_run
            if first_idx >= 0:
                bull_mid = (opens[first_idx] + closes[first_idx]) / 2
                if closes[-1] < bull_mid:
                    result["bearish_cisd"] = True
    return result

def detect_liquidity_sweep(df: pd.DataFrame, sh: list, sl: list,
                           direction: str) -> dict:
    """
    Liquidity Sweep (transkrip 4, 9, 15, 18).
    Valid: wick menembus swing, tapi close kembali di atas/bawah level.
    """
    result = {"type": "none", "level": None, "strength": 0}
    if direction == "bull" and sl:
        level = float(df["low"].iloc[sl[-1]])
        last_low = float(df["low"].iloc[-1])
        last_close = float(df["close"].iloc[-1])
        if last_low < level and last_close > level:
            depth = (level - last_low) / max(level, 1e-10)
            result = {
                "type": "sweep",
                "level": level,
                "strength": min(3, int(depth / 0.002) + 1),
            }
    elif direction == "bear" and sh:
        level = float(df["high"].iloc[sh[-1]])
        last_high = float(df["high"].iloc[-1])
        last_close = float(df["close"].iloc[-1])
        if last_high > level and last_close < level:
            depth = (last_high - level) / max(level, 1e-10)
            result = {
                "type": "sweep",
                "level": level,
                "strength": min(3, int(depth / 0.002) + 1),
            }
    return result

def detect_equal_highs_lows(df: pd.DataFrame, kind: str = "high",
                            lb: int = 80, tol: float = 0.003) -> list:
    """Equal Highs/Lows (transkrip 15, 24)."""
    sub = df.iloc[-lb:]
    vals = sub["high"] if kind == "high" else sub["low"]
    clusters, visited = [], set()
    for i in range(len(vals)):
        if i in visited:
            continue
        group = [float(vals.iloc[i])]
        for j in range(i + 1, len(vals)):
            if abs(vals.iloc[i] - vals.iloc[j]) / max(abs(float(vals.iloc[i])), 1e-10) < tol:
                group.append(float(vals.iloc[j]))
                visited.add(j)
        if len(group) >= 2:
            clusters.append(sum(group) / len(group))
    return sorted(clusters)

def detect_rsi_divergence(df: pd.DataFrame, direction: str, lb: int = 30) -> dict:
    """RSI Divergence (transkrip 19)."""
    result = {"bull_div": False, "bear_div": False, "strong": False}
    if len(df) < lb + 1 or "rsi" not in df.columns:
        return result
    sub = df.iloc[-lb:]
    price = sub["close"].values
    rsi_v = sub["rsi"].values
    n = len(price)
    lb3 = 3

    lows = [i for i in range(lb3, n - lb3)
            if price[i] == min(price[max(0, i - lb3): i + lb3 + 1])]
    if len(lows) >= 2:
        i1, i2 = lows[-2], lows[-1]
        if price[i2] < price[i1] and rsi_v[i2] > rsi_v[i1]:
            result["bull_div"] = True
            if rsi_v[i2] < 35:
                result["strong"] = True

    highs = [i for i in range(lb3, n - lb3)
             if price[i] == max(price[max(0, i - lb3): i + lb3 + 1])]
    if len(highs) >= 2:
        i1, i2 = highs[-2], highs[-1]
        if price[i2] > price[i1] and rsi_v[i2] < rsi_v[i1]:
            result["bear_div"] = True
            if rsi_v[i2] > 65:
                result["strong"] = True

    if direction == "bull" and not result["bull_div"]:
        return {"bull_div": False, "bear_div": False, "strong": False}
    if direction == "bear" and not result["bear_div"]:
        return {"bull_div": False, "bear_div": False, "strong": False}
    return result

def detect_failed_retest(df: pd.DataFrame, sh: list, sl: list,
                         atr: float) -> dict:
    """Failed retest (untuk kompatibilitas)."""
    result = {"failed_retest_sell": False, "failed_retest_buy": False}
    if len(df) < 3 or not sh or not sl:
        return result
    L = df.iloc[-1]
    P = df.iloc[-2]
    if len(sh) >= 2:
        res = float(df["high"].iloc[sh[-2]])
        if P["high"] >= res - atr * 0.5 and L["close"] < res - atr * 0.3 and L["close"] < L["open"]:
            result["failed_retest_sell"] = True
    if len(sl) >= 2:
        sup = float(df["low"].iloc[sl[-2]])
        if P["low"] <= sup + atr * 0.5 and L["close"] > sup + atr * 0.3 and L["close"] > L["open"]:
            result["failed_retest_buy"] = True
    return result


# =============================================================================
# SCORING — Confidence global tanpa bias sesi
# =============================================================================

def score_direction(df_h1: pd.DataFrame, df_m15: pd.DataFrame,
                    df_d1: Optional[pd.DataFrame] = None) -> Optional[dict]:
    """
    Tentukan arah dan skor confidence global (tanpa bias sesi).
    Faktor scoring (max ~165):
      - D1 bias (struktur/EMA)         +20
      - H1 structure                   +20
      - H1 ChoCH                       +10
      - H1 BOS                         +5
      - H1 EMA stack                   +10
      - M15 ChoCH                      +20
      - M15 BOS                        +10
      - M15 CISD                       +15
      - M15 Liquidity Sweep            +15
      - M15 OTE                        +10
      - M15 Fibonacci diskon/premium   +10
      - M15 RSI divergence             +10 (+5 if strong)
      - M15 failed retest              +10
    Penalti: jika setup M15 berlawanan dengan bias D1 → kurangi 50%.
    """
    h1 = build_df(df_h1)
    m15 = build_df(df_m15)
    if h1 is None or m15 is None:
        return None

    L1 = h1.iloc[-1]
    L15 = m15.iloc[-1]
    atr = max(float(L15["atr"]),
              float(L1["atr"]) / 4,
              float(L15["close"]) * 0.003)

    sh1, sl1 = swing_pts(h1, lb=5)
    sh15, sl15 = swing_pts(m15, lb=5)
    struct_h1 = _market_structure(h1, sh1, sl1)

    # ── D1 bias ──────────────────────────────────────────────────
    d1_bias = "neutral"
    try:
        if df_d1 is not None and len(df_d1) >= 65:
            d1 = build_df(df_d1)
        else:
            d1 = build_df(
                df_h1.resample("1D").agg(
                    {"open": "first", "high": "max", "low": "min",
                     "close": "last", "volume": "sum"}
                ).dropna()
            )
        if d1 is not None and len(d1) >= 10:
            LD = d1.iloc[-1]
            shd, sld = swing_pts(d1, lb=3)
            sd1 = _market_structure(d1, shd, sld)
            bull_d1 = sd1 == "bullish" or (LD["ema9"] > LD["ema21"] > LD["ema50"])
            bear_d1 = sd1 == "bearish" or (LD["ema9"] < LD["ema21"] < LD["ema50"])
            if bull_d1:
                d1_bias = "bullish"
            elif bear_d1:
                d1_bias = "bearish"
    except Exception:
        pass

    # ── H1 indikator ─────────────────────────────────────────────
    ema_h1_bull = L1["ema9"] > L1["ema21"] > L1["ema50"]
    ema_h1_bear = L1["ema9"] < L1["ema21"] < L1["ema50"]
    choch_h1 = detect_choch(h1, sh1, sl1)
    bos_h1 = detect_bos(h1, sh1, sl1)

    # ── M15 indikator ─────────────────────────────────────────────
    choch_m15 = detect_choch(m15, sh15, sl15)
    bos_m15 = detect_bos(m15, sh15, sl15)
    cisd_m15 = detect_cisd(m15, lb=8)
    liq_bull = detect_liquidity_sweep(m15, sh15, sl15, "bull")
    liq_bear = detect_liquidity_sweep(m15, sh15, sl15, "bear")
    fr_m15 = detect_failed_retest(m15, sh15, sl15, atr)

    # ── Fibonacci M15 context ─────────────────────────────────────
    fib_sh = float(m15["high"].iloc[sh15[-1]]) if sh15 else None
    fib_sl = float(m15["low"].iloc[sl15[-1]]) if sl15 else None
    fib_r = fib_position(float(L15["close"]), fib_sl or 0, fib_sh or 1) \
            if (fib_sh and fib_sl) else 0.5

    ote_bull = is_in_ote(float(L15["close"]), fib_sl or 0, fib_sh or 1, "bull") if (fib_sh and fib_sl) else False
    ote_bear = is_in_ote(float(L15["close"]), fib_sl or 0, fib_sh or 1, "bear") if (fib_sh and fib_sl) else False

    in_discount = fib_r < 0.45
    in_premium = fib_r > 0.55

    rdiv_bull = detect_rsi_divergence(m15, "bull", lb=30)
    rdiv_bear = detect_rsi_divergence(m15, "bear", lb=30)

    # ── Score BULL ───────────────────────────────────────────────
    bull = 0
    if d1_bias == "bullish":                    bull += 20
    if struct_h1 == "bullish":                  bull += 20
    elif struct_h1 == "ranging":                bull += 5
    if choch_h1["bullish_choch"]:               bull += 10
    if bos_h1["bullish_bos"]:                   bull += 5
    if ema_h1_bull:                             bull += 10
    if choch_m15["bullish_choch"]:              bull += 20
    if bos_m15["bullish_bos"]:                  bull += 10
    if cisd_m15["bullish_cisd"]:                bull += 15
    if liq_bull["type"] == "sweep":             bull += 15
    if ote_bull:                                bull += 10
    if in_discount:                             bull += 10
    if rdiv_bull.get("bull_div"):
        bull += 10 + (5 if rdiv_bull.get("strong") else 0)
    if fr_m15["failed_retest_buy"]:             bull += 10

    # ── Score BEAR ───────────────────────────────────────────────
    bear = 0
    if d1_bias == "bearish":                    bear += 20
    if struct_h1 == "bearish":                  bear += 20
    elif struct_h1 == "ranging":                bear += 5
    if choch_h1["bearish_choch"]:               bear += 10
    if bos_h1["bearish_bos"]:                   bear += 5
    if ema_h1_bear:                             bear += 10
    if choch_m15["bearish_choch"]:              bear += 20
    if bos_m15["bearish_bos"]:                  bear += 10
    if cisd_m15["bearish_cisd"]:                bear += 15
    if liq_bear["type"] == "sweep":             bear += 15
    if ote_bear:                                bear += 10
    if in_premium:                              bear += 10
    if rdiv_bear.get("bear_div"):
        bear += 10 + (5 if rdiv_bear.get("strong") else 0)
    if fr_m15["failed_retest_sell"]:            bear += 10

    # ── Penalti cross-HTF ─────────────────────────────────────────
    if d1_bias == "bullish" and bear > bull:
        bear = int(bear * 0.5)
    elif d1_bias == "bearish" and bull > bear:
        bull = int(bull * 0.5)

    direction = "bull" if bull >= bear else "bear"
    raw = bull if direction == "bull" else bear
    MAX_SCORE = 165
    confidence = min(int(raw / MAX_SCORE * 100), 99)

    return {
        "direction": direction,
        "confidence": confidence,
        "price": float(L15["close"]),
        "atr": atr,
        "struct_h1": struct_h1,
        "d1_bias": d1_bias,
        "choch_m15": choch_m15,
        "choch_h1": choch_h1,
        "cisd_m15": cisd_m15,
        "bos_m15": bos_m15,
        "bos_h1": bos_h1,
        "failed_retest": fr_m15,
        "liquidity_bull": liq_bull,
        "liquidity_bear": liq_bear,
        "sh15": sh15, "sl15": sl15,
        "sh1": sh1, "sl1": sl1,
        "fib_r": round(fib_r, 3),
        "ote_bull": ote_bull,
        "ote_bear": ote_bear,
    }


# =============================================================================
# STEP 1 — ENTRY CANDIDATES
# =============================================================================

def _collect_entry_candidates(m15: pd.DataFrame, h1: pd.DataFrame,
                              direction: str, current_price: float,
                              atr: float, score_ctx: dict) -> list:
    """
    Kumpulkan kandidat entry berdasarkan SMC/ICT.
    Prioritas (score tertinggi):
      1. OB + Liquidity Sweep (12–15)
      2. OB + ChoCH (9–12)
      3. OB + OTE (8–10)
      4. OB saja (5–8)
      5. FVG + CISD + LiqSweep (6–9)
      6. FVG saja (3–5)
      7. Equal Highs/Lows (2–4)
      8. Market entry fallback (1)
    """
    up = direction == "bull"
    cands = []

    liq = score_ctx.get("liquidity_bull" if up else "liquidity_bear", {})
    choch = score_ctx.get("choch_m15", {})
    cisd = score_ctx.get("cisd_m15", {})

    choch_ok = choch.get("bullish_choch") if up else choch.get("bearish_choch")
    cisd_ok = cisd.get("bullish_cisd") if up else cisd.get("bearish_cisd")
    liq_ok = liq.get("type") == "sweep"

    fib_sh = float(m15["high"].iloc[score_ctx["sh15"][-1]]) if score_ctx.get("sh15") else None
    fib_sl = float(m15["low"].iloc[score_ctx["sl15"][-1]]) if score_ctx.get("sl15") else None

    # ── Order Block ──────────────────────────────────────────────────
    obs = detect_order_block(m15, direction, lb=60,
                             sh=score_ctx.get("sh15", []),
                             sl=score_ctx.get("sl15", []))
    for z in obs:
        entry_pt = float(z["top"]) if up else float(z["bot"])
        invalid_pt = float(z["bot"]) if up else float(z["top"])

        if up and current_price < z["bot"] * 0.99:
            continue
        if not up and current_price > z["top"] * 1.01:
            continue

        sc = 3 + z["quality"]  # base 3 + quality

        if liq_ok:
            sweep_lev = liq.get("level", 0)
            if up and entry_pt >= float(sweep_lev) * 0.995:
                sc += 3
            elif not up and entry_pt <= float(sweep_lev) * 1.005:
                sc += 3
        if choch_ok:
            sc += 2
        if fib_sh and fib_sl and is_in_ote(entry_pt, fib_sl, fib_sh, direction):
            sc += 1

        cands.append({
            "price": round(entry_pt, 8),
            "invalid": round(invalid_pt, 8),
            "label": "ob",
            "score": sc,
        })

    # ── FVG ──────────────────────────────────────────────────────────
    fvgs = detect_fvg(m15, direction, lb=50)
    for f in fvgs:
        if not f["is_fresh"]:
            continue
        entry_pt = f["mid"]
        invalid_pt = f["top"] if up else f["bot"]
        sc = 3
        if cisd_ok: sc += 2
        if liq_ok: sc += 2
        if choch_ok: sc += 1
        cands.append({
            "price": round(entry_pt, 8),
            "invalid": round(invalid_pt, 8),
            "label": "fvg",
            "score": sc,
        })

    # ── Equal Highs/Lows ─────────────────────────────────────────────
    eqs = detect_equal_highs_lows(m15, "low" if up else "high", lb=80)
    for eq in eqs[:2]:
        # ── Proximity filter: EQ entry harus REACHABLE dari harga sekarang.
        #
        # Bug asal: tidak ada filter proximity → EQ yang harganya sudah
        # "tersapu" (liquidity sweep) tetap masuk sebagai kandidat entry.
        # Akibatnya SELL limit dipasang DI BAWAH harga pasar → Binance langsung
        # fill di harga pasar (slippage besar) → actual_entry > SL → geometri
        # rusak → auto-out terpicu.
        #
        # SELL (not up): entry di equal HIGH → level harus ≥ current_price
        #   supaya SELL limit menunggu harga naik ke sana, bukan fill sekarang.
        #   Toleransi 0.3%: jika eq < current_price * 0.997 → sudah tersapu.
        #
        # BUY (up): entry di equal LOW → level harus ≤ current_price
        #   supaya BUY limit menunggu harga turun ke sana, bukan fill sekarang.
        #   Toleransi 0.3%: jika eq > current_price * 1.003 → sudah tersapu.
        if not up and float(eq) < current_price * 0.997:
            continue   # EQ high sudah di bawah harga pasar → skip
        if up and float(eq) > current_price * 1.003:
            continue   # EQ low sudah di atas harga pasar → skip

        invalid_pt = eq - atr * 0.8 if up else eq + atr * 0.8
        sc = 2
        if liq_ok: sc += 1
        cands.append({
            "price": round(float(eq), 8),
            "invalid": round(float(invalid_pt), 8),
            "label": "eq",
            "score": sc,
        })

    # ── Market entry fallback ──────────────────────────────────────
    if not cands:
        invalid_pt = current_price - atr * 1.2 if up else current_price + atr * 1.2
        cands.append({
            "price": round(current_price, 8),
            "invalid": round(float(invalid_pt), 8),
            "label": "market",
            "score": 1,
        })

    cands.sort(key=lambda c: -c["score"])
    return cands


# =============================================================================
# STEP 2 — SL STRUKTURAL (anti‑Liquidity Sweep)
# =============================================================================

def _compute_sl(m15: pd.DataFrame, h1: pd.DataFrame, direction: str,
                entry: float, atr: float, liq_sweep: dict,
                invalid_level: Optional[float] = None) -> Tuple[float, float]:
    """
    Hitung SL yang tepat secara struktural, ditempatkan sedemikian rupa
    sehingga jika tersentuh berarti arah benar‑benar salah (bukan LS biasa).
    Buffer anti‑Liquidity Sweep = 0.5 ATR.
    Pilih kandidat dengan prioritas: ob_invalid > struct_h1 > ls_level > struct_m15,
    dan di antara prioritas yang sama pilih yang LEBIH JAUH dari entry
    (lebih tahan noise).
    """
    up = direction == "bull"
    ls_buffer = atr * 0.5
    min_risk = atr * 1.0
    max_risk = atr * 4.5

    cands = []

    # 1. invalid level dari OB/FVG
    if invalid_level is not None:
        sl_raw = invalid_level + (-ls_buffer if up else ls_buffer)
        risk = abs(sl_raw - entry)
        if min_risk <= risk <= max_risk:
            cands.append(("ob_invalid", sl_raw, risk))

    # 2. M15 swing (prioritas lebih rendah karena lebih rentan noise)
    sh15, sl15 = swing_pts(m15, lb=3)
    if up and sl15:
        struct_low = float(m15["low"].iloc[sl15[-1]])
        if struct_low < entry:
            sl_raw = struct_low - ls_buffer
            risk = abs(sl_raw - entry)
            if min_risk <= risk <= max_risk:
                cands.append(("struct_m15", sl_raw, risk))
    elif not up and sh15:
        struct_high = float(m15["high"].iloc[sh15[-1]])
        if struct_high > entry:
            sl_raw = struct_high + ls_buffer
            risk = abs(sl_raw - entry)
            if min_risk <= risk <= max_risk:
                cands.append(("struct_m15", sl_raw, risk))

    # 3. level yang disweep
    if liq_sweep and liq_sweep.get("type") == "sweep" and liq_sweep.get("level"):
        lev = float(liq_sweep["level"])
        if up and lev < entry:
            sl_raw = lev - ls_buffer
            risk = abs(sl_raw - entry)
            if min_risk <= risk <= max_risk:
                cands.append(("ls_level", sl_raw, risk))
        elif not up and lev > entry:
            sl_raw = lev + ls_buffer
            risk = abs(sl_raw - entry)
            if min_risk <= risk <= max_risk:
                cands.append(("ls_level", sl_raw, risk))

    # 4. H1 swing (lebih lebar, lebih tahan noise)
    sh1, sl1 = swing_pts(h1, lb=5)
    if up and sl1:
        h1_low = float(h1["low"].iloc[sl1[-1]])
        if h1_low < entry:
            sl_raw = h1_low - ls_buffer
            risk = abs(sl_raw - entry)
            if min_risk <= risk <= max_risk:
                cands.append(("struct_h1", sl_raw, risk))
    elif not up and sh1:
        h1_high = float(h1["high"].iloc[sh1[-1]])
        if h1_high > entry:
            sl_raw = h1_high + ls_buffer
            risk = abs(sl_raw - entry)
            if min_risk <= risk <= max_risk:
                cands.append(("struct_h1", sl_raw, risk))

    if cands:
        # Prioritas: ob_invalid (paling presisi), struct_h1 (paling tahan),
        # ls_level, struct_m15. Dalam prioritas sama, pilih risk terbesar.
        _PRIO = {"ob_invalid": 0, "struct_h1": 1, "ls_level": 2, "struct_m15": 3}
        cands.sort(key=lambda x: (_PRIO.get(x[0], 9), -x[2]))
        _, sl_price, risk = cands[0]
        return sl_price, risk

    # Fallback ATR
    sl_price = entry + (-min_risk if up else min_risk)
    return sl_price, min_risk


# =============================================================================
# STEP 3 — TP POOL DAN SELEKSI (dengan ekstensi jika RR < 2.0)
# =============================================================================

def _build_tp_pool(h1: pd.DataFrame, m15: pd.DataFrame, direction: str,
                   entry: float, atr: float,
                   sh1: list, sl1: list, sh15: list, sl15: list) -> list:
    """
    Bangun pool target TP dari berbagai sumber, terurut terdekat ke terjauh.
    Sumber (tier):
      1: EQ M15
      2: OB H1 (arah berlawanan)
      3: FVG H1
      4: Swing H1
      5: EQ H1
      6: Fibonacci extension (1.272, 1.618)
      7: Fibonacci extension (2.0, 2.414) untuk ekstensi
    """
    up = direction == "bull"
    sgn = 1 if up else -1
    pool = []

    # Tier 1: EQ M15
    eqs_m15 = detect_equal_highs_lows(m15, "high" if up else "low", lb=80)
    for v in eqs_m15:
        if sgn * (v - entry) > atr * 0.3:
            pool.append(("eq_m15", v, 1))

    # Tier 2: OB H1 (arah berlawanan = area resistance/support)
    opp_dir = "bear" if up else "bull"
    obs_h1_opp = detect_order_block(h1, opp_dir, lb=80, sh=sh1, sl=sl1)
    for z in obs_h1_opp:
        edge = float(z["bot"]) if up else float(z["top"])
        if sgn * (edge - entry) > atr * 0.5:
            pool.append(("ob_h1", edge, 2))

    # Tier 3: FVG H1
    fvgs_h1 = detect_fvg(h1, opp_dir, lb=60)
    for f in fvgs_h1:
        if sgn * (f["mid"] - entry) > atr * 0.5:
            pool.append(("fvg_h1", f["mid"], 3))

    # Tier 4: Swing H1
    sw_vals = ([float(h1["high"].iloc[i]) for i in sh1] if up
               else [float(h1["low"].iloc[i]) for i in sl1])
    for v in sw_vals:
        if sgn * (v - entry) > atr * 1.0:
            pool.append(("sw_h1", v, 4))

    # Tier 5: EQ H1
    eqs_h1 = detect_equal_highs_lows(h1, "high" if up else "low", lb=100)
    for v in eqs_h1:
        if sgn * (v - entry) > atr * 0.8:
            pool.append(("eq_h1", v, 5))

    # Tier 6 & 7: Fibonacci extensions
    if sh1 and sl1:
        sh_val = float(h1["high"].iloc[sh1[-1]])
        sl_val = float(h1["low"].iloc[sl1[-1]])
        leg = sh_val - sl_val
        if leg > 0:
            exts = [
                (FIB_EXT_1, "fib127", 6),
                (FIB_EXT_2, "fib162", 6),
                (1.0, "fib200", 7),
                (1.414, "fib241", 7),
            ]
            for ext, lbl, tier in exts:
                tp_v = (sh_val + leg * ext) if up else (sl_val - leg * ext)
                if sgn * (tp_v - entry) > atr * 0.5:
                    pool.append((lbl, tp_v, tier))

    # Sort by distance from entry (terdekat dulu)
    pool.sort(key=lambda x: abs(x[1] - entry))
    return pool

def _select_tp(pool: list, entry: float, risk: float,
               direction: str) -> Tuple[Optional[float], Optional[str], Optional[float]]:
    """
    Pilih TP terbaik dengan logika:
      1. Cari target RR 2.0–4.0 → ambil tier terkecil, RR paling dekat ke 2.0.
      2. Jika tidak ada dalam rentang ideal:
         - Jika ada target RR > 4.0 → CAP ke 4.0 (sesuai instruksi).
         - Jika semua target RR < 2.0 → coba cari yang lebih jauh
           (ekstensi Fibonacci) → jika tetap tidak ada, return None.
    """
    if not pool:
        return None, None, None

    sgn = 1 if direction == "bull" else -1
    qualified = []   # RR 2.0–4.0
    below_min = []   # RR < 2.0
    above_max = []   # RR > 4.0

    for lbl, v, tier in pool:
        if sgn * (v - entry) <= 0:
            continue
        rr = abs(v - entry) / max(risk, 1e-10)
        if MIN_RR <= rr <= MAX_RR:
            qualified.append((lbl, v, tier, rr))
        elif rr < MIN_RR:
            below_min.append((lbl, v, tier, rr))
        else:
            above_max.append((lbl, v, tier, rr))

    # 1. Ada target di zona ideal
    if qualified:
        best = min(qualified, key=lambda x: (x[2], x[3]))  # tier, lalu RR terkecil
        return round(best[1], 8), best[0], round(best[3], 2)

    # 2. Ada target terlalu jauh → cap ke 4.0
    if above_max:
        best = min(above_max, key=lambda x: x[3])
        capped = entry + sgn * risk * MAX_RR
        return round(capped, 8), best[0] + "_capped", MAX_RR

    # 3. Semua target terlalu dekat → sinyal tidak layak
    #    (tidak ada target yang bisa diekstensi karena pool sudah mencakup
    #     Fibonacci extension dan yang lain)
    return None, None, None


# =============================================================================
# FUNGSI UTAMA — Dipanggil oleh main.py
# =============================================================================

def full_analyze(df_h1: pd.DataFrame, df_m15: pd.DataFrame,
                 df_d1: Optional[pd.DataFrame] = None,
                 symbol: Optional[str] = None) -> Optional[dict]:
    """
    Analisa penuh satu koin: Entry → SL → TP.
    """
    try:
        if df_h1 is None or df_m15 is None or df_h1.empty or df_m15.empty:
            return None

        if symbol:
            log.info(f"[{symbol}] h1={len(df_h1)} m15={len(df_m15)}")

        score = score_direction(df_h1, df_m15, df_d1)
        if score is None:
            if symbol:
                log.debug(f"[{symbol}] score_direction=None (data kurang)")
            return None

        direction = score["direction"]
        cur_price = score["price"]
        atr = score["atr"]
        confidence = score["confidence"]
        up = direction == "bull"

        if symbol:
            log.info(
                f"[{symbol}] dir={direction} conf={confidence}% "
                f"struct_h1={score['struct_h1']} d1={score['d1_bias']}"
            )

        h1 = build_df(df_h1)
        m15 = build_df(df_m15)
        if h1 is None or m15 is None:
            return None

        # ── STEP 1: ENTRY ────────────────────────────────────────
        cands = _collect_entry_candidates(m15, h1, direction, cur_price, atr, score)
        if not cands:
            if symbol:
                log.debug(f"[{symbol}] no entry candidates")
            return None

        best = cands[0]
        entry = best["price"]
        entry_lbl = best["label"]
        invalid = best["invalid"]

        if entry_lbl == "market" and confidence < 70:
            if symbol:
                log.debug(f"[{symbol}] market entry ditolak (conf={confidence}<70)")
            return None

        if symbol:
            log.info(f"[{symbol}] ENTRY={entry:.6f} label={entry_lbl} score={best['score']}")

        # ── STEP 2: SL ──────────────────────────────────────────
        liq_ctx = score["liquidity_bull"] if up else score["liquidity_bear"]
        sl_price, risk = _compute_sl(m15, h1, direction, entry, atr,
                                     liq_ctx, invalid)

        if up and sl_price >= entry:
            if symbol:
                log.debug(f"[{symbol}] SL={sl_price:.6f} ≥ entry={entry:.6f}, skip")
            return None
        if not up and sl_price <= entry:
            if symbol:
                log.debug(f"[{symbol}] SL={sl_price:.6f} ≤ entry={entry:.6f}, skip")
            return None
        if risk <= 0:
            return None

        # ══ FILTER KRITIS #1 — SL vs HARGA PASAR SEKARANG ═══════════════════
        # Ini adalah filter paling penting untuk mencegah auto-out.
        # SL dihitung dari struktur (swing H1/M15), tapi harga bisa sudah
        # bergerak sehingga current_price sudah melewati SL sebelum order
        # bahkan terpasang. Kalau lolos di sini → order akan LANGSUNG auto-out
        # setelah fill karena "harga sudah melewati SL".
        #
        #   BUY : current_price harus di ATAS SL  (kalau sudah di bawah → skip)
        #   SELL: current_price harus di BAWAH SL (kalau sudah di atas → skip)
        if up and cur_price <= sl_price:
            if symbol:
                log.debug(
                    f"[{symbol}] DITOLAK (filter#1): BUY SL={sl_price:.6g} sudah "
                    f"ditembus current={cur_price:.6g} — akan auto-out, skip"
                )
            return None
        if not up and cur_price >= sl_price:
            if symbol:
                log.debug(
                    f"[{symbol}] DITOLAK (filter#1): SELL SL={sl_price:.6g} sudah "
                    f"ditembus current={cur_price:.6g} — akan auto-out, skip"
                )
            return None

        # ══ FILTER KRITIS #2 — ENTRY vs HARGA PASAR (LIMIT ORDER REACHABILITY) ══
        # Limit order harus MENUNGGU harga datang ke level entry, bukan
        # langsung fill di harga pasar. Kalau entry sudah "di belakang" harga,
        # Binance Futures fill langsung di harga pasar → actual_entry ≠ entry
        # → SL/TP dihitung untuk entry lama → geometri rusak → auto-out.
        #
        #   SELL limit: entry_target harus ≥ current_price
        #     (harga perlu NAIK dulu ke entry, baru fill)
        #     Kalau entry < current * 0.995 → limit sell sudah di bawah market
        #     → fill sekarang di harga pasar → actual_entry > SL → geometri rusak
        #
        #   BUY limit: entry_target harus ≤ current_price
        #     (harga perlu TURUN dulu ke entry, baru fill)
        #     Kalau entry > current * 1.005 → limit buy sudah di atas market
        #     → fill sekarang di harga pasar → actual_entry < SL → geometri rusak
        #
        #   Toleransi 0.5% untuk pembulatan tick / lag data minor.
        if not up and entry < cur_price * 0.995:
            if symbol:
                log.debug(
                    f"[{symbol}] DITOLAK (filter#2): SELL entry={entry:.6g} di bawah "
                    f"current={cur_price:.6g} — limit akan fill di harga salah, skip"
                )
            return None
        if up and entry > cur_price * 1.005:
            if symbol:
                log.debug(
                    f"[{symbol}] DITOLAK (filter#2): BUY entry={entry:.6g} di atas "
                    f"current={cur_price:.6g} — limit akan fill di harga salah, skip"
                )
            return None

        if symbol:
            log.info(f"[{symbol}] SL={sl_price:.6f} risk={risk:.6f}")

        # ── STEP 3: TP ──────────────────────────────────────────
        sh1 = score.get("sh1", [])
        sl1 = score.get("sl1", [])
        sh15 = score.get("sh15", [])
        sl15 = score.get("sl15", [])

        tp_pool = _build_tp_pool(h1, m15, direction, entry, atr,
                                 sh1, sl1, sh15, sl15)
        tp_price, tp_lbl, rr = _select_tp(tp_pool, entry, risk, direction)

        if tp_price is None:
            # Fallback: buat TP minimal 2.0
            sgn = 1 if up else -1
            tp_price = entry + sgn * risk * MIN_RR
            tp_lbl = "fallback_rr2"
            rr = MIN_RR

        if symbol:
            log.info(f"[{symbol}] TP={tp_price:.6f} label={tp_lbl} RR={rr:.2f}")

        if up and cur_price >= tp_price:
            if symbol:
                log.debug(f"[{symbol}] TP sudah lewat (price={cur_price:.6f})")
            return None
        if not up and cur_price <= tp_price:
            if symbol:
                log.debug(f"[{symbol}] TP sudah lewat")
            return None

        if rr < MIN_RR:
            if symbol:
                log.debug(f"[{symbol}] RR={rr:.2f} < {MIN_RR}, skip")
            return None

        rsi_val = round(float(m15["rsi"].iloc[-1]), 1)

        return {
            "symbol": symbol,
            "original_dir": direction,
            "decision": "BUY" if up else "SELL",
            "confidence": confidence,
            "price": cur_price,
            "entry": round(entry, 8),
            "entry_label": entry_lbl,
            "sl": round(sl_price, 8),
            "tp": round(tp_price, 8),
            "rr": rr,
            "atr": round(atr, 8),
            "rsi": rsi_val,
            "struct_h1": score["struct_h1"],
            "d1_bias": score.get("d1_bias", "neutral"),
            "choch_m15": score["choch_m15"],
            "choch_h1": score["choch_h1"],
            "cisd_m15": score["cisd_m15"],
            "failed_retest": score.get("failed_retest", {}),
            "tp_sl_reason": (
                f"Entry@{entry:.5g}({entry_lbl}) | "
                f"SL@{sl_price:.5g}(struct) | "
                f"TP@{tp_price:.5g}({tp_lbl}) | RR={rr:.2f}"
            ),
        }

    except Exception as e:
        if symbol:
            log.error(f"[full_analyze] {symbol}: {e}", exc_info=True)
        return None

def get_best_signal(candidates: list) -> Optional[dict]:
    """Pilih sinyal terbaik dari list kandidat."""
    if not candidates:
        return None
    label_bonus = {"ob": 4, "fvg": 2, "eq": 1, "market": 0}

    def _rank(sig):
        bonus = label_bonus.get(sig.get("entry_label", ""), 0)
        return sig["confidence"] + bonus + sig.get("rr", 0) * 0.5

    return max(candidates, key=_rank)


# =============================================================================
# VALIDASI PRE-ORDER & KOREKSI GEOMETRY (dipanggil oleh main.py)
# =============================================================================

def validate_and_adjust_geometry(
    entry: float, sl: float, tp: float,
    current_price: float, atr: float,
    direction: str,
) -> Optional[dict]:
    """
    Validasi dan (jika perlu) koreksi geometri entry/SL/TP sebelum order dipasang
    atau setelah order terisi di harga yang berbeda dari target.

    Mengapa fungsi ini penting:
    ─────────────────────────────────────────────────────────────────────────────
    Sinyal dihitung pada waktu T. Saat order terpasang (T+beberapa detik/menit),
    harga pasar bisa sudah bergerak — khususnya:

      • Kasus "geometri invalid setelah order terisi":
        SELL limit di entry_target, tapi actual_fill = harga pasar (lebih tinggi
        dari entry_target karena market sudah di atas limit sell). Akibatnya
        actual_fill > SL → geometri rusak.

      • Kasus "harga sudah melewati SL setelah order terisi":
        Harga sedikit melampaui SL (Liquidity Sweep) sesaat setelah fill.
        SL yang lama tidak relevan lagi, tapi kalau ini cuma sweep (< 3×ATR)
        masih bisa diselamatkan dengan relokasi SL ke luar area sweep.

    Logika:
    ─────────────────────────────────────────────────────────────────────────────
    1. Cek geometri dasar: SL di sisi yang benar dari entry, TP di sisi lain.
    2. Cek SL belum ditembus current_price.
    3. Jika SL ditembus tapi sweep-depth ≤ 3×ATR → relokasi SL ke luar sweep
       (current_price + 0.5×ATR buffer).
    4. Jika setelah relokasi SL geometri masih rusak → coba ganti entry ke
       current_price (harga sekarang = zona entry baru).
    5. Cek RR ≥ MIN_RR setelah semua penyesuaian.

    Return:
      dict  {entry, sl, tp, rr, adjusted} jika valid / bisa diselamatkan
      None  jika tidak bisa diperbaiki → TOLAK sinyal / auto-out
    """
    up = direction == "bull"
    ls_buf = atr * 0.5

    def _geo_ok(e: float, s: float, t: float) -> bool:
        return (s < e < t) if up else (t < e < s)

    def _rr(e: float, s: float, t: float) -> float:
        return abs(t - e) / max(abs(e - s), 1e-10)

    sl_breached = (current_price <= sl) if up else (current_price >= sl)

    # ─── Kasus 1: sudah valid ────────────────────────────────────────────────
    if _geo_ok(entry, sl, tp) and not sl_breached:
        rr = _rr(entry, sl, tp)
        if rr < MIN_RR:
            return None
        return {"entry": entry, "sl": sl, "tp": tp, "rr": round(rr, 2), "adjusted": False}

    # ─── Kasus 2: SL ditembus → cek apakah Liquidity Sweep ──────────────────
    new_sl = sl
    adjusted = False
    if sl_breached:
        sweep_depth = abs(current_price - sl)
        if sweep_depth > atr * 3.0:
            # Terlalu jauh — bukan sweep biasa → sinyal benar-benar invalid
            log.debug(
                f"[validate_geo] SL ditembus dalam ({sweep_depth:.6g} > 3×ATR {atr*3:.6g}) "
                f"— sinyal ditolak"
            )
            return None
        # Relokasi SL ke luar area sweep + buffer anti-re-sweep
        new_sl = current_price + (ls_buf if not up else -ls_buf)
        adjusted = True
        log.info(
            f"[validate_geo] Liquidity Sweep terdeteksi (depth={sweep_depth:.6g} ≤ 3×ATR) "
            f"— SL direlokasi {sl:.6g} → {new_sl:.6g}"
        )

    # ─── Kasus 3: geometri setelah SL baru ──────────────────────────────────
    if not _geo_ok(entry, new_sl, tp):
        # Geometri masih rusak (entry di luar range SL–TP).
        # Coba ganti entry ke current_price — harga sekarang sudah masuk zona.
        new_entry = round(current_price, 8)
        if _geo_ok(new_entry, new_sl, tp):
            log.info(
                f"[validate_geo] Entry digeser ke current_price "
                f"{entry:.6g} → {new_entry:.6g} supaya geometri valid"
            )
            entry = new_entry
            adjusted = True
        else:
            log.debug(
                f"[validate_geo] Geometri tetap rusak setelah SL relokasi & entry shift "
                f"(entry={new_entry:.6g}, sl={new_sl:.6g}, tp={tp:.6g}) — ditolak"
            )
            return None

    rr = _rr(entry, new_sl, tp)
    if rr < MIN_RR:
        log.debug(
            f"[validate_geo] RR={rr:.2f} < MIN_RR={MIN_RR} setelah koreksi — ditolak"
        )
        return None

    return {
        "entry": round(entry, 8),
        "sl":    round(new_sl, 8),
        "tp":    tp,
        "rr":    round(rr, 2),
        "adjusted": adjusted,
    }