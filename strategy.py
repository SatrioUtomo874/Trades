#!/usr/bin/env python3
"""Tactical V10 E_GE2_ATR — state machine adapter untuk main.py.

Menggunakan implementasi state machine dan parameter eksperimen V10 yang sama
dengan backtest. main.py memasok candle Binance M5 tertutup melalui provider
injeksi; modul ini tidak melakukan HTTP/WebSocket dan tidak memiliki cache data.

Pipeline: D1 bias -> AOI FVG D1 -> POI FVG H4 (LONG) / OB H4 (SHORT)
-> literal MSS M15 -> BOS M5 -> entry limit FVG/OB sesuai konfigurasi arah.
"""
import argparse
import hashlib
import json
import pickle
from collections import deque
from pathlib import Path
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view as swv

P = dict(NP=2, RR=2.0, COST=8e-4, SL_BUF=0.0,
         AOI_LB=30, AOI_AGE=40, AOI_MAX_ATR=0.0, POI_LB=30, POI_WAIT=2880,
         MSS_WIN=576, BOS_WIN=144, PEND_WIN=72, HOLD=2016,
         MSS_MODE='swing', ENTRY='prox', POI_MODE='fvg_or_ob',
         POI_MODE_LONG=None, POI_MODE_SHORT=None,
         SL_MODE='m15_12', ENTRY_ZONE='fvg_or_ob', ENTRY_ZONE_LONG=None, ENTRY_ZONE_SHORT=None,
         SHORT_BIAS_MODE='d1_only', LONG_RECOVERY_MODE='none', LONG_RECOVERY_MIN24H_PCT=-1.0,
         LONG_QUALITY_MODE='none', LONG_BIAS_AGE_MIN_DAYS=7.0, LONG_BIAS_AGE_MAX_DAYS=14.0,
         LONG_PROTECTED_SWING_MIN_ATR=1.0, LONG_PROTECTED_SWING_MAX_ATR=None, FRESH_ENTRY=True, SL_BUF_ATR=0.0,
         REQUIRE_POI_RETOUCH_AFTER_FAILED_ATTEMPT=False)
STEP, DAY = 300000, 86400000


def poi_mode_for_direction(params, dr):
    """Resolve H4 POI qualification after D1 bias has selected a direction.

    dr=+1 is LONG, dr=-1 is SHORT. A None direction override falls back to
    the legacy shared POI_MODE so the rest of V4 experiments remain compatible.
    """
    key = 'POI_MODE_LONG' if dr == 1 else 'POI_MODE_SHORT'
    mode = params.get(key)
    if mode in ('fvg_only', 'ob_only', 'fvg_or_ob'):
        return mode
    return params.get('POI_MODE', 'fvg_or_ob')


def poi_candidate_types(params, dr):
    """POI classes allowed by the selected direction's H4 qualification."""
    mode = poi_mode_for_direction(params, dr)
    if mode == 'fvg_only':
        return ('FVG',)
    if mode == 'ob_only':
        return ('OB',)
    return ('FVG', 'OB')


def entry_zone_for_direction(params, dr):
    """Resolve allowed M5 entry zone for LONG/SHORT independently."""
    key = 'ENTRY_ZONE_LONG' if dr == 1 else 'ENTRY_ZONE_SHORT'
    mode = params.get(key)
    if mode in ('fvg_only', 'ob_only', 'fvg_or_ob'):
        return mode
    return params.get('ENTRY_ZONE', 'fvg_or_ob')


def short_bias_confirmation_allowed(dr, mode, h4_bias):
    """D1 remains the primary bias; optional SHORT setup requires bearish H4 structure."""
    return not (dr == -1 and mode == 'd1_h4_confirm') or int(h4_bias) == -1


def trailing_return_pct(closes, bar, lookback):
    """Return (%) from the close lookback bars ago to this completed bar close."""
    if bar < lookback or lookback < 1:
        return None
    prev = float(closes[bar - lookback])
    curr = float(closes[bar])
    if not np.isfinite(prev) or not np.isfinite(curr) or prev <= 0:
        return None
    return (curr / prev - 1.0) * 100.0


def long_recovery_filter_decision(dr, mode, closes, bar, min_24h_pct=-1.0):
    """
    LONG-only guard evaluated on the completed BOS candle.
    At order-decision time the current completed candle and prior 288 M5 closes are known.
    SHORT is never changed. Missing 24h history fails closed when the filter is active.
    Returns (allowed, trailing_24h_return_pct).
    """
    ret24 = trailing_return_pct(closes, bar, 288)
    if dr != 1 or mode == 'none':
        return True, ret24
    if mode != 'avoid_24h_drawdown':
        raise ValueError(f'Unknown LONG_RECOVERY_MODE: {mode}')
    if ret24 is None:
        return False, None
    return ret24 >= (float(min_24h_pct) - 1e-9), ret24


def long_bias_quality_decision(dr, mode, context, age_min_days=7.0, age_max_days=14.0,
                               min_protected_distance_atr=1.0, max_protected_distance_atr=None):
    """Evaluate LONG-only qualification at the completed BOS close.

    Modes:
      none: no LONG filter
      bias_age_7_14: D1 direction age in [7, 14) days
      protected_distance_ge1atr: distance >= min_protected_distance_atr
      protected_distance_band: distance in [min_protected_distance_atr, max_protected_distance_atr)
      combined: age and >= min protected-distance conditions

    Missing features fail closed for active filters. SHORT is never filtered.
    Returns (allowed, reasons, age_days, distance_atr).
    """
    age = context.get('snapshot_d1_bias_age_days', np.nan)
    distance = context.get('snapshot_distance_to_d1_protected_swing_atr', np.nan)
    try:
        age = float(age)
    except (TypeError, ValueError):
        age = np.nan
    try:
        distance = float(distance)
    except (TypeError, ValueError):
        distance = np.nan

    if dr != 1 or mode == 'none':
        return True, [], age, distance

    valid_modes = {'bias_age_7_14', 'protected_distance_ge1atr', 'protected_distance_band', 'combined'}
    if mode not in valid_modes:
        raise ValueError(f'Unknown LONG_QUALITY_MODE: {mode}')

    failed = []
    if mode in {'bias_age_7_14', 'combined'}:
        age_ok = np.isfinite(age) and age >= float(age_min_days) and age < float(age_max_days)
        if not age_ok:
            failed.append('d1_bias_age_outside_7_to_14_days_or_unavailable')
    if mode in {'protected_distance_ge1atr', 'protected_distance_band', 'combined'}:
        min_dist = float(min_protected_distance_atr)
        max_dist = None if max_protected_distance_atr in (None, '') else float(max_protected_distance_atr)
        distance_ok = np.isfinite(distance) and distance >= min_dist
        if distance_ok and max_dist is not None:
            distance_ok = distance < max_dist
        if not distance_ok:
            if max_dist is None:
                label = f'distance_to_d1_protected_swing_below_{min_dist:g}_atr_or_unavailable'
            else:
                label = f'distance_to_d1_protected_swing_outside_{min_dist:g}_to_{max_dist:g}_atr_or_unavailable'
            failed.append(label)
    return len(failed) == 0, failed, age, distance


def find_data(arg):
    base = Path(__file__).resolve().parent
    cands = ([Path(arg)] if arg else []) + [base / 'M5_Data', base / 'M5_data', base / 'data', base.parent / 'M5_Data', base.parent / 'M5_data', base.parent / 'data',
                                            Path.cwd() / 'M5_Data', Path.cwd() / 'M5_data', Path.cwd() / 'data']
    for c in cands:
        if c.is_dir() and any(p.is_dir() and any(p.glob('*.csv')) for p in c.iterdir()): return c
    raise SystemExit('Folder data tidak ketemu. Jalankan get.py dulu, atau taruh script di folder yang berisi M5_Data (atau pakai --data PATH)')


def load(d):  # toleran: header, timestamp mikrodetik, candle hilang (diisi datar)
    fs = sorted(Path(d).glob('*.csv'))
    df = pd.concat([pd.read_csv(f, header=None, usecols=[0, 1, 2, 3, 4], names=['ot', 'o', 'h', 'l', 'c']) for f in fs], ignore_index=True)
    df = df.apply(pd.to_numeric, errors='coerce').dropna()
    df['ot'] = df.ot.astype('int64')
    if df.ot.max() > 1e15: df['ot'] = df.ot // 1000
    df = df.drop_duplicates('ot').sort_values('ot').set_index('ot')
    # Align to UTC day boundary so aggregation into H4/D1 candles is not shifted
    # when the first available M5 candle is not exactly 00:00 UTC.
    t0 = (int(df.index[0]) // DAY) * DAY
    t1 = (int(df.index[-1]) // STEP) * STEP
    grid = np.arange(t0, t1 + STEP, STEP, dtype=np.int64)
    df = df.reindex(grid); missing_mask = df.c.isna().to_numpy(); miss = int(missing_mask.sum())
    # Longest consecutive missing-candle run, before flat-fill.
    max_gap = 0; run_gap = 0
    for is_missing in missing_mask:
        run_gap = run_gap + 1 if is_missing else 0
        max_gap = max(max_gap, run_gap)
    df['c'] = df.c.ffill().bfill()
    for k in 'ohl': df[k] = df[k].fillna(df['c'])
    return {k: df[k].values.astype(float) for k in 'ohlc'}, grid, miss, max_gap


def agg(a, k):
    n = len(a['o']) // k
    f = lambda x: x[:n * k].reshape(n, k)
    return dict(o=f(a['o'])[:, 0], h=f(a['h']).max(1), l=f(a['l']).min(1), c=f(a['c'])[:, -1])


def frame(a5, dr):  # dr=-1: cermin harga supaya logika SHORT = logika LONG
    b = a5 if dr == 1 else dict(o=-a5['o'], h=-a5['l'], l=-a5['h'], c=-a5['c'])
    return dict(m5=b, m15=agg(b, 3), h4=agg(b, 48), d1=agg(b, 288))


def pivots(h, l, n):
    N = len(h); ph = np.zeros(N, bool); pl = np.zeros(N, bool); w = 2 * n + 1
    hv, lv = swv(h, w), swv(l, w)
    ph[n:N - n] = (hv[:, n] >= hv[:, :n].max(1)) & (hv[:, n] > hv[:, n + 1:].max(1))
    pl[n:N - n] = (lv[:, n] <= lv[:, :n].min(1)) & (lv[:, n] < lv[:, n + 1:].min(1))
    return ph, pl


def fvg_arrays(x):  # FVG bullish terbentuk saat candle j close
    h, l = x['h'], x['l']; top = np.full(len(h), np.nan); bot = top.copy()
    m = l[2:] > h[:-2]; top[2:][m] = l[2:][m]; bot[2:][m] = h[:-2][m]
    return top, bot


def ob_arrays(x):  # OB bullish: candle turun terakhir lalu candle naik yang close di atas high-nya
    o, h, l, c = x['o'], x['h'], x['l'], x['c']; top = np.full(len(h), np.nan); bot = top.copy()
    m = (c[:-1] < o[:-1]) & (c[1:] > o[1:]) & (c[1:] > h[:-1]); top[1:][m] = h[:-1][m]; bot[1:][m] = l[:-1][m]
    return top, bot


def atr_array(x, period=14):
    h, l, c = x['h'], x['l'], x['c']
    prev = np.r_[c[0], c[:-1]]
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev), np.abs(l - prev)))
    return pd.Series(tr).rolling(period, min_periods=period).mean().to_numpy()


def protected_swing_low_timeline(x, n):
    """Last confirmed, not-yet-close-broken swing low in the mirrored directional structure.

    A pivot at bar j is only made available at j+n, matching the engine's causal
    pivot confirmation convention. The level is invalidated on a close below it.
    """
    h, l, c = x['h'], x['l'], x['c']
    _, pl = pivots(h, l, n)
    out = np.full(len(c), np.nan, dtype=float)
    low = None
    for t in range(len(c)):
        j = t - n
        if j >= 0 and pl[j]:
            low = (j, float(l[j]))
        if low is not None and c[t] < low[1]:
            low = None
        if low is not None:
            out[t] = low[1]
    return out


def sequence_age_timeline(sequence):
    """Bars since latest observed bias-structure event; NaN before first event."""
    seq = np.asarray(sequence)
    out = np.full(len(seq), np.nan, dtype=float)
    last_change = None
    previous = None
    has_previous = False
    for i, value in enumerate(seq):
        if not has_previous or value != previous:
            last_change = i if value >= 0 else None
            previous = value
            has_previous = True
        if last_change is not None:
            out[i] = i - last_change
    return out


def direction_age_timeline(direction):
    """Bars since the current non-neutral bias direction began.

    Unlike sequence_age_timeline(), same-direction structure updates do not reset
    this clock. The clock resets only when the bias changes direction or becomes
    non-neutral after neutral. Neutral periods are reported as NaN.
    """
    values = np.asarray(direction, dtype=int)
    out = np.full(len(values), np.nan, dtype=float)
    current = 0
    start = None
    for i, value in enumerate(values):
        value = int(value)
        if value != current:
            current = value
            start = i if value != 0 else None
        if current != 0 and start is not None:
            out[i] = i - start
    return out


def bias_label(value):
    try:
        v = int(value)
    except (TypeError, ValueError):
        return 'NEUTRAL'
    return 'LONG' if v == 1 else 'SHORT' if v == -1 else 'NEUTRAL'


def price_fully_outside_poi(low, high, poi_top, poi_bottom):
    return bool(high < poi_bottom or low > poi_top)


def poi_touch_allowed(retouch_required, exit_seen, is_touch):
    return bool(is_touch and ((not retouch_required) or exit_seen))

def detect_post_mss_bos(k, k_mss, i_lo5, swing_highs, closes):
    """Return (bar, pivot_bar, pivot_price) only for a fresh close-cross after MSS.

    swing_highs must contain only pivots confirmed with data available by bar k.
    A BOS on the MSS confirmation bar itself is deliberately not allowed.
    """
    if k <= k_mss or k <= 0:
        return None
    eligible = [(j, price) for j, price in swing_highs if i_lo5 < j < k]
    if not eligible:
        return None
    pivot_bar, pivot_price = eligible[-1]  # latest confirmed swing after the reaction low
    if closes[k - 1] <= pivot_price < closes[k]:
        return (k, pivot_bar, float(pivot_price))
    return None


def d1_bull_breaks(d, n):  # (index close D1, index origin leg) tiap close di atas swing high
    h, l, c = d['h'], d['l'], d['c']; ph, pl = pivots(h, l, n); sh = sl = None; ev = []
    for t in range(len(c)):
        i = t - n
        if i >= 0:
            if ph[i]: sh = (i, h[i])
            if pl[i]: sl = (i, l[i])
        if sl is not None and c[t] < sl[1]: sl = None
        if sh is None or sl is None or c[t] <= sh[1]: continue
        ev.append((t, sl[0] + int(np.argmin(l[sl[0]:t + 1])))); sh = None
    return ev


def bias_timeline(d_up, d_dn, n):  # bias D1 = arah break struktur terakhir
    ev = sorted([(t, 1, o) for t, o in d1_bull_breaks(d_up, n)] + [(t, -1, o) for t, o in d1_bull_breaks(d_dn, n)])
    N = len(d_up['c']); bd = np.zeros(N, int); bi = np.full(N, -1); bo = np.zeros(N, int); e = 0; cur = (0, 0, -1)
    for t in range(N):
        while e < len(ev) and ev[e][0] == t: cur = (ev[e][1], ev[e][2], cur[2] + 1); e += 1
        bd[t], bo[t], bi[t] = cur
    return bd, bi, bo


def mk(sym, dr, ot, kf, k, fp, E, sl, tp, ex, res, zone, ctx, P):
    s = dr; risk = fp - sl
    return dict(sym=sym, dir='LONG' if dr == 1 else 'SHORT', t_fill=ot[kf], t_exit=ot[k], bar_fill=int(kf), bar_exit=int(k),
                entry=s * fp, sl=s * sl, tp=s * tp, exit=s * ex, res=res, entry_zone=zone,
                risk_pct=risk / abs(fp) * 100, R_gross=(ex - fp) / risk,
                R_net=(ex - fp) / risk - P['COST'] * abs(fp) / risk, **ctx)


def run_frame(sym, dr, F, bias, ot, P, include_state=False):
    NP = P['NP']; m5, m15, h4, d1 = F['m5'], F['m15'], F['h4'], F['d1']
    o5, h5, l5, c5 = m5['o'], m5['h'], m5['l'], m5['c']
    o15, h15, l15, c15 = m15['o'], m15['h'], m15['l'], m15['c']
    l4, l1, c1 = h4['l'], d1['l'], d1['c']
    dft, dfb = fvg_arrays(d1); hft, hfb = fvg_arrays(h4); hot, hob = ob_arrays(h4)
    atr1 = atr_array(d1, 14); atr4 = atr_array(h4, 14)
    ph5, _ = pivots(h5, l5, NP); ph15, pl15 = pivots(h15, l15, NP)
    # Bias arrays are generated from completed D1/H4 candles only. Sequence arrays
    # increment whenever the corresponding structure-break event updates the bias.
    bd, bid, bor, h4bd, h4bid, h4bor = bias
    # Two different clocks are deliberately kept separate:
    # 1) bias duration: elapsed completed D1/H4 bars since the current direction began;
    # 2) structure-update age: elapsed bars since the latest break event, including
    #    a same-direction update. V7 had exposed clock (2) under the ambiguous name
    #    "bias age"; V8 corrects the meaning and retains (2) under an explicit name.
    d1_bias_age = direction_age_timeline(bd)
    d1_bias_update_age = sequence_age_timeline(bid)
    h4_bias_age = direction_age_timeline(h4bd)
    h4_bias_update_age = sequence_age_timeline(h4bid)
    d1_protected_low = protected_swing_low_timeline(d1, NP)
    h4_protected_low = protected_swing_low_timeline(h4, NP)
    # Range/volatility context is precomputed once per symbol. For an intraday event,
    # the 30-day reference uses only the latest fully completed D1 candles.
    trail_hi24 = pd.Series(h5).rolling(288, min_periods=288).max().to_numpy()
    trail_lo24 = pd.Series(l5).rolling(288, min_periods=288).min().to_numpy()
    daily_range_pct = (d1['h'] - d1['l']) / np.maximum(np.abs(d1['c']), 1e-12) * 100.0
    median_daily_range_pct = pd.Series(daily_range_pct).rolling(30, min_periods=10).median().to_numpy()

    def snapshot_context(k):
        'Causal D1/H4 state and M5 context known by the close of bar k.'
        d_idx = (k + 1) // 288 - 1
        h_idx = (k + 1) // 48 - 1
        d_bias = int(bd[d_idx]) if 0 <= d_idx < len(bd) else 0
        h_bias = int(h4bd[h_idx]) if 0 <= h_idx < len(h4bd) else 0
        d_age = float(d1_bias_age[d_idx]) if 0 <= d_idx < len(d1_bias_age) and np.isfinite(d1_bias_age[d_idx]) else np.nan
        d_update_age = float(d1_bias_update_age[d_idx]) if 0 <= d_idx < len(d1_bias_update_age) and np.isfinite(d1_bias_update_age[d_idx]) else np.nan
        h_age = float(h4_bias_age[h_idx] * 4.0) if 0 <= h_idx < len(h4_bias_age) and np.isfinite(h4_bias_age[h_idx]) else np.nan
        h_update_age = float(h4_bias_update_age[h_idx] * 4.0) if 0 <= h_idx < len(h4_bias_update_age) and np.isfinite(h4_bias_update_age[h_idx]) else np.nan
        if d_bias == 0 and h_bias == 0: align = 'both_neutral'
        elif d_bias == 0: align = 'd1_neutral'
        elif h_bias == 0: align = 'h4_neutral'
        elif d_bias == h_bias: align = 'aligned'
        else: align = 'opposed'
        model_px = float(c5[k])
        def directional_return(back):
            j = k - back
            if j < 0 or not np.isfinite(c5[j]) or abs(c5[j]) < 1e-12:
                return np.nan
            return (model_px - float(c5[j])) / abs(float(c5[j])) * 100.0
        if k < 287 or not np.isfinite(trail_hi24[k]) or not np.isfinite(trail_lo24[k]) or abs(model_px) < 1e-12:
            range24_pct = np.nan
        else:
            range24_pct = (float(trail_hi24[k]) - float(trail_lo24[k])) / abs(model_px) * 100.0
        med_range = float(median_daily_range_pct[d_idx]) if 0 <= d_idx < len(median_daily_range_pct) and np.isfinite(median_daily_range_pct[d_idx]) else np.nan
        vol_ratio = range24_pct / med_range if np.isfinite(range24_pct) and np.isfinite(med_range) and med_range > 0 else np.nan
        d_sup = float(d1_protected_low[d_idx]) if 0 <= d_idx < len(d1_protected_low) and np.isfinite(d1_protected_low[d_idx]) else np.nan
        h_sup = float(h4_protected_low[h_idx]) if 0 <= h_idx < len(h4_protected_low) and np.isfinite(h4_protected_low[h_idx]) else np.nan
        d_atr = float(atr1[d_idx]) if 0 <= d_idx < len(atr1) and np.isfinite(atr1[d_idx]) else np.nan
        h_atr = float(atr4[h_idx]) if 0 <= h_idx < len(atr4) and np.isfinite(atr4[h_idx]) else np.nan
        def dist_atr(level, av):
            return (model_px - level) / av if np.isfinite(level) and np.isfinite(av) and av > 0 else np.nan
        d_distance = dist_atr(d_sup, d_atr)
        h_distance = dist_atr(h_sup, h_atr)
        if np.isfinite(aoi[0]) and np.isfinite(aoi[1]):
            aoi_mid = (aoi[0] + aoi[1]) / 2.0
            aoi_dist_atr = (model_px - aoi_mid) / d_atr if np.isfinite(d_atr) and d_atr > 0 else np.nan
            aoi_width_atr = abs(aoi[0] - aoi[1]) / d_atr if np.isfinite(d_atr) and d_atr > 0 else np.nan
        else:
            aoi_dist_atr = aoi_width_atr = np.nan
        if np.isfinite(poi[0]) and np.isfinite(poi[1]):
            poi_mid = (poi[0] + poi[1]) / 2.0
            poi_dist_atr = (model_px - poi_mid) / h_atr if np.isfinite(h_atr) and h_atr > 0 else np.nan
            poi_width_atr = abs(poi[0] - poi[1]) / h_atr if np.isfinite(h_atr) and h_atr > 0 else np.nan
        else:
            poi_dist_atr = poi_width_atr = np.nan
        atr1_pct = d_atr / abs(float(d1['c'][d_idx])) * 100.0 if np.isfinite(d_atr) and 0 <= d_idx < len(d1['c']) and abs(float(d1['c'][d_idx])) > 1e-12 else np.nan
        atr4_pct = h_atr / abs(float(h4['c'][h_idx])) * 100.0 if np.isfinite(h_atr) and 0 <= h_idx < len(h4['c']) and abs(float(h4['c'][h_idx])) > 1e-12 else np.nan
        return {
            'snapshot_d1_bias': bias_label(d_bias), 'snapshot_d1_bias_sign': d_bias,
            'snapshot_h4_bias': bias_label(h_bias), 'snapshot_h4_bias_sign': h_bias,
            'snapshot_d1_h4_alignment': align,
            'snapshot_setup_matches_d1_bias': bool(d_bias == dr),
            'snapshot_setup_matches_h4_bias': bool(h_bias == dr),
            'snapshot_d1_bias_age_days': d_age,
            'snapshot_d1_bias_update_age_days': d_update_age,
            'snapshot_h4_bias_age_hours': h_age,
            'snapshot_h4_bias_update_age_hours': h_update_age,
            'snapshot_directional_d1_protected_swing_level': d_sup * dr if np.isfinite(d_sup) else np.nan,
            'snapshot_directional_h4_protected_swing_level': h_sup * dr if np.isfinite(h_sup) else np.nan,
            'snapshot_distance_to_d1_protected_swing_atr': d_distance,
            'snapshot_distance_to_h4_protected_swing_atr': h_distance,
            'snapshot_price_below_d1_protected_swing': bool(np.isfinite(d_distance) and d_distance < 0),
            'snapshot_prior_1h_directional_return_pct': directional_return(12),
            'snapshot_prior_4h_directional_return_pct': directional_return(48),
            'snapshot_prior_24h_directional_return_pct': directional_return(288),
            'snapshot_prior_3d_directional_return_pct': directional_return(864),
            'snapshot_trailing_24h_range_pct': range24_pct,
            'snapshot_median_30d_daily_range_pct': med_range,
            'snapshot_volatility_ratio_24h_to_30d_median_daily_range': vol_ratio,
            'snapshot_atr_d1_pct': atr1_pct,
            'snapshot_atr_h4_pct': atr4_pct,
            'snapshot_distance_to_aoi_mid_atr': aoi_dist_atr,
            'snapshot_aoi_width_atr': aoi_width_atr,
            'snapshot_distance_to_poi_mid_atr': poi_dist_atr,
            'snapshot_poi_width_atr': poi_width_atr,
        }
    sh5 = deque(maxlen=400); sh15 = deque(maxlen=200); sl15 = deque(maxlen=200)
    cnt = dict(bias=0, aoi=0, aoi_touch=0, poi=0, poi_touch=0, mss=0, placed=0, filled=0, poi_broken=0, aoi_broken=0,
                mss_timeout=0, mss_low_broken=0, bos_timeout=0, order_expired=0, no_entry_zone=0, order_rejected=0, invalid_fill_touch=0, bias_reset=0, short_bias_h4_reject=0, long_momentum_filtered=0, long_quality_filtered=0)
    trades = []; events = []; setup_id = 0; poi_id = 0; poi_type = ''; st = -1; cur = -1; active = False; org1 = 0
    event_seq = 0; attempt_seq = 0; current_attempt_id = 0
    retouch_required = False; poi_exit_seen_after_failure = False
    aoi = (np.nan, np.nan); poi = (np.nan, np.nan); k_aoi = k_poi = k_touch = m_touch = k_mss = i_lo5 = k_set = kf = 0
    L = E = SL = TP = fp = np.nan; zone = ''; ctx = {}
    def emit(event, k, **extra):
        nonlocal event_seq
        event_seq += 1
        # Event fields are in actual market price coordinates, also for mirrored SHORT runs.
        rec = dict(sym=sym, dir='LONG' if dr == 1 else 'SHORT', event=event, event_seq=int(event_seq), bar=int(k),
                   time_ms=int(ot[min(max(k, 0), len(ot)-1)]), close_time_ms=int(ot[min(max(k, 0), len(ot)-1)] + STEP), price=float(c5[k] * dr),
                   setup_id=int(setup_id), poi_id=int(poi_id), attempt_id=int(current_attempt_id), state=int(st))
        if np.isfinite(aoi[0]) and np.isfinite(aoi[1]):
            rec['aoi_low'] = min(aoi[0] * dr, aoi[1] * dr); rec['aoi_high'] = max(aoi[0] * dr, aoi[1] * dr)
        if np.isfinite(poi[0]) and np.isfinite(poi[1]):
            rec['poi_low'] = min(poi[0] * dr, poi[1] * dr); rec['poi_high'] = max(poi[0] * dr, poi[1] * dr)
        rec.update(snapshot_context(k))
        rec.update(extra); events.append(rec)
    def rearm_after_failure(k):
        nonlocal st, k_touch, m_touch, current_attempt_id, retouch_required, poi_exit_seen_after_failure
        k_touch = k; m_touch = k // 3; current_attempt_id = 0
        if P.get('REQUIRE_POI_RETOUCH_AFTER_FAILED_ATTEMPT', False):
            st = 3
            retouch_required = True
            poi_exit_seen_after_failure = False
        else:
            st = 4
            retouch_required = False
            poi_exit_seen_after_failure = False
    progress_step = max(1, len(o5) // 10)
    for k in range(len(o5)):
        if k > 0 and k % progress_step == 0:
            print(f'    {sym} {"LONG" if dr == 1 else "SHORT"}: simulasi {min(100, round(k / max(len(o5)-1, 1) * 100))}% ({k:,}/{len(o5):,} candle)', flush=True)
        # --- intrabar: posisi & order ---
        if st == 7 and k > kf:
            ex = None
            if l5[k] <= SL: ex, res = min(SL, o5[k]), 'SL'
            elif h5[k] >= TP: ex, res = TP, 'TP'
            elif k - kf >= P['HOLD']: ex, res = c5[k], 'TO'
            if ex is not None:
                trades.append(mk(sym, dr, ot, kf, k, fp, E, SL, TP, ex, res, zone, ctx, P)); emit('TRADE_EXIT', k, result=res, exit_price=float(ex * dr), entry_price=float(fp * dr), sl_price=float(SL * dr), tp_price=float(TP * dr)); st = 0 if active else -1; current_attempt_id = 0
        elif st == 6:
            if l5[k] <= E:
                f = min(o5[k], E)
                if f > SL:
                    st = 7; kf = k; fp = f; cnt['filled'] += 1; ex = None
                    ctx.update({f'fill_{key.removeprefix("snapshot_")}': value for key, value in snapshot_context(k).items()})
                    emit('ENTRY_FILLED', k, entry_price=float(fp * dr), order_price=float(E * dr), entry_zone=zone, poi_type=poi_type, sl_price=float(SL * dr), tp_price=float(TP * dr), risk_pct=float((fp-SL)/abs(fp)*100), pending_bars=k-k_set, pending_minutes=(k-k_set)*5)
                    if l5[k] <= SL: ex, res = SL, 'SL'
                    elif h5[k] >= TP: ex, res = TP, 'TP'
                    if ex is not None:
                        trades.append(mk(sym, dr, ot, kf, k, fp, E, SL, TP, ex, res, zone, ctx, P)); emit('TRADE_EXIT', k, result=res, exit_price=float(ex * dr), entry_price=float(fp * dr), sl_price=float(SL * dr), tp_price=float(TP * dr)); st = 0 if active else -1; current_attempt_id = 0
                else:
                    cnt['invalid_fill_touch'] += 1; emit('ORDER_INVALIDATED_AT_TOUCH', k, reason='fill_price_not_above_sl', fill_price=float(f*dr), order_price=float(E*dr), sl_price=float(SL*dr))
                    rearm_after_failure(k)
            elif h5[k] >= TP or k - k_set > P['PEND_WIN']:
                cnt['order_expired'] += 1; emit('ORDER_EXPIRED', k, reason='price_reached_tp_before_fill' if h5[k] >= TP else 'pending_timeout', order_price=float(E * dr), entry_zone=zone)
                rearm_after_failure(k)
        if st == 1 and l5[k] <= aoi[0] and h5[k] >= aoi[1]:
            st = 2; cnt['aoi_touch'] += 1; emit('AOI_TOUCHED', k, aoi_age_hours=(k-k_aoi)*5/60)
        if st == 3:
            if retouch_required and not poi_exit_seen_after_failure:
                fully_outside = price_fully_outside_poi(l5[k], h5[k], poi[0], poi[1])
                if fully_outside:
                    poi_exit_seen_after_failure = True
                    emit('POI_EXIT_AFTER_FAILED_ATTEMPT', k, reason='price_fully_left_POI_before_retouch')
            is_poi_touch = l5[k] <= poi[0] and h5[k] >= poi[1]
            touch_allowed = poi_touch_allowed(retouch_required, poi_exit_seen_after_failure, is_poi_touch)
            if touch_allowed:
                was_retouch = bool(retouch_required and poi_exit_seen_after_failure)
                st = 4; k_touch = k; m_touch = k // 3; cnt['poi_touch'] += 1
                retouch_required = False; poi_exit_seen_after_failure = False
                emit('POI_TOUCHED', k, poi_type=poi_type, aoi_age_hours=(k-k_aoi)*5/60, poi_age_hours=(k-k_poi)*5/60,
                     is_retouch_after_failed_attempt=was_retouch)
        i = k - NP
        if i >= 0 and ph5[i]: sh5.append((i, h5[i]))
        # --- close M15: invalid POI / cari MSS ---
        if (k + 1) % 3 == 0:
            m = (k + 1) // 3 - 1; i = m - NP
            if i >= 0 and ph15[i]: sh15.append((i, h15[i]))
            if i >= 0 and pl15[i]: sl15.append((i, l15[i]))
            # While waiting for a post-failure retouch, keep the original POI
            # invalidation rule active. A close through its invalidation boundary
            # destroys the zone; it must not remain eligible merely because state=3.
            if st == 3 and retouch_required and c15[m] < poi[1]:
                st = 8; cnt['poi_broken'] += 1
                emit('POI_BROKEN', k, close_m15=float(c15[m] * dr), reason='closed_beyond_POI_while_waiting_for_retouch')
                current_attempt_id = 0; retouch_required = False; poi_exit_seen_after_failure = False
            if st == 4:
                a = m_touch
                if c15[m] < poi[1]:
                    st = 8; cnt['poi_broken'] += 1; emit('POI_BROKEN', k, close_m15=float(c15[m] * dr))
                elif k - k_touch > P['MSS_WIN']:
                    cnt['mss_timeout'] += 1; emit('MSS_TIMEOUT', k); st = 0
                else:
                    hit = False
                    if P['MSS_MODE'] == 'swing':
                        im = a + int(np.argmin(l15[a:m + 1]))
                        if im < m:
                            ref = next((p for (j, p) in reversed(sh15) if a - 16 <= j < im), None)
                            hit = ref is not None and c15[m] > ref
                    else:  # literal: close di atas high candle sebelumnya (yang bearish) setelah touch
                        hit = m - 1 >= a and c15[m] > h15[m - 1] and c15[m - 1] < o15[m - 1]
                    if hit:
                        # A new MSS confirmation starts a uniquely identifiable attempt.
                        attempt_seq += 1
                        current_attempt_id = attempt_seq
                        # Keep M5 local low for BOS anchoring, but use M15 structure for SL.
                        s5 = a * 3
                        i_lo5 = s5 + int(np.argmin(l5[s5:k + 1]))
                        if P['SL_MODE'] == 'legacy_m5':
                            L = l5[i_lo5]
                        elif P['SL_MODE'] == 'confirmed_swing':
                            latest_low = next((v for (j, v) in reversed(sl15) if a <= j < m), None)
                            L = float(latest_low) if latest_low is not None else float(np.min(l15[a:m + 1]))
                        else:  # m15_12
                            lo15 = max(0, m - 11)
                            L = float(np.min(l15[lo15:m + 1]))
                        st = 5; k_mss = k; cnt['mss'] += 1; emit('MSS_CONFIRMED', k, sl_anchor=float(L * dr), mss_close=float(c15[m] * dr), poi_touch_to_mss_hours=(k-k_touch)*5/60)
        # --- M5: BOS -> zona entry FVG/OB ---
        if st == 5:
            if c5[k] < L or k - k_mss > P['BOS_WIN']:
                failure_reason='mss_low_broken' if c5[k] < L else 'bos_timeout'
                if failure_reason == 'mss_low_broken': cnt['mss_low_broken'] += 1
                else: cnt['bos_timeout'] += 1
                emit('MSS_SETUP_FAILED', k, reason=failure_reason, sl_anchor=float(L * dr)); rearm_after_failure(k)
            else:
                # BOS must occur on a bar strictly AFTER the MSS confirmation.
                # Use a confirmed M5 swing high formed after the local reaction low,
                # and require a fresh close-cross above that level on the current bar.
                bos_candidate = detect_post_mss_bos(k, k_mss, i_lo5, list(sh5), c5)
                b5 = None; bos_ref = None; bos_pivot_bar = None
                if bos_candidate is not None:
                    b5, bos_pivot_bar, bos_ref = bos_candidate
                if b5 is not None:
                    emit('BOS_CONFIRMED', k, bos_bar=int(b5), bos_time_ms=int(ot[b5]), bos_price=float(c5[b5] * dr),
                         bos_ref_high=float(bos_ref * dr), bos_pivot_bar=int(bos_pivot_bar),
                         prior_close=float(c5[k-1] * dr), sl_anchor=float(L * dr),
                         mss_to_bos_minutes=(k-k_mss)*5, mss_to_bos_bars=int(k-k_mss),
                         bos_rule='fresh_close_cross_above_confirmed_m5_swing_after_mss')
                    # Only use zones formed after MSS when FRESH_ENTRY is enabled.
                    zstart = max(k_mss + 1 if P.get('FRESH_ENTRY', True) else i_lo5 + 2, 2)
                    entry_mode = entry_zone_for_direction(P, dr)
                    # The selected mode constrains which zones may create an order.
                    # In 'ob_only', do not silently prefer an available FVG.
                    fv = [j for j in range(zstart, b5 + 1) if l5[j] > h5[j - 2]] if entry_mode in ('fvg_only', 'fvg_or_ob') else []
                    cd = []
                    if fv:
                        j = fv[-1]; zt, zb, zone = l5[j], h5[j - 2], 'FVG'
                    elif entry_mode in ('ob_only', 'fvg_or_ob'):
                        cstart = max(k_mss + 1 if P.get('FRESH_ENTRY', True) else i_lo5 - 1, 0)
                        # Valid bullish OB proxy: bearish candle immediately followed
                        # by a bullish displacement candle closing above its high.
                        cd = [j for j in range(cstart, b5)
                              if j + 1 <= b5 and c5[j] < o5[j]
                              and c5[j + 1] > o5[j + 1] and c5[j + 1] > h5[j]]
                        if cd:
                            j = cd[-1]; zt, zb, zone = h5[j], l5[j], 'OB'
                    e = None if (not fv and not cd) else (zt if P['ENTRY'] == 'prox' else (zt + zb) / 2)
                    sl = L - P['SL_BUF'] * abs(L)
                    if P.get('SL_BUF_ATR', 0.0) > 0:
                        atr15 = atr_array(m15, 14)
                        av = atr15[min(k // 3, len(atr15) - 1)]
                        if np.isfinite(av): sl = L - P['SL_BUF_ATR'] * av
                    long_filter_mode = P.get('LONG_RECOVERY_MODE', 'none')
                    long_filter_min24 = float(P.get('LONG_RECOVERY_MIN24H_PCT', -1.0))
                    long_filter_allowed, ret24_at_bos = long_recovery_filter_decision(
                        dr, long_filter_mode, c5, k, long_filter_min24)
                    quality_mode = P.get('LONG_QUALITY_MODE', 'none')
                    bos_context = snapshot_context(k)
                    quality_allowed, quality_reasons, bias_age_at_bos, protected_distance_at_bos = long_bias_quality_decision(
                        dr, quality_mode, bos_context,
                        P.get('LONG_BIAS_AGE_MIN_DAYS', 7.0),
                        P.get('LONG_BIAS_AGE_MAX_DAYS', 14.0),
                        P.get('LONG_PROTECTED_SWING_MIN_ATR', 1.0),
                    P.get('LONG_PROTECTED_SWING_MAX_ATR', None))
                    if e is not None and e > sl and not long_filter_allowed:
                        cnt['long_momentum_filtered'] += 1
                        emit('LONG_MOMENTUM_FILTERED', k, reason='trailing_24h_return_below_threshold_or_unavailable',
                             trailing_24h_directional_return_pct=ret24_at_bos, threshold_pct=long_filter_min24,
                             entry_zone=zone, poi_type=poi_type, bos_bar=int(b5),
                             filter_evaluation='completed_BOS_bar_close_before_order_submission')
                        rearm_after_failure(k)
                    elif e is not None and e > sl and not quality_allowed:
                        cnt['long_quality_filtered'] += 1
                        emit('LONG_QUALITY_FILTERED', k, reason=';'.join(quality_reasons),
                             long_quality_mode=quality_mode,
                             d1_bias_age_days_at_bos=bias_age_at_bos,
                             distance_to_d1_protected_swing_atr_at_bos=protected_distance_at_bos,
                             entry_zone=zone, poi_type=poi_type, bos_bar=int(b5),
                             filter_evaluation='completed_BOS_bar_close_before_order_submission',
                             rearm_policy='legacy_same_POI_rearm; next MSS attempt may be evaluated without a fresh retouch')
                        rearm_after_failure(k)
                    elif e is not None and e > sl:
                        E = e; SL = sl; TP = E + P['RR'] * (E - SL); st = 6; k_set = k; cnt['placed'] += 1
                        ctx = dict(t_aoi=ot[k_aoi], t_poi=ot[k_poi], t_mss=ot[k_mss],
                                   poi_type=poi_type, poi_qualification_mode=poi_mode_for_direction(P, dr),
                                   entry_qualification_mode=entry_zone_for_direction(P, dr),
                                   short_bias_mode=P.get('SHORT_BIAS_MODE', 'd1_only'),
                                   long_recovery_mode=long_filter_mode, long_quality_mode=quality_mode,
                                   long_bias_age_min_days=float(P.get('LONG_BIAS_AGE_MIN_DAYS', 7.0)),
                                   long_bias_age_max_days=float(P.get('LONG_BIAS_AGE_MAX_DAYS', 14.0)),
                                   long_protected_swing_min_atr=float(P.get('LONG_PROTECTED_SWING_MIN_ATR', 1.0)),
                                   long_protected_swing_max_atr=(float(P['LONG_PROTECTED_SWING_MAX_ATR']) if P.get('LONG_PROTECTED_SWING_MAX_ATR') is not None else np.nan),
                                   trailing_24h_directional_return_pct_at_order=ret24_at_bos,
                                   long_recovery_min24h_pct=long_filter_min24,
                                   setup_id=int(setup_id), poi_id=int(poi_id), attempt_id=int(current_attempt_id),
                                   aoi_low=float(min(aoi[0]*dr,aoi[1]*dr)), aoi_high=float(max(aoi[0]*dr,aoi[1]*dr)),
                                   poi_low=float(min(poi[0]*dr,poi[1]*dr)), poi_high=float(max(poi[0]*dr,poi[1]*dr)),
                                   t_order=ot[k], bars_aoi_to_poi=int((k_poi-k_aoi)//48),
                                   bars_poi_to_mss=int((k_mss-k_poi)//3),
                                   **{f'order_{key.removeprefix("snapshot_")}': value for key, value in snapshot_context(k).items()})
                        emit('ORDER_PLACED', k, order_price=float(E*dr), entry_zone=zone, poi_type=poi_type, sl_price=float(SL*dr), tp_price=float(TP*dr), rr=float(P['RR']),
                             trailing_24h_directional_return_pct=ret24_at_bos, long_recovery_mode=long_filter_mode,
                             long_recovery_min24h_pct=long_filter_min24)
                    else:
                        if e is None:
                            cnt['no_entry_zone'] += 1; emit('ENTRY_ZONE_NOT_FOUND', k, reason=f'no_valid_{entry_mode}_after_bos', bos_bar=int(b5), poi_type=poi_type, entry_qualification_mode=entry_mode)
                        else:
                            cnt['order_rejected'] += 1; emit('ORDER_REJECTED', k, reason='entry_not_above_sl_in_mirrored_model', proposed_entry=float(e*dr), proposed_sl=float(sl*dr), poi_type=poi_type)
                        rearm_after_failure(k)
        # --- close H4: cari POI di dalam AOI ---
        if (k + 1) % 48 == 0:
            h = (k + 1) // 48 - 1
            if st == 3 and k - k_poi > P['POI_WAIT']:
                emit('POI_TIMEOUT', k); st = 0
            if st == 2:
                best = None
                poi_mode = poi_mode_for_direction(P, dr)
                allowed_poi_types = poi_candidate_types(P, dr)
                for j in range(h, max(1, h - P['POI_LB']) - 1, -1):
                    candidates = []
                    if 'FVG' in allowed_poi_types:
                        candidates.append((hft[j], hfb[j], 'FVG'))
                    if 'OB' in allowed_poi_types:
                        candidates.append((hot[j], hob[j], 'OB'))
                    for top, bot, typ in candidates:
                        if np.isnan(top) or top >= c5[k] or top < aoi[1] or bot > aoi[0]: continue
                        if j < h and l4[j + 1:h + 1].min() <= top: continue
                        if best is None or top > best[0]: best = (top, bot, typ)
                if best:
                    poi = (best[0], best[1]); zone = best[2]; poi_type = best[2]; k_poi = k; st = 3; cnt['poi'] += 1; poi_id += 1; retouch_required = False; poi_exit_seen_after_failure = False
                    emit('POI_CREATED', k, poi_type=poi_type, poi_qualification_mode=poi_mode_for_direction(P, dr))
        # --- close D1: bias baru, AOI ---
        if (k + 1) % 288 == 0:
            d = (k + 1) // 288 - 1
            # st=8 means AOI/POI invalidated. Previously it could remain stuck
            # forever if the D1 bias did not change. Re-arm on the next D1 close
            # and search for a fresh AOI under the still-valid bias.
            if st == 8:
                st = 0 if active else -1; current_attempt_id = 0
            if bid[d] != cur:
                if st in (1,2,3,4,5,6):
                    cnt['bias_reset'] += 1; emit('SETUP_RESET_BIAS_CHANGE', k, previous_state=int(st), next_bias='LONG' if bd[d] == 1 else 'SHORT' if bd[d] == -1 else 'NONE')
                cur = bid[d]; active = bd[d] == dr; org1 = bor[d]
                # Preserve the attempt key while a filled position is still open so
                # TRADE_EXIT remains joinable to its ENTRY_FILLED/BOS/MSS events.
                if st != 7:
                    current_attempt_id = 0; retouch_required = False; poi_exit_seen_after_failure = False
                    aoi = (np.nan, np.nan); poi = (np.nan, np.nan); poi_type = ''
                if active:
                    cnt['bias'] += 1; emit('BIAS_ACTIVATED', k, bias_dir='LONG' if dr == 1 else 'SHORT', bias_origin_d1_bar=int(org1))
                if st != 7: st = 0 if active else -1
            if 2 <= st <= 6 and c1[d] < aoi[1]:
                st = 8; cnt['aoi_broken'] += 1; emit('AOI_BROKEN', k, d1_close=float(c1[d]*dr)); current_attempt_id = 0
            if st == 1 and (c1[d] < aoi[1] or k - k_aoi > P['AOI_AGE'] * 288):
                emit('AOI_EXPIRED', k, reason='d1_close_below' if c1[d] < aoi[1] else 'age_limit'); st = 0
            if st == 0 and active:
                # Optional SHORT-bias experiment: D1 bearish bias remains primary,
                # but an AOI is armed only when the latest completed H4 structure is bearish.
                h4_idx = min((k + 1) // 48 - 1, len(h4bd) - 1)
                current_h4_bias = int(h4bd[h4_idx]) if h4_idx >= 0 else 0
                bias_ok = short_bias_confirmation_allowed(dr, P.get('SHORT_BIAS_MODE', 'd1_only'), current_h4_bias)
                if not bias_ok:
                    cnt['short_bias_h4_reject'] += 1
                    emit('BIAS_FILTERED_H4_CONFIRMATION', k, d1_bias='SHORT', h4_bias=current_h4_bias,
                         short_bias_mode=P.get('SHORT_BIAS_MODE', 'd1_only'))
                else:
                    best = None
                    for j in range(d, max(org1 + 2, d - P['AOI_LB'], 2) - 1, -1):
                        top, bot = dft[j], dfb[j]
                        if np.isnan(top) or top >= c1[d]: continue
                        if P.get('AOI_MAX_ATR', 0.0) > 0:
                            av = atr1[j] if j < len(atr1) else np.nan
                            if not np.isfinite(av) or av <= 0 or (top - bot) > P['AOI_MAX_ATR'] * av: continue
                        if j < d and l1[j + 1:d + 1].min() <= top: continue
                        if best is None or top > best[0]: best = (top, bot)
                    if best:
                        aoi = best; k_aoi = k; st = 1; cnt['aoi'] += 1; setup_id += 1; poi_id = 0; poi_type = ''
                        emit('AOI_CREATED', k, short_bias_mode=P.get('SHORT_BIAS_MODE', 'd1_only'),
                             h4_bias_at_arm=current_h4_bias)
    if include_state:
        last_k = len(o5) - 1
        def _finite(value):
            try:
                value = float(value)
                return value if np.isfinite(value) else None
            except (TypeError, ValueError):
                return None
        def _real_pair(values):
            if values is None or len(values) != 2:
                return None
            pair_values = [_finite(values[0] * dr), _finite(values[1] * dr)]
            if any(value is None for value in pair_values):
                return None
            return [min(pair_values), max(pair_values)]
        d_idx = (last_k + 1) // 288 - 1 if last_k >= 0 else -1
        current_d1_bias = int(bd[d_idx]) if 0 <= d_idx < len(bd) else 0
        state_snapshot = {
            'state': int(st),
            'active_bias': bool(active),
            'current_d1_bias': int(current_d1_bias),
            'aoi': _real_pair(aoi),
            'poi': _real_pair(poi),
            'poi_type': str(poi_type or ''),
            'entry_zone': str(zone or ''),
            'entry_order_price': _finite(E * dr),
            'entry_fill_price': _finite(fp * dr),
            'sl': _finite(SL * dr),
            'tp': _finite(TP * dr),
            'current_price': _finite(c5[last_k] * dr) if last_k >= 0 else None,
            'last_m5_open_time_ms': int(ot[last_k]) if 0 <= last_k < len(ot) else None,
            'last_m5_close_time_ms': int(ot[last_k] + STEP) if 0 <= last_k < len(ot) else None,
            'order_time_ms': int(ot[k_set]) if st in (6, 7) and 0 <= k_set < len(ot) else None,
            'order_close_time_ms': int(ot[k_set] + STEP) if st in (6, 7) and 0 <= k_set < len(ot) else None,
            'fill_time_ms': int(ot[kf]) if st == 7 and 0 <= kf < len(ot) else None,
            'mss_time_ms': int(ot[k_mss]) if st in (5, 6, 7) and 0 <= k_mss < len(ot) else None,
            'mss_close_time_ms': int(ot[k_mss] + STEP) if st in (5, 6, 7) and 0 <= k_mss < len(ot) else None,
            'mss_close_price': _finite(c5[k_mss] * dr) if st in (5, 6, 7) and 0 <= k_mss < len(c5) else None,
            'sl_anchor': _finite(L * dr) if st in (5, 6, 7) else None,
            'waiting_bos_bars_elapsed': int(last_k - k_mss) if st == 5 else None,
            'bos_timeout_m5_bars': int(P['BOS_WIN']),
            'pending_bars_elapsed': int(last_k - k_set) if st == 6 else None,
            'position_bars_held': int(last_k - kf) if st == 7 else None,
            'setup_id': int(setup_id),
            'poi_id': int(poi_id),
            'attempt_id': int(current_attempt_id),
            'entry_context': dict(ctx) if isinstance(ctx, dict) else {},
            'last_event': str(events[-1].get('event')) if events else 'NONE',
        }
        return trades, cnt, events, state_snapshot
    return trades, cnt, events



# ============================================================================
# LIVE/PAPER ADAPTER — TACTICAL V10 E_GE2_ATR
# ============================================================================
from dataclasses import dataclass
from datetime import datetime, timezone
import asyncio
import hashlib
import logging
import time
from typing import Any, Callable

STRATEGY_NAME = 'TACTICAL_V10_E_GE2_ATR'
STRATEGY_VERSION = '10.0.0'
STRATEGY_ENGINE = 'TACTICAL_V10_STATE_MACHINE'
STRATEGY_SCHEMA_VERSION = 10
EXCLUDED_SYMBOLS = frozenset({'BTCUSDT', 'ETHUSDT'})
SCAN_PRIMARY_DATA_SOURCE = 'BINANCE_PUBLIC'
PRIMARY_DATA_SOURCE = 'BINANCE_PUBLIC'
STRATEGY_DATA_SOURCE = 'BINANCE_PUBLIC'

M5_INTERVAL_MS = 300_000
M15_INTERVAL_MS = 900_000
H4_INTERVAL_MS = 14_400_000
D1_INTERVAL_MS = 86_400_000
M5_HISTORY_DAYS = 66  # buffer satu hari agar alignment UTC menyisakan >=65 hari lengkap
M5_HISTORY_CANDLES = M5_HISTORY_DAYS * 288
M5_MIN_HISTORY_CANDLES = 60 * 288
PENDING_WINDOW_M5 = 72
MAX_HOLD_M5 = 2016
ROUND_TRIP_COST_BPS = 8.0
RULE_PASS_SCORE = 100.0  # skor kelengkapan aturan, BUKAN probabilitas kemenangan

# Exact E_GE2_ATR research configuration from run_tactical_v10.py.
V10_LIVE_PARAMS = dict(P)
V10_LIVE_PARAMS.update({
    'NP': 2,
    'RR': 2.0,
    'COST': 8e-4,
    'AOI_LB': 30,
    'AOI_AGE': 40,
    'AOI_MAX_ATR': 0.0,
    'POI_LB': 30,
    'POI_WAIT': 2880,
    'MSS_WIN': 576,
    'BOS_WIN': 144,
    'PEND_WIN': PENDING_WINDOW_M5,
    'HOLD': MAX_HOLD_M5,
    'MSS_MODE': 'literal',
    'ENTRY': 'prox',
    'POI_MODE': 'fvg_or_ob',
    'POI_MODE_LONG': 'fvg_only',
    'POI_MODE_SHORT': 'ob_only',
    'SL_MODE': 'm15_12',
    'ENTRY_ZONE': 'fvg_or_ob',
    'ENTRY_ZONE_LONG': 'fvg_or_ob',
    'ENTRY_ZONE_SHORT': 'fvg_only',
    'SHORT_BIAS_MODE': 'd1_only',
    'LONG_RECOVERY_MODE': 'none',
    'LONG_QUALITY_MODE': 'protected_distance_band',
    'LONG_PROTECTED_SWING_MIN_ATR': 2.0,
    'LONG_PROTECTED_SWING_MAX_ATR': None,
    'FRESH_ENTRY': True,
    'SL_BUF': 0.0,
    'SL_BUF_ATR': 0.0,
    'REQUIRE_POI_RETOUCH_AFTER_FAILED_ATTEMPT': False,
})

@dataclass(frozen=True)
class Candle:
    time_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    turnover: float = 0.0
    taker_buy: float = 0.0

_PROVIDER: dict[str, Any] = {}
_LOG = logging.getLogger('strategy.v10')


def configure_market_data_provider(*, get_series, get_price=None, get_tick_size=None,
                                    get_cache_stats=None, trim_cache=None) -> None:
    """Inject main.py's sole Binance REST/WebSocket data owner."""
    if not callable(get_series):
        raise TypeError('get_series provider wajib callable')
    _PROVIDER.clear()
    _PROVIDER.update({
        'get_series': get_series,
        'get_price': get_price,
        'get_tick_size': get_tick_size,
        'get_cache_stats': get_cache_stats,
        'trim_cache': trim_cache,
        'configured_at': time.monotonic(),
    })


def market_data_provider_status() -> dict[str, Any]:
    return {
        'configured': callable(_PROVIDER.get('get_series')),
        'owner': 'main.py',
        'source': 'BINANCE_PUBLIC',
        'engine': STRATEGY_ENGINE,
        'variant': 'E_GE2_ATR',
        'network_access_in_strategy': False,
    }


def _as_candle_tuple(item: Any) -> tuple[int, float, float, float, float]:
    if isinstance(item, dict):
        get = item.get
    else:
        get = lambda key, default=None: getattr(item, key, default)
    t = int(get('time_ms'))
    o, h, l, c = (float(get(k)) for k in ('open', 'high', 'low', 'close'))
    if not all(np.isfinite(v) for v in (o, h, l, c)) or min(o, h, l, c) <= 0:
        raise ValueError('OHLC M5 invalid/non-finite')
    if h < max(o, l, c) or l > min(o, h, c):
        raise ValueError('Geometri OHLC M5 invalid')
    return t, o, h, l, c


def _prepare_m5_arrays(candles: list[Any]) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, Any]]:
    rows = []
    for item in candles:
        try:
            rows.append(_as_candle_tuple(item))
        except (TypeError, ValueError, AttributeError):
            continue
    if len(rows) < M5_MIN_HISTORY_CANDLES:
        raise ValueError(f'Riwayat M5 tidak cukup: {len(rows):,}/{M5_MIN_HISTORY_CANDLES:,} candle minimum')

    # V10 backtest aligns M5 -> M15/H4/D1 from 00:00 UTC. Reproduce that origin.
    by_time = {row[0]: row[1:] for row in rows}
    ordered_times = sorted(by_time)
    first_raw = int(ordered_times[0])
    t0 = ((first_raw + D1_INTERVAL_MS - 1) // D1_INTERVAL_MS) * D1_INTERVAL_MS
    t1 = (int(ordered_times[-1]) // M5_INTERVAL_MS) * M5_INTERVAL_MS
    if t1 < t0 or (t1 - t0) // M5_INTERVAL_MS + 1 < M5_MIN_HISTORY_CANDLES:
        raise ValueError('Riwayat M5 setelah alignment UTC kurang dari 60 hari')

    grid = np.arange(t0, t1 + M5_INTERVAL_MS, M5_INTERVAL_MS, dtype=np.int64)
    lookup = {t: values for t, values in by_time.items() if t0 <= t <= t1}
    n = len(grid)
    o = np.full(n, np.nan, dtype=float)
    h = np.full(n, np.nan, dtype=float)
    l = np.full(n, np.nan, dtype=float)
    c = np.full(n, np.nan, dtype=float)
    for i, t in enumerate(grid):
        values = lookup.get(int(t))
        if values is not None:
            o[i], h[i], l[i], c[i] = values
    missing = ~np.isfinite(c)
    missing_count = int(missing.sum())
    max_gap = 0
    gap = 0
    for absent in missing:
        gap = gap + 1 if absent else 0
        max_gap = max(max_gap, gap)
    if missing_count:
        c_series = pd.Series(c).ffill().bfill()
        c = c_series.to_numpy(dtype=float)
        o = np.where(np.isfinite(o), o, c)
        h = np.where(np.isfinite(h), h, c)
        l = np.where(np.isfinite(l), l, c)
    if not all(np.isfinite(x).all() for x in (o, h, l, c)):
        raise ValueError('Riwayat M5 masih mengandung nilai invalid setelah normalisasi')
    if max_gap > 12:
        raise ValueError(f'Gap data M5 terlalu panjang: {max_gap} candle / {max_gap * 5} menit')
    arrays = {'o': o, 'h': h, 'l': l, 'c': c}
    info = {
        'candles_m5': int(n),
        'missing_candles_filled': missing_count,
        'longest_missing_run': int(max_gap),
        'first_time_ms': int(grid[0]),
        'last_open_time_ms': int(grid[-1]),
        'last_close_time_ms': int(grid[-1] + M5_INTERVAL_MS),
        'history_days': round(n / 288.0, 2),
    }
    return arrays, grid, info


def _finite_or_none(value: Any) -> float | None:
    try:
        number = float(value)
        return number if np.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def _iso_from_ms(value: int | None) -> str | None:
    if value is None:
        return None
    return datetime.fromtimestamp(int(value) / 1000.0, tz=timezone.utc).isoformat()


def _analyze_v10_arrays(pair: str, a5: dict[str, np.ndarray], ot: np.ndarray,
                        data_info: dict[str, Any]) -> dict[str, Any]:
    frames = {direction: frame(a5, direction) for direction in (1, -1)}
    d1_bias = bias_timeline(frames[1]['d1'], frames[-1]['d1'], V10_LIVE_PARAMS['NP'])
    h4_bias = bias_timeline(frames[1]['h4'], frames[-1]['h4'], V10_LIVE_PARAMS['NP'])
    bias = (d1_bias[0], d1_bias[1], d1_bias[2], h4_bias[0], h4_bias[1], h4_bias[2])

    states = []
    waiting_states = []
    summaries = {}
    for direction in (1, -1):
        _trades, counts, _events, state = run_frame(
            pair, direction, frames[direction], bias, ot, V10_LIVE_PARAMS, include_state=True
        )
        state['direction'] = 'LONG' if direction == 1 else 'SHORT'
        state['counts'] = counts
        summaries[state['direction']] = {
            'state': state['state'], 'last_event': state['last_event'],
            'setup_id': state.get('setup_id'), 'poi_id': state.get('poi_id'),
            'attempt_id': state.get('attempt_id'), 'poi_type': state.get('poi_type'),
            'aoi': state.get('aoi'), 'poi': state.get('poi'),
            'mss_close_time_ms': state.get('mss_close_time_ms'),
            'placed': counts.get('placed', 0), 'filled': counts.get('filled', 0),
            'mss': counts.get('mss', 0), 'aoi_count': counts.get('aoi', 0), 'poi_count': counts.get('poi', 0),
        }
        if state['state'] in (6, 7):
            states.append(state)
        elif state['state'] == 5:
            waiting_states.append(state)

    # Prefer a strategy-confirmed V10 order/position. Otherwise expose state 5
    # (MSS already confirmed on M15; waiting for a fresh M5 BOS close-cross).
    states.sort(key=lambda x: int(x.get('order_time_ms') or 0), reverse=True)
    waiting_states.sort(key=lambda x: int(x.get('mss_close_time_ms') or 0), reverse=True)
    selected = states[0] if states else (waiting_states[0] if waiting_states else None)
    result = {
        'pair': pair,
        'selected_state': selected,
        'direction_states': summaries,
        'current_d1_bias': int(d1_bias[0][-1]) if len(d1_bias[0]) else 0,
        'current_h4_bias': int(h4_bias[0][-1]) if len(h4_bias[0]) else 0,
        'data': data_info,
        'active_state_count': len(states),
        'waiting_m5_bos_count': len(waiting_states),
    }
    return result


async def generate_setup(pair: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
    """Replay V10 on closed Binance M5 candles; expose state 5 candidates and state 6/7 paper setups."""
    symbol = str(pair or '').strip().upper()
    if not symbol:
        raise ValueError('Pair kosong')
    if symbol in EXCLUDED_SYMBOLS:
        raise ValueError(f'{symbol} termasuk simbol yang dikecualikan pada eksperimen Tactical V10.')
    provider = _PROVIDER.get('get_series')
    if not callable(provider):
        raise RuntimeError('Market-data provider belum dipasang oleh main.py')

    force_refresh = bool((context or {}).get('force_refresh', False))
    fetched = provider(symbol, '5m', M5_HISTORY_CANDLES, M5_INTERVAL_MS, force_refresh=force_refresh)
    if asyncio.iscoroutine(fetched):
        candles, source = await fetched
    else:
        candles, source = fetched
    if str(source or '').upper() not in {'BINANCE', 'BINANCE_PUBLIC', 'BINANCE_REST'}:
        raise RuntimeError(f'Sumber candle tidak diizinkan untuk V10: {source!r}')

    arrays, timestamps, data_info = _prepare_m5_arrays(list(candles))
    now_ms = int(time.time() * 1000)
    stale_ms = now_ms - int(data_info['last_close_time_ms'])
    if stale_ms > 10 * 60_000:
        raise RuntimeError(f'Candle M5 Binance stale: {stale_ms // 60_000} menit')

    analysis = await asyncio.to_thread(_analyze_v10_arrays, symbol, arrays, timestamps, data_info)
    state = analysis['selected_state']
    v10_analysis = {
        'engine': STRATEGY_ENGINE,
        'variant': 'E_GE2_ATR',
        'state': ('NO_ACTIVE_ORDER' if state is None else
                  'WAITING_M5_BOS' if state['state'] == 5 else
                  'PENDING_LIMIT' if state['state'] == 6 else 'OPEN_POSITION'),
        'direction_states': analysis['direction_states'],
        'd1_bias': 'LONG' if analysis['current_d1_bias'] == 1 else 'SHORT' if analysis['current_d1_bias'] == -1 else 'NEUTRAL',
        'h4_bias_context': 'LONG' if analysis['current_h4_bias'] == 1 else 'SHORT' if analysis['current_h4_bias'] == -1 else 'NEUTRAL',
        'configuration': {
            'long_d1_bias': 'BULLISH', 'long_poi_h4': 'FVG_ONLY',
            'short_d1_bias': 'BEARISH', 'short_poi_h4': 'OB_ONLY',
            'mss': 'LITERAL_M15', 'bos': 'FRESH_CLOSE_CROSS_M5_AFTER_MSS',
            'long_distance_to_d1_protected_swing': '[2.0, +inf) ATR_D1',
            'short_distance_filter': 'NONE', 'entry_long_m5': 'FVG_PRIORITY_OB_FALLBACK',
            'entry_short_m5': 'FVG_ONLY', 'sl': '12_COMPLETED_M15_CANDLES',
            'rr': 2.0, 'round_trip_cost_bps_backtest': ROUND_TRIP_COST_BPS,
            'pending_window_m5_bars': PENDING_WINDOW_M5, 'max_hold_m5_bars': MAX_HOLD_M5,
            'same_poi_rearm_after_failed_attempt': True,
        },
        'backtest_reference': {
            'variant': 'E_GE2_ATR', 'all_trades': 250, 'win_rate_pct': 38.8,
            'expectancy_R': 0.095607, 'profit_factor': 1.143644, 'total_R': 23.901690,
            'max_drawdown_R': 15.840089, 'long_trades': 64, 'long_expectancy_R': 0.109999,
            'long_profit_factor': 1.170716, 'short_trades': 186, 'short_expectancy_R': 0.090655,
            'short_profit_factor': 1.134724, 'oos_2026_available': False,
        },
        'data': data_info,
        'active_state_count': analysis['active_state_count'],
    }

    if state is not None and int(state.get('state', -1)) == 5:
        direction_name = str(state.get('direction') or '')
        direction = 'BUY' if direction_name == 'LONG' else 'SELL'
        watch_key = '|'.join((symbol, direction_name, str(state.get('setup_id')), str(state.get('poi_id')), str(state.get('attempt_id'))))
        watch_id = hashlib.sha256(watch_key.encode('utf-8')).hexdigest()[:24]
        poi = state.get('poi') if isinstance(state.get('poi'), list) else None
        aoi = state.get('aoi') if isinstance(state.get('aoi'), list) else None
        mss_time_ms = state.get('mss_time_ms')
        mss_close_time_ms = state.get('mss_close_time_ms')
        v10_analysis.update({
            'state': 'WAITING_M5_BOS',
            'direction': direction_name,
            'poi_type': state.get('poi_type'),
            'aoi': aoi,
            'poi': poi,
            'setup_id': state.get('setup_id'),
            'poi_id': state.get('poi_id'),
            'attempt_id': state.get('attempt_id'),
            'mss_time_ms': mss_time_ms,
            'mss_close_time_ms': mss_close_time_ms,
            'mss_close_price': state.get('mss_close_price'),
            'sl_anchor': state.get('sl_anchor'),
            'waiting_bos_bars_elapsed': state.get('waiting_bos_bars_elapsed'),
            'bos_timeout_m5_bars': state.get('bos_timeout_m5_bars'),
            'last_m5_close_time_ms': state.get('last_m5_close_time_ms'),
            'same_poi_rearm_after_failed_attempt': True,
        })
        v10_meta = {
            'name': STRATEGY_NAME, 'version': STRATEGY_VERSION, 'variant': 'E_GE2_ATR',
        }
        data_contract = {
            'source': 'BINANCE_PUBLIC', 'timeframe': '5m',
            'candles_used': int(data_info['candles_m5']),
            'fallback_used': False,
            'sources_by_timeframe': {
                '5m': 'BINANCE_PUBLIC', '15m': 'BINANCE_PUBLIC_DERIVED_M5',
                '4h': 'BINANCE_PUBLIC_DERIVED_M5', '1d': 'BINANCE_PUBLIC_DERIVED_M5',
            },
            'candles_requested': M5_HISTORY_CANDLES,
            'data_asof': {'5m_close_time_ms': data_info['last_close_time_ms']},
            **data_info,
        }
        return {
            'pair': symbol,
            'direction': direction,
            'price_now_reference': state.get('current_price'),
            'confidence': RULE_PASS_SCORE,
            'confidence_components': {'v10_prerequisites_passed': RULE_PASS_SCORE},
            'candidate_state': 'WAITING_M5_BOS',
            'state_reason': 'D1 bias + D1 FVG AOI + H4 directional POI + literal M15 MSS valid; waiting for fresh M5 close-cross BOS after MSS.',
            'strategy_name': STRATEGY_NAME, 'strategy_version': STRATEGY_VERSION,
            'strategy': v10_meta, 'analysis': {'v10': v10_analysis},
            'data_source': 'BINANCE_PUBLIC', 'analysis_source': 'BINANCE_PUBLIC',
            'data': data_contract,
            'entry_zone': ({'low': min(poi), 'high': max(poi), 'kind': f'H4_{state.get("poi_type") or "POI"}', 'timeframe': '4h'} if poi else {}),
            'thesis_signature': watch_id, 'candidate_id': watch_id, 'v10_candidate_id': watch_id,
            'v10_watch_id': watch_id,
            'v10_setup_id': int(state.get('setup_id') or 0),
            'v10_poi_id': int(state.get('poi_id') or 0),
            'v10_attempt_id': int(state.get('attempt_id') or 0),
            'v10_poi': poi, 'v10_aoi': aoi, 'v10_poi_type': state.get('poi_type'),
            'v10_mss_time_ms': mss_time_ms, 'v10_mss_close_time_ms': mss_close_time_ms,
            'v10_mss_close_price': state.get('mss_close_price'),
            'v10_sl_anchor': state.get('sl_anchor'),
            'v10_waiting_bos_bars': int(state.get('waiting_bos_bars_elapsed') or 0),
            'v10_bos_timeout_bars': int(state.get('bos_timeout_m5_bars') or V10_LIVE_PARAMS['BOS_WIN']),
            'v10_last_m5_close_ms': state.get('last_m5_close_time_ms'),
            'v10_last_m5_close': state.get('current_price'),
            'm5_confirmed_entry': False,
            'entry_model': 'V10_MSS_M15_WAIT_BOS_M5_CLOSE',
        }

    if state is None:
        waiting_reason = 'V10 state machine belum menghasilkan AOI/POI/MSS yang sedang menunggu BOS.'
        return {
            'pair': symbol, 'direction': 'BUY' if analysis['current_d1_bias'] >= 0 else 'SELL',
            'confidence': 0.0, 'candidate_state': 'WAITING_TRIGGER',
            'strategy_name': STRATEGY_NAME, 'strategy_version': STRATEGY_VERSION,
            'strategy': {'name': STRATEGY_NAME, 'version': STRATEGY_VERSION, 'variant': 'E_GE2_ATR'},
            'analysis': {'v10': v10_analysis},
            'price_now_reference': float(arrays['c'][-1]) if len(arrays.get('c', [])) else None,
            'v10_last_m5_close_ms': int(data_info['last_close_time_ms']),
            'v10_last_m5_close': float(arrays['c'][-1]) if len(arrays.get('c', [])) else None,
            'state_reason': waiting_reason, 'data_source': 'BINANCE_PUBLIC',
            'analysis_source': 'BINANCE_PUBLIC',
            'data': {
                'source': 'BINANCE_PUBLIC',
        'timeframe': '5m',
        'candles_used': int(data_info['candles_m5']),
                'fallback_used': False,
                'sources_by_timeframe': {
                    '5m': 'BINANCE_PUBLIC', '15m': 'BINANCE_PUBLIC_DERIVED_M5',
                    '4h': 'BINANCE_PUBLIC_DERIVED_M5', '1d': 'BINANCE_PUBLIC_DERIVED_M5',
                },
                'candles_requested': M5_HISTORY_CANDLES,
                'data_asof': {'5m_close_time_ms': data_info['last_close_time_ms']},
                **data_info,
            },
        }

    direction = 'BUY' if state['direction'] == 'LONG' else 'SELL'
    current_price = _finite_or_none(state.get('current_price'))
    limit_price = _finite_or_none(state.get('entry_order_price'))
    fill_price = _finite_or_none(state.get('entry_fill_price'))
    sl = _finite_or_none(state.get('sl'))
    tp = _finite_or_none(state.get('tp'))
    pending = int(state['state']) == 6
    entry = limit_price if pending else fill_price
    if any(value is None or value <= 0 for value in (current_price, entry, sl, tp)):
        raise RuntimeError(f'Snapshot V10 {symbol} tidak memiliki harga entry/SL/TP valid')
    if direction == 'BUY' and not (sl < entry and tp > entry):
        raise RuntimeError(f'Geometri LONG V10 tidak valid untuk {symbol}')
    if direction == 'SELL' and not (sl > entry and tp < entry):
        raise RuntimeError(f'Geometri SHORT V10 tidak valid untuk {symbol}')

    order_time_ms = state.get('order_time_ms')
    fill_time_ms = state.get('fill_time_ms')
    order_id_source = f"{symbol}|{state['direction']}|{order_time_ms}|{state.get('setup_id')}|{state.get('poi_id')}|{state.get('attempt_id')}"
    candidate_id = hashlib.sha256(order_id_source.encode('utf-8')).hexdigest()[:24]
    watch_id_source = '|'.join((symbol, str(state['direction']), str(state.get('setup_id')), str(state.get('poi_id')), str(state.get('attempt_id'))))
    watch_id = hashlib.sha256(watch_id_source.encode('utf-8')).hexdigest()[:24]
    order_distance = state.get('entry_context', {}).get('order_distance_to_d1_protected_swing_atr')
    if state['direction'] == 'LONG':
        try:
            distance = float(order_distance)
        except (TypeError, ValueError):
            raise RuntimeError(f'Jarak protected swing D1 LONG tidak tersedia untuk {symbol}')
        if not np.isfinite(distance) or distance < 2.0:
            raise RuntimeError(f'LONG {symbol} melanggar filter V10 E_GE2_ATR: distance={order_distance!r}')
    else:
        distance = _finite_or_none(order_distance)

    order_created_at = _iso_from_ms(order_time_ms)
    fill_created_at = _iso_from_ms(fill_time_ms)
    position_state = 'PENDING' if pending else 'FILLED'
    reason_common = (
        f"Tactical V10 E_GE2_ATR | {state['direction']} | D1 bias | H4 POI {state.get('poi_type')} | "
        f"MSS literal M15 → BOS M5 | M5 entry {state.get('entry_zone')}"
    )
    v10_analysis.update({
        'state': 'PENDING_LIMIT' if pending else 'OPEN_POSITION',
        'direction': state['direction'],
        'poi_type': state.get('poi_type'),
        'entry_zone': state.get('entry_zone'),
        'order_time': order_created_at,
        'fill_time': fill_created_at,
        'protected_distance_at_order_atr': distance,
        'aoi': state.get('aoi'), 'poi': state.get('poi'),
        'mss_time_ms': state.get('mss_time_ms'),
        'pending_bars_elapsed': state.get('pending_bars_elapsed'),
        'position_bars_held': state.get('position_bars_held'),
        'setup_id': state.get('setup_id'), 'poi_id': state.get('poi_id'),
        'attempt_id': state.get('attempt_id'), 'watch_id': watch_id,
    })
    data_contract = {
        'source': 'BINANCE_PUBLIC',
        'timeframe': '5m',
        'candles_used': int(data_info['candles_m5']),
                'fallback_used': False,
        'sources_by_timeframe': {
            '5m': 'BINANCE_PUBLIC', '15m': 'BINANCE_PUBLIC_DERIVED_M5',
            '4h': 'BINANCE_PUBLIC_DERIVED_M5', '1d': 'BINANCE_PUBLIC_DERIVED_M5',
        },
        'candles_requested': M5_HISTORY_CANDLES,
        'data_asof': {'5m_close_time_ms': data_info['last_close_time_ms']},
        **data_info,
    }
    return {
        'pair': symbol,
        'direction': direction,
        'price_now_reference': current_price,
        'entry': entry,
        'entry_reason': reason_common + (f" | order limit {limit_price}" if pending else f" | paper fill {fill_price}"),
        'price_exp': tp,
        'price_exp_reason': 'Pending V10: invalidasi bila harga menyentuh TP sebelum limit terisi atau melewati 72 candle M5.',
        'sl': sl,
        'sl_reason': 'V10 SL: ekstrem range 12 candle M15 terakhir pada konfirmasi MSS literal.',
        'tp': tp,
        'tp_reason': 'V10 fixed target dengan RR gross 1:2 dari harga order limit.',
        'confidence': RULE_PASS_SCORE,
        'confidence_components': {'v10_rule_pass': RULE_PASS_SCORE},
        'strategy_name': STRATEGY_NAME,
        'strategy_version': STRATEGY_VERSION,
        'strategy': {'name': STRATEGY_NAME, 'version': STRATEGY_VERSION, 'variant': 'E_GE2_ATR'},
        'analysis': {'v10': v10_analysis},
        'analysis_source': 'BINANCE_PUBLIC',
        'data_source': 'BINANCE_PUBLIC',
        'data': data_contract,
        'candidate_state': 'READY' if pending else 'FILLED',
        'state_reason': 'V10 state machine state 6 (pending limit)' if pending else 'V10 state machine state 7 (filled position)',
        'setup_archetype': 'TACTICAL_V10_E_GE2_ATR',
        'thesis_signature': candidate_id,
        'candidate_id': candidate_id,
        'v10_watch_id': watch_id,
        'v10_setup_id': int(state.get('setup_id') or 0),
        'v10_poi_id': int(state.get('poi_id') or 0),
        'v10_attempt_id': int(state.get('attempt_id') or 0),
        'v10_order_time_ms': int(order_time_ms) if order_time_ms is not None else None,
        'v10_order_close_time_ms': int(state.get('order_close_time_ms') or 0),
        'v10_bos_time_ms': int(order_time_ms) if order_time_ms is not None else None,
        'm5_confirmed_entry': True,
        'entry_model': 'V10_MSS_M15_BOS_M5_LIMIT',
        'v10_position_state': position_state,
        'v10_order_created_at': order_created_at,
        'v10_filled_at': fill_created_at,
        'v10_fill_price': fill_price,
        'v10_pending_bars_elapsed': state.get('pending_bars_elapsed'),
        'v10_position_bars_held': state.get('position_bars_held'),
        'v10_max_hold_seconds': MAX_HOLD_M5 * 5 * 60,
        'v10_candidate_id': candidate_id,
    }


async def analyze_scan_structure(pair: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
    # Compatibility shim: the updated main.py calls generate_setup only once per pair.
    result = await generate_setup(pair, context)
    v10 = (result.get('analysis') or {}).get('v10', {})
    bias = str(v10.get('d1_bias') or 'NEUTRAL').upper()
    return {
        'trend': 'BULLISH' if bias == 'LONG' else 'BEARISH' if bias == 'SHORT' else 'RANGE',
        'btc_h4_trend': 'NOT_USED_BY_V10',
        'analysis': {'v10': v10},
    }


async def validate_setup(pair: str, initial_setup: dict[str, Any], context: dict[str, Any] | None = None) -> dict[str, Any]:
    setup = dict(initial_setup)
    if setup.get('strategy_name') != STRATEGY_NAME:
        return {'valid': False, 'reason': 'Strategy bukan Tactical V10 E_GE2_ATR', 'confidence': 0.0}
    candidate_state = str(setup.get('candidate_state') or '').upper()
    if candidate_state == 'WAITING_M5_BOS':
        try:
            direction = str(setup.get('direction') or '').upper()
            watch_id = str(setup.get('v10_watch_id') or setup.get('v10_candidate_id') or '')
            setup_id = int(setup.get('v10_setup_id'))
            poi_id = int(setup.get('v10_poi_id'))
            attempt_id = int(setup.get('v10_attempt_id'))
            mss_close_ms = int(setup.get('v10_mss_close_time_ms'))
            poi = setup.get('v10_poi')
            if direction not in {'BUY', 'SELL'} or not watch_id or len(watch_id) < 16:
                raise ValueError('Identitas kandidat V10 tidak lengkap')
            if min(setup_id, poi_id, attempt_id) <= 0 or mss_close_ms <= 0 or not isinstance(poi, list) or len(poi) != 2:
                raise ValueError('State AOI/POI/MSS kandidat V10 tidak lengkap')
            low, high = sorted((float(poi[0]), float(poi[1])))
            if not (np.isfinite(low) and np.isfinite(high) and low > 0 and high >= low):
                raise ValueError('POI kandidat V10 tidak valid')
        except Exception as exc:
            return {'valid': False, 'reason': str(exc), 'confidence': 0.0}
        setup['pair'] = str(pair).upper()
        return {
            'valid': True, 'setup': setup, 'confidence': RULE_PASS_SCORE,
            'reason': 'Lulus prasyarat D1/H4/M15 V10; menunggu fresh M5 BOS close-cross.',
            'reason_codes': [],
        }
    if candidate_state not in {'READY', 'FILLED'}:
        return {'valid': False, 'reason': 'V10 belum berada pada WAITING_M5_BOS atau state order/position aktif', 'confidence': 0.0}
    try:
        direction = str(setup['direction']).upper()
        entry, sl, tp = (float(setup[k]) for k in ('entry', 'sl', 'tp'))
        if not all(np.isfinite(x) and x > 0 for x in (entry, sl, tp)):
            raise ValueError('Harga bukan finite-positive')
        if direction == 'BUY' and not (sl < entry < tp):
            raise ValueError('Geometri LONG tidak valid')
        if direction == 'SELL' and not (tp < entry < sl):
            raise ValueError('Geometri SHORT tidak valid')
        if direction not in {'BUY', 'SELL'}:
            raise ValueError('Direction harus BUY/SELL')
    except Exception as exc:
        return {'valid': False, 'reason': str(exc), 'confidence': 0.0}
    setup['pair'] = str(pair).upper()
    return {
        'valid': True,
        'setup': setup,
        'confidence': float(setup.get('confidence') or RULE_PASS_SCORE),
        'reason': 'Lulus validasi geometri; tidak menambahkan filter di luar V10.',
        'reason_codes': [],
    }


def analyze_trailing(trade: dict[str, Any], context: dict[str, Any] | None = None) -> None:
    # Tactical V10 uses fixed SL and 1:2 TP; no trailing adjustment exists in the backtest.
    return None
