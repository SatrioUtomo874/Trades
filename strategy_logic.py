import logging
import pandas as pd
import numpy as np
from datetime import timezone, datetime

log = logging.getLogger(__name__)

# ============================================================
# STRATEGY_LOGIC.PY — "OTAK" v4
# ============================================================
# ALUR WAJIB (jangan diacak urutannya): BIAS ARAH -> ENTRY -> SL -> TP
# -> CONFIDENCE. Lihat masing-masing fungsi: _bias_direction(),
# _find_best_entry(), _place_sl(), _find_tp(), _compute_confidence(),
# dirangkai di full_analyze() persis dengan urutan itu.
#
# RINGKASAN PRINSIP (sesuai instruksi):
#
# 1) RR DINAMIS 2R–4R, TIDAK PERNAH AUTO-TOLAK RR<2.
#    _find_tp() cari level struktural TERDEKAT dulu. Kalau RR-nya <
#    MIN_RR (2.0), sistem TIDAK langsung menolak — ia mencari target
#    LANJUTAN yang lebih jauh (level struktural lain, lalu proyeksi
#    fib-extension H1) sampai ketemu yang RR>=2. Kalau target yang
#    ketemu itu RR-nya > MAX_RR (4.0), TP di-CAP persis di harga RR=4
#    (bukan ditolak) — sesuai instruksi "batasi hingga 1:4 saja".
#
# 2) ENTRY DITENTUKAN LEBIH DULU (sebelum SL & TP), pakai hierarki SMC
#    dari catatan (combined.txt) + referensi umum ICT/SMC: liquidity
#    sweep terkonfirmasi diprioritaskan PALING TINGGI (tier 1), baru
#    Order Block segar, FVG segar, OTE, equal-high/low pool, dan
#    fallback market entry paling akhir. Lihat _find_best_entry().
#
# 3) SL = INVALIDATION ASLI, BUKAN LEVEL YANG GAMPANG KE-LIQUIDITY-
#    SWEEP. Buffer SL dibedakan per tier entry (SL_BUFFER_ATR) — tier
#    yang belum terbukti ada sweep (OB/FVG) dapat buffer lebih lebar
#    & dicek dulu apakah ada level likuiditas lebih dalam yang lebih
#    masuk akal jadi invalidation (_extend_beyond_liquidity), supaya
#    SL tidak gampang tersapu sebentar sebelum harga lanjut sesuai
#    bias. Tier 1 (sweep sudah TERJADI & terbukti) dapat buffer paling
#    kecil karena invalidasinya sudah konkret, bukan spekulasi lagi.
#    Kalau SL tersentuh setelah semua ini, itu memang representasi
#    "analisa awal salah" — bukan jaminan mutlak (tidak ada strategi
#    manapun yang bisa menjamin itu), tapi peluangnya jauh lebih besar
#    dibanding SL yang asal ditaruh di bawah/atas swing terdekat.
#
# 4) TRAILING = MURNI MENGIKUTI MARKET STRUCTURE M15, BUKAN PROFIT-LOCK
#    LADDER. TRAIL_R_LADDER sengaja dikosongkan (lihat komentar di
#    konfigurasi di bawah) supaya SL trailing di main.py 100% berasal
#    dari swing low/high M15 terbaru (kandidat B di monitor_position),
#    bukan dari skema "kunci sekian % dari profit". Begitu SL trailing
#    tersentuh, artinya trend kehabisan tenaga (swing baru gagal
#    terbentuk), bukan sekadar profit-taking paksa.
#
# 5) SETIAP KOIN SELALU DIANALISA & DAPAT CONFIDENCE (bias arah selalu
#    dipilih salah satu sisi, entry selalu ketemu minimal via fallback
#    tier 6, TP selalu ketemu minimal via fallback ATR) — tidak ada
#    hard-gate yang membuang symbol sebelum sempat dinilai.
#
# 6) CONFIDENCE 100% GLOBAL — TIDAK ADA PENYESUAIAN SESI/KILLZONE SAMA
#    SEKALI. _compute_confidence() murni dari kualitas chart: bias
#    struktur H1+D1, kualitas tier entry, trigger M15 (CHoCH/CISD/
#    failed-retest/BOS/sweep), lokasi zona (discount/premium+OTE), dan
#    konfluensi H4+volume — sesi hanya field informasi di output,
#    sudah diverifikasi lewat test (lihat pesan saya) confidence identik
#    walau timestamp/sesi digeser tanpa mengubah harga sama sekali.
#
# CATATAN JUJUR (baca ini juga): tidak ada strategy_logic yang bisa
# menjamin SL selalu = pembalikan arah murni & TP selalu tercapai —
# itu di luar kemampuan analisa apapun karena market intrinsically
# probabilistik. Yang bisa direkayasa dengan baik adalah: (a) SL
# diletakkan di invalidation level yang punya alasan struktural jelas
# + sadar liquidity-sweep, bukan di angka acak; (b) TP diprioritaskan
# ke level likuiditas/struktur nyata yang REALISTIS dicapai dalam
# rentang RR 2-4, bukan target jauh spekulatif; (c) confidence
# mencerminkan seberapa BANYAK konfluensi yang searah, sehingga makin
# tinggi confidence, makin besar edge statistiknya — bukan jaminan
# menang di trade individual manapun.
# ============================================================

# ── RISK / REWARD — DINAMIS 2R s/d 4R ───────────────────────
# Tidak ada lagi hard-reject kalau level struktural terdekat RR<2.
# Alurnya (lihat _find_tp): kalau kandidat terdekat < MIN_RR, cari
# kandidat lebih jauh (atau proyeksi H1/fib-extension) sampai dapat
# yang >= MIN_RR. Kalau kandidat yg qualify itu RR-nya > MAX_RR,
# TP di-CAP persis di harga yang mewakili MAX_RR (bukan ditolak) —
# sesuai instruksi: "batasi hingga 1:4 saja".
MIN_RR = 2.0
MAX_RR = 4.0
# RR di atas ini dianggap level basi/tidak realistis untuk 1x target,
# tidak dipakai sebagai kandidat TP struktural (beda dgn MAX_RR yang
# men-cap TP terpilih; ini membatasi kandidat MANA yang boleh dilihat).
RR_CANDIDATE_CEILING = MAX_RR + 2.5

# ── TRAILING STOP — MURNI BERBASIS MARKET STRUCTURE ─────────
# PENTING (baca sebelum mengubah): trail BUKAN mekanisme mengunci %
# profit — itu filosofi versi lama yang sudah diminta diganti. Trail
# di sini murni berarti: "geser SL mengikuti swing struktur M15
# terbaru", supaya begitu SL trailing tersentuh artinya trend
# sebelumnya sudah benar-benar kehabisan tenaga (swing baru gagal
# terbentuk / harga jebol swing sebelumnya) — BUKAN sekadar profit
# taking paksa.
#
# main.py (monitor_position) mengambil kandidat SL PALING PROTEKTIF
# antara (A) R-multiple ladder TRAIL_R_LADDER dan (B) structure-based
# dari STRUCT_TRAIL_*. Supaya trail 100% murni structure-based sesuai
# instruksi, TRAIL_R_LADDER sengaja DIKOSONGKAN — dengan begitu
# kandidat (A) tidak pernah aktif, dan SL trailing SELALU murni
# hasil kandidat (B): swing low/high M15 terbaru minus/plus buffer.
# (Ini satu-satunya cara mematikan komponen R-ladder tanpa mengubah
# main.py, karena TRAIL_R_LADDER dikonsumsi via `from strategy_logic
# import *`.)
TRAIL_R_LADDER = []

# STRUCT_TRAIL_LB: jumlah candle kiri-kanan utk konfirmasi swing point
# M15 yang dipakai trailing. 3 dipilih sebagai titik tengah: cukup
# signifikan (bukan noise 1-2 candle) tapi tidak terlalu lambat
# terkonfirmasi (yang bikin trail ketinggalan jauh dari harga).
STRUCT_TRAIL_LB = 3
# STRUCT_TRAIL_BUF_PCT: buffer di BAWAH swing low (BUY) / DI ATAS
# swing high (SELL) supaya trail SL tidak kena wick liquidity-sweep
# yang wajar terhadap swing itu sendiri (swing terbaru M15 juga rawan
# di-sweep sedikit sebelum trend lanjut). 0.3% dipilih lebih lebar
# dari default lama (0.15%) berdasarkan observasi noise wick M15
# crypto pada umumnya cukup untuk swing minor, tapi INI PERKIRAAN
# TETAP (main.py hardcode-nya sbg %, bukan ATR-adaptif per-koin) —
# lihat catatan "SARAN OPSIONAL main.py" di akhir file/pesan saya
# untuk cara membuatnya adaptif per-koin kalau kamu mau presisi lebih.
STRUCT_TRAIL_BUF_PCT = 0.003
STRUCT_TRAIL_LOOKBACK = 60

# FIBONACCI EXTENSION (proyeksi TP kalau level struktural M15/H1 tidak
# cukup jauh utk capai MIN_RR — lihat _find_tp)
FIB_EXT_1 = 0.272
FIB_EXT_2 = 0.618

# H4 RSI GATE (konfluensi tambahan utk bonus confidence & syarat
# proyeksi fib-extension TP — BUKAN gate keseluruhan sinyal)
H4_RSI_BUY_MIN  = 40
H4_RSI_BUY_MAX  = 65
H4_RSI_SELL_MIN = 35
H4_RSI_SELL_MAX = 60

# ── SESSION / KILLZONE ───────────────────────────────────────
# PENTING: sesuai instruksi, sesi/killzone TIDAK LAGI mempengaruhi
# confidence sama sekali (tidak ada bonus/penalti aditif). Field ini
# hanya dihitung untuk INFORMASI di pesan sinyal (main.py menampilkan
# "Sesi: ..."), confidence sepenuhnya global berdasarkan kualitas
# chart & konfluensi teknikal semata.
SESSION_NY_START     = 13
SESSION_NY_END       = 17
SESSION_LONDON_START = 7
SESSION_LONDON_END   = 12
SESSION_KILL_LDN_S   = 7
SESSION_KILL_LDN_E   = 10
SESSION_KILL_ASIA1_S = 20
SESSION_KILL_ASIA2_E = 5

# ============================================================
# FUNGSI BANTU (INDIKATOR & STRUKTUR)
# ============================================================

def ema(s, n): return s.ewm(span=n, adjust=False).mean()

def rsi(s, n=14):
    d = s.diff()
    g = d.clip(lower=0).rolling(n).mean()
    l = (-d.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + g / l.replace(0, np.nan))

def macd(s):
    line = ema(s, 12) - ema(s, 26)
    sig = ema(line, 9)
    return line, sig, line - sig

def atr_fn(df, n=14):
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"] - df["close"].shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def build_df(df):
    if len(df) < 60: return None
    df = df.copy()
    df["ema9"] = ema(df["close"], 9)
    df["ema21"] = ema(df["close"], 21)
    df["ema50"] = ema(df["close"], 50)
    df["ema200"] = ema(df["close"], 200) if len(df) >= 200 else ema(df["close"], 50)
    df["rsi"] = rsi(df["close"])
    df["ml"], df["ms"], df["mh"] = macd(df["close"])
    df["atr"] = atr_fn(df)
    df["atr_ma"] = df["atr"].rolling(50, min_periods=10).mean()
    df["vol_sma"] = df["volume"].rolling(20).mean()
    bm = df["close"].rolling(20).mean()
    bs = df["close"].rolling(20).std()
    df["bb_up"] = bm + 2 * bs
    df["bb_lo"] = bm - 2 * bs
    df["bb_mid"] = bm
    return df.dropna(subset=["ema9", "ema21", "ema50", "rsi", "atr"])

def swing_pts(df, lb=5):
    """Swing high/low pivot — dipakai juga oleh main.py (structure trail),
    JANGAN ubah signature (df, lb) & return type (list index, list index)."""
    sh, sl = [], []
    for i in range(lb, len(df) - lb):
        if df["high"].iloc[i] == df["high"].iloc[i - lb:i + lb + 1].max():
            sh.append(i)
        if df["low"].iloc[i] == df["low"].iloc[i - lb:i + lb + 1].min():
            sl.append(i)
    return sh, sl

def mkt_struct(df, sh, sl):
    if len(sh) < 2 or len(sl) < 2: return "ranging"
    hh = df["high"].iloc[sh[-1]] > df["high"].iloc[sh[-2]]
    hl = df["low"].iloc[sl[-1]] > df["low"].iloc[sl[-2]]
    lh = df["high"].iloc[sh[-1]] < df["high"].iloc[sh[-2]]
    ll = df["low"].iloc[sl[-1]] < df["low"].iloc[sl[-2]]
    if hh and hl: return "bullish"
    if lh and ll: return "bearish"
    return "ranging"

# ============================================================
# SESSION / KILLZONE — DIPAKAI UNTUK BONUS SKOR, BUKAN GATE
# ============================================================

def _to_utc_hour(bar_ts):
    try:
        if bar_ts is not None:
            if hasattr(bar_ts, 'tzinfo') and bar_ts.tzinfo is None:
                bar_ts = bar_ts.tz_localize("UTC")
            elif hasattr(bar_ts, 'tzinfo') and bar_ts.tzinfo is not None:
                bar_ts = bar_ts.tz_convert("UTC")
            return bar_ts.hour
        return datetime.now(timezone.utc).hour
    except Exception:
        return None

def _get_session(bar_ts=None):
    hour = _to_utc_hour(bar_ts)
    if hour is None: return "transition"
    if SESSION_NY_START <= hour < SESSION_NY_END: return "NY"
    if SESSION_LONDON_START <= hour < SESSION_LONDON_END: return "London"
    if hour >= SESSION_KILL_ASIA1_S or hour < SESSION_KILL_ASIA2_E: return "Asia"
    return "transition"

def _is_in_killzone(bar_ts=None):
    hour = _to_utc_hour(bar_ts)
    if hour is None: return False
    ldn_kill = SESSION_KILL_LDN_S <= hour < SESSION_KILL_LDN_E
    asia_kill = hour >= SESSION_KILL_ASIA1_S or hour < SESSION_KILL_ASIA2_E
    return ldn_kill or asia_kill

# ============================================================
# DETEKSI STRUKTUR SMART MONEY (SMC)
# ============================================================

def detect_liquidity_sweep(df, sh, sl, direction):
    """Liquidity sweep: wick menembus swing level lalu close kembali di
    dalam range -> indikasi stop-hunt/manipulasi sebelum reversal."""
    result = {"type": "none", "level": None}
    if direction == "bull" and len(sl) >= 1:
        low = df["low"].iloc[sl[-1]]
        if df["low"].iloc[-1] < low and df["close"].iloc[-1] > low:
            result = {"type": "sweep", "level": low}
    elif direction == "bear" and len(sh) >= 1:
        high = df["high"].iloc[sh[-1]]
        if df["high"].iloc[-1] > high and df["close"].iloc[-1] < high:
            result = {"type": "sweep", "level": high}
    return result

def detect_break_of_structure(df, sh, sl, direction):
    """BOS — WAJIB konfirmasi CLOSE candle melewati level (bukan cuma
    wick), sesuai prinsip: breakout baru valid kalau ada candle yang
    benar-benar close melewati level tersebut."""
    if direction == "bull" and len(sh) >= 2:
        prev_high = df["high"].iloc[sh[-2]]
        if df["close"].iloc[-1] > prev_high:
            return True
    elif direction == "bear" and len(sl) >= 2:
        prev_low = df["low"].iloc[sl[-2]]
        if df["close"].iloc[-1] < prev_low:
            return True
    return False

def detect_choch(df, sh, sl):
    """Change of Character — konfirmasi body close, bukan wick."""
    result = {"bearish_choch": False, "bullish_choch": False}
    if len(sh) < 2 or len(sl) < 2: return result
    close = df["close"].iloc[-1]
    prev_high = df["high"].iloc[sh[-2]]
    last_high = df["high"].iloc[sh[-1]]
    prev_low = df["low"].iloc[sl[-2]]
    last_low = df["low"].iloc[sl[-1]]
    if last_high > prev_high and last_low > prev_low and close > prev_low:
        result["bullish_choch"] = True
    if last_high < prev_high and last_low < prev_low and close < prev_low:
        result["bearish_choch"] = True
    return result

def detect_cisd(df, lb=6):
    """CISD — deretan candle searah yang tiba-tiba dilawan candle
    penutup kuat berlawanan arah -> sinyal reversal paling awal."""
    result = {"bullish_cisd": False, "bearish_cisd": False}
    if len(df) < lb + 1: return result
    sub = df.iloc[-lb:]
    closes = sub["close"].values
    opens = sub["open"].values
    n = len(closes)
    last_bull = closes[-1] > opens[-1]
    last_bear = closes[-1] < opens[-1]
    if not (last_bull or last_bear): return result
    if last_bull:
        cnt = 0
        for j in range(n - 2, -1, -1):
            if closes[j] < opens[j]: cnt += 1
            else: break
        if cnt >= 3: result["bullish_cisd"] = True
    else:
        cnt = 0
        for j in range(n - 2, -1, -1):
            if closes[j] > opens[j]: cnt += 1
            else: break
        if cnt >= 3: result["bearish_cisd"] = True
    return result

def detect_fvg(df, direction, lb=40):
    """Fair Value Gap — ketidakseimbangan harga 3-candle."""
    sub = df.iloc[-lb:]
    base_offset = len(df) - len(sub)
    out = []
    for i in range(len(sub) - 2):
        c0, c1, c2 = sub.iloc[i], sub.iloc[i+1], sub.iloc[i+2]
        gap = None
        if direction == "bull" and c2["low"] > c0["high"]:
            gap = {"top": c2["low"], "bot": c0["high"]}
        elif direction == "bear" and c2["high"] < c0["low"]:
            gap = {"top": c0["low"], "bot": c2["high"]}
        if gap:
            gap["mid"] = (gap["top"] + gap["bot"]) / 2
            gap["idx"] = base_offset + i + 2
            gap["is_fresh"] = is_zone_fresh(df, gap["top"], gap["bot"], gap["idx"])
            out.append(gap)
    return out[-3:] if out else []

def detect_order_block(df, direction, lb=40):
    """Order Block — candle terakhir sebelum impulsive move berlawanan
    arah candle tsb, dinilai quality-nya dari FVG+BOS+freshness."""
    is_demand = direction == "bull"
    sub = df.iloc[-lb:]
    base_offset = len(df) - len(sub)
    avg_body = (sub["close"] - sub["open"]).abs().mean()
    zones = []
    for i in range(1, len(sub) - 2):
        c = sub.iloc[i]
        nx = sub.iloc[i+1]
        impulse_body = abs(nx["close"] - nx["open"])
        if impulse_body < avg_body * 1.3: continue
        is_match = (c["close"] < c["open"] and nx["close"] > nx["open"]) if is_demand else (c["close"] > c["open"] and nx["close"] < nx["open"])
        if not is_match: continue
        top = max(c["open"], c["close"])
        bot = min(c["open"], c["close"])
        df_idx = base_offset + i
        has_fvg = False
        if i + 2 < len(sub):
            c2 = sub.iloc[i+2]
            if is_demand and c2["low"] > c["high"]: has_fvg = True
            if not is_demand and c2["high"] < c["low"]: has_fvg = True
        sh, sl = swing_pts(df, lb=5)
        has_bos = detect_break_of_structure(df, sh, sl, direction)
        fresh = is_zone_fresh(df, top, bot, df_idx)
        quality = int(has_fvg) + int(has_bos) + int(fresh)
        if quality >= 2:
            zones.append({
                "top": top, "bot": bot, "mid": (top + bot) / 2,
                "idx": df_idx, "has_fvg": has_fvg, "has_bos": has_bos,
                "is_fresh": fresh, "quality": quality,
            })
    return zones[-3:] if zones else []

def detect_equal_highs_lows(df, kind="high", lb=60, tol=0.0025):
    """Equal Highs/Lows — kluster level yang berulang -> kolam likuiditas."""
    sub = df.iloc[-lb:]
    vals = sub["high"] if kind == "high" else sub["low"]
    clusters = []
    visited = set()
    for i in range(len(vals)):
        if i in visited: continue
        group = [vals.iloc[i]]
        for j in range(i + 1, len(vals)):
            if abs(vals.iloc[i] - vals.iloc[j]) / max(vals.iloc[i], 0.0001) < tol:
                group.append(vals.iloc[j])
                visited.add(j)
        if len(group) >= 2:
            clusters.append(sum(group) / len(group))
    return sorted(clusters)

def detect_failed_retest(df, sh, sl, atr):
    """Failed retest — harga menyentuh level lalu ditolak dengan candle
    yang close jelas menjauh dari level (bukan sekadar wick)."""
    result = {"failed_retest_sell": False, "failed_retest_buy": False}
    if len(df) < 3: return result
    L = df.iloc[-1]
    P = df.iloc[-2]
    if len(sh) >= 2:
        resistance = df["high"].iloc[sh[-2]]
        touched = P["high"] >= resistance - atr * 0.5
        rejected = L["close"] < resistance - atr * 0.3
        bearish_c = L["close"] < L["open"]
        if touched and rejected and bearish_c:
            result["failed_retest_sell"] = True
    if len(sl) >= 2:
        support = df["low"].iloc[sl[-2]]
        touched = P["low"] <= support + atr * 0.5
        bounced = L["close"] > support + atr * 0.3
        bullish_c = L["close"] > L["open"]
        if touched and bounced and bullish_c:
            result["failed_retest_buy"] = True
    return result

def is_zone_fresh(df, top, bot, formed_idx, end_idx=None):
    if formed_idx is None or formed_idx + 2 >= len(df): return True
    start = formed_idx + 2
    end_idx = end_idx if end_idx is not None else len(df) - 1
    if start >= end_idx: return True
    sub = df.iloc[start:end_idx]
    if sub.empty: return True
    touched = ((sub["low"] <= top) & (sub["high"] >= bot)).any()
    return not bool(touched)

# ============================================================
# FIBONACCI — DISCOUNT/PREMIUM ZONE & OTE
# ============================================================

def get_fib_zone(price, swing_low, swing_high):
    """Premium/Discount/Equilibrium relatif terhadap swing range aktif.
    Prinsip SMC: beli di zona diskon (di bawah 0.5 dari range), jual di
    zona premium (di atas 0.5 dari range)."""
    rng = swing_high - swing_low
    if rng <= 0: return {"ratio": 0.5, "zone": "equilibrium"}
    ratio = (price - swing_low) / rng
    if ratio <= 0.45: zone = "discount"
    elif ratio >= 0.55: zone = "premium"
    else: zone = "equilibrium"
    return {"ratio": round(ratio, 4), "zone": zone}

def is_in_ote(df, direction, sh, sl):
    """Optimal Trade Entry — retracement 0.62-0.79 dari swing leg
    terakhir, area yang paling sering dipakai smart money re-entry."""
    if len(sh) < 1 or len(sl) < 1: return False
    swing_high = df["high"].iloc[sh[-1]]
    swing_low = df["low"].iloc[sl[-1]]
    fib = get_fib_zone(df["close"].iloc[-1], swing_low, swing_high)
    if direction == "bull":
        return 0.62 <= (1 - fib["ratio"]) <= 0.79
    else:
        return 0.62 <= fib["ratio"] <= 0.79

def _fib_extension_levels(h1, sh1, sl1, direction):
    if not sh1 or not sl1: return None, None
    swing_high = h1["high"].iloc[sh1[-1]]
    swing_low = h1["low"].iloc[sl1[-1]]
    leg = swing_high - swing_low
    if leg <= 0: return None, None
    if direction == "bull":
        return swing_high + leg * FIB_EXT_1, swing_high + leg * FIB_EXT_2
    else:
        return swing_low - leg * FIB_EXT_1, swing_low - leg * FIB_EXT_2

# ============================================================
# H4 CONFLUENCE GATE (bonus skor + syarat tambahan Fib-extension TP)
# ============================================================

def _h4_confluence(df_h1, direction, choch_m15=None):
    result = {"confluence": False, "full_confluence": False}
    try:
        df_h4 = build_df(df_h1.resample("4h").agg({
            "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
        }).dropna())
        if df_h4 is None or len(df_h4) < 20: return result
        L4 = df_h4.iloc[-1]
        sh4, sl4 = swing_pts(df_h4, lb=3)
        struct_h4 = mkt_struct(df_h4, sh4, sl4)
        rsi_h4 = L4["rsi"]
        if direction == "bull":
            ema_ok = L4["ema9"] > L4["ema21"] > L4["ema50"]
            struct_ok = struct_h4 == "bullish"
            rsi_ok = H4_RSI_BUY_MIN <= rsi_h4 <= H4_RSI_BUY_MAX
        else:
            ema_ok = L4["ema9"] < L4["ema21"] < L4["ema50"]
            struct_ok = struct_h4 == "bearish"
            rsi_ok = H4_RSI_SELL_MIN <= rsi_h4 <= H4_RSI_SELL_MAX
        result["confluence"] = bool(ema_ok and struct_ok and rsi_ok)
        if result["confluence"] and choch_m15:
            choch_agrees = (
                (direction == "bull" and choch_m15.get("bullish_choch")) or
                (direction == "bear" and choch_m15.get("bearish_choch"))
            )
            result["full_confluence"] = bool(choch_agrees)
    except Exception:
        pass
    return result

def _volatility_regime_penalty(m15):
    """Penalti kecil non-directional kalau kondisi market sedang tidak
    ideal untuk dianalisa: terlalu sepi/choppy (ATR jauh di bawah rata2
    -> range sempit, gampang whipsaw) atau terlalu liar (ATR jauh di
    atas rata2 -> kemungkinan spike berita, invalidation kurang bisa
    diandalkan). Ini BUKAN gate — cuma mengurangi confidence sedikit."""
    try:
        atr_now = m15["atr"].iloc[-1]
        atr_ma = m15["atr_ma"].iloc[-1]
        if atr_ma is None or atr_ma <= 0 or pd.isna(atr_ma): return 0
        ratio = atr_now / atr_ma
        if ratio < 0.55: return -6      # terlalu sepi/choppy
        if ratio > 2.3: return -5       # kemungkinan spike/berita
        return 0
    except Exception:
        return 0


# ============================================================
# STEP 0 — BIAS ARAH (top-down context, BUKAN confidence)
# ============================================================
# Ini cuma menentukan "kita nyari entry BUY atau SELL" — sesuai cara
# kerja SMC top-down yang benar: HTF (H1/D1) kasih bias, LTF (M15)
# kasih entry presisi searah bias itu. Ini BUKAN langkah yang menolak
# symbol (selalu balik salah satu arah) dan BUKAN bagian dari nilai
# confidence — confidence dihitung belakangan dari kualitas entry+SL+TP
# yang benar-benar ditemukan (lihat _compute_confidence).

def _get_d1_bias(df_h1, df_d1):
    try:
        if df_d1 is not None and len(df_d1) >= 65:
            df_d1_built = build_df(df_d1)
        else:
            df_d1_built = build_df(df_h1.resample("1D").agg({
                "open": "first", "high": "max", "low": "min",
                "close": "last", "volume": "sum"
            }).dropna())
        if df_d1_built is not None and len(df_d1_built) >= 10:
            LD = df_d1_built.iloc[-1]
            sh_d, sl_d = swing_pts(df_d1_built, lb=3)
            struct_d1 = mkt_struct(df_d1_built, sh_d, sl_d)
            ema_bear_d1 = LD["ema9"] < LD["ema21"] < LD["ema50"]
            ema_bull_d1 = LD["ema9"] > LD["ema21"] > LD["ema50"]
            if struct_d1 == "bearish" or ema_bear_d1: return "bearish", df_d1_built
            if struct_d1 == "bullish" or ema_bull_d1: return "bullish", df_d1_built
        return "neutral", df_d1_built
    except Exception:
        return "neutral", None

def _bias_direction(h1, df_d1_built, struct_h1, choch_h1):
    """Selalu mengembalikan 'bull' atau 'bear' — tidak pernah None,
    supaya setiap koin pasti punya arah yang dicari entry-nya."""
    if struct_h1 == "bullish": return "bull"
    if struct_h1 == "bearish": return "bear"
    if df_d1_built is not None and len(df_d1_built) >= 1:
        LD = df_d1_built.iloc[-1]
        if LD["ema9"] > LD["ema21"] > LD["ema50"]: return "bull"
        if LD["ema9"] < LD["ema21"] < LD["ema50"]: return "bear"
    if choch_h1.get("bullish_choch"): return "bull"
    if choch_h1.get("bearish_choch"): return "bear"
    L1 = h1.iloc[-1]
    return "bull" if L1["ema9"] >= L1["ema21"] else "bear"

# ============================================================
# STEP 1 — ENTRY (hierarki SMC, liquidity-sweep diprioritaskan)
# ============================================================
# Urutan prioritas (tier 1 = paling diutamakan), sesuai catatan/ilmu
# entry SMC (order block, FVG, liquidity sweep, OTE):
#   1. Liquidity sweep TERKONFIRMASI (candle terakhir sudah nge-wick
#      lewat swing level & close kembali masuk) — entry "smart money"
#      paling kuat karena stop-hunt-nya SUDAH terjadi, bukan spekulasi.
#   2. Order Block M15 segar (belum termitigasi) kualitas terbaik.
#   3. Fair Value Gap M15 segar.
#   4. OTE (retracement 0.618-0.79 dari swing leg M15 terakhir).
#   5. Equal High/Low pool yang BELUM di-sweep (antisipasi sweep).
#   6. Fallback market entry (kalau semua di atas tidak ada — supaya
#      symbol tetap teranalisa, bukan malah tidak ada sinyal).
ENTRY_TIER_SCORE = {1: 25, 2: 20, 3: 15, 4: 11, 5: 8, 6: 4}

def _reasonable_dist(current_price, entry_pt, atr, mult=6.0):
    return abs(current_price - entry_pt) < atr * mult

def _find_best_entry(m15, h1, direction, current_price, atr, sh15, sl15, sh1, sl1):
    up = direction == "bull"

    # TIER 1 — liquidity sweep
    sweep = detect_liquidity_sweep(m15, sh15, sl15, direction)
    if sweep["type"] == "sweep" and sweep["level"] is not None:
        return {"price": current_price, "label": "liquidity_sweep", "tier": 1,
                "invalid_base": sweep["level"], "quality": 3}

    # TIER 2 — Order Block M15 segar (utamakan quality tinggi, lalu terdekat)
    obs = [z for z in detect_order_block(m15, direction, lb=40) if z["is_fresh"]]
    if obs:
        entry_of = lambda z: z["top"] if up else z["bot"]
        obs_sorted = sorted(obs, key=lambda z: (-z["quality"], abs(current_price - entry_of(z))))
        best_ob = obs_sorted[0]
        entry_pt = entry_of(best_ob)
        invalid_pt = best_ob["bot"] if up else best_ob["top"]
        if _reasonable_dist(current_price, entry_pt, atr):
            return {"price": entry_pt, "label": "order_block", "tier": 2,
                    "invalid_base": invalid_pt, "quality": best_ob["quality"]}

    # TIER 3 — FVG M15 segar (paling baru/terdekat)
    fvgs = [f for f in detect_fvg(m15, direction, lb=40) if f["is_fresh"]]
    if fvgs:
        f = fvgs[-1]
        entry_pt = f["mid"]
        invalid_pt = f["bot"] if up else f["top"]
        if _reasonable_dist(current_price, entry_pt, atr):
            return {"price": entry_pt, "label": "fvg", "tier": 3,
                    "invalid_base": invalid_pt, "quality": 2}

    # TIER 4 — OTE zone dari swing leg M15 terakhir
    if len(sh15) >= 1 and len(sl15) >= 1 and is_in_ote(m15, direction, sh15, sl15):
        leg_start = m15["low"].iloc[sl15[-1]] if up else m15["high"].iloc[sh15[-1]]
        return {"price": current_price, "label": "ote", "tier": 4,
                "invalid_base": leg_start, "quality": 1}

    # TIER 5 — Equal High/Low pool (liquidity belum tersapu)
    eqs = detect_equal_highs_lows(m15, "low" if up else "high", lb=80)
    if up:
        cands = [e for e in eqs if e < current_price]
        eq = max(cands) if cands else None
    else:
        cands = [e for e in eqs if e > current_price]
        eq = min(cands) if cands else None
    if eq is not None and _reasonable_dist(current_price, eq, atr):
        invalid_pt = eq - atr * 0.8 if up else eq + atr * 0.8
        return {"price": eq, "label": "eq_pool", "tier": 5,
                "invalid_base": invalid_pt, "quality": 1}

    # TIER 6 — fallback market entry (jaminan symbol tetap teranalisa)
    invalid_pt = current_price - atr * 1.2 if up else current_price + atr * 1.2
    return {"price": current_price, "label": "market", "tier": 6,
            "invalid_base": invalid_pt, "quality": 0}

# ============================================================
# STEP 2 — SL (invalidation ASLI, sadar liquidity sweep)
# ============================================================
# Filosofi (sesuai instruksi): SL yang tersentuh harus benar-benar
# berarti "analisa awal salah", BUKAN cuma kena liquidity sweep biasa.
# Makanya buffer-nya dibedakan per tier entry — makin "mentah" alasan
# invalidation-nya (belum terbukti ada sweep), makin lebar buffer-nya,
# dan untuk tier OB/FVG kita cek juga apakah ada level likuiditas LEBIH
# DALAM yang lebih masuk akal jadi invalidation asli (supaya SL tidak
# gampang kena wick sweep dangkal sebelum harga lanjut sesuai bias).
def _extend_beyond_liquidity(direction, base_invalid, atr, df, sh, sl_pts, search_mult=1.6):
    up = direction == "bull"
    try:
        if up and sl_pts:
            deeper = [df["low"].iloc[i] for i in sl_pts
                      if df["low"].iloc[i] < base_invalid
                      and df["low"].iloc[i] > base_invalid - atr * search_mult]
            if deeper: return min(deeper)
        if not up and sh:
            deeper = [df["high"].iloc[i] for i in sh
                      if df["high"].iloc[i] > base_invalid
                      and df["high"].iloc[i] < base_invalid + atr * search_mult]
            if deeper: return max(deeper)
    except Exception:
        pass
    return base_invalid

# Buffer per tier (dalam satuan ATR) — tier 1 (sweep sudah terbukti
# terjadi) butuh buffer paling kecil; tier yang lebih spekulatif
# (belum ada bukti sweep) butuh buffer lebih lebar.
SL_BUFFER_ATR = {1: 0.15, 2: 0.6, 3: 0.7, 4: 0.5, 5: 0.8, 6: 1.2}

def _place_sl(direction, entry_meta, atr, m15, sh15, sl15):
    up = direction == "bull"
    tier = entry_meta["tier"]
    base = entry_meta["invalid_base"]
    entry_price = entry_meta["price"]

    if tier in (2, 3):   # OB & FVG -> cek liquidity pool lebih dalam
        base = _extend_beyond_liquidity(direction, base, atr, m15, sh15, sl15)

    buf = atr * SL_BUFFER_ATR.get(tier, 0.8)
    sl_price = base - buf if up else base + buf

    # SAFETY-NET arah — SL wajib di sisi risk, apapun yang terjadi
    if up and sl_price >= entry_price:
        sl_price = entry_price - atr * 1.2
    elif not up and sl_price <= entry_price:
        sl_price = entry_price + atr * 1.2

    risk = abs(entry_price - sl_price)
    risk_floor = max(atr * 0.8, entry_price * 0.003)
    if risk < risk_floor:
        sl_price = entry_price - risk_floor if up else entry_price + risk_floor
        risk = risk_floor
    return round(sl_price, 8), risk

# ============================================================
# STEP 3 — TP (eskalasi RR 2R–4R, tidak pernah auto-tolak RR<2)
# ============================================================
def _find_tp(direction, entry_price, risk, m15, h1, sh15, sl15, sh1, sl1,
             fib127, fib162):
    up = direction == "bull"
    sgn = 1 if up else -1
    opp = "bear" if up else "bull"   # OB/FVG lawan arah = target likuiditas
    pool = []

    for v in detect_equal_highs_lows(m15, "high" if up else "low", lb=80):
        if sgn * (v - entry_price) > 0: pool.append(("eq_m15", v))
    for z in detect_order_block(m15, opp, lb=40):
        edge = z["bot"] if up else z["top"]
        if sgn * (edge - entry_price) > 0: pool.append(("ob_m15", edge))
    for f in detect_fvg(m15, opp, lb=40):
        if sgn * (f["mid"] - entry_price) > 0: pool.append(("fvg_m15", f["mid"]))
    for v in detect_equal_highs_lows(h1, "high" if up else "low", lb=100):
        if sgn * (v - entry_price) > 0: pool.append(("eq_h1", v))
    for z in detect_order_block(h1, opp, lb=80):
        edge = z["bot"] if up else z["top"]
        if sgn * (edge - entry_price) > 0: pool.append(("ob_h1", edge))
    for i in (sh1 if up else sl1):
        v = h1["high" if up else "low"].iloc[i]
        if sgn * (v - entry_price) > 0: pool.append(("sw_h1", v))

    scored = [(lbl, v, abs(v - entry_price) / risk) for lbl, v in pool]
    scored = [c for c in scored if 0.3 <= c[2] <= RR_CANDIDATE_CEILING]
    scored.sort(key=lambda c: c[2])   # nearest (RR terkecil) duluan

    chosen = next((c for c in scored if c[2] >= MIN_RR), None)

    if chosen is None:
        # ── ESKALASI: RR<2 dari level terdekat -> cari target lanjutan ──
        ext = []
        if fib127 is not None:
            rr = abs(fib127 - entry_price) / risk
            if rr >= MIN_RR: ext.append(("fib_ext_127", fib127, rr))
        if fib162 is not None:
            rr = abs(fib162 - entry_price) / risk
            if rr >= MIN_RR: ext.append(("fib_ext_162", fib162, rr))
        if ext:
            ext.sort(key=lambda c: c[2])
            chosen = ext[0]
        else:
            # Tidak ada target lanjutan yang masuk akal sama sekali ->
            # fallback ATR murni, tetap jamin RR >= MIN_RR (2.0).
            tp_price = entry_price + sgn * risk * (MIN_RR + 0.2)
            return {"tp": round(tp_price, 8), "label": "atr_fallback",
                    "rr": round(MIN_RR + 0.2, 2), "weak_target": True, "capped": False}

    lbl, v, rr_c = chosen
    if rr_c > MAX_RR:
        # Target lanjutan kejauhan -> batasi persis di RR 1:4
        tp_price = entry_price + sgn * risk * MAX_RR
        return {"tp": round(tp_price, 8), "label": f"{lbl}_capped4R",
                "rr": MAX_RR, "weak_target": False, "capped": True}

    return {"tp": round(v, 8), "label": lbl, "rr": round(rr_c, 2),
            "weak_target": False, "capped": False}

# ============================================================
# STEP 4 — CONFIDENCE (GLOBAL — TANPA penyesuaian sesi sama sekali)
# ============================================================
# Sesuai instruksi eksplisit: confidence murni dari kualitas chart
# (bias + entry + trigger + zone + konfluensi), sesi/killzone HANYA
# informasi tampilan, tidak menambah/mengurangi angka ini lagi.
def _compute_confidence(direction, struct_h1, L1, d1_bias, choch_h1, choch_m15,
                         cisd_m15, fr, liq_bull, liq_bear, entry_meta, tp_meta,
                         h1, m15, sh1, sl1, sh15, sl15, h4_gate, L15):
    up = direction == "bull"

    # BIAS H1 + D1 (maks 35)
    bias = 0
    if (up and struct_h1 == "bullish") or (not up and struct_h1 == "bearish"): bias += 20
    elif struct_h1 == "ranging": bias += 6
    ema_align = (L1["ema9"] > L1["ema21"] > L1["ema50"]) if up else (L1["ema9"] < L1["ema21"] < L1["ema50"])
    if ema_align: bias += 8
    if (up and d1_bias == "bullish") or (not up and d1_bias == "bearish"): bias += 7
    if (up and choch_h1.get("bullish_choch")) or (not up and choch_h1.get("bearish_choch")): bias += 8
    bias = min(bias, 35)

    # KUALITAS ENTRY (maks ~28) — tier + bonus quality OB
    entry_score = min(ENTRY_TIER_SCORE.get(entry_meta["tier"], 4) + min(entry_meta.get("quality", 0), 3), 28)

    # TRIGGER M15 (maks 30)
    trig = 0
    if (up and choch_m15.get("bullish_choch")) or (not up and choch_m15.get("bearish_choch")): trig += 14
    if (up and cisd_m15.get("bullish_cisd")) or (not up and cisd_m15.get("bearish_cisd")): trig += 8
    if (up and fr.get("failed_retest_buy")) or (not up and fr.get("failed_retest_sell")): trig += 9
    sweep_ctx = liq_bull if up else liq_bear
    if sweep_ctx["type"] == "sweep": trig += 5
    if detect_break_of_structure(m15, sh15, sl15, direction): trig += 3
    if struct_h1 != "ranging" and ((up and struct_h1 == "bearish") or (not up and struct_h1 == "bullish")):
        trig *= 0.6   # bias H1 & trigger M15 berlawanan -> lemahkan (soft, bukan gate)
    trig = min(trig, 30)

    # ZONE — discount/premium H1 + OTE M15 (maks 15)
    zone = 0
    if len(sh1) >= 1 and len(sl1) >= 1:
        fib_h1 = get_fib_zone(L15["close"], h1["low"].iloc[sl1[-1]], h1["high"].iloc[sh1[-1]])
        if (up and fib_h1["zone"] == "discount") or (not up and fib_h1["zone"] == "premium"): zone += 8
    if is_in_ote(m15, direction, sh15, sl15): zone += 7
    zone = min(zone, 15)

    # KONFLUENSI H4 + volume (maks 10)
    conf_bonus = 0
    if h4_gate["confluence"]: conf_bonus += 6
    vol_ok = L15["volume"] > L15["vol_sma"] * 1.1 if pd.notna(L15.get("vol_sma")) else False
    trig_dir_ok = (L15["close"] > L15["open"]) if up else (L15["close"] < L15["open"])
    if vol_ok and trig_dir_ok: conf_bonus += 4
    conf_bonus = min(conf_bonus, 10)

    total = bias + entry_score + trig + zone + conf_bonus

    # Penalti GLOBAL (bukan sesi): konflik D1, regime volatilitas, kualitas TP
    d1_conflict = (d1_bias == "bearish" and up) or (d1_bias == "bullish" and not up)
    if d1_conflict: total -= 10
    total += _volatility_regime_penalty(m15)
    if tp_meta.get("capped"): total -= 4       # target asli lebih jauh dari yg diklaim
    if tp_meta.get("weak_target"): total -= 8  # tidak ada target struktural sama sekali

    total = max(0, min(int(round(total)), 99))
    breakdown = {
        "bias": round(bias, 1), "entry_quality": round(entry_score, 1),
        "trigger": round(trig, 1), "zone": zone, "confluence": conf_bonus,
        "d1_conflict": d1_conflict, "tp_capped": tp_meta.get("capped", False),
        "tp_weak": tp_meta.get("weak_target", False),
    }
    return total, breakdown

# ============================================================
# FUNGSI UTAMA — full_analyze()
# Urutan WAJIB: bias arah -> ENTRY -> SL -> TP -> CONFIDENCE (global)
# ============================================================
def full_analyze(df_h1, df_m15, df_d1=None, symbol=None):
    try:
        if df_h1 is None or df_m15 is None or df_h1.empty or df_m15.empty:
            return None
        h1 = build_df(df_h1)
        m15 = build_df(df_m15)
        if h1 is None or m15 is None: return None

        L1, L15 = h1.iloc[-1], m15.iloc[-1]
        atr = max(L15["atr"], L15["close"] * 0.003)
        current_price = L15["close"]

        sh1, sl1 = swing_pts(h1, 5)
        sh15, sl15 = swing_pts(m15, 5)
        struct_h1 = mkt_struct(h1, sh1, sl1)
        choch_h1 = detect_choch(h1, sh1, sl1)
        choch_m15 = detect_choch(m15, sh15, sl15)
        cisd_m15 = detect_cisd(m15, lb=8)
        fr = detect_failed_retest(m15, sh15, sl15, atr)
        liq_bull = detect_liquidity_sweep(m15, sh15, sl15, "bull")
        liq_bear = detect_liquidity_sweep(m15, sh15, sl15, "bear")

        d1_bias, df_d1_built = _get_d1_bias(df_h1, df_d1)

        # STEP 0: arah (top-down bias, bukan confidence)
        direction = _bias_direction(h1, df_d1_built, struct_h1, choch_h1)

        session = _get_session(m15.index[-1])
        in_killzone = _is_in_killzone(m15.index[-1])

        # STEP 1: ENTRY
        entry_meta = _find_best_entry(m15, h1, direction, current_price, atr, sh15, sl15, sh1, sl1)

        # STEP 2: SL
        sl_price, risk = _place_sl(direction, entry_meta, atr, m15, sh15, sl15)
        if risk <= 0: return None

        # STEP 3: TP (eskalasi RR 2-4)
        h4_gate = _h4_confluence(df_h1, direction, choch_m15)
        fib127, fib162 = _fib_extension_levels(h1, sh1, sl1, direction)
        tp_meta = _find_tp(direction, entry_meta["price"], risk, m15, h1, sh15, sl15, sh1, sl1, fib127, fib162)

        # Sanity akhir: harga sekarang belum boleh sudah lewat TP (sinyal basi)
        if direction == "bull" and current_price >= tp_meta["tp"]: return None
        if direction == "bear" and current_price <= tp_meta["tp"]: return None

        # STEP 4: CONFIDENCE (global — tanpa sesi)
        confidence, breakdown = _compute_confidence(
            direction, struct_h1, L1, d1_bias, choch_h1, choch_m15, cisd_m15, fr,
            liq_bull, liq_bear, entry_meta, tp_meta, h1, m15, sh1, sl1, sh15, sl15,
            h4_gate, L15)

        return {
            "symbol": symbol,
            "original_dir": direction,
            "decision": "BUY" if direction == "bull" else "SELL",
            "confidence": confidence,
            "price": current_price,
            "entry": entry_meta["price"],
            "entry_label": entry_meta["label"],
            "sl": round(sl_price, 8),
            "tp": round(tp_meta["tp"], 8),
            "rr": tp_meta["rr"],
            "rsi": round(L15["rsi"], 1),
            "struct_h1": struct_h1,
            "d1_bias": d1_bias,
            "choch_m15": choch_m15,
            "choch_h1": choch_h1,
            "cisd_m15": cisd_m15,
            "failed_retest": fr,
            "session": session,
            "in_killzone": in_killzone,
            "tp_sl_reason": (f"Entry@{entry_meta['price']:.5g}({entry_meta['label']}) | "
                              f"SL@{sl_price:.5g} | TP@{tp_meta['tp']:.5g}"
                              f"({tp_meta['label']}, RR {tp_meta['rr']})"),
            "score_breakdown": breakdown,
        }
    except Exception as e:
        log.debug(f"[full_analyze] {symbol}: {e}")
        return None

def get_best_signal(candidates):
    """Pilih sinyal terbaik lintas-symbol: confidence sebagai basis
    utama, RR & kualitas entry (tier) sebagai tie-breaker."""
    if not candidates: return None
    def _rank(sig):
        tier_bonus = {"liquidity_sweep": 4, "order_block": 3, "fvg": 2, "ote": 1}.get(sig.get("entry_label"), 0)
        return sig["confidence"] * 2 + sig["rr"] + tier_bonus
    return max(candidates, key=_rank)
