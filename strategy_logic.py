"""
strategy_logic.py — OTAK (logika analisa, swappable)

Murni fungsi analisa: indikator, SMC (BOS/CHoCH/OB/FVG/liquidity sweep),
scoring arah sinyal, entry/SL/TP. Tidak ada kode Telegram/API/state —
supaya aman diganti lewat /ganti tanpa menyentuh Mesin.

Interface: full_analyze(df_h1, df_m15, df_d1, symbol=None) -> dict | None
+ konstanta tuning: MIN_RR, TRAIL_R_LADDER, STRUCT_TRAIL_*, FIB_EXT_*, H4_RSI_*

Note: full_analyze() tidak fetch data sendiri (dikirim Mesin) — hindari
circular import ke main.py.
"""

import logging
import pandas as pd
import numpy as np

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Ambang minimum risk-reward — sinyal dengan RR di bawah ini ditolak
MIN_RR = 2.0

# TRAIL_R_LADDER = [(ambang_R, lock_ratio), ...] — begitu profit (dlm
# kelipatan risk R trade itu sendiri) capai ambang, kunci lock_ratio dari
# level itu (dlm R juga). SL final = kandidat PALING PROTEKTIF antara
# R-ladder ini vs komponen structure (swing-point) di bawah.
#
# RIWAYAT TUNING (evolusi, dari kecil ke besar sampel):
#   v1 flat-percent (bukan R)         → banyak masalah, digantikan R-based
#   v2 R-based (0.5/1.0/1.5/2.2/3.0R) → validasi 543 trade (M15 bar)
#   v3 ekor diperketat (2.0/2.8R)     → validasi silang 110+543 trade (M15)
#   v4 lock dinaikkan semua tahap     → validasi 356 trade (M1 presisi,
#                                        5 koin) — TERNYATA overfit, PnL
#                                        malah lebih rendah di sampel besar
#   v5 FINAL — kembali ke lock v3 + tambah 1 tahap ekor (3.5R) → divalidasi
#      di 1113 trade NYATA (15 koin, M1 presisi penuh menit-per-menit,
#      dataset PALING BESAR & PALING AKURAT sejauh ini): win rate PERSIS
#      SAMA (72.2%) dengan ladder v4, PnL lebih tinggi (144.35%→151.56%,
#      +7.2pp) — dan SL count SAMA SEKALI TIDAK BERUBAH (309), artinya
#      perbaikan ini murni menangkap lebih banyak upside dari trade yang
#      memang sudah menang, BUKAN mengambil risiko baru.
#
# Pelajaran dari v4→v5: tuning di sampel kecil (356 trade) bisa menyesatkan
# — begitu divalidasi ulang di sampel 3x lebih besar, hasilnya justru lebih
# baik pakai parameter yang lebih dekat ke versi SEBELUM v4. Sampel besar
# menang. Sudah dicoba juga menurunkan threshold R pertama (banyak trade
# SL cuma sempat MFE 0.28-0.36R sebelum reversal, di bawah 0.5R) — TERBUKTI
# menaikkan win rate signifikan (sampai 78-83%) TAPI PnL SELALU turun
# (127-146%) — tidak diambil krn bukan perbaikan bersih di kedua sisi,
# cuma trade-off WR-vs-PnL. Sudah dicoba juga grid search lebih luas di
# tahap ekor (3.2R/3.5R/4.5R/5.0R dgn macam2 lock) — hasil konvergen di
# kisaran 149-151.5%, F3b (di bawah) adalah titik terbaik yang ditemukan.
TRAIL_R_LADDER = [
    (0.5, 0.15),   # profit capai 0.5R → kunci 15% dari 0.5R
    (1.0, 0.35),   # 1.0R → kunci 35%
    (1.5, 0.50),   # 1.5R → kunci 50%
    (2.0, 0.65),   # 2.0R → kunci 65%
    (2.8, 0.80),   # 2.8R → kunci 80%
    (3.5, 0.85),   # 3.5R → kunci 85% (tahap tambahan v5 — tangkap sisa upside
                   #   trade yang sudah lari jauh, avg RR planned ~3.5-3.6R
                   #   jadi di titik ini biasanya sudah dekat TP)
]
# Trailing stop — KOMPONEN STRUKTUR (tetap dipakai, TIDAK berubah dari
# sebelumnya — divalidasi terpisah dan tetap jadi kandidat independen yg
# dibandingkan dgn ladder R di atas, SL final = paling protektif dari
# keduanya). Dibandingkan head-to-head di Datasheet.csv: fixed-pct
# SENDIRIAN WR 70.7% PnL 131.44%; structure SENDIRIAN WR 40-49% PnL
# 140-156% (lebih besar tapi jarang menang krn butuh lb*2+1 candle utk
# konfirmasi swing pertama); KOMBO WR 70.3% PnL 137.82% — lebih tahan-
# overfit krn separuh keputusan dari price action riil.
STRUCT_TRAIL_LB       = 2       # swing pivot lookback (kanan-kiri) di M15
STRUCT_TRAIL_BUF_PCT  = 0.0015  # buffer 0.15% di bawah/atas swing point
STRUCT_TRAIL_LOOKBACK = 60      # jumlah candle M15 ke belakang utk deteksi swing
# ── Fibonacci Extension TP (gated H4 confluence) ──
# Dipakai HANYA saat level struktural biasa sudah habis diperiksa DAN
# konteks H4 (trend besar) + RSI H4 (momentum belum jenuh) mendukung.
# Bukan cabang "penyelamat" RR gagal — ini kandidat TP tambahan yang
# dievaluasi berdampingan dengan level struktural lain di _select_best_tp.
FIB_EXT_1           = 0.272  # ekstensi 1.272 — butuh H4 trend + RSI band saja
FIB_EXT_2           = 0.618  # ekstensi 1.618 — butuh confluence penuh (+ CHoCH M15 searah)
H4_RSI_BUY_MIN      = 45     # RSI H4 BUY: momentum sudah established (bukan baru mulai)
H4_RSI_BUY_MAX      = 68     # tapi belum overbought / jenuh
H4_RSI_SELL_MIN     = 32     # RSI H4 SELL: kebalikan dari BUY
H4_RSI_SELL_MAX     = 55

def ema(s, n): return s.ewm(span=n, adjust=False).mean()

def rsi(s, n=14):
    d=s.diff()
    g=d.clip(lower=0).rolling(n).mean()
    l=(-d.clip(upper=0)).rolling(n).mean()
    return 100-100/(1+g/l.replace(0,np.nan))

def macd(s):
    line=ema(s,12)-ema(s,26); sig=ema(line,9)
    return line, sig, line-sig

def atr_fn(df, n=14):
    tr=pd.concat([
        df["high"]-df["low"],
        (df["high"]-df["close"].shift()).abs(),
        (df["low"]-df["close"].shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def build_df(df):
    if len(df)<60: return None
    df=df.copy()
    df["ema9"]   = ema(df["close"],9)
    df["ema21"]  = ema(df["close"],21)
    df["ema50"]  = ema(df["close"],50)
    df["ema200"] = ema(df["close"],200) if len(df)>=200 else ema(df["close"],50)
    df["rsi"]    = rsi(df["close"])
    df["ml"],df["ms"],df["mh"] = macd(df["close"])
    df["atr"]    = atr_fn(df)
    df["vol_sma"]= df["volume"].rolling(20).mean()
    bm=df["close"].rolling(20).mean(); bs=df["close"].rolling(20).std()
    df["bb_up"]=bm+2*bs; df["bb_lo"]=bm-2*bs; df["bb_mid"]=bm
    return df.dropna()


# ═════════════════════════════════════════════
# SMC / PRICE ACTION TOOLS
# ═════════════════════════════════════════════
def swing_pts(df, lb=5):
    sh,sl=[],[]
    for i in range(lb, len(df)-lb):
        if df["high"].iloc[i]==df["high"].iloc[i-lb:i+lb+1].max(): sh.append(i)
        if df["low"].iloc[i]==df["low"].iloc[i-lb:i+lb+1].min():   sl.append(i)
    return sh, sl

def mkt_struct(df, sh, sl):
    if len(sh)<2 or len(sl)<2: return "ranging"
    hh=df["high"].iloc[sh[-1]]>df["high"].iloc[sh[-2]]
    hl=df["low"].iloc[sl[-1]]>df["low"].iloc[sl[-2]]
    lh=df["high"].iloc[sh[-1]]<df["high"].iloc[sh[-2]]
    ll=df["low"].iloc[sl[-1]]<df["low"].iloc[sl[-2]]
    if hh and hl: return "bullish"
    if lh and ll: return "bearish"
    return "ranging"

def detect_bos(df, sh, sl):
    """
    BOS (Break of Structure) — konfirmasi kelanjutan trend.
    Sesuai materi: BOS valid CUKUP dengan shadow/wick candle menembus
    swing sebelumnya (tidak wajib body close, beda dengan CHoCH yang
    lebih ketat — lihat detect_choch()).
    """
    res={"bb":False,"bs":False,"cb":False,"cs":False}
    hi=df["high"].iloc[-1]; lo=df["low"].iloc[-1]
    if len(sh)>=2:
        ph=df["high"].iloc[sh[-2]]; lh=df["high"].iloc[sh[-1]]
        if hi>ph: res["bb" if lh>ph else "cb"]=True
    if len(sl)>=2:
        pl=df["low"].iloc[sl[-2]]; ll=df["low"].iloc[sl[-1]]
        if lo<pl: res["bs" if ll<pl else "cs"]=True
    return res

def find_snr_levels(df, lb=80):
    """
    Cari level Support & Resistance dari swing points.
    Level yang paling banyak disentuh = level terkuat.
    """
    sh, sl = swing_pts(df, lb=5)
    levels = []
    for i in sh:
        levels.append(("R", df["high"].iloc[i]))
    for i in sl:
        levels.append(("S", df["low"].iloc[i]))
    return levels

def find_zones(df, direction, lb=40, strict=False):
    """
    Deteksi ZONA TERPADU (Order Block = Supply/Demand Zone) — satu model
    sesuai materi: OB pada dasarnya adalah versi "dasar/minimal" dari
    Supply & Demand zone, bukan konsep terpisah. Fungsi ini menggantikan
    find_supply_demand() dan find_ob() versi lama yang terpisah.

    direction: "bull"/"demand" → base candle bearish diikuti rally (cari
               zona untuk BUY)
               "bear"/"supply" → base candle bullish diikuti drop (cari
               zona untuk SELL)
    strict   : True  → wajib 2 candle konfirmasi lanjutan searah setelah
                        impulse (perilaku find_ob lama, base candle lebih
                        "murni"/sempit — dipakai untuk entry precision)
               False → cukup 1 candle impulse (perilaku find_supply_demand
                        lama, zona bisa sedikit lebih lebar — dipakai
                        untuk SL/TP pool yang butuh lebih banyak kandidat)

    Setiap zona disertai VALIDASI 3-KRITERIA dari materi (dianggap valid
    kalau minimal salah satu terpenuhi, quality = jumlah yang terpenuhi):
    1. has_fvg   — ada Fair Value Gap yang menyertai impulse move
    2. has_bos   — impulse move menghasilkan break of structure
    3. is_fresh  — zona belum pernah disentuh ulang sejak terbentuk
    Plus:
    - pattern         : RBR/DBR (demand) atau DBD/RBD (supply)
    - strong_move_away: candle impulse body besar (bukan sekadar koreksi
      kecil — penanda smart money benar2 eksekusi order besar di sana)
    - fib_zone/fib_ratio/fib_aligned: posisi zona relatif ke range swing
      lb candle terakhir. Zona demand/bull idealnya di DISKON, zona
      supply/bear idealnya di PREMIUM (fib_aligned=True kalau selaras).
    """
    is_demand = direction in ("bull", "demand")
    sub = df.iloc[-lb:]
    base_offset = len(df) - len(sub)
    avg_body = (sub["close"] - sub["open"]).abs().mean()
    swing_hi = sub["high"].max()
    swing_lo = sub["low"].min()
    zones = []
    sh_all, sl_all = swing_pts(df, lb=5)

    end_range = len(sub) - 3 if strict else len(sub) - 2
    for i in range(1, end_range):
        c   = sub.iloc[i]
        nx  = sub.iloc[i + 1]
        nx2 = sub.iloc[i + 2] if i + 2 < len(sub) else None
        impulse_body = abs(nx["close"] - nx["open"])
        strong_move_away = impulse_body >= avg_body * 1.3
        min_impulse = avg_body * (1.5 if strict else 1.3)
        if impulse_body < min_impulse:
            continue

        if is_demand:
            is_match = c["close"] < c["open"] and nx["close"] > nx["open"]
            if strict and is_match:
                is_match = nx2 is not None and nx2["close"] > nx2["open"]
        else:
            is_match = c["close"] > c["open"] and nx["close"] < nx["open"]
            if strict and is_match:
                is_match = nx2 is not None and nx2["close"] < nx2["open"]
        if not is_match:
            continue

        top = max(c["open"], c["close"])
        bot = min(c["open"], c["close"])
        df_idx = base_offset + i

        # Kriteria 1: FVG menyertai impulse (celah antara c dan nx2)
        has_fvg = False
        if nx2 is not None:
            if is_demand and nx2["low"] > c["high"]:
                has_fvg = True
            if (not is_demand) and nx2["high"] < c["low"]:
                has_fvg = True

        # Kriteria 2: impulse ini menghasilkan BOS (harga break swing sebelumnya)
        has_bos = False
        try:
            if is_demand and len(sh_all) >= 1:
                prior_highs = [df["high"].iloc[k] for k in sh_all if k < df_idx]
                if prior_highs and nx["high"] > max(prior_highs[-1:] or [float("-inf")]):
                    has_bos = True
            if (not is_demand) and len(sl_all) >= 1:
                prior_lows = [df["low"].iloc[k] for k in sl_all if k < df_idx]
                if prior_lows and nx["low"] < min(prior_lows[-1:] or [float("inf")]):
                    has_bos = True
        except Exception:
            has_bos = False

        # Kriteria 3: fresh — belum pernah disentuh ulang sejak terbentuk
        fresh = is_zone_fresh(df, top, bot, df_idx)

        pattern = classify_sd_pattern(df, df_idx, "demand" if is_demand else "supply")

        fib = get_fib_zone((top + bot) / 2, swing_lo, swing_hi)
        fib_aligned = fib["zone"] in (("discount", "equilibrium") if is_demand
                                       else ("premium", "equilibrium"))

        zones.append({
            "top": top, "bot": bot,
            "mid": (top + bot) / 2,
            "high": c["high"], "low": c["low"],
            "idx": df_idx,
            "has_fvg": bool(has_fvg),
            "has_bos": bool(has_bos),
            "is_fresh": bool(fresh),
            "strong_move_away": bool(strong_move_away),
            "pattern": pattern,
            "fib_zone": fib["zone"],
            "fib_ratio": fib["ratio"],
            "fib_aligned": bool(fib_aligned),
            # quality: berapa dari 3 kriteria utama terpenuhi (fvg, bos, fresh)
            "quality": int(has_fvg) + int(has_bos) + int(fresh),
        })
    return zones[-3:] if zones else []


def find_supply_demand(df, direction, lb=40):
    """Kompatibilitas: alias tipis ke find_zones (mode non-strict/S&D)."""
    return find_zones(df, "demand" if direction == "demand" else "supply", lb=lb, strict=False)


def find_ob(df, direction, lb=40):
    """Kompatibilitas: alias tipis ke find_zones (mode strict/OB murni)."""
    return find_zones(df, direction, lb=lb, strict=True)



def find_fvg(df, direction, lb=40):
    """
    Fair Value Gap (FVG) — celah 3-candle yang menandakan pergerakan
    impulsif tak seimbang antara buyer/seller.

    Setiap FVG kini disertai:
    - is_fresh   : belum pernah disentuh ulang (bahkan oleh shadow) sejak terbentuk
    - candle3    : klasifikasi "breakaway" (ideal, searah & impulsif) vs
                   "rejection" (hindari, candle ke-3 melawan arah gap)
    - fib_zone   : apakah gap ini berada di area diskon/premium relatif
                   terhadap range swing lb candle terakhir (dipakai utk
                   preferensi entry FVG di area diskon utk BUY / premium
                   utk SELL, sesuai materi)
    """
    sub = df.iloc[-lb:]
    base_offset = len(df) - len(sub)
    out = []
    swing_hi = sub["high"].max()
    swing_lo = sub["low"].min()

    for i in range(len(sub) - 2):
        c0, c1, c2 = sub.iloc[i], sub.iloc[i + 1], sub.iloc[i + 2]
        df_idx_c0 = base_offset + i
        df_idx_c2 = base_offset + i + 2

        gap = None
        if direction == "bull" and c2["low"] > c0["high"]:
            gap = {"top": c2["low"], "bot": c0["high"]}
        if direction == "bear" and c2["high"] < c0["low"]:
            gap = {"top": c0["low"], "bot": c2["high"]}
        if gap is None:
            continue

        gap["mid"] = (gap["top"] + gap["bot"]) / 2
        gap["idx"] = df_idx_c2
        gap["is_fresh"] = is_zone_fresh(df, gap["top"], gap["bot"], df_idx_c0, end_idx=len(df)-1)
        gap["candle3"] = classify_fvg_candle3(df, df_idx_c2, direction)
        gap["fib_zone"] = get_fib_zone(gap["mid"], swing_lo, swing_hi)["zone"]
        out.append(gap)

    return out[-3:] if out else []

def find_equal_highs_lows(df, kind="high", lb=60, tol=0.0025):
    """
    Equal Highs/Lows = zona likuiditas (banyak stop loss retail di sana).
    Institusi sering sweeping level ini sebelum berbalik.
    """
    sub=df.iloc[-lb:]
    vals=sub["high"] if kind=="high" else sub["low"]
    clusters=[]
    visited=set()
    for i in range(len(vals)):
        if i in visited: continue
        group=[vals.iloc[i]]
        for j in range(i+1, len(vals)):
            if abs(vals.iloc[i]-vals.iloc[j])/max(vals.iloc[i],0.0001)<tol:
                group.append(vals.iloc[j])
                visited.add(j)
        if len(group)>=2:
            clusters.append(sum(group)/len(group))
    return sorted(clusters)

def nearest_snr(df, price, direction, margin=0.015):
    """
    Cari level S/R terdekat yang relevan untuk TP/SL.
    direction='above' → cari resistance di atas harga
    direction='below' → cari support di bawah harga
    """
    sh, sl = swing_pts(df, lb=4)
    if direction=="above":
        candidates = [df["high"].iloc[i] for i in sh
                      if df["high"].iloc[i] > price*(1+margin*0.3)]
        candidates += find_equal_highs_lows(df,"high")
        candidates = [c for c in candidates if c > price*(1+margin*0.3)]
        return min(candidates) if candidates else None
    else:
        candidates = [df["low"].iloc[i] for i in sl
                      if df["low"].iloc[i] < price*(1-margin*0.3)]
        candidates += find_equal_highs_lows(df,"low")
        candidates = [c for c in candidates if c < price*(1-margin*0.3)]
        return max(candidates) if candidates else None


def detect_choch(df, sh, sl):
    """
    CHoCH (Change of Character) — konfirmasi perubahan arah NYATA.
    Bearish CHoCH: harga break di bawah HL terakhir setelah LH terbentuk.
    Bullish CHoCH: harga break di atas LH terakhir setelah HL terbentuk.
    Lebih ketat dari BOS biasa — perlu dua swing point terkonfirmasi
    DAN wajib BODY CLOSE candle menembus level (bukan sekadar shadow/wick),
    karena CHoCH menandakan pembalikan karakter pasar yang butuh bukti
    lebih kuat dibanding BOS yang hanya kelanjutan trend. Fungsi ini
    sudah pakai df["close"] (bukan high/low) sehingga syarat body-close
    otomatis terpenuhi.
    """
    result = {"bearish_choch": False, "bullish_choch": False}
    close = df["close"].iloc[-1]

    # Bearish CHoCH: ada LH (lower high) DAN harga sekarang break bawah swing low sebelumnya
    if len(sh) >= 2 and len(sl) >= 2:
        prev_high = df["high"].iloc[sh[-2]]
        last_high = df["high"].iloc[sh[-1]]
        prev_low  = df["low"].iloc[sl[-2]]
        last_low  = df["low"].iloc[sl[-1]]

        lh_formed = last_high < prev_high          # LH terbentuk
        if lh_formed and close < prev_low:         # break bawah HL
            result["bearish_choch"] = True

        hh_formed = last_high > prev_high          # HH terbentuk
        if hh_formed and close > prev_low and last_low > prev_low:  # break atas + HL
            result["bullish_choch"] = True

    return result


def detect_failed_retest(df, sh, sl, atr):
    """
    Failed Retest — harga naik ke resistance/level struktural lalu ditolak keras.
    Ini trigger entry SELL yang paling valid di SMC.
    Syarat:
    - Ada resistance level yang jelas (swing high sebelumnya)
    - Harga candle sebelumnya menyentuh atau mendekati resistance (dalam 0.5 ATR)
    - Candle sekarang close jauh di bawah resistance (rejection)
    - Candle sekarang bearish (close < open)
    """
    result = {"failed_retest_sell": False, "failed_retest_buy": False,
              "resistance": None, "support": None}
    if len(df) < 3: return result

    L   = df.iloc[-1]   # candle sekarang
    P   = df.iloc[-2]   # candle sebelumnya

    # Failed retest SELL: candle sebelumnya menyentuh resistance, sekarang rejected
    if len(sh) >= 2:
        resistance = df["high"].iloc[sh[-2]]   # swing high terakhir = resistance
        touched    = P["high"] >= resistance - atr * 0.5   # candle sebelum menyentuh
        rejected   = L["close"] < resistance - atr * 0.3  # sekarang jauh di bawah
        bearish_c  = L["close"] < L["open"]               # candle bearish
        if touched and rejected and bearish_c:
            result["failed_retest_sell"] = True
            result["resistance"] = resistance

    # Failed retest BUY: candle sebelumnya menyentuh support, sekarang bounced
    if len(sl) >= 2:
        support  = df["low"].iloc[sl[-2]]      # swing low terakhir = support
        touched  = P["low"] <= support + atr * 0.5
        bounced  = L["close"] > support + atr * 0.3
        bullish_c = L["close"] > L["open"]
        if touched and bounced and bullish_c:
            result["failed_retest_buy"] = True
            result["support"] = support

    return result


# ═════════════════════════════════════════════
# SMC LANJUTAN — Ilmu dari materi edukasi:
# fresh/mitigated zone, fib diskon/premium, breakaway
# vs rejection FVG, validitas pullback, price action
# confirmation (pin bar/fakey), pola RBR/DBR/DBD/RBD,
# inducement & liquidity sweep/run.
# ═════════════════════════════════════════════

def is_zone_fresh(df, top, bot, formed_idx, end_idx=None):
    """
    Cek apakah sebuah zona (OB/S&D/FVG) masih FRESH — belum pernah
    disentuh oleh harga sejak zona itu terbentuk.

    "Disentuh" didefinisikan longgar (bahkan wick/shadow saja dianggap
    sudah memitigasi zona — sesuai penjelasan di materi FVG: "meskipun
    hanya tersentuh sedikit dengan shadow, kita tetap menganggapnya
    sudah tersentuh").

    formed_idx: index candle tempat zona ini terbentuk (posisi dalam df).
    end_idx   : index terakhir yang mau diperiksa (default: candle
                terakhir df). start diambil 2 candle setelah formed_idx
                supaya candle pembentuk zona itu sendiri tidak dihitung.

    Return: True jika fresh (belum tersentuh), False jika sudah termitigasi.
    """
    if formed_idx is None or top is None or bot is None:
        return True
    n = len(df)
    end_idx = end_idx if end_idx is not None else n - 1
    start = formed_idx + 2
    if start >= end_idx:
        return True
    sub = df.iloc[start:end_idx]
    if sub.empty:
        return True
    touched = ((sub["low"] <= top) & (sub["high"] >= bot)).any()
    return not bool(touched)


def get_fib_zone(price, swing_low, swing_high):
    """
    Tentukan posisi harga dalam rentang swing (retracement ratio) serta
    apakah harga berada di area DISKON, PREMIUM, atau EQUILIBRIUM.

    ratio dihitung sebagai posisi price relatif terhadap [swing_low, swing_high]:
      ratio kecil (<=0.45) → dekat swing_low  → "discount"
      ratio besar (>=0.55) → dekat swing_high → "premium"
      di antaranya         → "equilibrium"

    Return dict: {"ratio": float, "zone": str}
    """
    rng = swing_high - swing_low
    if rng <= 0:
        return {"ratio": 0.5, "zone": "equilibrium"}
    ratio = (price - swing_low) / rng
    if ratio <= 0.45:
        zone = "discount"
    elif ratio >= 0.55:
        zone = "premium"
    else:
        zone = "equilibrium"
    return {"ratio": round(ratio, 4), "zone": zone}


def adaptive_fib_target(df, sh, sl, direction):
    """
    Tentukan target retracement Fibonacci secara ADAPTIF berdasarkan
    kekuatan trend & kedalaman pullback (bukan angka fix 50%):

    - Trend SANGAT kuat (impuls dominan, nyaris tanpa pullback)
      → fokus area retracement 0.236 - 0.382 (paling dangkal)
    - Trend kuat (impuls dominan, pullback dangkal & lemah)
      → fokus area retracement 0.382 - 0.5 (dangkal)
    - Trend lemah (pullback agresif & dalam)
      → fokus area retracement 0.618 - 0.786 (dalam, termasuk OTE)

    Kekuatan trend diestimasi dari rasio panjang leg pullback vs leg
    impuls terakhir (di TF yang sama, m15/h1 tergantung caller).

    CATATAN (fix presisi-entry): tier "SANGAT kuat" ditambahkan setelah
    analisa data menemukan >600 sinyal (dari 2 backtest independen)
    gagal terisi krn TP sudah kena duluan sebelum harga sempat pullback
    ke zona entry — median jarak ke entry 2.2-2.5× lebih jauh drpd jarak
    ke TP itu sendiri. Root cause: trend yg SANGAT kuat (pullback_ratio
    mendekati 0, artinya harga nyaris tidak pullback sama sekali) tetap
    diminta retrace ke 0.382-0.5 — utk momentum se-ekstrem itu, itu
    sendiri sudah terlalu dalam & sering tidak pernah kejadian sebelum
    harga lanjut ke TP. Tier baru ini TIDAK melonggarkan syarat kualitas
    apa pun (freshness/FVG/BOS tetap sama) — cuma target retracement yg
    lebih realistis utk kondisi momentum paling ekstrem.

    Return: (fib_lo, fib_hi) sebagai rasio retracement (0..1).
    """
    default = (0.5, 0.618)   # fallback netral kalau data belum cukup
    if len(sh) < 2 or len(sl) < 2:
        return default
    try:
        if direction == "bull":
            impulse_len   = df["high"].iloc[sh[-1]] - df["low"].iloc[sl[-2]]
            pullback_len  = df["high"].iloc[sh[-1]] - df["close"].iloc[-1]
        else:
            impulse_len   = df["high"].iloc[sh[-2]] - df["low"].iloc[sl[-1]]
            pullback_len  = df["close"].iloc[-1] - df["low"].iloc[sl[-1]]
        if impulse_len <= 0:
            return default
        pullback_ratio = abs(pullback_len) / impulse_len
    except Exception:
        return default

    if pullback_ratio <= 0.12:
        return (0.236, 0.382)   # trend SANGAT kuat, pullback minimal
    elif pullback_ratio <= 0.30:
        return (0.382, 0.5)     # trend kuat, pullback dangkal
    elif pullback_ratio >= 0.55:
        return (0.618, 0.786)   # trend lemah, pullback dalam (OTE)
    else:
        return (0.5, 0.618)


def classify_fvg_candle3(df, fvg_idx_c2, direction):
    """
    Klasifikasi FVG berdasarkan candle ke-3 (candle "c2" pembentuk gap):
    - Breakaway Gap : candle ke-3 SEARAH gap (impulsif, melanjutkan) → IDEAL untuk entry
    - Rejection Gap : candle ke-3 BERLAWANAN arah gap → HINDARI, sinyal lemah

    direction: "bull" (bullish FVG) atau "bear" (bearish FVG)
    Return: "breakaway" atau "rejection"
    """
    if fvg_idx_c2 is None or fvg_idx_c2 >= len(df):
        return "unknown"
    c2 = df.iloc[fvg_idx_c2]
    is_bull_candle = c2["close"] > c2["open"]
    if direction == "bull":
        return "breakaway" if is_bull_candle else "rejection"
    else:
        return "rejection" if is_bull_candle else "breakaway"


def is_valid_pullback(df, direction, lookback=8):
    """
    Validasi pullback sesuai definisi price action yang ketat:
    pullback valid HANYA jika candle koreksi benar-benar men-BREAK
    high/low dari candle sebelumnya (bukan sekadar candle berganti warna).

    Bullish trend: pullback valid jika ada candle bearish yang close-nya
    menembus LOW dari candle bullish terakhir sebelum koreksi dimulai.
    Bearish trend: sebaliknya, candle bullish menembus HIGH candle
    bearish terakhir.

    Return: bool
    """
    if len(df) < lookback + 2:
        return False
    sub = df.iloc[-lookback:]

    if direction == "bull":
        last_bull_low = None
        found_i = None
        for i in range(len(sub) - 1, -1, -1):
            c = sub.iloc[i]
            if c["close"] > c["open"]:
                last_bull_low = c["low"]
                found_i = i
                break
        if last_bull_low is None:
            return False
        after = sub.iloc[found_i+1:]
        return bool((after["close"] < last_bull_low).any())
    else:
        last_bear_high = None
        found_i = None
        for i in range(len(sub) - 1, -1, -1):
            c = sub.iloc[i]
            if c["close"] < c["open"]:
                last_bear_high = c["high"]
                found_i = i
                break
        if last_bear_high is None:
            return False
        after = sub.iloc[found_i+1:]
        return bool((after["close"] > last_bear_high).any())


def classify_pullback_type(df, direction, atr, lookback=6):
    """
    Klasifikasi tipe pullback: aggressive / corrective / sweeping.

    - Aggressive : koreksi cepat & besar (candle body rerata > 1.2x ATR),
      momentum kuat melawan trend → probabilitas reaksi di zona RENDAH,
      sebaiknya tidak entry langsung.
    - Sweeping   : ada equal high/low (double top/bottom) tepat sebelum
      area, menandakan liquidity pool yang disapu dulu → probabilitas
      TINGGI setelah sweep + shift struktur.
    - Corrective : koreksi bertahap, beberapa struktur kecil → probabilitas
      entry paling ideal, terutama dengan konfirmasi CHoCH TF rendah.

    Return: "aggressive" | "corrective" | "sweeping"
    """
    if len(df) < lookback + 1:
        return "corrective"
    sub = df.iloc[-lookback:]
    bodies = (sub["close"] - sub["open"]).abs()
    avg_body = bodies.mean()

    highs = sub["high"].values
    lows  = sub["low"].values
    tol = atr * 0.15
    has_equal_high = False
    has_equal_low  = False
    for i in range(len(highs)):
        for j in range(i+1, len(highs)):
            if abs(highs[i] - highs[j]) < tol:
                has_equal_high = True
            if abs(lows[i] - lows[j]) < tol:
                has_equal_low = True

    if direction == "bull" and has_equal_low:
        return "sweeping"
    if direction == "bear" and has_equal_high:
        return "sweeping"

    if avg_body > atr * 1.2:
        return "aggressive"

    return "corrective"


def detect_pinbar(candle, min_wick_ratio=1.5):
    """
    Deteksi pola Pin Bar (Deception Candle): body kecil di salah satu
    ujung, shadow panjang di sisi berlawanan — menandakan rejection kuat.

    Return: {"is_pinbar": bool, "bullish_pinbar": bool, "bearish_pinbar": bool}
    """
    o, h, l, c = candle["open"], candle["high"], candle["low"], candle["close"]
    body = abs(c - o)
    rng  = h - l
    if rng <= 0:
        return {"is_pinbar": False, "bullish_pinbar": False, "bearish_pinbar": False}
    low_wick = min(o, c) - l
    up_wick  = h - max(o, c)

    bullish_pinbar = low_wick > body * min_wick_ratio and low_wick > up_wick * 1.5
    bearish_pinbar = up_wick > body * min_wick_ratio and up_wick > low_wick * 1.5
    return {
        "is_pinbar": bool(bullish_pinbar or bearish_pinbar),
        "bullish_pinbar": bool(bullish_pinbar),
        "bearish_pinbar": bool(bearish_pinbar),
    }


def detect_fakey(df):
    """
    Deteksi pola Fakey (false breakout dari inside bar):
    1. Ada inside bar (candle tertutup penuh dalam range candle sebelumnya)
    2. Harga breakout ke salah satu sisi (menembus high/low mother bar)
    3. Harga berbalik dan close kembali DI DALAM range mother bar

    Return: {"is_fakey": bool, "bullish_fakey": bool, "bearish_fakey": bool}
    bullish_fakey = false breakout ke bawah lalu balik naik (sinyal BUY)
    bearish_fakey = false breakout ke atas lalu balik turun (sinyal SELL)
    """
    result = {"is_fakey": False, "bullish_fakey": False, "bearish_fakey": False}
    if len(df) < 3:
        return result

    mother = df.iloc[-3]
    inside = df.iloc[-2]
    last   = df.iloc[-1]

    is_inside = inside["high"] <= mother["high"] and inside["low"] >= mother["low"]
    if not is_inside:
        return result

    broke_up   = last["high"] > mother["high"]
    broke_down = last["low"]  < mother["low"]
    closed_inside = mother["low"] <= last["close"] <= mother["high"]

    if broke_down and closed_inside and last["close"] > last["open"]:
        result["is_fakey"] = True
        result["bullish_fakey"] = True
    elif broke_up and closed_inside and last["close"] < last["open"]:
        result["is_fakey"] = True
        result["bearish_fakey"] = True

    return result


def classify_sd_pattern(df, zone_idx, direction, lb=6):
    """
    Klasifikasi pola pembentukan supply/demand berdasarkan rally/drop/base:
    - Demand: RBR (Rally-Base-Rally) atau DBR (Drop-Base-Rally)
    - Supply: DBD (Drop-Base-Drop) atau RBD (Rally-Base-Drop)

    zone_idx: index candle "base" (candle dasar pembentuk OB) dalam df.
    Return label string atau "unknown" kalau tidak cukup data.
    """
    if zone_idx is None or zone_idx < lb or zone_idx + lb >= len(df):
        return "unknown"

    before = df.iloc[max(0, zone_idx - lb):zone_idx]
    after  = df.iloc[zone_idx + 1: zone_idx + 1 + lb]
    if before.empty or after.empty:
        return "unknown"

    move_before = before["close"].iloc[-1] - before["close"].iloc[0]
    move_after  = after["close"].iloc[-1] - after["close"].iloc[0]

    before_up = move_before > 0
    after_up  = move_after > 0

    if direction == "demand":
        if before_up and after_up:       return "RBR"
        if (not before_up) and after_up: return "DBR"
        return "unknown"
    else:
        if (not before_up) and (not after_up): return "DBD"
        if before_up and (not after_up):        return "RBD"
        return "unknown"


def detect_liquidity_run_or_sweep(df, sh, sl, direction):
    """
    Bedakan Liquidity RUN (breakout bersih, close di luar swing) vs
    Liquidity SWEEP/SWIFT (wick menembus tapi GAGAL close di luar swing —
    liquidity grab, arah sebenarnya kemungkinan BERLAWANAN).

    direction: "bull" → cek terhadap swing high terdekat
               "bear" → cek terhadap swing low terdekat

    Return: {"type": "run"/"sweep"/"none", "level": float atau None}
    """
    result = {"type": "none", "level": None}
    if direction == "bull" and len(sh) >= 1:
        level = df["high"].iloc[sh[-1]]
        last  = df.iloc[-1]
        if last["high"] > level and last["close"] > level:
            result = {"type": "run", "level": level}
        elif last["high"] > level and last["close"] <= level:
            result = {"type": "sweep", "level": level}
    elif direction == "bear" and len(sl) >= 1:
        level = df["low"].iloc[sl[-1]]
        last  = df.iloc[-1]
        if last["low"] < level and last["close"] < level:
            result = {"type": "run", "level": level}
        elif last["low"] < level and last["close"] >= level:
            result = {"type": "sweep", "level": level}
    return result


def detect_inducement_move(df, direction, atr, lookback=5):
    """
    Deteksi kemungkinan inducement — gerakan kecil BERLAWANAN arah trend
    yang muncul TEPAT SEBELUM harga menyentuh level penting (OB/FVG/EQH/EQL).
    Ciri: gerakan kecil (< 0.6 ATR), searah pullback minor, terjadi di
    2-3 candle terakhir sebelum candle sekarang.

    Ini dipakai sebagai FLAG (bukan hard block) — kalau inducement barusan
    terjadi, kita minta konfirmasi CHoCH tambahan sebelum entry, alih-alih
    entry di breakout/gerakan pertama begitu saja.

    Return: bool (True = terindikasi inducement baru saja terjadi)
    """
    if len(df) < lookback + 1:
        return False
    sub = df.iloc[-lookback:-1]   # tidak termasuk candle sekarang
    if sub.empty:
        return False
    small_moves = ((sub["close"] - sub["open"]).abs() < atr * 0.6)
    if direction == "bull":
        counter = sub["close"] < sub["open"]
    else:
        counter = sub["close"] > sub["open"]
    return bool((small_moves & counter).tail(3).any())


# ═════════════════════════════════════════════
# TAHAP 1: SCORING NORMAL — cari sinyal terkuat
# ═════════════════════════════════════════════
def score_direction(df_h1, df_m15, df_d1=None):
    """
    Analisis HIERARKIS (bukan lagi additive scoring flat dari banyak
    indikator lepas): sesuai filosofi price-action/SMC di materi —
    tentukan BIAS dari struktur besar dulu, baru cari KONFIRMASI ENTRY
    dari price-action M15. Bias dan konfirmasi yang berlawanan saling
    melemahkan drastis (gate), bukan sekadar dijumlah rata.

    LAYER 1 — BIAS (konteks arah, dari struktur & momentum besar):
      • Market Structure H1 (HH/HL vs LH/LL) — bobot terbesar
      • D1 bias (EMA + struktur harian) — konfirmasi/veto di layer akhir
      • EMA H1 trend alignment
      • RSI M15 (dipertahankan sesuai preferensi — momentum filter,
        bukan sinyal SMC, tapi tetap berguna sebagai extra confluence)

    LAYER 2 — SETUP/KONFIRMASI (price-action & SMC murni dari 10 materi):
      • BOS (shadow) & CHoCH (wajib body-close) M15+H1
      • Failed Retest M15+H1
      • Validitas & tipe pullback (corrective/sweeping/aggressive)
      • Pin bar rejection, pola Fakey
      • Liquidity Run vs Sweep/Swift
      • OTE 0.62-0.79 + FVG/CHoCH pendukung
      • MACD/BB/Volume M15 — momentum confluence tambahan (ringan)

    LAYER 3 — GATE: kalau LAYER 2 (konfirmasi entry) berlawanan arah
    dengan LAYER 1 (bias struktural), konfirmasi itu dilemahkan drastis
    alih-alih dijumlah rata — mencegah sinyal yang sebenarnya melawan
    struktur besar tapi lolos hanya karena numpuk banyak micro-signal M15.

    df_d1 (BARU): klines D1 ASLI (bukan hasil resample df_h1). df_h1 di
    sini cuma window pendek (get_klines limit=250 ≈ 10 hari), resample
    ke "1D" dari situ cuma menghasilkan ~10 bar harian — DI BAWAH syarat
    minimum build_df (60 bar), jadi d1_bias SELALU "neutral" dan fitur
    ini tidak pernah benar-benar berkontribusi (bug lama). Kalau df_d1
    disediakan (fetch terpisah, histori panjang), d1_bias dihitung dari
    situ — kalau tidak, fallback ke cara lama (tetap sering neutral).

    Return: dict dengan symbol, direction asli, confidence, price
    """
    h1=build_df(df_h1); m15=build_df(df_m15)
    if h1 is None or m15 is None: return None

    L1=h1.iloc[-1]; P1=h1.iloc[-2]
    L15=m15.iloc[-1]; P15=m15.iloc[-2]
    rv=L15["rsi"]
    atr_val=max(L15["atr"], L15["close"]*0.003)

    sh1,sl1   = swing_pts(h1,5)
    sh15,sl15 = swing_pts(m15,5)
    struct_h1 = mkt_struct(h1,sh1,sl1)
    choch_h1  = detect_choch(h1, sh1, sl1)   # dihitung di sini krn ikut Layer 1

    # ── D1 bias — dihitung di awal supaya ikut Layer 1 scoring (bukan
    # cuma hard-block di akhir seperti sebelumnya). Syarat dilonggarkan
    # jadi OR (EMA alignment ATAU struct D1, tidak wajib bersamaan) —
    # sebelumnya syarat AND terlalu ketat sehingga d1_bias hampir selalu
    # "neutral" dan tidak pernah benar-benar berkontribusi ke arah,
    # sehingga H1 jadi satu-satunya penentu bias tanpa konteks D1 sama
    # sekali (H1 bisa saja cuma pullback minor dari downtrend D1 besar).
    d1_bias = "neutral"
    try:
        if df_d1 is not None and len(df_d1) >= 65:
            df_d1_built = build_df(df_d1)
        else:
            # fallback lama (hampir selalu gagal krn window terlalu pendek)
            df_d1_built = build_df(df_h1.resample("1D").agg({
                "open":"first","high":"max","low":"min",
                "close":"last","volume":"sum"
            }).dropna())
        if df_d1_built is not None and len(df_d1_built) >= 10:
            LD = df_d1_built.iloc[-1]
            sh_d, sl_d = swing_pts(df_d1_built, lb=3)
            struct_d1  = mkt_struct(df_d1_built, sh_d, sl_d)
            ema_bear_d1 = LD["ema9"] < LD["ema21"] < LD["ema50"]
            ema_bull_d1 = LD["ema9"] > LD["ema21"] > LD["ema50"]
            if struct_d1 == "bearish" or ema_bear_d1:
                d1_bias = "bearish"
            elif struct_d1 == "bullish" or ema_bull_d1:
                d1_bias = "bullish"
            # kalau struct_d1 dan ema saling kontradiksi (mis. struct
            # bullish tapi ema bearish), biarkan struct_d1 menang karena
            # itu representasi price action nyata, EMA cuma turunan lag.
    except Exception:
        pass

    # ══════════════════════════════════════════════════════════════
    # LAYER 1 — BIAS: arah besar dari struktur, bukan micro-indicator.
    # CHoCH H1 dimasukkan di layer ini (bukan Layer 2/setup) karena
    # secara konsep CHoCH = perubahan KARAKTER/BIAS pasar itu sendiri
    # (lihat materi video #6), bukan sekadar trigger entry M15 seperti
    # pin bar/fakey. struct_h1 dari swing HH/HL cenderung lagging —
    # CHoCH H1 sering jadi sinyal PALING AWAL bahwa bias lama sudah
    # tidak berlaku, jadi wajib ikut menentukan bias_dir, bukan cuma
    # jadi setup yang dipotong buta kalau "melawan" struct_h1 yang
    # sebenarnya sudah usang.
    # ══════════════════════════════════════════════════════════════
    bias_bull = bias_bear = 0

    if struct_h1=="bullish": bias_bull += 30
    if struct_h1=="bearish": bias_bear += 30

    if choch_h1["bullish_choch"]: bias_bull += 26
    if choch_h1["bearish_choch"]: bias_bear += 26

    if L1["ema9"]>L1["ema21"]>L1["ema50"]:  bias_bull += 15
    elif L1["ema9"]>L1["ema21"]:             bias_bull += 7
    if L1["ema9"]<L1["ema21"]<L1["ema50"]:  bias_bear += 15
    elif L1["ema9"]<L1["ema21"]:             bias_bear += 7
    if L1["close"]>L1["ema200"]: bias_bull += 8
    else:                          bias_bear += 8

    # D1 bias — bobot besar sengaja, karena ini representasi konteks
    # MAKRO/harian yang H1 sendiri tidak bisa lihat (H1 rentan salah baca
    # pullback lokal sebagai reversal, padahal trend besarnya masih sama).
    if d1_bias == "bullish": bias_bull += 24
    if d1_bias == "bearish": bias_bear += 24

    # RSI M15 — dipertahankan sebagai momentum filter (bukan SMC, tapi
    # tetap relevan): oversold/overbought memberi confluence ke bias.
    if rv<35:    bias_bull += 12
    elif rv<45:  bias_bull += 6
    if rv>65:    bias_bear += 12
    elif rv>55:  bias_bear += 6

    bias_dir = "bull" if bias_bull >= bias_bear else "bear"

    # ══════════════════════════════════════════════════════════════
    # LAYER 2 — SETUP: konfirmasi entry price-action/SMC (10 materi)
    # ══════════════════════════════════════════════════════════════
    setup_bull = setup_bear = 0

    # BOS (shadow cukup) & CHoCH (wajib body close) — inti SMC
    bos = detect_bos(m15, sh15, sl15)
    if bos["bb"]: setup_bull += 12
    if bos["cb"]: setup_bull += 7
    if bos["bs"]: setup_bear += 12
    if bos["cs"]: setup_bear += 7

    choch = detect_choch(m15, sh15, sl15)
    if choch["bullish_choch"]: setup_bull += 22
    if choch["bearish_choch"]: setup_bear += 22

    # (CHoCH H1 sudah dihitung di Layer 1/bias di atas — lihat komentar
    # di bagian bias_bull/bias_bear)

    # Failed Retest — trigger entry paling valid di SMC
    fr = detect_failed_retest(m15, sh15, sl15, atr_val)
    if fr["failed_retest_sell"]: setup_bear += 24
    if fr["failed_retest_buy"]:  setup_bull += 24

    fr_h1 = detect_failed_retest(h1, sh1, sl1, atr_val)
    if fr_h1["failed_retest_sell"]: setup_bear += 18
    if fr_h1["failed_retest_buy"]:  setup_bull += 18

    # Validitas & tipe pullback — corrective=ideal, sweeping=ideal
    # setelah sweep, aggressive=risiko tinggi (bobot dikurangi)
    pullback_valid_bull = is_valid_pullback(m15, "bull")
    pullback_valid_bear = is_valid_pullback(m15, "bear")
    pullback_type_bull  = classify_pullback_type(m15, "bull", atr_val)
    pullback_type_bear  = classify_pullback_type(m15, "bear", atr_val)

    if pullback_valid_bull:
        if pullback_type_bull == "aggressive": setup_bull += 3
        elif pullback_type_bull == "sweeping":  setup_bull += 14
        else:                                    setup_bull += 9
    if pullback_valid_bear:
        if pullback_type_bear == "aggressive": setup_bear += 3
        elif pullback_type_bear == "sweeping":  setup_bear += 14
        else:                                    setup_bear += 9

    # Pin bar rejection & pola Fakey — konfirmasi price action di zona
    pinbar = detect_pinbar(L15)
    if pinbar["bullish_pinbar"]: setup_bull += 10
    if pinbar["bearish_pinbar"]: setup_bear += 10

    fakey = detect_fakey(m15)
    if fakey["bullish_fakey"]: setup_bull += 10
    if fakey["bearish_fakey"]: setup_bear += 10

    # Liquidity Run vs Sweep/Swift
    liq_bull = detect_liquidity_run_or_sweep(m15, sh15, sl15, "bull")
    liq_bear = detect_liquidity_run_or_sweep(m15, sh15, sl15, "bear")
    if liq_bull["type"] == "run":    setup_bull += 10
    elif liq_bull["type"] == "sweep": setup_bear += 8
    if liq_bear["type"] == "run":    setup_bear += 10
    elif liq_bear["type"] == "sweep": setup_bull += 8

    # Inducement — flag saja, dipakai full_analyze/calc_discount_entry
    # untuk menunda entry, bukan mengubah skor di sini.
    inducement_bull = detect_inducement_move(m15, "bull", atr_val)
    inducement_bear = detect_inducement_move(m15, "bear", atr_val)

    # OTE (0.62-0.79) — TIDAK boleh berdiri sendiri, wajib CHoCH atau
    # FVG fresh searah sebagai pendamping.
    ote_bull = ote_bear = False
    if len(sh15) >= 1 and len(sl15) >= 1:
        swing_hi_m15 = m15["high"].iloc[sh15[-1]]
        swing_lo_m15 = m15["low"].iloc[sl15[-1]]
        fib_now = get_fib_zone(L15["close"], swing_lo_m15, swing_hi_m15)
        if 0.62 <= (1 - fib_now["ratio"]) <= 0.79: ote_bull = True
        if 0.62 <= fib_now["ratio"] <= 0.79:        ote_bear = True

    if ote_bull and (choch["bullish_choch"] or any(f.get("is_fresh") for f in find_fvg(m15, "bull", lb=30))):
        setup_bull += 10
    if ote_bear and (choch["bearish_choch"] or any(f.get("is_fresh") for f in find_fvg(m15, "bear", lb=30))):
        setup_bear += 10

    # Candle pattern dasar (hammer/shooting star) — pelengkap price action
    body=L15["close"]-L15["open"]
    low_wick=min(L15["open"],L15["close"])-L15["low"]
    up_wick=L15["high"]-max(L15["open"],L15["close"])
    if low_wick>abs(body)*1.5: setup_bull += 6
    if up_wick>abs(body)*1.5:  setup_bear += 6

    # Momentum confluence ringan (MACD/BB/Volume M15) — bukan SMC, tapi
    # masih dipakai selama relevan sesuai preferensi (bobot lebih kecil
    # dari layer SMC di atas, hanya sebagai pelengkap).
    if L15["mh"]>0 and P15["mh"]<=0:  setup_bull += 8
    elif L15["mh"]>0:                  setup_bull += 3
    if L15["mh"]<0 and P15["mh"]>=0:  setup_bear += 8
    elif L15["mh"]<0:                  setup_bear += 3

    if L15["close"]<=L15["bb_lo"]:    setup_bull += 7
    elif L15["close"]<L15["bb_mid"]:  setup_bull += 3
    if L15["close"]>=L15["bb_up"]:    setup_bear += 7
    elif L15["close"]>L15["bb_mid"]:  setup_bear += 3

    if L15["volume"]>L15["vol_sma"]*1.5:
        if L15["close"]>L15["open"]:  setup_bull += 6
        else:                          setup_bear += 6
    elif L15["volume"]>L15["vol_sma"]:
        if L15["close"]>L15["open"]:  setup_bull += 2
        else:                          setup_bear += 2

    # ══════════════════════════════════════════════════════════════
    # LAYER 3 — GATE: konfirmasi yang BERLAWANAN dengan bias struktural
    # dilemahkan drastis (dipotong separuh), bukan dijumlah rata.
    # Ini yang membedakan dari additive scoring lama — struktur besar
    # (bias) diperlakukan sebagai FILTER wajib, bukan sekadar satu
    # sumber poin di antara puluhan sumber poin lain yang setara.
    # ══════════════════════════════════════════════════════════════
    if bias_dir == "bull":
        setup_bear = setup_bear * 0.5
    else:
        setup_bull = setup_bull * 0.5

    bull = bias_bull + setup_bull
    bear = bias_bear + setup_bear

    direction="bull" if bull>=bear else "bear"
    raw=bull if direction=="bull" else bear
    conf=min(int(raw/264*100),99)

    # D1 berlawanan TOTAL dengan sinyal akhir → tetap hard block (bukan
    # cuma penalty scoring) — kalau sampai lolos scoring pun (krn Layer 2
    # setup sangat kuat) tapi D1 benar2 berlawanan, lebih aman ditolak.
    if d1_bias == "bearish" and direction == "bull": return None
    if d1_bias == "bullish" and direction == "bear": return None

    return {
        "direction"       : direction,
        "confidence"      : conf,
        "price"           : L15["close"],
        "atr"             : atr_val,
        "struct_h1"       : struct_h1,
        "d1_bias"         : d1_bias,
        "rsi"             : round(rv,1),
        "bull_pts"        : bull,
        "bear_pts"        : bear,
        "bias_dir"        : bias_dir,
        "choch_m15"       : choch,
        "choch_h1"        : choch_h1,
        "failed_retest"   : fr,
        "pullback_valid"  : pullback_valid_bull if direction == "bull" else pullback_valid_bear,
        "pullback_type"   : pullback_type_bull if direction == "bull" else pullback_type_bear,
        "pinbar"          : pinbar,
        "fakey"           : fakey,
        "liquidity_bull"  : liq_bull,
        "liquidity_bear"  : liq_bear,
        "inducement"      : inducement_bull if direction == "bull" else inducement_bear,
    }


# ═════════════════════════════════════════════
# TAHAP 2: ANALISIS ULANG — SL DULU, LALU TP
# ═════════════════════════════════════════════
# ── Tier kekuatan level untuk pemilihan TP ──────────────────────────
# Tier lebih rendah = level lebih kuat/reliable sebagai target liquidity.
def _h4_confluence(df_h1, direction, choch_m15=None):
    """
    Konfirmasi H4 untuk membuka kandidat TP Fibonacci extension.
    Resample dari H1 yang sudah di-fetch — TIDAK ada API call tambahan
    (pola sama persis dengan d1_bias di score_direction()).

    Syarat 'confluence' (unlock fib 1.272):
      BUY  : EMA9>EMA21>EMA50 H4 + struktur H4 bullish + RSI H4 di [45,68]
      SELL : EMA9<EMA21<EMA50 H4 + struktur H4 bearish + RSI H4 di [32,55]

    Syarat 'full_confluence' (unlock fib 1.618, tambahan):
      confluence di atas TERPENUHI + CHoCH M15 searah trade.
      Ini level paling jauh/spekulatif — baru boleh dipakai kalau H4
      DAN M15 dan RSI semuanya sepakat, bukan cuma H4 saja.

    Return: {"confluence": bool, "full_confluence": bool}
    """
    result = {"confluence": False, "full_confluence": False}
    try:
        df_h4 = build_df(df_h1.resample("4h").agg({
            "open":"first","high":"max","low":"min",
            "close":"last","volume":"sum"
        }).dropna())
        if df_h4 is None or len(df_h4) < 20:
            return result

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


def _fib_extension_levels(h1, sh1, sl1, direction):
    """
    Proyeksi Fibonacci extension dari leg swing H1 terakhir (low→high untuk
    BUY, high→low untuk SELL). Bukan angka dikarang — ini proyeksi dari
    RENTANG pergerakan H1 yang sudah benar-benar terjadi di chart.

    Return: (fib_127_price, fib_162_price) atau (None, None) kalau swing
    H1 belum cukup terbentuk.
    """
    if not sh1 or not sl1:
        return None, None
    swing_high = h1["high"].iloc[sh1[-1]]
    swing_low  = h1["low"].iloc[sl1[-1]]
    leg = swing_high - swing_low
    if leg <= 0:
        return None, None

    if direction == "bull":
        return swing_high + leg * FIB_EXT_1, swing_high + leg * FIB_EXT_2
    else:
        return swing_low - leg * FIB_EXT_1, swing_low - leg * FIB_EXT_2


TP_RR_CAP = MIN_RR * 2   # RR di atas ini ditarik mundur ke titik RR=cap, arah tetap ke level terkuat

def _select_best_tp(tp_pool, entry_price, risk):
    """
    Pilih TP dari level PALING KUAT (tier terendah) di antara semua
    kandidat yang lolos floor RR >= MIN_RR. Seri tier → ambil RR tertinggi.
    Kalau RR ke level itu > TP_RR_CAP, TP ditarik mundur (searah entry->level)
    ke titik yang menghasilkan RR = TP_RR_CAP — arah tetap ke level kuat itu,
    cuma jaraknya dipersingkat supaya lebih realistis tersentuh.
    """
    qualifying = []
    for lbl, v, tier in tp_pool:
        rr_c = abs(v - entry_price) / risk
        if rr_c >= MIN_RR:
            qualifying.append((lbl, v, tier, rr_c))
    if not qualifying:
        return None, None
    best_lbl, best_v, best_tier, best_rr = min(qualifying, key=lambda x: (x[2], -x[3]))
    if best_rr > TP_RR_CAP:
        sgn = 1 if best_v > entry_price else -1
        best_v = entry_price + sgn * risk * TP_RR_CAP
        best_lbl += "_capped"
    return round(best_v, 8), best_lbl


def _build_tp_pool(m15, h1, direction, entry_price, atr, sh15, sl15, sh1, sl1, h4_gate, fib_127, fib_162):
    """TP pool searah entry (bear=bawah, bull=atas), tier makin kecil makin kuat."""
    up = direction == "bull"
    zones_m15 = find_zones(m15, "demand" if up else "supply")
    zones_h1  = find_zones(h1, "demand" if up else "supply")
    fvgs      = find_fvg(m15, "bull" if up else "bear")
    eqs_m15   = find_equal_highs_lows(m15, "high" if up else "low", lb=80)
    eqs_h1    = find_equal_highs_lows(h1, "high" if up else "low", lb=50)
    sw_m15    = [m15["high" if up else "low"].iloc[i] for i in (sh15 if up else sl15)]
    sw_h1     = [h1["high" if up else "low"].iloc[i] for i in (sh1 if up else sl1)]
    sgn = 1 if up else -1
    pool = []

    for v in eqs_h1:
        if sgn*(v - entry_price) > atr*1.0: pool.append(("eq_h1", v, 1))
    for z in zones_h1:
        edge = z["bot"] if up else z["top"]
        if sgn*(edge - entry_price) > atr*1.0: pool.append(("zone_h1", edge, 2))
    for v in sw_h1:
        if sgn*(v - entry_price) > atr*1.0: pool.append(("sw_h1", v, 3))
    for v in eqs_m15:
        if sgn*(v - entry_price) > atr*0.5: pool.append(("eq_m15", v, 4))
    for z in zones_m15:
        edge = z["bot"] if up else z["top"]
        if sgn*(edge - entry_price) > atr*0.5:
            pool.append(("zone_m15", edge, 5 - (0.4 if z.get("is_fresh") else 0)))
    for f in fvgs:
        if sgn*(f["mid"] - entry_price) > atr*0.5:
            t = 6 - (0.4 if f.get("candle3") == "breakaway" else 0) - (0.2 if f.get("is_fresh") else 0)
            pool.append(("fvg_m15", f["mid"], t))
    for v in sw_m15:
        if sgn*(v - entry_price) > atr*0.5: pool.append(("sw_m15", v, 7))

    if fib_127 is not None and sgn*(fib_127 - entry_price) > atr*0.5 and h4_gate["confluence"]:
        pool.append(("fib127", fib_127, 8))
        if h4_gate["full_confluence"] and fib_162 is not None and sgn*(fib_162 - entry_price) > atr*0.5:
            pool.append(("fib162", fib_162, 9))
    return pool


def analyze_setup(df_h1, df_m15, direction, entry_price, score=None, invalid_level=None):
    """
    SL = seberang titik entry itu sendiri (invalid_level dari
    calc_discount_entry) + buffer noise kecil. Kalau harga sentuh SL,
    itu artinya struktur di entry ini TERBUKTI invalid (bukan sekadar
    "belum dikonfirmasi") — bukan liquidity pool berikutnya yang jauh.
    Tidak ada level jelas → return None (skip, cari koin lain).
    TP = tier pool terkuat dengan floor RR >= MIN_RR (lihat _select_best_tp).
    """
    h1, m15 = build_df(df_h1), build_df(df_m15)
    if h1 is None or m15 is None: return None

    # ATR M15 saja bisa under-estimate kalau harga baru selesai fase
    # spike/impulsif besar lalu masuk konsolidasi sempit (M15 "tenang"
    # tapi itu semu — koin baru saja terbukti bisa bergerak liar).
    # Pakai ATR H1 juga sebagai pembanding supaya buffer SL tetap
    # proporsional terhadap volatilitas riil koin, bukan cuma window
    # M15 saat ini yang mungkin kebetulan sempit.
    atr_m15 = m15["atr"].iloc[-1]
    atr_h1  = h1["atr"].iloc[-1] / 4   # ATR H1 diskalakan kasar ke basis M15
    atr = max(atr_m15, atr_h1, entry_price * 0.002)
    noise = atr * 0.6   # buffer anti-noise — dinaikkan dari 0.25x krn ATR
                         # M15 sendirian gampang under-estimate saat harga
                         # baru keluar dari candle spike besar (lihat kasus
                         # KAITOUSDT: SL kena dlm 4 menit oleh wick biasa)

    if invalid_level is None:
        return None

    sl_price = invalid_level + (noise if direction == "bear" else -noise)
    risk = abs(sl_price - entry_price)
    risk_floor = max(atr * 0.8, entry_price * 0.003)   # dinaikkan dari 0.4x/0.0015
    if risk < risk_floor:
        sl_price += (risk_floor - risk) * (1 if direction == "bear" else -1)
        risk = risk_floor
    if risk <= 0: return None

    sh15, sl15 = swing_pts(m15, lb=5)
    sh1, sl1   = swing_pts(h1, lb=5)
    choch_m15  = (score or {}).get("choch_m15", {})
    h4_gate    = _h4_confluence(df_h1, direction, choch_m15)
    fib_127, fib_162 = _fib_extension_levels(h1, sh1, sl1, direction)

    tp_pool = _build_tp_pool(m15, h1, direction, entry_price, atr,
                              sh15, sl15, sh1, sl1, h4_gate, fib_127, fib_162)
    tp_price, tp_label = _select_best_tp(tp_pool, entry_price, risk)
    if tp_price is None: return None

    reward = abs(tp_price - entry_price)
    rr = round(reward / risk, 2)
    if rr < MIN_RR: return None

    return {
        "sl": round(sl_price, 8), "tp": round(tp_price, 8), "rr": rr,
        "reason": f"SL@{sl_price:.5g}(invalidation) | TP@{tp_price:.5g}({tp_label})",
    }




def _zone_score(z):
    """Skor kekuatan zona OB/S&D: fresh + fvg + bos + breakaway fib align."""
    return z.get("quality", 0) + int(z.get("fib_aligned", False))


def _collect_entry_candidates(m15, direction, entry_ref, atr):
    """
    Kumpulkan semua kandidat entry (OB, FVG, sweep raw level, fib adaptif)
    dengan skor kekuatan masing-masing. direction: 'bull' cari di bawah
    entry_ref, 'bear' cari di atas entry_ref.
    sweep_side: sisi zona yang jadi TITIK ENTRY (ujung sweep, dekat harga)
    invalid_side: sisi seberang zona (dipakai sebagai basis SL nanti)

    FIX PRESISI-ENTRY (v2, REBALANCED): sebelumnya kandidat cuma
    dibandingkan lewat skor kualitas mentah (freshness/FVG/BOS), TANPA
    mempertimbangkan seberapa jauh zona itu dari harga saat ini — analisa
    2 backtest independen (>600 sinyal gabungan) menemukan median jarak
    ke entry 2.2-2.5× lebih jauh drpd jarak ke TP itu sendiri, bikin TP
    sering kesentuh duluan sebelum harga sempat pullback. v1 (penalti
    jarak besar + fib_adaptive ikut bersaing bebas) TERBUKTI kelewatan:
    trade naik 56% tapi SL naik 150%, Profit Factor nyaris separuh. v2
    ini menurunkan bobot penalti jarak jadi cuma tie-break TIPIS antar
    kandidat SEJENIS yang sudah sebanding kualitasnya, dan fib_adaptif
    dikembalikan jadi last-resort murni (tapi bug invalid_level=None-nya
    tetap diperbaiki, jadi minimal BISA menghasilkan trade saat memang
    tidak ada OB/FVG/EQ sama sekali — dulu jalur itu mati total).
    """
    up = direction == "bear"
    obs = find_zones(m15, direction, strict=True)
    fvgs = find_fvg(m15, direction)
    eqs = find_equal_highs_lows(m15, "high" if up else "low", lb=80)
    cands = []

    def _dist_penalty(price):
        # REBALANCE: bobot diturunkan 0.4→0.15 setelah data run pertama
        # menunjukkan versi 0.4 terlalu agresif — SL naik 150% padahal
        # trade cuma naik 56%, Profit Factor nyaris separuh (10.2→5.3).
        # Sekarang cuma nge-geser tie-break TIPIS antar kandidat SEJENIS
        # yang kualitasnya sudah sebanding (OB vs OB, FVG vs FVG dst),
        # bukan lagi cukup besar utk bikin zona lemah-tapi-dekat ngalahin
        # zona kuat-tapi-agak-jauh.
        if atr <= 0: return 0.0
        return (abs(price - entry_ref) / atr) * 0.15

    for z in obs:
        entry_pt, invalid_pt = (z["top"], z["bot"]) if up else (z["bot"], z["top"])
        if (up and entry_pt > entry_ref + atr*0.1) or (not up and entry_pt < entry_ref - atr*0.1):
            cands.append({"price": entry_pt, "invalid": invalid_pt, "label": "ob",
                           "score": 3 + _zone_score(z) - _dist_penalty(entry_pt)})
    for f in fvgs:
        if (up and f["mid"] > entry_ref + atr*0.1) or (not up and f["mid"] < entry_ref - atr*0.1):
            sc = 2 + int(f.get("is_fresh", False)) + 2*int(f.get("candle3") == "breakaway")
            invalid_pt = f["top"] if up else f["bot"]
            cands.append({"price": f["mid"], "invalid": invalid_pt, "label": "fvg",
                           "score": sc - _dist_penalty(f["mid"])})
    eqs_sorted = sorted(eqs) if up else sorted(eqs, reverse=True)
    for lv in eqs_sorted[:1]:
        if (up and lv > entry_ref + atr*0.2) or (not up and lv < entry_ref - atr*0.2):
            cands.append({"price": lv, "invalid": lv + (atr*0.6 if up else -atr*0.6),
                           "label": "eq", "score": 2 - _dist_penalty(lv)})

    # Fib adaptif — REBALANCE: dikembalikan jadi TRUE LAST RESORT (cuma
    # dipakai kalau BENAR-BENAR tidak ada OB/FVG/EQ sama sekali), bukan
    # ikut bersaing bebas di pool utama lagi. Data run pertama (versi
    # "ikut bersaing") menunjukkan fib generik terlalu sering menang
    # padahal secara struktural lebih lemah drpd OB/FVG asli — itu
    # kontributor utama SL melonjak. Yang TETAP diperbaiki dari versi
    # asli: dulu invalid_level selalu None di jalur ini → analyze_setup
    # SELALU menolaknya (bug lama, fib_adaptive tidak pernah benar2
    # menghasilkan trade). Sekarang dikasih invalid_level yang benar
    # (tepi dalam zona) supaya minimal BISA dipakai saat memang tidak
    # ada alternatif lain — lebih baik drpd skip sepenuhnya.
    if not cands:
        try:
            sh15, sl15 = swing_pts(m15, lb=5)
            if len(sh15) >= 1 and len(sl15) >= 1:
                lo, hi = adaptive_fib_target(m15, sh15, sl15, direction)
                swing_hi = m15["high"].iloc[sh15[-1]]
                swing_lo = m15["low"].iloc[sl15[-1]]
                leg = swing_hi - swing_lo
                px = (swing_lo + leg*lo) if up else (swing_hi - leg*lo)   # tepi dangkal = lo
                invalid_fib = (swing_lo + leg*hi) if up else (swing_hi - leg*hi)  # tepi dalam = SL basis
                if (up and px > entry_ref + atr*0.1) or (not up and px < entry_ref - atr*0.1):
                    cands.append({"price": px, "invalid": invalid_fib, "label": "fib_adaptive",
                                   "score": 1.5})
        except Exception:
            pass

    return cands


def calc_discount_entry(df_h1, df_m15, direction, current_price, atr):
    """
    Entry = kandidat terkuat (OB fresh > FVG breakaway > EQ > fib adaptif),
    dibandingkan lewat skor YANG SUDAH memperhitungkan jarak dari harga
    (lihat _collect_entry_candidates) — bukan cuma kualitas mentah. Fib
    adaptif sekarang ikut bersaing di pool yang sama (bukan fallback
    terakhir saja), jadi kalau OB/FVG yang ada semuanya jauh, alternatif
    fib yang lebih reachable bisa menang.
    Return (entry_price, label, invalid_level) — invalid_level dipakai
    analyze_setup() sebagai basis SL (seberang titik entry ini sendiri).
    """
    m15 = build_df(df_m15)
    if m15 is None: return current_price, "market", None
    cands = _collect_entry_candidates(m15, direction, current_price, atr)
    if cands:
        best = max(cands, key=lambda c: c["score"])
        return round(best["price"], 8), best["label"], best["invalid"]
    return current_price, "market", None


# ═════════════════════════════════════════════
# PIPELINE ANALISIS LENGKAP
# ═════════════════════════════════════════════
def full_analyze(df_h1, df_m15, df_d1=None, symbol=None):
    """
    Score arah (H1+M15+D1) -> entry diskon (OB/FVG/EQL/Fib) -> SL/TP.
    Dataframe dikirim pemanggil (main.py), fungsi ini tidak fetch sendiri.
    """
    try:
        if df_h1 is None or df_m15 is None or df_h1.empty or df_m15.empty:
            return None

        score = score_direction(df_h1, df_m15, df_d1)
        if score is None: return None

        original_dir  = score["direction"]
        current_price = score["price"]
        atr_val       = score["atr"]
        decision      = "BUY"  if original_dir == "bull" else "SELL"

        # ── Inducement-aware confidence adjustment ───────────────────────
        # Kalau terindikasi inducement (gerakan kecil pancingan) BARU SAJA
        # terjadi dan belum ada CHoCH searah yang mengkonfirmasi shift
        # struktur, turunkan confidence sedikit — mendorong sinyal ini
        # untuk tidak lolos MIN_CONFIDENCE kalau memang masih marginal,
        # alih-alih entry di gerakan/breakout pertama yang berisiko jadi
        # jebakan (bukan hard block, supaya sinyal yang memang sangat
        # kuat dari indikator lain tetap bisa lolos).
        confidence = score["confidence"]
        choch_confirms = (
            (original_dir == "bull" and score.get("choch_m15", {}).get("bullish_choch")) or
            (original_dir == "bear" and score.get("choch_m15", {}).get("bearish_choch"))
        )
        if score.get("inducement") and not choch_confirms:
            confidence = max(0, confidence - 8)

        # Kalau pullback yang mendasari sinyal ini AGGRESSIVE (momentum
        # kuat melawan, reaksi di zona rendah probabilitasnya) turunkan
        # sedikit juga, kecuali sudah ada CHoCH searah yang menguatkan.
        if score.get("pullback_type") == "aggressive" and not choch_confirms:
            confidence = max(0, confidence - 5)

        # Entry diskon dari zona struktural
        discount_entry, entry_label, invalid_level = calc_discount_entry(
            df_h1, df_m15, original_dir, current_price, atr_val)

        # SL/TP dihitung dari entry diskon
        setup = analyze_setup(df_h1, df_m15, original_dir, discount_entry,
                               score=score, invalid_level=invalid_level)
        if setup is None: return None

        # TP wajib MASIH di depan harga sekarang. Kalau entry diskon
        # dihitung dari zona struktural yang sudah ditinggalkan jauh oleh
        # rally/dump kuat (biasanya RSI sudah ekstrem), TP hasil analisa
        # dari zona lama itu bisa sudah KELEWAT harga sekarang — sinyal
        # ini mati sebelum pending order sempat dibuat. Tolak di sini,
        # bukan menunggu pending-cancel logic menangkapnya belakangan.
        if original_dir == "bull" and current_price >= setup["tp"]:
            return None
        if original_dir == "bear" and current_price <= setup["tp"]:
            return None

        return {
            "symbol"       : symbol,
            "original_dir" : original_dir,
            "decision"     : decision,
            "confidence"   : confidence,
            "price"        : current_price,
            "entry"        : discount_entry,
            "entry_label"  : entry_label,
            "sl"           : setup["sl"],
            "tp"           : setup["tp"],
            "rr"           : setup["rr"],
            "rsi"          : score["rsi"],
            "struct_h1"    : score["struct_h1"],
            "d1_bias"      : score.get("d1_bias", "neutral"),
            "choch_m15"    : score.get("choch_m15", {}),
            "choch_h1"     : score.get("choch_h1", {}),
            "failed_retest": score.get("failed_retest", {}),
            "tp_sl_reason" : f"Entry@{discount_entry:.5g}({entry_label}) | {setup['reason']}",
        }
    except Exception as e:
        log.debug(f"[full_analyze] {symbol}: {e}")
        return None


# ═════════════════════════════════════════════
# SCAN — 1 sinyal terbaik
# ═════════════════════════════════════════════
