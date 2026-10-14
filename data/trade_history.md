# Trading Journal

---

## RAREUSDT — SELL

Trade ID: `RAREUSDT-20261013-144941-C29320`

### Setup

- Price Now Reference: 0.015474
- Entry: 0.01617
- Reason Entry: RAREUSDT SELL: SELL_PULLBACK. H4 structure BEARISH, liquidity H1 0.016165 dipadukan dengan Fib PRIMARY; refinement BEARISH_FVG. Projected M15 RSI 87.8 pada entry zone.
- Price Exp: 0.01501
- Reason Price Exp: Price Exp 0.015012: retracement thesis dianggap basi jika harga bergerak terlalu jauh tanpa menyentuh entry zone sebelum menembus boundary relevansi.
- SL: 0.01667
- Reason SL: Prediction invalidation berasal dari H4_PROTECTED_HIGH di 0.016643; SL ditempatkan di luar invalidation dan liquidity guard 0.0166412.
- TP: 0.01451
- Reason TP: TP 0.014517 dipilih dari H4_LOWER_LOW dengan RR 3.33; target quality 78.

### Management

Tidak ada trailing.

### Result

- Result: EXPIRED
- Exit Price: 0.01501
- Result Reason: Price Exp 0.015012: retracement thesis dianggap basi jika harga bergerak terlalu jauh tanpa menyentuh entry zone sebelum menembus boundary relevansi.
- PnL: -
- Created: 2026-10-13T07:49:41.971126+00:00
- Filled: -
- Closed: 2026-10-13T10:54:36.083009+00:00
- Strategy: SMC_VLT_RSI v2.2.0

---

## CTRUSDT — BUY

Trade ID: `CTRUSDT-20261013-151532-E425E1`

### Setup

- Price Now Reference: 0.013748
- Entry: 0.013602
- Reason Entry: CTRUSDT BUY: BUY_PULLBACK. H4 structure BULLISH, liquidity M15 0.0134575 dipadukan dengan Fib PRIMARY; refinement BEARISH_FVG. Projected M15 RSI 43.4 pada entry zone.
- Price Exp: 0.014353
- Reason Price Exp: Price Exp 0.014353: retracement thesis dianggap basi jika harga bergerak terlalu jauh tanpa menyentuh entry zone sebelum menembus boundary relevansi.
- SL: 0.014371
- Reason SL: Prediction invalidation berasal dari H4_SELL_SIDE_LIQUIDITY di 0.013332; SL ditempatkan di luar invalidation dan liquidity guard 0.0133627.
- TP: 0.015209
- Reason TP: TP 0.015209 dipilih dari H4_HIGHER_HIGH dengan RR 5.17; target quality 88.

### Management

Trail #1
- Old SL: 0.013291
- New SL: 0.014269
- Price: 0.014395
- Reason: Auto trail [REJECTION_STRUCTURE]: trailing lanjut: SL di higher low M15 0.014299 (buffer ATR).
- Time: 14-10-2026, 08:30 WIB

Trail #2
- Old SL: 0.014269
- New SL: 0.014324
- Price: 0.014443
- Reason: Auto trail [REJECTION_STRUCTURE]: trailing lanjut: SL di higher low M15 0.01435 (buffer ATR).
- Time: 14-10-2026, 10:31 WIB

Trail #3
- Old SL: 0.014324
- New SL: 0.014371
- Price: 0.0145
- Reason: Auto trail [REJECTION_STRUCTURE]: trailing lanjut: SL di higher low M15 0.014392 (buffer ATR).
- Time: 14-10-2026, 12:15 WIB

### Result

- Result: TRAIL
- Exit Price: 0.01437
- Result Reason: SL trailing tercapai; profit terkunci. Prediction invalidation berasal dari H4_SELL_SIDE_LIQUIDITY di 0.013332; SL ditempatkan di luar invalidation dan liquidity guard 0.0133627.
- PnL: 5.646228495809439788266431407%
- Created: 2026-10-13T08:15:32.486672+00:00
- Filled: 2026-10-13T09:00:58.101082+00:00
- Closed: 2026-10-14T05:37:36.010519+00:00
- Strategy: SMC_VLT_RSI v2.2.0

---

## AKEUSDT — BUY

Trade ID: `AKEUSDT-20261014-110612-5A3DB9`

### Setup

- Price Now Reference: 0.025406
- Entry: 0.023195
- Reason Entry: AKEUSDT BUY: BUY_PULLBACK. H4 structure BULLISH, liquidity M15 0.0231772 dipadukan dengan Fib DEEP_0.618_0.786; refinement BULLISH_FVG. Projected M15 RSI 17.5 pada entry zone.
- Price Exp: 0.025667
- Reason Price Exp: Price Exp 0.025667: retracement thesis dianggap basi jika harga bergerak terlalu jauh tanpa menyentuh entry zone sebelum menembus boundary relevansi.
- SL: 0.02238
- Reason SL: Prediction invalidation berasal dari H4_PROTECTED_LOW di 0.022432; SL ditempatkan di luar invalidation dan liquidity guard 0.0224408.
- TP: 0.025803
- Reason TP: TP 0.025803 dipilih dari H4_HIGHER_HIGH dengan RR 3.20; target quality 84.

### Management

Tidak ada trailing.

### Result

- Result: EXPIRED
- Exit Price: 0.025667
- Result Reason: Price Exp 0.025667: retracement thesis dianggap basi jika harga bergerak terlalu jauh tanpa menyentuh entry zone sebelum menembus boundary relevansi.
- PnL: -
- Created: 2026-10-14T04:06:12.686827+00:00
- Filled: -
- Closed: 2026-10-14T06:19:00.187806+00:00
- Strategy: SMC_VLT_RSI v2.2.0

---

## ANIMEUSDT — SELL

Trade ID: `ANIMEUSDT-20261013-192453-964978`

### Setup

- Price Now Reference: 0.002986
- Entry: 0.003096
- Reason Entry: ANIMEUSDT SELL: SELL_PULLBACK_RSI_DIVERGENCE. H4 structure BEARISH, liquidity M15 0.0030955 dipadukan dengan Fib DEEP_0.618_0.786; refinement BEARISH_FVG. Projected M15 RSI 68.8 pada entry zone.
- Price Exp: 0.002958
- Reason Price Exp: Price Exp 0.0029579999999999997: retracement thesis dianggap basi jika harga bergerak terlalu jauh tanpa menyentuh entry zone sebelum menembus boundary relevansi.
- SL: 0.003106
- Reason SL: Prediction invalidation berasal dari BEARISH_SWEEP_EXTREME di 0.003102; SL ditempatkan di luar invalidation.
- TP: 0.002899
- Reason TP: TP 0.002899 dipilih dari H4_LOWER_LOW dengan RR 19.70; target quality 80.

### Management

Tidak ada trailing.

### Result

- Result: EXPIRED
- Exit Price: 0.002958
- Result Reason: Price Exp 0.0029579999999999997: retracement thesis dianggap basi jika harga bergerak terlalu jauh tanpa menyentuh entry zone sebelum menembus boundary relevansi.
- PnL: -
- Created: 2026-10-13T12:24:53.390471+00:00
- Filled: -
- Closed: 2026-10-14T12:30:10.778806+00:00
- Strategy: SMC_VLT_RSI v2.2.0

---

## 1000BONKUSDT — SELL

Trade ID: `1000BONKUSDT-20261013-190656-6785AD`

### Setup

- Price Now Reference: 0.003172
- Entry: 0.003323
- Reason Entry: 1000BONKUSDT SELL: SELL_PULLBACK_RSI_DIVERGENCE. H4 structure BEARISH, liquidity H1 0.003322 dipadukan dengan Fib PRIMARY; refinement BEARISH_FVG. Projected M15 RSI 73.6 pada entry zone.
- Price Exp: 0.003148
- Reason Price Exp: Price Exp 0.0031479999999999998: retracement thesis dianggap basi jika harga bergerak terlalu jauh tanpa menyentuh entry zone sebelum menembus boundary relevansi.
- SL: 0.003364
- Reason SL: Prediction invalidation berasal dari BEARISH_SWEEP_EXTREME di 0.003359; SL ditempatkan di luar invalidation.
- TP: 0.003113
- Reason TP: TP 0.003113 dipilih dari H4_LOWER_LOW dengan RR 5.12; target quality 82.

### Management

Tidak ada trailing.

### Result

- Result: EXPIRED
- Exit Price: 0.003148
- Result Reason: Price Exp 0.0031479999999999998: retracement thesis dianggap basi jika harga bergerak terlalu jauh tanpa menyentuh entry zone sebelum menembus boundary relevansi.
- PnL: -
- Created: 2026-10-13T12:06:56.583807+00:00
- Filled: -
- Closed: 2026-10-14T12:30:28.581721+00:00
- Strategy: SMC_VLT_RSI v2.2.0

---

## SLPUSDT — SELL

Trade ID: `SLPUSDT-20261013-192452-24C126`

### Setup

- Price Now Reference: 0.000601
- Entry: 0.0006312
- Reason Entry: SLPUSDT SELL: SELL_PULLBACK_RSI_DIVERGENCE. H4 structure BEARISH, liquidity H1 0.0006311 dipadukan dengan Fib SECONDARY; refinement BEARISH_FVG. Projected M15 RSI 76.0 pada entry zone.
- Price Exp: 0.0005974
- Reason Price Exp: Price Exp 0.0005974: retracement thesis dianggap basi jika harga bergerak terlalu jauh tanpa menyentuh entry zone sebelum menembus boundary relevansi.
- SL: 0.0006439
- Reason SL: Prediction invalidation berasal dari BEARISH_SWEEP_EXTREME di 0.0006431; SL ditempatkan di luar invalidation.
- TP: 0.0005973
- Reason TP: TP 0.0005973 dipilih dari H4_LOWER_LOW dengan RR 2.67; target quality 84.

### Management

Tidak ada trailing.

### Result

- Result: EXPIRED
- Exit Price: 0.0005974
- Result Reason: Price Exp 0.0005974: retracement thesis dianggap basi jika harga bergerak terlalu jauh tanpa menyentuh entry zone sebelum menembus boundary relevansi.
- PnL: -
- Created: 2026-10-13T12:24:52.381800+00:00
- Filled: -
- Closed: 2026-10-14T13:19:55.777553+00:00
- Strategy: SMC_VLT_RSI v2.2.0

---

## XANUSDT — SELL

Trade ID: `XANUSDT-20261013-192250-D74EF4`

### Setup

- Price Now Reference: 0.010259
- Entry: 0.010988
- Reason Entry: XANUSDT SELL: SELL_PULLBACK_RSI_DIVERGENCE. H4 structure BEARISH, liquidity M15 0.0109843 dipadukan dengan Fib SECONDARY; refinement BULLISH_FVG. Projected M15 RSI 75.8 pada entry zone.
- Price Exp: 0.010173
- Reason Price Exp: Price Exp 0.010173: retracement thesis dianggap basi jika harga bergerak terlalu jauh tanpa menyentuh entry zone sebelum menembus boundary relevansi.
- SL: 0.01118
- Reason SL: Prediction invalidation berasal dari H4_PROTECTED_HIGH di 0.011163; SL ditempatkan di luar invalidation.
- TP: 0.010086
- Reason TP: TP 0.010086 dipilih dari H1_SELL_SIDE_LIQUIDITY dengan RR 4.70; target quality 61.

### Management

Tidak ada trailing.

### Result

- Result: EXPIRED
- Exit Price: 0.010172
- Result Reason: Price Exp 0.010173: retracement thesis dianggap basi jika harga bergerak terlalu jauh tanpa menyentuh entry zone sebelum menembus boundary relevansi.
- PnL: -
- Created: 2026-10-13T12:22:50.189598+00:00
- Filled: -
- Closed: 2026-10-14T14:34:10.946351+00:00
- Strategy: SMC_VLT_RSI v2.2.0

---

## AKEUSDT — BUY

Trade ID: `AKEUSDT-20261014-213018-3D8F8C`

### Setup

- Price Now Reference: 0.025278
- Entry: 0.023334
- Reason Entry: AKEUSDT BUY: BUY_PULLBACK_RSI_DIVERGENCE. H4 structure BULLISH, liquidity M15 0.023077 dipadukan dengan Fib DEEP_0.618_0.786; refinement BULLISH_FVG. Projected M15 RSI 22.8 pada entry zone.
- Price Exp: 0.02564
- Reason Price Exp: Price Exp 0.02564: retracement thesis dianggap basi jika harga bergerak terlalu jauh tanpa menyentuh entry zone sebelum menembus boundary relevansi.
- SL: 0.022359
- Reason SL: Prediction invalidation berasal dari H4_PROTECTED_LOW di 0.022432; SL ditempatkan di luar invalidation dan liquidity guard 0.0224408.
- TP: 0.025803
- Reason TP: TP 0.025803 dipilih dari H4_HIGHER_HIGH dengan RR 2.53; target quality 87.

### Management

Tidak ada trailing.

### Result

- Result: EXPIRED
- Exit Price: 0.02564
- Result Reason: Price Exp 0.02564: retracement thesis dianggap basi jika harga bergerak terlalu jauh tanpa menyentuh entry zone sebelum menembus boundary relevansi.
- PnL: -
- Created: 2026-10-14T14:30:18.277697+00:00
- Filled: -
- Closed: 2026-10-14T15:13:15.600012+00:00
- Strategy: SMC_VLT_RSI v2.2.0

---

## LDOUSDT — SELL

Trade ID: `LDOUSDT-20261014-230610-0A4C4F`

### Setup

- Price Now Reference: 0.4063
- Entry: 0.4427
- Reason Entry: LDOUSDT SELL: SELL_PULLBACK. H4 structure BEARISH, liquidity H4 0.4413 dipadukan dengan Fib SECONDARY; refinement BEARISH_BREAKER. Projected M15 RSI 78.8 pada entry zone.
- Price Exp: 0.4059
- Reason Price Exp: Price Exp 0.40590000000000004: retracement thesis dianggap basi jika harga bergerak terlalu jauh tanpa menyentuh entry zone sebelum menembus boundary relevansi.
- SL: 0.4589
- Reason SL: Prediction invalidation berasal dari H4_BUY_SIDE_LIQUIDITY di 0.458192; SL ditempatkan di luar invalidation dan liquidity guard 0.4578.
- TP: 0.3902
- Reason TP: TP 0.3902 dipilih dari H4_LOWER_LOW dengan RR 3.24; target quality 79.

### Management

Tidak ada trailing.

### Result

- Result: EXPIRED
- Exit Price: 0.4059
- Result Reason: Price Exp 0.40590000000000004: retracement thesis dianggap basi jika harga bergerak terlalu jauh tanpa menyentuh entry zone sebelum menembus boundary relevansi.
- PnL: -
- Created: 2026-10-14T16:06:10.904370+00:00
- Filled: -
- Closed: 2026-10-14T16:11:28.988110+00:00
- Strategy: SMC_VLT_RSI v2.2.0
