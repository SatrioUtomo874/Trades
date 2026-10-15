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

---

## NEWTUSDT — SELL

Trade ID: `NEWTUSDT-20261013-192251-771ED5`

### Setup

- Price Now Reference: 0.04504
- Entry: 0.04572
- Reason Entry: NEWTUSDT SELL: SELL_PULLBACK. H4 structure BEARISH, liquidity M15 0.04572 dipadukan dengan Fib SECONDARY; refinement BEARISH_FVG. Projected M15 RSI 50.7 pada entry zone.
- Price Exp: 0.04427
- Reason Price Exp: Price Exp 0.044270000000000004: retracement thesis dianggap basi jika harga bergerak terlalu jauh tanpa menyentuh entry zone sebelum menembus boundary relevansi.
- SL: 0.0443
- Reason SL: Prediction invalidation berasal dari BEARISH_SWEEP_EXTREME di 0.04644; SL ditempatkan di luar invalidation.
- TP: 0.04366
- Reason TP: TP 0.04366 dipilih dari H1_SELL_SIDE_LIQUIDITY dengan RR 2.61; target quality 68.

### Management

Trail #1
- Old SL: 0.04651
- New SL: 0.04527
- Price: 0.04438
- Reason: Auto trail [REJECTION_STRUCTURE]: trailing lanjut: SL di lower high M15 0.04521 (buffer ATR).
- Time: 14-10-2026, 20:16 WIB

Trail #2
- Old SL: 0.04527
- New SL: 0.04494
- Price: 0.04398
- Reason: Auto trail [REJECTION_STRUCTURE]: trailing lanjut: SL di lower high M15 0.04487 (buffer ATR).
- Time: 14-10-2026, 21:46 WIB

Trail #3
- Old SL: 0.04494
- New SL: 0.0443
- Price: 0.04392
- Reason: Auto trail [REJECTION_STRUCTURE]: trailing lanjut: SL di lower high M15 0.04423 (buffer ATR).
- Time: 14-10-2026, 23:01 WIB

### Result

- Result: TP
- Exit Price: 0.04366
- Result Reason: TP 0.04366 dipilih dari H1_SELL_SIDE_LIQUIDITY dengan RR 2.61; target quality 68.
- PnL: 4.505686789151356080489938758%
- Created: 2026-10-13T12:22:51.193556+00:00
- Filled: 2026-10-13T13:38:46.207610+00:00
- Closed: 2026-10-14T16:15:09.101870+00:00
- Strategy: SMC_VLT_RSI v2.2.0

---

## IOSTUSDT — SELL

Trade ID: `IOSTUSDT-20261014-231957-4D093F`

### Setup

- Price Now Reference: 0.0008008
- Entry: 0.0008217
- Reason Entry: IOSTUSDT SELL: SELL_PULLBACK. H4 structure BEARISH, liquidity M15 0.0008189 dipadukan dengan Fib PRIMARY; refinement BULLISH_FVG. Projected M15 RSI 60.7 pada entry zone.
- Price Exp: 0.0007974
- Reason Price Exp: Price Exp 0.0007974: retracement thesis dianggap basi jika harga bergerak terlalu jauh tanpa menyentuh entry zone sebelum menembus boundary relevansi.
- SL: 0.000833
- Reason SL: Prediction invalidation berasal dari H1_PROTECTED_HIGH di 0.000832; SL ditempatkan di luar invalidation.
- TP: 0.000787
- Reason TP: TP 0.000787 dipilih dari H4_LOWER_LOW dengan RR 3.07; target quality 86.

### Management

Tidak ada trailing.

### Result

- Result: EXPIRED
- Exit Price: 0.0007974
- Result Reason: Price Exp 0.0007974: retracement thesis dianggap basi jika harga bergerak terlalu jauh tanpa menyentuh entry zone sebelum menembus boundary relevansi.
- PnL: -
- Created: 2026-10-14T16:19:57.507332+00:00
- Filled: -
- Closed: 2026-10-15T00:49:26.989026+00:00
- Strategy: SMC_VLT_RSI v2.2.0

---

## LAUSDT — SELL

Trade ID: `LAUSDT-20261014-232313-CE9929`

### Setup

- Price Now Reference: 0.05997
- Entry: 0.06243
- Reason Entry: LAUSDT SELL: SELL_PULLBACK. H4 structure BEARISH, liquidity M15 0.06242 dipadukan dengan Fib SECONDARY; refinement BEARISH_FVG. Projected M15 RSI 68.5 pada entry zone.
- Price Exp: 0.05938
- Reason Price Exp: Price Exp 0.05938: retracement thesis dianggap basi jika harga bergerak terlalu jauh tanpa menyentuh entry zone sebelum menembus boundary relevansi.
- SL: 0.06386
- Reason SL: Prediction invalidation berasal dari BEARISH_SWEEP_EXTREME di 0.06378; SL ditempatkan di luar invalidation.
- TP: 0.05808
- Reason TP: TP 0.05808 dipilih dari H4_LOWER_LOW dengan RR 3.04; target quality 83.

### Management

Tidak ada trailing.

### Result

- Result: EXPIRED
- Exit Price: 0.05938
- Result Reason: Price Exp 0.05938: retracement thesis dianggap basi jika harga bergerak terlalu jauh tanpa menyentuh entry zone sebelum menembus boundary relevansi.
- PnL: -
- Created: 2026-10-14T16:23:13.994946+00:00
- Filled: -
- Closed: 2026-10-15T00:49:42.203986+00:00
- Strategy: SMC_VLT_RSI v2.2.0

---

## ORDIUSDT — SELL

Trade ID: `ORDIUSDT-20261014-230922-62E619`

### Setup

- Price Now Reference: 3.977
- Entry: 4.263
- Reason Entry: ORDIUSDT SELL: SELL_PULLBACK. H4 structure BEARISH, liquidity H1 4.262 dipadukan dengan Fib PRIMARY; refinement BEARISH_FVG. Projected M15 RSI 79.5 pada entry zone.
- Price Exp: 3.952
- Reason Price Exp: Price Exp 3.952: retracement thesis dianggap basi jika harga bergerak terlalu jauh tanpa menyentuh entry zone sebelum menembus boundary relevansi.
- SL: 4.453
- Reason SL: Prediction invalidation berasal dari H1_BUY_SIDE_LIQUIDITY di 4.44778; SL ditempatkan di luar invalidation dan liquidity guard 4.444.
- TP: 3.865
- Reason TP: TP 3.865 dipilih dari H1_SWING_LOW dengan RR 2.09; target quality 64.

### Management

Tidak ada trailing.

### Result

- Result: EXPIRED
- Exit Price: 3.952
- Result Reason: Price Exp 3.952: retracement thesis dianggap basi jika harga bergerak terlalu jauh tanpa menyentuh entry zone sebelum menembus boundary relevansi.
- PnL: -
- Created: 2026-10-14T16:09:22.283096+00:00
- Filled: -
- Closed: 2026-10-15T01:57:05.736243+00:00
- Strategy: SMC_VLT_RSI v2.2.0

---

## BANANAUSDT — SELL

Trade ID: `BANANAUSDT-20261013-192454-47DE17`

### Setup

- Price Now Reference: 3.628
- Entry: 3.786
- Reason Entry: BANANAUSDT SELL: SELL_PULLBACK_RSI_DIVERGENCE. H4 structure BEARISH, liquidity M15 3.78475 dipadukan dengan Fib SECONDARY; refinement BEARISH_FVG. Projected M15 RSI 77.3 pada entry zone.
- Price Exp: 3.55
- Reason Price Exp: Price Exp 3.5500000000000003: retracement thesis dianggap basi jika harga bergerak terlalu jauh tanpa menyentuh entry zone sebelum menembus boundary relevansi.
- SL: 3.853
- Reason SL: Prediction invalidation berasal dari BEARISH_SWEEP_EXTREME di 3.843; SL ditempatkan di luar invalidation dan liquidity guard 3.84315.
- TP: 3.45
- Reason TP: TP 3.45 dipilih dari H1_SWING_LOW dengan RR 5.01; target quality 62.

### Management

Tidak ada trailing.

### Result

- Result: EXPIRED
- Exit Price: 3.55
- Result Reason: Price Exp 3.5500000000000003: retracement thesis dianggap basi jika harga bergerak terlalu jauh tanpa menyentuh entry zone sebelum menembus boundary relevansi.
- PnL: -
- Created: 2026-10-13T12:24:54.394491+00:00
- Filled: -
- Closed: 2026-10-15T02:22:21.682741+00:00
- Strategy: SMC_VLT_RSI v2.2.0

---

## PEOPLEUSDT — SELL

Trade ID: `PEOPLEUSDT-20261014-231956-D21138`

### Setup

- Price Now Reference: 0.007686
- Entry: 0.007828
- Reason Entry: PEOPLEUSDT SELL: SELL_PULLBACK. H4 structure BEARISH, liquidity M15 0.007828 dipadukan dengan Fib PRIMARY; refinement BEARISH_FVG. Projected M15 RSI 56.9 pada entry zone.
- Price Exp: 0.007522
- Reason Price Exp: Price Exp 0.007522: retracement thesis dianggap basi jika harga bergerak terlalu jauh tanpa menyentuh entry zone sebelum menembus boundary relevansi.
- SL: 0.007786
- Reason SL: Prediction invalidation berasal dari BEARISH_SWEEP_EXTREME di 0.007943; SL ditempatkan di luar invalidation.
- TP: 0.007391
- Reason TP: TP 0.007391 dipilih dari H4_LOWER_LOW dengan RR 3.47; target quality 86.

### Management

Trail #1
- Old SL: 0.007954
- New SL: 0.007786
- Price: 0.007632
- Reason: Auto trail [REJECTION_STRUCTURE]: rejection SWEEP: SL di lower high M15 0.007774 (buffer ATR).
- Time: 15-10-2026, 08:46 WIB

### Result

- Result: TP
- Exit Price: 0.00739
- Result Reason: TP 0.007391 dipilih dari H4_LOWER_LOW dengan RR 3.47; target quality 86.
- PnL: 5.595298926928972917731221257%
- Created: 2026-10-14T16:19:56.501590+00:00
- Filled: 2026-10-14T17:01:06.516699+00:00
- Closed: 2026-10-15T02:23:12.836741+00:00
- Strategy: SMC_VLT_RSI v2.2.0

---

## RENDERUSDT — SELL

Trade ID: `RENDERUSDT-20261014-230609-E51AF0`

### Setup

- Price Now Reference: 1.902
- Entry: 2.009
- Reason Entry: RENDERUSDT SELL: SELL_PULLBACK. H4 structure BEARISH, liquidity H1 2.00563 dipadukan dengan Fib SECONDARY; refinement BULLISH_FVG. Projected M15 RSI 69.3 pada entry zone.
- Price Exp: 1.832
- Reason Price Exp: Price Exp 1.832: retracement thesis dianggap basi jika harga bergerak terlalu jauh tanpa menyentuh entry zone sebelum menembus boundary relevansi.
- SL: 2.099
- Reason SL: Prediction invalidation berasal dari H4_PROTECTED_HIGH di 2.094; SL ditempatkan di luar invalidation.
- TP: 1.757
- Reason TP: TP 1.757 dipilih dari H4_LOWER_LOW dengan RR 2.80; target quality 82.

### Management

Tidak ada trailing.

### Result

- Result: EXPIRED
- Exit Price: 1.832
- Result Reason: Price Exp 1.832: retracement thesis dianggap basi jika harga bergerak terlalu jauh tanpa menyentuh entry zone sebelum menembus boundary relevansi.
- PnL: -
- Created: 2026-10-14T16:06:09.898912+00:00
- Filled: -
- Closed: 2026-10-15T02:24:19.109918+00:00
- Strategy: SMC_VLT_RSI v2.2.0

---

## PIEVERSEUSDT — SELL

Trade ID: `PIEVERSEUSDT-20261015-094304-36CCAC`

### Setup

- Price Now Reference: 1.1695
- Entry: 1.1882
- Reason Entry: PIEVERSEUSDT SELL: SELL_PULLBACK_RSI_DIVERGENCE. H4 structure BEARISH, liquidity M15 1.188 dipadukan dengan Fib PRIMARY; refinement BEARISH_FVG. Projected M15 RSI 62.0 pada entry zone.
- Price Exp: 1.1627
- Reason Price Exp: Price Exp 1.1627: retracement thesis dianggap basi jika harga bergerak terlalu jauh tanpa menyentuh entry zone sebelum menembus boundary relevansi.
- SL: 1.1984
- Reason SL: Prediction invalidation berasal dari H1_PROTECTED_HIGH di 1.1968; SL ditempatkan di luar invalidation.
- TP: 1.1518
- Reason TP: TP 1.1518 dipilih dari H1_SWING_LOW dengan RR 3.57; target quality 72.

### Management

Tidak ada trailing.

### Result

- Result: EXPIRED
- Exit Price: 1.1627
- Result Reason: Price Exp 1.1627: retracement thesis dianggap basi jika harga bergerak terlalu jauh tanpa menyentuh entry zone sebelum menembus boundary relevansi.
- PnL: -
- Created: 2026-10-15T02:43:04.782801+00:00
- Filled: -
- Closed: 2026-10-15T03:02:02.234644+00:00
- Strategy: SMC_VLT_RSI v2.2.0

---

## CETUSUSDT — SELL

Trade ID: `CETUSUSDT-20261015-095241-275401`

### Setup

- Price Now Reference: 0.023851
- Entry: 0.02525
- Reason Entry: CETUSUSDT SELL: SELL_PULLBACK. H4 structure BEARISH, liquidity M15 0.0252415 dipadukan dengan Fib PRIMARY; refinement BEARISH_FVG. Projected M15 RSI 68.6 pada entry zone.
- Price Exp: 0.02365
- Reason Price Exp: Price Exp 0.023653: retracement thesis dianggap basi jika harga bergerak terlalu jauh tanpa menyentuh entry zone sebelum menembus boundary relevansi.
- SL: 0.02583
- Reason SL: Prediction invalidation berasal dari H4_PROTECTED_HIGH di 0.025788; SL ditempatkan di luar invalidation dan liquidity guard 0.0257638.
- TP: 0.02341
- Reason TP: TP 0.023415 dipilih dari H4_SELL_SIDE_LIQUIDITY dengan RR 3.17; target quality 66.

### Management

Tidak ada trailing.

### Result

- Result: EXPIRED
- Exit Price: 0.02365
- Result Reason: Price Exp 0.023653: retracement thesis dianggap basi jika harga bergerak terlalu jauh tanpa menyentuh entry zone sebelum menembus boundary relevansi.
- PnL: -
- Created: 2026-10-15T02:52:41.089037+00:00
- Filled: -
- Closed: 2026-10-15T03:03:09.398928+00:00
- Strategy: SMC_VLT_RSI v2.2.0

---

## XAIUSDT — SELL

Trade ID: `XAIUSDT-20261015-094619-A57B66`

### Setup

- Price Now Reference: 0.008519
- Entry: 0.008909
- Reason Entry: XAIUSDT SELL: SELL_PULLBACK. H4 structure BEARISH, liquidity M15 0.0089065 dipadukan dengan Fib DEEP_0.618_0.786; refinement BEARISH_FVG. Projected M15 RSI 62.3 pada entry zone.
- Price Exp: 0.008442
- Reason Price Exp: Price Exp 0.008442: retracement thesis dianggap basi jika harga bergerak terlalu jauh tanpa menyentuh entry zone sebelum menembus boundary relevansi.
- SL: 0.009119
- Reason SL: Prediction invalidation berasal dari H1_PROTECTED_HIGH di 0.008954; SL ditempatkan di luar invalidation dan liquidity guard 0.0090440417.
- TP: 0.008243
- Reason TP: TP 0.008243 dipilih dari H4_LOWER_LOW dengan RR 3.17; target quality 83.

### Management

Tidak ada trailing.

### Result

- Result: EXPIRED
- Exit Price: 0.008442
- Result Reason: Price Exp 0.008442: retracement thesis dianggap basi jika harga bergerak terlalu jauh tanpa menyentuh entry zone sebelum menembus boundary relevansi.
- PnL: -
- Created: 2026-10-15T02:46:19.684920+00:00
- Filled: -
- Closed: 2026-10-15T03:27:54.067869+00:00
- Strategy: SMC_VLT_RSI v2.2.0

---

## AAVEUSDT — SELL

Trade ID: `AAVEUSDT-20261015-110303-BE3775`

### Setup

- Price Now Reference: 150.7
- Entry: 158.44
- Reason Entry: AAVEUSDT SELL: SELL_PULLBACK. H4 structure BEARISH, liquidity H1 158.42 dipadukan dengan Fib DEEP_0.618_0.786; refinement BEARISH_BREAKER. Projected M15 RSI 74.6 pada entry zone.
- Price Exp: 150.34
- Reason Price Exp: Price Exp 150.34: retracement thesis dianggap basi jika harga bergerak terlalu jauh tanpa menyentuh entry zone sebelum menembus boundary relevansi.
- SL: 161.68
- Reason SL: Prediction invalidation berasal dari BEARISH_SWEEP_EXTREME di 161.47; SL ditempatkan di luar invalidation dan liquidity guard 161.512.
- TP: 146.88
- Reason TP: TP 146.88 dipilih dari H4_SELL_SIDE_LIQUIDITY dengan RR 3.57; target quality 61.

### Management

Tidak ada trailing.

### Result

- Result: EXPIRED
- Exit Price: 150.34
- Result Reason: Price Exp 150.34: retracement thesis dianggap basi jika harga bergerak terlalu jauh tanpa menyentuh entry zone sebelum menembus boundary relevansi.
- PnL: -
- Created: 2026-10-15T04:03:03.777792+00:00
- Filled: -
- Closed: 2026-10-15T04:05:00.470479+00:00
- Strategy: SMC_VLT_RSI v2.2.0

---

## HOMEUSDT — SELL

Trade ID: `HOMEUSDT-20261015-111645-FBF396`

### Setup

- Price Now Reference: 0.00479
- Entry: 0.005161
- Reason Entry: HOMEUSDT SELL: SELL_PULLBACK_RSI_DIVERGENCE. H4 structure BEARISH, liquidity H1 0.005161 dipadukan dengan Fib DEEP_0.618_0.786; refinement BEARISH_FVG. Projected M15 RSI 71.3 pada entry zone.
- Price Exp: 0.004785
- Reason Price Exp: Price Exp 0.004785: retracement thesis dianggap basi jika harga bergerak terlalu jauh tanpa menyentuh entry zone sebelum menembus boundary relevansi.
- SL: 0.00539
- Reason SL: Prediction invalidation berasal dari H4_PROTECTED_HIGH di 0.005381; SL ditempatkan di luar invalidation.
- TP: 0.004624
- Reason TP: TP 0.004624 dipilih dari H4_SELL_SIDE_LIQUIDITY dengan RR 2.34; target quality 63.

### Management

Tidak ada trailing.

### Result

- Result: EXPIRED
- Exit Price: 0.004784
- Result Reason: Price Exp 0.004785: retracement thesis dianggap basi jika harga bergerak terlalu jauh tanpa menyentuh entry zone sebelum menembus boundary relevansi.
- PnL: -
- Created: 2026-10-15T04:16:45.877964+00:00
- Filled: -
- Closed: 2026-10-15T04:16:46.759098+00:00
- Strategy: SMC_VLT_RSI v2.2.0

---

## EULUSDT — SELL

Trade ID: `EULUSDT-20261015-112331-6794E0`

### Setup

- Price Now Reference: 1.2178
- Entry: 1.2996
- Reason Entry: EULUSDT SELL: SELL_PULLBACK. H4 structure BEARISH, liquidity H1 1.2994 dipadukan dengan Fib DEEP_0.618_0.786; refinement BULLISH_FVG. Projected M15 RSI 70.5 pada entry zone.
- Price Exp: 1.2149
- Reason Price Exp: Price Exp 1.2149: retracement thesis dianggap basi jika harga bergerak terlalu jauh tanpa menyentuh entry zone sebelum menembus boundary relevansi.
- SL: 1.3365
- Reason SL: Prediction invalidation berasal dari H1_BUY_SIDE_LIQUIDITY di 1.33425; SL ditempatkan di luar invalidation dan liquidity guard 1.3321.
- TP: 1.1787
- Reason TP: TP 1.1787 dipilih dari H4_SELL_SIDE_LIQUIDITY dengan RR 3.28; target quality 65.

### Management

Tidak ada trailing.

### Result

- Result: EXPIRED
- Exit Price: 1.2149
- Result Reason: Price Exp 1.2149: retracement thesis dianggap basi jika harga bergerak terlalu jauh tanpa menyentuh entry zone sebelum menembus boundary relevansi.
- PnL: -
- Created: 2026-10-15T04:23:31.384701+00:00
- Filled: -
- Closed: 2026-10-15T05:03:50.484683+00:00
- Strategy: SMC_VLT_RSI v2.2.0

---

## WAXPUSDT — SELL

Trade ID: `WAXPUSDT-20261015-120507-ED0C07`

### Setup

- Price Now Reference: 0.004962
- Entry: 0.005189
- Reason Entry: WAXPUSDT SELL: SELL_PULLBACK_RSI_DIVERGENCE. H4 structure BEARISH, liquidity H1 0.005188 dipadukan dengan Fib DEEP_0.618_0.786; refinement BEARISH_FVG. Projected M15 RSI 72.6 pada entry zone.
- Price Exp: 0.004946
- Reason Price Exp: Price Exp 0.004946: retracement thesis dianggap basi jika harga bergerak terlalu jauh tanpa menyentuh entry zone sebelum menembus boundary relevansi.
- SL: 0.005342
- Reason SL: Prediction invalidation berasal dari BEARISH_SWEEP_EXTREME di 0.005335; SL ditempatkan di luar invalidation.
- TP: 0.004842
- Reason TP: TP 0.004842 dipilih dari H4_SELL_SIDE_LIQUIDITY dengan RR 2.27; target quality 64.

### Management

Tidak ada trailing.

### Result

- Result: EXPIRED
- Exit Price: 0.004945
- Result Reason: Price Exp 0.004946: retracement thesis dianggap basi jika harga bergerak terlalu jauh tanpa menyentuh entry zone sebelum menembus boundary relevansi.
- PnL: -
- Created: 2026-10-15T05:05:07.882461+00:00
- Filled: -
- Closed: 2026-10-15T05:08:12.024415+00:00
- Strategy: SMC_VLT_RSI v2.2.0

---

## SXTUSDT — SELL

Trade ID: `SXTUSDT-20261015-112332-F29212`

### Setup

- Price Now Reference: 0.00812
- Entry: 0.008543
- Reason Entry: SXTUSDT SELL: SELL_PULLBACK. H4 structure BEARISH, liquidity M15 0.008542 dipadukan dengan Fib PRIMARY; refinement BEARISH_OB. Projected M15 RSI 69.3 pada entry zone.
- Price Exp: 0.008112
- Reason Price Exp: Price Exp 0.008112: retracement thesis dianggap basi jika harga bergerak terlalu jauh tanpa menyentuh entry zone sebelum menembus boundary relevansi.
- SL: 0.00871
- Reason SL: Prediction invalidation berasal dari H1_PROTECTED_HIGH di 0.008695; SL ditempatkan di luar invalidation.
- TP: 0.007928
- Reason TP: TP 0.007928 dipilih dari H4_SELL_SIDE_LIQUIDITY dengan RR 3.68; target quality 64.

### Management

Tidak ada trailing.

### Result

- Result: EXPIRED
- Exit Price: 0.008112
- Result Reason: Price Exp 0.008112: retracement thesis dianggap basi jika harga bergerak terlalu jauh tanpa menyentuh entry zone sebelum menembus boundary relevansi.
- PnL: -
- Created: 2026-10-15T04:23:32.388535+00:00
- Filled: -
- Closed: 2026-10-15T05:10:06.112135+00:00
- Strategy: SMC_VLT_RSI v2.2.0

---

## ARKMUSDT — SELL

Trade ID: `ARKMUSDT-20261015-094620-D104DE`

### Setup

- Price Now Reference: 0.12213
- Entry: 0.1275
- Reason Entry: ARKMUSDT SELL: SELL_PULLBACK. H4 structure BEARISH, liquidity M15 0.12739 dipadukan dengan Fib PRIMARY; refinement BULLISH_FVG. Projected M15 RSI 61.4 pada entry zone.
- Price Exp: 0.1195
- Reason Price Exp: Price Exp 0.11956000000000001: retracement thesis dianggap basi jika harga bergerak terlalu jauh tanpa menyentuh entry zone sebelum menembus boundary relevansi.
- SL: 0.1287
- Reason SL: Prediction invalidation berasal dari M15_BUY_SIDE_LIQUIDITY di 0.128405; SL ditempatkan di luar invalidation dan liquidity guard 0.1280775.
- TP: 0.1161
- Reason TP: TP 0.11619 dipilih dari H4_LOWER_LOW dengan RR 9.71; target quality 79.

### Management

Tidak ada trailing.

### Result

- Result: EXPIRED
- Exit Price: 0.1195
- Result Reason: Price Exp 0.11956000000000001: retracement thesis dianggap basi jika harga bergerak terlalu jauh tanpa menyentuh entry zone sebelum menembus boundary relevansi.
- PnL: -
- Created: 2026-10-15T02:46:20.692665+00:00
- Filled: -
- Closed: 2026-10-15T05:10:30.168707+00:00
- Strategy: SMC_VLT_RSI v2.2.0

---

## 1INCHUSDT — SELL

Trade ID: `1INCHUSDT-20261015-112007-861653`

### Setup

- Price Now Reference: 0.09121
- Entry: 0.09517
- Reason Entry: 1INCHUSDT SELL: SELL_PULLBACK. H4 structure BEARISH, liquidity H1 0.09516 dipadukan dengan Fib DEEP_0.618_0.786; refinement BEARISH_FVG. Projected M15 RSI 72.6 pada entry zone.
- Price Exp: 0.09111
- Reason Price Exp: Price Exp 0.09111000000000001: retracement thesis dianggap basi jika harga bergerak terlalu jauh tanpa menyentuh entry zone sebelum menembus boundary relevansi.
- SL: 0.09724
- Reason SL: Prediction invalidation berasal dari H1_BUY_SIDE_LIQUIDITY di 0.09713; SL ditempatkan di luar invalidation dan liquidity guard 0.0969173.
- TP: 0.08937
- Reason TP: TP 0.08937 dipilih dari H4_SELL_SIDE_LIQUIDITY dengan RR 2.80; target quality 64.

### Management

Tidak ada trailing.

### Result

- Result: EXPIRED
- Exit Price: 0.09111
- Result Reason: Price Exp 0.09111000000000001: retracement thesis dianggap basi jika harga bergerak terlalu jauh tanpa menyentuh entry zone sebelum menembus boundary relevansi.
- PnL: -
- Created: 2026-10-15T04:20:07.889062+00:00
- Filled: -
- Closed: 2026-10-15T05:31:39.297835+00:00
- Strategy: SMC_VLT_RSI v2.2.0

---

## DOGEUSDT — SELL

Trade ID: `DOGEUSDT-20261015-110304-9A96FA`

### Setup

- Price Now Reference: 0.08003
- Entry: 0.08322
- Reason Entry: DOGEUSDT SELL: SELL_PULLBACK. H4 structure BEARISH, liquidity H1 0.08321 dipadukan dengan Fib DEEP_0.618_0.786; refinement BEARISH_FVG. Projected M15 RSI 71.0 pada entry zone.
- Price Exp: 0.07923
- Reason Price Exp: Price Exp 0.07923000000000001: retracement thesis dianggap basi jika harga bergerak terlalu jauh tanpa menyentuh entry zone sebelum menembus boundary relevansi.
- SL: 0.08511
- Reason SL: Prediction invalidation berasal dari H1_BUY_SIDE_LIQUIDITY di 0.085005; SL ditempatkan di luar invalidation dan liquidity guard 0.0849525.
- TP: 0.07752
- Reason TP: TP 0.07752 dipilih dari H4_SELL_SIDE_LIQUIDITY dengan RR 3.02; target quality 64.

### Management

Tidak ada trailing.

### Result

- Result: EXPIRED
- Exit Price: 0.07923
- Result Reason: Price Exp 0.07923000000000001: retracement thesis dianggap basi jika harga bergerak terlalu jauh tanpa menyentuh entry zone sebelum menembus boundary relevansi.
- PnL: -
- Created: 2026-10-15T04:03:04.781309+00:00
- Filled: -
- Closed: 2026-10-15T05:47:37.616102+00:00
- Strategy: SMC_VLT_RSI v2.2.0

---

## QTUMUSDT — SELL

Trade ID: `QTUMUSDT-20261015-121052-E8A415`

### Setup

- Price Now Reference: 0.8818
- Entry: 0.9323
- Reason Entry: QTUMUSDT SELL: SELL_PULLBACK. H4 structure BEARISH, liquidity M15 0.93215 dipadukan dengan Fib DEEP_0.618_0.786; refinement BEARISH_FVG. Projected M15 RSI 70.7 pada entry zone.
- Price Exp: 0.8753
- Reason Price Exp: Price Exp 0.8753000000000001: retracement thesis dianggap basi jika harga bergerak terlalu jauh tanpa menyentuh entry zone sebelum menembus boundary relevansi.
- SL: 0.9478
- Reason SL: Prediction invalidation berasal dari BEARISH_SWEEP_EXTREME di 0.9444; SL ditempatkan di luar invalidation dan liquidity guard 0.9448036.
- TP: 0.8613
- Reason TP: TP 0.8613 dipilih dari H4_SELL_SIDE_LIQUIDITY dengan RR 4.58; target quality 61.

### Management

Tidak ada trailing.

### Result

- Result: EXPIRED
- Exit Price: 0.8753
- Result Reason: Price Exp 0.8753000000000001: retracement thesis dianggap basi jika harga bergerak terlalu jauh tanpa menyentuh entry zone sebelum menembus boundary relevansi.
- PnL: -
- Created: 2026-10-15T05:10:52.187703+00:00
- Filled: -
- Closed: 2026-10-15T05:51:31.395074+00:00
- Strategy: SMC_VLT_RSI v2.2.0

---

## NIGHTUSDT — SELL

Trade ID: `NIGHTUSDT-20261015-093641-F34106`

### Setup

- Price Now Reference: 0.04248
- Entry: 0.04465
- Reason Entry: NIGHTUSDT SELL: SELL_PULLBACK. H4 structure BEARISH, liquidity H1 0.044645 dipadukan dengan Fib PRIMARY; refinement BEARISH_FVG. Projected M15 RSI 70.8 pada entry zone.
- Price Exp: 0.04195
- Reason Price Exp: Price Exp 0.04195: retracement thesis dianggap basi jika harga bergerak terlalu jauh tanpa menyentuh entry zone sebelum menembus boundary relevansi.
- SL: 0.04527
- Reason SL: Prediction invalidation berasal dari H1_BUY_SIDE_LIQUIDITY di 0.045195; SL ditempatkan di luar invalidation dan liquidity guard 0.0451035.
- TP: 0.0408
- Reason TP: TP 0.0408 dipilih dari H4_LOWER_LOW dengan RR 6.21; target quality 79.

### Management

Tidak ada trailing.

### Result

- Result: EXPIRED
- Exit Price: 0.04195
- Result Reason: Price Exp 0.04195: retracement thesis dianggap basi jika harga bergerak terlalu jauh tanpa menyentuh entry zone sebelum menembus boundary relevansi.
- PnL: -
- Created: 2026-10-15T02:36:41.284548+00:00
- Filled: -
- Closed: 2026-10-15T05:54:32.884102+00:00
- Strategy: SMC_VLT_RSI v2.2.0

---

## MMTUSDT — SELL

Trade ID: `MMTUSDT-20261015-124916-0C8BD4`

### Setup

- Price Now Reference: 0.1582
- Entry: 0.1705
- Reason Entry: MMTUSDT SELL: SELL_PULLBACK. H4 structure BEARISH, liquidity H1 0.170425 dipadukan dengan Fib DEEP_0.618_0.786; refinement BEARISH_FVG. Projected M15 RSI 73.7 pada entry zone.
- Price Exp: 0.1568
- Reason Price Exp: Price Exp 0.1568: retracement thesis dianggap basi jika harga bergerak terlalu jauh tanpa menyentuh entry zone sebelum menembus boundary relevansi.
- SL: 0.1762
- Reason SL: Prediction invalidation berasal dari H4_PROTECTED_HIGH di 0.1759; SL ditempatkan di luar invalidation.
- TP: 0.1565
- Reason TP: TP 0.1565 dipilih dari H4_SELL_SIDE_LIQUIDITY dengan RR 2.46; target quality 65.

### Management

Tidak ada trailing.

### Result

- Result: EXPIRED
- Exit Price: 0.1568
- Result Reason: Price Exp 0.1568: retracement thesis dianggap basi jika harga bergerak terlalu jauh tanpa menyentuh entry zone sebelum menembus boundary relevansi.
- PnL: -
- Created: 2026-10-15T05:49:16.582817+00:00
- Filled: -
- Closed: 2026-10-15T05:54:48.505554+00:00
- Strategy: SMC_VLT_RSI v2.2.0

---

## AVAXUSDT — SELL

Trade ID: `AVAXUSDT-20261015-131552-27A41F`

### Setup

- Price Now Reference: 9.545
- Entry: 10.066
- Reason Entry: AVAXUSDT SELL: SELL_PULLBACK. H4 structure BEARISH, liquidity H1 10.065 dipadukan dengan Fib DEEP_0.618_0.786; refinement BULLISH_FVG. Projected M15 RSI 73.3 pada entry zone.
- Price Exp: 9.543
- Reason Price Exp: Price Exp 9.543000000000001: retracement thesis dianggap basi jika harga bergerak terlalu jauh tanpa menyentuh entry zone sebelum menembus boundary relevansi.
- SL: 10.392
- Reason SL: Prediction invalidation berasal dari BEARISH_SWEEP_EXTREME di 10.306; SL ditempatkan di luar invalidation dan liquidity guard 10.36685.
- TP: 9.32
- Reason TP: TP 9.32 dipilih dari H4_SELL_SIDE_LIQUIDITY dengan RR 2.29; target quality 66.

### Management

Tidak ada trailing.

### Result

- Result: EXPIRED
- Exit Price: 9.543
- Result Reason: Price Exp 9.543000000000001: retracement thesis dianggap basi jika harga bergerak terlalu jauh tanpa menyentuh entry zone sebelum menembus boundary relevansi.
- PnL: -
- Created: 2026-10-15T06:15:52.586444+00:00
- Filled: -
- Closed: 2026-10-15T06:15:58.469531+00:00
- Strategy: SMC_VLT_RSI v2.2.0

---

## EVAAUSDT — SELL

Trade ID: `EVAAUSDT-20261015-163613-2EA8EB`

### Setup

- Price Now Reference: 0.5924
- Entry: 0.6188
- Reason Entry: EVAAUSDT SELL: SELL_PULLBACK. H4 structure BEARISH, liquidity H1 0.618575 dipadukan dengan Fib DEEP_0.618_0.786; refinement BEARISH_FVG. Projected M15 RSI 82.1 pada entry zone.
- Price Exp: 0.5882
- Reason Price Exp: Price Exp 0.5882000000000001: retracement thesis dianggap basi jika harga bergerak terlalu jauh tanpa menyentuh entry zone sebelum menembus boundary relevansi.
- SL: 0.6371
- Reason SL: Prediction invalidation berasal dari H1_BUY_SIDE_LIQUIDITY di 0.6358962; SL ditempatkan di luar invalidation dan liquidity guard 0.63545.
- TP: 0.5751
- Reason TP: TP 0.5751 dipilih dari H1_SWING_LOW dengan RR 2.39; target quality 73.

### Management

Tidak ada trailing.

### Result

- Result: EXPIRED
- Exit Price: 0.5882
- Result Reason: Price Exp 0.5882000000000001: retracement thesis dianggap basi jika harga bergerak terlalu jauh tanpa menyentuh entry zone sebelum menembus boundary relevansi.
- PnL: -
- Created: 2026-10-15T09:36:13.763453+00:00
- Filled: -
- Closed: 2026-10-15T10:08:41.669153+00:00
- Strategy: SMC_VLT_RSI v2.2.0
