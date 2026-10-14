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
