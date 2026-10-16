# Trading Analysis & AI Reading Report

Generated: 17-10-2026, 04:48 WIB
Session: `20261016-161333-E3DA`

## 1. Scope and Data Source

- Market: Binance USDⓈ-M Futures
- Execution in this export: Simulation + Real (see real_enabled per trade)
- Price trigger: Binance `aggTrade` last price
- Historical trade records: 22
- Event records: 26
- User notes: 0
- Active setups at export time: 5

## 2. Executive Summary

- Total historical setup records: 22
- Historical filled-and-closed trades: 4
- Historical TP: 1
- Historical SL: 3
- Historical Expired: 13
- Historical Deleted: 0
- Historical fill rate: 18.18181818181818181818181818%
- Historical expiration rate: 59.09090909090909090909090909%
- Historical deletion rate: 0%
- Win Rate: 25%
- Historical PnL sum: -0.2141194940247596842325150714%
- Average win: 0.2649708532061579507123320833%
- Average loss: -0.1596967824103058783149490516%
- Average planned RR: 9.645087472721663065152662205
- Average pending duration among completed filled trades: 1j 9m 48s
- Average holding duration among completed filled trades: 2m 53s
- Trailing trade count: 0
- Total trail events: 0

**Win Rate definition:** TP / (TP + SL).
Expired dan Deleted tidak dihitung sebagai win/loss.
Historical metrics above only use records already closed and stored in history.
Active setup data is reported separately and is not counted as historical closed performance.

## 3. Current Active Session

- Active setups: 5
- Pending: 5
- Filled: 0

| Pair | Direction | Status | Entry | SL | TP | Price Exp | Trailing |
|---|---|---|---:|---:|---:|---:|---|
| BLESSUSDT | SELL | PENDING | 0.007825 | 0.007865 | 0.007195 | 0.007384 | NO |
| BSBUSDT | SELL | PENDING | 0.0895 | 0.09172 | 0.08447 | 0.08597 | NO |
| ZECUSDT | SELL | PENDING | 1240.96 | 1254.8 | 1122.89 | 1134.48 | NO |
| SLXUSDT | SELL | PENDING | 0.05211 | 0.05255 | 0.05013 | 0.05072 | NO |
| SXTUSDT | SELL | PENDING | 0.008689 | 0.00871 | 0.008226 | 0.008364 | NO |

## 4. By Pair

| Pair | Setup | Filled | TP | SL | Expired | Deleted | Win Rate | PnL % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1000FLOKIUSDT | 1 | 0 | 0 | 0 | 1 | 0 | 0% | 0 |
| ACEUSDT | 1 | 0 | 0 | 0 | 1 | 0 | 0% | 0 |
| AWEUSDT | 1 | 0 | 0 | 0 | 1 | 0 | 0% | 0 |
| BANANAUSDT | 1 | 0 | 0 | 0 | 0 | 0 | 0% | 0 |
| DEXEUSDT | 1 | 0 | 0 | 0 | 1 | 0 | 0% | 0 |
| DOGEUSDT | 1 | 1 | 0 | 1 | 0 | 0 | 0% | -0.1332364341085271317829457364 |
| FIDAUSDT | 1 | 0 | 0 | 0 | 1 | 0 | 0% | 0 |
| GASUSDT | 1 | 0 | 0 | 0 | 0 | 0 | 0% | 0 |
| HOMEUSDT | 1 | 0 | 0 | 0 | 1 | 0 | 0% | 0 |
| INJUSDT | 1 | 0 | 0 | 0 | 0 | 0 | 0% | 0 |
| KAITOUSDT | 1 | 0 | 0 | 0 | 1 | 0 | 0% | 0 |
| MEMEUSDT | 1 | 0 | 0 | 0 | 1 | 0 | 0% | 0 |
| MITOUSDT | 1 | 1 | 0 | 1 | 0 | 0 | 0% | -0.2121640735501979897929174611 |
| NEARUSDT | 1 | 0 | 0 | 0 | 0 | 0 | 0% | 0 |
| POWERUSDT | 1 | 0 | 0 | 0 | 1 | 0 | 0% | 0 |
| PROVEUSDT | 1 | 1 | 1 | 0 | 0 | 0 | 100% | 0.2649708532061579507123320833 |
| REUSDT | 1 | 0 | 0 | 0 | 1 | 0 | 0% | 0 |
| XANUSDT | 1 | 0 | 0 | 0 | 1 | 0 | 0% | 0 |
| XRPUSDT | 1 | 1 | 0 | 1 | 0 | 0 | 0% | -0.1336898395721925133689839572 |
| XVSUSDT | 1 | 0 | 0 | 0 | 1 | 0 | 0% | 0 |
| ZECUSDT | 1 | 0 | 0 | 0 | 0 | 0 | 0% | 0 |
| ZRXUSDT | 1 | 0 | 0 | 0 | 1 | 0 | 0% | 0 |

## 5. By Direction

| Direction | Setup | Filled | TP | SL | Expired | Deleted | Win Rate | PnL % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BUY | 6 | 1 | 1 | 0 | 0 | 0 | 100% | 0.2649708532061579507123320833 |
| SELL | 16 | 3 | 0 | 3 | 13 | 0 | 0% | -0.4790903472309176349448471547 |

## 6. By Strategy

| Strategy | Version | Setup | Filled | TP | SL | Expired | Win Rate | PnL % |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| MANUAL | 1.0 | 3 | 0 | 0 | 0 | 0 | 0% | 0 |
| SMC_VLT_RSI | 2.3.0 | 19 | 4 | 1 | 3 | 13 | 25% | -0.2141194940247596842325150714 |

## 7. Event Distribution

| Event | Count |
|---|---:|
| EXPIRED | 13 |
| FILLED | 4 |
| INVALID_BTC | 5 |
| SL | 3 |
| TP | 1 |

## 8. Expiration Reasons

| Reason | Count |
|---|---:|
| Price Exp 0.0005172: retracement thesis dianggap basi jika harga bergerak terlalu jauh tanpa menyentuh entry zone sebelum menembus boundary relevansi. | 1 |
| Price Exp 0.00507: retracement thesis dianggap basi jika harga bergerak terlalu jauh tanpa menyentuh entry zone sebelum menembus boundary relevansi. | 1 |
| Price Exp 0.0101: retracement thesis dianggap basi jika harga bergerak terlalu jauh tanpa menyentuh entry zone sebelum menembus boundary relevansi. | 1 |
| Price Exp 0.01921: retracement thesis dianggap basi jika harga bergerak terlalu jauh tanpa menyentuh entry zone sebelum menembus boundary relevansi. | 1 |
| Price Exp 0.02468: retracement thesis dianggap basi jika harga bergerak terlalu jauh tanpa menyentuh entry zone sebelum menembus boundary relevansi. | 1 |
| Price Exp 0.06303: retracement thesis dianggap basi jika harga bergerak terlalu jauh tanpa menyentuh entry zone sebelum menembus boundary relevansi. | 1 |
| Price Exp 0.07774: retracement thesis dianggap basi jika harga bergerak terlalu jauh tanpa menyentuh entry zone sebelum menembus boundary relevansi. | 1 |
| Price Exp 0.11233000000000001: retracement thesis dianggap basi jika harga bergerak terlalu jauh tanpa menyentuh entry zone sebelum menembus boundary relevansi. | 1 |
| Price Exp 0.16936: retracement thesis dianggap basi jika harga bergerak terlalu jauh tanpa menyentuh entry zone sebelum menembus boundary relevansi. | 1 |
| Price Exp 0.2923: retracement thesis dianggap basi jika harga bergerak terlalu jauh tanpa menyentuh entry zone sebelum menembus boundary relevansi. | 1 |
| Price Exp 0.4057: retracement thesis dianggap basi jika harga bergerak terlalu jauh tanpa menyentuh entry zone sebelum menembus boundary relevansi. | 1 |
| Price Exp 1.7510000000000001: retracement thesis dianggap basi jika harga bergerak terlalu jauh tanpa menyentuh entry zone sebelum menembus boundary relevansi. | 1 |
| Price Exp 3.031: retracement thesis dianggap basi jika harga bergerak terlalu jauh tanpa menyentuh entry zone sebelum menembus boundary relevansi. | 1 |

## 9. Data Quality / Interpretation Flags

- Records where Price Exp exactly equals TP: 0
- Expired records whose reason text mentions TP: 0
- These are flags for review, not automatic strategy judgments.

## 10. Trade-by-Trade Detail

### Trade 1: DEXEUSDT-20261016-163205-7639D8

- Pair: DEXEUSDT
- Direction: SELL
- Status: CLOSED
- Result: EXPIRED
- Price Now Reference: 1.753
- Entry: 1.831
- Reason Entry: DEXEUSDT SELL: SELL_PULLBACK. H4 structure BEARISH, liquidity H1 1.8305 dipadukan dengan Fib DEEP_0.618_0.786; refinement BULLISH_FVG. Projected M15 RSI 87.1 pada entry zone.
- Price Exp: 1.751
- Reason Price Exp: Price Exp 1.7510000000000001: retracement thesis dianggap basi jika harga bergerak terlalu jauh tanpa menyentuh entry zone sebelum menembus boundary relevansi.
- SL: 1.874
- Reason SL: Prediction invalidation berasal dari BEARISH_SWEEP_EXTREME di 1.872; SL ditempatkan di luar invalidation.
- TP: 1.718
- Reason TP: TP 1.718 dipilih dari H4_LOWER_LOW dengan RR 2.63; target quality 80.
- Created: 2026-10-16T09:32:05.790379+00:00
- Filled: None
- Closed: 2026-10-16T09:32:09.956041+00:00
- Fill Price: None
- Exit Price: 1.751
- Result Reason: Price Exp 1.7510000000000001: retracement thesis dianggap basi jika harga bergerak terlalu jauh tanpa menyentuh entry zone sebelum menembus boundary relevansi.
- PnL %: None
- Planned RR: 2.627906976744186046511627907
- Pending Seconds: None
- Holding Seconds: None
- Trailing: NO
- Trail History Count: 0
- Strategy: SMC_VLT_RSI
- Strategy Version: 2.3.0

### Trade 2: 1000FLOKIUSDT-20261016-162840-046FD0

- Pair: 1000FLOKIUSDT
- Direction: SELL
- Status: CLOSED
- Result: EXPIRED
- Price Now Reference: 0.02471
- Entry: 0.02491
- Reason Entry: 1000FLOKIUSDT SELL: SELL_PULLBACK. H4 structure BEARISH, liquidity H1 0.024912 dipadukan dengan Fib SECONDARY; refinement BEARISH_FVG. Projected M15 RSI 64.6 pada entry zone.
- Price Exp: 0.02468
- Reason Price Exp: Price Exp 0.02468: retracement thesis dianggap basi jika harga bergerak terlalu jauh tanpa menyentuh entry zone sebelum menembus boundary relevansi.
- SL: 0.02495
- Reason SL: Prediction invalidation berasal dari H1_PROTECTED_HIGH di 0.02492; SL ditempatkan di luar invalidation dan liquidity guard 0.024912.
- TP: 0.02459
- Reason TP: TP 0.02459 dipilih dari H1_SWING_LOW dengan RR 8.00; target quality 74.
- Created: 2026-10-16T09:28:40.889211+00:00
- Filled: None
- Closed: 2026-10-16T09:53:07.672905+00:00
- Fill Price: None
- Exit Price: 0.02468
- Result Reason: Price Exp 0.02468: retracement thesis dianggap basi jika harga bergerak terlalu jauh tanpa menyentuh entry zone sebelum menembus boundary relevansi.
- PnL %: None
- Planned RR: 8
- Pending Seconds: None
- Holding Seconds: None
- Trailing: NO
- Trail History Count: 0
- Strategy: SMC_VLT_RSI
- Strategy Version: 2.3.0

### Trade 3: MEMEUSDT-20261016-165452-CAA06B

- Pair: MEMEUSDT
- Direction: SELL
- Status: CLOSED
- Result: EXPIRED
- Price Now Reference: 0.0005182
- Entry: 0.0005203
- Reason Entry: MEMEUSDT SELL: SELL_PULLBACK. H4 structure BEARISH, liquidity M15 0.000520875 dipadukan dengan Fib DEEP_0.618_0.786; refinement BEARISH_FVG. Projected M15 RSI 60.3 pada entry zone.
- Price Exp: 0.0005172
- Reason Price Exp: Price Exp 0.0005172: retracement thesis dianggap basi jika harga bergerak terlalu jauh tanpa menyentuh entry zone sebelum menembus boundary relevansi.
- SL: 0.0005218
- Reason SL: Prediction invalidation berasal dari H1_PROTECTED_HIGH di 0.0005212; SL ditempatkan di luar invalidation dan liquidity guard 0.000520875.
- TP: 0.0005159
- Reason TP: TP 0.0005159 dipilih dari H4_LOWER_LOW dengan RR 2.93; target quality 97.
- Created: 2026-10-16T09:54:52.186391+00:00
- Filled: None
- Closed: 2026-10-16T09:58:39.754463+00:00
- Fill Price: None
- Exit Price: 0.0005172
- Result Reason: Price Exp 0.0005172: retracement thesis dianggap basi jika harga bergerak terlalu jauh tanpa menyentuh entry zone sebelum menembus boundary relevansi.
- PnL %: None
- Planned RR: 2.933333333333333333333333333
- Pending Seconds: None
- Holding Seconds: None
- Trailing: NO
- Trail History Count: 0
- Strategy: SMC_VLT_RSI
- Strategy Version: 2.3.0

### Trade 4: DOGEUSDT-20261016-162843-8EEB7E

- Pair: DOGEUSDT
- Direction: SELL
- Status: CLOSED
- Result: SL
- Price Now Reference: 0.08229
- Entry: 0.08256
- Reason Entry: DOGEUSDT SELL: SELL_PULLBACK. H4 structure BEARISH, liquidity M15 0.0825682 dipadukan dengan Fib DEEP_0.618_0.786; refinement BEARISH_FVG. Projected M15 RSI 66.2 pada entry zone.
- Price Exp: 0.08146
- Reason Price Exp: Price Exp 0.08146: retracement thesis dianggap basi jika harga bergerak terlalu jauh tanpa menyentuh entry zone sebelum menembus boundary relevansi.
- SL: 0.08267
- Reason SL: Prediction invalidation berasal dari M15_BUY_SIDE_LIQUIDITY di 0.0825891; SL ditempatkan di luar invalidation dan liquidity guard 0.0825682.
- TP: 0.081
- Reason TP: TP 0.081 dipilih dari H4_LOWER_LOW dengan RR 14.18; target quality 91.
- Created: 2026-10-16T09:28:43.769693+00:00
- Filled: 2026-10-16T11:12:53.378277+00:00
- Closed: 2026-10-16T11:13:26.464766+00:00
- Fill Price: 0.08256
- Exit Price: 0.08267
- Result Reason: Prediction invalidation berasal dari M15_BUY_SIDE_LIQUIDITY di 0.0825891; SL ditempatkan di luar invalidation dan liquidity guard 0.0825682.
- PnL %: -0.1332364341085271317829457364
- Planned RR: 14.18181818181818181818181818
- Pending Seconds: 6249.608584
- Holding Seconds: 33.086489
- Trailing: NO
- Trail History Count: 0
- Strategy: SMC_VLT_RSI
- Strategy Version: 2.3.0

### Trade 5: XRPUSDT-20261016-161934-D9931C

- Pair: XRPUSDT
- Direction: SELL
- Status: CLOSED
- Result: SL
- Price Now Reference: 1.3405
- Entry: 1.3464
- Reason Entry: XRPUSDT SELL: SELL_PULLBACK. H4 structure BEARISH, liquidity M15 1.34696 dipadukan dengan Fib SECONDARY; refinement BULLISH_FVG. Projected M15 RSI 66.2 pada entry zone.
- Price Exp: 1.326
- Reason Price Exp: Price Exp 1.326: retracement thesis dianggap basi jika harga bergerak terlalu jauh tanpa menyentuh entry zone sebelum menembus boundary relevansi.
- SL: 1.3482
- Reason SL: Prediction invalidation berasal dari H1_PROTECTED_HIGH di 1.347; SL ditempatkan di luar invalidation dan liquidity guard 1.34696.
- TP: 1.3173
- Reason TP: TP 1.3173 dipilih dari H4_LOWER_LOW dengan RR 16.17; target quality 90.
- Created: 2026-10-16T09:19:34.581675+00:00
- Filled: 2026-10-16T11:13:01.875414+00:00
- Closed: 2026-10-16T11:13:45.354396+00:00
- Fill Price: 1.3464
- Exit Price: 1.3482
- Result Reason: Prediction invalidation berasal dari H1_PROTECTED_HIGH di 1.347; SL ditempatkan di luar invalidation dan liquidity guard 1.34696.
- PnL %: -0.1336898395721925133689839572
- Planned RR: 16.16666666666666666666666667
- Pending Seconds: 6807.293739
- Holding Seconds: 43.478982
- Trailing: NO
- Trail History Count: 0
- Strategy: SMC_VLT_RSI
- Strategy Version: 2.3.0

### Trade 6: XANUSDT-20261016-170135-FE44A4

- Pair: XANUSDT
- Direction: SELL
- Status: CLOSED
- Result: EXPIRED
- Price Now Reference: 0.010181
- Entry: 0.010478
- Reason Entry: XANUSDT SELL: SELL_PULLBACK. H4 structure BEARISH, liquidity M15 0.010482 dipadukan dengan Fib PRIMARY; refinement BEARISH_FVG. Projected M15 RSI 76.2 pada entry zone.
- Price Exp: 0.0101
- Reason Price Exp: Price Exp 0.0101: retracement thesis dianggap basi jika harga bergerak terlalu jauh tanpa menyentuh entry zone sebelum menembus boundary relevansi.
- SL: 0.010498
- Reason SL: Prediction invalidation berasal dari H1_PROTECTED_HIGH di 0.010482; SL ditempatkan di luar invalidation.
- TP: 0.010086
- Reason TP: TP 0.010086 dipilih dari H4_LOWER_LOW dengan RR 19.60; target quality 85.
- Created: 2026-10-16T10:01:35.580854+00:00
- Filled: None
- Closed: 2026-10-16T11:18:43.166488+00:00
- Fill Price: None
- Exit Price: 0.0101
- Result Reason: Price Exp 0.0101: retracement thesis dianggap basi jika harga bergerak terlalu jauh tanpa menyentuh entry zone sebelum menembus boundary relevansi.
- PnL %: None
- Planned RR: 19.6
- Pending Seconds: None
- Holding Seconds: None
- Trailing: NO
- Trail History Count: 0
- Strategy: SMC_VLT_RSI
- Strategy Version: 2.3.0

### Trade 7: FIDAUSDT-20261016-181414-12362F

- Pair: FIDAUSDT
- Direction: SELL
- Status: CLOSED
- Result: EXPIRED
- Price Now Reference: 0.01932
- Entry: 0.02016
- Reason Entry: FIDAUSDT SELL: SELL_PULLBACK. H4 structure BEARISH, liquidity H1 0.02016 dipadukan dengan Fib DEEP_0.618_0.786; refinement BEARISH_FVG. Projected M15 RSI 85.8 pada entry zone.
- Price Exp: 0.01921
- Reason Price Exp: Price Exp 0.01921: retracement thesis dianggap basi jika harga bergerak terlalu jauh tanpa menyentuh entry zone sebelum menembus boundary relevansi.
- SL: 0.02028
- Reason SL: Prediction invalidation berasal dari H1_BUY_SIDE_LIQUIDITY di 0.0202551; SL ditempatkan di luar invalidation dan liquidity guard 0.02025.
- TP: 0.0191
- Reason TP: TP 0.0191 dipilih dari H4_LOWER_LOW dengan RR 8.83; target quality 80.
- Created: 2026-10-16T11:14:14.307062+00:00
- Filled: None
- Closed: 2026-10-16T12:02:26.849116+00:00
- Fill Price: None
- Exit Price: 0.01921
- Result Reason: Price Exp 0.01921: retracement thesis dianggap basi jika harga bergerak terlalu jauh tanpa menyentuh entry zone sebelum menembus boundary relevansi.
- PnL %: None
- Planned RR: 8.833333333333333333333333333
- Pending Seconds: None
- Holding Seconds: None
- Trailing: NO
- Trail History Count: 0
- Strategy: SMC_VLT_RSI
- Strategy Version: 2.3.0

### Trade 8: AWEUSDT-20261016-182018-DAD74B

- Pair: AWEUSDT
- Direction: SELL
- Status: CLOSED
- Result: EXPIRED
- Price Now Reference: 0.06355
- Entry: 0.06397
- Reason Entry: AWEUSDT SELL: SELL_PULLBACK. H4 structure BEARISH, liquidity H1 0.0640354 dipadukan dengan Fib DEEP_0.618_0.786; refinement BULLISH_FVG. Projected M15 RSI 64.0 pada entry zone.
- Price Exp: 0.06303
- Reason Price Exp: Price Exp 0.06303: retracement thesis dianggap basi jika harga bergerak terlalu jauh tanpa menyentuh entry zone sebelum menembus boundary relevansi.
- SL: 0.06418
- Reason SL: Prediction invalidation berasal dari H1_PROTECTED_HIGH di 0.06402; SL ditempatkan di luar invalidation dan liquidity guard 0.0640354.
- TP: 0.06264
- Reason TP: TP 0.06264 dipilih dari H4_LOWER_LOW dengan RR 6.33; target quality 91.
- Created: 2026-10-16T11:20:18.986906+00:00
- Filled: None
- Closed: 2026-10-16T12:57:57.873653+00:00
- Fill Price: None
- Exit Price: 0.06303
- Result Reason: Price Exp 0.06303: retracement thesis dianggap basi jika harga bergerak terlalu jauh tanpa menyentuh entry zone sebelum menembus boundary relevansi.
- PnL %: None
- Planned RR: 6.333333333333333333333333333
- Pending Seconds: None
- Holding Seconds: None
- Trailing: NO
- Trail History Count: 0
- Strategy: SMC_VLT_RSI
- Strategy Version: 2.3.0

### Trade 9: ZRXUSDT-20261016-181411-080C3A

- Pair: ZRXUSDT
- Direction: SELL
- Status: CLOSED
- Result: EXPIRED
- Price Now Reference: 0.11399
- Entry: 0.115
- Reason Entry: ZRXUSDT SELL: SELL_PULLBACK_RSI_DIVERGENCE. H4 structure BEARISH, liquidity M15 0.11495 dipadukan dengan Fib SECONDARY; refinement BULLISH_FVG. Projected M15 RSI 66.1 pada entry zone.
- Price Exp: 0.1123
- Reason Price Exp: Price Exp 0.11233000000000001: retracement thesis dianggap basi jika harga bergerak terlalu jauh tanpa menyentuh entry zone sebelum menembus boundary relevansi.
- SL: 0.1164
- Reason SL: Prediction invalidation berasal dari BEARISH_SWEEP_EXTREME di 0.1162; SL ditempatkan di luar invalidation.
- TP: 0.1111
- Reason TP: TP 0.11115 dipilih dari H4_LOWER_LOW dengan RR 2.80; target quality 87.
- Created: 2026-10-16T11:14:11.278460+00:00
- Filled: None
- Closed: 2026-10-16T14:10:57.973314+00:00
- Fill Price: None
- Exit Price: 0.1123
- Result Reason: Price Exp 0.11233000000000001: retracement thesis dianggap basi jika harga bergerak terlalu jauh tanpa menyentuh entry zone sebelum menembus boundary relevansi.
- PnL %: None
- Planned RR: 2.785714285714285714285714286
- Pending Seconds: None
- Holding Seconds: None
- Trailing: NO
- Trail History Count: 0
- Strategy: SMC_VLT_RSI
- Strategy Version: 2.3.0

### Trade 10: ACEUSDT-20261016-162233-7B3B06

- Pair: ACEUSDT
- Direction: SELL
- Status: CLOSED
- Result: EXPIRED
- Price Now Reference: 0.1737
- Entry: 0.18355
- Reason Entry: ACEUSDT SELL: SELL_PULLBACK. H4 structure BEARISH, liquidity M15 0.18376 dipadukan dengan Fib PRIMARY; refinement BULLISH_BREAKER. Projected M15 RSI 80.2 pada entry zone.
- Price Exp: 0.16936
- Reason Price Exp: Price Exp 0.16936: retracement thesis dianggap basi jika harga bergerak terlalu jauh tanpa menyentuh entry zone sebelum menembus boundary relevansi.
- SL: 0.18729
- Reason SL: Prediction invalidation berasal dari BEARISH_SWEEP_EXTREME di 0.18695; SL ditempatkan di luar invalidation.
- TP: 0.16328
- Reason TP: TP 0.16328 dipilih dari H4_LOWER_LOW dengan RR 5.42; target quality 79.
- Created: 2026-10-16T09:22:33.292933+00:00
- Filled: None
- Closed: 2026-10-16T15:01:56.871753+00:00
- Fill Price: None
- Exit Price: 0.16936
- Result Reason: Price Exp 0.16936: retracement thesis dianggap basi jika harga bergerak terlalu jauh tanpa menyentuh entry zone sebelum menembus boundary relevansi.
- PnL %: None
- Planned RR: 5.419786096256684491978609626
- Pending Seconds: None
- Holding Seconds: None
- Trailing: NO
- Trail History Count: 0
- Strategy: SMC_VLT_RSI
- Strategy Version: 2.3.0

### Trade 11: XVSUSDT-20261016-190811-389D11

- Pair: XVSUSDT
- Direction: SELL
- Status: CLOSED
- Result: EXPIRED
- Price Now Reference: 3.075
- Entry: 3.111
- Reason Entry: XVSUSDT SELL: SELL_PULLBACK. H4 structure BEARISH, liquidity M15 3.1102 dipadukan dengan Fib PRIMARY; refinement BULLISH_FVG. Projected M15 RSI 66.3 pada entry zone.
- Price Exp: 3.031
- Reason Price Exp: Price Exp 3.031: retracement thesis dianggap basi jika harga bergerak terlalu jauh tanpa menyentuh entry zone sebelum menembus boundary relevansi.
- SL: 3.116
- Reason SL: Prediction invalidation berasal dari M15_BUY_SIDE_LIQUIDITY di 3.113; SL ditempatkan di luar invalidation dan liquidity guard 3.1102.
- TP: 2.987
- Reason TP: TP 2.987 dipilih dari H4_LOWER_LOW dengan RR 24.80; target quality 80.
- Created: 2026-10-16T12:08:11.883120+00:00
- Filled: None
- Closed: 2026-10-16T15:21:55.057656+00:00
- Fill Price: None
- Exit Price: 3.03
- Result Reason: Price Exp 3.031: retracement thesis dianggap basi jika harga bergerak terlalu jauh tanpa menyentuh entry zone sebelum menembus boundary relevansi.
- PnL %: None
- Planned RR: 24.8
- Pending Seconds: None
- Holding Seconds: None
- Trailing: NO
- Trail History Count: 0
- Strategy: SMC_VLT_RSI
- Strategy Version: 2.3.0

### Trade 12: KAITOUSDT-20261016-162536-26C798

- Pair: KAITOUSDT
- Direction: SELL
- Status: CLOSED
- Result: EXPIRED
- Price Now Reference: 0.2976
- Entry: 0.302
- Reason Entry: KAITOUSDT SELL: SELL_PULLBACK. H4 structure BEARISH, liquidity M15 0.3022 dipadukan dengan Fib DEEP_0.618_0.786; refinement BEARISH_FVG. Projected M15 RSI 68.7 pada entry zone.
- Price Exp: 0.2923
- Reason Price Exp: Price Exp 0.2923: retracement thesis dianggap basi jika harga bergerak terlalu jauh tanpa menyentuh entry zone sebelum menembus boundary relevansi.
- SL: 0.3115
- Reason SL: Prediction invalidation berasal dari BEARISH_SWEEP_EXTREME di 0.31115; SL ditempatkan di luar invalidation dan liquidity guard 0.3110226.
- TP: 0.2807
- Reason TP: TP 0.2807 dipilih dari H1_SWING_LOW dengan RR 2.24; target quality 67.
- Created: 2026-10-16T09:25:36.491237+00:00
- Filled: None
- Closed: 2026-10-16T15:22:34.473098+00:00
- Fill Price: None
- Exit Price: 0.2923
- Result Reason: Price Exp 0.2923: retracement thesis dianggap basi jika harga bergerak terlalu jauh tanpa menyentuh entry zone sebelum menembus boundary relevansi.
- PnL %: None
- Planned RR: 2.242105263157894736842105263
- Pending Seconds: None
- Holding Seconds: None
- Trailing: NO
- Trail History Count: 0
- Strategy: SMC_VLT_RSI
- Strategy Version: 2.3.0

### Trade 13: GASUSDT-20261016-210318-1689D5

- Pair: GASUSDT
- Direction: BUY
- Status: CLOSED
- Result: INVALID_BTC
- Price Now Reference: 1.3228
- Entry: 1.269
- Reason Entry: GASUSDT BUY: BUY_PULLBACK. H4 structure BULLISH, liquidity M15 1.2694 dipadukan dengan Fib DEEP_0.618_0.786; refinement BULLISH_OB. Projected M15 RSI 11.5 pada entry zone.
- Price Exp: 1.329
- Reason Price Exp: Price Exp 1.3282: retracement thesis dianggap basi jika harga bergerak terlalu jauh tanpa menyentuh entry zone sebelum menembus boundary relevansi.
- SL: 1.249
- Reason SL: Prediction invalidation berasal dari H4_PROTECTED_LOW di 1.2508; SL ditempatkan di luar invalidation dan liquidity guard 1.2508.
- TP: 1.346
- Reason TP: TP 1.3453 dipilih dari H4_HIGHER_HIGH dengan RR 3.83; target quality 77.
- Created: 2026-10-16T14:03:18.388301+00:00
- Filled: None
- Closed: 2026-10-16T15:31:33.177438+00:00
- Fill Price: None
- Exit Price: None
- Result Reason: BTC berbalik: impuls BTC turun 1.56% (4 jam); setup BUY GASUSDT dibatalkan dan dianggap invalid.
- PnL %: None
- Planned RR: 3.85
- Pending Seconds: None
- Holding Seconds: None
- Trailing: NO
- Trail History Count: 0
- Strategy: SMC_VLT_RSI
- Strategy Version: 2.3.0

### Trade 14: ZECUSDT-20261016-210852-649267

- Pair: ZECUSDT
- Direction: BUY
- Status: CLOSED
- Result: INVALID_BTC
- Price Now Reference: 1217.11
- Entry: 1151.32
- Reason Entry: Liquidity Pool
- Price Exp: 1274
- Reason Price Exp: TP
- SL: 1131.62
- Reason SL: Liquidity Pool bawahnya
- TP: 1274.03
- Reason TP: Liquidity Pool, Swing High atas
- Created: 2026-10-16T14:08:52.157459+00:00
- Filled: None
- Closed: 2026-10-16T15:31:48.872369+00:00
- Fill Price: None
- Exit Price: None
- Result Reason: BTC berbalik: impuls BTC turun 1.56% (4 jam); setup BUY ZECUSDT dibatalkan dan dianggap invalid.
- PnL %: None
- Planned RR: 6.228934010152284263959390863
- Pending Seconds: None
- Holding Seconds: None
- Trailing: NO
- Trail History Count: 0
- Strategy: MANUAL
- Strategy Version: 1.0

### Trade 15: NEARUSDT-20261016-211316-1A20E5

- Pair: NEARUSDT
- Direction: BUY
- Status: CLOSED
- Result: INVALID_BTC
- Price Now Reference: 5.223
- Entry: 4.94
- Reason Entry: Skip
- Price Exp: 5.92
- Reason Price Exp: skip
- SL: 4.752
- Reason SL: skip
- TP: 5.925
- Reason TP: TP
- Created: 2026-10-16T14:13:16.022230+00:00
- Filled: None
- Closed: 2026-10-16T15:32:04.055156+00:00
- Fill Price: None
- Exit Price: None
- Result Reason: BTC berbalik: impuls BTC turun 1.56% (4 jam); setup BUY NEARUSDT dibatalkan dan dianggap invalid.
- PnL %: None
- Planned RR: 5.239361702127659574468085106
- Pending Seconds: None
- Holding Seconds: None
- Trailing: NO
- Trail History Count: 0
- Strategy: MANUAL
- Strategy Version: 1.0

### Trade 16: INJUSDT-20261016-212352-546F3F

- Pair: INJUSDT
- Direction: BUY
- Status: CLOSED
- Result: INVALID_BTC
- Price Now Reference: 9.106
- Entry: 8.488
- Reason Entry: skip
- Price Exp: 9.389
- Reason Price Exp: skip
- SL: 8.218
- Reason SL: skip
- TP: 9.39
- Reason TP: skip
- Created: 2026-10-16T14:23:52.814505+00:00
- Filled: None
- Closed: 2026-10-16T15:32:19.252920+00:00
- Fill Price: None
- Exit Price: None
- Result Reason: BTC berbalik: impuls BTC turun 1.56% (4 jam); setup BUY INJUSDT dibatalkan dan dianggap invalid.
- PnL %: None
- Planned RR: 3.340740740740740740740740741
- Pending Seconds: None
- Holding Seconds: None
- Trailing: NO
- Trail History Count: 0
- Strategy: MANUAL
- Strategy Version: 1.0

### Trade 17: BANANAUSDT-20261016-220843-F65465

- Pair: BANANAUSDT
- Direction: BUY
- Status: CLOSED
- Result: INVALID_BTC
- Price Now Reference: 3.606
- Entry: 3.48
- Reason Entry: BANANAUSDT BUY: BUY_PULLBACK. H4 structure BULLISH, liquidity M15 3.47925 dipadukan dengan Fib PRIMARY; refinement BULLISH_FVG. Projected M15 RSI 8.2 pada entry zone.
- Price Exp: 3.758
- Reason Price Exp: Price Exp 3.758: retracement thesis dianggap basi jika harga bergerak terlalu jauh tanpa menyentuh entry zone sebelum menembus boundary relevansi.
- SL: 3.332
- Reason SL: Prediction invalidation berasal dari H4_PROTECTED_LOW di 3.336; SL ditempatkan di luar invalidation dan liquidity guard 3.336.
- TP: 3.917
- Reason TP: TP 3.917 dipilih dari H4_HIGHER_HIGH dengan RR 2.95; target quality 78.
- Created: 2026-10-16T15:08:43.893682+00:00
- Filled: None
- Closed: 2026-10-16T15:32:33.873814+00:00
- Fill Price: None
- Exit Price: None
- Result Reason: BTC berbalik: impuls BTC turun 1.56% (4 jam); setup BUY BANANAUSDT dibatalkan dan dianggap invalid.
- PnL %: None
- Planned RR: 2.952702702702702702702702703
- Pending Seconds: None
- Holding Seconds: None
- Trailing: NO
- Trail History Count: 0
- Strategy: SMC_VLT_RSI
- Strategy Version: 2.3.0

### Trade 18: PROVEUSDT-20261016-213531-1E567F

- Pair: PROVEUSDT
- Direction: BUY
- Status: CLOSED
- Result: TP
- Price Now Reference: 0.1911
- Entry: 0.1887
- Reason Entry: PROVEUSDT BUY: BUY_PULLBACK. H4 structure BULLISH, liquidity H1 0.1886143 dipadukan dengan Fib SECONDARY; refinement BULLISH_BREAKER. Projected M15 RSI 28.3 pada entry zone.
- Price Exp: 0.1919
- Reason Price Exp: Price Exp 0.19190000000000002: retracement thesis dianggap basi jika harga bergerak terlalu jauh tanpa menyentuh entry zone sebelum menembus boundary relevansi.
- SL: 0.1883
- Reason SL: Prediction invalidation berasal dari H1_SELL_SIDE_LIQUIDITY di 0.1884976; SL ditempatkan di luar invalidation dan liquidity guard 0.1886143.
- TP: 0.1932
- Reason TP: TP 0.1932 dipilih dari H4_SWING_HIGH dengan RR 11.25; target quality 83.
- Created: 2026-10-16T14:35:31.583454+00:00
- Filled: 2026-10-16T15:22:54.678858+00:00
- Closed: 2026-10-16T15:32:50.573404+00:00
- Fill Price: 0.18869999999999998
- Exit Price: 0.1892
- Result Reason: BTC berbalik: impuls BTC turun 1.56% (4 jam) | Manual /close REAL. PnL positif → TP.
- PnL %: 0.2649708532061579507123320833
- Planned RR: 11.25
- Pending Seconds: 2843.095404
- Holding Seconds: 595.894546
- Trailing: NO
- Trail History Count: 0
- Strategy: SMC_VLT_RSI
- Strategy Version: 2.3.0

### Trade 19: MITOUSDT-20261016-233157-549E12

- Pair: MITOUSDT
- Direction: SELL
- Status: CLOSED
- Result: SL
- Price Now Reference: 0.01397
- Entry: 0.01414
- Reason Entry: MITOUSDT SELL: SELL_PULLBACK. H4 structure BEARISH, liquidity M15 0.0141308 dipadukan dengan Fib PRIMARY; refinement BEARISH_FVG. Projected M15 RSI 62.9 pada entry zone.
- Price Exp: 0.01369
- Reason Price Exp: Price Exp 0.01369: retracement thesis dianggap basi jika harga bergerak terlalu jauh tanpa menyentuh entry zone sebelum menembus boundary relevansi.
- SL: 0.01417
- Reason SL: Prediction invalidation berasal dari M15_BUY_SIDE_LIQUIDITY di 0.014145; SL ditempatkan di luar invalidation dan liquidity guard 0.0141308.
- TP: 0.01342
- Reason TP: TP 0.01342 dipilih dari H4_LOWER_LOW dengan RR 24.00; target quality 83.
- Created: 2026-10-16T16:31:57.584962+00:00
- Filled: 2026-10-16T16:46:12.447130+00:00
- Closed: 2026-10-16T16:46:35.078627+00:00
- Fill Price: 0.014140000000000002
- Exit Price: 0.01417
- Result Reason: Prediction invalidation berasal dari M15_BUY_SIDE_LIQUIDITY di 0.014145; SL ditempatkan di luar invalidation dan liquidity guard 0.0141308.
- PnL %: -0.2121640735501979897929174611
- Planned RR: 24
- Pending Seconds: 854.862168
- Holding Seconds: 22.631497
- Trailing: NO
- Trail History Count: 0
- Strategy: SMC_VLT_RSI
- Strategy Version: 2.3.0

### Trade 20: HOMEUSDT-20261016-232246-4F60D9

- Pair: HOMEUSDT
- Direction: SELL
- Status: CLOSED
- Result: EXPIRED
- Price Now Reference: 0.005112
- Entry: 0.005181
- Reason Entry: HOMEUSDT SELL: SELL_PULLBACK. H4 structure BEARISH, liquidity M15 0.005181 dipadukan dengan Fib DEEP_0.618_0.786; refinement BEARISH_OB. Projected M15 RSI 57.7 pada entry zone.
- Price Exp: 0.00507
- Reason Price Exp: Price Exp 0.00507: retracement thesis dianggap basi jika harga bergerak terlalu jauh tanpa menyentuh entry zone sebelum menembus boundary relevansi.
- SL: 0.005212
- Reason SL: Prediction invalidation berasal dari BEARISH_SWEEP_EXTREME di 0.005206; SL ditempatkan di luar invalidation.
- TP: 0.005023
- Reason TP: TP 0.005023 dipilih dari H4_LOWER_LOW dengan RR 5.10; target quality 90.
- Created: 2026-10-16T16:22:46.488738+00:00
- Filled: None
- Closed: 2026-10-16T17:04:37.174130+00:00
- Fill Price: None
- Exit Price: 0.00507
- Result Reason: Price Exp 0.00507: retracement thesis dianggap basi jika harga bergerak terlalu jauh tanpa menyentuh entry zone sebelum menembus boundary relevansi.
- PnL %: None
- Planned RR: 5.096774193548387096774193548
- Pending Seconds: None
- Holding Seconds: None
- Trailing: NO
- Trail History Count: 0
- Strategy: SMC_VLT_RSI
- Strategy Version: 2.3.0

### Trade 21: REUSDT-20261016-163521-7C0F6F

- Pair: REUSDT
- Direction: SELL
- Status: CLOSED
- Result: EXPIRED
- Price Now Reference: 0.4148
- Entry: 0.4242
- Reason Entry: REUSDT SELL: SELL_PULLBACK. H4 structure BEARISH, liquidity H1 0.4242 dipadukan dengan Fib PRIMARY; refinement BULLISH_FVG. Projected M15 RSI 78.4 pada entry zone.
- Price Exp: 0.4057
- Reason Price Exp: Price Exp 0.4057: retracement thesis dianggap basi jika harga bergerak terlalu jauh tanpa menyentuh entry zone sebelum menembus boundary relevansi.
- SL: 0.4323
- Reason SL: Prediction invalidation berasal dari BEARISH_SWEEP_EXTREME di 0.4319; SL ditempatkan di luar invalidation.
- TP: 0.3979
- Reason TP: TP 0.3979 dipilih dari H1_SWING_LOW dengan RR 3.25; target quality 66.
- Created: 2026-10-16T09:35:21.478999+00:00
- Filled: None
- Closed: 2026-10-16T17:05:18.658052+00:00
- Fill Price: None
- Exit Price: 0.4057
- Result Reason: Price Exp 0.4057: retracement thesis dianggap basi jika harga bergerak terlalu jauh tanpa menyentuh entry zone sebelum menembus boundary relevansi.
- PnL %: None
- Planned RR: 3.24691358024691358024691358
- Pending Seconds: None
- Holding Seconds: None
- Trailing: NO
- Trail History Count: 0
- Strategy: SMC_VLT_RSI
- Strategy Version: 2.3.0

### Trade 22: POWERUSDT-20261016-232614-754713

- Pair: POWERUSDT
- Direction: SELL
- Status: CLOSED
- Result: EXPIRED
- Price Now Reference: 0.0793
- Entry: 0.08145
- Reason Entry: POWERUSDT SELL: SELL_PULLBACK. H4 structure BEARISH, liquidity H1 0.08148 dipadukan dengan Fib DEEP_0.618_0.786; refinement BEARISH_BREAKER. Projected M15 RSI 73.8 pada entry zone.
- Price Exp: 0.07774
- Reason Price Exp: Price Exp 0.07774: retracement thesis dianggap basi jika harga bergerak terlalu jauh tanpa menyentuh entry zone sebelum menembus boundary relevansi.
- SL: 0.08161
- Reason SL: Prediction invalidation berasal dari H1_PROTECTED_HIGH di 0.08148; SL ditempatkan di luar invalidation dan liquidity guard 0.08148.
- TP: 0.07616
- Reason TP: TP 0.07616 dipilih dari H4_LOWER_LOW dengan RR 33.06; target quality 82.
- Created: 2026-10-16T16:26:14.379057+00:00
- Filled: None
- Closed: 2026-10-16T17:18:20.166297+00:00
- Fill Price: None
- Exit Price: 0.07774
- Result Reason: Price Exp 0.07774: retracement thesis dianggap basi jika harga bergerak terlalu jauh tanpa menyentuh entry zone sebelum menembus boundary relevansi.
- PnL %: None
- Planned RR: 33.0625
- Pending Seconds: None
- Holding Seconds: None
- Trailing: NO
- Trail History Count: 0
- Strategy: SMC_VLT_RSI
- Strategy Version: 2.3.0

## 11. User Notes (/catatan)

Belum ada catatan.

## 12. AI-Readable Conclusions

1. Dari 22 historical setup, 13 berakhir EXPIRED dan 4 berakhir sebagai filled trade yang kemudian closed.
2. Expiration rate historis adalah 59.09090909090909090909090909%; angka ini menggambarkan frekuensi setup berakhir sebelum menjadi trade yang selesai, bukan win/loss rate.
3. Historical filled sample berjumlah 4; TP=1 dan SL=3. Win Rate historis menurut definisi sistem adalah 25%.
4. Historical PnL sum hanya berasal dari record yang memiliki PnL, dengan nilai -0.2141194940247596842325150714%.
5. Active session saat export memiliki 5 setup (5 PENDING, 0 FILLED); setup aktif tidak dimasukkan ke historical closed performance.
6. Total trailing event yang tercatat adalah 0 dan jumlah trade historis yang memiliki trailing adalah 0.
7. Direction historis yang muncul: BUY, SELL.
8. Pair historis yang muncul: 1000FLOKIUSDT, ACEUSDT, AWEUSDT, BANANAUSDT, DEXEUSDT, DOGEUSDT, FIDAUSDT, GASUSDT, HOMEUSDT, INJUSDT, KAITOUSDT, MEMEUSDT, MITOUSDT, NEARUSDT, POWERUSDT, PROVEUSDT, REUSDT, XANUSDT, XRPUSDT, XVSUSDT, ZECUSDT, ZRXUSDT.
9. Reason EXPIRED yang paling sering tercatat adalah 'Price Exp 0.0005172: retracement thesis dianggap basi jika harga bergerak terlalu jauh tanpa menyentuh entry zone sebelum menembus boundary relevansi.' (1 record).
10. Dataset ini belum cukup untuk menyimpulkan efektivitas strategi secara umum; kesimpulan yang valid di sini bersifat deskriptif terhadap data yang tersimpan.

## 13. Guidance for Future AI Analysis

Gunakan bagian berikut sebagai aturan membaca dataset:

- `trades` = historical records yang sudah closed.
- `active_session.trades` = setup yang masih aktif saat export dan belum termasuk historical closed performance.
- `events` = kronologi event yang tercatat; satu trade dapat memiliki beberapa event.
- `notes` = catatan manual pengguna; jangan perlakukan isi catatan sebagai fakta pasar tanpa konteks tambahan.
- Expired dan Deleted tidak dihitung sebagai win/loss pada Win Rate.
- PnL hanya relevan pada trade yang memiliki `pnl_percent`.
- Reason Entry / Price Exp / SL / TP adalah alasan yang ditulis pengguna, bukan label otomatis yang diverifikasi oleh bot.
- Jangan menganggap active FILLED sebagai historical closed trade.
- Saat membandingkan periode atau strategi, gunakan sample size dan pisahkan pending/expired dari TP/SL.

## 14. Raw Data Location

- Full JSON export: `analysis/full_data.json`
- Full Markdown report: `analysis/analysis.md`
- Historical trades source: `data/trades.json`
- Event source: `data/events.jsonl`
- User notes source: `data/notes.json`

---
Generated by main.py analysis exporter.

## Riset & Resiliensi

- Kandidat: {'CLOSED': 4, 'EXPIRED': 14, 'REJECTED_QUALITY': 265, 'PENDING': 5, 'READY': 1, 'WAITING_PRICE': 11}
- Funnel scan terakhir: {'universe': 461, 'usable_data': 0, 'structure_analyzed': 0, 'direction_thesis': 0, 'candidates_evaluated': 0, 'hard_rejects': 0, 'quality_rejects': 0, 'waiting_price': 0, 'waiting_trigger': 0, 'ready': 0, 'validation_pass': 0, 'validation_reject': 0, 'validation_error': 0, 'risk_rejects': 0, 'pending_created': 0, 'structure_mismatch': 0, 'data_unavailable': 0, 'data_stale': 0, 'budget_stopped': 0, 'memory_skipped': 0, 'validation_skipped_topn': 0, 'rate_limited': 0, 'sweep_covered': 443, 'sweep_total': 443, 'sim_margin_blocked': 0, 'errors': {}, 'validation_reasons': {}}
- R-metrik histori: n=4 • E=-1.41R CI95[-2.68,-0.15] • win 25% • PF 0.08 • DD 5.66R  ⚠️ sampel kecil
- Cooldown recheck: {'INSUFFICIENT_HISTORY': 1, 'BELOW_THRESHOLD': 92, 'VALIDATION_REJECT:H4_STRUCTURE_SAME': 1}
