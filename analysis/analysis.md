# Trading Analysis & AI Reading Report

Generated: 03-10-2026, 16:32 WIB
Session: `20261003-114638-76EE`

## 1. Scope and Data Source

- Market: Binance USDⓈ-M Futures
- Execution in this export: Simulation + Real (see real_enabled per trade)
- Price trigger: Binance `aggTrade` last price
- Historical trade records: 25
- Event records: 56
- User notes: 0
- Active setups at export time: 0

## 2. Executive Summary

- Total historical setup records: 25
- Historical filled-and-closed trades: 15
- Historical TP: 3
- Historical SL: 12
- Historical Expired: 8
- Historical Deleted: 0
- Historical fill rate: 60%
- Historical expiration rate: 32%
- Historical deletion rate: 0%
- Win Rate: 20%
- Historical PnL sum: -2.628403170812966644640469966%
- Average win: 6.174576189768093536822651677%
- Average loss: -1.76267764500977060459236875%
- Average planned RR: 3.353827058292443878064368186
- Average pending duration among completed filled trades: 33m 47s
- Average holding duration among completed filled trades: 2j 23m 55s
- Trailing trade count: 0
- Total trail events: 0

**Win Rate definition:** TP / (TP + SL).
Expired dan Deleted tidak dihitung sebagai win/loss.
Historical metrics above only use records already closed and stored in history.
Active setup data is reported separately and is not counted as historical closed performance.

## 3. Current Active Session

- Active setups: 0
- Pending: 0
- Filled: 0

Tidak ada active setup saat export.

## 4. By Pair

| Pair | Setup | Filled | TP | SL | Expired | Deleted | Win Rate | PnL % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ASTRUSDT | 1 | 1 | 0 | 1 | 0 | 0 | 0% | -1.25658694770976895014187272 |
| ATUSDT | 1 | 0 | 0 | 0 | 1 | 0 | 0% | 0 |
| AUSDT | 1 | 1 | 0 | 1 | 0 | 0 | 0% | -1.727726894254787676935886761 |
| BELUSDT | 1 | 0 | 0 | 0 | 0 | 0 | 0% | 0 |
| BIGTIMEUSDT | 1 | 1 | 0 | 1 | 0 | 0 | 0% | -2.791301525478740668614086336 |
| CETUSUSDT | 1 | 1 | 0 | 1 | 0 | 0 | 0% | -1.396848137535816618911174785 |
| CHILLGUYUSDT | 1 | 1 | 1 | 0 | 0 | 0 | 100% | 3.012139605462830274062185543 |
| CLANKERUSDT | 1 | 1 | 0 | 1 | 0 | 0 | 0% | -1.715511079342387419585418156 |
| DASHUSDT | 1 | 1 | 0 | 1 | 0 | 0 | 0% | -2.309271296613068764967499145 |
| DOTUSDT | 1 | 1 | 0 | 1 | 0 | 0 | 0% | -1.680321016552415983949172379 |
| DUSKUSDT | 1 | 0 | 0 | 0 | 1 | 0 | 0% | 0 |
| DYDXUSDT | 2 | 0 | 0 | 0 | 2 | 0 | 0% | 0 |
| FILUSDT | 1 | 1 | 0 | 1 | 0 | 0 | 0% | -2.107819329771743769579342377 |
| IMXUSDT | 1 | 0 | 0 | 0 | 1 | 0 | 0% | 0 |
| KATUSDT | 1 | 1 | 0 | 1 | 0 | 0 | 0% | -1.24693376941946034341782502 |
| MONUSDT | 1 | 0 | 0 | 0 | 1 | 0 | 0% | 0 |
| OGNUSDT | 1 | 1 | 0 | 1 | 0 | 0 | 0% | -1.073170731707317073170731707 |
| RSRUSDT | 1 | 1 | 0 | 1 | 0 | 0 | 0% | -1.250744490768314472900536033 |
| SAFEUSDT | 1 | 1 | 0 | 1 | 0 | 0 | 0% | -2.595896520963425512934879572 |
| SLPUSDT | 1 | 1 | 1 | 0 | 0 | 0 | 100% | 7.349160743989112354453349463 |
| SPKUSDT | 1 | 1 | 1 | 0 | 0 | 0 | 100% | 8.162428219852337981952420016 |
| SUPERUSDT | 1 | 0 | 0 | 0 | 1 | 0 | 0% | 0 |
| VELVETUSDT | 2 | 0 | 0 | 0 | 1 | 0 | 0% | 0 |

## 5. By Direction

| Direction | Setup | Filled | TP | SL | Expired | Deleted | Win Rate | PnL % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BUY | 22 | 13 | 3 | 10 | 8 | 0 | 23.07692307692307692307692308% | 1.788687455571845889906371556 |
| SELL | 3 | 2 | 0 | 2 | 0 | 0 | 0% | -4.417090626384812534546841522 |

## 6. By Strategy

| Strategy | Version | Setup | Filled | TP | SL | Expired | Win Rate | PnL % |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| SMC_VLT_RSI | 0.7.0 | 25 | 15 | 3 | 12 | 8 | 20% | -2.628403170812966644640469966 |

## 7. Event Distribution

| Event | Count |
|---|---:|
| EXPIRED | 8 |
| FILLED | 31 |
| MANUAL_CLOSE_PENDING | 2 |
| SL | 12 |
| TP | 3 |

## 8. Expiration Reasons

| Reason | Count |
|---|---:|
| Price Exp 0.03488 adalah batas ekspansi thesis H4/H1 sebelum entry. Jika tercapai lebih dulu, setup lama dianggap expired dan market harus membentuk pattern baru. | 1 |
| Price Exp 0.0606387 adalah batas ekspansi thesis H4/H1 sebelum entry. Jika tercapai lebih dulu, setup lama dianggap expired dan market harus membentuk pattern baru. | 1 |
| Price Exp 0.089406 adalah batas ekspansi thesis H4/H1 sebelum entry. Jika tercapai lebih dulu, setup lama dianggap expired dan market harus membentuk pattern baru. | 1 |
| Price Exp 0.144034 adalah batas ekspansi thesis H4/H1 sebelum entry. Jika tercapai lebih dulu, setup lama dianggap expired dan market harus membentuk pattern baru. | 1 |
| Price Exp 0.152424 adalah batas ekspansi thesis H4/H1 sebelum entry. Jika tercapai lebih dulu, setup lama dianggap expired dan market harus membentuk pattern baru. | 1 |
| Price Exp 0.156513 adalah batas ekspansi thesis H4/H1 sebelum entry. Jika tercapai lebih dulu, setup lama dianggap expired dan market harus membentuk pattern baru. | 1 |
| Price Exp 0.1919 adalah batas ekspansi thesis H4/H1 sebelum entry. Jika tercapai lebih dulu, setup lama dianggap expired dan market harus membentuk pattern baru. | 1 |
| Price Exp 0.199914 adalah batas ekspansi thesis H4/H1 sebelum entry. Jika tercapai lebih dulu, setup lama dianggap expired dan market harus membentuk pattern baru. | 1 |

## 9. Data Quality / Interpretation Flags

- Records where Price Exp exactly equals TP: 1
- Expired records whose reason text mentions TP: 0
- These are flags for review, not automatic strategy judgments.

## 10. Trade-by-Trade Detail

### Trade 1: DYDXUSDT-20261001-194236-2820BE

- Pair: DYDXUSDT
- Direction: BUY
- Status: CLOSED
- Result: EXPIRED
- Price Now Reference: 0.14051
- Entry: 0.1388
- Reason Entry: DYDXUSDT BUY: thesis dimulai dari POI H4. (BULLISH_FVG 0.11756). H1 me-refine ke BULLISH_BREAKER 0.136395. M15 memberi execution POI BULLISH_FVG 0.13884. Fib H4 berada pada band DEEPER_THAN_0.786 (score 65). terdapat liquidity sweep searah reversal. diikuti MSS pada M15.
- Price Exp: 0.1441
- Reason Price Exp: Price Exp 0.144034 adalah batas ekspansi thesis H4/H1 sebelum entry. Jika tercapai lebih dulu, setup lama dianggap expired dan market harus membentuk pattern baru.
- SL: 0.1325
- Reason SL: SL di luar invalidasi struktur H1/H4 (0.13323) + buffer 0.30 ATR H1; risiko minimum 1.0 ATR H1.
- TP: 0.1515
- Reason TP: TP diarahkan ke H4_EQUAL_LEVELS 0.13828 sebagai liquidity target HTF.
- Created: 2026-10-01T12:42:36.616408+00:00
- Filled: None
- Closed: 2026-10-01T13:00:22.092187+00:00
- Fill Price: None
- Exit Price: 0.1441
- Result Reason: Price Exp 0.144034 adalah batas ekspansi thesis H4/H1 sebelum entry. Jika tercapai lebih dulu, setup lama dianggap expired dan market harus membentuk pattern baru.
- PnL %: None
- Planned RR: 2.015873015873015873015873016
- Pending Seconds: None
- Holding Seconds: None
- Trailing: NO
- Trail History Count: 0
- Strategy: SMC_VLT_RSI
- Strategy Version: 0.7.0

### Trade 2: RSRUSDT-20261001-194226-3F772D

- Pair: RSRUSDT
- Direction: BUY
- Status: CLOSED
- Result: SL
- Price Now Reference: 0.0016805
- Entry: 0.001679
- Reason Entry: RSRUSDT BUY: thesis dimulai dari POI H4. (BULLISH_OB 0.0016756). H1 me-refine ke BULLISH_FVG 0.00167975. M15 memberi execution POI BULLISH_FVG 0.00167775. Fib H4 berada pada band OVERLAPS_0.618 (score 96). terdapat liquidity sweep searah reversal. diikuti MSS pada M15.
- Price Exp: 0.001714
- Reason Price Exp: Price Exp 0.0017134 adalah batas ekspansi thesis H4/H1 sebelum entry. Jika tercapai lebih dulu, setup lama dianggap expired dan market harus membentuk pattern baru.
- SL: 0.001658
- Reason SL: SL di luar invalidasi struktur H1/H4 (0.0016683) + buffer 0.30 ATR H1; risiko minimum 1.0 ATR H1.
- TP: 0.001798
- Reason TP: TP diarahkan ke H4_EQUAL_LEVELS 0.001798 sebagai liquidity target HTF.
- Created: 2026-10-01T12:42:26.336227+00:00
- Filled: 2026-10-01T12:42:28.516437+00:00
- Closed: 2026-10-01T13:05:59.012167+00:00
- Fill Price: 0.001679
- Exit Price: 0.001658
- Result Reason: SL di luar invalidasi struktur H1/H4 (0.0016683) + buffer 0.30 ATR H1; risiko minimum 1.0 ATR H1.
- PnL %: -1.250744490768314472900536033
- Planned RR: 5.666666666666666666666666667
- Pending Seconds: 2.18021
- Holding Seconds: 1410.49573
- Trailing: NO
- Trail History Count: 0
- Strategy: SMC_VLT_RSI
- Strategy Version: 0.7.0

### Trade 3: SAFEUSDT-20261001-194227-A9D439

- Pair: SAFEUSDT
- Direction: BUY
- Status: CLOSED
- Result: SL
- Price Now Reference: 0.1124
- Entry: 0.1121
- Reason Entry: SAFEUSDT BUY: thesis dimulai dari POI H4. (BULLISH_FVG 0.099585). H1 me-refine ke BULLISH_BREAKER 0.11208. M15 memberi execution POI BULLISH_FVG 0.112205. Fib H4 berada pada band DEEPER_THAN_0.786 (score 69). terdapat liquidity sweep searah reversal. diikuti MSS pada M15. RSI 14 M15 50.3 mendukung momentum.
- Price Exp: 0.11514
- Reason Price Exp: Price Exp 0.115136 adalah batas ekspansi thesis H4/H1 sebelum entry. Jika tercapai lebih dulu, setup lama dianggap expired dan market harus membentuk pattern baru.
- SL: 0.1092
- Reason SL: SL di luar invalidasi struktur H1/H4 (0.1115) + buffer 0.30 ATR H1; risiko minimum 1.0 ATR H1.
- TP: 0.11696
- Reason TP: TP diarahkan ke H4_EQUAL_LEVELS 0.11696 sebagai liquidity target HTF.
- Created: 2026-10-01T12:42:27.345195+00:00
- Filled: 2026-10-01T13:03:29.264889+00:00
- Closed: 2026-10-01T13:15:50.307606+00:00
- Fill Price: 0.1121
- Exit Price: 0.10919
- Result Reason: SL di luar invalidasi struktur H1/H4 (0.1115) + buffer 0.30 ATR H1; risiko minimum 1.0 ATR H1.
- PnL %: -2.595896520963425512934879572
- Planned RR: 1.675862068965517241379310345
- Pending Seconds: 1261.919694
- Holding Seconds: 741.042717
- Trailing: NO
- Trail History Count: 0
- Strategy: SMC_VLT_RSI
- Strategy Version: 0.7.0

### Trade 4: CETUSUSDT-20261001-194224-34E8A9

- Pair: CETUSUSDT
- Direction: BUY
- Status: CLOSED
- Result: SL
- Price Now Reference: 0.027975
- Entry: 0.02792
- Reason Entry: CETUSUSDT BUY: thesis dimulai dari POI H4. (BULLISH_BREAKER 0.0273885). H1 me-refine ke BULLISH_FVG 0.0279255. M15 memberi execution POI BULLISH_FVG 0.0279165. Fib H4 berada pada band OVERLAPS_0.618 (score 96). terdapat liquidity sweep searah reversal. diikuti MSS pada M15.
- Price Exp: 0.02848
- Reason Price Exp: Price Exp 0.028476 adalah batas ekspansi thesis H4/H1 sebelum entry. Jika tercapai lebih dulu, setup lama dianggap expired dan market harus membentuk pattern baru.
- SL: 0.02753
- Reason SL: SL di luar invalidasi struktur H1/H4 (0.027651) + buffer 0.30 ATR H1; risiko minimum 1.0 ATR H1.
- TP: 0.02881
- Reason TP: TP diarahkan ke H4_EQUAL_LEVELS 0.02881 sebagai liquidity target HTF.
- Created: 2026-10-01T12:42:24.312334+00:00
- Filled: 2026-10-01T12:42:25.931956+00:00
- Closed: 2026-10-01T13:16:13.892322+00:00
- Fill Price: 0.02792
- Exit Price: 0.02753
- Result Reason: SL di luar invalidasi struktur H1/H4 (0.027651) + buffer 0.30 ATR H1; risiko minimum 1.0 ATR H1.
- PnL %: -1.396848137535816618911174785
- Planned RR: 2.282051282051282051282051282
- Pending Seconds: 1.619622
- Holding Seconds: 2027.960366
- Trailing: NO
- Trail History Count: 0
- Strategy: SMC_VLT_RSI
- Strategy Version: 0.7.0

### Trade 5: OGNUSDT-20261001-194232-D3FC40

- Pair: OGNUSDT
- Direction: BUY
- Status: CLOSED
- Result: SL
- Price Now Reference: 0.020514
- Entry: 0.0205
- Reason Entry: OGNUSDT BUY: thesis dimulai dari POI H4. (BULLISH_FVG 0.0181845). H1 me-refine ke BULLISH_OB 0.020501. M15 memberi execution POI BULLISH_FVG 0.0205105. Fib H4 berada pada band DEEPER_THAN_0.786 (score 69). terdapat liquidity sweep searah reversal. diikuti MSS pada M15.
- Price Exp: 0.02084
- Reason Price Exp: Price Exp 0.0208339 adalah batas ekspansi thesis H4/H1 sebelum entry. Jika tercapai lebih dulu, setup lama dianggap expired dan market harus membentuk pattern baru.
- SL: 0.02028
- Reason SL: SL di luar invalidasi struktur H1/H4 (0.020386) + buffer 0.30 ATR H1; risiko minimum 1.0 ATR H1.
- TP: 0.02118
- Reason TP: TP diarahkan ke H4_EQUAL_LEVELS 0.021179 sebagai liquidity target HTF.
- Created: 2026-10-01T12:42:32.566284+00:00
- Filled: 2026-10-01T12:57:36.403467+00:00
- Closed: 2026-10-01T13:17:18.379277+00:00
- Fill Price: 0.0205
- Exit Price: 0.02028
- Result Reason: SL di luar invalidasi struktur H1/H4 (0.020386) + buffer 0.30 ATR H1; risiko minimum 1.0 ATR H1.
- PnL %: -1.073170731707317073170731707
- Planned RR: 3.090909090909090909090909091
- Pending Seconds: 903.837183
- Holding Seconds: 1181.97581
- Trailing: NO
- Trail History Count: 0
- Strategy: SMC_VLT_RSI
- Strategy Version: 0.7.0

### Trade 6: KATUSDT-20261001-194230-2EDF02

- Pair: KATUSDT
- Direction: BUY
- Status: CLOSED
- Result: SL
- Price Now Reference: 0.004908
- Entry: 0.004892
- Reason Entry: KATUSDT BUY: thesis dimulai dari POI H4. (BULLISH_BREAKER 0.0049025). H1 me-refine ke BULLISH_BREAKER 0.0048915. M15 memberi execution POI BULLISH_BREAKER 0.004892. Fib H4 berada pada band DEEPER_THAN_0.786 (score 69). terdapat liquidity sweep searah reversal. diikuti MSS pada M15.
- Price Exp: 0.005002
- Reason Price Exp: Price Exp 0.0050011 adalah batas ekspansi thesis H4/H1 sebelum entry. Jika tercapai lebih dulu, setup lama dianggap expired dan market harus membentuk pattern baru.
- SL: 0.004831
- Reason SL: SL di luar invalidasi struktur H1/H4 (0.004859) + buffer 0.30 ATR H1; risiko minimum 1.0 ATR H1.
- TP: 0.005241
- Reason TP: TP diarahkan ke H4_EQUAL_LEVELS 0.0052405 sebagai liquidity target HTF.
- Created: 2026-10-01T12:42:30.545694+00:00
- Filled: 2026-10-01T12:53:01.464056+00:00
- Closed: 2026-10-01T14:05:16.154129+00:00
- Fill Price: 0.004892
- Exit Price: 0.004831
- Result Reason: SL di luar invalidasi struktur H1/H4 (0.004859) + buffer 0.30 ATR H1; risiko minimum 1.0 ATR H1.
- PnL %: -1.24693376941946034341782502
- Planned RR: 5.721311475409836065573770492
- Pending Seconds: 630.918362
- Holding Seconds: 4334.690073
- Trailing: NO
- Trail History Count: 0
- Strategy: SMC_VLT_RSI
- Strategy Version: 0.7.0

### Trade 7: ATUSDT-20261001-194223-A2EF37

- Pair: ATUSDT
- Direction: BUY
- Status: CLOSED
- Result: EXPIRED
- Price Now Reference: 0.15504
- Entry: 0.1536
- Reason Entry: ATUSDT BUY: thesis dimulai dari POI H4. (BULLISH_BREAKER 0.153095). H1 me-refine ke BULLISH_BREAKER 0.15351. M15 memberi execution POI BULLISH_BREAKER 0.15374. Fib H4 berada pada band OVERLAPS_0.618 (score 96). terdapat liquidity sweep searah reversal. diikuti MSS pada M15.
- Price Exp: 0.1566
- Reason Price Exp: Price Exp 0.156513 adalah batas ekspansi thesis H4/H1 sebelum entry. Jika tercapai lebih dulu, setup lama dianggap expired dan market harus membentuk pattern baru.
- SL: 0.152
- Reason SL: SL di luar invalidasi struktur H1/H4 (0.15246) + buffer 0.30 ATR H1; risiko minimum 1.0 ATR H1.
- TP: 0.1575
- Reason TP: TP diarahkan ke H4_EQUAL_LEVELS 0.157495 sebagai liquidity target HTF.
- Created: 2026-10-01T12:42:23.301605+00:00
- Filled: None
- Closed: 2026-10-01T14:19:10.690102+00:00
- Fill Price: None
- Exit Price: 0.1566
- Result Reason: Price Exp 0.156513 adalah batas ekspansi thesis H4/H1 sebelum entry. Jika tercapai lebih dulu, setup lama dianggap expired dan market harus membentuk pattern baru.
- PnL %: None
- Planned RR: 2.4375
- Pending Seconds: None
- Holding Seconds: None
- Trailing: NO
- Trail History Count: 0
- Strategy: SMC_VLT_RSI
- Strategy Version: 0.7.0

### Trade 8: SUPERUSDT-20261001-194231-3C7D81

- Pair: SUPERUSDT
- Direction: BUY
- Status: CLOSED
- Result: EXPIRED
- Price Now Reference: 0.19845
- Entry: 0.19274
- Reason Entry: SUPERUSDT BUY: thesis dimulai dari POI H4. (BULLISH_FVG 0.16847). H1 me-refine ke BULLISH_BREAKER 0.19292. M15 memberi execution POI BULLISH_FVG 0.192745. Fib H4 berada pada band OVERLAPS_0.618 (score 92). terdapat liquidity sweep searah reversal. diikuti MSS pada M15.
- Price Exp: 0.19992
- Reason Price Exp: Price Exp 0.199914 adalah batas ekspansi thesis H4/H1 sebelum entry. Jika tercapai lebih dulu, setup lama dianggap expired dan market harus membentuk pattern baru.
- SL: 0.19005
- Reason SL: SL di luar invalidasi struktur H1/H4 (0.19117) + buffer 0.30 ATR H1; risiko minimum 1.0 ATR H1.
- TP: 0.20089
- Reason TP: TP diarahkan ke H4_EQUAL_LEVELS 0.20089 sebagai liquidity target HTF.
- Created: 2026-10-01T12:42:31.554245+00:00
- Filled: None
- Closed: 2026-10-01T14:40:27.835995+00:00
- Fill Price: None
- Exit Price: 0.19993
- Result Reason: Price Exp 0.199914 adalah batas ekspansi thesis H4/H1 sebelum entry. Jika tercapai lebih dulu, setup lama dianggap expired dan market harus membentuk pattern baru.
- PnL %: None
- Planned RR: 3.029739776951672862453531599
- Pending Seconds: None
- Holding Seconds: None
- Trailing: NO
- Trail History Count: 0
- Strategy: SMC_VLT_RSI
- Strategy Version: 0.7.0

### Trade 9: DOTUSDT-20261001-194235-7162D7

- Pair: DOTUSDT
- Direction: BUY
- Status: CLOSED
- Result: SL
- Price Now Reference: 1.202
- Entry: 1.1962
- Reason Entry: DOTUSDT BUY: thesis dimulai dari POI H4. (BULLISH_FVG 1.01945). H1 me-refine ke BULLISH_BREAKER 1.19625. M15 memberi execution POI BULLISH_BREAKER 1.1971. Fib H4 berada pada band DEEPER_THAN_0.786 (score 69). terdapat liquidity sweep searah reversal. diikuti MSS pada M15.
- Price Exp: 1.2321
- Reason Price Exp: Price Exp 1.23208 adalah batas ekspansi thesis H4/H1 sebelum entry. Jika tercapai lebih dulu, setup lama dianggap expired dan market harus membentuk pattern baru.
- SL: 1.1761
- Reason SL: SL di luar invalidasi struktur H1/H4 (1.1878) + buffer 0.30 ATR H1; risiko minimum 1.0 ATR H1.
- TP: 1.2787
- Reason TP: TP diarahkan ke H4_EQUAL_LEVELS 1.27865 sebagai liquidity target HTF.
- Created: 2026-10-01T12:42:35.603783+00:00
- Filled: 2026-10-01T12:44:29.544957+00:00
- Closed: 2026-10-01T14:46:40.109444+00:00
- Fill Price: 1.1962
- Exit Price: 1.1761
- Result Reason: SL di luar invalidasi struktur H1/H4 (1.1878) + buffer 0.30 ATR H1; risiko minimum 1.0 ATR H1.
- PnL %: -1.680321016552415983949172379
- Planned RR: 4.104477611940298507462686567
- Pending Seconds: 113.941174
- Holding Seconds: 7330.564487
- Trailing: NO
- Trail History Count: 0
- Strategy: SMC_VLT_RSI
- Strategy Version: 0.7.0

### Trade 10: CLANKERUSDT-20261001-194233-D1CE80

- Pair: CLANKERUSDT
- Direction: BUY
- Status: CLOSED
- Result: SL
- Price Now Reference: 14.137
- Entry: 13.99
- Reason Entry: CLANKERUSDT BUY: thesis dimulai dari POI H4. (BULLISH_BREAKER 11.637). H1 me-refine ke BULLISH_FVG 13.959. M15 memberi execution POI BULLISH_BREAKER 13.9945. Fib H4 berada pada band DEEP_0.618_0.786 (score 100). terdapat liquidity sweep searah reversal. diikuti MSS pada M15. RSI 14 M15 51.4 mendukung momentum.
- Price Exp: 14.29
- Reason Price Exp: Price Exp 14.284 adalah batas ekspansi thesis H4/H1 sebelum entry. Jika tercapai lebih dulu, setup lama dianggap expired dan market harus membentuk pattern baru.
- SL: 13.75
- Reason SL: SL di luar invalidasi struktur H1/H4 (13.807) + buffer 0.30 ATR H1; risiko minimum 1.0 ATR H1.
- TP: 14.39
- Reason TP: TP diarahkan ke H4_EQUAL_LEVELS 14.382 sebagai liquidity target HTF.
- Created: 2026-10-01T12:42:33.579059+00:00
- Filled: 2026-10-01T13:22:44.698520+00:00
- Closed: 2026-10-01T14:53:27.575631+00:00
- Fill Price: 13.99
- Exit Price: 13.75
- Result Reason: SL di luar invalidasi struktur H1/H4 (13.807) + buffer 0.30 ATR H1; risiko minimum 1.0 ATR H1.
- PnL %: -1.715511079342387419585418156
- Planned RR: 1.666666666666666666666666667
- Pending Seconds: 2411.119461
- Holding Seconds: 5442.877111
- Trailing: NO
- Trail History Count: 0
- Strategy: SMC_VLT_RSI
- Strategy Version: 0.7.0

### Trade 11: AUSDT-20261001-194239-7E7187

- Pair: AUSDT
- Direction: BUY
- Status: CLOSED
- Result: SL
- Price Now Reference: 0.09688
- Entry: 0.09608
- Reason Entry: AUSDT BUY: thesis dimulai dari POI H4. (BULLISH_FVG 0.083125). H1 me-refine ke BULLISH_BREAKER 0.095985. M15 memberi execution POI BULLISH_FVG 0.09624. Fib H4 berada pada band DEEPER_THAN_0.786 (score 69). terdapat liquidity sweep searah reversal. diikuti MSS pada M15.
- Price Exp: 0.0986
- Reason Price Exp: Price Exp 0.0985942 adalah batas ekspansi thesis H4/H1 sebelum entry. Jika tercapai lebih dulu, setup lama dianggap expired dan market harus membentuk pattern baru.
- SL: 0.09442
- Reason SL: SL di luar invalidasi struktur H1/H4 (0.09477) + buffer 0.30 ATR H1; risiko minimum 1.0 ATR H1.
- TP: 0.10041
- Reason TP: TP diarahkan ke H4_EQUAL_LEVELS 0.10041 sebagai liquidity target HTF.
- Created: 2026-10-01T12:42:39.644103+00:00
- Filled: 2026-10-01T13:05:00.443645+00:00
- Closed: 2026-10-01T15:09:46.699281+00:00
- Fill Price: 0.09608
- Exit Price: 0.09442
- Result Reason: SL di luar invalidasi struktur H1/H4 (0.09477) + buffer 0.30 ATR H1; risiko minimum 1.0 ATR H1.
- PnL %: -1.727726894254787676935886761
- Planned RR: 2.608433734939759036144578313
- Pending Seconds: 1340.799542
- Holding Seconds: 7486.255636
- Trailing: NO
- Trail History Count: 0
- Strategy: SMC_VLT_RSI
- Strategy Version: 0.7.0

### Trade 12: VELVETUSDT-20261001-234509-4CF00F

- Pair: VELVETUSDT
- Direction: BUY
- Status: CLOSED
- Result: MANUAL_CLOSE_PENDING
- Price Now Reference: 0.0589
- Entry: 0.0576
- Reason Entry: VELVETUSDT BUY: thesis dimulai dari POI H4. (BULLISH_FVG 0.05735). H1 me-refine ke BULLISH_FVG 0.05765. M15 memberi execution POI BULLISH_BREAKER 0.05775. Fib H4 berada pada band OVERLAPS_0.618 (score 96). terdapat liquidity sweep searah reversal. diikuti MSS pada M15.
- Price Exp: 0.0607
- Reason Price Exp: Price Exp 0.0606387 adalah batas ekspansi thesis H4/H1 sebelum entry. Jika tercapai lebih dulu, setup lama dianggap expired dan market harus membentuk pattern baru.
- SL: 0.0564
- Reason SL: SL di luar invalidasi struktur H1/H4 (0.0574) + buffer 0.30 ATR H1; risiko minimum 1.0 ATR H1. SL dilebarkan ke risiko minimum 1.0 ATR H1.
- TP: 0.0619
- Reason TP: TP diarahkan ke H4_EQUAL_LEVELS 0.0619 sebagai liquidity target HTF.
- Created: 2026-10-01T16:45:09.156425+00:00
- Filled: None
- Closed: 2026-10-01T16:47:40.633985+00:00
- Fill Price: None
- Exit Price: None
- Result Reason: Manual /close pada setup REAL PENDING sebelum entry.
- PnL %: None
- Planned RR: 3.583333333333333333333333333
- Pending Seconds: None
- Holding Seconds: None
- Trailing: NO
- Trail History Count: 0
- Strategy: SMC_VLT_RSI
- Strategy Version: 0.7.0

### Trade 13: BELUSDT-20261001-234511-9A495D

- Pair: BELUSDT
- Direction: SELL
- Status: CLOSED
- Result: MANUAL_CLOSE_PENDING
- Price Now Reference: 0.12762
- Entry: 0.12893
- Reason Entry: BELUSDT SELL: thesis dimulai dari POI H4. (BEARISH_OB 0.133095). H1 me-refine ke BEARISH_BREAKER 0.129155. M15 memberi execution POI BEARISH_OB 0.12893. Fib H4 berada pada band OVERLAPS_0.618 (score 96). terdapat liquidity sweep searah reversal. diikuti MSS pada M15.
- Price Exp: 0.12576
- Reason Price Exp: Price Exp 0.1257601 adalah batas ekspansi thesis H4/H1 sebelum entry. Jika tercapai lebih dulu, setup lama dianggap expired dan market harus membentuk pattern baru.
- SL: 0.13017
- Reason SL: SL di luar invalidasi struktur H1/H4 (0.12959) + buffer 0.30 ATR H1; risiko minimum 1.0 ATR H1.
- TP: 0.12168
- Reason TP: TP diarahkan ke H4_EQUAL_LEVELS 0.12168 sebagai liquidity target HTF.
- Created: 2026-10-01T16:45:11.641488+00:00
- Filled: None
- Closed: 2026-10-01T16:47:59.650393+00:00
- Fill Price: None
- Exit Price: None
- Result Reason: Manual /close pada setup REAL PENDING sebelum entry.
- PnL %: None
- Planned RR: 5.846774193548387096774193548
- Pending Seconds: None
- Holding Seconds: None
- Trailing: NO
- Trail History Count: 0
- Strategy: SMC_VLT_RSI
- Strategy Version: 0.7.0

### Trade 14: CHILLGUYUSDT-20261001-235256-9FC32A

- Pair: CHILLGUYUSDT
- Direction: BUY
- Status: CLOSED
- Result: TP
- Price Now Reference: 0.013243
- Entry: 0.01318
- Reason Entry: CHILLGUYUSDT BUY: thesis dimulai dari POI H4. (BULLISH_BREAKER 0.0129215). H1 me-refine ke BULLISH_FVG 0.01318. M15 memberi execution POI BULLISH_FVG 0.013204. Fib H4 berada pada band OVERLAPS_0.618 (score 92). terdapat liquidity sweep searah reversal. diikuti MSS pada M15.
- Price Exp: 0.013444
- Reason Price Exp: Price Exp 0.0134431 adalah batas ekspansi thesis H4/H1 sebelum entry. Jika tercapai lebih dulu, setup lama dianggap expired dan market harus membentuk pattern baru.
- SL: 0.012936
- Reason SL: SL di luar invalidasi struktur H1/H4 (0.013148) + buffer 0.30 ATR H1; risiko minimum 1.0 ATR H1. SL dilebarkan ke risiko minimum 1.0 ATR H1.
- TP: 0.013577
- Reason TP: TP diarahkan ke H4_EQUAL_LEVELS 0.0135765 sebagai liquidity target HTF.
- Created: 2026-10-01T16:52:56.343409+00:00
- Filled: 2026-10-01T16:55:24.431451+00:00
- Closed: 2026-10-01T17:27:21.021789+00:00
- Fill Price: 0.013179999999999999
- Exit Price: 0.013577
- Result Reason: TP diarahkan ke H4_EQUAL_LEVELS 0.0135765 sebagai liquidity target HTF.
- PnL %: 3.012139605462830274062185543
- Planned RR: 1.627049180327868852459016393
- Pending Seconds: 148.088042
- Holding Seconds: 1916.590338
- Trailing: NO
- Trail History Count: 0
- Strategy: SMC_VLT_RSI
- Strategy Version: 0.7.0

### Trade 15: VELVETUSDT-20261001-235253-31517D

- Pair: VELVETUSDT
- Direction: BUY
- Status: CLOSED
- Result: EXPIRED
- Price Now Reference: 0.0589
- Entry: 0.0576
- Reason Entry: VELVETUSDT BUY: thesis dimulai dari POI H4. (BULLISH_FVG 0.05735). H1 me-refine ke BULLISH_FVG 0.05765. M15 memberi execution POI BULLISH_BREAKER 0.05775. Fib H4 berada pada band OVERLAPS_0.618 (score 96). terdapat liquidity sweep searah reversal. diikuti MSS pada M15.
- Price Exp: 0.0607
- Reason Price Exp: Price Exp 0.0606387 adalah batas ekspansi thesis H4/H1 sebelum entry. Jika tercapai lebih dulu, setup lama dianggap expired dan market harus membentuk pattern baru.
- SL: 0.0564
- Reason SL: SL di luar invalidasi struktur H1/H4 (0.0574) + buffer 0.30 ATR H1; risiko minimum 1.0 ATR H1. SL dilebarkan ke risiko minimum 1.0 ATR H1.
- TP: 0.0619
- Reason TP: TP diarahkan ke H4_EQUAL_LEVELS 0.0619 sebagai liquidity target HTF.
- Created: 2026-10-01T16:52:53.667609+00:00
- Filled: None
- Closed: 2026-10-02T02:58:22.333191+00:00
- Fill Price: None
- Exit Price: 0.0607
- Result Reason: Price Exp 0.0606387 adalah batas ekspansi thesis H4/H1 sebelum entry. Jika tercapai lebih dulu, setup lama dianggap expired dan market harus membentuk pattern baru.
- PnL %: None
- Planned RR: 3.583333333333333333333333333
- Pending Seconds: None
- Holding Seconds: None
- Trailing: NO
- Trail History Count: 0
- Strategy: SMC_VLT_RSI
- Strategy Version: 0.7.0

### Trade 16: DYDXUSDT-20261002-122115-0A0142

- Pair: DYDXUSDT
- Direction: BUY
- Status: CLOSED
- Result: EXPIRED
- Price Now Reference: 0.15006
- Entry: 0.1439
- Reason Entry: DYDXUSDT BUY: thesis dimulai dari POI H4. (BULLISH_OB 0.14208). H1 me-refine ke BULLISH_BREAKER 0.143195. M15 memberi execution POI BULLISH_BREAKER 0.143985. Fib H4 berada pada band OVERLAPS_0.618 (score 92). terdapat liquidity sweep searah reversal. diikuti MSS pada M15. RSI 14 M15 50.1 mendukung momentum.
- Price Exp: 0.1525
- Reason Price Exp: Price Exp 0.152424 adalah batas ekspansi thesis H4/H1 sebelum entry. Jika tercapai lebih dulu, setup lama dianggap expired dan market harus membentuk pattern baru.
- SL: 0.1388
- Reason SL: SL di luar invalidasi struktur H1/H4 (0.1397) + buffer 0.30 ATR H1; risiko minimum 1.0 ATR H1.
- TP: 0.154
- Reason TP: TP diarahkan ke H4_RECENT_RANGE_HIGH 0.154 sebagai liquidity target HTF.
- Created: 2026-10-02T05:21:15.073715+00:00
- Filled: None
- Closed: 2026-10-02T07:42:11.931417+00:00
- Fill Price: None
- Exit Price: 0.1525
- Result Reason: Price Exp 0.152424 adalah batas ekspansi thesis H4/H1 sebelum entry. Jika tercapai lebih dulu, setup lama dianggap expired dan market harus membentuk pattern baru.
- PnL %: None
- Planned RR: 1.980392156862745098039215686
- Pending Seconds: None
- Holding Seconds: None
- Trailing: NO
- Trail History Count: 0
- Strategy: SMC_VLT_RSI
- Strategy Version: 0.7.0

### Trade 17: DUSKUSDT-20261002-122117-CCEE29

- Pair: DUSKUSDT
- Direction: BUY
- Status: CLOSED
- Result: EXPIRED
- Price Now Reference: 0.0894
- Entry: 0.08425
- Reason Entry: DUSKUSDT BUY: thesis dimulai dari POI H4. (BULLISH_BREAKER 0.08405). H1 me-refine ke BULLISH_FVG 0.084255. M15 memberi execution POI BULLISH_BREAKER 0.084345. Fib H4 berada pada band DEEP_0.618_0.786 (score 96). terdapat liquidity sweep searah reversal. diikuti MSS pada M15. RSI 14 M15 60.5 mendukung momentum.
- Price Exp: 0.08941
- Reason Price Exp: Price Exp 0.089406 adalah batas ekspansi thesis H4/H1 sebelum entry. Jika tercapai lebih dulu, setup lama dianggap expired dan market harus membentuk pattern baru.
- SL: 0.08189
- Reason SL: SL di luar invalidasi struktur H1/H4 (0.08242) + buffer 0.30 ATR H1; risiko minimum 1.0 ATR H1.
- TP: 0.08941
- Reason TP: TP diarahkan ke H4_EQUAL_LEVELS 0.08941 sebagai liquidity target HTF.
- Created: 2026-10-02T05:21:17.840360+00:00
- Filled: None
- Closed: 2026-10-02T08:18:05.928167+00:00
- Fill Price: None
- Exit Price: 0.08941
- Result Reason: Price Exp 0.089406 adalah batas ekspansi thesis H4/H1 sebelum entry. Jika tercapai lebih dulu, setup lama dianggap expired dan market harus membentuk pattern baru.
- PnL %: None
- Planned RR: 2.186440677966101694915254237
- Pending Seconds: None
- Holding Seconds: None
- Trailing: NO
- Trail History Count: 0
- Strategy: SMC_VLT_RSI
- Strategy Version: 0.7.0

### Trade 18: SLPUSDT-20261002-003102-35D092

- Pair: SLPUSDT
- Direction: BUY
- Status: CLOSED
- Result: TP
- Price Now Reference: 0.0006696
- Entry: 0.0006613
- Reason Entry: SLPUSDT BUY: thesis dimulai dari POI H4. (BULLISH_FVG 0.0006239). H1 me-refine ke BULLISH_BREAKER 0.00065975. M15 memberi execution POI BULLISH_BREAKER 0.00066135. Fib H4 berada pada band DEEPER_THAN_0.786 (score 69). terdapat liquidity sweep searah reversal. diikuti MSS pada M15. RSI 14 M15 50.5 mendukung momentum.
- Price Exp: 0.0006813
- Reason Price Exp: Price Exp 0.0006812521 adalah batas ekspansi thesis H4/H1 sebelum entry. Jika tercapai lebih dulu, setup lama dianggap expired dan market harus membentuk pattern baru.
- SL: 0.0006497
- Reason SL: SL di luar invalidasi struktur H1/H4 (0.0006521) + buffer 0.30 ATR H1; risiko minimum 1.0 ATR H1.
- TP: 0.0007097
- Reason TP: TP diarahkan ke H4_EQUAL_LEVELS 0.0007097 sebagai liquidity target HTF.
- Created: 2026-10-01T17:31:02.860773+00:00
- Filled: 2026-10-01T19:20:59.936214+00:00
- Closed: 2026-10-02T09:24:48.822515+00:00
- Fill Price: 0.0006613
- Exit Price: 0.0007099
- Result Reason: TP diarahkan ke H4_EQUAL_LEVELS 0.0007097 sebagai liquidity target HTF.
- PnL %: 7.349160743989112354453349463
- Planned RR: 4.172413793103448275862068966
- Pending Seconds: 6597.075441
- Holding Seconds: 50628.886301
- Trailing: NO
- Trail History Count: 0
- Strategy: SMC_VLT_RSI
- Strategy Version: 0.7.0

### Trade 19: SPKUSDT-20261002-152410-D16C67

- Pair: SPKUSDT
- Direction: BUY
- Status: CLOSED
- Result: TP
- Price Now Reference: 0.02442
- Entry: 0.02438
- Reason Entry: SPKUSDT BUY: thesis dimulai dari POI H4. (BULLISH_FVG 0.024235). H1 me-refine ke BULLISH_BREAKER 0.024285. M15 memberi execution POI BULLISH_BREAKER 0.024385. Fib H4 berada pada band OVERLAPS_0.618 (score 92). terdapat liquidity sweep searah reversal. diikuti MSS pada M15. RSI 14 M15 63.0 mendukung momentum. VLT/OHLCV searah dengan setup.
- Price Exp: 0.02498
- Reason Price Exp: Price Exp 0.024975 adalah batas ekspansi thesis H4/H1 sebelum entry. Jika tercapai lebih dulu, setup lama dianggap expired dan market harus membentuk pattern baru.
- SL: 0.024
- Reason SL: SL di luar invalidasi struktur H1/H4 (0.02412) + buffer 0.30 ATR H1; risiko minimum 1.0 ATR H1.
- TP: 0.02637
- Reason TP: TP diarahkan ke H4_EQUAL_LEVELS 0.026365 sebagai liquidity target HTF.
- Created: 2026-10-02T08:24:10.972505+00:00
- Filled: 2026-10-02T08:25:13.429873+00:00
- Closed: 2026-10-02T13:03:09.141811+00:00
- Fill Price: 0.02438
- Exit Price: 0.02637
- Result Reason: TP diarahkan ke H4_EQUAL_LEVELS 0.026365 sebagai liquidity target HTF.
- PnL %: 8.162428219852337981952420016
- Planned RR: 5.236842105263157894736842105
- Pending Seconds: 62.457368
- Holding Seconds: 16675.711938
- Trailing: NO
- Trail History Count: 0
- Strategy: SMC_VLT_RSI
- Strategy Version: 0.7.0

### Trade 20: MONUSDT-20261002-200824-C1484F

- Pair: MONUSDT
- Direction: BUY
- Status: CLOSED
- Result: EXPIRED
- Price Now Reference: 0.03434
- Entry: 0.03313
- Reason Entry: MONUSDT BUY: thesis dimulai dari POI H4. (BULLISH_FVG 0.028655). H1 me-refine ke BULLISH_FVG 0.03309. M15 memberi execution POI BULLISH_FVG 0.033205. Fib H4 berada pada band DEEP_0.618_0.786 (score 100). terdapat liquidity sweep searah reversal. diikuti MSS pada M15. RSI 14 M15 61.4 mendukung momentum.
- Price Exp: 0.03488
- Reason Price Exp: Price Exp 0.03488 adalah batas ekspansi thesis H4/H1 sebelum entry. Jika tercapai lebih dulu, setup lama dianggap expired dan market harus membentuk pattern baru.
- SL: 0.03226
- Reason SL: SL di luar invalidasi struktur H1/H4 (0.03271) + buffer 0.30 ATR H1; risiko minimum 1.0 ATR H1. SL dilebarkan ke risiko minimum 1.0 ATR H1.
- TP: 0.03524
- Reason TP: TP diarahkan ke H4_RECENT_RANGE_HIGH 0.03524 sebagai liquidity target HTF.
- Created: 2026-10-02T13:08:24.555885+00:00
- Filled: None
- Closed: 2026-10-02T13:20:56.831863+00:00
- Fill Price: None
- Exit Price: 0.03488
- Result Reason: Price Exp 0.03488 adalah batas ekspansi thesis H4/H1 sebelum entry. Jika tercapai lebih dulu, setup lama dianggap expired dan market harus membentuk pattern baru.
- PnL %: None
- Planned RR: 2.425287356321839080459770115
- Pending Seconds: None
- Holding Seconds: None
- Trailing: NO
- Trail History Count: 0
- Strategy: SMC_VLT_RSI
- Strategy Version: 0.7.0

### Trade 21: DASHUSDT-20261003-055157-173218

- Pair: DASHUSDT
- Direction: SELL
- Status: CLOSED
- Result: SL
- Price Now Reference: 57.86
- Entry: 58.46
- Reason Entry: DASHUSDT SELL: thesis dimulai dari POI H4. (BEARISH_BREAKER 58.015). H1 me-refine ke BEARISH_BREAKER 58.525. M15 memberi execution POI BEARISH_BREAKER 58.46. Fib H4 berada pada band OVERLAPS_0.618 (score 92). terdapat liquidity sweep searah reversal. diikuti MSS pada M15. RSI 14 M15 49.3 mendukung momentum. VLT/OHLCV searah dengan setup.
- Price Exp: 56.31
- Reason Price Exp: Price Exp 56.31586 adalah batas ekspansi thesis H4/H1 sebelum entry. Jika tercapai lebih dulu, setup lama dianggap expired dan market harus membentuk pattern baru.
- SL: 59.81
- Reason SL: SL di luar invalidasi struktur H1/H4 (59.5) + buffer 0.30 ATR H1; risiko minimum 1.0 ATR H1.
- TP: 52.74
- Reason TP: TP diarahkan ke H4_EQUAL_LEVELS 52.745 sebagai liquidity target HTF.
- Created: 2026-10-02T22:51:57.858971+00:00
- Filled: 2026-10-02T23:35:39.726265+00:00
- Closed: 2026-10-03T01:16:13.135682+00:00
- Fill Price: 58.46
- Exit Price: 59.81
- Result Reason: SL di luar invalidasi struktur H1/H4 (59.5) + buffer 0.30 ATR H1; risiko minimum 1.0 ATR H1.
- PnL %: -2.309271296613068764967499145
- Planned RR: 4.237037037037037037037037037
- Pending Seconds: 2621.867294
- Holding Seconds: 6033.409417
- Trailing: NO
- Trail History Count: 0
- Strategy: SMC_VLT_RSI
- Strategy Version: 0.7.0

### Trade 22: FILUSDT-20261003-055205-538C5D

- Pair: FILUSDT
- Direction: SELL
- Status: CLOSED
- Result: SL
- Price Now Reference: 1.0241
- Entry: 1.0295
- Reason Entry: FILUSDT SELL: thesis dimulai dari POI H4. (BEARISH_FVG 1.0783). H1 me-refine ke BEARISH_BREAKER 1.03025. M15 memberi execution POI BEARISH_FVG 1.0266. Fib H4 berada pada band DEEP_0.618_0.786 (score 100). terdapat liquidity sweep searah reversal. diikuti MSS pada M15.
- Price Exp: 0.9921
- Reason Price Exp: Price Exp 0.992124 adalah batas ekspansi thesis H4/H1 sebelum entry. Jika tercapai lebih dulu, setup lama dianggap expired dan market harus membentuk pattern baru.
- SL: 1.0512
- Reason SL: SL di luar invalidasi struktur H1/H4 (1.046) + buffer 0.30 ATR H1; risiko minimum 1.0 ATR H1.
- TP: 0.9099
- Reason TP: TP diarahkan ke H4_EQUAL_LEVELS 0.9099 sebagai liquidity target HTF.
- Created: 2026-10-02T22:52:05.245426+00:00
- Filled: 2026-10-02T23:21:58.928671+00:00
- Closed: 2026-10-03T02:18:25.846492+00:00
- Fill Price: 1.0294999999999999
- Exit Price: 1.0512
- Result Reason: SL di luar invalidasi struktur H1/H4 (1.046) + buffer 0.30 ATR H1; risiko minimum 1.0 ATR H1.
- PnL %: -2.107819329771743769579342377
- Planned RR: 5.511520737327188940092165899
- Pending Seconds: 1793.683245
- Holding Seconds: 10586.917821
- Trailing: NO
- Trail History Count: 0
- Strategy: SMC_VLT_RSI
- Strategy Version: 0.7.0

### Trade 23: IMXUSDT-20261003-114933-BF716D

- Pair: IMXUSDT
- Direction: BUY
- Status: CLOSED
- Result: EXPIRED
- Price Now Reference: 0.1874
- Entry: 0.1816
- Reason Entry: IMXUSDT BUY: thesis dimulai dari POI H4. (BULLISH_OB 0.17795). H1 me-refine ke BULLISH_BREAKER 0.18045. M15 memberi execution POI BULLISH_FVG 0.18165. Fib H4 berada pada band OVERLAPS_0.618 (score 92). terdapat liquidity sweep searah reversal. diikuti MSS pada M15. RSI 14 M15 66.7 mendukung momentum. VLT/OHLCV searah dengan setup.
- Price Exp: 0.1919
- Reason Price Exp: Price Exp 0.1919 adalah batas ekspansi thesis H4/H1 sebelum entry. Jika tercapai lebih dulu, setup lama dianggap expired dan market harus membentuk pattern baru.
- SL: 0.1762
- Reason SL: SL di luar invalidasi struktur H1/H4 (0.1775) + buffer 0.30 ATR H1; risiko minimum 1.0 ATR H1.
- TP: 0.1949
- Reason TP: TP diarahkan ke H4_RECENT_RANGE_HIGH 0.1949 sebagai liquidity target HTF.
- Created: 2026-10-03T04:49:33.462526+00:00
- Filled: None
- Closed: 2026-10-03T06:17:44.645428+00:00
- Fill Price: None
- Exit Price: 0.1919
- Result Reason: Price Exp 0.1919 adalah batas ekspansi thesis H4/H1 sebelum entry. Jika tercapai lebih dulu, setup lama dianggap expired dan market harus membentuk pattern baru.
- PnL %: None
- Planned RR: 2.462962962962962962962962963
- Pending Seconds: None
- Holding Seconds: None
- Trailing: NO
- Trail History Count: 0
- Strategy: SMC_VLT_RSI
- Strategy Version: 0.7.0

### Trade 24: BIGTIMEUSDT-20261003-131904-E280C6

- Pair: BIGTIMEUSDT
- Direction: BUY
- Status: CLOSED
- Result: SL
- Price Now Reference: 0.00946
- Entry: 0.009243
- Reason Entry: BIGTIMEUSDT BUY: thesis dimulai dari POI H4. (BULLISH_OB 0.009074). H1 me-refine ke BULLISH_BREAKER 0.009267. M15 memberi execution POI BULLISH_BREAKER 0.0092375. Fib H4 berada pada band OVERLAPS_0.618 (score 92). terdapat liquidity sweep searah reversal. diikuti MSS pada M15. RSI 14 M15 53.3 mendukung momentum.
- Price Exp: 0.009847
- Reason Price Exp: Price Exp 0.0098467501 adalah batas ekspansi thesis H4/H1 sebelum entry. Jika tercapai lebih dulu, setup lama dianggap expired dan market harus membentuk pattern baru.
- SL: 0.008985
- Reason SL: SL di luar invalidasi struktur H1/H4 (0.009186) + buffer 0.30 ATR H1; risiko minimum 1.0 ATR H1. SL dilebarkan ke risiko minimum 1.0 ATR H1.
- TP: 0.010254
- Reason TP: TP diarahkan ke H4_RECENT_RANGE_HIGH 0.010254 sebagai liquidity target HTF.
- Created: 2026-10-03T06:19:04.362351+00:00
- Filled: 2026-10-03T07:12:02.531796+00:00
- Closed: 2026-10-03T08:55:56.845199+00:00
- Fill Price: 0.009243
- Exit Price: 0.008985
- Result Reason: SL di luar invalidasi struktur H1/H4 (0.009186) + buffer 0.30 ATR H1; risiko minimum 1.0 ATR H1. SL dilebarkan ke risiko minimum 1.0 ATR H1.
- PnL %: -2.791301525478740668614086336
- Planned RR: 3.918604651162790697674418605
- Pending Seconds: 3178.169445
- Holding Seconds: 6234.313403
- Trailing: NO
- Trail History Count: 0
- Strategy: SMC_VLT_RSI
- Strategy Version: 0.7.0

### Trade 25: ASTRUSDT-20261003-114935-8E06FD

- Pair: ASTRUSDT
- Direction: BUY
- Status: CLOSED
- Result: SL
- Price Now Reference: 0.007439
- Entry: 0.007401
- Reason Entry: ASTRUSDT BUY: thesis dimulai dari POI H4. (BULLISH_FVG 0.0074075). H1 me-refine ke BULLISH_FVG 0.007401. M15 memberi execution POI BULLISH_BREAKER 0.0074035. Fib H4 berada pada band DEEP_0.618_0.786 (score 96). terdapat liquidity sweep searah reversal. diikuti MSS pada M15. RSI 14 M15 56.3 mendukung momentum. VLT/OHLCV searah dengan setup.
- Price Exp: 0.007571
- Reason Price Exp: Price Exp 0.007571 adalah batas ekspansi thesis H4/H1 sebelum entry. Jika tercapai lebih dulu, setup lama dianggap expired dan market harus membentuk pattern baru.
- SL: 0.007308
- Reason SL: SL di luar invalidasi struktur H1/H4 (0.007375) + buffer 0.30 ATR H1; risiko minimum 1.0 ATR H1. SL dilebarkan ke risiko minimum 1.0 ATR H1.
- TP: 0.007659
- Reason TP: TP diarahkan ke H4_EQUAL_LEVELS 0.007659 sebagai liquidity target HTF.
- Created: 2026-10-03T04:49:35.936548+00:00
- Filled: 2026-10-03T07:25:16.835245+00:00
- Closed: 2026-10-03T09:30:21.325993+00:00
- Fill Price: 0.007401
- Exit Price: 0.007308
- Result Reason: SL di luar invalidasi struktur H1/H4 (0.007375) + buffer 0.30 ATR H1; risiko minimum 1.0 ATR H1. SL dilebarkan ke risiko minimum 1.0 ATR H1.
- PnL %: -1.25658694770976895014187272
- Planned RR: 2.774193548387096774193548387
- Pending Seconds: 9340.898697
- Holding Seconds: 7504.490748
- Trailing: NO
- Trail History Count: 0
- Strategy: SMC_VLT_RSI
- Strategy Version: 0.7.0

## 11. User Notes (/catatan)

Belum ada catatan.

## 12. AI-Readable Conclusions

1. Dari 25 historical setup, 8 berakhir EXPIRED dan 15 berakhir sebagai filled trade yang kemudian closed.
2. Expiration rate historis adalah 32%; angka ini menggambarkan frekuensi setup berakhir sebelum menjadi trade yang selesai, bukan win/loss rate.
3. Historical filled sample berjumlah 15; TP=3 dan SL=12. Win Rate historis menurut definisi sistem adalah 20%.
4. Historical PnL sum hanya berasal dari record yang memiliki PnL, dengan nilai -2.628403170812966644640469966%.
5. Active session saat export memiliki 0 setup (0 PENDING, 0 FILLED); setup aktif tidak dimasukkan ke historical closed performance.
6. Total trailing event yang tercatat adalah 0 dan jumlah trade historis yang memiliki trailing adalah 0.
7. Direction historis yang muncul: BUY, SELL.
8. Pair historis yang muncul: ASTRUSDT, ATUSDT, AUSDT, BELUSDT, BIGTIMEUSDT, CETUSUSDT, CHILLGUYUSDT, CLANKERUSDT, DASHUSDT, DOTUSDT, DUSKUSDT, DYDXUSDT, FILUSDT, IMXUSDT, KATUSDT, MONUSDT, OGNUSDT, RSRUSDT, SAFEUSDT, SLPUSDT, SPKUSDT, SUPERUSDT, VELVETUSDT.
9. Reason EXPIRED yang paling sering tercatat adalah 'Price Exp 0.03488 adalah batas ekspansi thesis H4/H1 sebelum entry. Jika tercapai lebih dulu, setup lama dianggap expired dan market harus membentuk pattern baru.' (1 record).
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
