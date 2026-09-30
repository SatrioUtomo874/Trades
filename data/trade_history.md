# Trading Journal

---

## AVAXUSDT — BUY

Trade ID: `AVAXUSDT-20260930-202212-B0B7CE`

### Setup

- Price Now Reference: 11.275
- Entry: 11.268
- Reason Entry: Model M15_SMC_FALLBACK: area BULLISH_BREAKER di 11.2685. Ada sell-side liquidity sweep sebelum perubahan struktur. M15 membentuk MSS bullish dengan displacement. Struktur pasangan mendukung bullish. RSI 14 M15 berada di 59.8, mendukung momentum bullish.
- Price Exp: 11.321
- Reason Price Exp: Price Exp 11.32071 menjadi batas ekspansi sebelum entry. Jika harga mencapai batas ini lebih dulu, setup lama dianggap expired dan pola baru perlu dicari.
- SL: 11.21
- Reason SL: SL di bawah low BULLISH_BREAKER (11.223) + buffer 0.15 ATR sebagai batas invalidasi struktur.
- TP: 11.377
- Reason TP: TP diarahkan ke M15_EQUAL_LEVELS 11.3465 (0.87 ATR dari current) sebagai target liquidity.

### Management

Tidak ada trailing.

### Result

- Result: SL
- Exit Price: 11.21
- Result Reason: SL di bawah low BULLISH_BREAKER (11.223) + buffer 0.15 ATR sebagai batas invalidasi struktur.
- PnL: -0.5147319843805466808661696841%
- Created: 2026-09-30T13:22:12.176530+00:00
- Filled: 2026-09-30T13:22:13.726120+00:00
- Closed: 2026-09-30T13:35:08.209631+00:00
- Strategy: SMC_VLT_RSI v0.6.0

---

## JUPUSDT — SELL

Trade ID: `JUPUSDT-20260930-202745-AC28C6`

### Setup

- Price Now Reference: 0.3391
- Entry: 0.3438
- Reason Entry: JUPUSDT SELL: thesis dimulai dari POI H4. (BEARISH_FVG 0.3468). H1 me-refine ke BEARISH_BREAKER 0.34665. M15 memberi execution POI BEARISH_BREAKER 0.34375. Fib H4 berada pada band 0.382_0.500 (score 72). terdapat liquidity sweep searah reversal. diikuti MSS pada M15.
- Price Exp: 0.3347
- Reason Price Exp: Price Exp 0.33478 adalah batas ekspansi thesis H4/H1 sebelum entry. Jika tercapai lebih dulu, setup lama dianggap expired dan market harus membentuk pattern baru.
- SL: 0.3453
- Reason SL: SL ditempatkan di bawah/atas structural invalidation terdekat (0.3448) + buffer 0.15 ATR M15.
- TP: 0.3319
- Reason TP: TP diarahkan ke H4_EQUAL_LEVELS 0.3319 sebagai liquidity target HTF.

### Management

Tidak ada trailing.

### Result

- Result: EXPIRED
- Exit Price: 0.3347
- Result Reason: Price Exp 0.33478 adalah batas ekspansi thesis H4/H1 sebelum entry. Jika tercapai lebih dulu, setup lama dianggap expired dan market harus membentuk pattern baru.
- PnL: -
- Created: 2026-09-30T13:27:45.963509+00:00
- Filled: -
- Closed: 2026-09-30T13:35:29.653344+00:00
- Strategy: SMC_VLT_RSI v0.6.0

---

## HYPEUSDT — SELL

Trade ID: `HYPEUSDT-20260930-202207-27A594`

### Setup

- Price Now Reference: 87.54
- Entry: 88.36
- Reason Entry: HYPEUSDT SELL: thesis dimulai dari POI H4. (BEARISH_OB 88.46). H1 me-refine ke BEARISH_OB 88.235. M15 memberi execution POI BEARISH_BREAKER 88.36. Fib H4 berada pada band OVERLAPS_0.618 (score 96). terdapat liquidity sweep searah reversal. diikuti MSS pada M15.
- Price Exp: 86.55
- Reason Price Exp: Price Exp 86.556 adalah batas ekspansi thesis H4/H1 sebelum entry. Jika tercapai lebih dulu, setup lama dianggap expired dan market harus membentuk pattern baru.
- SL: 88.57
- Reason SL: SL ditempatkan di bawah/atas structural invalidation terdekat (88.5) + buffer 0.15 ATR M15.
- TP: 85.56
- Reason TP: TP diarahkan ke H4_SWING_LOW 85.56 sebagai liquidity target HTF.

### Management

Tidak ada trailing.

### Result

- Result: EXPIRED
- Exit Price: 86.55
- Result Reason: Price Exp 86.556 adalah batas ekspansi thesis H4/H1 sebelum entry. Jika tercapai lebih dulu, setup lama dianggap expired dan market harus membentuk pattern baru.
- PnL: -
- Created: 2026-09-30T13:22:07.116085+00:00
- Filled: -
- Closed: 2026-09-30T13:35:54.856308+00:00
- Strategy: SMC_VLT_RSI v0.6.0

---

## HBARUSDT — BUY

Trade ID: `HBARUSDT-20260930-202218-7CE6A4`

### Setup

- Price Now Reference: 0.10831
- Entry: 0.10775
- Reason Entry: Model M15_SMC_FALLBACK: area BULLISH_FVG di 0.107755. Struktur pasangan mendukung bullish. RSI 14 M15 berada di 62.9, mendukung momentum bullish.
- Price Exp: 0.1096
- Reason Price Exp: Price Exp 0.1095921 menjadi batas ekspansi sebelum entry. Jika harga mencapai batas ini lebih dulu, setup lama dianggap expired dan pola baru perlu dicari.
- SL: 0.10756
- Reason SL: SL di bawah low BULLISH_FVG (0.10772) + buffer 0.15 ATR sebagai batas invalidasi struktur.
- TP: 0.131
- Reason TP: Tidak ada liquidity target yang cukup dekat; TP fallback 0.131 berdasarkan ATR / range.

### Management

Tidak ada trailing.

### Result

- Result: SL
- Exit Price: 0.10756
- Result Reason: SL di bawah low BULLISH_FVG (0.10772) + buffer 0.15 ATR sebagai batas invalidasi struktur.
- PnL: -0.1763341067285382830626450116%
- Created: 2026-09-30T13:22:18.233724+00:00
- Filled: 2026-09-30T13:35:51.257666+00:00
- Closed: 2026-09-30T13:36:18.962515+00:00
- Strategy: SMC_VLT_RSI v0.6.0

---

## LTCUSDT — SELL

Trade ID: `LTCUSDT-20260930-202210-9544AF`

### Setup

- Price Now Reference: 67.98
- Entry: 68.47
- Reason Entry: LTCUSDT SELL: thesis dimulai dari POI H4. (BEARISH_FVG 68.095). H1 me-refine ke BEARISH_FVG 68.29. M15 memberi execution POI BEARISH_OB 68.47. Fib H4 berada pada band OVERLAPS_0.618 (score 92). terdapat liquidity sweep searah reversal. diikuti MSS pada M15.
- Price Exp: 67.39
- Reason Price Exp: Price Exp 67.392 adalah batas ekspansi thesis H4/H1 sebelum entry. Jika tercapai lebih dulu, setup lama dianggap expired dan market harus membentuk pattern baru.
- SL: 68.68
- Reason SL: SL ditempatkan di bawah/atas structural invalidation terdekat (68.62) + buffer 0.15 ATR M15.
- TP: 67
- Reason TP: TP diarahkan ke H4_SWING_LOW 67.0 sebagai liquidity target HTF.

### Management

Tidak ada trailing.

### Result

- Result: EXPIRED
- Exit Price: 67.39
- Result Reason: Price Exp 67.392 adalah batas ekspansi thesis H4/H1 sebelum entry. Jika tercapai lebih dulu, setup lama dianggap expired dan market harus membentuk pattern baru.
- PnL: -
- Created: 2026-09-30T13:22:10.155870+00:00
- Filled: -
- Closed: 2026-09-30T13:36:40.139686+00:00
- Strategy: SMC_VLT_RSI v0.6.0

---

## APTUSDT — SELL

Trade ID: `APTUSDT-20260930-202747-553074`

### Setup

- Price Now Reference: 0.8183
- Entry: 0.8288
- Reason Entry: APTUSDT SELL: thesis dimulai dari POI H4. (BEARISH_FVG 0.82). H1 me-refine ke BEARISH_FVG 0.82295. M15 memberi execution POI BEARISH_BREAKER 0.8288. Fib H4 berada pada band OVERLAPS_0.618 (score 92). terdapat liquidity sweep searah reversal. diikuti MSS pada M15.
- Price Exp: 0.8057
- Reason Price Exp: Price Exp 0.8057 adalah batas ekspansi thesis H4/H1 sebelum entry. Jika tercapai lebih dulu, setup lama dianggap expired dan market harus membentuk pattern baru.
- SL: 0.8316
- Reason SL: SL ditempatkan di bawah/atas structural invalidation terdekat (0.8304) + buffer 0.15 ATR M15.
- TP: 0.7951
- Reason TP: TP diarahkan ke H4_EQUAL_LEVELS 0.79515 sebagai liquidity target HTF.

### Management

Tidak ada trailing.

### Result

- Result: EXPIRED
- Exit Price: 0.8057
- Result Reason: Price Exp 0.8057 adalah batas ekspansi thesis H4/H1 sebelum entry. Jika tercapai lebih dulu, setup lama dianggap expired dan market harus membentuk pattern baru.
- PnL: -
- Created: 2026-09-30T13:27:47.979148+00:00
- Filled: -
- Closed: 2026-09-30T13:37:00.282733+00:00
- Strategy: SMC_VLT_RSI v0.6.0

---

## ZENUSDT — SELL

Trade ID: `ZENUSDT-20260930-203634-63F343`

### Setup

- Price Now Reference: 7.493
- Entry: 7.703
- Reason Entry: ZENUSDT SELL: thesis dimulai dari POI H4. (BEARISH_FVG 7.6735). H1 me-refine ke BEARISH_BREAKER 7.703. M15 memberi execution POI BEARISH_BREAKER 7.703. Fib H4 berada pada band OVERLAPS_0.618 (score 92). terdapat liquidity sweep searah reversal. diikuti MSS pada M15.
- Price Exp: 7.481
- Reason Price Exp: Price Exp 7.4816 adalah batas ekspansi thesis H4/H1 sebelum entry. Jika tercapai lebih dulu, setup lama dianggap expired dan market harus membentuk pattern baru.
- SL: 7.819
- Reason SL: SL ditempatkan di bawah/atas structural invalidation terdekat (7.806) + buffer 0.15 ATR M15.
- TP: 7.474
- Reason TP: TP diarahkan ke H4_EQUAL_LEVELS 7.474 sebagai liquidity target HTF.

### Management

Tidak ada trailing.

### Result

- Result: EXPIRED
- Exit Price: 7.44
- Result Reason: Price Exp 7.4816 adalah batas ekspansi thesis H4/H1 sebelum entry. Jika tercapai lebih dulu, setup lama dianggap expired dan market harus membentuk pattern baru.
- PnL: -
- Created: 2026-09-30T13:36:34.658750+00:00
- Filled: -
- Closed: 2026-09-30T13:37:28.745214+00:00
- Strategy: SMC_VLT_RSI v0.6.0

---

## USELESSUSDT — SELL

Trade ID: `USELESSUSDT-20260930-202213-8A255F`

### Setup

- Price Now Reference: 0.25472
- Entry: 0.27143
- Reason Entry: USELESSUSDT SELL: thesis dimulai dari POI H4. (BEARISH_BREAKER 0.27298). H1 me-refine ke BEARISH_FVG 0.27065. M15 memberi execution POI BEARISH_FVG 0.27143. Fib H4 berada pada band OVERLAPS_0.618 (score 92). terdapat liquidity sweep searah reversal. diikuti MSS pada M15.
- Price Exp: 0.24937
- Reason Price Exp: Price Exp 0.249374 adalah batas ekspansi thesis H4/H1 sebelum entry. Jika tercapai lebih dulu, setup lama dianggap expired dan market harus membentuk pattern baru.
- SL: 0.27322
- Reason SL: SL ditempatkan di bawah/atas structural invalidation terdekat (0.2725) + buffer 0.15 ATR M15.
- TP: 0.2458
- Reason TP: TP diarahkan ke H4_EQUAL_LEVELS 0.24581 sebagai liquidity target HTF.

### Management

Tidak ada trailing.

### Result

- Result: EXPIRED
- Exit Price: 0.24918
- Result Reason: Price Exp 0.249374 adalah batas ekspansi thesis H4/H1 sebelum entry. Jika tercapai lebih dulu, setup lama dianggap expired dan market harus membentuk pattern baru.
- PnL: -
- Created: 2026-09-30T13:22:13.187087+00:00
- Filled: -
- Closed: 2026-09-30T13:37:50.753656+00:00
- Strategy: SMC_VLT_RSI v0.6.0

---

## BRUSDT — SELL

Trade ID: `BRUSDT-20260930-203633-CE5760`

### Setup

- Price Now Reference: 0.75302
- Entry: 0.76571
- Reason Entry: BRUSDT SELL: thesis dimulai dari POI H4. (BEARISH_FVG 0.804255). H1 me-refine ke BEARISH_FVG 0.77457. M15 memberi execution POI BEARISH_OB 0.765705. Fib H4 berada pada band SHALLOW (score 45). terdapat liquidity sweep searah reversal. diikuti MSS pada M15. RSI 14 M15 35.0 mendukung momentum. VLT/OHLCV searah dengan setup.
- Price Exp: 0.7482
- Reason Price Exp: Price Exp 0.748208 adalah batas ekspansi thesis H4/H1 sebelum entry. Jika tercapai lebih dulu, setup lama dianggap expired dan market harus membentuk pattern baru.
- SL: 0.77061
- Reason SL: SL ditempatkan di bawah/atas structural invalidation terdekat (0.76923) + buffer 0.15 ATR M15.
- TP: 0.745
- Reason TP: TP diarahkan ke H4_SWING_LOW 0.745 sebagai liquidity target HTF.

### Management

Tidak ada trailing.

### Result

- Result: EXPIRED
- Exit Price: 0.74487
- Result Reason: Price Exp 0.748208 adalah batas ekspansi thesis H4/H1 sebelum entry. Jika tercapai lebih dulu, setup lama dianggap expired dan market harus membentuk pattern baru.
- PnL: -
- Created: 2026-09-30T13:36:33.650661+00:00
- Filled: -
- Closed: 2026-09-30T13:38:21.348748+00:00
- Strategy: SMC_VLT_RSI v0.6.0

---

## COTIUSDT — SELL

Trade ID: `COTIUSDT-20260930-203632-D4A888`

### Setup

- Price Now Reference: 0.013401
- Entry: 0.014139
- Reason Entry: COTIUSDT SELL: thesis dimulai dari POI H4. (BEARISH_FVG 0.014213). H1 me-refine ke BEARISH_FVG 0.014101. M15 memberi execution POI BEARISH_FVG 0.0141385. Fib H4 berada pada band OVERLAPS_0.618 (score 92). terdapat liquidity sweep searah reversal. diikuti MSS pada M15.
- Price Exp: 0.013332
- Reason Price Exp: Price Exp 0.013332 adalah batas ekspansi thesis H4/H1 sebelum entry. Jika tercapai lebih dulu, setup lama dianggap expired dan market harus membentuk pattern baru.
- SL: 0.014284
- Reason SL: SL ditempatkan di bawah/atas structural invalidation terdekat (0.014267) + buffer 0.15 ATR M15.
- TP: 0.013286
- Reason TP: TP diarahkan ke H1_SWING_LOW 0.013286 sebagai liquidity target HTF.

### Management

Tidak ada trailing.

### Result

- Result: EXPIRED
- Exit Price: 0.013317
- Result Reason: Price Exp 0.013332 adalah batas ekspansi thesis H4/H1 sebelum entry. Jika tercapai lebih dulu, setup lama dianggap expired dan market harus membentuk pattern baru.
- PnL: -
- Created: 2026-09-30T13:36:32.554797+00:00
- Filled: -
- Closed: 2026-09-30T13:38:48.911826+00:00
- Strategy: SMC_VLT_RSI v0.6.0

---

## ALGOUSDT — BUY

Trade ID: `ALGOUSDT-20260930-202746-1C1366`

### Setup

- Price Now Reference: 0.12688
- Entry: 0.12606
- Reason Entry: ALGOUSDT BUY: thesis dimulai dari POI H4. (BULLISH_FVG 0.098285). H1 me-refine ke BULLISH_FVG 0.12369. M15 memberi execution POI BULLISH_FVG 0.126065. Fib H4 berada pada band DEEPER_THAN_0.786 (score 69). terdapat liquidity sweep searah reversal. diikuti MSS pada M15. RSI 14 M15 66.0 mendukung momentum.
- Price Exp: 0.12716
- Reason Price Exp: Price Exp 0.1271559 adalah batas ekspansi thesis H4/H1 sebelum entry. Jika tercapai lebih dulu, setup lama dianggap expired dan market harus membentuk pattern baru.
- SL: 0.12529
- Reason SL: SL ditempatkan di bawah/atas structural invalidation terdekat (0.12545) + buffer 0.15 ATR M15.
- TP: 0.12734
- Reason TP: TP diarahkan ke H1_SWING_HIGH 0.12704 sebagai liquidity target HTF.

### Management

Tidak ada trailing.

### Result

- Result: SL
- Exit Price: 0.12468
- Result Reason: SL ditempatkan di bawah/atas structural invalidation terdekat (0.12545) + buffer 0.15 ATR M15.
- PnL: -1.094716801523084245597334603%
- Created: 2026-09-30T13:27:46.971433+00:00
- Filled: 2026-09-30T13:29:25.499848+00:00
- Closed: 2026-09-30T13:39:14.864506+00:00
- Strategy: SMC_VLT_RSI v0.6.0

---

## QUSDT — SELL

Trade ID: `QUSDT-20260930-203630-2A2D9F`

### Setup

- Price Now Reference: 0.02311
- Entry: 0.02319
- Reason Entry: Model M15_SMC_FALLBACK: area BEARISH_FVG di 0.023185. Ada buy-side liquidity sweep sebelum perubahan struktur. M15 membentuk MSS bearish dengan displacement. RSI 14 M15 berada di 49.5, mendukung momentum bearish.
- Price Exp: 0.02295
- Reason Price Exp: Price Exp 0.022957 menjadi batas ekspansi sebelum entry. Jika harga mencapai batas ini lebih dulu, setup lama dianggap expired dan pola baru perlu dicari.
- SL: 0.02323
- Reason SL: SL di atas high BEARISH_FVG (0.02319) + buffer 0.15 ATR sebagai batas invalidasi struktur.
- TP: 0.02277
- Reason TP: TP diarahkan ke H1_SWING 0.02277 (1.32 ATR dari current) sebagai target liquidity.

### Management

Tidak ada trailing.

### Result

- Result: EXPIRED
- Exit Price: 0.02294
- Result Reason: Price Exp 0.022957 menjadi batas ekspansi sebelum entry. Jika harga mencapai batas ini lebih dulu, setup lama dianggap expired dan pola baru perlu dicari.
- PnL: -
- Created: 2026-09-30T13:36:30.676962+00:00
- Filled: -
- Closed: 2026-09-30T13:39:43.769008+00:00
- Strategy: SMC_VLT_RSI v0.6.0

---

## AZTECUSDT — BUY

Trade ID: `AZTECUSDT-20260930-203628-468630`

### Setup

- Price Now Reference: 0.01741
- Entry: 0.01739
- Reason Entry: AZTECUSDT BUY: thesis dimulai dari POI H4. (BULLISH_FVG 0.015785). H1 me-refine ke BULLISH_BREAKER 0.017005. M15 memberi execution POI BULLISH_BREAKER 0.01739. Fib H4 berada pada band DEEPER_THAN_0.786 (score 69). terdapat liquidity sweep searah reversal. diikuti MSS pada M15. RSI 14 M15 61.5 mendukung momentum. VLT/OHLCV searah dengan setup.
- Price Exp: 0.01759
- Reason Price Exp: Price Exp 0.017584 adalah batas ekspansi thesis H4/H1 sebelum entry. Jika tercapai lebih dulu, setup lama dianggap expired dan market harus membentuk pattern baru.
- SL: 0.01728
- Reason SL: SL ditempatkan di bawah/atas structural invalidation terdekat (0.01731) + buffer 0.15 ATR M15.
- TP: 0.0177
- Reason TP: TP diarahkan ke H4_SWING_HIGH 0.0177 sebagai liquidity target HTF.

### Management

Tidak ada trailing.

### Result

- Result: SL
- Exit Price: 0.01724
- Result Reason: SL ditempatkan di bawah/atas structural invalidation terdekat (0.01731) + buffer 0.15 ATR M15.
- PnL: -0.8625646923519263944795859689%
- Created: 2026-09-30T13:36:28.661886+00:00
- Filled: 2026-09-30T13:38:17.147381+00:00
- Closed: 2026-09-30T13:40:05.647700+00:00
- Strategy: SMC_VLT_RSI v0.6.0
