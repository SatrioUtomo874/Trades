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

---

## RUNEUSDT — BUY

Trade ID: `RUNEUSDT-20260930-203629-3B6426`

### Setup

- Price Now Reference: 0.8014
- Entry: 0.7993
- Reason Entry: RUNEUSDT BUY: thesis dimulai dari POI H4. (BULLISH_FVG 0.70385). H1 me-refine ke BULLISH_FVG 0.69685. M15 memberi execution POI BULLISH_FVG 0.79935. Fib H4 berada pada band SHALLOW (score 45). terdapat liquidity sweep searah reversal. diikuti MSS pada M15. RSI 14 M15 77.4 mendukung momentum. VLT/OHLCV searah dengan setup.
- Price Exp: 0.8055
- Reason Price Exp: Price Exp 0.80549 adalah batas ekspansi thesis H4/H1 sebelum entry. Jika tercapai lebih dulu, setup lama dianggap expired dan market harus membentuk pattern baru.
- SL: 0.7982
- Reason SL: SL ditempatkan di bawah/atas structural invalidation terdekat (0.7993) + buffer 0.15 ATR M15.
- TP: 0.8083
- Reason TP: TP diarahkan ke H4_EQUAL_LEVELS 0.801 sebagai liquidity target HTF.

### Management

Tidak ada trailing.

### Result

- Result: SL
- Exit Price: 0.789
- Result Reason: SL ditempatkan di bawah/atas structural invalidation terdekat (0.7993) + buffer 0.15 ATR M15.
- PnL: -1.288627549105467283873389216%
- Created: 2026-09-30T13:36:29.670012+00:00
- Filled: 2026-09-30T13:39:38.876158+00:00
- Closed: 2026-09-30T13:40:27.651006+00:00
- Strategy: SMC_VLT_RSI v0.6.0

---

## BCHUSDT — SELL

Trade ID: `BCHUSDT-20260930-202215-0D5028`

### Setup

- Price Now Reference: 315.1
- Entry: 315.9
- Reason Entry: Model M15_SMC_FALLBACK: area BEARISH_OB di 315.85. Struktur pasangan mendukung bearish.
- Price Exp: 313.2
- Reason Price Exp: Price Exp 313.255 menjadi batas ekspansi sebelum entry. Jika harga mencapai batas ini lebih dulu, setup lama dianggap expired dan pola baru perlu dicari.
- SL: 317.4
- Reason SL: SL di atas high BEARISH_OB (317.1) + buffer 0.15 ATR sebagai batas invalidasi struktur.
- TP: 311
- Reason TP: TP diarahkan ke M15_SWING_LOW 311.0 (2.52 ATR dari current) sebagai target liquidity.

### Management

Tidak ada trailing.

### Result

- Result: TP
- Exit Price: 307.98
- Result Reason: TP diarahkan ke M15_SWING_LOW 311.0 (2.52 ATR dari current) sebagai target liquidity.
- PnL: 2.507122507122507122507122507%
- Created: 2026-09-30T13:22:15.204815+00:00
- Filled: 2026-09-30T13:26:44.605883+00:00
- Closed: 2026-09-30T13:40:55.718504+00:00
- Strategy: SMC_VLT_RSI v0.6.0

---

## NOMUSDT — BUY

Trade ID: `NOMUSDT-20260930-204252-144C61`

### Setup

- Price Now Reference: 0.002386
- Entry: 0.002321
- Reason Entry: Model M15_SMC_FALLBACK: area BULLISH_FVG di 0.002321. Ada sell-side liquidity sweep sebelum perubahan struktur. M15 membentuk MSS bullish dengan displacement. Struktur pasangan mendukung bullish. RSI 14 M15 berada di 68.2, mendukung momentum bullish. Volume/price pressure OHLCV mendukung bullish.
- Price Exp: 0.002414
- Reason Price Exp: Price Exp 0.0024139556 menjadi batas ekspansi sebelum entry. Jika harga mencapai batas ini lebih dulu, setup lama dianggap expired dan pola baru perlu dicari.
- SL: 0.002287
- Reason SL: SL di bawah low BULLISH_FVG (0.002295) + buffer 0.15 ATR sebagai batas invalidasi struktur.
- TP: 0.002449
- Reason TP: TP diarahkan ke M15_RECENT_RANGE_HIGH 0.00244 (1.31 ATR dari current) sebagai target liquidity.

### Management

Tidak ada trailing.

### Result

- Result: EXPIRED
- Exit Price: 0.002438
- Result Reason: Price Exp 0.0024139556 menjadi batas ekspansi sebelum entry. Jika harga mencapai batas ini lebih dulu, setup lama dianggap expired dan pola baru perlu dicari.
- PnL: -
- Created: 2026-09-30T13:42:52.458335+00:00
- Filled: -
- Closed: 2026-09-30T13:42:52.898814+00:00
- Strategy: SMC_VLT_RSI v0.6.0

---

## LDOUSDT — SELL

Trade ID: `LDOUSDT-20260930-204302-DC68E6`

### Setup

- Price Now Reference: 0.4711
- Entry: 0.4715
- Reason Entry: Model M15_SMC_FALLBACK: area BEARISH_FVG di 0.4715. Struktur pasangan mendukung bearish.
- Price Exp: 0.468
- Reason Price Exp: Price Exp 0.4680605 menjadi batas ekspansi sebelum entry. Jika harga mencapai batas ini lebih dulu, setup lama dianggap expired dan pola baru perlu dicari.
- SL: 0.473
- Reason SL: SL di atas high BEARISH_FVG (0.4721) + buffer 0.15 ATR sebagai batas invalidasi struktur.
- TP: 0.4643
- Reason TP: TP diarahkan ke H4_SWING 0.4668 (0.80 ATR dari current) sebagai target liquidity.

### Management

Tidak ada trailing.

### Result

- Result: SL
- Exit Price: 0.473
- Result Reason: SL di atas high BEARISH_FVG (0.4721) + buffer 0.15 ATR sebagai batas invalidasi struktur.
- PnL: -0.3181336161187698833510074231%
- Created: 2026-09-30T13:43:02.705955+00:00
- Filled: 2026-09-30T13:43:12.245390+00:00
- Closed: 2026-09-30T13:43:19.863046+00:00
- Strategy: SMC_VLT_RSI v0.6.0

---

## PAXGUSDT — SELL

Trade ID: `PAXGUSDT-20260930-202214-73AA43`

### Setup

- Price Now Reference: 4199.4
- Entry: 4201.5
- Reason Entry: Model M15_SMC_FALLBACK: area BEARISH_FVG di 4201.5. Ada buy-side liquidity sweep sebelum perubahan struktur. M15 membentuk MSS bearish dengan displacement. Struktur pasangan mendukung bearish.
- Price Exp: 4195
- Reason Price Exp: Price Exp 4195.0 menjadi batas ekspansi sebelum entry. Jika harga mencapai batas ini lebih dulu, setup lama dianggap expired dan pola baru perlu dicari.
- SL: 4204.5
- Reason SL: SL di atas high BEARISH_FVG (4203.3) + buffer 0.15 ATR sebagai batas invalidasi struktur.
- TP: 4189.6
- Reason TP: TP diarahkan ke M15_SWING_LOW 4190.9 (2.21 ATR dari current) sebagai target liquidity.

### Management

Tidak ada trailing.

### Result

- Result: TP
- Exit Price: 4189.59
- Result Reason: TP diarahkan ke M15_SWING_LOW 4190.9 (2.21 ATR dari current) sebagai target liquidity.
- PnL: 0.2834701892181363798643341664%
- Created: 2026-09-30T13:22:14.195473+00:00
- Filled: 2026-09-30T13:25:21.146677+00:00
- Closed: 2026-09-30T13:43:57.002389+00:00
- Strategy: SMC_VLT_RSI v0.6.0

---

## STRKUSDT — BUY

Trade ID: `STRKUSDT-20260930-204258-1064C6`

### Setup

- Price Now Reference: 0.04233
- Entry: 0.04223
- Reason Entry: STRKUSDT BUY: thesis dimulai dari POI H4. (BULLISH_FVG 0.03877). H1 me-refine ke BULLISH_BREAKER 0.03954. M15 memberi execution POI BULLISH_BREAKER 0.04223. Fib H4 berada pada band OVERLAPS_0.618 (score 92). terdapat liquidity sweep searah reversal. diikuti MSS pada M15. RSI 14 M15 63.1 mendukung momentum.
- Price Exp: 0.04259
- Reason Price Exp: Price Exp 0.0425849 adalah batas ekspansi thesis H4/H1 sebelum entry. Jika tercapai lebih dulu, setup lama dianggap expired dan market harus membentuk pattern baru.
- SL: 0.04201
- Reason SL: SL ditempatkan di bawah/atas structural invalidation terdekat (0.04208) + buffer 0.15 ATR M15.
- TP: 0.04276
- Reason TP: TP diarahkan ke H4_SWING_HIGH 0.04225 sebagai liquidity target HTF.

### Management

Tidak ada trailing.

### Result

- Result: EXPIRED
- Exit Price: 0.04259
- Result Reason: Price Exp 0.0425849 adalah batas ekspansi thesis H4/H1 sebelum entry. Jika tercapai lebih dulu, setup lama dianggap expired dan market harus membentuk pattern baru.
- PnL: -
- Created: 2026-09-30T13:42:58.672056+00:00
- Filled: -
- Closed: 2026-09-30T13:47:06.345054+00:00
- Strategy: SMC_VLT_RSI v0.6.0

---

## FLOWUSDT — SELL

Trade ID: `FLOWUSDT-20260930-204836-34F0B1`

### Setup

- Price Now Reference: 0.03256
- Entry: 0.03269
- Reason Entry: FLOWUSDT SELL: thesis dimulai dari POI H4. (BEARISH_FVG 0.03307). H1 me-refine ke BEARISH_FVG 0.032725. M15 memberi execution POI BEARISH_FVG 0.03269. Fib H4 berada pada band OVERLAPS_0.618 (score 96). terdapat liquidity sweep searah reversal. diikuti MSS pada M15.
- Price Exp: 0.03237
- Reason Price Exp: Price Exp 0.0323706 adalah batas ekspansi thesis H4/H1 sebelum entry. Jika tercapai lebih dulu, setup lama dianggap expired dan market harus membentuk pattern baru.
- SL: 0.0328
- Reason SL: SL ditempatkan di bawah/atas structural invalidation terdekat (0.03274) + buffer 0.15 ATR M15.
- TP: 0.03224
- Reason TP: TP diarahkan ke H4_SWING_LOW 0.03253 sebagai liquidity target HTF.

### Management

Tidak ada trailing.

### Result

- Result: EXPIRED
- Exit Price: 0.03237
- Result Reason: Price Exp 0.0323706 adalah batas ekspansi thesis H4/H1 sebelum entry. Jika tercapai lebih dulu, setup lama dianggap expired dan market harus membentuk pattern baru.
- PnL: -
- Created: 2026-09-30T13:48:36.159070+00:00
- Filled: -
- Closed: 2026-09-30T13:48:43.905158+00:00
- Strategy: SMC_VLT_RSI v0.6.0

---

## UAIUSDT — SELL

Trade ID: `UAIUSDT-20260930-204835-C93E1A`

### Setup

- Price Now Reference: 0.2977
- Entry: 0.2979
- Reason Entry: UAIUSDT SELL: thesis dimulai dari POI H4. (BEARISH_FVG 0.36905). H1 me-refine ke BEARISH_FVG 0.3725. M15 memberi execution POI BEARISH_BREAKER 0.29785. Fib H4 berada pada band DEEPER_THAN_0.786 (score 65). terdapat liquidity sweep searah reversal. diikuti MSS pada M15.
- Price Exp: 0.2912
- Reason Price Exp: Price Exp 0.29126 adalah batas ekspansi thesis H4/H1 sebelum entry. Jika tercapai lebih dulu, setup lama dianggap expired dan market harus membentuk pattern baru.
- SL: 0.2995
- Reason SL: SL ditempatkan di bawah/atas structural invalidation terdekat (0.299) + buffer 0.15 ATR M15.
- TP: 0.2747
- Reason TP: TP diarahkan ke H4_RECENT_RANGE_LOW 0.2747 sebagai liquidity target HTF.

### Management

Tidak ada trailing.

### Result

- Result: SL
- Exit Price: 0.2995
- Result Reason: SL ditempatkan di bawah/atas structural invalidation terdekat (0.299) + buffer 0.15 ATR M15.
- PnL: -0.5370929842228935884525008392%
- Created: 2026-09-30T13:48:35.072888+00:00
- Filled: 2026-09-30T13:48:35.834320+00:00
- Closed: 2026-09-30T13:55:14.165137+00:00
- Strategy: SMC_VLT_RSI v0.6.0

---

## BTWUSDT — BUY

Trade ID: `BTWUSDT-20260930-203635-AEC528`

### Setup

- Price Now Reference: 1.20279
- Entry: 1.1851
- Reason Entry: BTWUSDT BUY: thesis dimulai dari POI H4. (BULLISH_FVG 1.1772). H1 me-refine ke BULLISH_FVG 1.17849. M15 memberi execution POI BULLISH_FVG 1.18511. Fib H4 berada pada band SHALLOW (score 45). terdapat liquidity sweep searah reversal. diikuti MSS pada M15. RSI 14 M15 60.3 mendukung momentum.
- Price Exp: 1.2691
- Reason Price Exp: Price Exp 1.26905 adalah batas ekspansi thesis H4/H1 sebelum entry. Jika tercapai lebih dulu, setup lama dianggap expired dan market harus membentuk pattern baru.
- SL: 1.173
- Reason SL: SL ditempatkan di bawah/atas structural invalidation terdekat (1.17949) + buffer 0.15 ATR M15.
- TP: 1.4395
- Reason TP: TP diarahkan ke H4_EQUAL_LEVELS 1.43945 sebagai liquidity target HTF.

### Management

Tidak ada trailing.

### Result

- Result: EXPIRED
- Exit Price: 1.2691
- Result Reason: Price Exp 1.26905 adalah batas ekspansi thesis H4/H1 sebelum entry. Jika tercapai lebih dulu, setup lama dianggap expired dan market harus membentuk pattern baru.
- PnL: -
- Created: 2026-09-30T13:36:35.667328+00:00
- Filled: -
- Closed: 2026-09-30T14:02:23.889542+00:00
- Strategy: SMC_VLT_RSI v0.6.0

---

## MINAUSDT — BUY

Trade ID: `MINAUSDT-20260930-204837-8ED1BA`

### Setup

- Price Now Reference: 0.14671
- Entry: 0.14326
- Reason Entry: MINAUSDT BUY: thesis dimulai dari POI H4. (BULLISH_FVG 0.14387). H1 me-refine ke BULLISH_FVG 0.14298. M15 memberi execution POI BULLISH_FVG 0.14326. Fib H4 berada pada band OVERLAPS_0.618 (score 92). RSI 14 M15 58.7 mendukung momentum. VLT/OHLCV searah dengan setup.
- Price Exp: 0.14677
- Reason Price Exp: Price Exp 0.146764 adalah batas ekspansi thesis H4/H1 sebelum entry. Jika tercapai lebih dulu, setup lama dianggap expired dan market harus membentuk pattern baru.
- SL: 0.14293
- Reason SL: SL ditempatkan di bawah/atas structural invalidation terdekat (0.14317) + buffer 0.15 ATR M15.
- TP: 0.1468
- Reason TP: TP diarahkan ke H4_SWING_HIGH 0.1468 sebagai liquidity target HTF.

### Management

Tidak ada trailing.

### Result

- Result: EXPIRED
- Exit Price: 0.14679
- Result Reason: Price Exp 0.146764 adalah batas ekspansi thesis H4/H1 sebelum entry. Jika tercapai lebih dulu, setup lama dianggap expired dan market harus membentuk pattern baru.
- PnL: -
- Created: 2026-09-30T13:48:37.172053+00:00
- Filled: -
- Closed: 2026-09-30T14:04:18.551282+00:00
- Strategy: SMC_VLT_RSI v0.6.0

---

## BANKUSDT — SELL

Trade ID: `BANKUSDT-20260930-210024-A21514`

### Setup

- Price Now Reference: 0.03198
- Entry: 0.03203
- Reason Entry: Model M15_SMC_FALLBACK: area BEARISH_FVG di 0.032025. Ada buy-side liquidity sweep sebelum perubahan struktur. M15 membentuk MSS bearish dengan displacement.
- Price Exp: 0.03187
- Reason Price Exp: Price Exp 0.0318741 menjadi batas ekspansi sebelum entry. Jika harga mencapai batas ini lebih dulu, setup lama dianggap expired dan pola baru perlu dicari.
- SL: 0.03208
- Reason SL: SL di atas high BEARISH_FVG (0.03205) + buffer 0.15 ATR sebagai batas invalidasi struktur.
- TP: 0.03174
- Reason TP: TP diarahkan ke H1_SWING 0.03183 (0.80 ATR dari current) sebagai target liquidity.

### Management

Tidak ada trailing.

### Result

- Result: SL
- Exit Price: 0.03208
- Result Reason: SL di atas high BEARISH_FVG (0.03205) + buffer 0.15 ATR sebagai batas invalidasi struktur.
- PnL: -0.1561036528254761161411177022%
- Created: 2026-09-30T14:00:24.460282+00:00
- Filled: 2026-09-30T14:03:17.596423+00:00
- Closed: 2026-09-30T14:05:46.512942+00:00
- Strategy: SMC_VLT_RSI v0.6.0

---

## SOONUSDT — BUY

Trade ID: `SOONUSDT-20260930-202219-CD6C43`

### Setup

- Price Now Reference: 0.4694
- Entry: 0.4415
- Reason Entry: Model M15_SMC_FALLBACK: area BULLISH_FVG di 0.44155. Ada sell-side liquidity sweep sebelum perubahan struktur. M15 membentuk MSS bullish dengan displacement. Struktur pasangan mendukung bullish. RSI 14 M15 berada di 62.2, mendukung momentum bullish.
- Price Exp: 0.477
- Reason Price Exp: Price Exp 0.4769211 menjadi batas ekspansi sebelum entry. Jika harga mencapai batas ini lebih dulu, setup lama dianggap expired dan pola baru perlu dicari.
- SL: 0.4384
- Reason SL: SL di bawah low BULLISH_FVG (0.4405) + buffer 0.15 ATR sebagai batas invalidasi struktur.
- TP: 0.4862
- Reason TP: TP diarahkan ke M15_SWING_HIGH 0.4654 (0.16 ATR dari current) sebagai target liquidity.

### Management

Tidak ada trailing.

### Result

- Result: EXPIRED
- Exit Price: 0.477
- Result Reason: Price Exp 0.4769211 menjadi batas ekspansi sebelum entry. Jika harga mencapai batas ini lebih dulu, setup lama dianggap expired dan pola baru perlu dicari.
- PnL: -
- Created: 2026-09-30T13:22:19.245440+00:00
- Filled: -
- Closed: 2026-09-30T14:08:24.853406+00:00
- Strategy: SMC_VLT_RSI v0.6.0
