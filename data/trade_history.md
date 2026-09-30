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
