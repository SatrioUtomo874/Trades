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
