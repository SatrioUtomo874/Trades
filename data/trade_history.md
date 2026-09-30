# Trading Journal

---

## HYPERUSDT — BUY

Trade ID: `HYPERUSDT-20260922-192342-026541`

### Setup

- Price Now Reference: 0.07387
- Entry: 0.07335
- Reason Entry: FVG Terakir H4 yang paspasan dengan Order Block H4, Liquidity Pool 14% H4, tipis di atas Volumatic Trend konsisten yang terbentuk di M15
- Price Exp: 0.07473
- Reason Price Exp: TP
- SL: 0.07284
- Reason SL: di bawah close candle order block dan di atas Liquidity pool 24% (konfirmasi pembalikan arah sekaligus agar RR logis)
- TP: 0.07473
- Reason TP: swing high H4 dan Liquidity Pool bearish 34%

### Management

Tidak ada trailing.

### Result

- Result: EXPIRED
- Exit Price: 0.07474
- Result Reason: TP
- PnL: -
- Created: 2026-09-22T12:23:42.760106+00:00
- Filled: -
- Closed: 2026-09-22T12:58:06.000106+00:00
- Strategy: MANUAL v1.0

---

## UMAUSDT — BUY

Trade ID: `UMAUSDT-20260922-194400-B89139`

### Setup

- Price Now Reference: 0.4032
- Entry: 0.384
- Reason Entry: RSI H4 Divergent + OverBought (65.19). Liquidity Pool 100%. RSI M15 sedang netral (bisa turun lebih). FVG M15, Discount Zone 0.618 an (naiknya struggle buat nembus resistance H4)
- Price Exp: 0.408
- Reason Price Exp: Terlalu jauh dari entry (telat)
- SL: 0.3787
- Reason SL: tepat di bawah Liquidity Sweep
- TP: 0.4033
- Reason TP: Swing High H4

### Management

Tidak ada trailing.

### Result

- Result: EXPIRED
- Exit Price: 0.408
- Result Reason: Terlalu jauh dari entry (telat)
- PnL: -
- Created: 2026-09-22T12:44:00.468810+00:00
- Filled: -
- Closed: 2026-09-22T13:03:00.212173+00:00
- Strategy: MANUAL v1.0

---

## HYPERUSDT — BUY

Trade ID: `HYPERUSDT-20260922-211028-14AAAF`

### Setup

- Price Now Reference: 0.07462
- Entry: 0.07335
- Reason Entry: FVG Terakir H4 yang paspasan dengan Order Block H4, Liquidity Pool 14% H4, tipis di atas Volumatic Trend konsisten yang terbentuk di M15
- Price Exp: 0.7555
- Reason Price Exp: market membentuk pola baru
- SL: 0.07284
- Reason SL: di bawah close candle order block dan di atas Liquidity pool 24% (konfirmasi pembalikan arah sekaligus agar RR logis)
- TP: 0.07473
- Reason TP: swing high H4 dan Liquidity Pool bearish 34%

### Management

Tidak ada trailing.

### Result

- Result: SL
- Exit Price: 0.07281
- Result Reason: di bawah close candle order block dan di atas Liquidity pool 24% (konfirmasi pembalikan arah sekaligus agar RR logis)
- PnL: -0.7361963190184049079754601227%
- Created: 2026-09-22T14:10:28.713982+00:00
- Filled: 2026-09-23T14:13:11.546243+00:00
- Closed: 2026-09-23T14:13:22.783389+00:00
- Strategy: MANUAL v1.0

---

## SOLUSDT — BUY

Trade ID: `SOLUSDT-20260922-195004-54BB93`

### Setup

- Price Now Reference: 117.29
- Entry: 109.38
- Reason Entry: Entry di Liquidity Sweep (Pool 100%). sedikit di atas Order Block. RSI H4 OverBought (69.01) + Divergent tipis.
- Price Exp: 120
- Reason Price Exp: Telat Entry
- SL: 106.22
- Reason SL: di bawah Volumatik Trend
- TP: 118.86
- Reason TP: Swing High H4

### Management

Tidak ada trailing.

### Result

- Result: EXPIRED
- Exit Price: 120
- Result Reason: Telat Entry
- PnL: -
- Created: 2026-09-22T12:50:04.634431+00:00
- Filled: -
- Closed: 2026-09-25T11:06:32.034813+00:00
- Strategy: MANUAL v1.0

---

## WUSDT — BUY

Trade ID: `WUSDT-20260926-235356-AA6F23`

### Setup

- Price Now Reference: 0.012857
- Entry: 0.012008
- Reason Entry: Order Block. Liquidity Pool Tipis (semua tipis juga). Fibo 0.618. RSI Masih tinggi jadi jauh
- Price Exp: 0.013
- Reason Price Exp: Harga Terlalu Jauh naik
- SL: 0.011718
- Reason SL: di bawah Liquidity Pool Entry
- TP: 0.01288
- Reason TP: Swing High H4

### Real Execution

- Real Enabled: False
- Margin: 0.5 USDT
- Leverage: 10x
- Quantity: None
- Target Notional: None USDT
- Actual Notional: None USDT
- Entry Order ID: None
- TP Algo ID: None
- SL Algo ID: None

### Management

Tidak ada trailing.

### Result

- Result: EXPIRED
- Exit Price: 0.013
- Result Reason: Harga Terlalu Jauh naik
- PnL: -
- Created: 2026-09-26T16:53:56.965752+00:00
- Filled: -
- Closed: 2026-09-26T16:56:35.447646+00:00
- Strategy: MANUAL v1.0

---

## UNIUSDT — BUY

Trade ID: `UNIUSDT-20260927-213358-2441B9`

### Setup

- Price Now Reference: 9.751
- Entry: 9.675
- Reason Entry: Model BREAKER_RETEST: area BULLISH_BREAKER di 9.675. Struktur pasangan mendukung bullish.
- Price Exp: 9.799
- Reason Price Exp: Price Exp 9.7989 menjadi batas ekspansi sebelum entry. Jika harga mencapai batas ini lebih dulu, setup lama dianggap expired dan pola baru perlu dicari.
- SL: 9.635
- Reason SL: SL di bawah low BULLISH_BREAKER (9.648) + buffer 0.15 ATR sebagai batas invalidasi struktur.
- TP: 9.858
- Reason TP: TP diarahkan ke EQUAL_LEVELS 9.7525 (1.19 ATR dari current) sebagai target liquidity.

### Management

Tidak ada trailing.

### Result

- Result: SL
- Exit Price: 9.625
- Result Reason: SL di bawah low BULLISH_BREAKER (9.648) + buffer 0.15 ATR sebagai batas invalidasi struktur.
- PnL: -0.5167958656330749354005167959%
- Created: 2026-09-27T14:33:58.303540+00:00
- Filled: 2026-09-27T14:48:29.716538+00:00
- Closed: 2026-09-27T15:14:47.101775+00:00
- Strategy: SMC_VLT_RSI v0.1.0

---

## UNIUSDT — BUY

Trade ID: `UNIUSDT-20260927-213358-2441B9`

### Setup

- Price Now Reference: 9.751
- Entry: 9.675
- Reason Entry: Model BREAKER_RETEST: area BULLISH_BREAKER di 9.675. Struktur pasangan mendukung bullish.
- Price Exp: 9.799
- Reason Price Exp: Price Exp 9.7989 menjadi batas ekspansi sebelum entry. Jika harga mencapai batas ini lebih dulu, setup lama dianggap expired dan pola baru perlu dicari.
- SL: 9.635
- Reason SL: SL di bawah low BULLISH_BREAKER (9.648) + buffer 0.15 ATR sebagai batas invalidasi struktur.
- TP: 9.858
- Reason TP: TP diarahkan ke EQUAL_LEVELS 9.7525 (1.19 ATR dari current) sebagai target liquidity.

### Management

Tidak ada trailing.

### Result

- Result: SL
- Exit Price: 9.635
- Result Reason: SL di bawah low BULLISH_BREAKER (9.648) + buffer 0.15 ATR sebagai batas invalidasi struktur.
- PnL: -0.4134366925064599483204134367%
- Created: 2026-09-27T14:33:58.303540+00:00
- Filled: 2026-09-27T14:48:29.716538+00:00
- Closed: 2026-09-27T15:17:52.358020+00:00
- Strategy: SMC_VLT_RSI v0.1.0

---

## FILUSDT — BUY

Trade ID: `FILUSDT-20260927-222825-FAD7A9`

### Setup

- Price Now Reference: 1.1269
- Entry: 1.1265
- Reason Entry: FILUSDT BUY: thesis dimulai dari POI H4. (BULLISH_FVG 0.9006). H1 me-refine ke BULLISH_OB 1.1259. M15 memberi execution POI BULLISH_FVG 1.12655. Fib H4 berada pada band DEEP_0.618_0.786 (score 96).
- Price Exp: 1.1327
- Reason Price Exp: Price Exp 1.13263 adalah batas ekspansi thesis H4/H1 sebelum entry. Jika tercapai lebih dulu, setup lama dianggap expired dan market harus membentuk pattern baru.
- SL: 1.1215
- Reason SL: SL ditempatkan di bawah/atas structural invalidation terdekat (1.1227) + buffer 0.15 ATR M15.
- TP: 1.1365
- Reason TP: TP diarahkan ke H4_SWING_HIGH 1.1331 sebagai liquidity target HTF.

### Management

Tidak ada trailing.

### Result

- Result: EXPIRED
- Exit Price: 1.1327
- Result Reason: Price Exp 1.13263 adalah batas ekspansi thesis H4/H1 sebelum entry. Jika tercapai lebih dulu, setup lama dianggap expired dan market harus membentuk pattern baru.
- PnL: -
- Created: 2026-09-27T15:28:25.444660+00:00
- Filled: -
- Closed: 2026-09-27T15:29:03.104545+00:00
- Strategy: SMC_VLT_RSI v0.4.0

---

## NEARUSDT — BUY

Trade ID: `NEARUSDT-20260927-213711-BE2DC5`

### Setup

- Price Now Reference: 5.227
- Entry: 5.079
- Reason Entry: Model BREAKER_RETEST: area BULLISH_BREAKER di 5.079. Struktur pasangan mendukung bullish.
- Price Exp: 5.262
- Reason Price Exp: Price Exp 5.26129 menjadi batas ekspansi sebelum entry. Jika harga mencapai batas ini lebih dulu, setup lama dianggap expired dan pola baru perlu dicari.
- SL: 5.014
- Reason SL: SL di bawah low BULLISH_BREAKER (5.024) + buffer 0.15 ATR sebagai batas invalidasi struktur.
- TP: 5.304
- Reason TP: TP diarahkan ke EQUAL_LEVELS 5.1625 (0.12 ATR dari current) sebagai target liquidity.

### Management

Tidak ada trailing.

### Result

- Result: EXPIRED
- Exit Price: 5.262
- Result Reason: Price Exp 5.26129 menjadi batas ekspansi sebelum entry. Jika harga mencapai batas ini lebih dulu, setup lama dianggap expired dan pola baru perlu dicari.
- PnL: -
- Created: 2026-09-27T14:37:11.782125+00:00
- Filled: -
- Closed: 2026-09-27T15:29:30.086173+00:00
- Strategy: SMC_VLT_RSI v0.1.0

---

## IOTAUSDT — BUY

Trade ID: `IOTAUSDT-20260927-223001-7BCB0D`

### Setup

- Price Now Reference: 0.04974
- Entry: 0.04946
- Reason Entry: IOTAUSDT BUY: thesis dimulai dari POI H4. (BULLISH_BREAKER 0.0494). H1 me-refine ke BULLISH_BREAKER 0.04966. M15 memberi execution POI BULLISH_FVG 0.04946. Fib H4 berada pada band OVERLAPS_0.618 (score 92).
- Price Exp: 0.04979
- Reason Price Exp: Price Exp 0.049782 adalah batas ekspansi thesis H4/H1 sebelum entry. Jika tercapai lebih dulu, setup lama dianggap expired dan market harus membentuk pattern baru.
- SL: 0.04935
- Reason SL: SL ditempatkan di bawah/atas structural invalidation terdekat (0.0494) + buffer 0.15 ATR M15.
- TP: 0.04981
- Reason TP: TP diarahkan ke H4_SWING_HIGH 0.04981 sebagai liquidity target HTF.

### Management

Tidak ada trailing.

### Result

- Result: EXPIRED
- Exit Price: 0.04979
- Result Reason: Price Exp 0.049782 adalah batas ekspansi thesis H4/H1 sebelum entry. Jika tercapai lebih dulu, setup lama dianggap expired dan market harus membentuk pattern baru.
- PnL: -
- Created: 2026-09-27T15:30:01.692092+00:00
- Filled: -
- Closed: 2026-09-27T15:30:06.513115+00:00
- Strategy: SMC_VLT_RSI v0.4.0

---

## BATUSDT — BUY

Trade ID: `BATUSDT-20260927-223019-7510CD`

### Setup

- Price Now Reference: 0.09539
- Entry: 0.09377
- Reason Entry: BATUSDT BUY: thesis dimulai dari POI H4. (BULLISH_OB 0.094275). H1 me-refine ke BULLISH_FVG 0.09409. M15 memberi execution POI BULLISH_FVG 0.09377. Fib H4 berada pada band OVERLAPS_0.618 (score 92).
- Price Exp: 0.09554
- Reason Price Exp: Price Exp 0.09554 adalah batas ekspansi thesis H4/H1 sebelum entry. Jika tercapai lebih dulu, setup lama dianggap expired dan market harus membentuk pattern baru.
- SL: 0.09353
- Reason SL: SL ditempatkan di bawah/atas structural invalidation terdekat (0.09363) + buffer 0.15 ATR M15.
- TP: 0.09564
- Reason TP: TP diarahkan ke H4_EQUAL_LEVELS 0.09564 sebagai liquidity target HTF.

### Management

Tidak ada trailing.

### Result

- Result: EXPIRED
- Exit Price: 0.09554
- Result Reason: Price Exp 0.09554 adalah batas ekspansi thesis H4/H1 sebelum entry. Jika tercapai lebih dulu, setup lama dianggap expired dan market harus membentuk pattern baru.
- PnL: -
- Created: 2026-09-27T15:30:19.511454+00:00
- Filled: -
- Closed: 2026-09-27T15:34:10.265791+00:00
- Strategy: SMC_VLT_RSI v0.4.0

---

## KSMUSDT — BUY

Trade ID: `KSMUSDT-20260927-213650-BA13C3`

### Setup

- Price Now Reference: 4.787
- Entry: 4.721
- Reason Entry: Model BREAKER_RETEST: area BULLISH_BREAKER di 4.721. Struktur pasangan mendukung bullish.
- Price Exp: 4.802
- Reason Price Exp: Price Exp 4.80196 menjadi batas ekspansi sebelum entry. Jika harga mencapai batas ini lebih dulu, setup lama dianggap expired dan pola baru perlu dicari.
- SL: 4.712
- Reason SL: SL di bawah low BULLISH_BREAKER (4.716) + buffer 0.15 ATR sebagai batas invalidasi struktur.
- TP: 4.821
- Reason TP: TP diarahkan ke EQUAL_LEVELS 4.7865 (0.51 ATR dari current) sebagai target liquidity.

### Management

Tidak ada trailing.

### Result

- Result: EXPIRED
- Exit Price: 4.803
- Result Reason: Price Exp 4.80196 menjadi batas ekspansi sebelum entry. Jika harga mencapai batas ini lebih dulu, setup lama dianggap expired dan pola baru perlu dicari.
- PnL: -
- Created: 2026-09-27T14:36:50.630832+00:00
- Filled: -
- Closed: 2026-09-27T15:48:05.598325+00:00
- Strategy: SMC_VLT_RSI v0.1.0

---

## ONTUSDT — BUY

Trade ID: `ONTUSDT-20260927-222939-7FD4B9`

### Setup

- Price Now Reference: 0.05867
- Entry: 0.05811
- Reason Entry: ONTUSDT BUY: thesis dimulai dari POI H4. (BULLISH_FVG 0.057885). H1 me-refine ke BULLISH_FVG 0.05797. M15 memberi execution POI BULLISH_BREAKER 0.05811. Fib H4 berada pada band OVERLAPS_0.618 (score 92).
- Price Exp: 0.05896
- Reason Price Exp: Price Exp 0.058952 adalah batas ekspansi thesis H4/H1 sebelum entry. Jika tercapai lebih dulu, setup lama dianggap expired dan market harus membentuk pattern baru.
- SL: 0.05785
- Reason SL: SL ditempatkan di bawah/atas structural invalidation terdekat (0.05789) + buffer 0.15 ATR M15.
- TP: 0.05914
- Reason TP: TP diarahkan ke H4_SWING_HIGH 0.05914 sebagai liquidity target HTF.

### Management

Tidak ada trailing.

### Result

- Result: EXPIRED
- Exit Price: 0.05896
- Result Reason: Price Exp 0.058952 adalah batas ekspansi thesis H4/H1 sebelum entry. Jika tercapai lebih dulu, setup lama dianggap expired dan market harus membentuk pattern baru.
- PnL: -
- Created: 2026-09-27T15:29:39.501601+00:00
- Filled: -
- Closed: 2026-09-27T15:48:48.416346+00:00
- Strategy: SMC_VLT_RSI v0.4.0

---

## ATOMUSDT — BUY

Trade ID: `ATOMUSDT-20260927-222918-1E1EE9`

### Setup

- Price Now Reference: 1.86
- Entry: 1.841
- Reason Entry: ATOMUSDT BUY: thesis dimulai dari POI H4. (BULLISH_BREAKER 1.833). H1 me-refine ke BULLISH_FVG 1.8405. M15 memberi execution POI BULLISH_FVG 1.841. Fib H4 berada pada band OVERLAPS_0.618 (score 92).
- Price Exp: 1.883
- Reason Price Exp: Price Exp 1.8828 adalah batas ekspansi thesis H4/H1 sebelum entry. Jika tercapai lebih dulu, setup lama dianggap expired dan market harus membentuk pattern baru.
- SL: 1.836
- Reason SL: SL ditempatkan di bawah/atas structural invalidation terdekat (1.838) + buffer 0.15 ATR M15.
- TP: 1.925
- Reason TP: TP diarahkan ke H4_RECENT_RANGE_HIGH 1.925 sebagai liquidity target HTF.

### Management

Tidak ada trailing.

### Result

- Result: EXPIRED
- Exit Price: 1.883
- Result Reason: Price Exp 1.8828 adalah batas ekspansi thesis H4/H1 sebelum entry. Jika tercapai lebih dulu, setup lama dianggap expired dan market harus membentuk pattern baru.
- PnL: -
- Created: 2026-09-27T15:29:18.535723+00:00
- Filled: -
- Closed: 2026-09-27T16:11:04.328323+00:00
- Strategy: SMC_VLT_RSI v0.4.0

---

## ALGOUSDT — BUY

Trade ID: `ALGOUSDT-20260927-223155-EEA17C`

### Setup

- Price Now Reference: 0.11642
- Entry: 0.11519
- Reason Entry: ALGOUSDT BUY: thesis dimulai dari POI H4. (BULLISH_FVG 0.098285). H1 me-refine ke BULLISH_BREAKER 0.115355. M15 memberi execution POI BULLISH_BREAKER 0.11519. Fib H4 berada pada band DEEPER_THAN_0.786 (score 65).
- Price Exp: 0.11778
- Reason Price Exp: Price Exp 0.117773 adalah batas ekspansi thesis H4/H1 sebelum entry. Jika tercapai lebih dulu, setup lama dianggap expired dan market harus membentuk pattern baru.
- SL: 0.11483
- Reason SL: SL ditempatkan di bawah/atas structural invalidation terdekat (0.11495) + buffer 0.15 ATR M15.
- TP: 0.11868
- Reason TP: TP diarahkan ke H4_EQUAL_LEVELS 0.118675 sebagai liquidity target HTF.

### Management

Tidak ada trailing.

### Result

- Result: EXPIRED
- Exit Price: 0.11778
- Result Reason: Price Exp 0.117773 adalah batas ekspansi thesis H4/H1 sebelum entry. Jika tercapai lebih dulu, setup lama dianggap expired dan market harus membentuk pattern baru.
- PnL: -
- Created: 2026-09-27T15:31:55.151279+00:00
- Filled: -
- Closed: 2026-09-27T17:46:01.094879+00:00
- Strategy: SMC_VLT_RSI v0.4.0

---

## VETUSDT — BUY

Trade ID: `VETUSDT-20260927-223114-D84788`

### Setup

- Price Now Reference: 0.009198
- Entry: 0.009181
- Reason Entry: VETUSDT BUY: thesis dimulai dari POI H4. (BULLISH_FVG 0.007674). H1 me-refine ke BULLISH_BREAKER 0.009183. M15 memberi execution POI BULLISH_FVG 0.0091815. Fib H4 berada pada band DEEPER_THAN_0.786 (score 65).
- Price Exp: 0.009271
- Reason Price Exp: Price Exp 0.0092705171 adalah batas ekspansi thesis H4/H1 sebelum entry. Jika tercapai lebih dulu, setup lama dianggap expired dan market harus membentuk pattern baru.
- SL: 0.009136
- Reason SL: SL ditempatkan di bawah/atas structural invalidation terdekat (0.009145) + buffer 0.15 ATR M15.
- TP: 0.009335
- Reason TP: TP diarahkan ke H4_SWING_HIGH 0.009335 sebagai liquidity target HTF.

### Management

Tidak ada trailing.

### Result

- Result: TP
- Exit Price: 0.009335
- Result Reason: TP diarahkan ke H4_SWING_HIGH 0.009335 sebagai liquidity target HTF.
- PnL: 1.67737719202701230802744799%
- Created: 2026-09-27T15:31:14.730299+00:00
- Filled: 2026-09-27T15:32:09.292029+00:00
- Closed: 2026-09-27T18:12:20.111009+00:00
- Strategy: SMC_VLT_RSI v0.4.0

---

## NEOUSDT — BUY

Trade ID: `NEOUSDT-20260927-223128-B38F67`

### Setup

- Price Now Reference: 2.629
- Entry: 2.628
- Reason Entry: NEOUSDT BUY: thesis dimulai dari POI H4. (BULLISH_FVG 2.495). H1 me-refine ke BULLISH_FVG 2.4955. M15 memberi execution POI BULLISH_BREAKER 2.6285. Fib H4 berada pada band OVERLAPS_0.618 (score 96).
- Price Exp: 2.647
- Reason Price Exp: Price Exp 2.64627 adalah batas ekspansi thesis H4/H1 sebelum entry. Jika tercapai lebih dulu, setup lama dianggap expired dan market harus membentuk pattern baru.
- SL: 2.619
- Reason SL: SL ditempatkan di bawah/atas structural invalidation terdekat (2.622) + buffer 0.15 ATR M15.
- TP: 2.68
- Reason TP: TP diarahkan ke H4_EQUAL_LEVELS 2.6795 sebagai liquidity target HTF.

### Management

Tidak ada trailing.

### Result

- Result: TP
- Exit Price: 2.68
- Result Reason: TP diarahkan ke H4_EQUAL_LEVELS 2.6795 sebagai liquidity target HTF.
- PnL: 1.978691019786910197869101979%
- Created: 2026-09-27T15:31:28.983345+00:00
- Filled: 2026-09-27T15:31:43.276207+00:00
- Closed: 2026-09-27T18:44:38.002359+00:00
- Strategy: SMC_VLT_RSI v0.4.0

---

## AVAXUSDT — BUY

Trade ID: `AVAXUSDT-20260922-195523-78EDEF`

### Setup

- Price Now Reference: 10.815
- Entry: 10.13
- Reason Entry: RSI H4 OverBought (62.50) dan Divergent. Middle FVG + OB. Liquidity Pool 34%. RSI M15 sempat oversold tapi sekarang netral (ada kesempatan turun)
- Price Exp: 11.45
- Reason Price Exp: Telat Entry
- SL: 9.485
- Reason SL: di bawah OB
- TP: 11.448
- Reason TP: Swing High H4 (RR Supaya 1:2)

### Management

Tidak ada trailing.

### Result

- Result: TP
- Exit Price: 11.448
- Result Reason: Swing High H4 (RR Supaya 1:2)
- PnL: 13.01085883514313919052319842%
- Created: 2026-09-22T12:55:23.645025+00:00
- Filled: 2026-09-23T14:13:14.890022+00:00
- Closed: 2026-09-29T07:54:00.636719+00:00
- Strategy: MANUAL v1.0

---

## TREEUSDT — SELL

Trade ID: `TREEUSDT-20260930-175242-449B94`

### Setup

- Price Now Reference: 0.0453
- Entry: 0.04536
- Reason Entry: TREEUSDT SELL: thesis dimulai dari POI H4. (BEARISH_FVG 0.04783). H1 me-refine ke BEARISH_FVG 0.045395. M15 memberi execution POI BEARISH_FVG 0.04536. Fib H4 berada pada band DEEP_0.618_0.786 (score 100).
- Price Exp: 0.045
- Reason Price Exp: Price Exp 0.0450028 adalah batas ekspansi thesis H4/H1 sebelum entry. Jika tercapai lebih dulu, setup lama dianggap expired dan market harus membentuk pattern baru.
- SL: 0.04546
- Reason SL: SL ditempatkan di bawah/atas structural invalidation terdekat (0.04542) + buffer 0.15 ATR M15.
- TP: 0.0446
- Reason TP: TP diarahkan ke H4_SWING_LOW 0.0446 sebagai liquidity target HTF.

### Management

Tidak ada trailing.

### Result

- Result: SL
- Exit Price: 0.04551
- Result Reason: SL ditempatkan di bawah/atas structural invalidation terdekat (0.04542) + buffer 0.15 ATR M15.
- PnL: -0.3306878306878306878306878307%
- Created: 2026-09-30T10:52:42.864237+00:00
- Filled: 2026-09-30T10:52:59.026474+00:00
- Closed: 2026-09-30T10:53:01.789409+00:00
- Strategy: SMC_VLT_RSI v0.5.0

---

## MAVUSDT — SELL

Trade ID: `MAVUSDT-20260930-175240-ACDAE2`

### Setup

- Price Now Reference: 0.011865
- Entry: 0.011911
- Reason Entry: MAVUSDT SELL: thesis dimulai dari POI H4. (BEARISH_FVG 0.011908). M15 memberi execution POI BEARISH_FVG 0.011911. Fib H4 berada pada band 0.382_0.500 (score 72).
- Price Exp: 0.011838
- Reason Price Exp: Price Exp 0.0118388 adalah batas ekspansi thesis H4/H1 sebelum entry. Jika tercapai lebih dulu, setup lama dianggap expired dan market harus membentuk pattern baru.
- SL: 0.011923
- Reason SL: SL ditempatkan di bawah/atas structural invalidation terdekat (0.011912) + buffer 0.15 ATR M15.
- TP: 0.011821
- Reason TP: TP diarahkan ke H4_SWING_LOW 0.011889 sebagai liquidity target HTF.

### Management

Tidak ada trailing.

### Result

- Result: EXPIRED
- Exit Price: 0.011838
- Result Reason: Price Exp 0.0118388 adalah batas ekspansi thesis H4/H1 sebelum entry. Jika tercapai lebih dulu, setup lama dianggap expired dan market harus membentuk pattern baru.
- PnL: -
- Created: 2026-09-30T10:52:40.848037+00:00
- Filled: -
- Closed: 2026-09-30T10:53:35.831418+00:00
- Strategy: SMC_VLT_RSI v0.5.0

---

## COAIUSDT — SELL

Trade ID: `COAIUSDT-20260930-175232-C9AA83`

### Setup

- Price Now Reference: 0.3081
- Entry: 0.3175
- Reason Entry: COAIUSDT SELL: thesis dimulai dari POI H4. (BEARISH_FVG 0.31705). H1 me-refine ke BEARISH_FVG 0.3172. M15 memberi execution POI BEARISH_FVG 0.31745. Fib H4 berada pada band DEEP_0.618_0.786 (score 100).
- Price Exp: 0.308
- Reason Price Exp: Price Exp 0.30798 adalah batas ekspansi thesis H4/H1 sebelum entry. Jika tercapai lebih dulu, setup lama dianggap expired dan market harus membentuk pattern baru.
- SL: 0.3179
- Reason SL: SL ditempatkan di bawah/atas structural invalidation terdekat (0.3176) + buffer 0.15 ATR M15.
- TP: 0.3079
- Reason TP: TP diarahkan ke H4_EQUAL_LEVELS 0.3079 sebagai liquidity target HTF.

### Management

Tidak ada trailing.

### Result

- Result: EXPIRED
- Exit Price: 0.308
- Result Reason: Price Exp 0.30798 adalah batas ekspansi thesis H4/H1 sebelum entry. Jika tercapai lebih dulu, setup lama dianggap expired dan market harus membentuk pattern baru.
- PnL: -
- Created: 2026-09-30T10:52:32.799668+00:00
- Filled: -
- Closed: 2026-09-30T11:00:45.410094+00:00
- Strategy: SMC_VLT_RSI v0.5.0

---

## CGPTUSDT — SELL

Trade ID: `CGPTUSDT-20260930-175243-777E28`

### Setup

- Price Now Reference: 0.02185
- Entry: 0.02224
- Reason Entry: CGPTUSDT SELL: thesis dimulai dari POI H4. (BEARISH_FVG 0.02226). H1 me-refine ke BEARISH_FVG 0.022255. M15 memberi execution POI BEARISH_FVG 0.022235. Fib H4 berada pada band DEEPER_THAN_0.786 (score 69).
- Price Exp: 0.02173
- Reason Price Exp: Price Exp 0.021736 adalah batas ekspansi thesis H4/H1 sebelum entry. Jika tercapai lebih dulu, setup lama dianggap expired dan market harus membentuk pattern baru.
- SL: 0.02227
- Reason SL: SL ditempatkan di bawah/atas structural invalidation terdekat (0.02225) + buffer 0.15 ATR M15.
- TP: 0.02166
- Reason TP: TP diarahkan ke H4_SWING_LOW 0.02166 sebagai liquidity target HTF.

### Management

Tidak ada trailing.

### Result

- Result: EXPIRED
- Exit Price: 0.02173
- Result Reason: Price Exp 0.021736 adalah batas ekspansi thesis H4/H1 sebelum entry. Jika tercapai lebih dulu, setup lama dianggap expired dan market harus membentuk pattern baru.
- PnL: -
- Created: 2026-09-30T10:52:43.871065+00:00
- Filled: -
- Closed: 2026-09-30T11:00:55.663021+00:00
- Strategy: SMC_VLT_RSI v0.5.0

---

## PIPPINUSDT — SELL

Trade ID: `PIPPINUSDT-20260930-175247-8EE2A2`

### Setup

- Price Now Reference: 0.019
- Entry: 0.01901
- Reason Entry: PIPPINUSDT SELL: thesis dimulai dari POI H4. (BEARISH_OB 0.019075). H1 me-refine ke BEARISH_FVG 0.01901. M15 memberi execution POI BEARISH_FVG 0.019005. Fib H4 berada pada band OVERLAPS_0.618 (score 96).
- Price Exp: 0.01884
- Reason Price Exp: Price Exp 0.0188468 adalah batas ekspansi thesis H4/H1 sebelum entry. Jika tercapai lebih dulu, setup lama dianggap expired dan market harus membentuk pattern baru.
- SL: 0.01907
- Reason SL: SL ditempatkan di bawah/atas structural invalidation terdekat (0.01903) + buffer 0.15 ATR M15.
- TP: 0.01874
- Reason TP: TP diarahkan ke H4_SWING_LOW 0.01877 sebagai liquidity target HTF.

### Management

Tidak ada trailing.

### Result

- Result: EXPIRED
- Exit Price: 0.01884
- Result Reason: Price Exp 0.0188468 adalah batas ekspansi thesis H4/H1 sebelum entry. Jika tercapai lebih dulu, setup lama dianggap expired dan market harus membentuk pattern baru.
- PnL: -
- Created: 2026-09-30T10:52:47.899622+00:00
- Filled: -
- Closed: 2026-09-30T11:01:38.692413+00:00
- Strategy: SMC_VLT_RSI v0.5.0

---

## CLOUSDT — SELL

Trade ID: `CLOUSDT-20260930-175250-443619`

### Setup

- Price Now Reference: 0.06068
- Entry: 0.0608
- Reason Entry: CLOUSDT SELL: thesis dimulai dari POI H4. (BEARISH_FVG 0.06072). H1 me-refine ke BEARISH_FVG 0.060685. M15 memberi execution POI BEARISH_FVG 0.060795. Fib H4 berada pada band DEEP_0.618_0.786 (score 100). RSI 14 M15 49.6 mendukung momentum.
- Price Exp: 0.06016
- Reason Price Exp: Price Exp 0.0601676 adalah batas ekspansi thesis H4/H1 sebelum entry. Jika tercapai lebih dulu, setup lama dianggap expired dan market harus membentuk pattern baru.
- SL: 0.06086
- Reason SL: SL ditempatkan di bawah/atas structural invalidation terdekat (0.0608) + buffer 0.15 ATR M15.
- TP: 0.05885
- Reason TP: TP diarahkan ke H4_EQUAL_LEVELS 0.05885 sebagai liquidity target HTF.

### Management

Tidak ada trailing.

### Result

- Result: EXPIRED
- Exit Price: 0.06016
- Result Reason: Price Exp 0.0601676 adalah batas ekspansi thesis H4/H1 sebelum entry. Jika tercapai lebih dulu, setup lama dianggap expired dan market harus membentuk pattern baru.
- PnL: -
- Created: 2026-09-30T10:52:50.913352+00:00
- Filled: -
- Closed: 2026-09-30T11:01:55.074628+00:00
- Strategy: SMC_VLT_RSI v0.5.0

---

## 1000000MOGUSDT — SELL

Trade ID: `1000000MOGUSDT-20260930-175241-E9C48F`

### Setup

- Price Now Reference: 0.1194
- Entry: 0.1197
- Reason Entry: 1000000MOGUSDT SELL: thesis dimulai dari POI H4. (BEARISH_BREAKER 0.1215). H1 me-refine ke BEARISH_BREAKER 0.12005. M15 memberi execution POI BEARISH_BREAKER 0.11965. Fib H4 berada pada band DEEPER_THAN_0.786 (score 65).
- Price Exp: 0.1188
- Reason Price Exp: Price Exp 0.1188 adalah batas ekspansi thesis H4/H1 sebelum entry. Jika tercapai lebih dulu, setup lama dianggap expired dan market harus membentuk pattern baru.
- SL: 0.1204
- Reason SL: SL ditempatkan di bawah/atas structural invalidation terdekat (0.1203) + buffer 0.15 ATR M15.
- TP: 0.1184
- Reason TP: TP diarahkan ke H4_EQUAL_LEVELS 0.1184 sebagai liquidity target HTF.

### Management

Tidak ada trailing.

### Result

- Result: EXPIRED
- Exit Price: 0.1188
- Result Reason: Price Exp 0.1188 adalah batas ekspansi thesis H4/H1 sebelum entry. Jika tercapai lebih dulu, setup lama dianggap expired dan market harus membentuk pattern baru.
- PnL: -
- Created: 2026-09-30T10:52:41.857309+00:00
- Filled: -
- Closed: 2026-09-30T11:02:12.103623+00:00
- Strategy: SMC_VLT_RSI v0.5.0

---

## NILUSDT — SELL

Trade ID: `NILUSDT-20260930-175239-19063D`

### Setup

- Price Now Reference: 0.08952
- Entry: 0.08961
- Reason Entry: Model M15_SMC_FALLBACK: area BEARISH_OB di 0.089605. Struktur pasangan mendukung bearish.
- Price Exp: 0.08806
- Reason Price Exp: Price Exp 0.0880665 menjadi batas ekspansi sebelum entry. Jika harga mencapai batas ini lebih dulu, setup lama dianggap expired dan pola baru perlu dicari.
- SL: 0.09052
- Reason SL: SL di atas high BEARISH_OB (0.09029) + buffer 0.15 ATR sebagai batas invalidasi struktur.
- TP: 0.08629
- Reason TP: TP diarahkan ke M15_SWING_LOW 0.08629 (1.84 ATR dari current) sebagai target liquidity.

### Management

Tidak ada trailing.

### Result

- Result: SL
- Exit Price: 0.09052
- Result Reason: SL di atas high BEARISH_OB (0.09029) + buffer 0.15 ATR sebagai batas invalidasi struktur.
- PnL: -1.015511661644905702488561544%
- Created: 2026-09-30T10:52:39.840837+00:00
- Filled: 2026-09-30T10:52:40.307302+00:00
- Closed: 2026-09-30T11:08:30.727167+00:00
- Strategy: SMC_VLT_RSI v0.5.0

---

## RVNUSDT — SELL

Trade ID: `RVNUSDT-20260930-175251-ECE25F`

### Setup

- Price Now Reference: 0.002368
- Entry: 0.002372
- Reason Entry: RVNUSDT SELL: thesis dimulai dari POI H4. (BEARISH_FVG 0.002381). H1 me-refine ke BEARISH_FVG 0.0023955. M15 memberi execution POI BEARISH_FVG 0.002372. Fib H4 berada pada band SHALLOW (score 45).
- Price Exp: 0.00236
- Reason Price Exp: Price Exp 0.0023602557 adalah batas ekspansi thesis H4/H1 sebelum entry. Jika tercapai lebih dulu, setup lama dianggap expired dan market harus membentuk pattern baru.
- SL: 0.002377
- Reason SL: SL ditempatkan di bawah/atas structural invalidation terdekat (0.002374) + buffer 0.15 ATR M15.
- TP: 0.002355
- Reason TP: TP diarahkan ke H4_SWING_LOW 0.002362 sebagai liquidity target HTF.

### Management

Tidak ada trailing.

### Result

- Result: SL
- Exit Price: 0.002377
- Result Reason: SL ditempatkan di bawah/atas structural invalidation terdekat (0.002374) + buffer 0.15 ATR M15.
- PnL: -0.2107925801011804384485666105%
- Created: 2026-09-30T10:52:51.921046+00:00
- Filled: 2026-09-30T10:54:26.412862+00:00
- Closed: 2026-09-30T11:09:26.283889+00:00
- Strategy: SMC_VLT_RSI v0.5.0

---

## ZECUSDT — SELL

Trade ID: `ZECUSDT-20260930-202208-E8EC1F`

### Setup

- Price Now Reference: 1474
- Entry: 1484.75
- Reason Entry: ZECUSDT SELL: thesis dimulai dari POI H4. (BEARISH_FVG 1630.4). H1 me-refine ke BEARISH_OB 1480.78. M15 memberi execution POI BEARISH_OB 1484.75. Fib H4 berada pada band DEEPER_THAN_0.786 (score 65). terdapat liquidity sweep searah reversal. diikuti MSS pada M15.
- Price Exp: 1457.74
- Reason Price Exp: Price Exp 1457.74 adalah batas ekspansi thesis H4/H1 sebelum entry. Jika tercapai lebih dulu, setup lama dianggap expired dan market harus membentuk pattern baru.
- SL: 1489.17
- Reason SL: SL ditempatkan di bawah/atas structural invalidation terdekat (1487.22) + buffer 0.15 ATR M15.
- TP: 1442.62
- Reason TP: TP diarahkan ke H4_EQUAL_LEVELS 1442.62 sebagai liquidity target HTF.

### Management

Tidak ada trailing.

### Result

- Result: SL
- Exit Price: 1489.17
- Result Reason: SL ditempatkan di bawah/atas structural invalidation terdekat (1487.22) + buffer 0.15 ATR M15.
- PnL: -0.2976932143458494696076780603%
- Created: 2026-09-30T13:22:08.127756+00:00
- Filled: 2026-09-30T13:27:11.130008+00:00
- Closed: 2026-09-30T13:27:16.451312+00:00
- Strategy: SMC_VLT_RSI v0.6.0

---

## 1000PEPEUSDT — BUY

Trade ID: `1000PEPEUSDT-20260922-193547-29F3A0`

### Setup

- Price Now Reference: 0.0049115
- Entry: 0.0041686
- Reason Entry: RSI H4 OverBought + RSI Divergent tapi, RSI M15 Oversold (41.89). Liquidity Pool Bullish 36%. Middle FVG H4. Discount Zone Fibonachi H4
- Price Exp: 0.005275
- Reason Price Exp: di atas TP
- SL: 0.0038454
- Reason SL: Dibawah garih volumatic trend dan order block, di bawah Liquidity Pool
- TP: 0.0052748
- Reason TP: Swing High H4 RR: 3.42

### Management

Tidak ada trailing.

### Result

- Result: TP
- Exit Price: 0.0044809
- Result Reason: Manual /close. PnL positif, sehingga hasil dicatat sebagai TP (PnL +7.49%).
- PnL: 7.491723840138175886388715636%
- Created: 2026-09-22T12:35:47.195079+00:00
- Filled: 2026-09-28T03:32:59.723994+00:00
- Closed: 2026-09-30T13:30:49.991690+00:00
- Strategy: MANUAL v1.0

---

## ENAUSDT — BUY

Trade ID: `ENAUSDT-20260926-234822-1F3682`

### Setup

- Price Now Reference: 0.27441
- Entry: 0.24743
- Reason Entry: Liquidity Pool 79%. Zona Fibo 0.382. RSI H4 (76). Middle FVG
- Price Exp: 0.29613
- Reason Price Exp: TP
- SL: 0.22794
- Reason SL: di bawah Pool Liquidity
- TP: 0.29613
- Reason TP: Support D1

### Management

Tidak ada trailing.

### Result

- Result: TP
- Exit Price: 0.26842
- Result Reason: Manual /close. PnL positif, sehingga hasil dicatat sebagai TP (PnL +8.48%).
- PnL: 8.48320737178191811825566827%
- Created: 2026-09-26T16:48:22.527777+00:00
- Filled: 2026-09-29T01:37:35.825454+00:00
- Closed: 2026-09-30T13:31:20.744880+00:00
- Strategy: MANUAL v1.0

---

## ASTERUSDT — BUY

Trade ID: `ASTERUSDT-20260930-202211-84901C`

### Setup

- Price Now Reference: 0.775
- Entry: 0.7688
- Reason Entry: Model M15_SMC_FALLBACK: area BULLISH_OB di 0.76885. Ada sell-side liquidity sweep sebelum perubahan struktur. M15 membentuk MSS bullish dengan displacement. Struktur pasangan mendukung bullish. RSI 14 M15 berada di 62.9, mendukung momentum bullish.
- Price Exp: 0.7791
- Reason Price Exp: Price Exp 0.7790772 menjadi batas ekspansi sebelum entry. Jika harga mencapai batas ini lebih dulu, setup lama dianggap expired dan pola baru perlu dicari.
- SL: 0.7655
- Reason SL: SL di bawah low BULLISH_OB (0.7666) + buffer 0.15 ATR sebagai batas invalidasi struktur.
- TP: 0.7841
- Reason TP: TP diarahkan ke M15_RECENT_RANGE_HIGH 0.7815 (0.90 ATR dari current) sebagai target liquidity.

### Management

Tidak ada trailing.

### Result

- Result: EXPIRED
- Exit Price: 0.7791
- Result Reason: Price Exp 0.7790772 menjadi batas ekspansi sebelum entry. Jika harga mencapai batas ini lebih dulu, setup lama dianggap expired dan pola baru perlu dicari.
- PnL: -
- Created: 2026-09-30T13:22:11.166641+00:00
- Filled: -
- Closed: 2026-09-30T13:32:07.293880+00:00
- Strategy: SMC_VLT_RSI v0.6.0
