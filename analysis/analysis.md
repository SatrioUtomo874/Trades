# Trading Analysis & AI Reading Report

Generated: 17-10-2026, 13:35 WIB
Session: `20261017-045444-D4D9`

## 1. Scope and Data Source

- Market: Binance USDⓈ-M Futures
- Execution in this export: Simulation + Real (see real_enabled per trade)
- Price trigger: Binance `aggTrade` last price
- Historical trade records: 6
- Event records: 7
- User notes: 0
- Active setups at export time: 6

## 2. Executive Summary

- Total historical setup records: 6
- Historical filled-and-closed trades: 0
- Historical TP: 0
- Historical SL: 0
- Historical Expired: 5
- Historical Deleted: 0
- Historical fill rate: 0%
- Historical expiration rate: 83.33333333333333333333333333%
- Historical deletion rate: 0%
- Win Rate: 0%
- Historical PnL sum: 0%
- Average win: 0%
- Average loss: 0%
- Average planned RR: 3.254827462900053638476667263
- Average pending duration among completed filled trades: -
- Average holding duration among completed filled trades: -
- Trailing trade count: 0
- Total trail events: 0

**Win Rate definition:** TP / (TP + SL).
Expired dan Deleted tidak dihitung sebagai win/loss.
Historical metrics above only use records already closed and stored in history.
Active setup data is reported separately and is not counted as historical closed performance.

## 3. Current Active Session

- Active setups: 6
- Pending: 5
- Filled: 1

| Pair | Direction | Status | Entry | SL | TP | Price Exp | Trailing |
|---|---|---|---:|---:|---:|---:|---|
| BLESSUSDT | SELL | PENDING | 0.007819 | 0.007891 | 0.007195 | 0.007382 | NO |
| MANAUSDT | SELL | PENDING | 0.09141 | 0.09197 | 0.0874 | 0.08763 | NO |
| EULUSDT | SELL | PENDING | 1.2911 | 1.3308 | 1.1787 | 1.2124 | NO |
| EVAAUSDT | SELL | PENDING | 0.5955 | 0.6164 | 0.5497 | 0.5743 | NO |
| SXTUSDT | SELL | FILLED | 0.008515 | 0.008705 | 0.007953 | 0.008208 | NO |
| OGUSDT | SELL | PENDING | 2.72 | 2.737 | 2.669 | 2.684 | NO |

## 4. By Pair

| Pair | Setup | Filled | TP | SL | Expired | Deleted | Win Rate | PnL % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BSBUSDT | 1 | 0 | 0 | 0 | 1 | 0 | 0% | 0 |
| DYMUSDT | 1 | 0 | 0 | 0 | 1 | 0 | 0% | 0 |
| ENJUSDT | 1 | 0 | 0 | 0 | 0 | 0 | 0% | 0 |
| GRAMUSDT | 1 | 0 | 0 | 0 | 1 | 0 | 0% | 0 |
| STBLUSDT | 1 | 0 | 0 | 0 | 1 | 0 | 0% | 0 |
| ZRXUSDT | 1 | 0 | 0 | 0 | 1 | 0 | 0% | 0 |

## 5. By Direction

| Direction | Setup | Filled | TP | SL | Expired | Deleted | Win Rate | PnL % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BUY | 3 | 0 | 0 | 0 | 2 | 0 | 0% | 0 |
| SELL | 3 | 0 | 0 | 0 | 3 | 0 | 0% | 0 |

## 6. By Strategy

| Strategy | Version | Setup | Filled | TP | SL | Expired | Win Rate | PnL % |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| SMC_VLT_RSI | 2.3.1 | 6 | 0 | 0 | 0 | 5 | 0% | 0 |

## 7. Event Distribution

| Event | Count |
|---|---:|
| EXPIRED | 5 |
| FILLED | 1 |
| INVALID_BTC | 1 |

## 8. Expiration Reasons

| Reason | Count |
|---|---:|
| Price Exp 0.01828: retracement thesis dianggap basi jika harga bergerak terlalu jauh tanpa menyentuh entry zone sebelum menembus boundary relevansi. | 1 |
| Price Exp 0.025150000000000002: retracement thesis dianggap basi jika harga bergerak terlalu jauh tanpa menyentuh entry zone sebelum menembus boundary relevansi. | 1 |
| Price Exp 0.08754: retracement thesis dianggap basi jika harga bergerak terlalu jauh tanpa menyentuh entry zone sebelum menembus boundary relevansi. | 1 |
| Price Exp 0.11138: retracement thesis dianggap basi jika harga bergerak terlalu jauh tanpa menyentuh entry zone sebelum menembus boundary relevansi. | 1 |
| Price Exp 1.347: retracement thesis dianggap basi jika harga bergerak terlalu jauh tanpa menyentuh entry zone sebelum menembus boundary relevansi. | 1 |

## 9. Data Quality / Interpretation Flags

- Records where Price Exp exactly equals TP: 0
- Expired records whose reason text mentions TP: 0
- These are flags for review, not automatic strategy judgments.

## 10. Trade-by-Trade Detail

### Trade 1: STBLUSDT-20261017-070247-F7A64E

- Pair: STBLUSDT
- Direction: BUY
- Status: CLOSED
- Result: EXPIRED
- Price Now Reference: 0.02506
- Entry: 0.02298
- Reason Entry: STBLUSDT BUY: BUY_PULLBACK_RSI_DIVERGENCE. H4 structure BULLISH, liquidity H4 0.0229775 dipadukan dengan Fib SECONDARY; refinement BULLISH_FVG. Projected M15 RSI 33.9 pada entry zone.
- Price Exp: 0.02515
- Reason Price Exp: Price Exp 0.025150000000000002: retracement thesis dianggap basi jika harga bergerak terlalu jauh tanpa menyentuh entry zone sebelum menembus boundary relevansi.
- SL: 0.02204
- Reason SL: Prediction invalidation berasal dari H1_SELL_SIDE_LIQUIDITY di 0.02291; SL ditempatkan di luar invalidation dan liquidity guard 0.0232527.
- TP: 0.02608
- Reason TP: TP 0.02608 dipilih dari H4_SWING_HIGH dengan RR 3.30; target quality 82.
- Created: 2026-10-17T00:02:47.891440+00:00
- Filled: None
- Closed: 2026-10-17T00:06:38.972677+00:00
- Fill Price: None
- Exit Price: 0.02515
- Result Reason: Price Exp 0.025150000000000002: retracement thesis dianggap basi jika harga bergerak terlalu jauh tanpa menyentuh entry zone sebelum menembus boundary relevansi.
- PnL %: None
- Planned RR: 3.297872340425531914893617021
- Pending Seconds: None
- Holding Seconds: None
- Trailing: NO
- Trail History Count: 0
- Strategy: SMC_VLT_RSI
- Strategy Version: 2.3.1

### Trade 2: DYMUSDT-20261017-071908-FA2341

- Pair: DYMUSDT
- Direction: BUY
- Status: CLOSED
- Result: EXPIRED
- Price Now Reference: 0.01827
- Entry: 0.01797
- Reason Entry: DYMUSDT BUY: BUY_PULLBACK_RSI_DIVERGENCE. H4 structure BULLISH, liquidity H4 0.01796 dipadukan dengan Fib PRIMARY; refinement BEARISH_FVG. Projected M15 RSI 32.9 pada entry zone.
- Price Exp: 0.01828
- Reason Price Exp: Price Exp 0.01828: retracement thesis dianggap basi jika harga bergerak terlalu jauh tanpa menyentuh entry zone sebelum menembus boundary relevansi.
- SL: 0.0178
- Reason SL: Prediction invalidation berasal dari H4_PROTECTED_LOW di 0.01796; SL ditempatkan di luar invalidation.
- TP: 0.01841
- Reason TP: TP 0.01841 dipilih dari H4_SWING_HIGH dengan RR 2.59; target quality 88.
- Created: 2026-10-17T00:19:08.681751+00:00
- Filled: None
- Closed: 2026-10-17T00:19:25.656435+00:00
- Fill Price: None
- Exit Price: 0.0183
- Result Reason: Price Exp 0.01828: retracement thesis dianggap basi jika harga bergerak terlalu jauh tanpa menyentuh entry zone sebelum menembus boundary relevansi.
- PnL %: None
- Planned RR: 2.588235294117647058823529412
- Pending Seconds: None
- Holding Seconds: None
- Trailing: NO
- Trail History Count: 0
- Strategy: SMC_VLT_RSI
- Strategy Version: 2.3.1

### Trade 3: BSBUSDT-20261017-045634-740604

- Pair: BSBUSDT
- Direction: SELL
- Status: CLOSED
- Result: EXPIRED
- Price Now Reference: 0.08762
- Entry: 0.08924
- Reason Entry: BSBUSDT SELL: SELL_PULLBACK. H4 structure BEARISH, liquidity H1 0.0893363 dipadukan dengan Fib DEEP_0.618_0.786; refinement BULLISH_FVG. Projected M15 RSI 72.4 pada entry zone.
- Price Exp: 0.08754
- Reason Price Exp: Price Exp 0.08754: retracement thesis dianggap basi jika harga bergerak terlalu jauh tanpa menyentuh entry zone sebelum menembus boundary relevansi.
- SL: 0.09001
- Reason SL: Prediction invalidation berasal dari H1_PROTECTED_HIGH di 0.08928; SL ditempatkan di luar invalidation dan liquidity guard 0.0893363.
- TP: 0.08682
- Reason TP: TP 0.08682 dipilih dari H4_LOWER_LOW dengan RR 3.14; target quality 90.
- Created: 2026-10-16T21:56:34.981879+00:00
- Filled: None
- Closed: 2026-10-17T00:48:18.160981+00:00
- Fill Price: None
- Exit Price: 0.08754
- Result Reason: Price Exp 0.08754: retracement thesis dianggap basi jika harga bergerak terlalu jauh tanpa menyentuh entry zone sebelum menembus boundary relevansi.
- PnL %: None
- Planned RR: 3.142857142857142857142857143
- Pending Seconds: None
- Holding Seconds: None
- Trailing: NO
- Trail History Count: 0
- Strategy: SMC_VLT_RSI
- Strategy Version: 2.3.1

### Trade 4: GRAMUSDT-20261017-070928-AFB848

- Pair: GRAMUSDT
- Direction: SELL
- Status: CLOSED
- Result: EXPIRED
- Price Now Reference: 1.352
- Entry: 1.368
- Reason Entry: GRAMUSDT SELL: SELL_PULLBACK. H4 structure BEARISH, liquidity M15 1.368 dipadukan dengan Fib PRIMARY; refinement BULLISH_FVG. Projected M15 RSI 67.0 pada entry zone.
- Price Exp: 1.347
- Reason Price Exp: Price Exp 1.347: retracement thesis dianggap basi jika harga bergerak terlalu jauh tanpa menyentuh entry zone sebelum menembus boundary relevansi.
- SL: 1.382
- Reason SL: Prediction invalidation berasal dari BEARISH_SWEEP_EXTREME di 1.381; SL ditempatkan di luar invalidation.
- TP: 1.339
- Reason TP: TP 1.339 dipilih dari H4_LOWER_LOW dengan RR 2.07; target quality 88.
- Created: 2026-10-17T00:09:28.278385+00:00
- Filled: None
- Closed: 2026-10-17T00:49:47.157349+00:00
- Fill Price: None
- Exit Price: 1.347
- Result Reason: Price Exp 1.347: retracement thesis dianggap basi jika harga bergerak terlalu jauh tanpa menyentuh entry zone sebelum menembus boundary relevansi.
- PnL %: None
- Planned RR: 2.071428571428571428571428571
- Pending Seconds: None
- Holding Seconds: None
- Trailing: NO
- Trail History Count: 0
- Strategy: SMC_VLT_RSI
- Strategy Version: 2.3.1

### Trade 5: ENJUSDT-20261017-071914-18B5C2

- Pair: ENJUSDT
- Direction: BUY
- Status: CLOSED
- Result: INVALID_BTC
- Price Now Reference: 0.02943
- Entry: 0.02905
- Reason Entry: ENJUSDT BUY: BUY_PULLBACK. H4 structure BULLISH, liquidity H1 0.0288884 dipadukan dengan Fib DEEP_0.618_0.786; refinement BEARISH_FVG. Projected M15 RSI 35.6 pada entry zone.
- Price Exp: 0.03001
- Reason Price Exp: Price Exp 0.030010000000000002: retracement thesis dianggap basi jika harga bergerak terlalu jauh tanpa menyentuh entry zone sebelum menembus boundary relevansi.
- SL: 0.02863
- Reason SL: Prediction invalidation berasal dari H1_PROTECTED_LOW di 0.02891; SL ditempatkan di luar invalidation dan liquidity guard 0.0288884.
- TP: 0.03042
- Reason TP: TP 0.03042 dipilih dari H4_HIGHER_HIGH dengan RR 3.26; target quality 86.
- Created: 2026-10-17T00:19:14.266754+00:00
- Filled: None
- Closed: 2026-10-17T04:02:18.257618+00:00
- Fill Price: None
- Exit Price: None
- Result Reason: BTC berbalik: BTC H4 berbalik BEARISH; setup BUY ENJUSDT dibatalkan dan dianggap invalid.
- PnL %: None
- Planned RR: 3.261904761904761904761904762
- Pending Seconds: None
- Holding Seconds: None
- Trailing: NO
- Trail History Count: 0
- Strategy: SMC_VLT_RSI
- Strategy Version: 2.3.1

### Trade 6: ZRXUSDT-20261017-051516-3D9C85

- Pair: ZRXUSDT
- Direction: SELL
- Status: CLOSED
- Result: EXPIRED
- Price Now Reference: 0.11193
- Entry: 0.1142
- Reason Entry: ZRXUSDT SELL: SELL_PULLBACK. H4 structure BEARISH, liquidity H1 0.1142529 dipadukan dengan Fib PRIMARY; refinement BEARISH_OB. Projected M15 RSI 74.0 pada entry zone.
- Price Exp: 0.1113
- Reason Price Exp: Price Exp 0.11138: retracement thesis dianggap basi jika harga bergerak terlalu jauh tanpa menyentuh entry zone sebelum menembus boundary relevansi.
- SL: 0.1148
- Reason SL: Prediction invalidation berasal dari H1_PROTECTED_HIGH di 0.11425; SL ditempatkan di luar invalidation dan liquidity guard 0.1142529.
- TP: 0.1111
- Reason TP: TP 0.11115 dipilih dari H4_LOWER_LOW dengan RR 4.29; target quality 85.
- Created: 2026-10-16T22:15:16.982103+00:00
- Filled: None
- Closed: 2026-10-17T06:23:32.559461+00:00
- Fill Price: None
- Exit Price: 0.1113
- Result Reason: Price Exp 0.11138: retracement thesis dianggap basi jika harga bergerak terlalu jauh tanpa menyentuh entry zone sebelum menembus boundary relevansi.
- PnL %: None
- Planned RR: 5.166666666666666666666666667
- Pending Seconds: None
- Holding Seconds: None
- Trailing: NO
- Trail History Count: 0
- Strategy: SMC_VLT_RSI
- Strategy Version: 2.3.1

## 11. User Notes (/catatan)

Belum ada catatan.

## 12. AI-Readable Conclusions

1. Dari 6 historical setup, 5 berakhir EXPIRED dan 0 berakhir sebagai filled trade yang kemudian closed.
2. Expiration rate historis adalah 83.33333333333333333333333333%; angka ini menggambarkan frekuensi setup berakhir sebelum menjadi trade yang selesai, bukan win/loss rate.
3. Historical filled sample berjumlah 0; TP=0 dan SL=0. Win Rate historis menurut definisi sistem adalah 0%.
4. Historical PnL sum hanya berasal dari record yang memiliki PnL, dengan nilai 0%.
5. Active session saat export memiliki 6 setup (5 PENDING, 1 FILLED); setup aktif tidak dimasukkan ke historical closed performance.
6. Total trailing event yang tercatat adalah 0 dan jumlah trade historis yang memiliki trailing adalah 0.
7. Direction historis yang muncul: BUY, SELL.
8. Pair historis yang muncul: BSBUSDT, DYMUSDT, ENJUSDT, GRAMUSDT, STBLUSDT, ZRXUSDT.
9. Reason EXPIRED yang paling sering tercatat adalah 'Price Exp 0.01828: retracement thesis dianggap basi jika harga bergerak terlalu jauh tanpa menyentuh entry zone sebelum menembus boundary relevansi.' (1 record).
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

- Kandidat: {'PENDING': 5, 'EXPIRED': 6, 'READY': 5, 'REJECTED_QUALITY': 222, 'FILLED': 1, 'WAITING_PRICE': 61}
- Funnel scan terakhir: {'universe': 461, 'usable_data': 15, 'structure_analyzed': 15, 'direction_thesis': 15, 'candidates_evaluated': 15, 'hard_rejects': 0, 'quality_rejects': 0, 'waiting_price': 15, 'waiting_trigger': 0, 'ready': 0, 'validation_pass': 0, 'validation_reject': 0, 'validation_error': 0, 'risk_rejects': 0, 'pending_created': 0, 'structure_mismatch': 0, 'data_unavailable': 0, 'data_stale': 0, 'budget_stopped': 0, 'memory_skipped': 0, 'validation_skipped_topn': 0, 'rate_limited': 0, 'sweep_covered': 444, 'sweep_total': 444, 'sim_margin_blocked': 0, 'errors': {}, 'validation_reasons': {}}
- R-metrik histori: belum ada sampel
- Cooldown recheck: {'INSUFFICIENT_HISTORY': 1, 'STRUCTURE_MISMATCH': 273, 'BELOW_THRESHOLD': 53, 'VALIDATION_REJECT:H4_STRUCTURE_SAME': 3}
