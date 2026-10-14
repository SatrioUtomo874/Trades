# Trading Analysis & AI Reading Report

Generated: 14-10-2026, 20:32 WIB
Session: `20261013-123651-A253`

## 1. Scope and Data Source

- Market: Binance USDⓈ-M Futures
- Execution in this export: Simulation + Real (see real_enabled per trade)
- Price trigger: Binance `aggTrade` last price
- Historical trade records: 6
- Event records: 11
- User notes: 0
- Active setups at export time: 3

## 2. Executive Summary

- Total historical setup records: 6
- Historical filled-and-closed trades: 1
- Historical TP: 1
- Historical SL: 0
- Historical Expired: 5
- Historical Deleted: 0
- Historical fill rate: 16.66666666666666666666666667%
- Historical expiration rate: 83.33333333333333333333333333%
- Historical deletion rate: 0%
- Win Rate: 100%
- Historical PnL sum: 5.646228495809439788266431407%
- Average win: 5.646228495809439788266431407%
- Average loss: 0%
- Average planned RR: 6.016828246028382485682264438
- Average pending duration among completed filled trades: 45m 25s
- Average holding duration among completed filled trades: 20j 36m 37s
- Trailing trade count: 1
- Total trail events: 3

**Win Rate definition:** TP / (TP + SL).
Expired dan Deleted tidak dihitung sebagai win/loss.
Historical metrics above only use records already closed and stored in history.
Active setup data is reported separately and is not counted as historical closed performance.

## 3. Current Active Session

- Active setups: 3
- Pending: 2
- Filled: 1

| Pair | Direction | Status | Entry | SL | TP | Price Exp | Trailing |
|---|---|---|---:|---:|---:|---:|---|
| XANUSDT | SELL | PENDING | 0.010988 | 0.01118 | 0.010086 | 0.010173 | NO |
| NEWTUSDT | SELL | FILLED | 0.04572 | 0.04527 | 0.04366 | 0.04427 | YES |
| BANANAUSDT | SELL | PENDING | 3.786 | 3.853 | 3.45 | 3.55 | NO |

## 4. By Pair

| Pair | Setup | Filled | TP | SL | Expired | Deleted | Win Rate | PnL % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1000BONKUSDT | 1 | 0 | 0 | 0 | 1 | 0 | 0% | 0 |
| AKEUSDT | 1 | 0 | 0 | 0 | 1 | 0 | 0% | 0 |
| ANIMEUSDT | 1 | 0 | 0 | 0 | 1 | 0 | 0% | 0 |
| CTRUSDT | 1 | 1 | 1 | 0 | 0 | 0 | 100% | 5.646228495809439788266431407 |
| RAREUSDT | 1 | 0 | 0 | 0 | 1 | 0 | 0% | 0 |
| SLPUSDT | 1 | 0 | 0 | 0 | 1 | 0 | 0% | 0 |

## 5. By Direction

| Direction | Setup | Filled | TP | SL | Expired | Deleted | Win Rate | PnL % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BUY | 2 | 1 | 1 | 0 | 1 | 0 | 100% | 5.646228495809439788266431407 |
| SELL | 4 | 0 | 0 | 0 | 4 | 0 | 0% | 0 |

## 6. By Strategy

| Strategy | Version | Setup | Filled | TP | SL | Expired | Win Rate | PnL % |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| SMC_VLT_RSI | 2.2.0 | 6 | 1 | 1 | 0 | 5 | 100% | 5.646228495809439788266431407 |

## 7. Event Distribution

| Event | Count |
|---|---:|
| EXPIRED | 5 |
| FILLED | 1 |
| TRAIL | 5 |

## 8. Expiration Reasons

| Reason | Count |
|---|---:|
| Price Exp 0.0005974: retracement thesis dianggap basi jika harga bergerak terlalu jauh tanpa menyentuh entry zone sebelum menembus boundary relevansi. | 1 |
| Price Exp 0.0029579999999999997: retracement thesis dianggap basi jika harga bergerak terlalu jauh tanpa menyentuh entry zone sebelum menembus boundary relevansi. | 1 |
| Price Exp 0.0031479999999999998: retracement thesis dianggap basi jika harga bergerak terlalu jauh tanpa menyentuh entry zone sebelum menembus boundary relevansi. | 1 |
| Price Exp 0.015012: retracement thesis dianggap basi jika harga bergerak terlalu jauh tanpa menyentuh entry zone sebelum menembus boundary relevansi. | 1 |
| Price Exp 0.025667: retracement thesis dianggap basi jika harga bergerak terlalu jauh tanpa menyentuh entry zone sebelum menembus boundary relevansi. | 1 |

## 9. Data Quality / Interpretation Flags

- Records where Price Exp exactly equals TP: 0
- Expired records whose reason text mentions TP: 0
- These are flags for review, not automatic strategy judgments.

## 10. Trade-by-Trade Detail

### Trade 1: RAREUSDT-20261013-144941-C29320

- Pair: RAREUSDT
- Direction: SELL
- Status: CLOSED
- Result: EXPIRED
- Price Now Reference: 0.015474
- Entry: 0.01617
- Reason Entry: RAREUSDT SELL: SELL_PULLBACK. H4 structure BEARISH, liquidity H1 0.016165 dipadukan dengan Fib PRIMARY; refinement BEARISH_FVG. Projected M15 RSI 87.8 pada entry zone.
- Price Exp: 0.01501
- Reason Price Exp: Price Exp 0.015012: retracement thesis dianggap basi jika harga bergerak terlalu jauh tanpa menyentuh entry zone sebelum menembus boundary relevansi.
- SL: 0.01667
- Reason SL: Prediction invalidation berasal dari H4_PROTECTED_HIGH di 0.016643; SL ditempatkan di luar invalidation dan liquidity guard 0.0166412.
- TP: 0.01451
- Reason TP: TP 0.014517 dipilih dari H4_LOWER_LOW dengan RR 3.33; target quality 78.
- Created: 2026-10-13T07:49:41.971126+00:00
- Filled: None
- Closed: 2026-10-13T10:54:36.083009+00:00
- Fill Price: None
- Exit Price: 0.01501
- Result Reason: Price Exp 0.015012: retracement thesis dianggap basi jika harga bergerak terlalu jauh tanpa menyentuh entry zone sebelum menembus boundary relevansi.
- PnL %: None
- Planned RR: 3.32
- Pending Seconds: None
- Holding Seconds: None
- Trailing: NO
- Trail History Count: 0
- Strategy: SMC_VLT_RSI
- Strategy Version: 2.2.0

### Trade 2: CTRUSDT-20261013-151532-E425E1

- Pair: CTRUSDT
- Direction: BUY
- Status: CLOSED
- Result: TRAIL
- Price Now Reference: 0.013748
- Entry: 0.013602
- Reason Entry: CTRUSDT BUY: BUY_PULLBACK. H4 structure BULLISH, liquidity M15 0.0134575 dipadukan dengan Fib PRIMARY; refinement BEARISH_FVG. Projected M15 RSI 43.4 pada entry zone.
- Price Exp: 0.014353
- Reason Price Exp: Price Exp 0.014353: retracement thesis dianggap basi jika harga bergerak terlalu jauh tanpa menyentuh entry zone sebelum menembus boundary relevansi.
- SL: 0.014371
- Reason SL: Prediction invalidation berasal dari H4_SELL_SIDE_LIQUIDITY di 0.013332; SL ditempatkan di luar invalidation dan liquidity guard 0.0133627.
- TP: 0.015209
- Reason TP: TP 0.015209 dipilih dari H4_HIGHER_HIGH dengan RR 5.17; target quality 88.
- Created: 2026-10-13T08:15:32.486672+00:00
- Filled: 2026-10-13T09:00:58.101082+00:00
- Closed: 2026-10-14T05:37:36.010519+00:00
- Fill Price: 0.013602
- Exit Price: 0.01437
- Result Reason: SL trailing tercapai; profit terkunci. Prediction invalidation berasal dari H4_SELL_SIDE_LIQUIDITY di 0.013332; SL ditempatkan di luar invalidation dan liquidity guard 0.0133627.
- PnL %: 5.646228495809439788266431407
- Planned RR: 2.089726918075422626788036411
- Pending Seconds: 2725.61441
- Holding Seconds: 74197.909437
- Trailing: YES
- Trail History Count: 3
- Strategy: SMC_VLT_RSI
- Strategy Version: 2.2.0

### Trade 3: AKEUSDT-20261014-110612-5A3DB9

- Pair: AKEUSDT
- Direction: BUY
- Status: CLOSED
- Result: EXPIRED
- Price Now Reference: 0.025406
- Entry: 0.023195
- Reason Entry: AKEUSDT BUY: BUY_PULLBACK. H4 structure BULLISH, liquidity M15 0.0231772 dipadukan dengan Fib DEEP_0.618_0.786; refinement BULLISH_FVG. Projected M15 RSI 17.5 pada entry zone.
- Price Exp: 0.025667
- Reason Price Exp: Price Exp 0.025667: retracement thesis dianggap basi jika harga bergerak terlalu jauh tanpa menyentuh entry zone sebelum menembus boundary relevansi.
- SL: 0.02238
- Reason SL: Prediction invalidation berasal dari H4_PROTECTED_LOW di 0.022432; SL ditempatkan di luar invalidation dan liquidity guard 0.0224408.
- TP: 0.025803
- Reason TP: TP 0.025803 dipilih dari H4_HIGHER_HIGH dengan RR 3.20; target quality 84.
- Created: 2026-10-14T04:06:12.686827+00:00
- Filled: None
- Closed: 2026-10-14T06:19:00.187806+00:00
- Fill Price: None
- Exit Price: 0.025667
- Result Reason: Price Exp 0.025667: retracement thesis dianggap basi jika harga bergerak terlalu jauh tanpa menyentuh entry zone sebelum menembus boundary relevansi.
- PnL %: None
- Planned RR: 3.2
- Pending Seconds: None
- Holding Seconds: None
- Trailing: NO
- Trail History Count: 0
- Strategy: SMC_VLT_RSI
- Strategy Version: 2.2.0

### Trade 4: ANIMEUSDT-20261013-192453-964978

- Pair: ANIMEUSDT
- Direction: SELL
- Status: CLOSED
- Result: EXPIRED
- Price Now Reference: 0.002986
- Entry: 0.003096
- Reason Entry: ANIMEUSDT SELL: SELL_PULLBACK_RSI_DIVERGENCE. H4 structure BEARISH, liquidity M15 0.0030955 dipadukan dengan Fib DEEP_0.618_0.786; refinement BEARISH_FVG. Projected M15 RSI 68.8 pada entry zone.
- Price Exp: 0.002958
- Reason Price Exp: Price Exp 0.0029579999999999997: retracement thesis dianggap basi jika harga bergerak terlalu jauh tanpa menyentuh entry zone sebelum menembus boundary relevansi.
- SL: 0.003106
- Reason SL: Prediction invalidation berasal dari BEARISH_SWEEP_EXTREME di 0.003102; SL ditempatkan di luar invalidation.
- TP: 0.002899
- Reason TP: TP 0.002899 dipilih dari H4_LOWER_LOW dengan RR 19.70; target quality 80.
- Created: 2026-10-13T12:24:53.390471+00:00
- Filled: None
- Closed: 2026-10-14T12:30:10.778806+00:00
- Fill Price: None
- Exit Price: 0.002958
- Result Reason: Price Exp 0.0029579999999999997: retracement thesis dianggap basi jika harga bergerak terlalu jauh tanpa menyentuh entry zone sebelum menembus boundary relevansi.
- PnL %: None
- Planned RR: 19.7
- Pending Seconds: None
- Holding Seconds: None
- Trailing: NO
- Trail History Count: 0
- Strategy: SMC_VLT_RSI
- Strategy Version: 2.2.0

### Trade 5: 1000BONKUSDT-20261013-190656-6785AD

- Pair: 1000BONKUSDT
- Direction: SELL
- Status: CLOSED
- Result: EXPIRED
- Price Now Reference: 0.003172
- Entry: 0.003323
- Reason Entry: 1000BONKUSDT SELL: SELL_PULLBACK_RSI_DIVERGENCE. H4 structure BEARISH, liquidity H1 0.003322 dipadukan dengan Fib PRIMARY; refinement BEARISH_FVG. Projected M15 RSI 73.6 pada entry zone.
- Price Exp: 0.003148
- Reason Price Exp: Price Exp 0.0031479999999999998: retracement thesis dianggap basi jika harga bergerak terlalu jauh tanpa menyentuh entry zone sebelum menembus boundary relevansi.
- SL: 0.003364
- Reason SL: Prediction invalidation berasal dari BEARISH_SWEEP_EXTREME di 0.003359; SL ditempatkan di luar invalidation.
- TP: 0.003113
- Reason TP: TP 0.003113 dipilih dari H4_LOWER_LOW dengan RR 5.12; target quality 82.
- Created: 2026-10-13T12:06:56.583807+00:00
- Filled: None
- Closed: 2026-10-14T12:30:28.581721+00:00
- Fill Price: None
- Exit Price: 0.003148
- Result Reason: Price Exp 0.0031479999999999998: retracement thesis dianggap basi jika harga bergerak terlalu jauh tanpa menyentuh entry zone sebelum menembus boundary relevansi.
- PnL %: None
- Planned RR: 5.121951219512195121951219512
- Pending Seconds: None
- Holding Seconds: None
- Trailing: NO
- Trail History Count: 0
- Strategy: SMC_VLT_RSI
- Strategy Version: 2.2.0

### Trade 6: SLPUSDT-20261013-192452-24C126

- Pair: SLPUSDT
- Direction: SELL
- Status: CLOSED
- Result: EXPIRED
- Price Now Reference: 0.000601
- Entry: 0.0006312
- Reason Entry: SLPUSDT SELL: SELL_PULLBACK_RSI_DIVERGENCE. H4 structure BEARISH, liquidity H1 0.0006311 dipadukan dengan Fib SECONDARY; refinement BEARISH_FVG. Projected M15 RSI 76.0 pada entry zone.
- Price Exp: 0.0005974
- Reason Price Exp: Price Exp 0.0005974: retracement thesis dianggap basi jika harga bergerak terlalu jauh tanpa menyentuh entry zone sebelum menembus boundary relevansi.
- SL: 0.0006439
- Reason SL: Prediction invalidation berasal dari BEARISH_SWEEP_EXTREME di 0.0006431; SL ditempatkan di luar invalidation.
- TP: 0.0005973
- Reason TP: TP 0.0005973 dipilih dari H4_LOWER_LOW dengan RR 2.67; target quality 84.
- Created: 2026-10-13T12:24:52.381800+00:00
- Filled: None
- Closed: 2026-10-14T13:19:55.777553+00:00
- Fill Price: None
- Exit Price: 0.0005974
- Result Reason: Price Exp 0.0005974: retracement thesis dianggap basi jika harga bergerak terlalu jauh tanpa menyentuh entry zone sebelum menembus boundary relevansi.
- PnL %: None
- Planned RR: 2.669291338582677165354330709
- Pending Seconds: None
- Holding Seconds: None
- Trailing: NO
- Trail History Count: 0
- Strategy: SMC_VLT_RSI
- Strategy Version: 2.2.0

## 11. User Notes (/catatan)

Belum ada catatan.

## 12. AI-Readable Conclusions

1. Dari 6 historical setup, 5 berakhir EXPIRED dan 1 berakhir sebagai filled trade yang kemudian closed.
2. Expiration rate historis adalah 83.33333333333333333333333333%; angka ini menggambarkan frekuensi setup berakhir sebelum menjadi trade yang selesai, bukan win/loss rate.
3. Historical filled sample berjumlah 1; TP=1 dan SL=0. Win Rate historis menurut definisi sistem adalah 100%.
4. Historical PnL sum hanya berasal dari record yang memiliki PnL, dengan nilai 5.646228495809439788266431407%.
5. Active session saat export memiliki 3 setup (2 PENDING, 1 FILLED); setup aktif tidak dimasukkan ke historical closed performance.
6. Total trailing event yang tercatat adalah 3 dan jumlah trade historis yang memiliki trailing adalah 1.
7. Direction historis yang muncul: BUY, SELL.
8. Pair historis yang muncul: 1000BONKUSDT, AKEUSDT, ANIMEUSDT, CTRUSDT, RAREUSDT, SLPUSDT.
9. Reason EXPIRED yang paling sering tercatat adalah 'Price Exp 0.0005974: retracement thesis dianggap basi jika harga bergerak terlalu jauh tanpa menyentuh entry zone sebelum menembus boundary relevansi.' (1 record).
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

- Kandidat: {'EXPIRED': 9, 'CLOSED': 1, 'FILLED': 1, 'PENDING': 2, 'REJECTED_QUALITY': 284, 'WAITING_PRICE': 3}
- Funnel scan terakhir: {'universe': 462, 'usable_data': 0, 'structure_analyzed': 0, 'direction_thesis': 0, 'candidates_evaluated': 0, 'hard_rejects': 0, 'quality_rejects': 0, 'waiting_price': 0, 'waiting_trigger': 0, 'ready': 0, 'validation_pass': 0, 'validation_reject': 0, 'validation_error': 0, 'risk_rejects': 0, 'pending_created': 0, 'structure_mismatch': 0, 'data_unavailable': 0, 'data_stale': 0, 'budget_stopped': 0, 'memory_skipped': 0, 'validation_skipped_topn': 0, 'rate_limited': 0, 'sweep_covered': 453, 'sweep_total': 453, 'sim_margin_blocked': 0, 'errors': {}}
- R-metrik histori: n=1 • E=+2.40R • win 100% • PF - • DD 0.00R  ⚠️ sampel kecil
- Cooldown recheck: {'INSUFFICIENT_HISTORY': 1, 'BELOW_THRESHOLD': 440}
