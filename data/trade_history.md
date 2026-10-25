# Trading Journal

---

## DOGEUSDT — SELL

Trade ID: `DOGEUSDT-20261024-110350-C8B90A`

### Setup

- Price Now Reference: 0.07893
- Entry: 0.07952
- Reason Entry: DOGEUSDT SELL: SELL_PULLBACK. H4 structure BEARISH, liquidity H1 0.07957 dipadukan dengan Fib SECONDARY; refinement BULLISH_BREAKER. Projected M15 RSI 66.3 pada entry zone.
- Price Exp: 0.0776
- Reason Price Exp: Price Exp 0.0776: retracement thesis dianggap basi jika harga bergerak terlalu jauh tanpa menyentuh entry zone sebelum menembus boundary relevansi.
- SL: 0.08032
- Reason SL: Prediction invalidation berasal dari H1_PROTECTED_HIGH di 0.07958; SL ditempatkan di luar invalidation.
- TP: 0.07744
- Reason TP: TP 0.07744 dipilih dari H4_LOWER_LOW dengan RR 2.60; target quality 92.

### Management

Tidak ada trailing.

### Result

- Result: DELETED
- Exit Price: -
- Result Reason: Order LIMIT sudah tidak aktif di Binance (CANCELED); dibatalkan manual atau oleh Binance.
- PnL: -
- Created: 2026-10-24T04:03:50.493093+00:00
- Filled: -
- Closed: 2026-10-24T07:54:22.678578+00:00
- Strategy: SMC_VLT_RSI v3.0.0

---

## WCTUSDT — BUY

Trade ID: `WCTUSDT-20261025-174008-ED1892`

### Setup

- Price Now Reference: 0.03538
- Entry: 0.03538
- Reason Entry: M5 closed-candle confirmation score 77.53/72.00; POI_TOUCH_CURRENT_BAR;DIRECTIONAL_STRUCTURE_BREAK;TAKER_FLOW_SUPPORT;RELATIVE_VOLUME_SUPPORT;OPPOSING_MOVE_STALL_OR_SWEEP; entry modeled at closed M5 price 0.03538.
- Price Exp: 0.03657
- Reason Price Exp: Reference ekspansi 0.0365658 dari target liquidity M5_BUY_SIDE_LIQUIDITY; bukan target TP.
- SL: 0.03487
- Reason SL: SL struktur M15_PROTECTED_LOW di luar anchor 0.03493 dengan buffer 5.43172e-05; risk 0.55 ATR H1.
- TP: 0.03678
- Reason TP: TP ke liquidity M5_BUY_SIDE_LIQUIDITY di 0.036775; sumber EQUAL_LEVELS; RR gross 2.77R (minimum 2.0R, tanpa batas maksimum).

### Management

Tidak ada trailing.

### Result

- Result: SL
- Exit Price: 0.03487
- Result Reason: SL struktur M15_PROTECTED_LOW di luar anchor 0.03493 dengan buffer 5.43172e-05; risk 0.55 ATR H1.
- PnL: -1.441492368569813453928773318%
- Created: 2026-10-25T10:40:08.293969+00:00
- Filled: 2026-10-25T10:40:12.074151+00:00
- Closed: 2026-10-25T12:31:20.576186+00:00
- Strategy: SMC_VLT_RSI v5.4.0-CONFIRMED-STRUCTURAL-RR2-M15-STRUCTURE-TRAIL
