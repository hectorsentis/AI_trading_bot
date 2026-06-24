# Data and features

## Current data

- **Source:** Binance REST `klines` only (`download_data.py`, `data_loader.py`). Realtime
  websocket ingestion exists (`realtime_ingestor.py`) but `ENABLE_REALTIME_INGESTION` defaults
  off.
- **Symbols / timeframes:** default `BTCUSDT, ETHUSDT, SOLUSDT`; default `1h` (supports
  `15m, 1h, 4h`).
- **Candle fields stored:** `open, high, low, close, volume, close_time_utc,
  quote_asset_volume, number_of_trades, taker_buy_base_volume, taker_buy_quote_volume,
  provider, ingestion_ts_utc`. (Taker buy volumes and trade count are kept — good, they enable
  buyer/seller-pressure features.)
- **History:** `INITIAL_BACKFILL_DAYS = 365` by default; full mode backfills from `2017-01-01`.
- **External data:** **none today** — no funding, open interest, fear/greed, order book or news.

## Current features

~46–56 features across ~14 families (`config.py:190-246`, computed in `features.py` and
`technical_patterns.py`): returns, range/spread, volatility/ATR, volume, moving-average distance
& slope, MACD, Bollinger, stochastic, price-action/breakouts, candlestick patterns, simple
statistics/z-scores, reversal proxies, short sequences, and cyclical time encodings.

The [`initial_roadmap`](initial_roadmap) sections 10.1–10.23 enumerate ~1000 features across
20+ families. Today's set is a useful core but a small fraction of that target.

## Target feature expansion (Phase C)

Prioritized, leakage-safe additions:

1. **Multi-timeframe** — features from higher TFs (only **closed** candles) aligned to the base
   TF: `rsi_14_4h`, `trend_strength_1d`, `multi_tf_trend_alignment`, etc.
2. **Cross-asset context** — BTC/ETH returns, rolling correlation/beta, relative strength,
   market-breadth proxies. Crypto is highly correlated; BTC context is high-value.
3. **Regime** — trend/volatility/liquidity regime scores and percentiles to let models and the
   allocator condition on market state.
4. **Microstructure proxies from taker data** — buyer/seller imbalance, taker pressure vs price
   divergence, Amihud illiquidity, slippage/market-impact proxies.
5. **Cost/execution features** — fee/slippage/spread estimates, min-notional/step/tick, cost-
   adjusted expected return and risk/reward (needed for net-of-cost objectives).

## External data integration (Phase C, optional)

Only integrate a source if it is **available, versioned, timestamped and leakage-free**:

- Macro/crypto: BTC dominance, total market cap return, **fear/greed index**.
- Derivatives context (even though execution is Spot): **funding rate**, **open interest** change,
  long/short ratio, liquidations.
- Sentiment/news: only with a reliable, timestamped pipeline.

Rules: store each external feed in its own timestamped table; join **as-of** (use only the last
value known at or before `t`); never forward-fill from the future; mark and exclude stale feeds.
New module: `src/external_data.py`.

## Leakage rules (absolute)

```
A feature at t may use only information available up to t.
No feature may look into the future.
No feature may encode the outcome of the trade it is trying to predict.
Higher-timeframe candles may be used only once closed.
Normalization/scaling must never use post-t data.
Never select parameters using the final OOS set.
```

Forbidden as features: future return, future MFE/MAE, `hit_tp_before_sl`, exit price, realized
PnL, or any label used as an input.

## Labels (Phase B expansion)

Today: triple-barrier `triple_barrier_tp_sl_v2`. Target additional label families to support
native models: horizon returns (`future_return_{1,4,8,12,24}h`), MFE/MAE per horizon, return
quantiles, and trade-lifecycle labels (`hit_tp_before_sl`, `realized_return_after_costs`,
`optimal_holding_bars`). Triple-barrier remains valid but must not be the only target.
