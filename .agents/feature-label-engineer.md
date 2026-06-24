# Agent: Feature & Label Engineer

## Mission
Generate versioned, leakage-free features and labels that let models learn direction, magnitude,
uncertainty, favorable/adverse excursion, horizon and risk/reward — without ever seeing the
future.

## Owns
`features.py`, `technical_patterns.py`, `labels.py`, `feature_store.py`. Tables: `features`
(and label columns).

## Invariants / red-lines
- A feature at `t` uses only information available up to `t`. Higher-TF candles only once closed.
- No normalization using post-`t` data; never use a label/outcome as an input.
- Features computed per symbol; never mix rolling windows across symbols.
- Version every feature set (current `FEATURE_VERSION = v3_symbol_pattern_regime`,
  `LABEL_VERSION = triple_barrier_tp_sl_v2`); bump on change.

## Current state
~46–56 features / ~14 families; labels are triple-barrier TP/SL only.

## Backlog (see docs/ROADMAP.md, docs/DATA_AND_FEATURES.md)
- Phase B: add label families — horizon returns, MFE/MAE, quantiles, trade-lifecycle labels.
- Phase C: multi-timeframe, cross-asset (BTC/ETH), regime, microstructure, cost/execution
  features; keep each leakage-safe and versioned.

## Acceptance criteria
`feature_store.py` rebuilds deterministically; a leakage check confirms no future-derived column;
new feature/label families are versioned and documented.
