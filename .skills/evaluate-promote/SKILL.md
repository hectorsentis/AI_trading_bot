# Skill: evaluate-promote

## When to use
Periodically, to evaluate active models' paper performance and move them through lifecycle states
(paper_active -> paper_validated/paper_degraded/paper_rejected -> real_ready).

## Preconditions
Models have accumulated paper trades (respecting `MIN_PAPER_VALIDATION_DAYS` /
`MIN_PAPER_VALIDATION_TRADES`). Reliable paper PnL requires Phase A (persisted accounting).

## Commands
```bash
python src/paper_model_evaluator.py --evaluate-active
```

## Verification
Models meeting `PAPER_MIN_PROFIT_FACTOR` / `PAPER_MAX_DRAWDOWN` / `PAPER_MIN_WIN_RATE` etc. with
sufficient sample move to `paper_validated` / `real_ready`; failures with sufficient sample move
to `paper_rejected`. Insufficient sample does not force promotion or rejection (unless a hard
safety condition trips). Check `model_lifecycle_events` and `paper_model_metrics`.

## Red-lines
No model becomes `real_active` automatically. `ALLOW_AUTO_PROMOTE_TO_REAL=false` by default; real
also requires all four live flags. Never claim profitability from a short paper window.
