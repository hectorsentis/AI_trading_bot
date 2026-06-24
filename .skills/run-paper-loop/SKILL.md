# Skill: run-paper-loop

## When to use
Run the trading loop in paper mode (local-paper or Binance testnet) to accumulate real,
attributable paper performance.

## Preconditions
- Accepted/active model(s) in the registry.
- Recent data/features. For testnet: `BINANCE_TESTNET_API_KEY/SECRET` set,
  `ENABLE_TESTNET_PAPER_TRADING=true`. Safe defaults keep real trading off.

## Commands
```bash
# Single iteration
python src/trading_bot.py --mode paper --paper-mode per-model --run-once

# Continuous loop
python src/trading_bot.py --mode paper --paper-mode per-model --loop

# Controlled testnet order probe (refuses if DRY_RUN=false or any real flag is on)
python src/paper_demo_probe.py --symbol BTCUSDT --timeframe 1h
```

## Verification
A trade traces prediction -> proposal -> allocation -> trade -> order -> fill. After Phase A:
trades reach `OPEN -> CLOSING -> CLOSED` with booked `realized_pnl_usdt`, and
`portfolio_snapshots`/`account_snapshots` persist across runs. Orders are `dry_run=1` or
`account_mode IN ('local_paper','testnet_paper')`, never `real`.

## Red-lines
Never bypass risk/kill-switch/reconciliation; never fall back to real on testnet failure.
