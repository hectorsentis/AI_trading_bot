# Security policy

This project can be configured to place real orders on a real Binance account. Security is
treated as a first-class requirement. The full threat model, hardening controls and test
checklist live in [docs/SECURITY.md](docs/SECURITY.md).

## Reporting a vulnerability

If you discover a security issue (key handling, authentication, order-safety bypass, injection,
or anything that could risk funds), please report it privately to the maintainer rather than
opening a public issue. Include reproduction steps and impact. You will get an acknowledgment
and a remediation timeline.

## Safe defaults

Out of the box the system cannot send a real order:

```
DRY_RUN=true
ENABLE_LIVE_TRADING=false
ENABLE_REAL_ORDER_EXECUTION=false
ENABLE_REAL_BINANCE_ACCOUNT=false
KILL_SWITCH_ENABLED=true
```

Never commit `.env`, API keys/secrets, the SQLite database, model artifacts, logs or reports.
Real Binance API keys must be **trade-enabled, withdrawal-disabled and IP-allowlisted**.
