# Skill: run-tests

## When to use
After any code change, and in CI. The test harness is being established starting with Phase A.

## Preconditions
`pytest` installed. Tests run against a fixture/temp SQLite DB, never the operational DB.

## Commands
```bash
pytest -q                                # full suite
pytest tests/test_paper_loop_smoke.py -q # Phase A smoke test
pytest --cov=src --cov-report=term-missing
```

## Verification
- Phase A smoke test: a local_paper `--run-once` on a fixture DB yields an
  `OPEN -> CLOSING -> CLOSED` trade with non-null `realized_pnl_usdt` and a persisted snapshot;
  a second run carries starting equity over.
- Security suite (Phase F) green; coverage rising on safety-critical modules.

## Red-lines
Never point tests at the real/operational DB or send real/testnet orders. Keep tests
deterministic (seed data, fixed timestamps).
