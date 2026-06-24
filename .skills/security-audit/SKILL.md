# Skill: security-audit

## When to use
Before any real-trading consideration, and routinely in CI. Confirms the system cannot leak keys,
cannot be driven by an unauthorized user, and cannot send a real order by accident.

## Preconditions
Phase F controls in progress/landed. Tools available: `bandit`, `pip-audit`, a secrets scanner
(`gitleaks`/`detect-secrets`).

## Commands
```bash
python src/broker_client.py --healthcheck
python src/platform_checks.py            # expect real_orders_blocked_by_default: true
pytest tests/security -q                 # adversarial safety/auth/injection/kill-switch suite
bandit -r src                            # static analysis
pip-audit                                # dependency vulnerabilities
detect-secrets scan                      # no committed secrets
```

## Verification
- Every real-order path blocked unless all four live flags are set and `DRY_RUN=false`.
- Auth resists brute-force, session forgery and CSRF.
- SQL paths parameterized; reconciliation failure halts trading.
- Real-key healthcheck refuses live if the key can withdraw or is not IP-allowlisted.
- Scanners clean.

## Red-lines
Do not weaken a safety gate to make a test pass. Never print or commit a secret. See
[docs/SECURITY.md](../../docs/SECURITY.md).
