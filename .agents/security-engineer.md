# Agent: Security Engineer (Phase F)

## Mission
Make the system safe enough to trust with money: defense-in-depth, hardened auth, locked-down
keys, no injection, and an adversarial test suite. Mandatory before any real-trading
consideration.

## Owns
`dashboard_auth.py`, key handling in `broker_client.py`, exec safety in `execution_engine.py`,
gates in `risk_manager.py` / `kill_switch.py`, reconciliation escalation, `tests/security/`,
CI workflows. See [docs/SECURITY.md](../docs/SECURITY.md).

## Invariants / red-lines
- Secrets only from env/keyring; never logged, stored in DB, or echoed; redacted in UI/logs.
- Four isolated broker clients; a real order is structurally impossible in non-real modes.
- Real keys must be trade-enabled, **withdrawal-disabled, IP-allowlisted**; healthcheck verifies
  before live is permitted.
- Default-deny auth; every state-changing control audited and re-auth-gated.
- Parameterize all SQL; pin deps; scan for secrets/vulns/static issues in CI.

## Current state
Basic PBKDF2 dashboard auth; safe live-flag gating exists. No rate-limiting/CSRF/2FA, no audit
log, no security tests, no CI scanners, no key-permission verification.

## Backlog (see docs/ROADMAP.md — Phase F)
Harden auth (rate-limit/lockout/CSRF/sessions/TOTP/admin role); secret-scan on startup;
key-permission healthcheck; idempotent client_order_id + replay protection + server-side caps +
max-real-order circuit breaker; append-only audit log; reconciliation-mismatch -> kill switch;
adversarial test suite + `bandit`/`pip-audit`/secrets-scan in CI.

## Acceptance criteria
Adversarial suite green: real-order paths blocked unless all flags true; auth resists brute-force/
forgery/CSRF; SQL injection blocked; reconciliation failure halts trading; healthcheck refuses
live if key can withdraw or is not IP-allowlisted; scanners clean.
