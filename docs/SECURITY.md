# Security and cybersecurity (Phase F)

This system can be configured to move real funds. Treat it accordingly: defense-in-depth, full
test coverage, and a hardened attack surface are **mandatory before any real-trading
consideration**. This document is the threat model and the hardening + test checklist. The
public vulnerability-disclosure policy is in the repo-root [`SECURITY.md`](../SECURITY.md).

## Assets to protect

```
real Binance API keys (and the funds they can reach)
the dashboard (it can change runtime state)
the SQLite ledger (audit integrity)
the live-trading flags and kill switch
the host/server the bot runs on
```

## Threat model (primary risks)

1. **Key theft / leakage** — keys in logs, reports, DB, git, or screen output.
2. **Unauthorized control** — attacker reaches the dashboard and flips flags, clears the kill
   switch, or forces trades.
3. **Mode confusion** — a real order sent while believing it is testnet, or vice versa.
4. **Withdrawal abuse** — a stolen key with withdrawal permission drains the account.
5. **Injection / supply chain** — SQL injection, malicious/compromised dependency.
6. **Tampering** — silent edits to ledger/audit history hiding losses or actions.
7. **Network exposure** — dashboard reachable on the open internet without TLS/auth.

## Controls (the Phase F checklist)

### Secrets & keys
- Load secrets only from `.env` / OS keyring / environment — never DB, reports, or code.
- Redact secrets in all logs and dashboard output; never echo a key.
- Startup secret-scan: refuse to run if `.env` is committed or world-readable.
- `broker_client.py` keeps **four separate clients**: public data, testnet/paper, real-read,
  real-execute. A real order is structurally impossible in any non-real mode.

### Authentication & access control (`dashboard_auth.py`)
- Strong password hashing (PBKDF2/argon2; current PBKDF2 acceptable, document params).
- Login rate-limiting + lockout; signed, expiring session tokens.
- CSRF protection on every state-changing action.
- Optional 2FA/TOTP. Audited **admin** role required for any control that changes state.
- Default-deny: no control works without an authenticated, authorized session.

### API-key permission model
- Real Binance keys must be **trade-enabled, withdrawal-DISABLED, IP-allowlisted**.
- The healthcheck must verify the key cannot withdraw and is IP-restricted **before** live is
  permitted; otherwise live stays blocked.

### Order / execution safety
- Idempotent `client_order_id`; replay protection on retries/timeouts.
- Server-side notional/qty sanity caps independent of model output.
- A hard **max real-order notional circuit breaker** separate from `RiskManager` (belt and
  braces).

### Input & supply chain
- Parameterize every SQL statement; audit for injection.
- Pin `requirements.txt` with hashes.
- CI + pre-commit gates: `bandit` (static), `pip-audit`/`safety` (deps), `gitleaks`/
  `detect-secrets` (secrets).

### Network / deployment
- Dashboard bound to localhost by default; exposed only behind HTTPS + reverse proxy.
- Firewall + fail2ban guidance; locked-down systemd units.

### Tamper-evidence
- Append-only audit log of every control action, kill-switch event and live-flag change, with
  actor + UTC timestamp.
- A reconciliation mismatch escalates to the kill switch.

## Required tests (adversarial; wired into CI)

```
safety-gate: every real-order path is blocked unless all 4 flags are true
auth: brute-force lockout, session forgery rejection, CSRF rejection
sql-injection: malicious inputs cannot alter queries
kill-switch: active kill switch blocks all execution
reconciliation: a mismatch halts new entries
key-permission: healthcheck refuses live if key can withdraw or is not IP-allowlisted
```

Target high coverage on `risk_manager.py`, `kill_switch.py`, `broker_client.py`,
`execution_engine.py`, `dashboard_auth.py`.

## Verification

```bash
python src/broker_client.py --healthcheck
python src/platform_checks.py          # expect real_orders_blocked_by_default: true
pytest tests/security -q               # adversarial suite green
bandit -r src && pip-audit             # static + dependency scans clean
```
