# Documentation index

This folder is the canonical documentation for the autonomous Binance Spot trading
platform in this repository. The operational quick-start lives in the top-level
[`readme.md`](../readme.md); the documents here add architecture, current-state, strategy
and roadmap depth.

## Reading order

1. [ARCHITECTURE.md](ARCHITECTURE.md) — what the system is, module map, data flow, the
   attribution chain, operating modes and the safety model.
2. [CURRENT_STATE.md](CURRENT_STATE.md) — what is actually implemented vs aspirational,
   with file evidence and the confirmed gaps. Read this before changing code.
3. [ROADMAP.md](ROADMAP.md) — **the single source of truth** for sequenced execution
   (Phases A–G). `.agents`, `.skills` and `.codex` all reference this file.
4. [MODELING.md](MODELING.md) — is the model adequate, the model-pool strategy,
   train-new-vs-retrain, native-model plan, anti-leakage discipline.
5. [DATA_AND_FEATURES.md](DATA_AND_FEATURES.md) — current vs target feature families,
   leakage rules, and the external-data integration plan.
6. [SECURITY.md](SECURITY.md) — threat model, secrets, auth, API-key permissions, and
   the hardening + test checklist (Phase F). Disclosure policy is in the repo-root
   [`SECURITY.md`](../SECURITY.md).
7. [UI_SPEC.md](UI_SPEC.md) — control-panel information architecture and visual standards
   (Phase G).
8. [DECISIONS/](DECISIONS/) — architecture decision records (ADRs).

## Source vision

[`initial_roadmap`](initial_roadmap) is the original, exhaustive target vision (2645 lines).
It is preserved as the long-form reference. [ROADMAP.md](ROADMAP.md) organizes that vision
into a sequenced, executable plan and is what day-to-day work should follow.

## Honest framing

This is a research and paper-trading platform. **No profitability is implied or guaranteed.**
A model is only considered useful if its complete trades — executed with the same logic used
in production, net of fees, slippage and risk constraints — keep positive expectancy across
temporal validation, out-of-sample backtest and continuous paper validation.
