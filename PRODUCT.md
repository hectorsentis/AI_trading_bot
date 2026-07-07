# Product

## Register

product

## Users

The primary user is the operator responsible for supervising an autonomous Binance Spot
trading system on a workstation or server. The operator may be reviewing paper or shadow
activity most of the time, and must be able to recognize unsafe conditions, stale data,
reconciliation failures, model degradation, and open exposure without reconstructing state
from logs or raw database rows.

## Product Purpose

The product is an operational control panel for safely supervising the bot's complete
prediction-to-ledger lifecycle. It exists to make current state, capital, risk, model ownership,
trade protection, exchange synchronization, and required operator action understandable in
seconds. It does not exist to prove profitability or to encourage casual live trading.

## Brand Personality

Calm, exact, accountable.

The product should communicate professional trading operations, not speculation. It should feel
like a dependable internal instrument: dense where density helps, restrained in color and motion,
explicit about uncertainty, and severe when safety is compromised.

## Anti-references

- Default Streamlit report layouts with a long vertical stack of unrelated charts.
- Consumer crypto dashboards that use neon decoration, gamified profit signals, or promotional
  language.
- Generic admin templates with identical cards, vague labels, and hidden operational state.
- Grafana used as the primary trading control surface.
- Raw JSON or database dumps presented as the normal operator experience.
- Interfaces where live trading, manual close, or kill-switch recovery appear as casual toggles.

## Design Principles

1. Safety before performance: unsafe state must visually outrank profit and analytics.
2. Operator before chart: every main-screen element must help answer a current operational
   question.
3. Attribution without reconstruction: model, proposal, allocation, trade, order, and fill
   identity must remain visible through drilldowns.
4. Progressive investigation: the cockpit gives immediate answers, while deeper evidence is
   available without overwhelming the first view.
5. Guarded action: monitoring is read-only by default; every state-changing action follows an
   authenticated, confirmed, audited command path.

## Accessibility & Inclusion

- Target WCAG 2.2 AA contrast and keyboard accessibility.
- Never communicate safety, profit/loss, or lifecycle state by color alone.
- Provide reduced-motion behavior; frequent operational interactions should be effectively
  instant.
- Use explicit units and UTC timestamps.
- Keep dense tables readable at 100 to 125 percent browser zoom and support horizontal table
  inspection without truncating critical identifiers.
