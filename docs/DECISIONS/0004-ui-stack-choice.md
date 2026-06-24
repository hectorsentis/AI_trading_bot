# ADR 0004 — UI stack: harden Streamlit first, migrate only if needed

Status: Proposed (decide at the start of Phase G)

## Context

The control panel must become a real operator console with audited, re-auth-gated controls
(see [../UI_SPEC.md](../UI_SPEC.md)). The current dashboard is Streamlit + Plotly with read-only
(`mode=ro`) DB access and basic auth (`dashboard_auth.py`).

## Decision (proposed)

Harden and restructure the **existing Streamlit app first**: it is already integrated, renders
the data, and uses read-only DB access. Migrate to a lightweight web stack (e.g. FastAPI +
React/Svelte) **only if** interactivity, role separation, or auth requirements (Phase F) exceed
what Streamlit can do cleanly.

## Consequences

- Fastest path to a usable control panel; preserves existing launch workflow
  (`streamlit run src/dashboard.py`).
- Revisit this ADR if audited controls + CSRF + session management prove awkward in Streamlit;
  at that point a dedicated web backend may be justified.
- Update status to Accepted/Superseded once the call is made.
