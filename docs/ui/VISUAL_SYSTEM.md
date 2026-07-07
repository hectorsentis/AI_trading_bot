# Visual System

## 1. Physical scene and theme choice

The operator uses the cockpit for prolonged sessions on a desktop monitor, often alongside
terminal windows and logs, in moderate or low ambient light. A dark-first theme reduces glare
and aligns with the existing dashboard, while retaining a complete light-theme token set for
accessibility and daytime use.

Dark mode is the recommended default, not the only supported theme.

## 2. Visual character

- restrained and technical
- dense but not cramped
- flat hierarchy with borders and spacing rather than decorative shadows
- minimal accent use
- clear semantic state colors
- no neon crypto aesthetic, gradients, glassmorphism, or ornamental market imagery

## 3. Typography

Recommended final web stack:

- UI family: Geist Sans or a system-ui fallback.
- Numeric/identifier family: Geist Mono or IBM Plex Mono.
- Use tabular numerals for prices, quantities, percentages, time, and PnL.

Scale:

| Role | Size | Weight | Use |
| --- | ---: | ---: | --- |
| Page title | 20 px | 650 | Cockpit/page name |
| Section title | 15 px | 650 | Major panel |
| Subsection | 13 px | 600 | Table group, drawer section |
| Body | 13 px | 400 | Explanations and messages |
| Table | 12 to 13 px | 400/550 | Dense operational rows |
| Label | 11 to 12 px | 550 | Metric and field labels |
| Micro | 10 to 11 px | 500 | Source, age, helper text |

Rules:

- Avoid oversized display typography.
- Do not use uppercase tracking as a repeated section pattern.
- Identifiers and timestamps may use monospace.
- Negative signs, decimal precision, and units must align consistently.

## 4. Spacing and density

Base spacing unit: 4 px.

Allowed scale: 4, 8, 12, 16, 20, 24, 32.

- Application shell gap: 12 px.
- Panel padding: 12 to 16 px.
- Dense table row: 32 px.
- Comfortable table row: 40 px.
- Status bar height: 44 to 48 px.
- Left rail width: 208 px expanded, 56 px collapsed.
- Right rail width: 300 to 340 px.

Cards are used only for real grouping. Metrics in one family share a strip or grid rather than
becoming many floating cards. Nested cards are prohibited.

## 5. Shape system

- Panels: 8 px radius.
- Inputs and compact buttons: 6 px radius.
- Dialogs and drawers: 10 px radius.
- Badges: full pill only because they are status labels.
- Tables: square internal cells; radius only on outer container.

No card radius above 16 px.

## 6. Color strategy

Restrained neutral system with one informational accent and semantic colors.

Suggested dark tokens:

| Token | Value | Role |
| --- | --- | --- |
| `bg.canvas` | `#090C12` | Application background |
| `bg.rail` | `#0C111A` | Navigation and safety rail |
| `bg.panel` | `#101722` | Primary panel |
| `bg.elevated` | `#151E2B` | Drawer, popover, selected row |
| `border.default` | `#263244` | Structural border |
| `border.subtle` | `#1B2533` | Table and separator |
| `text.primary` | `#F2F5F8` | Main text |
| `text.secondary` | `#B5C0CE` | Secondary text |
| `text.muted` | `#8491A3` | Metadata with AA verification |
| `info` | `#4F9CF9` | Selection and information |
| `success` | `#3FBF7F` | Healthy and profit |
| `warning` | `#E6A23C` | Warning, degraded, paused |
| `danger` | `#F05B65` | Loss, error, live risk |
| `critical` | `#D9364A` | Kill switch, hard breach |
| `unknown` | `#6F7B8B` | Unknown or inactive |

Color meanings:

- green: healthy, profit, reconciled
- red: loss, error, live risk, kill switch
- yellow/amber: warning, degraded, paused
- gray: inactive, unknown, unavailable
- blue: informational, selection, running process

Profit/loss and health use separate labels even if they share semantic colors.

## 7. Status badges

Badges include:

- icon or shape
- short state label
- optional age or count

Examples:

- `OK`
- `WARNING`
- `PAUSED`
- `ERROR`
- `LIVE RISK`
- `RECONCILIATION REQUIRED`
- `STALE 4m`

Badge text uses sentence case except fixed operational labels. No status is represented by a
colored dot alone.

## 8. Tables

- Sticky header and pinned identity/status columns.
- Right-align numbers, left-align text, center short statuses.
- Use tabular numerals.
- Default sorting reflects operational risk, not alphabetical convenience.
- Long IDs show a compact prefix and copy affordance.
- Row selection uses one consistent blue-neutral tint.
- Zebra striping is optional and subtle; separators are preferred.
- Critical rows may receive a faint semantic background, not a bright full fill.
- Column visibility and density controls belong in table settings.
- Horizontal scrolling is acceptable; hiding attribution IDs is not.

## 9. Charts

- No 3D charts.
- No donut chart when a ranked bar answers the question better.
- No dual axis unless the relationship is essential and clearly labeled.
- Always show units, time range, source, and last update.
- UTC on time axes.
- Crosshair and tooltip values use tabular numerals.
- Main-screen charts avoid legends with more than five series.
- Missing data is visible as a gap, not interpolated silently.
- Profit green and loss red are reserved for outcomes; market series use neutral/info colors.

## 10. Five-second scan pattern

The eye should travel:

1. top-left mode and bot state
2. top-center safety and reconciliation
3. top-right Action Required count
4. capital/risk strip
5. open-trade protection table
6. right-side critical queue

Techniques:

- stable placement of safety state
- no competing decorative header
- short labels and aligned metrics
- limited semantic colors
- explicit age for stale data
- risk-first row ordering
- no chart above Open Trades unless there are no open trades

## 11. Empty, loading, and error states

### Loading

Use skeleton rows matching the final table and metric geometry. Do not use a central spinner for
the entire cockpit.

### Empty

State what is absent, why it may be absent, and the safe next step. Example:

`No proposals in the selected period. Last prediction: 2026-06-25 12:00 UTC.`

### Error

Keep unaffected panels available. Show:

- failed source
- last successful update
- operational consequence
- recommended recovery command or investigation link

## 12. Responsive rules

- Desktop-first because the task is operational supervision.
- At 1280 px, move the safety rail into a drawer.
- At 1024 px, collapse navigation and stack secondary panels.
- On mobile, preserve monitoring but lock critical controls.
- Tables keep pinned identity columns and horizontal scroll.
- Never reduce critical labels to icon-only without accessible text.

## 13. Accessibility

- WCAG 2.2 AA contrast.
- Visible focus ring on all interactive elements.
- Keyboard operation for navigation, filters, drawers, tables, and dialogs.
- Screen-reader labels include value and unit.
- Charts have textual summaries or table alternatives.
- Reduced motion removes drawer translation and uses near-instant fade.
- Color-blind-safe labels and shapes.

## 14. Anti-clutter rules

- Maximum one primary chart above the fold.
- Maximum one accent color outside semantic state.
- No repeated mini-card grids.
- No raw JSON outside DebugDrawer.
- No more than two levels of panel nesting.
- No decorative icons next to every label.
- No animation for live number updates.
- No auto-rotating content.
