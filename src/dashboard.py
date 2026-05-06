"""Professional operational dashboard for the Binance Spot AI Trading Bot.

Run:
    streamlit run src/dashboard.py
"""
from __future__ import annotations

import itertools
import json
from html import escape
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import dashboard_auth as auth
import dashboard_controls as controls
import dashboard_data as data


_PLOTLY_KEY_COUNTER = itertools.count()


STATUS_COLORS = {
    "Running": "#22c55e",
    "Stopped": "#f97316",
    "Error": "#ef4444",
    "Unknown": "#94a3b8",
    "LIVE TRADING": "#ef4444",
    "PAPER TRADING": "#22c55e",
    "DRY RUN": "#38bdf8",
    "RESEARCH": "#a855f7",
}


def fmt_money(value: Any) -> str:
    try:
        if value is None or pd.isna(value):
            return "N/A"
        return f"${float(value):,.2f}"
    except Exception:
        return "N/A"


def fmt_pct(value: Any) -> str:
    try:
        if value is None or pd.isna(value):
            return "N/A"
        return f"{float(value) * 100:,.2f}%"
    except Exception:
        return "N/A"


def numeric_or_none(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def boolish(value: Any) -> bool:
    try:
        if value is None or pd.isna(value):
            return False
    except Exception:
        pass
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def fmt_num(value: Any, digits: int = 2) -> str:
    try:
        if value is None or pd.isna(value):
            return "N/A"
        return f"{float(value):,.{digits}f}"
    except Exception:
        return "N/A"


def pnl_color(value: Any) -> str:
    number = numeric_or_none(value)
    if number is None:
        return "color: #cbd5e1"
    if number > 0:
        return "color: #86efac; font-weight: 700"
    if number < 0:
        return "color: #fecaca; font-weight: 700"
    return "color: #cbd5e1"


def status_color(value: Any) -> str:
    text = str(value).lower()
    if any(x in text for x in ["active", "accepted", "validated", "ready", "open", "running"]):
        return "color: #86efac; font-weight: 700"
    if any(x in text for x in ["rejected", "quarantine", "error", "failed", "paused", "degraded"]):
        return "color: #fecaca; font-weight: 700"
    return "color: #cbd5e1"


def simple_display_df(df: pd.DataFrame, *, pct_cols: list[str] | None = None, money_cols: list[str] | None = None, digits: int = 3) -> pd.DataFrame:
    """Return a plain dataframe for Streamlit; no Styler, no complex rendering."""
    if df.empty:
        return df
    out = df.copy()
    for col in pct_cols or []:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").map(lambda v: "N/A" if pd.isna(v) else f"{v * 100:,.2f}%")
    for col in money_cols or []:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").map(lambda v: "N/A" if pd.isna(v) else f"${v:,.2f}")
    for col in out.select_dtypes(include=["float", "float64", "float32"]).columns:
        if col not in set((pct_cols or []) + (money_cols or [])):
            out[col] = out[col].round(digits)
    return out


def show_simple_df(st, df: pd.DataFrame, *, height: int = 280, empty: str = "No data available.", pct_cols: list[str] | None = None, money_cols: list[str] | None = None) -> None:
    if df.empty:
        st.info(df.attrs.get("message") or empty)
        return
    st.dataframe(simple_display_df(df, pct_cols=pct_cols, money_cols=money_cols), use_container_width=True, height=height, hide_index=True)


def inject_css(st) -> None:
    st.markdown(
        """
        <style>
        :root { --bg:#08111f; --panel:#0f172a; --muted:#94a3b8; --line:rgba(148,163,184,.22); }
        .stApp { background: linear-gradient(180deg, #07111f 0%, #0b1220 48%, #111827 100%); color:#e5e7eb; }
        .block-container { padding-top: 1rem; padding-bottom: 2rem; max-width: 1680px; }
        [data-testid="stSidebar"] { background:#07111f; border-right:1px solid rgba(148,163,184,.18); }
        div[data-testid="stMetric"] {
            background: linear-gradient(135deg, rgba(15,23,42,.98), rgba(17,24,39,.92));
            border:1px solid rgba(148,163,184,.22); border-radius:16px; padding:14px 16px;
            box-shadow:0 14px 35px rgba(0,0,0,.22);
        }
        div[data-testid="stMetricLabel"] { color:#cbd5e1; font-weight:700; }
        div[data-testid="stMetricValue"] { color:#f8fafc; font-size:1.45rem; }
        .hero {
            position:relative; padding:20px 24px; border-radius:22px; margin-bottom:14px;
            border:1px solid rgba(148,163,184,.24);
            background:
                radial-gradient(circle at top left, rgba(56,189,248,.18), transparent 32%),
                radial-gradient(circle at top right, rgba(34,197,94,.16), transparent 28%),
                linear-gradient(135deg, #020617 0%, #0f172a 54%, #111827 100%);
            box-shadow:0 20px 60px rgba(0,0,0,.32);
        }
        .hero-live {
            border:1px solid rgba(239,68,68,.70);
            box-shadow:0 0 0 2px rgba(239,68,68,.16), 0 20px 60px rgba(127,29,29,.32);
        }
        .title { margin:0; font-size:2.15rem; line-height:1.1; letter-spacing:-.035em; color:#f8fafc; }
        .subtitle { margin:.4rem 0 0 0; color:#cbd5e1; }
        .pill { display:inline-flex; align-items:center; gap:6px; padding:5px 10px; border-radius:999px;
                font-size:.78rem; font-weight:800; margin:3px 4px 0 0; border:1px solid rgba(148,163,184,.32); }
        .pill-green { background:rgba(34,197,94,.13); color:#86efac; border-color:rgba(34,197,94,.45); }
        .pill-red { background:rgba(239,68,68,.15); color:#fecaca; border-color:rgba(239,68,68,.55); }
        .pill-blue { background:rgba(56,189,248,.13); color:#bae6fd; border-color:rgba(56,189,248,.45); }
        .pill-orange { background:rgba(249,115,22,.13); color:#fed7aa; border-color:rgba(249,115,22,.45); }
        .pill-gray { background:rgba(148,163,184,.12); color:#cbd5e1; border-color:rgba(148,163,184,.32); }
        .panel {
            background:rgba(15,23,42,.72); border:1px solid rgba(148,163,184,.18);
            border-radius:18px; padding:14px 16px; margin:8px 0 14px 0;
        }
        .small { color:#94a3b8; font-size:.86rem; }
        .warnbox { border-left:4px solid #f59e0b; padding:10px 12px; background:rgba(245,158,11,.10); border-radius:10px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def pill(text: str, css: str) -> str:
    return f'<span class="pill {css}">{escape(text)}</span>'


def status_pill(text: str) -> str:
    if text == "Running":
        return pill("Running", "pill-green")
    if text == "Error":
        return pill("Error", "pill-red")
    if text == "Stopped":
        return pill("Stopped", "pill-orange")
    return pill("Unknown", "pill-gray")


def mode_pill(mode: str, real_possible: bool) -> str:
    if mode == "LIVE TRADING" or real_possible:
        return pill("LIVE TRADING - REAL ORDERS POSSIBLE", "pill-red")
    if mode == "PAPER TRADING":
        return pill("PAPER TRADING", "pill-green")
    if mode == "DRY RUN":
        return pill("DRY RUN", "pill-blue")
    return pill(mode or "Unknown mode", "pill-gray")


def show_df(st, df: pd.DataFrame, *, height: int = 320, empty: str = "No data available.") -> None:
    if df.empty:
        msg = df.attrs.get("message") or empty
        st.info(msg)
        return
    display = df.copy()
    for col in display.columns:
        if str(col).endswith("_json"):
            display[col] = display[col].apply(parse_jsonish_for_display)
        if display[col].map(lambda v: isinstance(v, (dict, list))).any():
            display[col] = display[col].apply(humanize_nested_value)
    st.dataframe(display, use_container_width=True, height=height)


def show_colored_df(
    st,
    df: pd.DataFrame,
    *,
    height: int = 320,
    empty: str = "No data available.",
    money_cols: list[str] | None = None,
    pct_cols: list[str] | None = None,
    pnl_cols: list[str] | None = None,
    status_cols: list[str] | None = None,
) -> None:
    """Compact dataframe renderer with operational green/red styling."""
    if df.empty:
        st.info(df.attrs.get("message") or empty)
        return
    display = df.copy()
    money_cols = [c for c in (money_cols or []) if c in display.columns]
    pct_cols = [c for c in (pct_cols or []) if c in display.columns]
    pnl_cols = [c for c in (pnl_cols or []) if c in display.columns]
    status_cols = [c for c in (status_cols or []) if c in display.columns]
    for col in set(money_cols + pct_cols + pnl_cols):
        display[col] = pd.to_numeric(display[col], errors="coerce")
    formatter: dict[str, Any] = {}
    formatter.update({c: "${:,.2f}" for c in money_cols + pnl_cols})
    formatter.update({c: "{:,.2%}" for c in pct_cols})
    styled = display.style.format(formatter, na_rep="N/A")
    if pnl_cols:
        try:
            styled = styled.map(pnl_color, subset=pnl_cols)
        except AttributeError:  # pandas < 2.1
            styled = styled.applymap(pnl_color, subset=pnl_cols)
    if status_cols:
        try:
            styled = styled.map(status_color, subset=status_cols)
        except AttributeError:  # pandas < 2.1
            styled = styled.applymap(status_color, subset=status_cols)
    st.dataframe(styled, use_container_width=True, height=height)


def render_selectable_colored_df(
    st,
    df: pd.DataFrame,
    *,
    key: str,
    height: int = 320,
    empty: str = "No data available.",
    money_cols: list[str] | None = None,
    pct_cols: list[str] | None = None,
    pnl_cols: list[str] | None = None,
    status_cols: list[str] | None = None,
) -> int | None:
    """Render a selectable dataframe and return the selected row position."""
    if df.empty:
        st.info(df.attrs.get("message") or empty)
        return None
    display = df.copy()
    money_cols = [c for c in (money_cols or []) if c in display.columns]
    pct_cols = [c for c in (pct_cols or []) if c in display.columns]
    pnl_cols = [c for c in (pnl_cols or []) if c in display.columns]
    status_cols = [c for c in (status_cols or []) if c in display.columns]
    for col in set(money_cols + pct_cols + pnl_cols):
        display[col] = pd.to_numeric(display[col], errors="coerce")
    formatter: dict[str, Any] = {}
    formatter.update({c: "${:,.2f}" for c in money_cols + pnl_cols})
    formatter.update({c: "{:,.2%}" for c in pct_cols})
    styled = display.style.format(formatter, na_rep="N/A")
    if pnl_cols:
        try:
            styled = styled.map(pnl_color, subset=pnl_cols)
        except AttributeError:
            styled = styled.applymap(pnl_color, subset=pnl_cols)
    if status_cols:
        try:
            styled = styled.map(status_color, subset=status_cols)
        except AttributeError:
            styled = styled.applymap(status_color, subset=status_cols)
    try:
        event = st.dataframe(
            styled,
            use_container_width=True,
            height=height,
            key=key,
            on_select="rerun",
            selection_mode="single-row",
        )
        rows = getattr(getattr(event, "selection", None), "rows", None)
        if rows:
            return int(rows[0])
    except Exception:
        try:
            event = st.dataframe(
                display,
                use_container_width=True,
                height=height,
                key=f"{key}_plain",
                on_select="rerun",
                selection_mode="single-row",
            )
            rows = getattr(getattr(event, "selection", None), "rows", None)
            if rows:
                return int(rows[0])
        except Exception:
            st.dataframe(display, use_container_width=True, height=height)
    return None


def parse_jsonish_for_display(value: Any) -> Any:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if isinstance(value, (dict, list)):
        return value
    text = str(value).strip()
    if not text:
        return ""
    try:
        return json.loads(text)
    except Exception:
        return text


def humanize_nested_value(value: Any) -> str:
    """Display nested dict/list payloads without raw JSON widgets."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if isinstance(value, dict):
        parts = []
        for k, v in value.items():
            if isinstance(v, dict):
                inner = ", ".join(f"{ik}={iv}" for ik, iv in list(v.items())[:5])
                parts.append(f"{k}: {inner}")
            elif isinstance(v, list):
                parts.append(f"{k}: {', '.join(map(str, v[:5]))}")
            else:
                parts.append(f"{k}: {v}")
        return "; ".join(parts[:8])
    if isinstance(value, list):
        return ", ".join(map(str, value[:8]))
    return str(value)


def flatten_nested_metrics(payload: Any) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def walk(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            for k, v in value.items():
                walk(f"{prefix}.{k}" if prefix else str(k), v)
        elif isinstance(value, list):
            rows.append({"metric": prefix, "value": ", ".join(map(str, value[:10]))})
        else:
            rows.append({"metric": prefix, "value": value})

    walk("", payload if isinstance(payload, dict) else {})
    return pd.DataFrame(rows)


def render_key_value_table(st, title: str, payload: dict[str, Any], *, height: int = 260) -> None:
    st.markdown(f"#### {title}")
    rows = [{"field": str(k), "value": humanize_nested_value(v)} for k, v in payload.items()]
    show_df(st, pd.DataFrame(rows), height=height, empty="No data available.")


def filter_df(st, df: pd.DataFrame, key: str, columns: list[str] | None = None) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    columns = columns or ["model_id", "symbol", "timeframe", "account_mode", "status"]
    cols = st.columns(min(4, len(columns)))
    for idx, col in enumerate(columns):
        if col in out.columns:
            values = sorted(out[col].dropna().astype(str).unique().tolist())
            selected = cols[idx % len(cols)].multiselect(col, values, key=f"{key}_{col}")
            if selected:
                out = out[out[col].astype(str).isin(selected)]
    return out


def render_header(st, status: dict[str, Any]) -> None:
    live = bool(status.get("real_order_possible"))
    hero_class = "hero hero-live" if live else "hero"
    symbols = ", ".join(status.get("symbols", [])[:10]) or "Unknown"
    missing = "" if status.get("db_exists") else "<div class='warnbox' style='margin-top:10px'>SQLite DB not found. Run ingestion/training first.</div>"
    st.markdown(
        f"""
        <div class="{hero_class}">
          <div style="display:flex;justify-content:space-between;gap:16px;align-items:flex-start;flex-wrap:wrap;">
            <div>
              <h1 class="title">AI Trading Bot</h1>
              <p class="subtitle">Operational dashboard · Binance Spot · SQLite/read-only · risk-first monitoring.</p>
            </div>
            <div style="text-align:right">
              <div class="small">Last refresh UTC</div>
              <div style="font-weight:800;color:#f8fafc">{escape(str(status.get("last_refresh_utc", "N/A")))}</div>
            </div>
          </div>
          <div style="margin-top:12px">
            {mode_pill(str(status.get("mode", "Unknown")), live)}
            {status_pill(str(status.get("state", "Unknown")))}
            {pill("DRY_RUN=true" if status.get("safety_flags", {}).get("DRY_RUN") else "DRY_RUN=false", "pill-blue" if status.get("safety_flags", {}).get("DRY_RUN") else "pill-red")}
            {pill("Exchange: Binance Spot", "pill-gray")}
            {pill("Timeframe: " + escape(str(status.get("timeframe", "Unknown"))), "pill-gray")}
            {pill("Symbols: " + escape(symbols), "pill-gray")}
            {pill("Model: " + escape(str(status.get("active_model_id", "N/A"))), "pill-gray")}
            {pill("Model status: " + escape(str(status.get("active_model_status", "unknown"))), "pill-gray")}
          </div>
          {missing}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpis(st, summary: dict[str, Any], status: dict[str, Any], model_row: pd.Series | None, coverage: pd.DataFrame, gaps: pd.DataFrame) -> None:
    st.subheader("Mission-critical KPIs")
    k = st.columns(6)
    k[0].metric("Total equity", fmt_money(summary.get("total_equity")))
    k[1].metric("Cash USDT", fmt_money(summary.get("cash_usdt")))
    k[2].metric("Unrealized PnL", fmt_money(summary.get("unrealized_pnl")))
    k[3].metric("Realized PnL", fmt_money(summary.get("realized_pnl")))
    k[4].metric("Daily PnL", fmt_money(summary.get("daily_pnl")))
    k[5].metric("Exposure", fmt_pct(summary.get("exposure_pct")))

    k = st.columns(6)
    k[0].metric("Total return", fmt_pct(summary.get("total_return")))
    k[1].metric("Max drawdown", fmt_pct(summary.get("max_drawdown")))
    k[2].metric("Trades", "N/A" if summary.get("number_of_trades") is None else int(summary.get("number_of_trades")))
    k[3].metric("Win rate", fmt_pct(summary.get("win_rate")))
    k[4].metric("Profit factor", fmt_num(summary.get("profit_factor")))
    k[5].metric("Sharpe", fmt_num(summary.get("sharpe")))

    k = st.columns(6)
    k[0].metric("Active model_id", status.get("active_model_id", "N/A"))
    k[1].metric("Model status", status.get("active_model_status", "unknown"))
    k[2].metric("OOS return", fmt_pct(model_row.get("strategy_return") if model_row is not None else None))
    k[3].metric("OOS Sharpe", fmt_num(model_row.get("sharpe") if model_row is not None else None))
    k[4].metric("F1 macro", fmt_num(model_row.get("f1_macro") if model_row is not None else None))
    k[5].metric("Accuracy", fmt_num(model_row.get("accuracy") if model_row is not None else None))

    latest_candle = "N/A"
    min_cov = max_cov = "N/A"
    if not coverage.empty:
        if "max_datetime_utc" in coverage.columns:
            latest_candle = str(coverage["max_datetime_utc"].dropna().max()) if coverage["max_datetime_utc"].dropna().any() else "N/A"
            max_cov = latest_candle
        if "min_datetime_utc" in coverage.columns:
            min_cov = str(coverage["min_datetime_utc"].dropna().min()) if coverage["min_datetime_utc"].dropna().any() else "N/A"
    open_gaps = len(gaps) if not gaps.empty else 0
    k = st.columns(4)
    k[0].metric("Latest candle", latest_candle)
    k[1].metric("Coverage min", min_cov)
    k[2].metric("Coverage max", max_cov)
    k[3].metric("Open gaps", open_gaps)


def render_compact_kpis(st, summary: dict[str, Any], status: dict[str, Any], registry: pd.DataFrame, positions: pd.DataFrame, gaps: pd.DataFrame) -> None:
    """Small operator-first KPI strip for the simplified control panel."""
    active_models = 0
    if not registry.empty:
        if "is_active" in registry.columns:
            active_models = int(pd.to_numeric(registry["is_active"], errors="coerce").fillna(0).sum())
        elif "status" in registry.columns:
            active_models = int(registry["status"].astype(str).str.contains("active|ready|validated|accepted", case=False, regex=True).sum())
    open_positions = 0
    if not positions.empty and "quantity" in positions.columns:
        open_positions = int((pd.to_numeric(positions["quantity"], errors="coerce").fillna(0).abs() > 1e-12).sum())
    cards = st.columns(6)
    cards[0].metric("Portfolio", fmt_money(summary.get("total_equity")))
    cards[1].metric("Total PnL", fmt_money((summary.get("realized_pnl") or 0) + (summary.get("unrealized_pnl") or 0)))
    cards[2].metric("Daily PnL", fmt_money(summary.get("daily_pnl")))
    cards[3].metric("Bot", str(status.get("state") or "Unknown"))
    cards[4].metric("Models", active_models)
    cards[5].metric("Positions", open_positions)
    if not gaps.empty:
        st.warning(f"DATA WARNING: {len(gaps)} open data gaps.")


def fig_template(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0f172a",
        plot_bgcolor="#111827",
        margin=dict(l=10, r=10, t=36, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        font=dict(color="#e5e7eb"),
        hovermode="x unified",
    )
    fig.update_xaxes(gridcolor="rgba(148,163,184,.16)", zerolinecolor="rgba(148,163,184,.25)")
    fig.update_yaxes(gridcolor="rgba(148,163,184,.16)", zerolinecolor="rgba(148,163,184,.25)")
    return fig


def safe_plotly_chart(st, fig: go.Figure, *, height: int = 420, key: str | None = None) -> None:
    """Render Plotly with a fixed height and a native fallback if Streamlit fails.

    Some local Streamlit/Plotly combinations render transparent/auto-sized
    figures poorly inside tabs/columns. A fixed height plus non-transparent
    background makes the charts consistently visible.
    """
    fig.update_layout(height=height, autosize=True)
    trace_count = len(fig.data)
    visible_points = 0
    for trace in fig.data:
        y = getattr(trace, "y", None)
        close = getattr(trace, "close", None)
        if y is not None:
            try:
                visible_points += int(pd.Series(y).notna().sum())
            except Exception:
                pass
        elif close is not None:
            try:
                visible_points += int(pd.Series(close).notna().sum())
            except Exception:
                pass
    if trace_count == 0 or visible_points == 0:
        st.warning("Chart has no plottable points after filtering. Check symbol/timeframe and upstream data.")
        return
    try:
        chart_key = key or f"plotly_chart_{next(_PLOTLY_KEY_COUNTER)}"
        st.plotly_chart(fig_template(fig), use_container_width=True, config={"displaylogo": False}, key=chart_key)
    except Exception as exc:
        st.warning(f"Plotly chart could not be rendered: {exc}")


def render_equity(st, equity: pd.DataFrame, *, key_prefix: str = "equity") -> None:
    st.subheader("Equity curve and drawdown")
    if equity.empty:
        st.info(equity.attrs.get("message", "No equity curve available. Run paper trading/backtest first."))
        return
    required = {"datetime_utc", "equity"}
    if not required.issubset(equity.columns):
        st.warning(f"Equity data exists but is missing required columns: {sorted(required - set(equity.columns))}")
        show_df(st, equity.head(20), height=220)
        return
    equity = equity.dropna(subset=["datetime_utc", "equity"]).copy()
    if equity.empty:
        st.warning("Equity data has no plottable datetime/equity rows.")
        return
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.68, 0.32], vertical_spacing=0.05)
    mode = "lines+markers" if len(equity) < 5 else "lines"
    fig.add_trace(go.Scatter(x=equity["datetime_utc"], y=equity["equity"], mode=mode, name="Bot equity", line=dict(color="#22c55e", width=2.4)), row=1, col=1)
    if "benchmark_equity" in equity.columns and equity["benchmark_equity"].notna().any():
        fig.add_trace(go.Scatter(x=equity["datetime_utc"], y=equity["benchmark_equity"], mode=mode, name="Buy & hold", line=dict(color="#cbd5e1", width=1.6, dash="dot")), row=1, col=1)
    if "drawdown" in equity.columns and equity["drawdown"].notna().any():
        fig.add_trace(go.Scatter(x=equity["datetime_utc"], y=equity["drawdown"], fill="tozeroy", mode="lines", name="Drawdown", line=dict(color="#ef4444", width=1.8)), row=2, col=1)
    fig.update_yaxes(title_text="Equity", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown", tickformat=".1%", row=2, col=1)
    safe_plotly_chart(st, fig, height=560, key=f"{key_prefix}_equity_{equity['source'].iloc[0] if 'source' in equity.columns and len(equity) else 'unknown'}_{len(equity)}")
    st.caption(f"Source: {equity['source'].iloc[0] if 'source' in equity.columns and len(equity) else 'N/A'}")


def render_trade_pnl(st, trades: pd.DataFrame, *, key_prefix: str = "trade_pnl") -> None:
    st.subheader("PnL by operation / backtest decision")
    if trades.empty:
        st.info(trades.attrs.get("message", "No trade PnL available. Orders do not contain realized PnL yet; run backtest for report-based view."))
        return
    if "pnl" not in trades.columns:
        st.warning("Trade/PnL data exists but has no `pnl` column.")
        show_df(st, trades.head(20), height=220)
        return
    trades = trades.dropna(subset=["pnl"]).copy()
    if trades.empty:
        st.warning("Trade/PnL data has no non-null PnL rows.")
        return
    x = trades["datetime_utc"] if "datetime_utc" in trades.columns else trades.index
    # Report-based PnL is often per-bar return; show bps so bars are visible.
    values = trades["pnl"]
    y_title = "PnL / return"
    if values.abs().max() < 0.05:
        values = values * 10_000
        y_title = "Return (bps)"
        if "cumulative_pnl" in trades.columns:
            trades["cumulative_plot"] = trades["cumulative_pnl"] * 100
    else:
        trades["cumulative_plot"] = trades.get("cumulative_pnl")
    colors = ["#22c55e" if v >= 0 else "#ef4444" for v in trades["pnl"].fillna(0)]
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=x, y=values, marker_color=colors, name=y_title), secondary_y=False)
    if "cumulative_plot" in trades.columns and trades["cumulative_plot"].notna().any():
        fig.add_trace(go.Scatter(x=x, y=trades["cumulative_plot"], mode="lines", name="Cumulative (%)", line=dict(color="#38bdf8", width=2)), secondary_y=True)
    fig.update_yaxes(title_text=y_title, secondary_y=False)
    fig.update_yaxes(title_text="Cumulative (%)", secondary_y=True)
    safe_plotly_chart(st, fig, height=380, key=f"{key_prefix}_trade_pnl_{trades['source'].iloc[0] if 'source' in trades.columns and len(trades) else 'unknown'}_{len(trades)}")
    st.caption(f"Source: {trades['source'].iloc[0] if 'source' in trades.columns and len(trades) else 'N/A'}")


def render_price_signals(
    st,
    symbol: str,
    timeframe: str,
    price: pd.DataFrame,
    signals: pd.DataFrame,
    orders: pd.DataFrame,
    positions: pd.DataFrame | None = None,
    *,
    key_prefix: str = "price",
) -> None:
    st.subheader(f"Gráfico de velas - {symbol} - {timeframe}")
    if price.empty:
        st.info(price.attrs.get("message", "No price data available for this symbol/timeframe."))
        return
    if "datetime_utc" not in price.columns or "close" not in price.columns:
        st.warning("Price data exists but is missing `datetime_utc` or `close`.")
        show_df(st, price.head(20), height=220)
        return
    price = price.dropna(subset=["datetime_utc", "close"]).copy()
    if price.empty:
        st.warning("Price data has no plottable rows.")
        return

    has_volume = "volume" in price.columns and pd.to_numeric(price["volume"], errors="coerce").notna().any()
    fig = make_subplots(
        rows=2 if has_volume else 1,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.76, 0.24] if has_volume else [1.0],
        vertical_spacing=0.03,
    )
    if {"open", "high", "low", "close"}.issubset(price.columns) and price[["open", "high", "low", "close"]].notna().all(axis=1).any():
        fig.add_trace(
            go.Candlestick(
                x=price["datetime_utc"],
                open=price["open"],
                high=price["high"],
                low=price["low"],
                close=price["close"],
                name="Candles",
                increasing_line_color="#22c55e",
                decreasing_line_color="#ef4444",
            ),
            row=1,
            col=1,
        )
    else:
        fig.add_trace(go.Scatter(x=price["datetime_utc"], y=price["close"], mode="lines", name="Close", line=dict(color="#38bdf8", width=2)), row=1, col=1)

    if has_volume:
        fig.add_trace(go.Bar(x=price["datetime_utc"], y=pd.to_numeric(price["volume"], errors="coerce"), name="Volume", marker_color="rgba(148,163,184,.45)"), row=2, col=1)

    if not signals.empty and {"symbol", "datetime_utc"}.issubset(signals.columns):
        tf_series = signals["timeframe"].astype(str) if "timeframe" in signals.columns else pd.Series([timeframe] * len(signals), index=signals.index)
        sig = signals[(signals["symbol"].astype(str) == symbol) & (tf_series == timeframe)].copy()
        if not sig.empty:
            sig["datetime_utc"] = pd.to_datetime(sig["datetime_utc"], utc=True, errors="coerce")
            sig = sig.merge(price[["datetime_utc", "close"]], on="datetime_utc", how="left")
            label_series = sig["signal"].astype(str) if "signal" in sig.columns else (sig["signal_label"].astype(str) if "signal_label" in sig.columns else pd.Series([""] * len(sig), index=sig.index))
            for label, color, marker in [("LONG", "#22c55e", "triangle-up"), ("SHORT", "#ef4444", "triangle-down"), ("FLAT", "#94a3b8", "circle")]:
                part = sig[label_series.str.upper() == label]
                if not part.empty:
                    fig.add_trace(go.Scatter(x=part["datetime_utc"], y=part["close"], mode="markers", name=f"Signal {label}", marker=dict(color=color, size=11, symbol=marker, line=dict(color="#020617", width=1))), row=1, col=1)
    if not orders.empty and {"symbol", "created_at_utc"}.issubset(orders.columns):
        od = orders[orders["symbol"].astype(str) == symbol].copy()
        if not od.empty:
            od["created_at_utc"] = pd.to_datetime(od["created_at_utc"], utc=True, errors="coerce")
            od["plot_price"] = pd.to_numeric(od.get("fill_price", od.get("requested_price")), errors="coerce")
            od = od.dropna(subset=["created_at_utc", "plot_price"])
            if not od.empty:
                fig.add_trace(go.Scatter(x=od["created_at_utc"], y=od["plot_price"], mode="markers", name="Orders", marker=dict(color="#facc15", size=11, symbol="x")), row=1, col=1)

    if positions is not None and not positions.empty and {"symbol", "quantity"}.issubset(positions.columns):
        pos = positions[positions["symbol"].astype(str) == symbol].copy()
        if not pos.empty:
            pos["quantity"] = pd.to_numeric(pos["quantity"], errors="coerce")
            pos = pos[pos["quantity"].fillna(0).abs() > 1e-12]
        if not pos.empty:
            x0 = price["datetime_utc"].min()
            x1 = price["datetime_utc"].max()
            entry_col = "avg_entry_price" if "avg_entry_price" in pos.columns else ("avg_price" if "avg_price" in pos.columns else None)
            if entry_col:
                for idx, row in pos.head(12).iterrows():
                    entry = numeric_or_none(row.get(entry_col))
                    if entry is None:
                        continue
                    model_id = str(row.get("model_id", "model N/A"))
                    qty = numeric_or_none(row.get("quantity")) or 0.0
                    pnl = numeric_or_none(row.get("unrealized_pnl"))
                    fig.add_trace(
                        go.Scatter(
                            x=[x0, x1],
                            y=[entry, entry],
                            mode="lines",
                            name=f"Open position {model_id}",
                            line=dict(color="#38bdf8", width=1.6, dash="dash"),
                            hovertemplate=f"model={model_id}<br>qty={qty:,.8f}<br>entry={entry:,.4f}<br>unrealized={fmt_money(pnl)}<extra></extra>",
                        ),
                        row=1,
                        col=1,
                    )
            marker_price = pd.to_numeric(pos.get("current_price"), errors="coerce") if "current_price" in pos.columns else pd.Series([pd.NA] * len(pos), index=pos.index)
            if marker_price.notna().any():
                fig.add_trace(
                    go.Scatter(
                        x=[price["datetime_utc"].max()] * int(marker_price.notna().sum()),
                        y=marker_price.dropna(),
                        mode="markers",
                        name="Open positions",
                        marker=dict(color="#38bdf8", size=12, symbol="diamond", line=dict(color="#020617", width=1)),
                    ),
                    row=1,
                    col=1,
                )
    fig.update_xaxes(rangeslider_visible=False)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    if has_volume:
        fig.update_yaxes(title_text="Volume", row=2, col=1)
    safe_plotly_chart(st, fig, height=560, key=f"{key_prefix}_price_signals_{symbol}_{timeframe}_{len(price)}")


def render_exposure(st, exposure: pd.DataFrame, *, key_prefix: str = "exposure") -> None:
    st.subheader("Exposure by asset")
    if exposure.empty:
        st.info("No open exposure data available. Run paper trading or create portfolio snapshots first.")
        return
    exposure = exposure.dropna(subset=["value_usdt"]).copy()
    if exposure.empty:
        st.info("Exposure rows exist, but all values are empty.")
        return
    fig = px.bar(exposure, x="asset", y="value_usdt", color="asset")
    fig.update_layout(showlegend=False, yaxis_title="USDT value")
    safe_plotly_chart(st, fig, height=320, key=f"{key_prefix}_exposure_{len(exposure)}")


def apply_selection_filters(
    df: pd.DataFrame,
    *,
    model_ids: list[str] | None = None,
    symbols: list[str] | None = None,
    account_modes: list[str] | None = None,
    statuses: list[str] | None = None,
    signal_labels: list[str] | None = None,
) -> pd.DataFrame:
    """Apply global operator filters only when matching columns exist."""
    if df.empty:
        return df
    out = df.copy()
    expanded_model_ids = expand_model_ids_for_filter(model_ids)
    filters = [
        ("model_id", expanded_model_ids),
        ("symbol", symbols),
        ("account_mode", account_modes),
        ("status", statuses),
    ]
    for col, selected in filters:
        if selected and col in out.columns:
            out = out[out[col].astype(str).isin([str(v) for v in selected])]
    if signal_labels:
        label_col = "signal" if "signal" in out.columns else ("signal_label" if "signal_label" in out.columns else None)
        if label_col:
            out = out[out[label_col].astype(str).isin([str(v) for v in signal_labels])]
    return out


def expand_model_ids_for_filter(model_ids: list[str] | None) -> list[str] | None:
    if not model_ids:
        return model_ids
    selected = {str(v) for v in model_ids if str(v)}
    try:
        aliases = data.load_model_aliases()
    except Exception:
        aliases = pd.DataFrame()
    if not aliases.empty and {"alias_model_id", "canonical_model_id"}.issubset(aliases.columns):
        for _, row in aliases.iterrows():
            alias = str(row.get("alias_model_id"))
            canonical = str(row.get("canonical_model_id"))
            if canonical in selected or alias in selected:
                selected.add(alias)
                selected.add(canonical)
    return sorted(selected)


def build_equity_curve_from_snapshots(snapshots: pd.DataFrame) -> pd.DataFrame:
    """Build aggregate equity curve after dashboard filters are applied."""
    if snapshots.empty or not {"datetime_utc", "equity"}.issubset(snapshots.columns):
        return pd.DataFrame()
    df = snapshots.copy()
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True, errors="coerce")
    for col in ["cash", "equity", "realized_pnl", "unrealized_pnl"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["datetime_utc", "equity"])
    if df.empty:
        return pd.DataFrame()
    agg_map = {"equity": ("equity", "sum")}
    for col in ["cash", "realized_pnl", "unrealized_pnl"]:
        if col in df.columns:
            agg_map[col] = (col, "sum")
    curve = df.groupby("datetime_utc", dropna=False).agg(**agg_map).reset_index().sort_values("datetime_utc")
    curve["drawdown"] = curve["equity"] / curve["equity"].cummax() - 1
    curve["source"] = "portfolio_snapshots_filtered"
    return curve


def choose_account_modes(all_accounts: list[str], view_real: bool) -> list[str]:
    """Map the real/demo switch to known account_mode values without assuming one exact label."""
    if not all_accounts:
        return []
    real_keywords = ("real", "live")
    real_modes = [m for m in all_accounts if any(k in str(m).lower() for k in real_keywords)]
    if view_real:
        return real_modes or ["__NO_REAL_ACCOUNT_MODE__"]
    return [m for m in all_accounts if m not in real_modes]


def calculate_portfolio_view_summary(snapshots: pd.DataFrame, positions: pd.DataFrame, fallback: dict[str, Any]) -> dict[str, Any]:
    """Current portfolio position and global/daily PnL for the selected account/symbol filters."""
    summary = dict(fallback)
    if not snapshots.empty and {"datetime_utc", "equity"}.issubset(snapshots.columns):
        df = snapshots.copy()
        df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True, errors="coerce")
        for col in ["equity", "cash", "realized_pnl", "unrealized_pnl"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["datetime_utc", "equity"])
        if not df.empty:
            group_cols = [c for c in ["model_id", "account_mode"] if c in df.columns]
            latest = _latest_by_group_dashboard(df, group_cols, "datetime_utc") if group_cols else df.sort_values("datetime_utc").tail(1)
            summary["total_equity"] = latest["equity"].sum()
            if "cash" in latest.columns:
                summary["cash_usdt"] = latest["cash"].sum()
            if "realized_pnl" in latest.columns:
                summary["realized_pnl"] = latest["realized_pnl"].sum()
            if "unrealized_pnl" in latest.columns:
                summary["unrealized_pnl"] = latest["unrealized_pnl"].sum()
            equity_curve = df.groupby("datetime_utc", dropna=False)["equity"].sum().sort_index()
            if len(equity_curve) > 1:
                first = float(equity_curve.iloc[0])
                last = float(equity_curve.iloc[-1])
                summary["total_return"] = (last / first - 1) if first else None
                summary["max_drawdown"] = (equity_curve / equity_curve.cummax() - 1).min()
                recent = equity_curve[equity_curve.index >= pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=24)]
                if len(recent):
                    base = float(recent.iloc[0])
                    summary["daily_pnl"] = last - base
                    summary["daily_return"] = (last / base - 1) if base else None
            summary["source"] = "portfolio_snapshots_filtered"

    total_pnl = (numeric_or_none(summary.get("realized_pnl")) or 0.0) + (numeric_or_none(summary.get("unrealized_pnl")) or 0.0)
    summary["total_pnl"] = total_pnl
    if not positions.empty and "market_value" in positions.columns:
        exposure = pd.to_numeric(positions["market_value"], errors="coerce").abs().sum()
        equity = numeric_or_none(summary.get("total_equity"))
        summary["invested_value"] = exposure
        summary["exposure_pct"] = exposure / equity if equity else summary.get("exposure_pct")
    return summary


def _latest_by_group_dashboard(df: pd.DataFrame, group_cols: list[str], ts_col: str) -> pd.DataFrame:
    if not group_cols:
        return df.sort_values(ts_col).tail(1)
    return df.sort_values(ts_col).groupby(group_cols, dropna=False).tail(1)


def render_portfolio_position_header(st, summary: dict[str, Any], account_label: str) -> None:
    st.subheader(f"Posición actual de cartera · {account_label}")
    cols = st.columns(7)
    total_pnl = numeric_or_none(summary.get("total_pnl"))
    daily_pnl = numeric_or_none(summary.get("daily_pnl"))
    cols[0].metric("Equity total", fmt_money(summary.get("total_equity")))
    cols[1].metric("USDT libre", fmt_money(summary.get("cash_usdt")))
    cols[2].metric("Invertido", fmt_money(summary.get("invested_value")))
    cols[3].metric("PnL total", fmt_money(total_pnl), delta=fmt_pct(summary.get("total_return")))
    cols[4].metric("PnL diario", fmt_money(daily_pnl), delta=fmt_pct(summary.get("daily_return")))
    cols[5].metric("Exposición", fmt_pct(summary.get("exposure_pct")))
    cols[6].metric("Drawdown", fmt_pct(summary.get("max_drawdown")))


def build_compact_model_view(registry: pd.DataFrame, model_control: pd.DataFrame, selected_models: list[str] | None = None) -> pd.DataFrame:
    merged = merge_model_controls(registry, model_control)
    if not merged.empty:
        merged = apply_selection_filters(merged, model_ids=selected_models or [])
    if merged.empty:
        return merged
    rows = []
    for _, row in merged.iterrows():
        rows.append(
            {
                "model_id": row.get("model_id"),
                "estado": row.get("status", row.get("acceptance_status")),
                "símbolos": row.get("symbol_scope"),
                "tf": row.get("timeframe"),
                "activo": boolish(row.get("dashboard_signal_enabled")),
                "paper": boolish(row.get("dashboard_paper_enabled")),
                "total_return": row.get("strategy_return"),
                "exceso_vs_bh": row.get("excess_return"),
                "drawdown": row.get("max_drawdown"),
                "profit_factor": row.get("profit_factor"),
                "trades": row.get("trade_count"),
                "score": row.get("risk_adjusted_score"),
                "entrenado_utc": row.get("training_ts_utc"),
            }
        )
    out = pd.DataFrame(rows)
    if "score" in out.columns:
        out = out.sort_values("score", ascending=False, na_position="last")
    return out


def render_main_models_table(st, registry: pd.DataFrame, model_control: pd.DataFrame, requested_by: str, selected_models: list[str]) -> str | None:
    st.subheader("Modelos")
    view = build_compact_model_view(registry, model_control, selected_models)
    if view.empty:
        st.info("No hay modelos registrados todavía. Ejecuta model_maintenance/train primero.")
        return None
    visible_cols = ["model_id", "estado", "símbolos", "tf", "activo", "paper", "total_return", "drawdown", "profit_factor", "trades", "score"]
    table = view[[c for c in visible_cols if c in view.columns]].copy()
    st.caption("Puedes clicar una fila si tu versión de Streamlit lo permite. Si no, usa el selector debajo.")
    clicked_model_id = None
    try:
        event = st.dataframe(
            simple_display_df(table, pct_cols=["total_return", "drawdown"]),
            use_container_width=True,
            height=330,
            hide_index=True,
            key="models_plain_select_table",
            on_select="rerun",
            selection_mode="single-row",
        )
        selection = getattr(event, "selection", None)
        if selection is None and isinstance(event, dict):
            selection = event.get("selection")
        rows = getattr(selection, "rows", None)
        if rows is None and isinstance(selection, dict):
            rows = selection.get("rows")
        if rows:
            clicked_model_id = str(view.iloc[int(rows[0])].get("model_id"))
    except Exception:
        show_simple_df(st, table, height=330, pct_cols=["total_return", "drawdown"])

    model_ids = view["model_id"].dropna().astype(str).tolist()
    current = clicked_model_id or st.session_state.get("dashboard_selected_model_id")
    index = model_ids.index(current) if current in model_ids else 0
    selected = st.selectbox("Modelo seleccionado", model_ids, index=index, key="main_model_selectbox")
    if selected:
        st.session_state["dashboard_selected_model_id"] = selected
    selected_row_df = view[view["model_id"].astype(str) == str(selected)]
    if not selected_row_df.empty:
        row = selected_row_df.iloc[0]
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Return", fmt_pct(row.get("total_return")))
        c2.metric("Drawdown", fmt_pct(row.get("drawdown")))
        c3.metric("Profit factor", fmt_num(row.get("profit_factor")))
        signal_new = c4.toggle("Signals", value=bool(row.get("activo")), key=f"selected_signal_toggle_{selected}")
        paper_new = c5.toggle("Paper", value=bool(row.get("paper")), key=f"selected_paper_toggle_{selected}")
        if signal_new != bool(row.get("activo")):
            render_action_result(st, controls.activate_model if signal_new else controls.deactivate_model, selected, requested_by)
            st.rerun()
        if paper_new != bool(row.get("paper")):
            render_action_result(st, controls.set_model_paper, selected, paper_new, requested_by)
            st.rerun()
    return selected


def build_open_positions_view(positions: pd.DataFrame, trades: pd.DataFrame | None = None) -> pd.DataFrame:
    if positions.empty:
        return positions
    out = positions.copy()
    for col in ["quantity", "avg_entry_price", "avg_price", "current_price", "market_value", "unrealized_pnl", "realized_pnl", "exposure_pct"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    if "avg_entry_price" not in out.columns and "avg_price" in out.columns:
        out["avg_entry_price"] = out["avg_price"]
    if {"current_price", "avg_entry_price"}.issubset(out.columns):
        out["pnl_pct"] = out["current_price"] / out["avg_entry_price"] - 1
    if trades is not None and not trades.empty and "symbol" in out.columns and "symbol" in trades.columns:
        tr = trades.copy()
        if "status" in tr.columns:
            tr = tr[tr["status"].astype(str).str.upper().isin(["OPEN", "PARTIALLY_FILLED", "REDUCING", "CLOSING", "ORDER_SENT", "ORDER_PENDING"])]
        keep = [c for c in ["trade_id", "model_id", "symbol", "tp_price", "sl_price", "emergency_sl_price", "horizon_bars", "opened_at_utc", "exit_reason"] if c in tr.columns]
        if keep:
            dedupe = [c for c in ["model_id", "symbol"] if c in keep]
            if dedupe:
                tr = tr[keep].drop_duplicates(dedupe, keep="last")
                out = out.merge(tr, on=dedupe, how="left", suffixes=("", "_trade"))
    if {"current_price", "tp_price"}.issubset(out.columns):
        out["dist_tp_pct"] = pd.to_numeric(out["tp_price"], errors="coerce") / out["current_price"] - 1
    if {"current_price", "sl_price"}.issubset(out.columns):
        out["dist_sl_pct"] = pd.to_numeric(out["sl_price"], errors="coerce") / out["current_price"] - 1
    preferred = [
        "symbol", "model_id", "account_mode", "quantity", "avg_entry_price", "current_price",
        "market_value", "unrealized_pnl", "pnl_pct", "realized_pnl", "exposure_pct",
        "tp_price", "sl_price", "dist_tp_pct", "dist_sl_pct", "trade_id", "updated_at_utc",
    ]
    return out[[c for c in preferred if c in out.columns]]


def build_filtered_exposure_breakdown(positions: pd.DataFrame, summary: dict[str, Any]) -> pd.DataFrame:
    if not positions.empty and "market_value" in positions.columns and "symbol" in positions.columns:
        by_asset = positions.copy()
        by_asset["market_value"] = pd.to_numeric(by_asset["market_value"], errors="coerce")
        by_asset = by_asset.groupby("symbol", dropna=False)["market_value"].sum().reset_index()
        by_asset = by_asset.rename(columns={"symbol": "asset", "market_value": "value_usdt"})
        by_asset["asset"] = by_asset["asset"].astype(str).str.replace("USDT", "", regex=False)
        cash = numeric_or_none(summary.get("cash_usdt"))
        if cash is not None:
            by_asset = pd.concat([by_asset, pd.DataFrame([{"asset": "USDT", "value_usdt": cash}])], ignore_index=True)
        return by_asset.dropna(subset=["value_usdt"])
    cash = numeric_or_none(summary.get("cash_usdt"))
    if cash is not None:
        return pd.DataFrame([{"asset": "USDT", "value_usdt": cash}])
    return pd.DataFrame(columns=["asset", "value_usdt"])


def filter_model_context_df(df: pd.DataFrame, model_id: str, selected_symbols: list[str], selected_accounts: list[str]) -> pd.DataFrame:
    return apply_selection_filters(df, model_ids=[model_id], symbols=selected_symbols, account_modes=selected_accounts)


def render_selected_model_context(
    st,
    *,
    model_id: str | None,
    registry: pd.DataFrame,
    signals: pd.DataFrame,
    orders: pd.DataFrame,
    fills: pd.DataFrame,
    positions: pd.DataFrame,
    snapshots: pd.DataFrame,
    trades: pd.DataFrame,
    selected_symbols: list[str],
    selected_accounts: list[str],
) -> None:
    if not model_id:
        st.info("Clica un modelo en la tabla para abrir su ficha operativa y usarlo como contexto de navegación.")
        return
    st.markdown(f"### Modelo seleccionado: `{model_id}`")
    c0, c1, c2 = st.columns([1.2, 1, 1])
    apply_filter = c0.toggle(
        "Usar este modelo como filtro global",
        value=bool(st.session_state.get("dashboard_apply_selected_model_filter", True)),
        key="selected_model_filter_toggle",
    )
    st.session_state["dashboard_apply_selected_model_filter"] = bool(apply_filter)
    if c1.button("Limpiar modelo seleccionado", use_container_width=True):
        st.session_state.pop("dashboard_selected_model_id", None)
        st.session_state["dashboard_apply_selected_model_filter"] = False
        st.rerun()
    c2.caption("Atajo: usa las pestañas con este mismo contexto model_id/símbolo/cuenta.")

    row = pd.DataFrame()
    if not registry.empty and "model_id" in registry.columns:
        row = registry[registry["model_id"].astype(str) == str(model_id)].head(1)
    sig_m = filter_model_context_df(signals, model_id, selected_symbols, selected_accounts)
    ord_m = filter_model_context_df(orders, model_id, selected_symbols, selected_accounts)
    fill_m = filter_model_context_df(fills, model_id, selected_symbols, selected_accounts)
    pos_m = filter_model_context_df(positions, model_id, selected_symbols, selected_accounts)
    snap_m = apply_selection_filters(snapshots, model_ids=[model_id], account_modes=selected_accounts)
    trade_m = filter_model_context_df(trades, model_id, selected_symbols, selected_accounts)
    prop_m = filter_model_context_df(data.read_table("trade_proposals", 500, "created_at_utc"), model_id, selected_symbols, selected_accounts)
    alloc_m = filter_model_context_df(data.read_table("allocations", 500, "created_at_utc"), model_id, selected_symbols, selected_accounts)
    pred_m = filter_model_context_df(data.read_table("model_predictions", 500, "created_at_utc"), model_id, selected_symbols, selected_accounts)
    perf_m = filter_model_context_df(data.read_table("model_performance", 500, "timestamp_utc"), model_id, selected_symbols, selected_accounts)
    metrics_m = filter_model_context_df(data.load_paper_model_metrics(2000), model_id, selected_symbols, selected_accounts)

    realized = pd.to_numeric(pos_m.get("realized_pnl"), errors="coerce").sum() if not pos_m.empty and "realized_pnl" in pos_m.columns else None
    unrealized = pd.to_numeric(pos_m.get("unrealized_pnl"), errors="coerce").sum() if not pos_m.empty and "unrealized_pnl" in pos_m.columns else None
    exposure = pd.to_numeric(pos_m.get("market_value"), errors="coerce").abs().sum() if not pos_m.empty and "market_value" in pos_m.columns else None
    latest_metric = metrics_m.sort_values("evaluated_at_utc").tail(1) if not metrics_m.empty and "evaluated_at_utc" in metrics_m.columns else pd.DataFrame()
    k = st.columns(7)
    k[0].metric("Señales", len(sig_m))
    k[1].metric("Propuestas", len(prop_m))
    k[2].metric("Trades", len(trade_m) if not trade_m.empty else len(ord_m))
    k[3].metric("Exposición", fmt_money(exposure))
    k[4].metric("PnL realizado", fmt_money(realized))
    k[5].metric("PnL abierto", fmt_money(unrealized))
    k[6].metric("Return paper", fmt_pct(latest_metric.iloc[0].get("total_return") if not latest_metric.empty else None))

    sub = st.tabs(["Resumen", "Predicciones/señales", "Propuestas/allocator", "Trades/posiciones", "Órdenes/fills", "Métricas"])
    with sub[0]:
        cols = [c for c in [
            "model_id", "status", "acceptance_status", "symbol_scope", "timeframe", "training_scope",
            "is_active", "training_ts_utc", "strategy_return", "excess_return", "max_drawdown",
            "profit_factor", "trade_count", "sharpe", "f1_macro", "accuracy",
        ] if c in row.columns]
        show_colored_df(st, row[cols] if cols and not row.empty else row, height=150, pct_cols=["strategy_return", "excess_return", "max_drawdown"], pnl_cols=["strategy_return", "excess_return"], status_cols=["status", "acceptance_status"], empty="No hay fila de registry para este modelo.")
        render_symbol_bot_matrix(st, row if not row.empty else registry[registry.get("model_id", pd.Series(dtype=str)).astype(str) == str(model_id)] if "model_id" in registry.columns else row, pos_m, ord_m)
    with sub[1]:
        pred_cols = [c for c in ["timestamp_utc", "symbol", "timeframe", "direction", "confidence", "expected_return_pct", "expected_adverse_move_pct", "horizon_bars", "signal_valid_until_utc"] if c in pred_m.columns]
        show_colored_df(st, pred_m[pred_cols].head(80) if pred_cols and not pred_m.empty else pred_m.head(80), height=240, pct_cols=["expected_return_pct", "expected_adverse_move_pct"], pnl_cols=["expected_return_pct"], empty="No hay model_predictions para este modelo.")
        sig_cols = [c for c in ["datetime_utc", "symbol", "timeframe", "account_mode", "signal", "confidence", "entry_price", "take_profit_price", "stop_loss_price", "risk_reward"] if c in sig_m.columns]
        show_colored_df(st, sig_m[sig_cols].head(120) if sig_cols and not sig_m.empty else sig_m.head(120), height=260, money_cols=["entry_price", "take_profit_price", "stop_loss_price"], empty="No hay señales recientes.")
    with sub[2]:
        prop_cols = [c for c in ["created_at_utc", "proposal_id", "prediction_id", "symbol", "side", "confidence", "expected_return_pct", "expected_adverse_move_pct", "requested_notional_usdt", "proposal_score", "status", "rejection_reason"] if c in prop_m.columns]
        show_colored_df(st, prop_m[prop_cols].head(120) if prop_cols and not prop_m.empty else prop_m.head(120), height=260, money_cols=["requested_notional_usdt"], pct_cols=["expected_return_pct", "expected_adverse_move_pct"], pnl_cols=["expected_return_pct"], status_cols=["status"], empty="No hay propuestas para este modelo.")
        alloc_cols = [c for c in ["created_at_utc", "allocation_id", "proposal_id", "symbol", "decision", "requested_notional_usdt", "approved_notional_usdt", "allocator_score", "rejection_reason", "shadow_trade_id"] if c in alloc_m.columns]
        show_colored_df(st, alloc_m[alloc_cols].head(120) if alloc_cols and not alloc_m.empty else alloc_m.head(120), height=240, money_cols=["requested_notional_usdt", "approved_notional_usdt"], status_cols=["decision"], empty="No hay decisiones de allocator para este modelo.")
    with sub[3]:
        show_colored_df(st, build_open_positions_view(pos_m, trade_m), height=280, money_cols=["avg_entry_price", "current_price", "market_value", "realized_pnl", "tp_price", "sl_price"], pct_cols=["pnl_pct", "exposure_pct", "dist_tp_pct", "dist_sl_pct"], pnl_cols=["unrealized_pnl", "realized_pnl", "pnl_pct"], empty="No hay posiciones abiertas para este modelo.")
        trade_cols = [c for c in ["status", "trade_id", "proposal_id", "allocation_id", "symbol", "side", "approved_notional_usdt", "qty", "avg_entry_price", "tp_price", "sl_price", "unrealized_pnl_usdt", "realized_pnl_usdt", "exit_reason", "opened_at_utc", "updated_at_utc"] if c in trade_m.columns]
        show_colored_df(st, trade_m[trade_cols].head(120) if trade_cols and not trade_m.empty else trade_m.head(120), height=260, money_cols=["approved_notional_usdt", "avg_entry_price", "tp_price", "sl_price"], pnl_cols=["unrealized_pnl_usdt", "realized_pnl_usdt"], status_cols=["status"], empty="No hay trades persistidos para este modelo.")
    with sub[4]:
        order_cols = [c for c in ["created_at_utc", "order_id", "trade_id", "proposal_id", "allocation_id", "symbol", "side", "type", "quantity", "fill_price", "status", "account_mode", "fees_usdt", "slippage_usdt", "reason"] if c in ord_m.columns]
        show_colored_df(st, ord_m[order_cols].head(150) if order_cols and not ord_m.empty else ord_m.head(150), height=280, money_cols=["fill_price", "fees_usdt", "slippage_usdt"], status_cols=["status"], empty="No hay órdenes para este modelo.")
        fill_cols = [c for c in ["timestamp_utc", "fill_id", "order_id", "trade_id", "proposal_id", "allocation_id", "symbol", "quantity", "price", "fee_usdt", "slippage_usdt", "account_mode"] if c in fill_m.columns]
        show_colored_df(st, fill_m[fill_cols].head(150) if fill_cols and not fill_m.empty else fill_m.head(150), height=260, money_cols=["price", "fee_usdt", "slippage_usdt"], empty="No hay fills para este modelo.")
    with sub[5]:
        show_colored_df(st, metrics_m.head(80), height=260, money_cols=["realized_pnl", "unrealized_pnl", "total_pnl", "equity", "current_exposure"], pct_cols=["total_return", "max_drawdown", "win_rate"], pnl_cols=["realized_pnl", "unrealized_pnl", "total_pnl", "total_return"], status_cols=["validation_status"], empty="No hay paper_model_metrics para este modelo.")
        perf_cols = [c for c in ["timestamp_utc", "symbol", "timeframe", "account_mode", "predictions", "proposals", "accepted_proposals", "rejected_proposals", "open_trades", "closed_trades", "realized_pnl_usdt", "unrealized_pnl_usdt", "total_return_pct", "max_drawdown_pct", "win_rate", "profit_factor", "degradation_status"] if c in perf_m.columns]
        show_colored_df(st, perf_m[perf_cols].head(120) if perf_cols and not perf_m.empty else perf_m.head(120), height=260, money_cols=["realized_pnl_usdt", "unrealized_pnl_usdt"], pct_cols=["total_return_pct", "max_drawdown_pct", "win_rate"], pnl_cols=["realized_pnl_usdt", "unrealized_pnl_usdt", "total_return_pct"], status_cols=["degradation_status"], empty="No hay model_performance para este modelo.")


def render_selected_model_simple(
    st,
    *,
    model_id: str | None,
    registry: pd.DataFrame,
    signals: pd.DataFrame,
    orders: pd.DataFrame,
    fills: pd.DataFrame,
    positions: pd.DataFrame,
    trades: pd.DataFrame,
    selected_symbols: list[str],
    selected_accounts: list[str],
) -> None:
    """Simple model detail page: few sections, plain visible tables."""
    if not model_id:
        st.info("Selecciona un modelo en la pantalla principal.")
        return
    st.subheader(f"Modelo seleccionado: {model_id}")
    apply_filter = st.toggle(
        "Aplicar este modelo como filtro global",
        value=bool(st.session_state.get("dashboard_apply_selected_model_filter", True)),
        key="simple_selected_model_global_filter",
    )
    st.session_state["dashboard_apply_selected_model_filter"] = bool(apply_filter)
    if st.button("Limpiar selección de modelo", use_container_width=False):
        st.session_state.pop("dashboard_selected_model_id", None)
        st.session_state["dashboard_apply_selected_model_filter"] = False
        st.rerun()

    sig_m = filter_model_context_df(signals, model_id, selected_symbols, selected_accounts)
    ord_m = filter_model_context_df(orders, model_id, selected_symbols, selected_accounts)
    fill_m = filter_model_context_df(fills, model_id, selected_symbols, selected_accounts)
    pos_m = filter_model_context_df(positions, model_id, selected_symbols, selected_accounts)
    trade_m = filter_model_context_df(trades, model_id, selected_symbols, selected_accounts)
    row = registry[registry["model_id"].astype(str) == str(model_id)].head(1) if not registry.empty and "model_id" in registry.columns else pd.DataFrame()

    realized = pd.to_numeric(pos_m.get("realized_pnl"), errors="coerce").sum() if not pos_m.empty and "realized_pnl" in pos_m.columns else 0
    unrealized = pd.to_numeric(pos_m.get("unrealized_pnl"), errors="coerce").sum() if not pos_m.empty and "unrealized_pnl" in pos_m.columns else 0
    exposure = pd.to_numeric(pos_m.get("market_value"), errors="coerce").abs().sum() if not pos_m.empty and "market_value" in pos_m.columns else 0
    k = st.columns(6)
    k[0].metric("Señales", len(sig_m))
    k[1].metric("Órdenes", len(ord_m))
    k[2].metric("Fills", len(fill_m))
    k[3].metric("Posiciones", len(pos_m))
    k[4].metric("Exposición", fmt_money(exposure))
    k[5].metric("PnL abierto", fmt_money(unrealized))

    st.markdown("#### Resumen")
    reg_cols = [c for c in ["model_id", "status", "acceptance_status", "symbol_scope", "timeframe", "training_scope", "is_active", "strategy_return", "max_drawdown", "profit_factor", "trade_count"] if c in row.columns]
    show_simple_df(st, row[reg_cols] if reg_cols and not row.empty else row, height=120, pct_cols=["strategy_return", "max_drawdown"])

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Posiciones")
        show_simple_df(
            st,
            build_open_positions_view(pos_m, trade_m),
            height=260,
            pct_cols=["pnl_pct", "exposure_pct", "dist_tp_pct", "dist_sl_pct"],
            money_cols=["avg_entry_price", "current_price", "market_value", "realized_pnl"],
            empty="Sin posiciones para este modelo.",
        )
    with c2:
        st.markdown("#### Últimas señales")
        sig_cols = [c for c in ["datetime_utc", "symbol", "timeframe", "account_mode", "signal", "confidence", "entry_price", "take_profit_price", "stop_loss_price"] if c in sig_m.columns]
        show_simple_df(st, sig_m[sig_cols].head(50) if sig_cols and not sig_m.empty else sig_m.head(50), height=260, money_cols=["entry_price", "take_profit_price", "stop_loss_price"], empty="Sin señales para este modelo.")

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("#### Órdenes")
        order_cols = [c for c in ["created_at_utc", "order_id", "symbol", "side", "quantity", "fill_price", "status", "account_mode", "reason"] if c in ord_m.columns]
        show_simple_df(st, ord_m[order_cols].head(80) if order_cols and not ord_m.empty else ord_m.head(80), height=260, money_cols=["fill_price"], empty="Sin órdenes para este modelo.")
    with c4:
        st.markdown("#### Fills")
        fill_cols = [c for c in ["timestamp_utc", "fill_id", "order_id", "symbol", "quantity", "price", "fee_usdt", "account_mode"] if c in fill_m.columns]
        show_simple_df(st, fill_m[fill_cols].head(80) if fill_cols and not fill_m.empty else fill_m.head(80), height=260, money_cols=["price", "fee_usdt"], empty="Sin fills para este modelo.")


def render_model_equity_matrix(st, snapshots: pd.DataFrame, *, key_prefix: str) -> None:
    st.subheader("Equity by model")
    if snapshots.empty or not {"datetime_utc", "equity", "model_id"}.issubset(snapshots.columns):
        st.info("No per-model portfolio snapshots available for the selected filters.")
        return
    df = snapshots.copy()
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True, errors="coerce")
    df["equity"] = pd.to_numeric(df["equity"], errors="coerce")
    df = df.dropna(subset=["datetime_utc", "equity", "model_id"])
    if df.empty:
        st.info("No plottable per-model equity rows for the selected filters.")
        return
    fig = px.line(df.sort_values("datetime_utc"), x="datetime_utc", y="equity", color="model_id", line_group="account_mode" if "account_mode" in df.columns else None)
    fig.update_layout(yaxis_title="Equity")
    safe_plotly_chart(st, fig, height=420, key=f"{key_prefix}_model_equity_{len(df)}")


def render_symbol_bot_matrix(st, registry: pd.DataFrame, positions: pd.DataFrame, orders: pd.DataFrame) -> None:
    st.subheader("Symbol-separated bot allocation")
    if registry.empty or "model_id" not in registry.columns:
        st.info("No registered models available to map by symbol.")
        return
    rows: list[dict[str, Any]] = []
    for _, row in registry.iterrows():
        symbols_raw = str(row.get("symbol_scope", "") or "")
        symbols = [s.strip().upper() for s in symbols_raw.split(",") if s.strip()]
        for symbol in symbols:
            model_id = str(row.get("model_id", ""))
            pos_qty = None
            if not positions.empty and {"model_id", "symbol", "quantity"}.issubset(positions.columns):
                match = positions[(positions["model_id"].astype(str) == model_id) & (positions["symbol"].astype(str) == symbol)]
                if not match.empty:
                    pos_qty = pd.to_numeric(match["quantity"], errors="coerce").sum()
            last_order = None
            if not orders.empty and {"model_id", "symbol", "created_at_utc"}.issubset(orders.columns):
                match = orders[(orders["model_id"].astype(str) == model_id) & (orders["symbol"].astype(str) == symbol)].copy()
                if not match.empty:
                    match["created_at_utc"] = pd.to_datetime(match["created_at_utc"], utc=True, errors="coerce")
                    last_order = match.sort_values("created_at_utc").tail(1).iloc[0].get("status")
            rows.append(
                {
                    "symbol": symbol,
                    "model_id": model_id,
                    "status": row.get("status"),
                    "training_scope": row.get("training_scope"),
                    "signal_enabled": row.get("dashboard_signal_enabled"),
                    "paper_enabled": row.get("dashboard_paper_enabled"),
                    "return": row.get("strategy_return"),
                    "excess_return": row.get("excess_return"),
                    "sharpe": row.get("sharpe"),
                    "drawdown": row.get("max_drawdown"),
                    "profit_factor": row.get("profit_factor"),
                    "risk_adjusted_score": row.get("risk_adjusted_score"),
                    "position_qty": pos_qty,
                    "last_order_status": last_order,
                }
            )
    matrix = pd.DataFrame(rows)
    if matrix.empty:
        st.info("No symbol scope data found in model registry.")
        return
    sort_cols = ["symbol", "risk_adjusted_score"] if "risk_adjusted_score" in matrix.columns else ["symbol"]
    show_df(st, matrix.sort_values(sort_cols, ascending=[True, False] if len(sort_cols) == 2 else True), height=280)


def merge_model_controls(registry: pd.DataFrame, model_control: pd.DataFrame) -> pd.DataFrame:
    if registry.empty:
        return registry
    out = registry.copy()
    if not model_control.empty and "model_id" in model_control.columns:
        cols = [c for c in ["model_id", "signal_enabled", "paper_enabled", "live_enabled", "updated_by", "updated_at_utc"] if c in model_control.columns]
        out = out.merge(model_control[cols].drop_duplicates("model_id"), on="model_id", how="left", suffixes=("", "_control"))
    for col in ["signal_enabled", "paper_enabled", "live_enabled"]:
        if col not in out.columns:
            out[col] = pd.NA
    if "is_active" in out.columns:
        out["dashboard_signal_enabled"] = out["signal_enabled"].fillna(out["is_active"]).fillna(0).astype(int)
    else:
        out["dashboard_signal_enabled"] = out["signal_enabled"].fillna(0).astype(int)
    out["dashboard_paper_enabled"] = out["paper_enabled"].fillna(0).astype(int)
    out["dashboard_live_enabled"] = out["live_enabled"].fillna(0).astype(int)
    return out


def render_action_result(st, fn, *args, **kwargs) -> None:
    try:
        action_id = fn(*args, **kwargs)
        st.success(f"Audited action recorded. action_id={action_id}")
    except Exception as exc:
        st.error(f"Action rejected: {exc}")


def json_safe_value(value: Any) -> Any:
    if isinstance(value, (dict, list, int, float, bool)) or value is None:
        return value
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return str(value)


def render_models_tab(st, registry: pd.DataFrame, model_control: pd.DataFrame, requested_by: str, selected_models: list[str]) -> None:
    st.subheader("Model registry, metrics and safe controls")
    merged = merge_model_controls(registry, model_control)
    if not merged.empty:
        merged = apply_selection_filters(merged, model_ids=selected_models)
    comparison = data.load_model_comparison()
    comparison = apply_selection_filters(comparison, model_ids=selected_models)
    if not comparison.empty and {"model_id", "strategy_return"}.issubset(comparison.columns):
        plot_df = comparison.dropna(subset=["strategy_return"]).head(30)
        if not plot_df.empty:
            fig = px.bar(plot_df, x="model_id", y="strategy_return", color="status", hover_data=[c for c in ["sharpe", "max_drawdown", "profit_factor", "trade_count"] if c in plot_df.columns])
            fig.update_yaxes(tickformat=".1%")
            safe_plotly_chart(st, fig, height=390, key=f"models_comparison_{len(plot_df)}_{','.join(selected_models) if selected_models else 'all'}")

    preferred = [
        "model_id", "symbol_scope", "timeframe", "status", "acceptance_status", "is_active",
        "dashboard_signal_enabled", "dashboard_paper_enabled", "dashboard_live_enabled",
        "training_ts_utc", "train_start", "train_end", "test_start", "test_end",
        "feature_version", "label_version", "model_path", "risk_adjusted_score",
        "strategy_return", "buy_hold_return", "excess_return", "sharpe",
        "max_drawdown", "profit_factor", "trade_count",
        "f1_macro", "accuracy", "precision_macro", "recall_macro",
        "rejection_reasons_json_parsed",
    ]
    show_df(st, merged[[c for c in preferred if c in merged.columns]] if not merged.empty else merged, height=430, empty="No models registered. Run model maintenance/training first.")

    if merged.empty or "model_id" not in merged.columns:
        st.info("Model controls are pending until model_registry has rows.")
        return

    model_ids = merged["model_id"].dropna().astype(str).tolist()
    selected_model = st.selectbox("Inspect/control model", model_ids, key="model_detail_select")
    row = merged[merged["model_id"].astype(str) == selected_model].iloc[0]
    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("#### Model details")
        detail = {k: row.get(k) for k in preferred if k in row.index}
        render_key_value_table(st, "Selected model", {k: json_safe_value(v) for k, v in detail.items()}, height=300)
        metrics_raw = row.get("metrics_json_parsed", {})
        if isinstance(metrics_raw, dict) and metrics_raw:
            st.markdown("#### Metrics")
            metrics_df = flatten_nested_metrics(metrics_raw)
            show_df(st, metrics_df, height=260, empty="No metrics available for this model.")
        else:
            st.info("No metrics available for this model.")
    with c2:
        st.markdown("#### Safe model controls")
        st.caption("These controls update model_control/model_registry and queue pending actions. They do not send orders.")
        a, b = st.columns(2)
        if a.button("Activate signal generation", key=f"activate_{selected_model}", use_container_width=True):
            render_action_result(st, controls.activate_model, selected_model, requested_by)
        if b.button("Deactivate signal generation", key=f"deactivate_{selected_model}", use_container_width=True):
            render_action_result(st, controls.deactivate_model, selected_model, requested_by)
        a, b = st.columns(2)
        if a.button("Enable paper trading", key=f"paper_on_{selected_model}", use_container_width=True):
            render_action_result(st, controls.set_model_paper, selected_model, True, requested_by)
        if b.button("Disable paper trading", key=f"paper_off_{selected_model}", use_container_width=True):
            render_action_result(st, controls.set_model_paper, selected_model, False, requested_by)
        st.warning("Live runtime is controlled from the Control tab and requires password plus external environment gates.")
        st.caption(controls.live_trading_locked_reason())

    reports = data.load_latest_report_summary()
    report_rows = []
    for name, payload in reports.items():
        if isinstance(payload, dict) and payload:
            economic = payload.get("economic", {}) if isinstance(payload.get("economic"), dict) else {}
            classification = payload.get("classification", {}) if isinstance(payload.get("classification"), dict) else {}
            report_rows.append(
                {
                    "report": name,
                    "model_id": payload.get("model_id"),
                    "status": payload.get("status") or payload.get("acceptance_status"),
                    "return": economic.get("strategy_return"),
                    "sharpe": economic.get("sharpe"),
                    "drawdown": economic.get("max_drawdown"),
                    "trades": economic.get("trade_count"),
                    "accuracy": classification.get("accuracy"),
                    "f1_macro": classification.get("f1_macro"),
                    "source": payload.get("_source_file"),
                }
            )
    st.markdown("#### Latest report summaries")
    show_df(st, pd.DataFrame(report_rows), height=220, empty="No report summaries available.")


def render_bot_control_tab(st, status: dict[str, Any], requested_by: str) -> None:
    st.subheader("Bot Control - audited operator intents")
    st.caption("Authenticated buttons execute safe server-side operations on the VPS and audit every action in SQLite. They do not place exchange orders or enable live trading.")
    if controls.server_actions_enabled():
        st.success("Server actions enabled: dashboard can start/stop paper bot services and launch maintenance jobs.")
    else:
        st.warning("Server actions disabled: actions will be queued but not executed. Set DASHBOARD_ALLOW_SERVER_ACTIONS=true on the VPS.")
    c = st.columns(4)
    action_buttons = [
        ("Request bot start", "START_BOT"),
        ("Request bot stop", "STOP_BOT"),
        ("Pause signal generation", "PAUSE_SIGNALS"),
        ("Resume signal generation", "RESUME_SIGNALS"),
        ("Enable paper trading", "ENABLE_PAPER"),
        ("Disable paper trading", "DISABLE_PAPER"),
        ("Refresh data", "REFRESH_DATA"),
        ("Run data quality check", "RUN_DATA_CHECK"),
    ]
    for idx, (label, action) in enumerate(action_buttons):
        if c[idx % 4].button(label, key=f"bot_action_{action}", use_container_width=True):
            params = {"source": "dashboard", "mode": status.get("mode"), "dry_run": status.get("safety_flags", {}).get("DRY_RUN")}
            params.update({"symbols": ",".join(status.get("symbols") or []), "timeframe": status.get("timeframe")})
            render_action_result(st, controls.execute_bot_action, action, params, requested_by)

    st.markdown("#### Request new model training")
    with st.form("request_training"):
        symbols = st.text_input("Symbols", value=",".join(status.get("symbols") or []))
        timeframe = st.text_input("Timeframe", value=str(status.get("timeframe") or "1h"))
        training_scope = st.selectbox("Training scope", ["both", "multi_symbol", "per_symbol"], index=0)
        max_attempts = st.number_input("Max attempts", min_value=1, max_value=500, value=50, step=1)
        submitted = st.form_submit_button("Queue training request")
    if submitted:
        render_action_result(
            st,
            controls.execute_bot_action,
            "REQUEST_RETRAIN",
            {"symbols": symbols, "timeframe": timeframe, "training_scope": training_scope, "max_attempts": int(max_attempts)},
            requested_by,
        )

    st.markdown("#### Runtime/safety state")
    c1, c2 = st.columns([1, 1])
    with c1:
        render_key_value_table(st, "Runtime/safety state", status, height=520)
    with c2:
        st.markdown("#### Pending/recent actions")
        show_df(st, controls.load_control_actions(300), height=520, empty="No dashboard actions recorded yet.")


def render_simple_control_panel(st, status: dict[str, Any], requested_by: str) -> None:
    st.subheader("Panel de control")
    st.caption("Panel simplificado para VPS. El login permite gestionar el sistema; live trading conserva los guardarraíles obligatorios de entorno.")

    st.markdown("### Kill switch")
    st.error("Detiene procesos del bot y desactiva signals, paper y live en runtime_config. No borra datos ni modelos.")
    kill_cols = st.columns([1, 2])
    if kill_cols[0].button("KILL SWITCH", type="primary", use_container_width=True):
        render_action_result(st, controls.execute_bot_action, "KILL_SWITCH", {"source": "dashboard"}, requested_by)
    kill_cols[1].caption("Usar ante comportamiento inesperado, pérdidas, conectividad mala o cualquier duda operativa.")

    st.markdown("### Bot")
    c = st.columns(4)
    action_specs = [
        ("Start", "START_BOT"),
        ("Stop", "STOP_BOT"),
        ("Pause signals", "PAUSE_SIGNALS"),
        ("Resume signals", "RESUME_SIGNALS"),
        ("Enable paper", "ENABLE_PAPER"),
        ("Disable paper", "DISABLE_PAPER"),
        ("Refresh data", "REFRESH_DATA"),
        ("Data check", "RUN_DATA_CHECK"),
    ]
    for idx, (label, action) in enumerate(action_specs):
        if c[idx % 4].button(label, key=f"simple_{action}", use_container_width=True):
            params = {"symbols": ",".join(status.get("symbols") or []), "timeframe": status.get("timeframe"), "source": "dashboard"}
            render_action_result(st, controls.execute_bot_action, action, params, requested_by)

    st.markdown("### Entrenamiento")
    with st.form("simple_training"):
        c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
        symbols = c1.text_input("Symbols", value=",".join(status.get("symbols") or []))
        timeframe = c2.text_input("Timeframe", value=str(status.get("timeframe") or "1h"))
        scope = c3.selectbox("Scope", ["both", "multi_symbol", "per_symbol"], index=0)
        attempts = c4.number_input("Attempts", min_value=1, max_value=500, value=50, step=1)
        if st.form_submit_button("Train new model", use_container_width=True):
            render_action_result(
                st,
                controls.execute_bot_action,
                "REQUEST_RETRAIN",
                {"symbols": symbols, "timeframe": timeframe, "training_scope": scope, "max_attempts": int(attempts)},
                requested_by,
            )

    st.markdown("### Trading real")
    st.warning("No se permite activar trading real saltándose `.env`. El botón requiere contraseña y que el VPS ya tenga activados explícitamente ENABLE_LIVE_TRADING, ENABLE_REAL_ORDER_EXECUTION, ENABLE_REAL_BINANCE_ACCOUNT, DRY_RUN=false y LIVE_TRADING_ALLOWED=true.")
    st.caption(controls.live_trading_locked_reason())
    live_cols = st.columns([1, 1])
    with live_cols[0].form("enable_live_form"):
        password = st.text_input("Password para activar live runtime", type="password")
        confirm = st.checkbox("Confirmo que entiendo que puede enviar órdenes reales si los flags externos ya están activos")
        submitted = st.form_submit_button("Activate live runtime", use_container_width=True)
        if submitted:
            if not confirm:
                st.error("Confirma explícitamente la acción.")
            elif not auth.verify_user_password(password):
                st.error("Password incorrecta.")
            else:
                render_action_result(st, controls.execute_bot_action, "ENABLE_LIVE_RUNTIME", {"source": "dashboard_reauth"}, requested_by)
    with live_cols[1]:
        if st.button("Disable live runtime", use_container_width=True):
            render_action_result(st, controls.execute_bot_action, "DISABLE_LIVE_RUNTIME", {"source": "dashboard"}, requested_by)

    st.markdown("### Últimas acciones")
    show_df(st, controls.load_control_actions(80), height=300, empty="No actions yet.")


def render_configuration_tab(st, requested_by: str) -> None:
    st.subheader("Runtime configuration")
    st.caption("Safe runtime settings are stored in SQLite runtime_config with audit history. Secrets and API keys are never editable here.")
    cfg = controls.load_runtime_config()
    show_df(st, cfg, height=360, empty="No runtime config found. It will be initialized when the DB is available.")
    schema = controls.RUNTIME_CONFIG_SCHEMA
    key = st.selectbox("Configuration key", sorted(schema.keys()), key="config_key")
    spec = schema[key]
    current_value = None
    if not cfg.empty and "key" in cfg.columns:
        rows = cfg[cfg["key"].astype(str) == key]
        if not rows.empty:
            current_value = rows.iloc[0].get("value")
    if current_value is None:
        current_value = spec.get("default", "")
    st.info(f"{spec.get('description', '')} Type: {spec.get('type')}.")
    if spec.get("dangerous"):
        st.warning("Dangerous setting. Enabling live trading is blocked by dashboard guardrails unless external env gates are already explicitly set.")
    with st.form("runtime_config_editor"):
        if spec["type"] == "bool":
            proposed = st.checkbox("New value", value=str(current_value).lower() in {"true", "1", "yes", "on"})
        else:
            proposed = st.text_input("New value", value=str(current_value))
        reason = st.text_input("Change reason", value="dashboard_operator_update")
        st.write("Preview")
        show_df(
            st,
            pd.DataFrame(
                [
                    {"field": "key", "value": key},
                    {"field": "old_value", "value": current_value},
                    {"field": "new_value", "value": proposed},
                    {"field": "requested_by", "value": requested_by},
                    {"field": "audit", "value": True},
                ]
            ),
            height=190,
        )
        confirm = st.checkbox("Confirm audited configuration update")
        submitted = st.form_submit_button("Save configuration change")
    if submitted:
        if not confirm:
            st.error("Please confirm the audited update before saving.")
        else:
            try:
                controls.update_runtime_config(key, proposed, requested_by, reason=reason)
                st.success("Configuration saved to runtime_config and audit log.")
            except Exception as exc:
                st.error(f"Configuration rejected: {exc}")

    st.markdown("#### Configuration audit")
    audit = data.read_table("runtime_config_audit", 300, "updated_at_utc")
    show_df(st, audit, height=330, empty="No configuration changes recorded yet.")


def render_no_data_page(st, inventory: dict[str, Any]) -> None:
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.error("No SQLite database is available for the dashboard.")
    st.write(f"Expected DB path: `{inventory['db_path']}`")
    st.write("Run these commands first:")
    st.code(
        "python src/db_utils.py --init --check-schema\n"
        "python src/realtime_ingestor.py --symbols BTCUSDT ETHUSDT SOLUSDT --timeframe 1h\n"
        "python src/model_maintenance.py --target-accepted-models 5 --max-attempts 50\n"
        "python src/trading_bot.py --mode paper --paper-mode per-model --run-once",
        language="bash",
    )
    st.write("Available reports/logs can still be used once the DB path is configured correctly.")
    render_key_value_table(st, "Dashboard inventory", inventory, height=260)
    st.markdown("</div>", unsafe_allow_html=True)


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="AI Trading Bot - Operations", page_icon=None, layout="wide")
    inject_css(st)
    requested_by = auth.require_login(st)
    if requested_by is None:
        return

    status = data.load_system_status()
    inventory = data.load_data_inventory()
    if inventory.get("db_exists"):
        try:
            controls.ensure_dashboard_tables()
            inventory = data.load_data_inventory()
        except Exception as exc:
            st.warning(f"Dashboard control tables are not writable: {exc}")
    render_header(st, status)

    with st.sidebar:
        st.header("Operator")
        st.write(f"User: **{requested_by}**")
        if auth.load_auth_config().enabled:
            st.success("AUTH ENABLED")
            if st.button("Logout", use_container_width=True):
                auth.logout(st)
                st.rerun()
        else:
            st.warning("AUTH DISABLED")
        st.divider()
        st.header("Controls")
        if st.button("Refresh now", use_container_width=True):
            st.rerun()
        st.caption(f"Recommended refresh: {status.get('refresh_seconds')}s")
        st.divider()
        st.subheader("Runtime")
        st.write(f"Mode: **{status.get('mode')}**")
        st.write(f"State: **{status.get('state')}**")
        st.write(f"DB: `{status.get('db_path')}`")
        st.write(f"Latest data: `{status.get('latest_data_ts') or 'N/A'}`")
        st.divider()
        st.subheader("Safety flags")
        flags = pd.DataFrame([{"flag": k, "value": v} for k, v in (status.get("safety_flags") or {}).items()])
        if flags.empty:
            st.info("No safety flags available.")
        else:
            st.dataframe(flags, use_container_width=True, height=230, hide_index=True)
        if status.get("real_order_possible"):
            st.error("LIVE TRADING flags allow real orders. Verify intentionally.")
        else:
            st.success("Real order execution is blocked by default.")

    if not inventory.get("db_exists"):
        render_no_data_page(st, inventory)
        return

    summary = data.load_portfolio_summary()
    coverage = data.load_data_coverage()
    gaps = data.load_data_gaps(open_only=True)
    registry = data.load_model_registry()
    model_row = None
    if not registry.empty and "model_id" in registry.columns:
        active = registry[registry["model_id"].astype(str) == str(status.get("active_model_id"))]
        model_row = active.iloc[0] if not active.empty else registry.iloc[0]

    signals = data.load_recent_signals(1000)
    orders = data.load_recent_orders(1000)
    positions = data.load_open_positions()
    fills = data.load_recent_fills(1000)
    snapshots = data.load_portfolio_snapshots()
    model_control = data.load_model_control()

    all_symbols = sorted(set(status.get("symbols") or ["BTCUSDT"]) | set(signals["symbol"].dropna().astype(str).tolist() if "symbol" in signals.columns else []) | set(orders["symbol"].dropna().astype(str).tolist() if "symbol" in orders.columns else []) | set(positions["symbol"].dropna().astype(str).tolist() if "symbol" in positions.columns else []))
    all_models = sorted(set(registry["model_id"].dropna().astype(str).tolist() if "model_id" in registry.columns else []) | set(signals["model_id"].dropna().astype(str).tolist() if "model_id" in signals.columns else []) | set(orders["model_id"].dropna().astype(str).tolist() if "model_id" in orders.columns else []) | set(snapshots["model_id"].dropna().astype(str).tolist() if "model_id" in snapshots.columns else []))
    all_accounts = sorted(set(orders["account_mode"].dropna().astype(str).tolist() if "account_mode" in orders.columns else []) | set(positions["account_mode"].dropna().astype(str).tolist() if "account_mode" in positions.columns else []) | set(snapshots["account_mode"].dropna().astype(str).tolist() if "account_mode" in snapshots.columns else []))
    all_statuses = sorted(orders["status"].dropna().astype(str).unique().tolist()) if "status" in orders.columns else []
    all_signal_labels = sorted((signals["signal"].dropna().astype(str).unique().tolist() if "signal" in signals.columns else signals["signal_label"].dropna().astype(str).unique().tolist() if "signal_label" in signals.columns else []))

    with st.sidebar:
        st.divider()
        st.subheader("Filtros")
        selected_symbols: list[str] = []
        selected_models: list[str] = []
        selected_accounts: list[str] = []
        selected_order_statuses: list[str] = []
        selected_signal_labels: list[str] = []
        symbol_options = all_symbols or ["BTCUSDT"]
        view_real_portfolio = st.toggle("Ver cartera real", value=False, help="OFF = demo/paper/testnet/local. ON = real/live si existen datos en SQLite.")
        selected_accounts = choose_account_modes(all_accounts, view_real_portfolio)
        account_label = "REAL" if view_real_portfolio else "DEMO / PAPER"
        if view_real_portfolio and selected_accounts == ["__NO_REAL_ACCOUNT_MODE__"]:
            st.warning("No hay datos de cartera real en SQLite para los filtros actuales.")
        else:
            st.caption(f"Vista cartera: {account_label}")
        selected_symbols = st.multiselect("Símbolos", symbol_options, default=[], help="Vacío = todos los símbolos en tablas/KPIs.")
        chart_symbol_options = selected_symbols or symbol_options
        symbol = st.selectbox("Símbolo del gráfico", chart_symbol_options, index=0)
        timeframe = st.text_input("Timeframe", value=str(status.get("timeframe") or "1h"))
        price_limit = st.slider("Velas", 100, 3000, 800, 100)
        with st.expander("Advanced filters", expanded=False):
            selected_models = st.multiselect("Models", all_models, default=[], help="Empty = all models")
            override_accounts = st.multiselect("Account modes override", all_accounts, default=[], help="Vacío = usa el switch real/demo")
            if override_accounts:
                selected_accounts = override_accounts
            selected_order_statuses = st.multiselect("Order statuses", all_statuses, default=[], help="Affects order charts/tables")
            selected_signal_labels = st.multiselect("Signal labels", all_signal_labels, default=[], help="Affects signal charts/tables")
            if selected_symbols and symbol not in selected_symbols:
                st.caption("The chart symbol is independent from table filters.")

    context_model_id = st.session_state.get("dashboard_selected_model_id")
    if context_model_id:
        with st.sidebar:
            st.divider()
            st.subheader("Modelo clicado")
            st.code(str(context_model_id), language=None)
            st.checkbox(
                "Aplicar modelo clicado a todo",
                value=bool(st.session_state.get("dashboard_apply_selected_model_filter", True)),
                key="dashboard_apply_selected_model_filter",
                help="Cuando está activo, todas las tablas con model_id se filtran por el modelo seleccionado.",
            )
            if st.button("Quitar modelo clicado", use_container_width=True):
                st.session_state.pop("dashboard_selected_model_id", None)
                st.session_state["dashboard_apply_selected_model_filter"] = False
                st.rerun()
    if context_model_id and st.session_state.get("dashboard_apply_selected_model_filter", True):
        selected_models = [str(context_model_id)]

    signals_f = apply_selection_filters(signals, model_ids=selected_models, symbols=selected_symbols, account_modes=selected_accounts, signal_labels=selected_signal_labels)
    orders_f = apply_selection_filters(orders, model_ids=selected_models, symbols=selected_symbols, account_modes=selected_accounts, statuses=selected_order_statuses)
    positions_f = apply_selection_filters(positions, model_ids=selected_models, symbols=selected_symbols, account_modes=selected_accounts)
    fills_f = apply_selection_filters(fills, model_ids=selected_models, symbols=selected_symbols, account_modes=selected_accounts)
    snapshots_f = apply_selection_filters(snapshots, model_ids=selected_models, account_modes=selected_accounts)
    registry_f = apply_selection_filters(registry, model_ids=selected_models)
    trades_all = data.read_table("trades", 1000, "created_at_utc")
    trades_f = apply_selection_filters(trades_all, model_ids=selected_models, symbols=selected_symbols, account_modes=selected_accounts)
    if selected_accounts == ["__NO_REAL_ACCOUNT_MODE__"]:
        view_summary = {
            "total_equity": None, "cash_usdt": None, "invested_value": None, "total_pnl": None,
            "daily_pnl": None, "total_return": None, "daily_return": None, "exposure_pct": None,
            "max_drawdown": None, "source": "no_real_account_data",
        }
    else:
        view_summary = calculate_portfolio_view_summary(snapshots_f, positions_f, summary)
    coverage_f = apply_selection_filters(coverage, symbols=selected_symbols)
    gaps_f = apply_selection_filters(gaps, symbols=selected_symbols)
    filtered_equity = build_equity_curve_from_snapshots(snapshots_f)
    if filtered_equity.empty and not selected_models and not view_real_portfolio:
        filtered_equity = data.load_equity_curve()

    tabs = st.tabs(["Principal", "Posiciones", "Modelo", "Sistema", "Control"])

    with tabs[0]:
        render_portfolio_position_header(st, view_summary, account_label)
        render_price_signals(
            st,
            symbol,
            timeframe,
            data.load_price_series(symbol, timeframe, price_limit),
            signals_f,
            orders_f,
            positions_f,
            key_prefix="principal",
        )
        selected_model_id = render_main_models_table(st, registry_f, model_control, requested_by, selected_models)
        st.caption("Vista simplificada: cartera, gr?fico y modelos. Usa las pesta?as para detalle.")

    with tabs[1]:
        st.subheader("Posiciones abiertas")
        positions_view = build_open_positions_view(positions_f, trades_f)
        show_simple_df(
            st,
            positions_view,
            height=420,
            pct_cols=["pnl_pct", "exposure_pct", "dist_tp_pct", "dist_sl_pct"],
            money_cols=["avg_entry_price", "current_price", "market_value", "realized_pnl", "tp_price", "sl_price"],
            empty="No hay posiciones abiertas para la vista seleccionada.",
        )
        c_pos1, c_pos2 = st.columns([1.4, 1])
        with c_pos1:
            render_equity(st, filtered_equity, key_prefix="positions_simple")
        with c_pos2:
            render_exposure(st, build_filtered_exposure_breakdown(positions_f, view_summary), key_prefix="positions_simple")

    with tabs[2]:
        model_for_detail = st.session_state.get("dashboard_selected_model_id")
        if not model_for_detail and not registry_f.empty and "model_id" in registry_f.columns:
            model_for_detail = str(registry_f.iloc[0].get("model_id"))
        render_selected_model_simple(
            st,
            model_id=model_for_detail,
            registry=registry,
            signals=signals,
            orders=orders,
            fills=fills,
            positions=positions,
            trades=trades_all,
            selected_symbols=selected_symbols,
            selected_accounts=selected_accounts,
        )

    with tabs[3]:
        st.subheader("Sistema y relaciones")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Datos")
            show_simple_df(st, coverage_f, height=220, empty="No coverage rows.")
            show_simple_df(st, gaps_f, height=220, empty="No open gaps.")
        with c2:
            st.markdown("#### Seguridad")
            risk_events = apply_selection_filters(
                data.read_table("risk_events", 300, "created_at_utc"),
                model_ids=selected_models,
                symbols=selected_symbols,
                account_modes=selected_accounts,
            )
            show_simple_df(st, risk_events, height=220, empty="No risk events.")
            relationship_issues = apply_selection_filters(data.load_relationship_issues(500), model_ids=selected_models, symbols=selected_symbols)
            if relationship_issues.empty:
                st.success("Relaciones OK: no hay errores model/order/proposal/trade en la vista actual.")
            else:
                st.warning(f"{len(relationship_issues)} relaciones rotas detectadas.")
                show_simple_df(st, relationship_issues, height=220)
        with st.expander("Sincronizaci?n Binance / cuenta", expanded=False):
            account_snapshots = apply_selection_filters(data.read_table("account_snapshots", 100, "created_at_utc"), account_modes=selected_accounts)
            balance_snapshots = apply_selection_filters(data.read_table("balance_snapshots", 200, "timestamp_utc"), account_modes=selected_accounts)
            reconciliation_events = apply_selection_filters(data.read_table("reconciliation_events", 200, "created_at_utc"), account_modes=selected_accounts)
            show_simple_df(st, account_snapshots, height=180, empty="No account snapshots.")
            show_simple_df(st, balance_snapshots, height=180, empty="No balance snapshots.")
            show_simple_df(st, reconciliation_events, height=180, empty="No reconciliation events.")

    with tabs[4]:
        render_compact_kpis(st, view_summary, status, registry_f, positions_f, gaps_f)
        render_simple_control_panel(st, status, requested_by)


if __name__ == "__main__":
    main()
