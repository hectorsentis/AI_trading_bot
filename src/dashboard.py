import json
import sqlite3
from html import escape

import pandas as pd

from config import (
    BINANCE_ENV,
    BINANCE_REAL_API_KEY,
    BINANCE_TESTNET_API_KEY,
    DASHBOARD_REFRESH_SECONDS,
    DB_FILE,
    DRY_RUN,
    ENABLE_LIVE_TRADING,
    ENABLE_REAL_BINANCE_ACCOUNT,
    ENABLE_REAL_ORDER_EXECUTION,
    ENABLE_TESTNET_PAPER_TRADING,
    MAX_DAILY_LOSS_USDT,
    MAX_EXPOSURE_PER_MODEL_USDT,
    MAX_EXPOSURE_TOTAL_USDT,
    MAX_ORDER_NOTIONAL_USDT,
    MIN_ORDER_NOTIONAL_USDT,
)
from db_utils import init_research_tables


STATUS_COLORS = {
    "candidate": "#94a3b8",
    "validation_rejected": "#ef4444",
    "validation_accepted": "#38bdf8",
    "backtest_rejected": "#f97316",
    "backtest_accepted": "#22c55e",
    "paper_active": "#14b8a6",
    "paper_rejected": "#dc2626",
    "paper_validated": "#84cc16",
    "real_ready": "#a855f7",
    "real_active": "#facc15",
    "real_paused": "#fb923c",
    "real_rejected": "#b91c1c",
    "archived": "#64748b",
}


def _query(table: str, limit: int = 1000, order_by: str | None = None) -> pd.DataFrame:
    init_research_tables()
    conn = sqlite3.connect(DB_FILE)
    try:
        order = f" ORDER BY {order_by}" if order_by else ""
        return pd.read_sql_query(f"SELECT * FROM {table}{order} LIMIT {int(limit)}", conn)
    except Exception:
        return pd.DataFrame()
    finally:
        conn.close()


def _counts() -> pd.DataFrame:
    init_research_tables()
    conn = sqlite3.connect(DB_FILE)
    try:
        tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name", conn)
        rows = []
        for table in tables["name"].tolist():
            try:
                n = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            except Exception:
                n = None
            rows.append({"table": table, "rows": n})
        return pd.DataFrame(rows)
    finally:
        conn.close()


def _fmt_money(value) -> str:
    try:
        return f"${float(value):,.2f}"
    except Exception:
        return "$0.00"


def _fmt_pct(value) -> str:
    try:
        return f"{100 * float(value):.2f}%"
    except Exception:
        return "0.00%"


def _inject_css(st):
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.1rem; padding-bottom: 2rem; max-width: 1500px;}
        div[data-testid="stMetric"] {
            background: linear-gradient(135deg, rgba(15,23,42,.98), rgba(30,41,59,.92));
            border: 1px solid rgba(148,163,184,.18); border-radius: 18px; padding: 14px 16px;
            box-shadow: 0 14px 35px rgba(2,6,23,.20);
        }
        div[data-testid="stMetricLabel"] {color: #cbd5e1;}
        div[data-testid="stMetricValue"] {color: #f8fafc;}
        .hero {
            padding: 20px 24px; border-radius: 24px;
            background: radial-gradient(circle at top left, rgba(20,184,166,.35), transparent 35%),
                        linear-gradient(135deg, #020617 0%, #0f172a 50%, #111827 100%);
            color: white; border: 1px solid rgba(148,163,184,.22);
            box-shadow: 0 20px 55px rgba(2,6,23,.30); margin-bottom: 18px;
        }
        .hero h1 {margin: 0 0 8px 0; font-size: 2.1rem; letter-spacing: -0.03em;}
        .hero p {margin: 0; color: #cbd5e1;}
        .pill {display:inline-block; padding: 4px 10px; border-radius:999px; font-size:.78rem; font-weight:700; margin:2px 4px 2px 0;}
        .pill-ok {background: rgba(34,197,94,.16); color:#86efac; border:1px solid rgba(34,197,94,.35)}
        .pill-warn {background: rgba(245,158,11,.16); color:#fde68a; border:1px solid rgba(245,158,11,.35)}
        .pill-bad {background: rgba(239,68,68,.16); color:#fecaca; border:1px solid rgba(239,68,68,.35)}
        .section-card {border:1px solid rgba(148,163,184,.18); border-radius:18px; padding:16px; background: rgba(15,23,42,.035);}
        .small-muted {color:#64748b; font-size:.88rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _status_badges(registry: pd.DataFrame) -> str:
    if registry.empty or "status" not in registry:
        return '<span class="pill pill-warn">sin modelos</span>'
    counts = registry["status"].fillna("unknown").value_counts().to_dict()
    html = []
    for status, n in counts.items():
        color = STATUS_COLORS.get(str(status), "#94a3b8")
        html.append(
            f'<span class="pill" style="background:{color}22;color:{color};border:1px solid {color}66">{escape(str(status))}: {int(n)}</span>'
        )
    return "".join(html)


def _filter_df(st, df: pd.DataFrame, key_prefix: str) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    cols = st.columns(4)
    for idx, col in enumerate(["model_id", "symbol", "account_mode", "status"]):
        if col in out.columns:
            values = sorted([str(v) for v in out[col].dropna().unique().tolist()])
            selected = cols[idx % 4].multiselect(col, values, key=f"{key_prefix}_{col}")
            if selected:
                out = out[out[col].astype(str).isin(selected)]
    return out


def _streamlit_main():
    import streamlit as st

    st.set_page_config(page_title="Binance Spot Demo Trading Dashboard", page_icon="📈", layout="wide")
    _inject_css(st)

    registry = _query("model_registry", limit=3000, order_by="training_ts_utc DESC")
    orders = _query("orders", limit=3000, order_by="created_at_utc DESC")
    fills = _query("fills", limit=3000, order_by="timestamp_utc DESC")
    positions = _query("positions", limit=3000, order_by="updated_at_utc DESC")
    snaps = _query("portfolio_snapshots", limit=8000, order_by="datetime_utc DESC")
    metrics = _query("paper_model_metrics", limit=3000, order_by="evaluated_at_utc DESC")
    gaps = _query("data_gaps", limit=3000, order_by="detected_at_utc DESC")
    coverage = _query("data_coverage", limit=3000, order_by="updated_at_utc DESC")
    signals = _query("signals", limit=3000, order_by="created_at_utc DESC")
    risk_events = _query("risk_events", limit=3000, order_by="created_at_utc DESC")
    bot_status = _query("bot_status", limit=100, order_by="component ASC")
    if not bot_status.empty and "last_heartbeat_utc" in bot_status.columns:
        now_utc = pd.Timestamp.now(tz="UTC")
        bot_status["heartbeat_age_seconds"] = (now_utc - pd.to_datetime(bot_status["last_heartbeat_utc"], utc=True, errors="coerce")).dt.total_seconds()
        bot_status["effective_status"] = bot_status.apply(lambda r: "stale" if str(r.get("status")) == "running" and float(r.get("heartbeat_age_seconds") or 999999) > 180 else r.get("status"), axis=1)

    real_enabled = ENABLE_LIVE_TRADING and ENABLE_REAL_ORDER_EXECUTION and ENABLE_REAL_BINANCE_ACCOUNT and not DRY_RUN
    services_running = (not bot_status.empty and "effective_status" in bot_status and (bot_status["effective_status"].astype(str) == "running").any())
    runner_running = (not bot_status.empty and "component" in bot_status and "effective_status" in bot_status and ((bot_status["component"].astype(str) == "autonomous_runner") & (bot_status["effective_status"].astype(str) == "running")).any())
    mode_pill = "pill-bad" if real_enabled else "pill-ok"
    st.markdown(
        f"""
        <div class="hero">
          <h1>📈 Binance Spot AI Trading Platform</h1>
          <p>Panel operativo SQLite · Paper multi-modelo · Demo/Testnet separado de real · Sin promesas de rentabilidad.</p>
          <div style="margin-top:12px">
            <span class="pill {mode_pill}">Real execution: {'ENABLED' if real_enabled else 'BLOCKED BY DEFAULT'}</span>
            <span class="pill pill-ok">Environment: {escape(str(BINANCE_ENV))}</span>
            <span class="pill {'pill-ok' if runner_running else 'pill-bad'}">Bot: {'RUNNING' if runner_running else 'OFF/STALE'}</span>
            <span class="pill {'pill-ok' if ENABLE_TESTNET_PAPER_TRADING else 'pill-warn'}">Spot Demo Paper: {ENABLE_TESTNET_PAPER_TRADING}</span>
            <span class="pill {'pill-ok' if DRY_RUN else 'pill-bad'}">DRY_RUN: {DRY_RUN}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    active = int((registry.get("status", pd.Series(dtype=str)) == "paper_active").sum()) if not registry.empty else 0
    validated = int((registry.get("status", pd.Series(dtype=str)) == "paper_validated").sum()) if not registry.empty else 0
    ready = int((registry.get("status", pd.Series(dtype=str)) == "real_ready").sum()) if not registry.empty else 0
    rejected = int(registry.get("status", pd.Series(dtype=str)).astype(str).str.contains("rejected", na=False).sum()) if not registry.empty else 0
    filled = int((orders.get("status", pd.Series(dtype=str)).astype(str).str.upper() == "FILLED").sum()) if not orders.empty else 0
    latest_equity = snaps["equity"].iloc[0] if not snaps.empty and "equity" in snaps else 0

    k1, k2, k3, k4, k5, k6, k7 = st.columns(7)
    k1.metric("Paper active", active)
    k2.metric("Paper validated", validated)
    k3.metric("Real ready", ready)
    k4.metric("Rejected", rejected)
    k5.metric("Filled orders", filled)
    k6.metric("Latest equity", _fmt_money(latest_equity))
    k7.metric("Bot", "RUNNING" if runner_running else "OFF")

    st.markdown(_status_badges(registry), unsafe_allow_html=True)

    with st.sidebar:
        st.header("⚙️ Runtime")
        st.caption(f"DB: `{DB_FILE}`")
        st.caption(f"Refresh recomendado: {DASHBOARD_REFRESH_SECONDS}s")
        st.markdown(f"<span class='pill {'pill-ok' if runner_running else 'pill-bad'}'>Bot: {'RUNNING' if runner_running else 'OFF/STALE'}</span>", unsafe_allow_html=True)
        st.divider()
        st.subheader("Conexiones")
        st.markdown(f"<span class='pill {'pill-ok' if BINANCE_TESTNET_API_KEY else 'pill-warn'}'>Demo API key: {'present' if BINANCE_TESTNET_API_KEY else 'missing'}</span>", unsafe_allow_html=True)
        st.markdown(f"<span class='pill {'pill-warn' if BINANCE_REAL_API_KEY else 'pill-ok'}'>Real API key: {'present' if BINANCE_REAL_API_KEY else 'empty'}</span>", unsafe_allow_html=True)
        st.divider()
        st.subheader("Safety flags")
        st.json({
            "DRY_RUN": DRY_RUN,
            "ENABLE_LIVE_TRADING": ENABLE_LIVE_TRADING,
            "ENABLE_REAL_ORDER_EXECUTION": ENABLE_REAL_ORDER_EXECUTION,
            "ENABLE_REAL_BINANCE_ACCOUNT": ENABLE_REAL_BINANCE_ACCOUNT,
            "real_order_possible": real_enabled,
        })

    tabs = st.tabs(["🏠 Overview", "🧠 Modelos", "💼 Paper trading", "🧾 Órdenes", "📊 Performance", "🧪 Data quality", "🛡️ Risk"])

    with tabs[0]:
        c1, c2 = st.columns([1.2, 1])
        with c1:
            st.subheader("Última cobertura de datos")
            st.dataframe(coverage, use_container_width=True, height=260)
        with c2:
            st.subheader("Conteo de tablas")
            st.dataframe(_counts(), use_container_width=True, height=260)
        st.subheader("Estado live de procesos")
        if bot_status.empty:
            st.warning("No hay heartbeats. El bot aut?nomo parece apagado. Ejecuta .tools\\run.cmd")
        else:
            st.dataframe(bot_status, use_container_width=True, height=220)
        if not orders.empty and "created_at_utc" in orders:
            st.subheader("Actividad reciente")
            st.dataframe(orders.head(20), use_container_width=True, height=300)

    with tabs[1]:
        st.subheader("Registro de modelos")
        view = _filter_df(st, registry, "registry")
        st.dataframe(view, use_container_width=True, height=520)

    with tabs[2]:
        a, b = st.columns([1, 1])
        with a:
            st.subheader("Posiciones por modelo")
            st.dataframe(_filter_df(st, positions, "positions"), use_container_width=True, height=360)
        with b:
            st.subheader("Últimas señales")
            st.dataframe(_filter_df(st, signals, "signals"), use_container_width=True, height=360)
        st.subheader("Equity curve por modelo")
        if not snaps.empty and {"datetime_utc", "equity", "model_id"}.issubset(snaps.columns):
            chart_df = snaps.copy()
            chart_df["datetime_utc"] = pd.to_datetime(chart_df["datetime_utc"], utc=True, errors="coerce")
            chart_df = chart_df.sort_values("datetime_utc")
            st.line_chart(chart_df.pivot_table(index="datetime_utc", columns="model_id", values="equity", aggfunc="last"), height=320)
        else:
            st.info("Aún no hay snapshots de cartera.")

    with tabs[3]:
        st.subheader("Órdenes")
        st.dataframe(_filter_df(st, orders, "orders"), use_container_width=True, height=460)
        st.subheader("Fills")
        st.dataframe(_filter_df(st, fills, "fills"), use_container_width=True, height=320)

    with tabs[4]:
        st.subheader("Ranking y métricas paper")
        if not metrics.empty:
            cols = [c for c in ["model_id", "account_mode", "validation_status", "equity", "total_return", "profit_factor", "win_rate", "max_drawdown", "filled_trades", "evaluated_at_utc"] if c in metrics.columns]
            st.dataframe(metrics[cols], use_container_width=True, height=420)
            numeric = metrics.copy()
            for col in ["total_return", "profit_factor", "win_rate", "max_drawdown"]:
                if col in numeric:
                    numeric[col] = pd.to_numeric(numeric[col], errors="coerce")
            if {"model_id", "total_return"}.issubset(numeric.columns):
                st.bar_chart(numeric.dropna(subset=["total_return"]).set_index("model_id")["total_return"], height=280)
        else:
            st.info("Ejecuta `python src/paper_model_evaluator.py --evaluate-active` para poblar métricas.")

    with tabs[5]:
        c1, c2 = st.columns([1, 1])
        with c1:
            st.subheader("Coverage")
            st.dataframe(coverage, use_container_width=True, height=420)
        with c2:
            st.subheader("Gaps")
            st.dataframe(gaps, use_container_width=True, height=420)

    with tabs[6]:
        st.subheader("Límites configurados")
        r1, r2, r3, r4, r5 = st.columns(5)
        r1.metric("Max/model", _fmt_money(MAX_EXPOSURE_PER_MODEL_USDT))
        r2.metric("Max total", _fmt_money(MAX_EXPOSURE_TOTAL_USDT))
        r3.metric("Max order", _fmt_money(MAX_ORDER_NOTIONAL_USDT))
        r4.metric("Min order", _fmt_money(MIN_ORDER_NOTIONAL_USDT))
        r5.metric("Daily loss", _fmt_money(MAX_DAILY_LOSS_USDT))
        st.subheader("Risk events")
        st.dataframe(_filter_df(st, risk_events, "risk"), use_container_width=True, height=460)


if __name__ == "__main__":
    try:
        _streamlit_main()
    except ModuleNotFoundError:
        print("Streamlit is not installed. Install requirements and run: streamlit run src/dashboard.py")
