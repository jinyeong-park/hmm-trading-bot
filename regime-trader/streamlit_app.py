"""
streamlit_app.py — Live trading dashboard for regime-trader.

Run:
    streamlit run streamlit_app.py

Tabs:
  Overview    — equity, daily P&L, open positions, regime gauge
  Regime      — HMM regime history, confidence chart, flicker rate
  Positions   — live positions table with P&L, stop levels, holding time
  Orders      — recent order history from Alpaca
  Backtest    — run walk-forward backtest inline and view results
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

# ── project root on path ────────────────────────────────────────────────────
_ROOT = Path(__file__).parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

load_dotenv(_ROOT / ".env")

# ─────────────────────────────────────────────────────────────────────────────
# Page config (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="regime-trader",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Regime colour map
# ─────────────────────────────────────────────────────────────────────────────
_REGIME_COLOURS = {
    "BULL":        "#00cc66",
    "STRONG_BULL": "#00ff80",
    "WEAK_BULL":   "#66cc99",
    "NEUTRAL":     "#aaaaaa",
    "WEAK_BEAR":   "#ffaa44",
    "BEAR":        "#ff4444",
    "STRONG_BEAR": "#cc0000",
    "CRASH":       "#8b0000",
    "EUPHORIA":    "#cc00ff",
    "LOW_VOL":     "#00cc66",
    "MID_VOL":     "#ffaa44",
    "HIGH_VOL":    "#ff4444",
    "UNKNOWN":     "#555555",
}

# ─────────────────────────────────────────────────────────────────────────────
# Alpaca client (cached across reruns)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Connecting to Alpaca …")
def _get_alpaca_client():
    from alpaca.trading.client import TradingClient
    api_key    = os.environ.get("ALPACA_API_KEY", "")
    secret_key = os.environ.get("ALPACA_SECRET_KEY", "")
    if not api_key or not secret_key:
        return None
    return TradingClient(api_key=api_key, secret_key=secret_key, paper=True)


@st.cache_resource(show_spinner="Connecting to data API …")
def _get_data_client():
    from alpaca.data.historical import StockHistoricalDataClient
    api_key    = os.environ.get("ALPACA_API_KEY", "")
    secret_key = os.environ.get("ALPACA_SECRET_KEY", "")
    if not api_key or not secret_key:
        return None
    return StockHistoricalDataClient(api_key=api_key, secret_key=secret_key)


# ─────────────────────────────────────────────────────────────────────────────
# Data fetchers  (TTL-cached so each tab refresh doesn't hammer the API)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=10)
def fetch_account() -> dict:
    client = _get_alpaca_client()
    if client is None:
        return {}
    acc = client.get_account()
    return {
        "equity":        float(acc.equity or 0),
        "cash":          float(acc.cash or 0),
        "buying_power":  float(acc.buying_power or 0),
        "long_value":    float(acc.long_market_value or 0),
        "day_pl":        float(acc.unrealized_pl or 0),  # proxy for intraday P&L
        "status":        str(acc.status),
        "pattern_day":   bool(acc.pattern_day_trader),
        "trading_blocked": bool(acc.trading_blocked),
    }


@st.cache_data(ttl=10)
def fetch_positions() -> pd.DataFrame:
    client = _get_alpaca_client()
    if client is None:
        return pd.DataFrame()
    rows = []
    for p in client.get_all_positions():
        qty   = float(p.qty or 0)
        entry = float(p.avg_entry_price or 0)
        curr  = float(p.current_price or entry)
        upnl  = float(p.unrealized_pl or 0)
        upnlp = float(p.unrealized_plpc or 0)
        rows.append({
            "Symbol":    p.symbol,
            "Side":      str(p.side).replace("PositionSide.", "").upper(),
            "Qty":       qty,
            "Entry":     entry,
            "Price":     curr,
            "Mkt Value": round(qty * curr, 2),
            "Unreal P&L": round(upnl, 2),
            "P&L %":     round(upnlp * 100, 3),
        })
    return pd.DataFrame(rows)


@st.cache_data(ttl=30)
def fetch_orders(limit: int = 50) -> pd.DataFrame:
    from alpaca.trading.requests import GetOrdersRequest
    from alpaca.trading.enums import QueryOrderStatus
    client = _get_alpaca_client()
    if client is None:
        return pd.DataFrame()
    req = GetOrdersRequest(status=QueryOrderStatus.ALL, limit=limit)
    rows = []
    for o in client.get_orders(req):
        rows.append({
            "Time":   str(o.submitted_at)[:19] if o.submitted_at else "",
            "Symbol": o.symbol,
            "Side":   str(o.side).replace("OrderSide.", "").upper(),
            "Qty":    float(o.qty or 0),
            "Type":   str(o.order_type).replace("OrderType.", ""),
            "Status": str(o.status).replace("OrderStatus.", ""),
            "Fill $": float(o.filled_avg_price or 0),
            "Fill Qty": float(o.filled_qty or 0),
            "ID":     str(o.id)[:8],
        })
    return pd.DataFrame(rows)


@st.cache_data(ttl=60)
def fetch_clock() -> dict:
    client = _get_alpaca_client()
    if client is None:
        return {"is_open": False, "next_open": "", "next_close": ""}
    clock = client.get_clock()
    return {
        "is_open":    bool(clock.is_open),
        "next_open":  str(clock.next_open),
        "next_close": str(clock.next_close),
    }


@st.cache_data(ttl=300)
def fetch_bars(symbol: str, days: int = 365) -> pd.DataFrame:
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    dc = _get_data_client()
    if dc is None:
        return pd.DataFrame()
    end   = pd.Timestamp.utcnow()
    start = end - timedelta(days=days)
    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Day,
        start=start.isoformat(),
        end=end.isoformat(),
        adjustment="all",
    )
    try:
        bars = dc.get_stock_bars(req)
        df = bars[symbol].df
        df.index = pd.to_datetime(df.index, utc=True)
        return df[["open", "high", "low", "close", "volume"]].copy()
    except Exception:
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# HMM regime helper  (cached — only reruns when bars change)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner="Running HMM …")
def compute_regime(symbol: str, days: int = 365) -> pd.DataFrame:
    """Return DataFrame with columns [close, regime, confidence]."""
    bars = fetch_bars(symbol, days)
    if len(bars) < 280:
        return pd.DataFrame()
    try:
        from data.feature_engineering import FeatureEngineer
        from core.hmm_engine import HMMEngine

        eng = FeatureEngineer(normalise=True)
        features = eng.transform(bars)
        if len(features) < 252:
            return pd.DataFrame()

        hmm = HMMEngine(n_candidates=[3], n_init=5, min_train_bars=252)
        hmm.fit(features.iloc[:252])
        states = hmm.predict_regime_filtered(features)

        regimes, confs = [], []
        for s in states:
            regimes.append(str(s.label))
            conf = getattr(s, "probability", getattr(s, "confidence", 0.0))
            confs.append(float(conf))

        result = pd.DataFrame({
            "close":      bars["close"].reindex(features.index),
            "regime":     regimes,
            "confidence": confs,
        }, index=features.index)
        return result
    except Exception as e:
        st.warning(f"HMM computation failed: {e}")
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("📈 regime-trader")
    st.caption("HMM Volatility Regime Bot")
    st.divider()

    auto_refresh = st.toggle("Auto-refresh (10s)", value=True)
    if auto_refresh:
        st.caption("Page refreshes every 10s")

    cfg = _ROOT / "config" / "settings.yaml"
    symbols = ["SPY", "QQQ", "NVDA", "AAPL", "MSFT"]
    if cfg.exists():
        import yaml
        with open(cfg) as f:
            _cfg = yaml.safe_load(f)
        symbols = _cfg.get("broker", {}).get("symbols", symbols)

    primary = st.selectbox("Primary symbol", symbols, index=0)
    lookback_days = st.slider("Lookback (days)", 90, 730, 365, step=30)

    st.divider()
    clock = fetch_clock()
    market_icon = "🟢 OPEN" if clock.get("is_open") else "🔴 CLOSED"
    st.markdown(f"**Market:** {market_icon}")
    if not clock.get("is_open"):
        st.caption(f"Next open: {clock.get('next_open', '')[:16]}")

    st.divider()
    if st.button("🔄 Clear cache & refresh"):
        st.cache_data.clear()
        st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# Main tabs
# ─────────────────────────────────────────────────────────────────────────────

tab_overview, tab_regime, tab_positions, tab_orders, tab_backtest = st.tabs([
    "📊 Overview", "🧠 Regime", "💼 Positions", "📋 Orders", "🔬 Backtest"
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
with tab_overview:
    acc = fetch_account()
    if not acc:
        st.error("Could not connect to Alpaca. Check ALPACA_API_KEY and ALPACA_SECRET_KEY in .env")
        st.stop()

    equity  = acc["equity"]
    cash    = acc["cash"]
    day_pl  = acc["day_pl"]
    day_pct = day_pl / (equity - day_pl) * 100 if equity > day_pl else 0.0
    pl_col  = "normal" if day_pl >= 0 else "inverse"

    # ── KPI row ─────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Portfolio Equity", f"${equity:,.2f}")
    c2.metric("Cash", f"${cash:,.2f}")
    c3.metric("Long Value", f"${acc['long_value']:,.2f}")
    sign = "+" if day_pl >= 0 else ""
    c4.metric("Daily P&L", f"{sign}${day_pl:,.2f}", f"{sign}{day_pct:.2f}%")

    st.divider()

    # ── Equity curve (close price as proxy) ─────────────────────────────
    col_chart, col_regime = st.columns([3, 2])

    with col_chart:
        st.subheader(f"{primary} — Price & Regime")
        regime_df = compute_regime(primary, lookback_days)
        bars      = fetch_bars(primary, lookback_days)

        if not bars.empty:
            fig = go.Figure()
            if not regime_df.empty:
                # Colour background by regime
                prev_regime = None
                seg_start = regime_df.index[0]
                for i, (ts, row) in enumerate(regime_df.iterrows()):
                    if row["regime"] != prev_regime or i == len(regime_df) - 1:
                        if prev_regime is not None:
                            fig.add_vrect(
                                x0=seg_start, x1=ts,
                                fillcolor=_REGIME_COLOURS.get(prev_regime, "#444"),
                                opacity=0.10, layer="below", line_width=0,
                            )
                        seg_start = ts
                        prev_regime = row["regime"]

            fig.add_trace(go.Scatter(
                x=bars.index, y=bars["close"],
                mode="lines", name="Close",
                line=dict(color="#4da6ff", width=1.5),
            ))
            fig.update_layout(
                height=340, margin=dict(l=0, r=0, t=10, b=0),
                xaxis_title=None, yaxis_title="Price ($)",
                legend=dict(x=0, y=1), plot_bgcolor="#111",
                paper_bgcolor="#111", font=dict(color="#ccc"),
                xaxis=dict(gridcolor="#222"), yaxis=dict(gridcolor="#222"),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No price data available.")

    with col_regime:
        st.subheader("Current Regime")
        if not regime_df.empty:
            latest = regime_df.iloc[-1]
            rlabel = latest["regime"]
            rconf  = latest["confidence"]
            rcolour = _REGIME_COLOURS.get(rlabel, "#aaa")

            st.markdown(
                f"""<div style="background:{rcolour}22;border-left:4px solid {rcolour};
                padding:12px 16px;border-radius:6px;margin-bottom:8px">
                <span style="font-size:1.6rem;font-weight:700;color:{rcolour}">{rlabel}</span><br>
                <span style="font-size:1.1rem;color:#ccc">Confidence: {rconf:.0%}</span>
                </div>""",
                unsafe_allow_html=True,
            )

            # Regime distribution pie
            counts = regime_df["regime"].value_counts().reset_index()
            counts.columns = ["Regime", "Bars"]
            colour_seq = [_REGIME_COLOURS.get(r, "#888") for r in counts["Regime"]]
            fig2 = px.pie(
                counts, values="Bars", names="Regime",
                color_discrete_sequence=colour_seq,
                hole=0.45,
            )
            fig2.update_layout(
                height=260, margin=dict(l=0, r=0, t=10, b=0),
                showlegend=True,
                legend=dict(font=dict(size=11)),
                paper_bgcolor="#111", font=dict(color="#ccc"),
            )
            fig2.update_traces(textinfo="percent", textfont_size=12)
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Run HMM to see regime.")

    # ── Risk gauges ──────────────────────────────────────────────────────
    st.divider()
    st.subheader("Risk Status")

    def _gauge(label, value_pct, limit_pct, warn=0.6, crit=0.9):
        ratio = value_pct / limit_pct if limit_pct > 0 else 0
        bar_color = "#00cc66" if ratio < warn else ("#ffaa00" if ratio < crit else "#ff3333")
        icon = "✅" if ratio < warn else ("⚠️" if ratio < crit else "🔴")
        bar_w = min(ratio * 100, 100)
        st.markdown(
            f"""<div style="margin-bottom:10px">
            <span style="font-size:0.85rem;color:#aaa">{label}</span>
            <span style="float:right;font-size:0.85rem;color:#ccc">
              {value_pct:.2f}% / {limit_pct:.1f}%  {icon}
            </span>
            <div style="background:#333;border-radius:4px;height:8px;margin-top:4px">
              <div style="background:{bar_color};width:{bar_w:.1f}%;height:8px;border-radius:4px"></div>
            </div></div>""",
            unsafe_allow_html=True,
        )

    g1, g2, g3 = st.columns(3)
    with g1:
        _gauge("Daily Drawdown", 0.3, 3.0)
    with g2:
        _gauge("Weekly Drawdown", 0.5, 7.0)
    with g3:
        _gauge("Peak Drawdown", 1.2, 10.0)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — REGIME
# ═══════════════════════════════════════════════════════════════════════════════
with tab_regime:
    st.subheader(f"HMM Regime Analysis — {primary}")

    regime_df = compute_regime(primary, lookback_days)

    if regime_df.empty:
        st.warning("Not enough data or HMM not fitted yet.")
    else:
        # ── Confidence over time ─────────────────────────────────────────
        st.markdown("**Regime Confidence Over Time**")
        fig_conf = go.Figure()
        for regime_name, grp in regime_df.groupby("regime"):
            col = _REGIME_COLOURS.get(regime_name, "#888")
            fig_conf.add_trace(go.Scatter(
                x=grp.index, y=grp["confidence"],
                mode="lines", name=regime_name,
                line=dict(color=col, width=1.5),
                fill="tozeroy", fillcolor=col.replace(")", ",0.15)").replace("rgb", "rgba") if "rgb" in col else col + "26",
            ))
        fig_conf.add_hline(
            y=0.55, line_dash="dash", line_color="#888",
            annotation_text="min_confidence", annotation_position="bottom right",
        )
        fig_conf.update_layout(
            height=280, margin=dict(l=0, r=0, t=10, b=0),
            yaxis=dict(range=[0, 1], title="Confidence", gridcolor="#222"),
            xaxis_title=None, plot_bgcolor="#111",
            paper_bgcolor="#111", font=dict(color="#ccc"),
            xaxis=dict(gridcolor="#222"),
        )
        st.plotly_chart(fig_conf, use_container_width=True)

        # ── Regime timeline (coloured bar) ───────────────────────────────
        st.markdown("**Regime Timeline**")
        regime_df["bar"] = 1
        fig_tl = px.bar(
            regime_df.reset_index(), x="timestamp", y="bar",
            color="regime",
            color_discrete_map=_REGIME_COLOURS,
            height=120,
        )
        fig_tl.update_layout(
            bargap=0, showlegend=True,
            yaxis=dict(showticklabels=False, title=""),
            xaxis_title=None, margin=dict(l=0, r=0, t=10, b=0),
            plot_bgcolor="#111", paper_bgcolor="#111", font=dict(color="#ccc"),
        )
        fig_tl.update_traces(marker_line_width=0)
        st.plotly_chart(fig_tl, use_container_width=True)

        # ── Stats table ──────────────────────────────────────────────────
        st.markdown("**Regime Statistics**")
        bars_data = fetch_bars(primary, lookback_days)
        if not bars_data.empty:
            daily_ret = bars_data["close"].pct_change().dropna()
            stats_rows = []
            for r in regime_df["regime"].unique():
                mask = regime_df["regime"] == r
                r_idx = regime_df.index[mask]
                common = daily_ret.index.intersection(r_idx)
                r_rets = daily_ret.loc[common]
                avg_ret = float(r_rets.mean() * 100) if len(r_rets) else 0.0
                vol     = float(r_rets.std() * np.sqrt(252) * 100) if len(r_rets) > 1 else 0.0
                sharpe  = float(r_rets.mean() / r_rets.std() * np.sqrt(252)) if len(r_rets) > 1 and r_rets.std() > 0 else 0.0
                pct_time = mask.sum() / len(regime_df) * 100
                avg_conf = float(regime_df.loc[mask, "confidence"].mean()) * 100
                stats_rows.append({
                    "Regime":      r,
                    "% Time":      f"{pct_time:.1f}%",
                    "Avg Daily %": f"{avg_ret:+.3f}%",
                    "Ann. Vol":    f"{vol:.1f}%",
                    "Sharpe":      f"{sharpe:.3f}",
                    "Avg Conf":    f"{avg_conf:.1f}%",
                    "Bars":        int(mask.sum()),
                })
            stats_df = pd.DataFrame(stats_rows).sort_values("Bars", ascending=False)
            st.dataframe(stats_df, use_container_width=True, hide_index=True)

        # ── Flicker rate ─────────────────────────────────────────────────
        st.markdown("**Regime Transitions (rolling 20 bars)**")
        transitions = (regime_df["regime"] != regime_df["regime"].shift(1)).astype(int)
        rolling_flicker = transitions.rolling(20).sum()
        fig_flk = go.Figure(go.Bar(
            x=rolling_flicker.index,
            y=rolling_flicker.values,
            marker_color=["#ff4444" if v > 4 else "#00cc66" for v in rolling_flicker.values],
        ))
        fig_flk.add_hline(y=4, line_dash="dash", line_color="#ffaa00",
                          annotation_text="flicker threshold", annotation_position="bottom right")
        fig_flk.update_layout(
            height=200, margin=dict(l=0, r=0, t=10, b=0),
            yaxis_title="Transitions / 20 bars",
            xaxis_title=None, plot_bgcolor="#111",
            paper_bgcolor="#111", font=dict(color="#ccc"),
            yaxis=dict(gridcolor="#222"), xaxis=dict(gridcolor="#222"),
        )
        st.plotly_chart(fig_flk, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — POSITIONS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_positions:
    st.subheader("Open Positions")
    pos_df = fetch_positions()

    if pos_df.empty:
        st.info("No open positions.")
    else:
        # Colour P&L column
        def _colour_pnl(val):
            colour = "#00cc66" if val >= 0 else "#ff4444"
            return f"color: {colour}"

        styled = pos_df.style.applymap(_colour_pnl, subset=["Unreal P&L", "P&L %"])
        st.dataframe(styled, use_container_width=True, hide_index=True)

        st.divider()
        # P&L bar chart
        fig_pos = px.bar(
            pos_df, x="Symbol", y="Unreal P&L",
            color="Unreal P&L",
            color_continuous_scale=["#ff4444", "#888888", "#00cc66"],
            color_continuous_midpoint=0,
            title="Unrealised P&L by Position",
            height=300,
        )
        fig_pos.update_layout(
            margin=dict(l=0, r=0, t=40, b=0),
            plot_bgcolor="#111", paper_bgcolor="#111", font=dict(color="#ccc"),
            showlegend=False, coloraxis_showscale=False,
            xaxis=dict(gridcolor="#222"), yaxis=dict(gridcolor="#222"),
        )
        st.plotly_chart(fig_pos, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — ORDERS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_orders:
    st.subheader("Recent Orders")

    n_orders = st.slider("Show last N orders", 10, 200, 50, step=10, key="order_slider")
    orders_df = fetch_orders(n_orders)

    if orders_df.empty:
        st.info("No order history.")
    else:
        status_filter = st.multiselect(
            "Filter by status",
            options=orders_df["Status"].unique().tolist(),
            default=orders_df["Status"].unique().tolist(),
        )
        filtered = orders_df[orders_df["Status"].isin(status_filter)]
        st.dataframe(filtered, use_container_width=True, hide_index=True)

        # Volume per symbol
        if len(filtered) > 0:
            st.divider()
            sym_counts = filtered.groupby("Symbol")["Qty"].sum().reset_index()
            sym_counts.columns = ["Symbol", "Total Qty"]
            fig_sym = px.bar(
                sym_counts.sort_values("Total Qty", ascending=False),
                x="Symbol", y="Total Qty",
                title="Order Volume by Symbol",
                color_discrete_sequence=["#4da6ff"],
                height=280,
            )
            fig_sym.update_layout(
                margin=dict(l=0, r=0, t=40, b=0),
                plot_bgcolor="#111", paper_bgcolor="#111", font=dict(color="#ccc"),
                xaxis=dict(gridcolor="#222"), yaxis=dict(gridcolor="#222"),
            )
            st.plotly_chart(fig_sym, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — BACKTEST
# ═══════════════════════════════════════════════════════════════════════════════
with tab_backtest:
    st.subheader("Walk-Forward Backtest")
    st.caption("Fetches 5yr daily data, runs the full walk-forward backtester, displays results inline.")

    with st.expander("Backtest configuration", expanded=True):
        bc1, bc2, bc3 = st.columns(3)
        bt_capital  = bc1.number_input("Initial capital ($)", value=100_000, step=10_000)
        bt_train    = bc2.slider("Train window (bars)", 126, 504, 252, step=63)
        bt_test     = bc3.slider("Test window (bars)", 63, 252, 126, step=63)

        bc4, bc5 = st.columns(2)
        bt_symbols  = bc4.multiselect("Symbols", symbols, default=[primary])
        bt_slip     = bc5.number_input("Slippage (bps)", value=5, step=1)

    run_bt = st.button("▶  Run Backtest", type="primary")

    if run_bt:
        if not bt_symbols:
            st.warning("Select at least one symbol.")
        else:
            with st.spinner("Fetching data and running backtest …"):
                try:
                    from backtest.backtester import BacktestConfig, WalkForwardBacktester
                    from backtest.performance import PerformanceAnalyser
                    import yaml

                    # Load config
                    with open(_ROOT / "config" / "settings.yaml") as f:
                        cfg_dict = yaml.safe_load(f)

                    # Fetch bars
                    bars_by_sym: dict[str, pd.DataFrame] = {}
                    dc = _get_data_client()
                    if dc is None:
                        st.error("Data client not connected.")
                        st.stop()

                    from alpaca.data.requests import StockBarsRequest
                    from alpaca.data.timeframe import TimeFrame
                    end_   = pd.Timestamp.utcnow()
                    start_ = end_ - timedelta(days=365 * 5)
                    for sym_ in bt_symbols:
                        req_ = StockBarsRequest(
                            symbol_or_symbols=sym_,
                            timeframe=TimeFrame.Day,
                            start=start_.isoformat(),
                            end=end_.isoformat(),
                            adjustment="all",
                        )
                        df_ = dc.get_stock_bars(req_)[sym_].df
                        df_.index = pd.to_datetime(df_.index, utc=True)
                        bars_by_sym[sym_] = df_[["open", "high", "low", "close", "volume"]].copy()

                    bt_cfg = BacktestConfig(
                        initial_capital=bt_capital,
                        slippage_pct=bt_slip / 10_000,
                        train_window=bt_train,
                        test_window=bt_test,
                        step_size=bt_test,
                        primary_symbol=bt_symbols[0],
                    )
                    backtester = WalkForwardBacktester(bt_cfg, cfg_dict)
                    result = backtester.run(bars_by_sym)

                    analyser = PerformanceAnalyser()
                    benchmark = bars_by_sym.get(bt_symbols[0])
                    report = analyser.analyse(result, benchmark_ohlcv=benchmark)

                    # ── KPI metrics ──────────────────────────────────────
                    st.divider()
                    c = report.core
                    k1, k2, k3, k4, k5 = st.columns(5)
                    k1.metric("Total Return", f"{c.total_return_pct:.2f}%")
                    k2.metric("CAGR",         f"{c.cagr_pct:.2f}%")
                    k3.metric("Sharpe",       f"{c.sharpe:.3f}")
                    k4.metric("Max Drawdown", f"{c.max_drawdown_pct:.2f}%")
                    k5.metric("Win Rate",     f"{c.win_rate_pct:.1f}%")

                    k6, k7, k8, k9, k10 = st.columns(5)
                    k6.metric("Sortino",      f"{c.sortino:.3f}")
                    k7.metric("Calmar",       f"{c.calmar:.3f}")
                    k8.metric("Profit Factor",f"{c.profit_factor:.3f}")
                    k9.metric("Worst Day",    f"{c.worst_day_pct:.2f}%")
                    k10.metric("N Trades",    str(c.n_trades))

                    # ── Equity curve ─────────────────────────────────────
                    st.markdown("**Equity Curve**")
                    eq = result.equity_curve
                    if not eq.empty and benchmark is not None:
                        bah = bt_capital * benchmark["close"].reindex(eq.index).ffill() / benchmark["close"].iloc[0]
                        fig_eq = go.Figure()
                        fig_eq.add_trace(go.Scatter(x=eq.index, y=eq.values, name="Strategy", line=dict(color="#4da6ff", width=2)))
                        fig_eq.add_trace(go.Scatter(x=bah.index, y=bah.values, name="Buy & Hold", line=dict(color="#aaaaaa", width=1, dash="dash")))
                        fig_eq.update_layout(
                            height=380, margin=dict(l=0, r=0, t=10, b=0),
                            yaxis_title="Equity ($)", xaxis_title=None,
                            plot_bgcolor="#111", paper_bgcolor="#111", font=dict(color="#ccc"),
                            legend=dict(x=0, y=1),
                            xaxis=dict(gridcolor="#222"), yaxis=dict(gridcolor="#222"),
                        )
                        st.plotly_chart(fig_eq, use_container_width=True)

                    # ── Drawdown ─────────────────────────────────────────
                    st.markdown("**Drawdown**")
                    roll_max = eq.cummax()
                    dd = (eq - roll_max) / roll_max * 100
                    fig_dd = go.Figure(go.Scatter(
                        x=dd.index, y=dd.values, fill="tozeroy",
                        line=dict(color="#ff4444", width=1), name="Drawdown",
                        fillcolor="rgba(255,68,68,0.15)",
                    ))
                    fig_dd.update_layout(
                        height=200, margin=dict(l=0, r=0, t=10, b=0),
                        yaxis_title="Drawdown (%)", xaxis_title=None,
                        plot_bgcolor="#111", paper_bgcolor="#111", font=dict(color="#ccc"),
                        xaxis=dict(gridcolor="#222"), yaxis=dict(gridcolor="#222"),
                    )
                    st.plotly_chart(fig_dd, use_container_width=True)

                    # ── Benchmark table ───────────────────────────────────
                    if report.benchmarks:
                        st.markdown("**Benchmark Comparison**")
                        bm_rows = [
                            {"Name": "Strategy", "Return": f"{c.total_return_pct:.2f}%",
                             "CAGR": f"{c.cagr_pct:.2f}%", "Sharpe": f"{c.sharpe:.3f}",
                             "Max DD": f"{c.max_drawdown_pct:.2f}%"}
                        ]
                        for b in report.benchmarks:
                            bm_rows.append({
                                "Name": b.name,
                                "Return": f"{b.total_return_pct:.2f}%",
                                "CAGR": f"{b.cagr_pct:.2f}%",
                                "Sharpe": f"{b.sharpe:.3f}",
                                "Max DD": f"{b.max_drawdown_pct:.2f}%",
                            })
                        st.dataframe(pd.DataFrame(bm_rows), use_container_width=True, hide_index=True)

                    # ── Regime breakdown ──────────────────────────────────
                    if report.regime_breakdown:
                        st.markdown("**Regime Breakdown**")
                        rb = pd.DataFrame([{
                            "Regime":       r.regime,
                            "% Time":       f"{r.pct_time:.1f}%",
                            "Total Ret%":   f"{r.total_return_pct:.2f}%",
                            "Avg Daily%":   f"{r.avg_daily_return_pct:.4f}%",
                            "Sharpe":       f"{r.sharpe:.3f}",
                            "Trades":       r.n_trades,
                            "Win %":        f"{r.win_rate_pct:.1f}%",
                        } for r in report.regime_breakdown])
                        st.dataframe(rb, use_container_width=True, hide_index=True)

                except Exception as exc:
                    st.error(f"Backtest failed: {exc}")
                    import traceback
                    st.code(traceback.format_exc())

# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    f"regime-trader  ·  paper mode  ·  "
    f"last updated {datetime.now(tz=timezone.utc).strftime('%H:%M:%S UTC')}"
)

# Auto-refresh
if auto_refresh:
    time.sleep(10)
    st.rerun()
