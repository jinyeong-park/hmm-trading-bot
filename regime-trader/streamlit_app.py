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
    api_key = os.environ.get("ALPACA_API_KEY", "")
    secret_key = os.environ.get("ALPACA_SECRET_KEY", "")
    if not api_key or not secret_key:
        return None
    return TradingClient(api_key=api_key, secret_key=secret_key, paper=True)


@st.cache_resource(show_spinner="Connecting to data API …")
def _get_data_client():
    from alpaca.data.historical import StockHistoricalDataClient
    api_key = os.environ.get("ALPACA_API_KEY", "")
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
    # Debug: print raw account values
    print(
        f"DEBUG: Alpaca account raw - equity: {acc.equity}, cash: {acc.cash}, last_equity: {acc.last_equity}, status: {acc.status}")
    day_pl_calc = float((acc.equity or 0)) - float((acc.last_equity or 0))
    print(f"DEBUG: day_pl calculation: {day_pl_calc}")
    return {
        "equity":        float(acc.equity or 0),
        "cash":          float(acc.cash or 0),
        "buying_power":  float(acc.buying_power or 0),
        "long_value":    float(acc.long_market_value or 0),
        "last_equity":   float(acc.last_equity or 0),
        # intraday P&L
        "day_pl":        day_pl_calc,
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
        qty = float(p.qty or 0)
        entry = float(p.avg_entry_price or 0)
        curr = float(p.current_price or entry)
        upnl = float(p.unrealized_pl or 0)
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
    """Fetch daily OHLCV bars via the free IEX feed."""
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.enums import DataFeed
    dc = _get_data_client()
    if dc is None:
        return pd.DataFrame()
    end = pd.Timestamp.utcnow()
    start = end - timedelta(days=days)
    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Day,
        start=start.isoformat(),
        end=end.isoformat(),
        adjustment="all",
        feed=DataFeed.IEX,
    )
    try:
        df = dc.get_stock_bars(req).df
        if isinstance(df.index, pd.MultiIndex):
            df = df.reset_index()
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df = df.set_index("timestamp")
        else:
            df.index = pd.to_datetime(df.index, utc=True)
        return df[["open", "high", "low", "close", "volume"]].copy()
    except Exception as e:
        st.warning(f"fetch_bars failed for {symbol}: {e}")
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# HMM regime helper  (cached — only reruns when bars change)
# ─────────────────────────────────────────────────────────────────────────────

# Bars needed: 200-bar SMA warmup + 252-bar z-score window ≈ 325 warmup rows.
# Fetch a fixed 5-year window so the HMM always has enough training data,
# then trim the returned regime series to the user's display window.
_HMM_FETCH_DAYS = 1825  # 5 years


@st.cache_data(ttl=300, show_spinner="Running HMM …")
def compute_regime(symbol: str, days: int = 365) -> pd.DataFrame:
    """Return DataFrame[close, regime, confidence] trimmed to `days`."""
    bars_full = fetch_bars(symbol, _HMM_FETCH_DAYS)
    if len(bars_full) < 400:
        st.warning(f"Not enough bars for HMM ({len(bars_full)} fetched).")
        return pd.DataFrame()
    try:
        from data.feature_engineering import FeatureEngineer
        from core.hmm_engine import HMMEngine

        eng = FeatureEngineer(normalise=True)
        features = eng.transform(bars_full)
        if len(features) < 252:
            st.warning(f"Not enough features after warmup ({len(features)}).")
            return pd.DataFrame()

        hmm = HMMEngine(n_candidates=[3], n_init=5, min_train_bars=252)
        hmm.fit(features.iloc[:252])
        states = hmm.predict_regime_filtered(features)

        regimes, vol_regimes, confs, confirmed_list, consec_list = [], [], [], [], []
        posteriors_list: list = []
        state_indices: list[int] = []
        state_label_map: dict[int, str] = {}

        for s in states:
            # Use .value so we get "BEAR" not "RegimeLabel.BEAR"
            lbl = s.label
            lbl_str = lbl.value if hasattr(lbl, "value") else str(lbl)
            regimes.append(lbl_str)
            vol = getattr(s, "regime", None)
            vol_regimes.append(vol.value if hasattr(
                vol, "value") else str(vol))
            confs.append(
                float(getattr(s, "probability", getattr(s, "confidence", 0.0))))
            confirmed_list.append(bool(getattr(s, "is_confirmed", True)))
            consec_list.append(int(getattr(s, "consecutive_bars", 1)))
            probs = getattr(s, "state_probabilities", np.array([]))
            posteriors_list.append(
                list(probs.astype(float)) if len(probs) else [])
            sid = int(getattr(s, "state_id", 0))
            state_indices.append(sid)
            state_label_map[sid] = lbl_str

        import json as _json
        _label_json = _json.dumps(
            {str(k): v for k, v in state_label_map.items()})

        result = pd.DataFrame({
            "close":            bars_full["close"].reindex(features.index),
            "regime":           regimes,
            "vol_regime":       vol_regimes,
            "confidence":       confs,
            "is_confirmed":     confirmed_list,
            "consecutive_bars": consec_list,
            "posteriors":       posteriors_list,
            "state_index":      state_indices,
            "state_label_json": _label_json,
        }, index=features.index)

        # Trim to the user's display window
        cutoff = pd.Timestamp.utcnow() - timedelta(days=days)
        cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff
        return result[result.index >= cutoff]
    except Exception as e:
        import traceback
        st.warning(
            f"HMM computation failed: {e}\n\n```\n{traceback.format_exc()}\n```")
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("Regime Trader")
    st.divider()

    st.subheader("Settings")
    refresh_interval = st.slider("Refresh interval (s)", 5, 60, 10)

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
    st.subheader("Debug")
    show_price_chart = st.checkbox("Show price chart", value=True)
    show_regime_history = st.checkbox("Show regime history", value=False)
    show_transition_matrix = st.checkbox("Show transition matrix", value=False)
    show_logs = st.checkbox("Show logs", value=False)
    show_model_info = st.checkbox("Show model info", value=False)

    st.divider()
    if st.button("🔄 Clear cache & refresh"):
        st.cache_data.clear()
        st.rerun()

    st.divider()
    clock = fetch_clock()
    market_icon = "🟢" if clock.get("is_open") else "🔴"
    st.caption(
        f"**Market:** {market_icon} {'OPEN' if clock.get('is_open') else 'CLOSED'}")
    st.caption(f"Last refresh: {datetime.now().strftime('%H:%M:%S')}")

# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# Overview section
# ─────────────────────────────────────────────────────────────────────────────
fetch_clock()  # Cache clock at top level
clock = fetch_clock()


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
# tab_overview
acc = fetch_account()
if not acc:
    st.error(
        "Could not connect to Alpaca. Check ALPACA_API_KEY and ALPACA_SECRET_KEY in .env")
    st.stop()

equity = acc["equity"]
cash = acc["cash"]
day_pl = acc["day_pl"]
last_equity = acc["last_equity"]
day_pct = (day_pl / last_equity) * 100 if last_equity > 0 else 0.0

# ────────────────────────────────────────────────────────────────────────
# TOP HEADER: Mode, Equity, Cash, Day P&L
# ────────────────────────────────────────────────────────────────────────
h1, h2, h3, h4 = st.columns(4)

with h1:
    st.markdown(
        "<div style='text-align:center'>"
        "<div style='font-size:0.75rem;color:#aaa'>Mode</div>"
        "<div style='font-size:1.8rem;font-weight:700;color:#4da6ff'>PAPER</div>"
        "</div>",
        unsafe_allow_html=True,
    )

with h2:
    st.markdown(
        "<div style='text-align:center'>"
        "<div style='font-size:0.75rem;color:#aaa'>Equity</div>"
        f"<div style='font-size:1.4rem;font-weight:700;color:#fff'>${equity:,.2f}</div>"
        "</div>",
        unsafe_allow_html=True,
    )

with h3:
    st.markdown(
        "<div style='text-align:center'>"
        "<div style='font-size:0.75rem;color:#aaa'>Cash</div>"
        f"<div style='font-size:1.4rem;font-weight:700;color:#fff'>${cash:,.2f}</div>"
        "</div>",
        unsafe_allow_html=True,
    )

with h4:
    sign_pl = "+" if day_pl >= 0 else ""
    st.markdown(
        "<div style='text-align:center'>"
        "<div style='font-size:0.75rem;color:#aaa'>Daily P&L</div>"
        f"<div style='font-size:1.4rem;font-weight:700;color:#00cc66'>{sign_pl}${day_pl:,.2f}</div>"
        f"<div style='font-size:0.75rem;color:#aaa'>{day_pct:.2f}%</div>"
        "</div>",
        unsafe_allow_html=True,
    )

st.divider()

# ────────────────────────────────────────────────────────────────────────
# REGIME DETECTION (left) & RISK STATUS (right)
# ────────────────────────────────────────────────────────────────────────
regime_col, risk_col = st.columns([2, 1])

with regime_col:
    st.subheader("Regime Detection")
    regime_df = compute_regime(primary, lookback_days)

    if not regime_df.empty:
        latest = regime_df.iloc[-1]
        rlabel = latest["regime"]
        rconf = latest["confidence"]
        rcolour = _REGIME_COLOURS.get(rlabel, "#aaa")

        # Stability: consecutive_bars field from HMM (or count from tail of series)
        stability_bars = int(latest.get("consecutive_bars", 1))
        is_confirmed = bool(latest.get("is_confirmed", True))

        # Vol rank: map Regime value (low_vol / mid_vol / high_vol) → 0–1
        _vol_rank_map = {"low_vol": 0.15, "mid_vol": 0.50, "high_vol": 0.85}
        vol_str = str(latest.get("vol_regime", "unknown")).lower()
        vol_rank = _vol_rank_map.get(vol_str, 0.50)

        # ── Stats row (top) ───────────────────────────────────────────
        r1, r2, r3, r4 = st.columns(4)
        with r1:
            st.markdown(
                f"<div style='text-align:center'><div style='font-size:0.8rem;color:#aaa'>Regime</div>"
                f"<div style='font-size:2rem;font-weight:700;color:{rcolour}'>{rlabel}</div></div>",
                unsafe_allow_html=True)
        with r2:
            st.markdown(
                f"<div style='text-align:center'><div style='font-size:0.8rem;color:#aaa'>Confidence</div>"
                f"<div style='font-size:2rem;font-weight:700;color:#fff'>{rconf:.1%}</div></div>",
                unsafe_allow_html=True)
        with r3:
            st.markdown(
                f"<div style='text-align:center'><div style='font-size:0.8rem;color:#aaa'>Stability</div>"
                f"<div style='font-size:2rem;font-weight:700;color:#fff'>{stability_bars} bars</div></div>",
                unsafe_allow_html=True)
        with r4:
            st.markdown(
                f"<div style='text-align:center'><div style='font-size:0.8rem;color:#aaa'>Vol Rank</div>"
                f"<div style='font-size:2rem;font-weight:700;color:#fff'>{vol_rank:.2f}</div></div>",
                unsafe_allow_html=True)

        st.markdown("")

        # Status badge
        if is_confirmed:
            status_color, status_text = "#00cc66", "↑ CONFIRMED"
        else:
            status_color, status_text = "#ffaa00", "⏳ PENDING"
        st.markdown(
            f"<div style='display:inline-block;background:{status_color};color:#000;"
            f"padding:3px 12px;border-radius:20px;font-size:0.8rem;font-weight:600'>{status_text}</div>",
            unsafe_allow_html=True)

        st.markdown("")

        # ── Semicircle SVG gauge (confidence-based) ───────────────────
        import math as _math
        import json as _json

        gauge_val = rconf          # 0–1, shown as the filled arc
        gauge_pct = gauge_val * 100
        cx, cy, r = 120, 128, 96   # arc centre, radius
        sw_track = 24             # dark track stroke-width
        sw_fill = 16             # orange fill stroke-width (inset)

        # Arc endpoint for fill: angle sweeps from left (0) to right (1)
        nx = cx - r * _math.cos(_math.pi * gauge_val)
        ny = cy - r * _math.sin(_math.pi * gauge_val)
        arc_large = 1 if gauge_val > 0.5 else 0

        # Tick labels outside the arc at 0%, 20%, 40%, 60%, 80%, 100% positions
        r_lbl = r + 18
        tick_svg = ""
        for f, lab in [(0.0, "0%"), (0.2, "20%"), (0.4, "40%"), (0.6, "60%"), (0.8, "80%"), (1.0, "100%")]:
            tx = cx - r_lbl * _math.cos(_math.pi * f)
            ty = cy - r_lbl * _math.sin(_math.pi * f)
            anc = "end" if f < 0.5 else "start"
            tick_svg += (
                f'<text x="{tx:.1f}" y="{ty:.1f}" text-anchor="{anc}" '
                f'dominant-baseline="middle" font-size="10" fill="#888">{lab}</text>'
            )

        # Runner-up states from posteriors
        runner_up_html = ""
        try:
            posteriors = latest.get("posteriors", [])
            label_map = _json.loads(latest.get("state_label_json", "{}"))
            main_idx = int(latest.get("state_index", -1))
            if isinstance(posteriors, list) and len(posteriors) > 1:
                parts = []
                for i, p in enumerate(posteriors):
                    if i == main_idx:
                        continue
                    lbl_ru = label_map.get(str(i), f"State{i}")
                    if p > 1e-6:
                        parts.append(f"{lbl_ru}: {p:.2%}")
                    elif p > 0:
                        import math as _m
                        exp = int(_m.floor(_m.log10(p)))
                        parts.append(f"{lbl_ru}: 10<sup>{exp}</sup>")
                    else:
                        parts.append(f"{lbl_ru}: ~0")
                if parts:
                    runner_up_html = (
                        "<div style='text-align:center;font-size:0.75rem;color:#666;"
                        "margin-top:4px'>Runner-up states: " +
                        " &nbsp;|&nbsp; ".join(parts) + "</div>"
                    )
        except Exception:
            pass

        st.markdown(
            f"""<div style="text-align:center">
            <svg viewBox="-10 -12 260 158" width="100%" style="max-width:380px">
              <!-- dark gray track -->
              <path d="M{cx-r},{cy} A{r},{r} 0 0,1 {cx+r},{cy}"
                    fill="none" stroke="#2a2a2a" stroke-width="{sw_track}" stroke-linecap="butt"/>
              <!-- orange fill arc -->
              <path d="M{cx-r},{cy} A{r},{r} 0 {arc_large},1 {nx:.2f},{ny:.2f}"
                    fill="none" stroke="#ff8c00" stroke-width="{sw_fill}" stroke-linecap="butt"/>
              <!-- tick labels -->
              {tick_svg}
              <!-- "Regime: LABEL" subtitle -->
              <text x="{cx}" y="{cy - 24}" text-anchor="middle"
                    font-size="13" fill="#aaa">Regime: {rlabel}</text>
              <!-- large percentage value -->
              <text x="{cx}" y="{cy + 6}" text-anchor="middle"
                    font-size="34" font-weight="bold" fill="#ff8c00">{gauge_pct:.0f}%</text>
            </svg>
            {runner_up_html}
            </div>""",
            unsafe_allow_html=True,
        )

        st.markdown("")
    else:
        # No-data fallback — same layout as live section
        r1, r2, r3, r4 = st.columns(4)
        with r1:
            st.markdown(
                "<div style='text-align:center'><div style='font-size:0.8rem;color:#aaa'>Regime</div>"
                "<div style='font-size:2rem;font-weight:700;color:#888'>UNKNOWN</div></div>",
                unsafe_allow_html=True)
        with r2:
            st.markdown(
                "<div style='text-align:center'><div style='font-size:0.8rem;color:#aaa'>Confidence</div>"
                "<div style='font-size:2rem;font-weight:700;color:#888'>0.0%</div></div>",
                unsafe_allow_html=True)
        with r3:
            st.markdown(
                "<div style='text-align:center'><div style='font-size:0.8rem;color:#aaa'>Stability</div>"
                "<div style='font-size:2rem;font-weight:700;color:#888'>0 bars</div></div>",
                unsafe_allow_html=True)
        with r4:
            st.markdown(
                "<div style='text-align:center'><div style='font-size:0.8rem;color:#aaa'>Vol Rank</div>"
                "<div style='font-size:2rem;font-weight:700;color:#888'>0.00</div></div>",
                unsafe_allow_html=True)
        st.markdown("")
        st.markdown(
            "<div style='display:inline-block;background:#ffaa00;color:#000;padding:3px 12px;"
            "border-radius:20px;font-size:0.8rem;font-weight:600'>⚠ AWAITING DATA</div>",
            unsafe_allow_html=True)
        st.markdown("")

        import math as _math_nd
        cx_nd, cy_nd, r_nd = 120, 128, 96
        st.markdown(
            f"""<div style="text-align:center">
            <svg viewBox="-10 -12 260 158" width="100%" style="max-width:380px">
              <path d="M{cx_nd-r_nd},{cy_nd} A{r_nd},{r_nd} 0 0,1 {cx_nd+r_nd},{cy_nd}"
                    fill="none" stroke="#2a2a2a" stroke-width="24" stroke-linecap="butt"/>
              <text x="{cx_nd}" y="{cy_nd - 24}" text-anchor="middle"
                    font-size="13" fill="#555">Regime: UNKNOWN</text>
              <text x="{cx_nd}" y="{cy_nd + 6}" text-anchor="middle"
                    font-size="34" font-weight="bold" fill="#555">–%</text>
            </svg>
            </div>""",
            unsafe_allow_html=True,
        )
        st.info("Run HMM to see regime. Need at least 252 trading days of data.")

with risk_col:
    st.subheader("Risk Status")

    # Risk gauge functions
    def _risk_gauge(label, value_pct, limit_pct, warn=0.6, crit=0.9):
        ratio = value_pct / limit_pct if limit_pct > 0 else 0
        bar_color = "#00cc66" if ratio < warn else (
            "#ffaa00" if ratio < crit else "#ff3333")
        icon = "✅" if ratio < warn else ("⚠️" if ratio < crit else "🔴")
        bar_w = min(ratio * 100, 100)

        st.markdown(
            f"""<div style="margin-bottom:12px">
            <span style="font-size:0.8rem;color:#aaa">{label}</span>
            <span style="float:right;font-size:0.8rem;color:#ccc;font-weight:600">{value_pct:.2f}% / {limit_pct:.1f}%</span>
            <div style="clear:both;background:#333;border-radius:4px;height:6px;margin-top:4px;overflow:hidden">
              <div style="background:{bar_color};width:{bar_w:.1f}%;height:6px;border-radius:4px"></div>
            </div>
            </div>""",
            unsafe_allow_html=True,
        )

    # Daily DD: equity dropped from last close
    _daily_dd = max(0.0, (acc["last_equity"] - acc["equity"]) /
                    acc["last_equity"] * 100) if acc["last_equity"] > 0 else 0.0
    # Leverage: long market value as fraction of equity
    _leverage_pct = (acc["long_value"] / acc["equity"]
                     * 100) if acc["equity"] > 0 else 0.0

    _risk_gauge("Daily DD",  _daily_dd,    3.0)
    # proxy until equity history tracked
    _risk_gauge("Peak DD",   _daily_dd,   10.0)
    _risk_gauge("Leverage",  _leverage_pct, 125.0)

    st.markdown("")
    _cb_ok = _daily_dd < 3.0
    _cb_color = "#00cc66" if _cb_ok else "#ff3333"
    _cb_text = "✓ All circuit breakers OK" if _cb_ok else "⚠ Daily DD limit approaching"
    st.markdown(
        f"<div style='background:{_cb_color}22;padding:8px 12px;border-radius:6px;font-size:0.85rem;color:{_cb_color};font-weight:600'>{_cb_text}</div>", unsafe_allow_html=True)

st.divider()

# ────────────────────────────────────────────────────────────────────────
# PRICE CHART & REGIME HISTORY
# ────────────────────────────────────────────────────────────────────────
if show_price_chart:
    st.subheader(f"{primary} — Price & Regime")
    bars = fetch_bars(primary, lookback_days)

    if not bars.empty and not regime_df.empty:
        fig = go.Figure()

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
            legend=dict(x=0, y=1), plot_bgcolor="#000000",
            paper_bgcolor="#000000", font=dict(color="#ccc"),
            xaxis=dict(gridcolor="#222"), yaxis=dict(gridcolor="#222"),
        )
        st.plotly_chart(fig, use_container_width=True)

if show_regime_history:
    st.subheader("Regime History")
    if not regime_df.empty:
        regime_df_display = regime_df.reset_index()
        regime_df_display.columns = [
            "Timestamp", "Close", "Regime", "Confidence"]
        st.dataframe(regime_df_display.tail(
            50), use_container_width=True, hide_index=True)

if show_model_info:
    st.subheader("Model Info")
    st.info(
        "HMM model info would display here (BIC scores, n_states, training date, etc.)")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — REGIME
# ═══════════════════════════════════════════════════════════════════════════════
# tab_regime
st.subheader(f"HMM Regime Analysis — {primary}")

regime_df = compute_regime(primary, lookback_days)

if regime_df.empty:
    st.warning("Not enough data or HMM not fitted yet.")
else:
    # ── Confidence over time ─────────────────────────────────────────
    st.markdown("**Regime Confidence Over Time**")
    fig_conf = go.Figure()

    # Shade background by regime period
    prev_r = regime_df["regime"].iloc[0]
    seg_start = regime_df.index[0]
    for ts, row in regime_df.iterrows():
        if row["regime"] != prev_r:
            fig_conf.add_vrect(
                x0=seg_start, x1=ts,
                fillcolor=_REGIME_COLOURS.get(prev_r, "#444"),
                opacity=0.12, layer="below", line_width=0,
            )
            seg_start = ts
            prev_r = row["regime"]
    # Final segment
    fig_conf.add_vrect(
        x0=seg_start, x1=regime_df.index[-1],
        fillcolor=_REGIME_COLOURS.get(prev_r, "#444"),
        opacity=0.12, layer="below", line_width=0,
    )

    # Single continuous confidence line
    fig_conf.add_trace(go.Scatter(
        x=regime_df.index,
        y=regime_df["confidence"],
        mode="lines",
        name="Confidence",
        line=dict(color="#ffffff", width=1.5),
        hovertemplate="%{x|%Y-%m-%d}<br>Confidence: %{y:.1%}<extra></extra>",
    ))

    fig_conf.add_hline(
        y=0.55, line_dash="dash", line_color="#888",
        annotation_text="min_confidence", annotation_position="bottom right",
    )
    fig_conf.update_layout(
        height=280, margin=dict(l=0, r=0, t=10, b=0),
        yaxis=dict(range=[0, 1], title="Confidence",
                   gridcolor="#222", tickformat=".0%"),
        xaxis_title=None, plot_bgcolor="#000000",
        paper_bgcolor="#000000", font=dict(color="#ccc"),
        xaxis=dict(gridcolor="#222"),
        showlegend=False,
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
        plot_bgcolor="#000000", paper_bgcolor="#000000", font=dict(color="#ccc"),
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
            vol = float(r_rets.std() * np.sqrt(252) *
                        100) if len(r_rets) > 1 else 0.0
            sharpe = float(r_rets.mean() / r_rets.std() * np.sqrt(252)
                           ) if len(r_rets) > 1 and r_rets.std() > 0 else 0.0
            pct_time = mask.sum() / len(regime_df) * 100
            avg_conf = float(
                regime_df.loc[mask, "confidence"].mean()) * 100
            stats_rows.append({
                "Regime":      r,
                "% Time":      f"{pct_time:.1f}%",
                "Avg Daily %": f"{avg_ret:+.3f}%",
                "Ann. Vol":    f"{vol:.1f}%",
                "Sharpe":      f"{sharpe:.3f}",
                "Avg Conf":    f"{avg_conf:.1f}%",
                "Bars":        int(mask.sum()),
            })
        stats_df = pd.DataFrame(stats_rows).sort_values(
            "Bars", ascending=False)
        st.dataframe(stats_df, use_container_width=True, hide_index=True)

    # ── Flicker rate ─────────────────────────────────────────────────
    st.markdown("**Regime Transitions (rolling 20 bars)**")
    transitions = (regime_df["regime"] !=
                   regime_df["regime"].shift(1)).astype(int)
    rolling_flicker = transitions.rolling(20).sum()
    fig_flk = go.Figure(go.Bar(
        x=rolling_flicker.index,
        y=rolling_flicker.values,
        marker_color=["#ff4444" if v >
                      4 else "#00cc66" for v in rolling_flicker.values],
    ))
    fig_flk.add_hline(y=4, line_dash="dash", line_color="#ffaa00",
                      annotation_text="flicker threshold", annotation_position="bottom right")
    fig_flk.update_layout(
        height=200, margin=dict(l=0, r=0, t=10, b=0),
        yaxis_title="Transitions / 20 bars",
        xaxis_title=None, plot_bgcolor="#000000",
        paper_bgcolor="#000000", font=dict(color="#ccc"),
        yaxis=dict(gridcolor="#222"), xaxis=dict(gridcolor="#222"),
    )
    st.plotly_chart(fig_flk, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — POSITIONS
# ═══════════════════════════════════════════════════════════════════════════════
# tab_positions
st.subheader("Open Positions")
pos_df = fetch_positions()

if pos_df.empty:
    st.info("No open positions.")
else:
    # Colour P&L column
    def _colour_pnl(val):
        colour = "#00cc66" if val >= 0 else "#ff4444"
        return f"color: {colour}"

    styled = pos_df.style.map(_colour_pnl, subset=["Unreal P&L", "P&L %"])
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
        plot_bgcolor="#000000", paper_bgcolor="#000000", font=dict(color="#ccc"),
        showlegend=False, coloraxis_showscale=False,
        xaxis=dict(gridcolor="#222"), yaxis=dict(gridcolor="#222"),
    )
    st.plotly_chart(fig_pos, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — ORDERS
# ═══════════════════════════════════════════════════════════════════════════════
# tab_orders
st.subheader("Recent Orders")

n_orders = st.slider("Show last N orders", 10, 200,
                     50, step=10, key="order_slider")
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
            plot_bgcolor="#000000", paper_bgcolor="#000000", font=dict(color="#ccc"),
            xaxis=dict(gridcolor="#222"), yaxis=dict(gridcolor="#222"),
        )
        st.plotly_chart(fig_sym, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — BACKTEST
# ═══════════════════════════════════════════════════════════════════════════════
# tab_backtest
st.subheader("Walk-Forward Backtest")
st.caption(
    "Fetches 5yr daily data, runs the full walk-forward backtester, displays results inline.")

with st.expander("Backtest configuration", expanded=True):
    bc1, bc2, bc3 = st.columns(3)
    bt_capital = bc1.number_input(
        "Initial capital ($)", value=100_000, step=10_000)
    bt_train = bc2.slider("Train window (bars)", 126, 504, 252, step=63)
    bt_test = bc3.slider("Test window (bars)", 63, 252, 126, step=63)

    bc4, bc5 = st.columns(2)
    bt_symbols = bc4.multiselect("Symbols", symbols, default=[primary])
    bt_slip = bc5.number_input("Slippage (bps)", value=5, step=1)

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
                # Use historical data to avoid SIP subscription limitations
                end_ = pd.Timestamp('2023-12-31')
                start_ = end_ - timedelta(days=365 * 5)
                for sym_ in bt_symbols:
                    req_ = StockBarsRequest(
                        symbol_or_symbols=sym_,
                        timeframe=TimeFrame.Day,
                        start=start_.isoformat(),
                        end=end_.isoformat(),
                        adjustment="all",
                    )
                    bars = dc.get_stock_bars(req_)
                    df_ = bars.df
                    # Handle MultiIndex from Alpaca API
                    if isinstance(df_.index, pd.MultiIndex):
                        df_ = df_.reset_index()
                        df_['timestamp'] = pd.to_datetime(
                            df_['timestamp'], utc=True)
                        df_ = df_.set_index('timestamp')
                    else:
                        df_.index = pd.to_datetime(df_.index, utc=True)
                    bars_by_sym[sym_] = df_[
                        ["open", "high", "low", "close", "volume"]].copy()

                bt_cfg = BacktestConfig(
                    initial_capital=bt_capital,
                    slippage_pct=bt_slip / 10_000,
                    train_window=bt_train,
                    test_window=bt_test,
                    step_size=bt_test,
                    primary_symbol=bt_symbols[0],
                )
                backtester = WalkForwardBacktester(
                    bt_cfg,
                    hmm_config=cfg_dict.get("hmm", {}),
                    orchestrator_config=cfg_dict.get("strategy", {}),
                )
                result = backtester.run(bars_by_sym)

                analyser = PerformanceAnalyser()
                benchmark = bars_by_sym.get(bt_symbols[0])
                report = analyser.analyse(
                    result, benchmark_ohlcv=benchmark)

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
                k8.metric("Profit Factor", f"{c.profit_factor:.3f}")
                k9.metric("Worst Day",    f"{c.worst_day_pct:.2f}%")
                k10.metric("N Trades",    str(c.n_trades))

                # ── Equity curve ─────────────────────────────────────
                st.markdown("**Equity Curve**")
                eq = result.equity_curve
                if not eq.empty and benchmark is not None:
                    bah = bt_capital * \
                        benchmark["close"].reindex(
                            eq.index).ffill() / benchmark["close"].iloc[0]
                    fig_eq = go.Figure()
                    fig_eq.add_trace(go.Scatter(
                        x=eq.index, y=eq.values, name="Strategy", line=dict(color="#4da6ff", width=2)))
                    fig_eq.add_trace(go.Scatter(x=bah.index, y=bah.values, name="Buy & Hold", line=dict(
                        color="#aaaaaa", width=1, dash="dash")))
                    fig_eq.update_layout(
                        height=380, margin=dict(l=0, r=0, t=10, b=0),
                        yaxis_title="Equity ($)", xaxis_title=None,
                        plot_bgcolor="#000000", paper_bgcolor="#000000", font=dict(color="#ccc"),
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
                    plot_bgcolor="#000000", paper_bgcolor="#000000", font=dict(color="#ccc"),
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
                    st.dataframe(pd.DataFrame(bm_rows),
                                 use_container_width=True, hide_index=True)

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
                    st.dataframe(rb, use_container_width=True,
                                 hide_index=True)

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
