"""
Reusable Streamlit UI components for the AlphaForge dashboard.

Each function encapsulates a logical section of the UI, keeping
app.py clean and focused on orchestration.
"""

import base64
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np

# ── Logo helpers ──────────────────────────────────────────────────────────────
# Drop assets/logo.png or assets/logo_mark.png into the assets/ folder to
# activate the image-based header.  Falls back to the styled text header
# automatically when no file is present.

_ASSETS         = Path(__file__).resolve().parent.parent.parent / "assets"
_LOGO_PATH      = _ASSETS / "logo.png"
_LOGO_MARK_PATH = _ASSETS / "logo_mark.png"


def _b64(path: Path) -> str | None:
    if path.exists():
        return base64.b64encode(path.read_bytes()).decode()
    return None


def _img_tag(path: Path, height: int, style: str = "") -> str:
    b64 = _b64(path)
    if not b64:
        return ""
    base_style = "display:inline-block; vertical-align:middle; border-radius:4px;"
    return (
        f'<img src="data:image/png;base64,{b64}" height="{height}" '
        f'style="{base_style} {style}" alt="AlphaForge">'
    )


def _get_logo_b64() -> str | None:
    return _b64(_LOGO_PATH)

def _logo_img_tag(height: int = 48, extra_style: str = "") -> str:
    return _img_tag(_LOGO_PATH, height, extra_style)


# ── CSS injection ─────────────────────────────────────────────────────────────

def inject_custom_css() -> None:
    """Inject custom CSS for a polished dark-mode professional look."""
    st.markdown(
        """
        <style>
        /* ── Metric cards ───────────────────────────────── */
        [data-testid="metric-container"] {
            background: linear-gradient(135deg, #151B2E 0%, #1A2238 100%);
            border: 1px solid #1E2A3A;
            border-radius: 12px;
            padding: 0.9rem 1rem;
            box-shadow: 0 4px 20px rgba(0,0,0,0.4);
            transition: transform 0.2s, box-shadow 0.2s;
        }
        [data-testid="metric-container"]:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 25px rgba(0,212,255,0.12);
        }
        [data-testid="stMetricLabel"] {
            font-size: 0.75rem !important;
            font-weight: 600 !important;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #888 !important;
        }
        [data-testid="stMetricValue"] {
            font-size: 1.4rem !important;
            font-weight: 700 !important;
            color: #E8E8E8 !important;
        }
        /* ── Tabs ──────────────────────────────────────── */
        .stTabs [data-baseweb="tab-list"] {
            gap: 6px;
            background-color: #0E1117;
            border-radius: 10px;
            padding: 4px;
            border: 1px solid #1E2A3A;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 8px;
            color: #888;
            font-weight: 500;
            font-size: 0.85rem;
            padding: 6px 14px;
        }
        .stTabs [aria-selected="true"] {
            background-color: #00D4FF !important;
            color: #0E1117 !important;
            font-weight: 700 !important;
        }
        /* ── Sidebar ────────────────────────────────────── */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0E1117 0%, #151B2E 100%);
            border-right: 1px solid #1E2A3A;
        }
        /* ── Buttons ─────────────────────────────────────── */
        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #00D4FF 0%, #0099CC 100%);
            color: #0E1117;
            border: none;
            border-radius: 10px;
            font-weight: 700;
            font-size: 0.95rem;
            padding: 0.6rem 1.2rem;
            transition: transform 0.15s, box-shadow 0.15s;
        }
        .stButton > button[kind="primary"]:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 15px rgba(0,212,255,0.35);
        }
        /* ── Section headers ─────────────────────────────── */
        h3 {
            font-weight: 700 !important;
            letter-spacing: -0.02em !important;
            border-bottom: 1px solid #1E2A3A;
            padding-bottom: 0.4rem;
            margin-bottom: 1rem !important;
        }
        /* ── Expanders ───────────────────────────────────── */
        [data-testid="stExpander"] {
            border: 1px solid #1E2A3A;
            border-radius: 10px;
        }
        /* ── Dataframes ──────────────────────────────────── */
        [data-testid="stDataFrame"] {
            border-radius: 8px;
            overflow: hidden;
        }
        /* ── Info / warning boxes ─────────────────────────── */
        [data-testid="stAlert"] {
            border-radius: 10px;
        }
        /* ── Dividers ─────────────────────────────────────── */
        hr {
            border-color: #1E2A3A !important;
            margin: 1.5rem 0 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ── Helpers ──────────────────────────────────────────────────────────────────

def _pct(val, decimals: int = 2) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    sign = "+" if val >= 0 else ""
    return f"{sign}{val * 100:.{decimals}f}%"

def _usd(val) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    return f"${val:,.2f}"

def _fmt(val, decimals: int = 3) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    return f"{val:.{decimals}f}"

def _delta_color(val) -> str:
    if val is None:
        return "off"
    return "normal" if val >= 0 else "inverse"


# ── Header ───────────────────────────────────────────────────────────────────

def render_header() -> None:
    """AlphaForge branded header. Uses image if assets/logo.png present, else styled text."""
    logo_tag = _img_tag(_LOGO_PATH, height=160, style="margin:0 auto; display:block;")

    if logo_tag:
        st.markdown(
            f"<div style='text-align:center; padding:1.2rem 0 0.2rem 0;'>{logo_tag}</div>"
            "<hr style='border:1px solid #1E2A3A; margin-top:0.8rem;'>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div style='text-align:center; padding:1.5rem 0 0.3rem 0;'>
                <h1 style='font-size:3rem; font-weight:900; margin-bottom:0.1rem;
                           background:linear-gradient(135deg,#00D4FF,#0099CC);
                           -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                           background-clip:text; letter-spacing:-2px;'>
                    ⚡ AlphaForge
                </h1>
                <p style='font-size:1rem; color:#666; margin-top:0; letter-spacing:0.1em;
                          text-transform:uppercase; font-weight:600;'>
                    Quantitative Trading Research Platform
                </p>
            </div>
            <hr style='border:1px solid #1E2A3A;'>
            """,
            unsafe_allow_html=True,
        )


# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar() -> dict:
    """
    Render all sidebar controls and return user selections as a dict.
    """
    import datetime

    # ── Sidebar brand block ─────────────────────────────────────────────────
    mark_tag = _img_tag(_LOGO_MARK_PATH, height=48,
                        style="margin:0 auto 4px auto; display:block;")
    icon_html = mark_tag if mark_tag else "<div style='font-size:2rem;text-align:center;'>⚡</div>"

    st.sidebar.markdown(
        f"""
        <div style='text-align:center; padding:0.6rem 0 0.2rem 0;'>
            {icon_html}
            <div style='color:#00D4FF; font-size:1.0rem; font-weight:800;
                        letter-spacing:0.12em; text-transform:uppercase; margin-top:4px;'>
                AlphaForge
            </div>
            <div style='color:#444; font-size:0.62rem; text-transform:uppercase;
                        letter-spacing:0.13em; margin-top:1px;'>
                Quant Research Platform
            </div>
        </div>
        <hr style='border:1px solid #1E2A3A; margin:0.6rem 0 0.8rem 0;'>
        <h2 style='color:#00D4FF; font-size:1.0rem; letter-spacing:0.05em;
                   margin-bottom:0.6rem;'>⚙️ CONFIGURATION</h2>
        """,
        unsafe_allow_html=True,
    )

    # ── Asset ──
    from src.data.asset_universe import (
        get_all_labels, get_asset_types, get_label_for_symbol,
        parse_symbol_from_label, load_asset_universe,
    )

    st.sidebar.markdown("**📈 Asset Selection**")

    # Asset type filter
    _all_types = get_asset_types()
    _selected_types = st.sidebar.multiselect(
        "Filter by Asset Type",
        options=_all_types,
        default=[],
        placeholder="All types",
        help="Narrow the asset list below by type.",
    )

    # Filtered option labels
    _filtered_labels = get_all_labels(_selected_types if _selected_types else None)

    # Compute default selection — AAPL if visible, else first option
    _aapl_label = get_label_for_symbol("AAPL")
    _default_selection = (
        [_aapl_label]
        if _aapl_label and _aapl_label in _filtered_labels
        else ([_filtered_labels[0]] if _filtered_labels else [])
    )

    selected_labels = st.sidebar.multiselect(
        "Selected Assets",
        options=_filtered_labels,
        default=_default_selection,
        placeholder="Type to search by ticker or name…",
        help=(
            "Search by ticker (e.g. AAPL) or company name (e.g. Apple). "
            "Select multiple assets to run a multi-asset comparison."
        ),
    )

    # Manual / custom ticker fallback
    with st.sidebar.expander("➕ Add unlisted ticker"):
        st.caption("Enter any yfinance-compatible ticker not in the list above.")
        _manual_raw = st.text_input(
            "Custom ticker symbol",
            value="",
            placeholder="e.g. BRK-B, BTC-USD, ^VIX",
            key="manual_ticker_input",
        ).upper().strip()
        if _manual_raw:
            st.caption(f"Will add: **{_manual_raw}**")

    # Build primary tickers list
    tickers: list[str] = [parse_symbol_from_label(lbl) for lbl in selected_labels]
    if _manual_raw and _manual_raw not in tickers:
        tickers.append(_manual_raw)

    # Fall back to AAPL so the rest of the app always has at least one ticker
    if not tickers:
        tickers = ["AAPL"]
        st.sidebar.warning("⚠️ No asset selected — defaulting to AAPL.")

    ticker = tickers[0]  # primary ticker (backward-compat with single-asset flow)

    # Benchmark — full universe selectbox
    _bm_labels   = get_all_labels()           # all types, not filtered
    _spy_label   = get_label_for_symbol("SPY")
    _bm_default  = _bm_labels.index(_spy_label) if _spy_label in _bm_labels else 0
    _bm_selected = st.sidebar.selectbox(
        "Benchmark",
        options=_bm_labels,
        index=_bm_default,
        help="Comparison benchmark — type to search (e.g. 'S&P' or 'SPY')",
    )
    benchmark = parse_symbol_from_label(_bm_selected)

    # ── Dates ──
    st.sidebar.markdown("**📅 Date Range**")
    col1, col2 = st.sidebar.columns(2)
    start_date = col1.date_input(
        "Start", value=datetime.date(2020, 1, 1),
        min_value=datetime.date(2000, 1, 1),
        max_value=datetime.date.today(),
    )
    end_date = col2.date_input(
        "End", value=datetime.date(2024, 12, 31),
        min_value=datetime.date(2000, 1, 2),
        max_value=datetime.date.today(),
    )

    # ── Portfolio ──
    st.sidebar.markdown("**💰 Portfolio**")
    initial_capital = st.sidebar.number_input(
        "Initial Capital ($)", min_value=1_000, max_value=100_000_000,
        value=100_000, step=10_000,
    )
    transaction_cost = st.sidebar.slider(
        "Transaction Cost (%)", 0.0, 1.0, 0.1, 0.01, format="%.2f%%"
    ) / 100
    slippage = st.sidebar.slider(
        "Slippage (%)", 0.0, 0.5, 0.05, 0.01, format="%.2f%%"
    ) / 100

    # ── Volatility targeting ──
    enable_vol_target = st.sidebar.checkbox("Volatility Targeting", value=False,
        help="Scale position size to target a specific annualised vol")
    vol_target = None
    if enable_vol_target:
        vol_target = st.sidebar.slider(
            "Target Volatility (%)", 5, 30, 15, 1, format="%d%%"
        ) / 100

    # ── Strategy ──
    st.sidebar.markdown("**🧠 Strategy**")
    strategy = st.sidebar.selectbox(
        "Strategy",
        options=[
            "Moving Average Crossover",
            "Mean Reversion",
            "Momentum",
            "RSI Mean Reversion",
            "Bollinger Bands",
            "Buy & Hold",
        ],
    )

    strategy_params = {}
    if strategy == "Moving Average Crossover":
        st.sidebar.markdown("*MA Parameters*")
        short_w = st.sidebar.slider("Short Window", 5, 100, 20)
        long_w  = st.sidebar.slider("Long Window", short_w + 5, 300, 50)
        strategy_params = {"short_window": short_w, "long_window": long_w}

    elif strategy == "Mean Reversion":
        st.sidebar.markdown("*Mean Reversion Parameters*")
        window    = st.sidebar.slider("Rolling Window", 5, 100, 20)
        entry_thr = st.sidebar.slider("Entry Z-Score", 0.5, 4.0, 2.0, 0.1)
        exit_thr  = st.sidebar.slider(
            "Exit Z-Score", 0.0, float(entry_thr) - 0.1, 0.5, 0.1
        )
        strategy_params = {
            "window": window,
            "entry_threshold": entry_thr,
            "exit_threshold": exit_thr,
        }

    elif strategy == "Momentum":
        st.sidebar.markdown("*Momentum Parameters*")
        lookback  = st.sidebar.slider("Lookback Window", 20, 252, 90)
        skip_days = st.sidebar.slider("Skip Recent Days", 0, 20, 1)
        strategy_params = {"lookback": lookback, "skip_days": skip_days}

    elif strategy == "RSI Mean Reversion":
        st.sidebar.markdown("*RSI Parameters*")
        period    = st.sidebar.slider("RSI Period", 2, 50, 14)
        oversold  = st.sidebar.slider("Oversold Level", 10, 45, 30)
        overbought= st.sidebar.slider("Overbought Level", 55, 90, 70)
        exit_lv   = st.sidebar.slider("Exit Level", oversold + 1, overbought - 1, 50)
        strategy_params = {
            "period": period,
            "oversold": float(oversold),
            "overbought": float(overbought),
            "exit_level": float(exit_lv),
        }

    elif strategy == "Bollinger Bands":
        st.sidebar.markdown("*Bollinger Band Parameters*")
        window    = st.sidebar.slider("Window", 5, 100, 20)
        n_std     = st.sidebar.slider("Std Deviations", 0.5, 4.0, 2.0, 0.1)
        exit_mid  = st.sidebar.checkbox("Exit at Middle Band", value=True)
        strategy_params = {
            "window": window,
            "n_std": n_std,
            "exit_at_middle": exit_mid,
        }

    # ── Research Lab advanced options ──
    st.sidebar.markdown("**🔬 Research Lab Options**")
    comparison_mode = False   # handled by Strategy Comparison tab
    enable_opt      = st.sidebar.checkbox(
        "Parameter Optimisation", value=False,
        help="Grid-search + walk-forward (MA Crossover & Mean Reversion only)",
    )
    enable_mc  = st.sidebar.checkbox(
        "Monte Carlo Simulation", value=False,
        help="500-path fan chart after backtest",
    )
    n_mc    = 300
    mc_days = 252
    if enable_mc:
        n_mc    = st.sidebar.slider("# Simulations", 100, 1000, 300, 50)
        mc_days = st.sidebar.slider("Horizon (days)", 63, 504, 252, 21)

    risk_free_rate = st.sidebar.slider(
        "Risk-Free Rate (%)", 0.0, 8.0, 0.0, 0.25, format="%.2f%%"
    ) / 100

    st.sidebar.markdown("---")
    run_backtest = st.sidebar.button(
        "🚀 Run Backtest", use_container_width=True, type="primary",
        help="Runs the Research Lab backtest for the primary selected asset",
    )

    return {
        "ticker":            ticker,       # primary ticker string (backward-compat)
        "tickers":           tickers,      # full list for multi-asset mode
        "benchmark":         benchmark,
        "start_date":        str(start_date),
        "end_date":          str(end_date),
        "initial_capital":   float(initial_capital),
        "transaction_cost":  transaction_cost,
        "slippage":          slippage,
        "vol_target":        vol_target,
        "strategy":          strategy,
        "strategy_params":   strategy_params,
        "run_backtest":      run_backtest,
        "comparison_mode":   comparison_mode,
        "enable_optimization": enable_opt,
        "enable_monte_carlo":  enable_mc,
        "n_mc_simulations":    n_mc,
        "mc_days":             mc_days,
        "risk_free_rate":      risk_free_rate,
    }


# ── Config summary ────────────────────────────────────────────────────────────

def render_config_summary(config: dict) -> None:
    """Compact row of chips summarising the current backtest config."""
    st.markdown("### 📋 Backtest Configuration")
    cols = st.columns(7)
    items = [
        ("Ticker",      config["ticker"]),
        ("Benchmark",   config["benchmark"]),
        ("Strategy",    config["strategy"]),
        ("Capital",     _usd(config["initial_capital"])),
        ("Tx Cost",     _pct(config["transaction_cost"])),
        ("Slippage",    _pct(config["slippage"])),
        ("Vol Target",  _pct(config["vol_target"]) if config["vol_target"] else "Off"),
    ]
    for col, (label, value) in zip(cols, items):
        col.metric(label, value)
    st.caption(
        f"📅 {config['start_date']} → {config['end_date']}  |  "
        f"Params: {config['strategy_params'] or 'N/A'}  |  "
        f"Risk-Free Rate: {config['risk_free_rate']*100:.2f}%"
    )


# ── Key metrics grid ──────────────────────────────────────────────────────────

def render_metric_cards(metrics: dict) -> None:
    """Display three rows of metric cards covering all major performance dimensions."""
    st.markdown("### 📊 Performance Metrics")

    # Row 1 — Returns
    st.markdown(
        "<p style='font-size:0.7rem; color:#555; text-transform:uppercase; "
        "letter-spacing:0.1em; margin-bottom:0.5rem;'>RETURNS</p>",
        unsafe_allow_html=True,
    )
    cols = st.columns(4)
    cols[0].metric("Cumulative Return", _pct(metrics.get("Cumulative_Return")),
                   delta=_pct(metrics.get("Cumulative_Return")),
                   delta_color=_delta_color(metrics.get("Cumulative_Return")))
    cols[1].metric("Annualised Return (CAGR)", _pct(metrics.get("Annualised_Return")),
                   delta=_pct(metrics.get("Annualised_Return")),
                   delta_color=_delta_color(metrics.get("Annualised_Return")))
    cols[2].metric("Best Day",  _pct(metrics.get("Best_Day")))
    cols[3].metric("Worst Day", _pct(metrics.get("Worst_Day")),
                   delta=_pct(metrics.get("Worst_Day")), delta_color="inverse")

    # Row 2 — Risk-adjusted ratios
    st.markdown(
        "<p style='font-size:0.7rem; color:#555; text-transform:uppercase; "
        "letter-spacing:0.1em; margin-bottom:0.5rem; margin-top:0.7rem;'>RISK-ADJUSTED RATIOS</p>",
        unsafe_allow_html=True,
    )
    cols2 = st.columns(5)
    cols2[0].metric("Sharpe Ratio",   _fmt(metrics.get("Sharpe_Ratio"), 2))
    cols2[1].metric("Sortino Ratio",  _fmt(metrics.get("Sortino_Ratio"), 2))
    cols2[2].metric("Calmar Ratio",   _fmt(metrics.get("Calmar_Ratio"), 2))
    cols2[3].metric("Omega Ratio",    _fmt(metrics.get("Omega_Ratio"), 2))
    cols2[4].metric("Serenity Ratio", _fmt(metrics.get("Serenity_Ratio"), 2))

    # Row 3 — Risk
    st.markdown(
        "<p style='font-size:0.7rem; color:#555; text-transform:uppercase; "
        "letter-spacing:0.1em; margin-bottom:0.5rem; margin-top:0.7rem;'>RISK</p>",
        unsafe_allow_html=True,
    )
    cols3 = st.columns(5)
    cols3[0].metric("Max Drawdown",   _pct(metrics.get("Max_Drawdown")),
                    delta=_pct(metrics.get("Max_Drawdown")), delta_color="inverse")
    cols3[1].metric("Ann. Volatility", _pct(metrics.get("Annualised_Volatility")))
    cols3[2].metric("Ulcer Index",    _fmt(metrics.get("Ulcer_Index"), 4))
    cols3[3].metric("VaR (95%, 1d)",  _pct(metrics.get("VaR_Historical_95")))
    cols3[4].metric("Expected Shortfall (95%)", _pct(metrics.get("Expected_Shortfall_95")))

    # Row 4 — Benchmark
    if any(k in metrics for k in ["Alpha", "Beta", "Correlation"]):
        st.markdown(
            "<p style='font-size:0.7rem; color:#555; text-transform:uppercase; "
            "letter-spacing:0.1em; margin-bottom:0.5rem; margin-top:0.7rem;'>BENCHMARK</p>",
            unsafe_allow_html=True,
        )
        cols4 = st.columns(5)
        cols4[0].metric("Alpha (Ann.)",       _pct(metrics.get("Alpha")))
        cols4[1].metric("Beta",               _fmt(metrics.get("Beta"), 2))
        cols4[2].metric("Correlation",        _fmt(metrics.get("Correlation"), 2))
        cols4[3].metric("Tracking Error",     _pct(metrics.get("Tracking_Error")))
        cols4[4].metric("Information Ratio",  _fmt(metrics.get("Information_Ratio"), 2))

    # Row 5 — Trade stats
    st.markdown(
        "<p style='font-size:0.7rem; color:#555; text-transform:uppercase; "
        "letter-spacing:0.1em; margin-bottom:0.5rem; margin-top:0.7rem;'>TRADE STATISTICS</p>",
        unsafe_allow_html=True,
    )
    cols5 = st.columns(5)
    cols5[0].metric("Win Rate",      _pct(metrics.get("Win_Rate")))
    cols5[1].metric("Profit Factor", _fmt(metrics.get("Profit_Factor"), 2))
    cols5[2].metric("Tail Ratio",    _fmt(metrics.get("Tail_Ratio"), 2))
    cols5[3].metric("Payoff Ratio",  _fmt(metrics.get("Payoff_Ratio"), 2))
    cols5[4].metric("Gain / Pain",   _fmt(metrics.get("Gain_to_Pain"), 2))


# ── Trade log ─────────────────────────────────────────────────────────────────

def render_trade_log(trade_df: pd.DataFrame) -> None:
    """Display a styled trade log table for BUY/SELL events only."""
    st.markdown("### 📒 Trade Log")

    if trade_df is None or trade_df.empty:
        st.info("No trades recorded.")
        return

    display = trade_df[trade_df["Direction"].isin(["BUY", "SELL"])].copy()
    if display.empty:
        st.info("No BUY/SELL orders executed during the backtest period.")
        return

    display["Date"] = pd.to_datetime(display["Date"]).dt.strftime("%Y-%m-%d")

    def _color_dir(v):
        if v == "BUY":
            return "color:#00D4FF; font-weight:600"
        if v == "SELL":
            return "color:#FF6B35; font-weight:600"
        return ""

    st.dataframe(
        display.style.applymap(_color_dir, subset=["Direction"]),
        use_container_width=True,
        height=min(400, 35 * len(display) + 38),
    )
    total_commission = display["Commission"].sum()
    st.caption(
        f"**{len(display)} trades** | "
        f"Total commission paid: **{_usd(total_commission)}**"
    )


# ── Comparison metrics table ──────────────────────────────────────────────────

def render_comparison_table(comparison_metrics: dict[str, dict]) -> None:
    """
    Render a side-by-side metrics table comparing all strategies.

    Parameters
    ----------
    comparison_metrics : dict[strategy_name → metrics_dict]
    """
    st.markdown("### 🏆 Strategy Comparison")

    KEY_METRICS = [
        ("Cumulative_Return",    "Cumulative Return",    True),
        ("Annualised_Return",    "Ann. Return (CAGR)",   True),
        ("Sharpe_Ratio",         "Sharpe Ratio",         False),
        ("Sortino_Ratio",        "Sortino Ratio",        False),
        ("Calmar_Ratio",         "Calmar Ratio",         False),
        ("Omega_Ratio",          "Omega Ratio",          False),
        ("Max_Drawdown",         "Max Drawdown",         False),
        ("Annualised_Volatility","Ann. Volatility",      False),
        ("Win_Rate",             "Win Rate",             True),
        ("Profit_Factor",        "Profit Factor",        False),
        ("Alpha",                "Alpha",                True),
        ("Beta",                 "Beta",                 False),
        ("VaR_Historical_95",    "VaR 95%",              False),
    ]

    rows = {}
    for key, label, is_pct in KEY_METRICS:
        row = {}
        for name, metrics in comparison_metrics.items():
            val = metrics.get(key)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                row[name] = "—"
            elif is_pct:
                row[name] = _pct(val)
            else:
                row[name] = _fmt(val, 3)
        rows[label] = row

    df = pd.DataFrame(rows).T
    df.index.name = "Metric"

    st.dataframe(df, use_container_width=True)

    # Best strategy by Sharpe
    best_sharpe = max(
        comparison_metrics.items(),
        key=lambda x: x[1].get("Sharpe_Ratio", -999),
    )
    st.success(
        f"🏅 Highest Sharpe Ratio: **{best_sharpe[0]}** "
        f"({_fmt(best_sharpe[1].get('Sharpe_Ratio'), 3)})"
    )


# ── Walk-forward ──────────────────────────────────────────────────────────────

def render_walk_forward(wf_df: pd.DataFrame, summary: dict) -> None:
    """Display walk-forward validation results."""
    st.markdown("### 🔄 Walk-Forward Validation")

    if wf_df.empty:
        st.warning("Walk-forward returned no results. Try a longer date range.")
        return

    col1, col2, col3 = st.columns(3)
    col1.metric("Avg In-Sample Sharpe",     _fmt(summary.get("Avg_IS_Sharpe")))
    col2.metric("Avg Out-of-Sample Sharpe", _fmt(summary.get("Avg_OOS_Sharpe")))
    col3.metric(
        "Degradation Ratio",
        _fmt(summary.get("Sharpe_Degradation_Ratio")),
        help="OOS / IS Sharpe. Values near 1.0 = low overfitting risk.",
    )

    st.dataframe(wf_df, use_container_width=True)

    ratio = summary.get("Sharpe_Degradation_Ratio", 1.0)
    if ratio < 0.5:
        st.error("⚠️ Severe overfitting detected. OOS performance is less than 50% of IS.")
    elif ratio < 0.75:
        st.warning("⚠️ Moderate overfitting. OOS Sharpe significantly lags IS Sharpe.")
    else:
        st.success("✅ Acceptable IS/OOS degradation. Strategy generalises reasonably well.")


# ── Optimisation results ──────────────────────────────────────────────────────

def render_optimization_results(opt_df: pd.DataFrame, strategy_name: str) -> None:
    """Display grid-search optimisation results with overfitting warning."""
    st.markdown("### ⚗️ Parameter Optimisation Results")
    st.warning(
        "⚠️ **Overfitting Risk**: Parameters optimised on historical data may not "
        "generalise to future markets. Always pair with walk-forward validation."
    )
    if opt_df.empty:
        st.info("No optimisation results.")
        return
    st.markdown(f"**Top 10 Combinations for {strategy_name}** (sorted by Sharpe Ratio)")
    st.dataframe(
        opt_df.head(10).style.background_gradient(
            subset=["Sharpe_Ratio"], cmap="RdYlGn"
        ),
        use_container_width=True,
    )


# ── Monte Carlo summary ───────────────────────────────────────────────────────

def render_monte_carlo_summary(mc_results: dict) -> None:
    """Render Monte Carlo stats."""
    st.markdown("### 🎲 Monte Carlo Simulation")
    st.caption(
        f"{mc_results.get('n_simulations','?')} paths · "
        f"{mc_results.get('n_days','?')}-day horizon · "
        f"μ = {mc_results.get('mu',0)*100:.3f}%/day · "
        f"σ = {mc_results.get('sigma',0)*100:.3f}%/day"
    )

    cols = st.columns(5)
    cols[0].metric("5th Pct",  _usd(mc_results.get("percentile_5")))
    cols[1].metric("25th Pct", _usd(mc_results.get("percentile_25")))
    cols[2].metric("Median",   _usd(mc_results.get("percentile_50")))
    cols[3].metric("75th Pct", _usd(mc_results.get("percentile_75")))
    cols[4].metric("95th Pct", _usd(mc_results.get("percentile_95")))

    prob = mc_results.get("prob_loss", 0)
    col_a, col_b = st.columns(2)
    col_a.metric(
        "Probability of Loss",
        _pct(prob),
        delta=_pct(prob),
        delta_color="inverse",
    )
    col_b.metric(
        "Expected Final Value",
        _usd(mc_results.get("mean_final")),
    )
    st.caption(
        "**Disclaimer**: Assumes i.i.d. Gaussian returns. Real markets exhibit "
        "fat tails, volatility clustering, and regime shifts. Results are "
        "illustrative only — not predictive."
    )


# ── Export section ────────────────────────────────────────────────────────────

def render_export_section(
    metrics: dict,
    trade_df: pd.DataFrame,
    equity_df: pd.DataFrame,
    ticker: str,
    strategy: str,
) -> None:
    """Render CSV download buttons."""
    from ..storage.exporter import dataframe_to_csv_bytes, metrics_to_csv_bytes

    st.markdown("### 💾 Export Results")
    col1, col2, col3 = st.columns(3)

    col1.download_button(
        label="📥 Metrics CSV",
        data=metrics_to_csv_bytes(metrics),
        file_name=f"{ticker}_{strategy.replace(' ','_')}_metrics.csv",
        mime="text/csv",
        use_container_width=True,
    )
    if trade_df is not None and not trade_df.empty:
        col2.download_button(
            label="📥 Trade Log CSV",
            data=dataframe_to_csv_bytes(trade_df),
            file_name=f"{ticker}_{strategy.replace(' ','_')}_trades.csv",
            mime="text/csv",
            use_container_width=True,
        )
    if equity_df is not None and not equity_df.empty:
        col3.download_button(
            label="📥 Portfolio CSV",
            data=dataframe_to_csv_bytes(equity_df),
            file_name=f"{ticker}_{strategy.replace(' ','_')}_portfolio.csv",
            mime="text/csv",
            use_container_width=True,
        )


# ── Run history ───────────────────────────────────────────────────────────────

def render_run_history(runs_df: pd.DataFrame) -> None:
    """Display past backtest runs from the database."""
    st.markdown("### 🗂️ Backtest Run History")
    if runs_df.empty:
        st.info("No previous backtest runs stored yet.")
        return

    display_cols = [
        "run_id", "timestamp", "ticker", "strategy",
        "start_date", "end_date", "initial_capital",
        "sharpe_ratio", "max_drawdown", "annualised_return", "cumulative_return",
    ]
    available = [c for c in display_cols if c in runs_df.columns]
    st.dataframe(
        runs_df[available].style.background_gradient(
            subset=["sharpe_ratio"] if "sharpe_ratio" in available else [],
            cmap="RdYlGn",
        ),
        use_container_width=True,
        height=300,
    )
