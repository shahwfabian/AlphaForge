"""
AlphaForge — Institutional Quantitative Research Platform
==========================================================
Entry point for the Streamlit dashboard.

Tabs
----
1. 🔬 Research Lab       — single-asset / single-strategy backtest
2. 📊 Portfolio Builder  — multi-asset weighted portfolio construction
3. ⚔️  Strategy Comparison — rank strategies side-by-side on one asset
4. 🧮 Risk Analytics     — factor exposure, stress testing, forward testing
5. 📋 Tear Sheet         — generate & download professional report
6. 💾 Saved Runs         — persist, reload, and manage configurations

Run locally:
    streamlit run app.py
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("alphaforge.app")

# ── Page config (must be first Streamlit call) ────────────────────────────────
# Favicon: uses assets/logo_mark.png when present, otherwise the ⚡ emoji.
try:
    from PIL import Image as _PILImage
    _icon_candidates = [ROOT / "assets" / "logo_mark.png", ROOT / "assets" / "logo.png"]
    _page_icon: "str | _PILImage.Image" = "⚡"
    for _p in _icon_candidates:
        if _p.exists():
            _page_icon = _PILImage.open(_p)
            break
except Exception:
    _page_icon = "⚡"

st.set_page_config(
    page_title="AlphaForge",
    page_icon=_page_icon,
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "AlphaForge — Institutional Quantitative Research Platform"},
)

# ── AlphaForge imports ────────────────────────────────────────────────────────
from src.data.data_loader import load_market_data
from src.data.preprocessing import preprocess
from src.data.validation import (
    ValidationError, validate_tickers, validate_date_range,
    validate_capital, validate_cost_params,
)
from src.strategies.moving_average import MovingAverageCrossover
from src.strategies.mean_reversion import MeanReversion
from src.strategies.momentum import Momentum
from src.strategies.rsi import RSIMeanReversion
from src.strategies.bollinger import BollingerBands
from src.strategies.buy_hold import BuyAndHold
from src.backtester.engine import BacktestEngine
from src.analytics.performance import (
    compute_all_metrics, monthly_returns_table, annual_returns,
)
from src.analytics.risk import compute_risk_metrics
from src.analytics.benchmark import compute_benchmark_metrics
from src.analytics.monte_carlo import run_monte_carlo
from src.analytics.optimizer import grid_search_ma, grid_search_mr
from src.analytics.walk_forward import walk_forward_ma, summarise_walk_forward
from src.analytics.comparison import (
    rank_strategies, format_comparison_table, best_strategy,
)
from src.analytics.factor_analysis import (
    fetch_factor_returns, run_factor_regression, FACTOR_PROXIES,
)
from src.analytics.stress_testing import run_stress_scenarios, custom_scenario, STRESS_SCENARIOS
from src.analytics.diversification import (
    correlation_matrix, diversification_report,
)
from src.analytics.forward_testing import run_forward_test, split_data
from src.portfolio.allocation import (
    equal_weight_allocation, normalize_weights, validate_weights,
    calculate_portfolio_returns, allocation_summary, dominant_asset_warning,
    cumulative_portfolio_value,
)
from src.reports.tear_sheet import generate_html_tear_sheet, generate_csv_report
from src.storage.database import save_run, load_runs
from src.storage.strategy_configs import (
    save_config, load_config, list_configs, delete_config, rename_config,
)
from src.dashboard.components import (
    inject_custom_css, render_header, render_sidebar,
    render_config_summary, render_metric_cards, render_trade_log,
    render_walk_forward, render_optimization_results,
    render_monte_carlo_summary, render_export_section,
    render_run_history, render_comparison_table,
)
from src.dashboard.charts import (
    equity_curve_chart, drawdown_chart, returns_distribution_chart,
    rolling_volatility_chart, monte_carlo_chart, cumulative_returns_comparison,
    optimization_heatmap, signal_overlay_chart, monthly_returns_heatmap,
    annual_returns_chart, rolling_sharpe_chart, returns_scatter_chart,
    underwater_chart, strategy_comparison_chart,
    allocation_pie_chart, correlation_heatmap, factor_exposure_chart,
    stress_test_chart, drawdown_comparison_chart,
)

# ── Strategy registry ─────────────────────────────────────────────────────────
ALL_STRATEGIES = [
    "Moving Average Crossover",
    "Mean Reversion",
    "Momentum",
    "RSI Mean Reversion",
    "Bollinger Bands",
    "Buy & Hold",
]

DEFAULT_PARAMS: dict[str, dict] = {
    "Moving Average Crossover": {"short_window": 20, "long_window": 50},
    "Mean Reversion":           {"window": 20, "entry_threshold": 2.0, "exit_threshold": 0.5},
    "Momentum":                 {"lookback": 90, "skip_days": 1},
    "RSI Mean Reversion":       {"period": 14, "oversold": 30.0, "overbought": 70.0, "exit_level": 50.0},
    "Bollinger Bands":          {"window": 20, "n_std": 2.0, "exit_at_middle": True},
    "Buy & Hold":               {},
}

# ── Cached data helpers ───────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def _cached_load(tickers: list[str], start: str, end: str) -> dict:
    return load_market_data(tickers, start, end)


@st.cache_data(ttl=3600, show_spinner=False)
def _cached_preprocess(ticker: str, start: str, end: str) -> pd.DataFrame:
    raw = load_market_data([ticker], start, end)
    if ticker not in raw:
        return pd.DataFrame()
    return preprocess(raw[ticker], ticker, save=True)


@st.cache_data(ttl=3600, show_spinner=False)
def _cached_factor_returns(start: str, end: str) -> dict:
    return fetch_factor_returns(start, end)


# ── Strategy builder ──────────────────────────────────────────────────────────

def _build_strategy(name: str, params: dict):
    if name == "Moving Average Crossover":
        return MovingAverageCrossover(**params)
    if name == "Mean Reversion":
        return MeanReversion(**params)
    if name == "Momentum":
        return Momentum(**params)
    if name == "RSI Mean Reversion":
        return RSIMeanReversion(**params)
    if name == "Bollinger Bands":
        return BollingerBands(**params)
    return BuyAndHold()


def _run_backtest(
    df: pd.DataFrame, ticker: str, strategy_name: str, params: dict,
    capital: float, tx_cost: float, slippage: float, vol_target=None,
) -> dict:
    strategy = _build_strategy(strategy_name, params)
    engine   = BacktestEngine(
        strategy=strategy,
        initial_capital=capital,
        transaction_cost=tx_cost,
        slippage=slippage,
        vol_target=vol_target,
    )
    return engine.run(df, ticker)


# ── Validation ────────────────────────────────────────────────────────────────

def _validate(config: dict) -> list[str]:
    errors = []
    if not config.get("tickers"):
        errors.append("Select at least one asset before running a backtest.")
        return errors
    try:
        validate_tickers(config["tickers"] + [config["benchmark"]])
    except ValidationError as e:
        errors.append(str(e))
    try:
        validate_date_range(config["start_date"], config["end_date"])
    except ValidationError as e:
        errors.append(str(e))
    try:
        validate_capital(config["initial_capital"])
    except ValidationError as e:
        errors.append(str(e))
    try:
        validate_cost_params(config["transaction_cost"], config["slippage"])
    except ValidationError as e:
        errors.append(str(e))
    return errors


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    inject_custom_css()
    render_header()
    config = render_sidebar()

    tabs = st.tabs([
        "🔬 Research Lab",
        "📊 Portfolio Builder",
        "⚔️  Strategy Comparison",
        "🧮 Risk Analytics",
        "📋 Tear Sheet",
        "💾 Saved Runs",
    ])

    with tabs[0]:
        _tab_research_lab(config)
    with tabs[1]:
        _tab_portfolio_builder(config)
    with tabs[2]:
        _tab_strategy_comparison(config)
    with tabs[3]:
        _tab_risk_analytics(config)
    with tabs[4]:
        _tab_tear_sheet(config)
    with tabs[5]:
        _tab_saved_runs(config)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — RESEARCH LAB
# ═══════════════════════════════════════════════════════════════════════════════

def _tab_research_lab(config: dict) -> None:
    """Single-asset, single-strategy backtest with full analytics suite."""

    if not config.get("run_backtest") and "af_result" not in st.session_state:
        _render_landing()
        return

    if config.get("run_backtest"):
        errors = _validate(config)
        if errors:
            for e in errors:
                st.error(f"❌ {e}")
            return

        _run_and_store(config)

    # ── Retrieve stored results ───────────────────────────────────────────────
    if "af_result" not in st.session_state:
        st.info("Configure your backtest in the sidebar and click **🚀 Run Backtest**.")
        return

    result      = st.session_state["af_result"]
    all_metrics = st.session_state["af_metrics"]
    equity_df   = st.session_state["af_equity"]
    signals_df  = st.session_state["af_signals"]
    asset_df    = st.session_state["af_asset_df"]
    trade_df    = result["trade_log"]
    bm_result   = st.session_state.get("af_bm_result")
    snap        = st.session_state["af_config"]
    daily_rets  = equity_df["Daily_Return"].dropna()

    # ── Multi-asset check ─────────────────────────────────────────────────────
    if len(snap.get("tickers", [])) > 1:
        st.info(
            f"Multiple assets selected ({', '.join(snap['tickers'])}). "
            "Switch to **📊 Portfolio Builder** for weighted multi-asset analysis."
        )

    # ── Dashboard ─────────────────────────────────────────────────────────────
    render_config_summary(snap)
    st.markdown("---")
    render_metric_cards(all_metrics)
    st.markdown("---")

    _render_charts(equity_df, signals_df, asset_df, daily_rets, bm_result, snap, all_metrics)

    st.markdown("---")
    render_trade_log(trade_df)

    # Monte Carlo
    if snap.get("enable_monte_carlo"):
        st.markdown("---")
        with st.spinner("🎲 Running Monte Carlo simulation…"):
            mc = run_monte_carlo(
                returns=daily_rets,
                initial_value=snap["initial_capital"],
                n_simulations=snap.get("n_mc_simulations", 300),
                n_days=snap.get("mc_days", 252),
            )
        if mc:
            render_monte_carlo_summary(mc)
            st.plotly_chart(monte_carlo_chart(mc), use_container_width=True)

    # Optimisation + walk-forward
    if snap.get("enable_optimization"):
        _render_optimisation(asset_df, snap)

    # Forward testing
    with st.expander("🔭 Forward Testing (Out-of-Sample)", expanded=False):
        _render_forward_testing_section(asset_df, snap)

    # Export
    st.markdown("---")
    render_export_section(
        all_metrics, trade_df, equity_df,
        snap["ticker"], snap["strategy"],
    )

    # Run history
    with st.expander("🗂️ Backtest Run History", expanded=False):
        render_run_history(load_runs())


def _run_and_store(config: dict) -> None:
    """Execute a backtest and persist results in session_state."""
    ticker = config["ticker"]
    prog   = st.progress(0, text="📡 Fetching market data…")

    asset_df = _cached_preprocess(ticker, config["start_date"], config["end_date"])
    prog.progress(25, text="🔧 Preprocessing…")

    if asset_df.empty:
        st.error(f"❌ No data for **{ticker}**. Check ticker and date range.")
        prog.empty()
        return

    bm_df = _cached_preprocess(config["benchmark"], config["start_date"], config["end_date"])
    if bm_df.empty:
        st.warning(f"⚠️ Benchmark **{config['benchmark']}** data unavailable.")

    prog.progress(50, text="⚙️ Running backtest…")

    try:
        result = _run_backtest(
            df=asset_df, ticker=ticker,
            strategy_name=config["strategy"],
            params=config.get("strategy_params", {}),
            capital=config["initial_capital"],
            tx_cost=config["transaction_cost"],
            slippage=config["slippage"],
            vol_target=config.get("vol_target"),
        )
    except ValueError as exc:
        st.error(f"❌ Strategy error: {exc}")
        prog.empty()
        return

    equity_df  = result["equity_curve"]
    signals_df = result["signals"]

    if equity_df.empty:
        st.error("Backtest produced an empty equity curve.")
        prog.empty()
        return

    prog.progress(75, text="📊 Computing analytics…")

    daily_rets  = equity_df["Daily_Return"].dropna()
    perf        = compute_all_metrics(equity_df, risk_free_rate=config.get("risk_free_rate", 0.0))
    risk        = compute_risk_metrics(daily_rets)
    all_metrics = {**perf, **risk}

    bm_result = None
    if not bm_df.empty:
        bm_eng    = BacktestEngine(BuyAndHold(), config["initial_capital"], 0.0, 0.0)
        bm_result = bm_eng.run(bm_df, config["benchmark"])
        bm_daily  = bm_result["equity_curve"]["Daily_Return"].dropna()
        bm_m      = compute_benchmark_metrics(
            daily_rets, bm_daily, config.get("risk_free_rate", 0.0)
        )
        all_metrics = {**all_metrics, **bm_m}

    prog.progress(90, text="💾 Saving run…")

    try:
        save_run(
            ticker=ticker, benchmark=config["benchmark"],
            strategy_name=config["strategy"],
            parameters=config.get("strategy_params", {}),
            start_date=config["start_date"], end_date=config["end_date"],
            initial_capital=config["initial_capital"],
            transaction_cost=config["transaction_cost"],
            slippage=config["slippage"],
            metrics=all_metrics,
        )
    except Exception as exc:
        logger.warning("DB save failed: %s", exc)

    # Persist in session_state for cross-tab access
    st.session_state["af_result"]    = result
    st.session_state["af_metrics"]   = all_metrics
    st.session_state["af_equity"]    = equity_df
    st.session_state["af_signals"]   = signals_df
    st.session_state["af_asset_df"]  = asset_df
    st.session_state["af_bm_result"] = bm_result
    st.session_state["af_config"]    = config.copy()
    st.session_state["af_daily_rets"] = daily_rets

    prog.empty()


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — PORTFOLIO BUILDER
# ═══════════════════════════════════════════════════════════════════════════════

def _tab_portfolio_builder(config: dict) -> None:
    """Multi-asset weighted portfolio construction and analytics."""

    st.markdown("## 📊 Portfolio Builder")
    tickers = config.get("tickers", [config.get("ticker", "AAPL")])

    if len(tickers) < 2:
        st.info(
            "ℹ️ Select **2 or more assets** in the sidebar to build a portfolio. "
            "Use the **Selected Assets** multiselect to add more."
        )
        return

    st.markdown(
        f"Constructing a portfolio of **{len(tickers)} assets**: "
        f"{', '.join(f'**{t}**' for t in tickers)}"
    )

    # ── Weight assignment ─────────────────────────────────────────────────────
    st.markdown("### ⚖️ Allocation Weights")
    use_equal = st.checkbox("Equal-weight allocation", value=True)

    weights: dict[str, float] = {}
    if use_equal:
        weights = equal_weight_allocation(tickers)
        st.caption(f"Each asset: **{100/len(tickers):.2f}%**")
    else:
        st.caption("Assign weights (will be normalised to 100%)")
        cols = st.columns(min(len(tickers), 4))
        for i, t in enumerate(tickers):
            with cols[i % len(cols)]:
                weights[t] = st.number_input(
                    t, min_value=0.0, max_value=100.0,
                    value=round(100 / len(tickers), 1), step=1.0, key=f"w_{t}"
                ) / 100.0

        weights = normalize_weights(weights)
        valid, msg = validate_weights(weights)
        if not valid:
            st.warning(f"⚠️ {msg}")

    # Allocation summary
    alloc_df  = allocation_summary(weights)
    warn_msg  = dominant_asset_warning(weights)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.plotly_chart(allocation_pie_chart(weights), use_container_width=True)
    with col2:
        st.markdown("#### Allocation Table")
        st.dataframe(
            alloc_df.style.format({"Weight": "{:.4f}", "Weight_Pct": "{:.2f}%"}),
            use_container_width=True,
        )
        if warn_msg:
            st.warning(warn_msg)

    st.markdown("---")

    # ── Run portfolio backtest ────────────────────────────────────────────────
    run_port = st.button("🚀 Run Portfolio Backtest", type="primary",
                         use_container_width=True, key="run_portfolio")

    if not run_port and "af_port_result" not in st.session_state:
        return

    if run_port:
        with st.spinner("Loading market data and computing portfolio returns…"):
            asset_returns: dict[str, pd.Series] = {}
            for tkr in tickers:
                df = _cached_preprocess(tkr, config["start_date"], config["end_date"])
                if not df.empty and "Daily_Return" in df.columns:
                    asset_returns[tkr] = df["Daily_Return"].fillna(0.0)

        if len(asset_returns) < 2:
            st.error("❌ Could not load data for enough assets.")
            return

        rets_df     = pd.DataFrame(asset_returns).dropna()
        port_rets   = calculate_portfolio_returns(rets_df, weights)
        port_value  = cumulative_portfolio_value(port_rets, config["initial_capital"])

        port_equity = pd.DataFrame({
            "Portfolio_Value": port_value,
            "Daily_Return":    port_rets,
        })

        perf = compute_all_metrics(port_equity, config.get("risk_free_rate", 0.0))
        risk = compute_risk_metrics(port_rets)
        div  = diversification_report(rets_df, weights)

        st.session_state["af_port_result"]  = port_equity
        st.session_state["af_port_metrics"] = {**perf, **risk}
        st.session_state["af_port_div"]     = div
        st.session_state["af_port_rets_df"] = rets_df
        st.session_state["af_port_weights"] = weights

    # ── Display results ───────────────────────────────────────────────────────
    if "af_port_result" not in st.session_state:
        return

    port_equity = st.session_state["af_port_result"]
    port_mets   = st.session_state["af_port_metrics"]
    div_report  = st.session_state["af_port_div"]
    rets_df     = st.session_state["af_port_rets_df"]

    st.markdown("### 📈 Portfolio Performance")
    render_metric_cards(port_mets)

    st.plotly_chart(
        equity_curve_chart(
            port_equity, None,
            strategy_name="Weighted Portfolio",
            benchmark_name=config["benchmark"],
            initial_capital=config["initial_capital"],
        ),
        use_container_width=True,
    )

    # ── Diversification analytics ─────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🧩 Diversification Analytics")

    c1, c2, c3, c4 = st.columns(4)
    avg_corr = div_report.get("avg_correlation", float("nan"))
    dr       = div_report.get("diversification_ratio", float("nan"))
    hhi      = div_report.get("concentration_score", float("nan"))
    pv       = div_report.get("portfolio_vol_ann", float("nan"))

    c1.metric("Avg Pairwise Corr", f"{avg_corr:.3f}" if not np.isnan(avg_corr) else "N/A",
              help="Lower = more diversification benefit")
    c2.metric("Diversification Ratio", f"{dr:.3f}" if not np.isnan(dr) else "N/A",
              help=">1 means diversification reduces portfolio vol")
    c3.metric("Concentration (HHI)", f"{hhi:.4f}" if not np.isnan(hhi) else "N/A",
              help="Herfindahl index; 1/N = fully equal-weighted")
    c4.metric("Portfolio Vol (Ann.)", f"{pv*100:.2f}%" if not np.isnan(pv) else "N/A")

    corr = div_report.get("corr_matrix", pd.DataFrame())
    if not corr.empty:
        st.plotly_chart(
            correlation_heatmap(corr, "Asset Correlation Matrix"),
            use_container_width=True,
        )

    most = div_report.get("most_correlated", ("", "", float("nan")))
    least = div_report.get("least_correlated", ("", "", float("nan")))
    if most[0]:
        st.markdown(
            f"🔴 **Most correlated pair**: {most[0]} / {most[1]} "
            f"(ρ = {most[2]:.3f})"
            f"&emsp;&emsp;"
            f"🟢 **Least correlated pair**: {least[0]} / {least[1]} "
            f"(ρ = {least[2]:.3f})"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — STRATEGY COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════

def _tab_strategy_comparison(config: dict) -> None:
    """Rank all strategies on the same asset side-by-side."""

    st.markdown("## ⚔️ Strategy Comparison")
    ticker = config["ticker"]
    st.markdown(
        f"Compare strategies on **{ticker}** with identical cost assumptions."
    )

    # Strategy selection
    st.markdown("**Select strategies to compare:**")
    cols = st.columns(3)
    selected: list[str] = []
    for i, name in enumerate(ALL_STRATEGIES):
        with cols[i % 3]:
            if st.checkbox(name, value=True, key=f"cmp_{name}"):
                selected.append(name)

    sort_by = st.selectbox(
        "Rank by",
        ["Sharpe_Ratio", "CAGR", "Calmar_Ratio", "Max_Drawdown"],
        key="cmp_sort",
    )

    run_cmp = st.button(
        "🚀 Run Comparison", type="primary",
        use_container_width=True, key="run_comparison",
    )

    if not run_cmp and "af_cmp_results" not in st.session_state:
        st.caption("Select strategies above and click **Run Comparison**.")
        return

    if run_cmp:
        if len(selected) < 2:
            st.warning("Select at least 2 strategies to compare.")
            return

        asset_df = _cached_preprocess(ticker, config["start_date"], config["end_date"])
        if asset_df.empty:
            st.error(f"❌ No data for {ticker}.")
            return

        bm_df = _cached_preprocess(config["benchmark"], config["start_date"], config["end_date"])
        bm_daily = None
        if not bm_df.empty:
            bm_eng   = BacktestEngine(BuyAndHold(), config["initial_capital"], 0.0, 0.0)
            bm_res   = bm_eng.run(bm_df, config["benchmark"])
            bm_daily = bm_res["equity_curve"]["Daily_Return"].dropna()

        prog = st.progress(0, text="Running strategies…")
        cmp_metrics: dict[str, dict] = {}
        cmp_equity:  dict[str, pd.DataFrame] = {}

        for i, name in enumerate(selected):
            prog.progress(int((i + 1) / len(selected) * 100), text=f"Running {name}…")
            try:
                res  = _run_backtest(
                    df=asset_df, ticker=ticker, strategy_name=name,
                    params=DEFAULT_PARAMS[name],
                    capital=config["initial_capital"],
                    tx_cost=config["transaction_cost"],
                    slippage=config["slippage"],
                )
                eq   = res["equity_curve"]
                rets = eq["Daily_Return"].dropna()
                perf = compute_all_metrics(eq, config.get("risk_free_rate", 0.0))
                risk = compute_risk_metrics(rets)
                bm_m = (
                    compute_benchmark_metrics(rets, bm_daily, config.get("risk_free_rate", 0.0))
                    if bm_daily is not None else {}
                )
                cmp_metrics[name] = {**perf, **risk, **bm_m}
                cmp_equity[name]  = eq
            except Exception as exc:
                logger.warning("Comparison %s failed: %s", name, exc)

        prog.empty()
        st.session_state["af_cmp_results"] = cmp_metrics
        st.session_state["af_cmp_equity"]  = cmp_equity
        st.session_state["af_cmp_sort"]    = sort_by

    # Display
    if "af_cmp_results" not in st.session_state:
        return

    cmp_metrics = st.session_state["af_cmp_results"]
    cmp_equity  = st.session_state["af_cmp_equity"]
    sort_key    = st.session_state.get("af_cmp_sort", sort_by)

    st.markdown("### 🏆 Strategy Rankings")
    ranked = rank_strategies(cmp_metrics, by=sort_key)
    fmt    = format_comparison_table(ranked)
    st.dataframe(fmt, use_container_width=True)

    winner = best_strategy(cmp_metrics, by=sort_key)
    if winner:
        st.success(f"🥇 Best strategy by **{sort_key}**: **{winner}**")

    st.markdown("### 📈 Equity Curves")
    st.plotly_chart(
        strategy_comparison_chart(cmp_equity, config["initial_capital"]),
        use_container_width=True,
    )

    st.markdown("### 📉 Drawdown Comparison")
    st.plotly_chart(
        drawdown_comparison_chart(cmp_equity, "Strategy Drawdown Comparison"),
        use_container_width=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — RISK ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════════

def _tab_risk_analytics(config: dict) -> None:
    """Factor exposure, stress testing, and forward testing analytics."""

    st.markdown("## 🧮 Risk Analytics")

    has_results = "af_daily_rets" in st.session_state

    if not has_results:
        st.info(
            "Run a backtest in **🔬 Research Lab** first to enable "
            "factor exposure and stress testing analysis."
        )
        st.markdown("---")
        st.markdown("**📡 Forward Testing** is available without a prior backtest:")
        _render_forward_testing_section(None, config)
        return

    daily_rets = st.session_state["af_daily_rets"]
    snap       = st.session_state["af_config"]

    # ── Factor Analysis ───────────────────────────────────────────────────────
    with st.expander("📡 Factor Exposure Analysis (ETF Proxies)", expanded=True):
        st.markdown(
            "> *Simplified educational factor analysis using liquid ETFs as proxies.*  "
            "> *Results are approximations — not a formal Fama-French model.*"
        )

        sel_factors = st.multiselect(
            "Select factor proxies",
            options=list(FACTOR_PROXIES.keys()),
            default=list(FACTOR_PROXIES.keys()),
            key="factor_select",
        )

        run_fa = st.button("Run Factor Analysis", key="run_factor")
        if run_fa and sel_factors:
            with st.spinner("Fetching factor proxy returns…"):
                proxy_rets = _cached_factor_returns(snap["start_date"], snap["end_date"])
                proxy_sel  = {k: v for k, v in proxy_rets.items() if k in sel_factors}

            if not proxy_sel:
                st.error("Could not fetch factor proxy data. Check your internet connection.")
            else:
                fa_result = run_factor_regression(
                    daily_rets, proxy_sel, snap.get("risk_free_rate", 0.0)
                )
                st.session_state["af_fa_result"] = fa_result

        if "af_fa_result" in st.session_state:
            fa = st.session_state["af_fa_result"]
            if fa:
                c1, c2, c3 = st.columns(3)
                c1.metric("Alpha (Ann.)", f"{fa.get('alpha_ann', 0)*100:+.3f}%")
                c2.metric("R²", f"{fa.get('r_squared', 0):.4f}")
                c3.metric("Observations", str(fa.get("n_obs", "—")))

                loadings = fa.get("factor_loadings", {})
                if loadings:
                    st.plotly_chart(
                        factor_exposure_chart(loadings, "Factor Loadings"),
                        use_container_width=True,
                    )

                    # Regression table
                    rows = []
                    for name in ["alpha"] + list(loadings.keys()):
                        rows.append({
                            "Factor":   name,
                            "Loading":  round(fa["t_stats"].get(name, float("nan")), 4),
                            "t-stat":   round(fa["t_stats"].get(name, float("nan")), 3),
                            "p-value":  round(fa["p_values"].get(name, float("nan")), 4),
                            "Sig.":     "***" if fa["p_values"].get(name, 1) < 0.01
                                        else "**" if fa["p_values"].get(name, 1) < 0.05
                                        else "*"  if fa["p_values"].get(name, 1) < 0.10
                                        else "",
                        })
                    st.dataframe(pd.DataFrame(rows).set_index("Factor"), use_container_width=True)

    # ── Stress Testing ────────────────────────────────────────────────────────
    with st.expander("⚡ Stress Testing", expanded=True):
        st.markdown(
            "> *Estimated portfolio impact under historical and hypothetical stress scenarios.*  "
            "> *Losses are approximated using strategy volatility as a beta proxy.*"
        )

        c_left, c_right = st.columns([2, 1])
        with c_right:
            st.markdown("**Custom Shock**")
            custom_shock = st.slider(
                "Market shock (%)", -60, -1, -20, key="stress_shock"
            ) / 100
            custom_dur = st.number_input(
                "Duration (days)", 1, 504, 21, key="stress_dur"
            )
            cust_r = custom_scenario(
                daily_rets, config["initial_capital"],
                custom_shock, int(custom_dur),
            )
            st.metric(
                "Custom scenario impact",
                f"{cust_r['estimated_loss_pct']*100:.2f}%",
                delta=f"${cust_r['estimated_loss_usd']:,.0f}",
                delta_color="inverse",
            )

        with c_left:
            stress_df = run_stress_scenarios(
                daily_rets, config["initial_capital"]
            )
            if not stress_df.empty:
                worst = stress_df.iloc[0]
                st.error(
                    f"🔴 **Worst-case scenario**: *{worst['Scenario']}* — "
                    f"estimated loss **{worst['Estimated_Loss_Pct']*100:.1f}%** "
                    f"(~${abs(worst['Estimated_Loss_USD']):,.0f})"
                )

        if not stress_df.empty:
            st.plotly_chart(
                stress_test_chart(stress_df, config["initial_capital"]),
                use_container_width=True,
            )
            disp = stress_df[["Severity_Rank", "Scenario", "Description",
                              "Market_Shock_Pct", "Duration_Days",
                              "Estimated_Loss_Pct", "Estimated_Loss_USD",
                              "Ending_Value"]].copy()
            for col in ["Market_Shock_Pct", "Estimated_Loss_Pct"]:
                disp[col] = disp[col].apply(lambda v: f"{v*100:.2f}%")
            for col in ["Estimated_Loss_USD", "Ending_Value"]:
                disp[col] = disp[col].apply(lambda v: f"${v:,.0f}")
            st.dataframe(disp.set_index("Severity_Rank"), use_container_width=True)

    # ── Forward Testing ───────────────────────────────────────────────────────
    with st.expander("🔭 Forward Testing (Out-of-Sample)", expanded=False):
        _render_forward_testing_section(
            st.session_state.get("af_asset_df"),
            st.session_state["af_config"],
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — TEAR SHEET
# ═══════════════════════════════════════════════════════════════════════════════

def _tab_tear_sheet(config: dict) -> None:
    """Generate and download a professional HTML strategy tear sheet."""

    st.markdown("## 📋 Strategy Tear Sheet")

    if "af_result" not in st.session_state:
        st.info(
            "ℹ️ Run a backtest in **🔬 Research Lab** first to generate a tear sheet."
        )
        return

    metrics   = st.session_state["af_metrics"]
    equity_df = st.session_state["af_equity"]
    snap      = st.session_state["af_config"]

    st.markdown(
        f"**Strategy:** {snap['strategy']} &nbsp;·&nbsp; "
        f"**Asset:** {snap['ticker']} &nbsp;·&nbsp; "
        f"**Period:** {snap['start_date']} → {snap['end_date']}"
    )

    st.markdown(
        "> ⚠️ *This tear sheet is for research and educational purposes only. "
        "Past performance does not guarantee future results. "
        "This is not investment advice.*"
    )

    gen_col, csv_col, html_col = st.columns(3)

    with gen_col:
        generate = st.button(
            "📋 Generate Tear Sheet", type="primary",
            use_container_width=True, key="gen_ts",
        )

    if generate or "af_tear_sheet_html" in st.session_state:
        if generate:
            with st.spinner("Building tear sheet…"):
                html = generate_html_tear_sheet(metrics, equity_df, snap)
                csv  = generate_csv_report(metrics, snap)
                st.session_state["af_tear_sheet_html"] = html
                st.session_state["af_tear_sheet_csv"]  = csv

        html = st.session_state["af_tear_sheet_html"]
        csv  = st.session_state["af_tear_sheet_csv"]

        with csv_col:
            st.download_button(
                "⬇️ Download CSV",
                data=csv,
                file_name=f"alphaforge_{snap['strategy'].replace(' ', '_')}_{snap['ticker']}.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with html_col:
            st.download_button(
                "⬇️ Download HTML Report",
                data=html,
                file_name=f"alphaforge_{snap['strategy'].replace(' ', '_')}_{snap['ticker']}.html",
                mime="text/html",
                use_container_width=True,
            )

        st.markdown("---")
        st.markdown("### 👁️ Report Preview")
        st.components.v1.html(html, height=860, scrolling=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6 — SAVED RUNS
# ═══════════════════════════════════════════════════════════════════════════════

def _tab_saved_runs(config: dict) -> None:
    """Save, load, rename, and delete strategy configurations."""

    st.markdown("## 💾 Saved Configurations")

    # ── Save current config ───────────────────────────────────────────────────
    with st.expander("💾 Save Current Configuration", expanded=True):
        cfg_name = st.text_input(
            "Configuration name",
            value=f"{config.get('strategy', 'Strategy')} — {config.get('ticker', '?')} "
                  f"({config.get('start_date', '')[:4]}–{config.get('end_date', '')[:4]})",
            key="save_cfg_name",
        )
        if st.button("Save", key="save_cfg_btn", type="primary"):
            rid = save_config(cfg_name, config)
            st.success(f"✅ Saved as **{cfg_name}** (ID {rid})")

    # ── Load saved config ─────────────────────────────────────────────────────
    with st.expander("📂 Load Saved Configuration", expanded=False):
        cfgs_df = list_configs()
        if cfgs_df.empty:
            st.info("No saved configurations yet.")
        else:
            options = {
                f"{row['name']} (ID {row['id']})": row["id"]
                for _, row in cfgs_df.iterrows()
            }
            chosen = st.selectbox("Select configuration", list(options.keys()), key="load_sel")
            if st.button("Load", key="load_btn"):
                loaded = load_config(options[chosen])
                if loaded:
                    st.success("✅ Configuration loaded — settings shown below.")
                    st.json(
                        {k: v for k, v in loaded.items()
                         if k not in ("strategy_params",)}
                    )
                    if "strategy_params" in loaded:
                        st.json(loaded["strategy_params"])
                    st.info(
                        "To apply this configuration, manually set the sidebar "
                        "controls to the values shown above."
                    )
                else:
                    st.error("Failed to load configuration.")

    # ── Rename / delete ───────────────────────────────────────────────────────
    with st.expander("✏️ Rename / 🗑️ Delete Configuration", expanded=False):
        cfgs_df = list_configs()
        if cfgs_df.empty:
            st.info("No saved configurations.")
        else:
            options = {
                f"{row['name']} (ID {row['id']})": row["id"]
                for _, row in cfgs_df.iterrows()
            }
            chosen_mgmt = st.selectbox(
                "Select configuration to manage", list(options.keys()), key="mgmt_sel"
            )
            cfg_id = options[chosen_mgmt]

            r_col, d_col = st.columns(2)
            with r_col:
                new_name = st.text_input("New name", key="rename_input")
                if st.button("Rename", key="rename_btn") and new_name:
                    rename_config(cfg_id, new_name)
                    st.success(f"✅ Renamed to **{new_name}**")
                    st.rerun()
            with d_col:
                st.markdown("&nbsp;")
                if st.button("🗑️ Delete", key="delete_btn", type="secondary"):
                    delete_config(cfg_id)
                    st.warning(f"Deleted configuration ID {cfg_id}.")
                    st.rerun()

    # ── Run history ───────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🗂️ Backtest Run History")
    render_run_history(load_runs())


# ═══════════════════════════════════════════════════════════════════════════════
# CHART GALLERY (Research Lab helper)
# ═══════════════════════════════════════════════════════════════════════════════

def _render_charts(
    equity_df, signals_df, asset_df, daily_rets,
    bm_result, config, all_metrics,
) -> None:
    st.markdown("### 📈 Charts")

    bm_equity = bm_result["equity_curve"] if bm_result else None
    bm_rets   = (
        bm_result["equity_curve"]["Daily_Return"].dropna() if bm_result else None
    )

    tabs = st.tabs([
        "Equity Curve", "Signal Overlay", "Monthly Returns",
        "Annual Returns", "Drawdown", "Underwater", "Returns Dist.",
        "Rolling Vol", "Rolling Sharpe", "Beta Scatter", "Benchmark",
    ])

    with tabs[0]:
        st.plotly_chart(
            equity_curve_chart(
                equity_df, bm_equity,
                strategy_name=config["strategy"],
                benchmark_name=config["benchmark"],
                initial_capital=config["initial_capital"],
            ), use_container_width=True,
        )

    with tabs[1]:
        st.plotly_chart(
            signal_overlay_chart(asset_df, signals_df,
                                  strategy_name=config["strategy"],
                                  ticker=config["ticker"]),
            use_container_width=True,
        )
        st.caption(
            "↑ Cyan = entries · Orange = exits · Blue shading = in-position"
        )

    with tabs[2]:
        st.plotly_chart(
            monthly_returns_heatmap(daily_rets, strategy_name=config["strategy"]),
            use_container_width=True,
        )
        with st.expander("📋 Monthly Returns Table", expanded=False):
            mr_tbl = monthly_returns_table(daily_rets)
            if not mr_tbl.empty:
                st.dataframe(
                    (mr_tbl * 100).round(2).style.background_gradient(
                        cmap="RdYlGn", axis=None, vmin=-10, vmax=10
                    ).format("{:.2f}%"),
                    use_container_width=True,
                )

    with tabs[3]:
        st.plotly_chart(
            annual_returns_chart(daily_rets, bm_rets,
                                  strategy_name=config["strategy"],
                                  benchmark_name=config["benchmark"]),
            use_container_width=True,
        )

    with tabs[4]:
        st.plotly_chart(
            drawdown_chart(equity_df, strategy_name=config["strategy"],
                            benchmark_df=bm_equity,
                            benchmark_name=config["benchmark"]),
            use_container_width=True,
        )

    with tabs[5]:
        st.plotly_chart(
            underwater_chart(equity_df, strategy_name=config["strategy"]),
            use_container_width=True,
        )

    with tabs[6]:
        st.plotly_chart(
            returns_distribution_chart(daily_rets, bm_rets,
                                        strategy_name=config["strategy"]),
            use_container_width=True,
        )

    with tabs[7]:
        st.plotly_chart(
            rolling_volatility_chart(equity_df, strategy_name=config["strategy"]),
            use_container_width=True,
        )

    with tabs[8]:
        st.plotly_chart(
            rolling_sharpe_chart(daily_rets, bm_rets,
                                  strategy_name=config["strategy"],
                                  benchmark_name=config["benchmark"]),
            use_container_width=True,
        )

    with tabs[9]:
        if bm_rets is not None:
            st.plotly_chart(
                returns_scatter_chart(daily_rets, bm_rets,
                                       strategy_name=config["strategy"],
                                       benchmark_name=config["benchmark"]),
                use_container_width=True,
            )
        else:
            st.info("Benchmark data required for this chart.")

    with tabs[10]:
        if bm_rets is not None:
            st.plotly_chart(
                cumulative_returns_comparison(daily_rets, bm_rets,
                                              strategy_name=config["strategy"],
                                              benchmark_name=config["benchmark"]),
                use_container_width=True,
            )
        else:
            st.info("Benchmark data unavailable.")


# ── Forward testing shared section ────────────────────────────────────────────

def _render_forward_testing_section(asset_df, config: dict) -> None:
    """Shared forward-testing UI, used in Research Lab and Risk Analytics."""
    st.markdown(
        "Evaluate strategy parameters on a held-out test period to check "
        "for out-of-sample robustness."
    )

    train_pct = st.slider(
        "Training fraction", 0.50, 0.90, 0.70, 0.05,
        format="%.0f%%",
        key="ft_train_pct",
        help="Portion of data used for in-sample (IS) evaluation. Remainder is OOS.",
    )

    if asset_df is None or asset_df.empty:
        ticker   = config["ticker"]
        asset_df = _cached_preprocess(ticker, config["start_date"], config["end_date"])

    if asset_df is None or asset_df.empty:
        st.error("No asset data available. Run a backtest first.")
        return

    run_ft = st.button("Run Forward Test", key="run_ft_btn")
    if not run_ft:
        return

    strategy = config.get("strategy", "Buy & Hold")
    params   = config.get("strategy_params", {})

    with st.spinner("Running forward test…"):
        try:
            ft_result = run_forward_test(
                df=asset_df,
                strategy_name=strategy,
                params=params,
                capital=config["initial_capital"],
                transaction_cost=config["transaction_cost"],
                slippage=config["slippage"],
                train_pct=train_pct,
                risk_free_rate=config.get("risk_free_rate", 0.0),
                ticker=config["ticker"],
            )
        except ValueError as exc:
            st.error(f"Forward test failed: {exc}")
            return

    # Display
    c1, c2, c3 = st.columns(3)
    c1.metric("IS Sharpe",  f"{ft_result.is_sharpe:.3f}")
    c2.metric("OOS Sharpe", f"{ft_result.oos_sharpe:.3f}",
              delta=f"{(ft_result.oos_sharpe - ft_result.is_sharpe):+.3f}",
              delta_color="normal")
    c3.metric("Degradation Ratio",
              f"{ft_result.degradation_ratio:.3f}" if not np.isnan(ft_result.degradation_ratio) else "N/A",
              help="OOS Sharpe / IS Sharpe. >0.5 is acceptable.")

    if ft_result.overfitting_flag:
        st.warning(
            "⚠️ **Overfitting Risk**: OOS Sharpe is less than 50% of IS Sharpe. "
            "The strategy may be overfit to historical data."
        )
    else:
        st.success("✅ OOS performance is within acceptable range of IS performance.")

    st.dataframe(
        pd.DataFrame([ft_result.to_dict()]).T.rename(columns={0: "Value"}),
        use_container_width=True,
    )


# ── Optimisation helper ────────────────────────────────────────────────────────

def _render_optimisation(asset_df: pd.DataFrame, config: dict) -> None:
    st.markdown("---")
    if config.get("strategy") == "Moving Average Crossover":
        with st.spinner("⚗️ Optimising parameters…"):
            opt_df = grid_search_ma(
                asset_df,
                short_windows=list(range(5, 65, 5)),
                long_windows=list(range(20, 160, 10)),
                initial_capital=config["initial_capital"],
                transaction_cost=config["transaction_cost"],
                slippage=config["slippage"],
            )
        render_optimization_results(opt_df, config["strategy"])
        if not opt_df.empty:
            st.plotly_chart(
                optimization_heatmap(opt_df, "Sharpe_Ratio"),
                use_container_width=True,
            )
        st.markdown("---")
        with st.spinner("🔄 Walk-forward validation…"):
            wf_df = walk_forward_ma(
                asset_df,
                short_windows=list(range(5, 65, 5)),
                long_windows=list(range(20, 160, 10)),
                initial_capital=config["initial_capital"],
                transaction_cost=config["transaction_cost"],
                slippage=config["slippage"],
            )
        render_walk_forward(wf_df, summarise_walk_forward(wf_df))

    elif config.get("strategy") == "Mean Reversion":
        with st.spinner("⚗️ Optimising parameters…"):
            opt_df = grid_search_mr(
                asset_df,
                windows=[10, 15, 20, 30, 40, 60],
                entry_thresholds=[1.0, 1.5, 2.0, 2.5, 3.0],
                initial_capital=config["initial_capital"],
                transaction_cost=config["transaction_cost"],
                slippage=config["slippage"],
            )
        render_optimization_results(opt_df, config["strategy"])


# ── Landing page ───────────────────────────────────────────────────────────────

def _render_landing() -> None:
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(
            """
## Welcome to AlphaForge

**AlphaForge** is a modular quantitative trading research platform designed for
rigorous systematic strategy research — not stock price prediction.

Configure your backtest in the sidebar and click **🚀 Run Backtest** to begin.

---
### 🧩 Platform Capabilities

| Module | Feature |
|---|---|
| **Research Lab** | Single-asset backtesting · 11 interactive charts · Trade log |
| **Portfolio Builder** | Multi-asset allocation · Diversification analytics |
| **Strategy Comparison** | Rank 6 strategies · Overlaid equity curves |
| **Risk Analytics** | ETF factor analysis · Stress testing · Forward testing |
| **Tear Sheet** | Professional HTML/CSV report · Download |
| **Saved Runs** | SQLite config storage · Load · Rename · Delete |
| **Data** | yfinance OHLCV · Parquet cache · 400+ asset universe |
| **Strategies** | MA Crossover · Mean Reversion · Momentum · RSI · Bollinger · Buy & Hold |
| **Risk** | VaR · ES · Ulcer · Omega · Serenity · Tail Ratio |
| **Simulation** | Monte Carlo (Gaussian, Block Bootstrap, GBM, Stress Scenarios) |
""")
    with col2:
        st.markdown(
            """
### Quick start
1. Search an asset in the sidebar
2. Set your date range
3. Choose a strategy
4. Click **🚀 Run Backtest**
5. Explore the tabs

---
### Benchmark
Default: **SPY** (S&P 500)

Changeable via the Benchmark selector in the sidebar.
"""
        )


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
else:
    main()
