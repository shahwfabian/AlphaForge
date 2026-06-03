"""
Plotly chart builders for the AlphaForge Streamlit dashboard.

All functions return a plotly.graph_objects.Figure that can be rendered
with st.plotly_chart(fig, use_container_width=True).
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Brand colours ─────────────────────────────────────────────────────────────
STRATEGY_COLOR  = "#00D4FF"   # Cyan
BENCHMARK_COLOR = "#FF6B35"   # Orange
NEGATIVE_COLOR  = "#FF4444"   # Red
POSITIVE_COLOR  = "#00CC66"   # Green
GOLD_COLOR      = "#FFD700"   # Gold (rolling metrics)
PURPLE_COLOR    = "#BB86FC"   # Purple (secondary)
BACKGROUND      = "#0E1117"
SECONDARY_BG    = "#151B2E"
GRID_COLOR      = "#1E2A3A"
TEXT_COLOR      = "#E0E0E0"

LAYOUT_DEFAULTS = dict(
    paper_bgcolor=BACKGROUND,
    plot_bgcolor=SECONDARY_BG,
    font=dict(color=TEXT_COLOR, family="Inter, sans-serif", size=12),
    margin=dict(l=55, r=30, t=55, b=50),
    legend=dict(bgcolor="rgba(0,0,0,0.4)", bordercolor=GRID_COLOR,
                borderwidth=1, font=dict(size=11)),
    xaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR,
               showline=True, linecolor=GRID_COLOR),
    yaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR,
               showline=True, linecolor=GRID_COLOR),
)

# Strategy-colour palette for comparison mode
PALETTE = [STRATEGY_COLOR, BENCHMARK_COLOR, POSITIVE_COLOR,
           GOLD_COLOR, PURPLE_COLOR, "#FF69B4", "#20B2AA"]


# ── 1. Equity curve ───────────────────────────────────────────────────────────

def equity_curve_chart(
    equity_df: pd.DataFrame,
    benchmark_df: pd.DataFrame | None = None,
    strategy_name: str = "Strategy",
    benchmark_name: str = "Benchmark",
    initial_capital: float = 100_000.0,
) -> go.Figure:
    """Line chart of portfolio value over time vs benchmark."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=equity_df.index, y=equity_df["Portfolio_Value"],
        name=strategy_name,
        line=dict(color=STRATEGY_COLOR, width=2.5),
        hovertemplate="%{x|%Y-%m-%d}<br><b>$%{y:,.0f}</b><extra></extra>",
    ))

    if benchmark_df is not None and "Portfolio_Value" in benchmark_df.columns:
        bm = benchmark_df["Portfolio_Value"]
        scale = initial_capital / bm.iloc[0] if bm.iloc[0] != 0 else 1.0
        fig.add_trace(go.Scatter(
            x=benchmark_df.index, y=bm * scale,
            name=benchmark_name,
            line=dict(color=BENCHMARK_COLOR, width=2, dash="dot"),
            hovertemplate="%{x|%Y-%m-%d}<br><b>$%{y:,.0f}</b><extra></extra>",
        ))

    # Initial capital reference line
    fig.add_hline(y=initial_capital, line_color=GRID_COLOR,
                  line_dash="dot", line_width=1)

    fig.update_layout(
        title=dict(text="Portfolio Equity Curve", font=dict(size=16)),
        yaxis_title="Portfolio Value (USD)",
        xaxis_title="Date",
        hovermode="x unified",
        **LAYOUT_DEFAULTS,
    )
    return fig


# ── 2. Drawdown ───────────────────────────────────────────────────────────────

def drawdown_chart(
    equity_df: pd.DataFrame,
    strategy_name: str = "Strategy",
    benchmark_df: pd.DataFrame | None = None,
    benchmark_name: str = "Benchmark",
) -> go.Figure:
    """Area chart showing drawdown series (percentage from peak)."""
    pv = equity_df["Portfolio_Value"]
    dd_strat = (pv - pv.cummax()) / pv.cummax() * 100

    fig = go.Figure()

    if benchmark_df is not None and "Portfolio_Value" in benchmark_df.columns:
        bm_pv = benchmark_df["Portfolio_Value"]
        dd_bm = (bm_pv - bm_pv.cummax()) / bm_pv.cummax() * 100
        fig.add_trace(go.Scatter(
            x=dd_bm.index, y=dd_bm.values,
            name=benchmark_name,
            line=dict(color=BENCHMARK_COLOR, width=1.5, dash="dot"),
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:.2f}%<extra></extra>",
        ))

    fig.add_trace(go.Scatter(
        x=dd_strat.index, y=dd_strat.values,
        name=strategy_name,
        fill="tozeroy",
        line=dict(color=NEGATIVE_COLOR, width=1.5),
        fillcolor="rgba(255,68,68,0.18)",
        hovertemplate="%{x|%Y-%m-%d}<br><b>%{y:.2f}%</b><extra></extra>",
    ))

    fig.update_layout(
        title=dict(text=f"{strategy_name} — Drawdown", font=dict(size=16)),
        yaxis_title="Drawdown (%)",
        xaxis_title="Date",
        hovermode="x unified",
        **LAYOUT_DEFAULTS,
    )
    return fig


# ── 3. Returns distribution ───────────────────────────────────────────────────

def returns_distribution_chart(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series | None = None,
    strategy_name: str = "Strategy",
) -> go.Figure:
    """Histogram of daily returns with optional benchmark overlay."""
    fig = go.Figure()
    clean_s = strategy_returns.dropna() * 100

    fig.add_trace(go.Histogram(
        x=clean_s, name=strategy_name, nbinsx=80,
        marker_color=STRATEGY_COLOR, opacity=0.75,
        hovertemplate="Return: %{x:.2f}%<br>Count: %{y}<extra></extra>",
    ))

    if benchmark_returns is not None:
        clean_b = benchmark_returns.dropna() * 100
        fig.add_trace(go.Histogram(
            x=clean_b, name="Benchmark", nbinsx=80,
            marker_color=BENCHMARK_COLOR, opacity=0.55,
            hovertemplate="Return: %{x:.2f}%<br>Count: %{y}<extra></extra>",
        ))

    # Normal distribution overlay for comparison
    mu, sigma = clean_s.mean(), clean_s.std()
    x_range = np.linspace(clean_s.min(), clean_s.max(), 200)
    from scipy.stats import norm
    normal_y = norm.pdf(x_range, mu, sigma)
    # Scale to histogram counts
    bin_width = (clean_s.max() - clean_s.min()) / 80
    normal_y_scaled = normal_y * len(clean_s) * bin_width
    fig.add_trace(go.Scatter(
        x=x_range, y=normal_y_scaled,
        name="Normal Fit",
        line=dict(color=GOLD_COLOR, width=2, dash="dash"),
        hoverinfo="skip",
    ))

    fig.add_vline(x=0, line_color=TEXT_COLOR, line_dash="dot", line_width=1)

    fig.update_layout(
        title=dict(text="Daily Returns Distribution", font=dict(size=16)),
        xaxis_title="Daily Return (%)",
        yaxis_title="Frequency",
        barmode="overlay",
        **LAYOUT_DEFAULTS,
    )
    return fig


# ── 4. Rolling volatility ─────────────────────────────────────────────────────

def rolling_volatility_chart(
    equity_df: pd.DataFrame,
    window: int = 21,
    strategy_name: str = "Strategy",
) -> go.Figure:
    """Line chart of rolling annualised volatility."""
    rets = (
        equity_df["Daily_Return"].dropna()
        if "Daily_Return" in equity_df.columns
        else equity_df["Portfolio_Value"].pct_change().dropna()
    )
    roll_vol = rets.rolling(window=window).std() * np.sqrt(252) * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=roll_vol.index, y=roll_vol.values,
        name=f"{window}d Rolling Vol",
        line=dict(color=GOLD_COLOR, width=2),
        fill="tozeroy", fillcolor="rgba(255,215,0,0.08)",
        hovertemplate="%{x|%Y-%m-%d}<br>Vol: <b>%{y:.1f}%</b><extra></extra>",
    ))

    fig.update_layout(
        title=dict(text=f"Rolling {window}-Day Annualised Volatility", font=dict(size=16)),
        yaxis_title="Volatility (% ann.)",
        xaxis_title="Date",
        **LAYOUT_DEFAULTS,
    )
    return fig


# ── 5. Monte Carlo fan chart ──────────────────────────────────────────────────

def monte_carlo_chart(mc_results: dict) -> go.Figure:
    """Fan chart showing Monte Carlo simulation percentile bands."""
    from ..analytics.monte_carlo import mc_percentile_df
    pct_df = mc_percentile_df(mc_results)
    if pct_df.empty:
        return go.Figure()

    initial = mc_results.get("initial_value", 100_000)
    days = pct_df["Day"].values

    fig = go.Figure()

    for lo, hi, opacity, label in [
        ("P5",  "P95", 0.10, "5th–95th"),
        ("P25", "P75", 0.20, "25th–75th"),
    ]:
        fig.add_trace(go.Scatter(
            x=list(days) + list(days[::-1]),
            y=list(pct_df[hi]) + list(pct_df[lo][::-1]),
            fill="toself",
            fillcolor=f"rgba(0,212,255,{opacity})",
            line=dict(color="rgba(255,255,255,0)"),
            name=label, hoverinfo="skip",
        ))

    fig.add_trace(go.Scatter(
        x=days, y=pct_df["P50"],
        name="Median", line=dict(color=STRATEGY_COLOR, width=2.5),
        hovertemplate="Day %{x}<br><b>$%{y:,.0f}</b><extra></extra>",
    ))

    fig.add_hline(y=initial, line_color=TEXT_COLOR, line_dash="dot",
                  annotation_text="Initial Capital", annotation_position="top right")

    fig.update_layout(
        title=dict(
            text=f"Monte Carlo — {mc_results.get('n_simulations', 500)} Paths × "
                 f"{mc_results.get('n_days', 252)} Days",
            font=dict(size=16),
        ),
        xaxis_title="Trading Days Forward",
        yaxis_title="Portfolio Value (USD)",
        **LAYOUT_DEFAULTS,
    )
    return fig


# ── 6. Cumulative returns comparison ─────────────────────────────────────────

def cumulative_returns_comparison(
    strategy_rets: pd.Series,
    benchmark_rets: pd.Series,
    strategy_name: str = "Strategy",
    benchmark_name: str = "Benchmark",
) -> go.Figure:
    """Cumulative return comparison chart, both starting at 0 %."""
    s_cum = ((1 + strategy_rets).cumprod() - 1) * 100
    b_cum = ((1 + benchmark_rets).cumprod() - 1) * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=s_cum.index, y=s_cum.values, name=strategy_name,
        line=dict(color=STRATEGY_COLOR, width=2.5),
        hovertemplate="%{x|%Y-%m-%d}<br>%{y:.1f}%<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=b_cum.index, y=b_cum.values, name=benchmark_name,
        line=dict(color=BENCHMARK_COLOR, width=2, dash="dot"),
        hovertemplate="%{x|%Y-%m-%d}<br>%{y:.1f}%<extra></extra>",
    ))
    fig.add_hline(y=0, line_color=GRID_COLOR, line_width=1)

    fig.update_layout(
        title=dict(text="Cumulative Return vs Benchmark", font=dict(size=16)),
        yaxis_title="Cumulative Return (%)",
        xaxis_title="Date",
        hovermode="x unified",
        **LAYOUT_DEFAULTS,
    )
    return fig


# ── 7. Optimisation heatmap ───────────────────────────────────────────────────

def optimization_heatmap(opt_df: pd.DataFrame, metric: str = "Sharpe_Ratio") -> go.Figure:
    """Heatmap of MA parameter combinations vs a selected metric."""
    if opt_df.empty or "short_window" not in opt_df.columns:
        return go.Figure()

    pivot = opt_df.pivot_table(
        index="long_window", columns="short_window", values=metric, aggfunc="mean"
    )

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=[str(c) for c in pivot.columns],
        y=[str(r) for r in pivot.index],
        colorscale="Viridis",
        colorbar=dict(title=metric.replace("_", " ")),
        hovertemplate=(
            "Short: %{x}<br>Long: %{y}<br>"
            + metric.replace("_", " ") + ": %{z:.3f}<extra></extra>"
        ),
    ))
    fig.update_layout(
        title=dict(
            text=f"Parameter Optimisation: {metric.replace('_', ' ')}",
            font=dict(size=16),
        ),
        xaxis_title="Short Window (days)",
        yaxis_title="Long Window (days)",
        **LAYOUT_DEFAULTS,
    )
    return fig


# ── 8. Signal overlay on price chart ─────────────────────────────────────────

def signal_overlay_chart(
    price_df: pd.DataFrame,
    signals_df: pd.DataFrame,
    strategy_name: str = "Strategy",
    ticker: str = "",
) -> go.Figure:
    """
    Price line chart with buy/sell entry markers overlaid.

    Visually shows every moment the strategy entered or exited a position.
    BUY markers = cyan upward triangles below the price.
    SELL markers = orange downward triangles above the price.
    Active position shaded in background.
    """
    close = price_df["Close"]
    position = signals_df["Position"] if "Position" in signals_df.columns else pd.Series(0, index=price_df.index)
    signal = signals_df["Signal"] if "Signal" in signals_df.columns else pd.Series(0, index=price_df.index)

    # Detect transitions
    pos_shifted = position.shift(1).fillna(0)
    buys  = price_df.index[(position == 1) & (pos_shifted == 0)]
    sells = price_df.index[(position == 0) & (pos_shifted == 1)]

    fig = go.Figure()

    # Shaded background when in position
    in_pos = (position == 1)
    # Build shaded regions
    start_shade = None
    for i, (ts, in_p) in enumerate(in_pos.items()):
        if in_p and start_shade is None:
            start_shade = ts
        elif not in_p and start_shade is not None:
            fig.add_vrect(
                x0=start_shade, x1=ts,
                fillcolor="rgba(0,212,255,0.06)",
                layer="below", line_width=0,
            )
            start_shade = None
    if start_shade is not None:
        fig.add_vrect(
            x0=start_shade, x1=close.index[-1],
            fillcolor="rgba(0,212,255,0.06)",
            layer="below", line_width=0,
        )

    # Price line
    fig.add_trace(go.Scatter(
        x=close.index, y=close.values,
        name=f"{ticker} Close",
        line=dict(color=TEXT_COLOR, width=1.5),
        hovertemplate="%{x|%Y-%m-%d}<br>$%{y:.2f}<extra></extra>",
    ))

    # BUY markers
    if len(buys) > 0:
        buy_prices = close.reindex(buys)
        fig.add_trace(go.Scatter(
            x=buys, y=buy_prices * 0.985,
            mode="markers",
            name="Buy",
            marker=dict(
                symbol="triangle-up",
                size=10,
                color=STRATEGY_COLOR,
                line=dict(color="#004466", width=1),
            ),
            hovertemplate="%{x|%Y-%m-%d}<br>BUY @ $%{y:.2f}<extra></extra>",
        ))

    # SELL markers
    if len(sells) > 0:
        sell_prices = close.reindex(sells)
        fig.add_trace(go.Scatter(
            x=sells, y=sell_prices * 1.015,
            mode="markers",
            name="Sell",
            marker=dict(
                symbol="triangle-down",
                size=10,
                color=BENCHMARK_COLOR,
                line=dict(color="#662200", width=1),
            ),
            hovertemplate="%{x|%Y-%m-%d}<br>SELL @ $%{y:.2f}<extra></extra>",
        ))

    trade_count = len(buys)
    fig.update_layout(
        title=dict(
            text=f"{strategy_name} — Signal Overlay  ({trade_count} entries)",
            font=dict(size=16),
        ),
        yaxis_title=f"{ticker} Price (USD)",
        xaxis_title="Date",
        hovermode="x unified",
        **LAYOUT_DEFAULTS,
    )
    return fig


# ── 9. Monthly returns heatmap ────────────────────────────────────────────────

def monthly_returns_heatmap(
    returns: pd.Series,
    strategy_name: str = "Strategy",
) -> go.Figure:
    """
    Calendar heatmap: Year × Month grid of compounded monthly returns.

    This is the signature visualisation on any institutional performance
    tear sheet and is instantly recognisable to quant practitioners.
    """
    from ..analytics.performance import monthly_returns_table
    pivot = monthly_returns_table(returns)
    if pivot.empty:
        return go.Figure()

    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    years = pivot.index.tolist()
    z_vals = pivot[months].values * 100  # convert to %

    # Annotation text (show the % value in each cell)
    text = []
    for row in z_vals:
        row_text = []
        for val in row:
            row_text.append(f"{val:.1f}%" if not np.isnan(val) else "")
        text.append(row_text)

    # Symmetric colour scale centred on zero
    abs_max = np.nanmax(np.abs(z_vals))
    abs_max = max(abs_max, 1.0)

    colorscale = [
        [0.000, "#8B0000"],
        [0.250, "#CC3333"],
        [0.450, "#FF9999"],
        [0.500, "#1E2A3A"],   # neutral (matches grid)
        [0.550, "#99FFBB"],
        [0.750, "#33CC66"],
        [1.000, "#006400"],
    ]

    fig = go.Figure(data=go.Heatmap(
        z=z_vals,
        x=months,
        y=[str(y) for y in years],
        colorscale=colorscale,
        zmid=0,
        zmin=-abs_max,
        zmax=abs_max,
        text=text,
        texttemplate="%{text}",
        textfont=dict(size=10, color="white"),
        colorbar=dict(
            title="Return %",
            tickformat=".0f",
            ticksuffix="%",
        ),
        hovertemplate="Year: %{y}  Month: %{x}<br>Return: <b>%{z:.2f}%</b><extra></extra>",
    ))

    # Annual return column as a separate bar-style trace using annotation
    for i, year in enumerate(years):
        annual_val = pivot.loc[year, "Annual"] * 100
        if not np.isnan(annual_val):
            color = POSITIVE_COLOR if annual_val >= 0 else NEGATIVE_COLOR
            fig.add_annotation(
                x=len(months) - 0.5,
                y=str(year),
                text=f"{annual_val:.1f}%",
                showarrow=False,
                font=dict(color=color, size=10, family="monospace"),
                xref="x", yref="y",
            )

    fig.update_layout(
        title=dict(
            text=f"{strategy_name} — Monthly Returns",
            font=dict(size=16),
        ),
        xaxis=dict(
            title="Month",
            side="top",
            gridcolor=BACKGROUND,
            tickfont=dict(size=11),
        ),
        yaxis=dict(
            title="Year",
            autorange="reversed",
            gridcolor=BACKGROUND,
            tickfont=dict(size=11),
        ),
        paper_bgcolor=BACKGROUND,
        plot_bgcolor=BACKGROUND,
        font=dict(color=TEXT_COLOR, family="Inter, sans-serif", size=12),
        margin=dict(l=60, r=30, t=80, b=30),
    )
    return fig


# ── 10. Annual returns bar chart ──────────────────────────────────────────────

def annual_returns_chart(
    strategy_rets: pd.Series,
    benchmark_rets: pd.Series | None = None,
    strategy_name: str = "Strategy",
    benchmark_name: str = "Benchmark",
) -> go.Figure:
    """
    Grouped bar chart of calendar-year returns: strategy vs benchmark.
    """
    from ..analytics.performance import annual_returns

    s_annual = annual_returns(strategy_rets) * 100
    fig = go.Figure()

    bar_colors = [POSITIVE_COLOR if v >= 0 else NEGATIVE_COLOR for v in s_annual.values]
    fig.add_trace(go.Bar(
        x=[str(y) for y in s_annual.index],
        y=s_annual.values,
        name=strategy_name,
        marker_color=bar_colors,
        opacity=0.85,
        hovertemplate="Year: %{x}<br>Return: <b>%{y:.1f}%</b><extra></extra>",
    ))

    if benchmark_rets is not None:
        b_annual = annual_returns(benchmark_rets) * 100
        fig.add_trace(go.Bar(
            x=[str(y) for y in b_annual.index],
            y=b_annual.values,
            name=benchmark_name,
            marker_color=BENCHMARK_COLOR,
            opacity=0.55,
            hovertemplate="Year: %{x}<br>Return: <b>%{y:.1f}%</b><extra></extra>",
        ))

    fig.add_hline(y=0, line_color=GRID_COLOR, line_width=1)

    fig.update_layout(
        title=dict(text="Annual Returns", font=dict(size=16)),
        xaxis_title="Year",
        yaxis_title="Return (%)",
        barmode="group",
        **LAYOUT_DEFAULTS,
    )
    return fig


# ── 11. Rolling Sharpe chart ──────────────────────────────────────────────────

def rolling_sharpe_chart(
    strategy_rets: pd.Series,
    benchmark_rets: pd.Series | None = None,
    window: int = 63,
    strategy_name: str = "Strategy",
    benchmark_name: str = "Benchmark",
) -> go.Figure:
    """
    Rolling annualised Sharpe ratio over a sliding window.

    A declining rolling Sharpe often precedes a drawdown and signals
    deteriorating strategy performance — the key real-time risk monitor.
    """
    from ..analytics.performance import rolling_sharpe

    s_roll = rolling_sharpe(strategy_rets, window=window)
    fig = go.Figure()

    # Colour the Sharpe line by positive/negative
    fig.add_trace(go.Scatter(
        x=s_roll.index, y=s_roll.values,
        name=f"{strategy_name} ({window}d Sharpe)",
        line=dict(color=STRATEGY_COLOR, width=2),
        hovertemplate="%{x|%Y-%m-%d}<br>Sharpe: <b>%{y:.2f}</b><extra></extra>",
    ))

    if benchmark_rets is not None:
        b_roll = rolling_sharpe(benchmark_rets, window=window)
        fig.add_trace(go.Scatter(
            x=b_roll.index, y=b_roll.values,
            name=f"{benchmark_name} ({window}d Sharpe)",
            line=dict(color=BENCHMARK_COLOR, width=2, dash="dot"),
            hovertemplate="%{x|%Y-%m-%d}<br>Sharpe: <b>%{y:.2f}</b><extra></extra>",
        ))

    fig.add_hline(y=0, line_color=GRID_COLOR, line_width=1)
    fig.add_hline(y=1.0, line_color=POSITIVE_COLOR, line_dash="dot",
                  line_width=1, annotation_text="Sharpe = 1",
                  annotation_position="right")

    fig.update_layout(
        title=dict(text=f"Rolling {window}-Day Sharpe Ratio", font=dict(size=16)),
        yaxis_title="Sharpe Ratio",
        xaxis_title="Date",
        hovermode="x unified",
        **LAYOUT_DEFAULTS,
    )
    return fig


# ── 12. Returns scatter (beta visualisation) ──────────────────────────────────

def returns_scatter_chart(
    strategy_rets: pd.Series,
    benchmark_rets: pd.Series,
    strategy_name: str = "Strategy",
    benchmark_name: str = "Benchmark",
) -> go.Figure:
    """
    Scatter plot of daily strategy returns vs benchmark returns with OLS fit line.

    The slope of the regression line is visually interpretable as beta.
    """
    aligned = pd.concat(
        [strategy_rets.rename("s"), benchmark_rets.rename("b")], axis=1
    ).dropna()

    if len(aligned) < 10:
        return go.Figure()

    # OLS fit
    coeffs = np.polyfit(aligned["b"], aligned["s"], 1)
    beta, alpha = coeffs
    x_fit = np.linspace(aligned["b"].min(), aligned["b"].max(), 200)
    y_fit = beta * x_fit + alpha

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=aligned["b"] * 100,
        y=aligned["s"] * 100,
        mode="markers",
        name="Daily returns",
        marker=dict(color=STRATEGY_COLOR, size=4, opacity=0.5),
        hovertemplate=(
            f"{benchmark_name}: %{{x:.2f}}%<br>{strategy_name}: %{{y:.2f}}%<extra></extra>"
        ),
    ))

    fig.add_trace(go.Scatter(
        x=x_fit * 100, y=y_fit * 100,
        mode="lines",
        name=f"OLS fit (β={beta:.2f})",
        line=dict(color=BENCHMARK_COLOR, width=2),
        hoverinfo="skip",
    ))

    fig.add_hline(y=0, line_color=GRID_COLOR, line_width=1)
    fig.add_vline(x=0, line_color=GRID_COLOR, line_width=1)

    fig.update_layout(
        title=dict(
            text=f"Return Scatter — β={beta:.2f}, α={alpha*252*100:.1f}% ann.",
            font=dict(size=16),
        ),
        xaxis_title=f"{benchmark_name} Daily Return (%)",
        yaxis_title=f"{strategy_name} Daily Return (%)",
        **LAYOUT_DEFAULTS,
    )
    return fig


# ── 13. Underwater (time below peak) chart ────────────────────────────────────

def underwater_chart(
    equity_df: pd.DataFrame,
    strategy_name: str = "Strategy",
) -> go.Figure:
    """
    Underwater chart: shows continuous drawdown depth over time.

    Distinguishes between deep-but-short drawdowns and shallow-but-prolonged ones.
    Red intensity encodes drawdown severity.
    """
    pv = equity_df["Portfolio_Value"]
    peak = pv.cummax()
    dd = (pv - peak) / peak * 100

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=dd.index, y=dd.values,
        fill="tozeroy",
        line=dict(color="rgba(255,68,68,0.8)", width=1),
        fillcolor="rgba(255,68,68,0.15)",
        name="Drawdown",
        hovertemplate="%{x|%Y-%m-%d}<br>Underwater: <b>%{y:.2f}%</b><extra></extra>",
    ))

    # Annotate max drawdown point
    min_idx = dd.idxmin()
    min_val = dd.min()
    fig.add_annotation(
        x=min_idx, y=min_val,
        text=f"Max DD: {min_val:.1f}%",
        showarrow=True, arrowhead=2,
        arrowcolor=NEGATIVE_COLOR,
        font=dict(color=NEGATIVE_COLOR, size=11),
        bgcolor=BACKGROUND, bordercolor=NEGATIVE_COLOR,
    )

    fig.update_layout(
        title=dict(text=f"{strategy_name} — Underwater Chart", font=dict(size=16)),
        yaxis_title="Drawdown from Peak (%)",
        xaxis_title="Date",
        **LAYOUT_DEFAULTS,
    )
    return fig


# ── 14. Multi-strategy comparison ────────────────────────────────────────────

def strategy_comparison_chart(
    results: dict[str, pd.DataFrame],
    initial_capital: float = 100_000.0,
) -> go.Figure:
    """
    Overlaid equity curves for multiple strategies on the same asset.

    Parameters
    ----------
    results : dict[strategy_name → equity_curve DataFrame]
    """
    fig = go.Figure()

    for i, (name, equity_df) in enumerate(results.items()):
        color = PALETTE[i % len(PALETTE)]
        fig.add_trace(go.Scatter(
            x=equity_df.index,
            y=equity_df["Portfolio_Value"],
            name=name,
            line=dict(color=color, width=2),
            hovertemplate=f"{name}: $%{{y:,.0f}}<extra></extra>",
        ))

    fig.add_hline(y=initial_capital, line_color=GRID_COLOR,
                  line_dash="dot", line_width=1)

    fig.update_layout(
        title=dict(text="Strategy Comparison — Equity Curves", font=dict(size=16)),
        yaxis_title="Portfolio Value (USD)",
        xaxis_title="Date",
        hovermode="x unified",
        **LAYOUT_DEFAULTS,
    )
    return fig


# ── 15. Allocation pie chart ──────────────────────────────────────────────────

def allocation_pie_chart(
    weights: dict[str, float],
    title: str = "Portfolio Allocation",
) -> go.Figure:
    """Donut chart of portfolio weights."""
    labels = list(weights.keys())
    values = [max(v, 0) for v in weights.values()]
    total  = sum(values)
    if total > 0:
        values = [v / total for v in values]

    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.45,
        textinfo="label+percent",
        hovertemplate="%{label}: %{percent}<extra></extra>",
        marker=dict(colors=PALETTE[:len(labels)]),
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=15)),
        showlegend=True,
        legend=dict(orientation="v", x=1.02, y=0.5),
        **LAYOUT_DEFAULTS,
        height=380,
    )
    return fig


# ── 16. Correlation heatmap ───────────────────────────────────────────────────

def correlation_heatmap(
    corr_matrix: pd.DataFrame,
    title: str = "Correlation Matrix",
) -> go.Figure:
    """
    Annotated correlation heatmap.

    Green = positive correlation, Red = negative.
    Diagonal is always 1.0.
    """
    if corr_matrix.empty:
        return go.Figure()

    labels = list(corr_matrix.columns)
    z      = corr_matrix.values.tolist()

    fig = go.Figure(go.Heatmap(
        z=z,
        x=labels, y=labels,
        zmin=-1, zmax=1,
        colorscale=[
            [0.0, "#FF4444"],
            [0.5, "#151B2E"],
            [1.0, "#00CC66"],
        ],
        text=[[f"{v:.2f}" for v in row] for row in z],
        texttemplate="%{text}",
        hovertemplate="%{x} / %{y}: <b>%{z:.4f}</b><extra></extra>",
        showscale=True,
        colorbar=dict(title="ρ", thickness=14),
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=15)),
        xaxis=dict(tickangle=-35),
        **LAYOUT_DEFAULTS,
        height=max(300, 80 * len(labels)),
    )
    return fig


# ── 17. Factor exposure bar chart ─────────────────────────────────────────────

def factor_exposure_chart(
    factor_loadings: dict[str, float],
    title: str = "Factor Exposure (ETF Proxy)",
) -> go.Figure:
    """
    Horizontal bar chart of factor loadings from OLS regression.

    Positive loading = long exposure; negative = inverse/hedging.
    """
    if not factor_loadings:
        return go.Figure()

    factors = list(factor_loadings.keys())
    values  = list(factor_loadings.values())
    colours = [POSITIVE_COLOR if v >= 0 else NEGATIVE_COLOR for v in values]

    fig = go.Figure(go.Bar(
        y=factors,
        x=values,
        orientation="h",
        marker_color=colours,
        hovertemplate="%{y}: <b>%{x:.4f}</b><extra></extra>",
    ))

    fig.add_vline(x=0, line_color=GRID_COLOR, line_width=1)

    fig.update_layout(
        title=dict(text=title, font=dict(size=15)),
        xaxis_title="Loading (β)",
        **LAYOUT_DEFAULTS,
        height=max(280, 50 * len(factors)),
        margin=dict(l=160, r=30, t=55, b=50),
    )
    return fig


# ── 18. Stress test loss chart ────────────────────────────────────────────────

def stress_test_chart(
    stress_df: pd.DataFrame,
    initial_value: float = 100_000.0,
    title: str = "Stress Test — Estimated Portfolio Impact",
) -> go.Figure:
    """
    Horizontal bar chart of estimated portfolio losses under stress scenarios.

    Bars are coloured by severity: worst losses in deep red.
    """
    if stress_df.empty:
        return go.Figure()

    scenarios = stress_df["Scenario"].tolist()
    pct_vals  = (stress_df["Estimated_Loss_Pct"] * 100).tolist()
    usd_vals  = stress_df["Estimated_Loss_USD"].tolist()

    # Colour gradient: more negative = more red
    max_loss = min(pct_vals) if pct_vals else -1
    colours  = []
    for v in pct_vals:
        intensity = min(abs(v) / max(abs(max_loss), 1), 1.0)
        r = int(255 * intensity)
        g = int(50  * (1 - intensity))
        colours.append(f"rgb({r},{g},50)")

    fig = go.Figure(go.Bar(
        y=scenarios,
        x=pct_vals,
        orientation="h",
        marker_color=colours,
        text=[f"${abs(u):,.0f}" for u in usd_vals],
        textposition="outside",
        hovertemplate=(
            "%{y}<br>Loss: <b>%{x:.2f}%</b><br>"
            "Est. USD: $%{text}<extra></extra>"
        ),
    ))

    fig.add_vline(x=0, line_color=GRID_COLOR, line_width=1)

    fig.update_layout(
        title=dict(text=title, font=dict(size=15)),
        xaxis_title="Estimated Loss (%)",
        **LAYOUT_DEFAULTS,
        height=max(320, 48 * len(scenarios)),
        margin=dict(l=240, r=80, t=55, b=50),
    )
    return fig


# ── 19. Drawdown comparison ───────────────────────────────────────────────────

def drawdown_comparison_chart(
    equity_dict: dict[str, pd.DataFrame],
    title: str = "Drawdown Comparison",
) -> go.Figure:
    """
    Overlaid drawdown curves for multiple strategies or assets.

    Parameters
    ----------
    equity_dict : dict[name → equity_curve DataFrame]
    """
    fig = go.Figure()

    for i, (name, eq) in enumerate(equity_dict.items()):
        pv   = eq["Portfolio_Value"]
        peak = pv.cummax()
        dd   = (pv - peak) / peak * 100
        color = PALETTE[i % len(PALETTE)]

        fig.add_trace(go.Scatter(
            x=dd.index, y=dd.values,
            name=name,
            line=dict(color=color, width=1.5),
            hovertemplate=f"{name}: %{{y:.2f}}%<extra></extra>",
        ))

    fig.add_hline(y=0, line_color=GRID_COLOR, line_width=1)

    fig.update_layout(
        title=dict(text=title, font=dict(size=15)),
        yaxis_title="Drawdown from Peak (%)",
        xaxis_title="Date",
        hovermode="x unified",
        **LAYOUT_DEFAULTS,
    )
    return fig
