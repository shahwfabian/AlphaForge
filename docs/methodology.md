# AlphaForge Methodology

## Backtesting Assumptions

1. **Daily bar data**: All strategies execute on daily OHLCV data.
2. **Execution at next open**: Signals generated on day _t_ are executed on
   day _t+1_ (implemented by shifting positions by 1 bar).
3. **Long-only**: Strategies take long positions only (no short selling).
4. **Fully invested**: On entry, approximately 95 % of available cash is deployed.
5. **No partial fills**: Orders are assumed to fill completely at the simulated price.
6. **No margin**: The portfolio cannot hold more notional than its cash balance.

## Transaction Costs

Every BUY and SELL fill incurs a commission charge:

```
commission = quantity * fill_price * transaction_cost_rate
```

A typical setting is 0.1 % (10 bps), representing retail brokerage costs.
Zero-commission brokers (e.g., Interactive Brokers for large accounts) can
use 0.0.

## Slippage

Market impact / bid-ask spread is approximated as a fixed percentage of price:

```
buy_fill  = close_price * (1 + slippage)
sell_fill = close_price * (1 - slippage)
```

A typical setting is 0.05 % (5 bps), appropriate for liquid large-cap equities.
Increase to 0.1–0.2 % for mid/small-cap or low-liquidity assets.

## Lookahead Bias Prevention

The most common error in backtesting is using future data to generate
present-day signals.  AlphaForge prevents this in two ways:

1. **Signal shift**: `BaseStrategy._finalise()` applies `Signal.shift(1)`,
   ensuring the position on day _t_ is determined by the signal from day _t-1_.

2. **Rolling windows**: All moving averages and z-scores use
   `min_periods = window`, meaning no signal is generated until a full
   window of data is available.

## Benchmark Comparison

The default benchmark is SPY (S&P 500 ETF).  A Buy & Hold position on SPY
is simulated with zero transaction costs for a fair comparison.

**Alpha (Jensen's Alpha)**:
```
α = E[r_strategy] - [rf + β × (E[r_benchmark] - rf)]   (annualised)
```

**Beta**:
```
β = Cov(r_strategy, r_benchmark) / Var(r_benchmark)
```

## Risk Metrics

### Value at Risk (VaR)

Historical (non-parametric) VaR at confidence level _c_:
```
VaR_c = Quantile(returns, 1 - c)
```
Reads directly off the empirical return distribution.  Answers:
"With probability _c_, I will not lose more than |VaR| in one day."

### Expected Shortfall (ES / CVaR)

Average return in the worst (1-_c_) fraction of days:
```
ES_c = E[r | r < VaR_c]
```
ES is a coherent risk measure required by Basel III/IV for internal model
capital calculations.  It captures tail severity, not just the threshold.

### Rolling Volatility

Annualised using the square-root-of-time rule:
```
σ_annual = σ_daily * sqrt(252)
```

## Monte Carlo Simulation

The simulation draws from a Gaussian return distribution calibrated to the
historical strategy returns:

```
r_sim ~ N(μ_historical, σ_historical)
```

Limitations:
- Assumes i.i.d. returns (no autocorrelation, no volatility clustering)
- Assumes normally distributed returns (ignores fat tails)
- Does not model regime changes, crashes, or liquidity crises
- **Results are illustrative only — not predictive**

## Overfitting and Limitations

### Parameter Optimisation Overfitting

Grid search over strategy parameters finds the combination that maximises
an in-sample metric (e.g., Sharpe ratio).  This is almost guaranteed to
**overfit** historical data.  The selected parameters may reflect noise
rather than genuine signal.

Mitigation:
- Always validate optimised parameters on a separate out-of-sample period.
- Use walk-forward validation to estimate realistic degradation.
- Prefer parameters with economic intuition over purely data-mined ones.

### Survivorship Bias

AlphaForge uses yfinance which only returns data for tickers that currently
exist (or existed long enough to have historical data).  Companies that were
delisted, merged, or went bankrupt are **not** included.  Backtests on equity
universe data therefore carry upward survivorship bias.

### Other Limitations

- No corporate actions adjustment beyond yfinance's auto-adjust.
- No multi-asset portfolio rebalancing.
- Transaction costs are fixed-rate; real costs vary by broker and volume.
- No financing costs for leveraged positions.
- No accounting for taxes.
