# AlphaForge Architecture

## System Overview

AlphaForge is a layered research platform. Data flows from raw market feeds through
preprocessing, strategy signal generation, event-driven simulation, and finally into
analytics and the dashboard.

```
┌─────────────────────────────────────────────────────────────┐
│                        Streamlit UI (app.py)                │
│  Sidebar Config → Run Button → Charts + Metrics + Tables    │
└───────────────────────────┬─────────────────────────────────┘
                            │
          ┌─────────────────▼──────────────────┐
          │          Dashboard Layer            │
          │  components.py  ·  charts.py        │
          └─────────────────┬──────────────────┘
                            │
          ┌─────────────────▼──────────────────┐
          │          Analytics Layer            │
          │  performance · risk · benchmark     │
          │  monte_carlo · optimizer · walk_fwd │
          └─────────────────┬──────────────────┘
                            │
          ┌─────────────────▼──────────────────┐
          │         Backtesting Engine          │
          │  engine · portfolio · trade_log     │
          │  events (Market/Signal/Order/Fill)  │
          └─────────────────┬──────────────────┘
                            │
          ┌─────────────────▼──────────────────┐
          │          Strategy Layer             │
          │  BaseStrategy → MA / MR / Mom / B&H │
          └─────────────────┬──────────────────┘
                            │
          ┌─────────────────▼──────────────────┐
          │            Data Layer               │
          │  data_loader · preprocessing        │
          │  validation                         │
          └─────────────────┬──────────────────┘
                            │
          ┌─────────────────▼──────────────────┐
          │          Storage Layer              │
          │  SQLite (database.py)               │
          │  CSV exports (exporter.py)          │
          └────────────────────────────────────┘
```

## Layer Descriptions

### Data Layer (`src/data/`)

| Module | Responsibility |
|---|---|
| `data_loader.py` | Download OHLCV data from Yahoo Finance via yfinance; disk caching via Parquet |
| `preprocessing.py` | Clean data, compute returns, log returns, rolling volatility |
| `validation.py` | Input validation (tickers, dates, DataFrames, capital amounts) |

### Strategy Layer (`src/strategies/`)

All strategies inherit from `BaseStrategy`.  The base class enforces the
`generate_signals(df) → DataFrame` contract and provides the 1-day position
shift that prevents lookahead bias.

| Strategy | Signal Logic |
|---|---|
| `MovingAverageCrossover` | Long when SMA(short) > SMA(long) |
| `MeanReversion` | Long when z-score < –entry_threshold |
| `Momentum` | Long when trailing return > 0 |
| `BuyAndHold` | Always long; passive baseline |

### Backtesting Engine (`src/backtester/`)

Event-driven loop inspired by the Hurst (2014) architecture.

```
[MarketEvent] → [SignalEvent] → [OrderEvent] → [FillEvent] → Portfolio update
```

- **No lookahead bias**: signals use `Signal.shift(1)` so each day's
  execution uses yesterday's signal.
- **Transaction costs**: applied as a percentage of notional on each fill.
- **Slippage**: buy fills at `price * (1 + slippage)`, sell at `price * (1 - slippage)`.
- **Position sizing**: a fixed fraction (~95 %) of available cash on entry.

### Analytics Layer (`src/analytics/`)

| Module | Key Outputs |
|---|---|
| `performance.py` | CAGR, Sharpe, Sortino, Calmar, Max Drawdown, Win Rate, Profit Factor |
| `risk.py` | Historical VaR, Parametric VaR, Expected Shortfall, Rolling Volatility |
| `benchmark.py` | Alpha, Beta, R², Correlation, Tracking Error, Information Ratio |
| `monte_carlo.py` | 500 simulated paths; percentile fan chart |
| `optimizer.py` | Grid search over MA and MR parameters |
| `walk_forward.py` | IS vs OOS Sharpe comparison across rolling windows |

### Storage Layer (`src/storage/`)

- **SQLite** (`data/alphaforge.db`): One row per backtest run; stores all
  metadata and computed metrics for comparison.
- **Exporter** (`exporter.py`): Converts DataFrames to CSV bytes for
  Streamlit `st.download_button`.
