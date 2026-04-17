# HMM Regime-Based Trading Bot

A volatility-aware systematic trading system using Hidden Markov Models for regime detection, with strict risk management and Alpaca integration.

---

## PHASE 1: Project Structure & Environment Setup

regime-trader/
├── config/
│ ├── settings.yaml # All configurable parameters
│ └── credentials.yaml.example
├── core/
│ ├── init.py
│ ├── hmm_engine.py # HMM regime detection engine
│ ├── regime_strategies.py # Vol-based allocation strategies
│ ├── risk_manager.py # Position sizing, leverage, drawdown limits
│ └── signal_generator.py # Combines HMM + strategy into signals
├── broker/
│ ├── init.py
│ ├── alpaca_client.py # Alpaca API wrapper
│ ├── order_executor.py # Order placement, modification, cancellation
│ └── position_tracker.py # Track open positions, P&L
├── data/
│ ├── init.py
│ ├── market_data.py # Real-time and historical data fetching
│ └── feature_engineering.py # Technical indicators, feature computation
├── monitoring/
│ ├── init.py
│ ├── logger.py # Structured logging
│ ├── dashboard.py # Terminal-based live dashboard
│ └── alerts.py # Email/webhook alerts for critical events
├── backtest/
│ ├── init.py
│ ├── backtester.py # Walk-forward allocation backtester
│ ├── performance.py # Sharpe, drawdown, regime breakdown, benchmarks
│ └── stress_test.py # Crash injection, gap simulation
├── tests/
│ ├── test_hmm.py
│ ├── test_look_ahead.py
│ ├── test_strategies.py
│ ├── test_risk.py
│ └── test_orders.py
├── main.py # Entry point
├── requirements.txt
├── .env.example
└── README.md

Set up requirements.txt with:
hmmlearn, alpaca-trade-api, pandas, numpy, scipy, ta (technical analysis library), scikit-learn, pyyaml, python-dotenv, streamlit, websocket-client, schedule, rich.

## Phase 2: HMM Regime Detection Engine

Design Philosophy
The HMM is a volatility classifier. It detects whether the market is in a calm, moderate, or turbulent volatility environment. It does NOT predict price direction.

Requirements
Gaussian HMM with Automatic Model Selection:

Test n_components = [3, 4, 5, 6, 7] during training

Calculate BIC (Bayesian Information Criterion) for each candidate

Select the lowest BIC model

Labeling: After training, sort regimes by mean return (ascending):

3 regimes: CRASH, NEUTRAL, BULL

4 regimes: CRASH, BEAR, BULL, EUPHORIA

5 regimes: CRASH, BEAR, NEUTRAL, BULL, EUPHORIA

Feature Engineering (from OHLCV):

Log returns (1, 5, 20 periods)

Volatility: realized vol (20-period rolling std), vol ratio (5-period / 20-period)

Trend: ADX (14-period), slope of 50-period SMA

Mean reversion: RSI (14-period), z-score vs 200 SMA

Momentum: ROC (10, 20)

Range: normalized ATR (14-period ATR / close)

No Look-Ahead Bias:

Do NOT use model.predict()

Use the Forward Algorithm (only past and current data)

Regime Stability Filter:

Regime change only "confirmed" after persisting N bars (default: 3)

## Phase 3: Volatility-Based Allocation Strategy

Implemented in core/regime_strategies.py

Three Strategy Classes (based on volatility rank)
Strategy Volatility Rank Direction Allocation Leverage
LowVolBullStrategy Lowest third LONG 95% of portfolio 1.25x
MidVolCautiousStrategy Middle third LONG if price > 50 SMA, else FLAT 60% of portfolio 1.0x
HighVolDefensiveStrategy Highest third FLAT (cash) or 10% defensive 0-10% of portfolio 0.5x
Rebalancing: Only when target allocation differs from current by >10%

## Phase 4: Walk-Forward Backtesting & Validation

Walk-Forward Optimization Engine
In-Sample (IS): 252 trading days (1 year) for training

Out-of-Sample (OOS): 126 trading days (6 months) for evaluation

Performance Metrics
Total return (%), CAGR, Sharpe ratio, Sortino ratio

Max drawdown, Win rate, Profit factor

Benchmark comparisons: Buy-and-hold SPY, 200 SMA trend following

Stress Testing
Crash injection: Insert -5% to -15% single-day gaps at random points

Gap risk: Insert overnight gaps of 2-5x ATR

## Phase 5: Risk Management Layer

The risk manager operates INDEPENDENTLY of the HMM. It has ABSOLUTE VETO POWER over any signal.

Portfolio-Level Limits
Max total exposure: 80% of portfolio (20% cash minimum)

Max single position: 15%

Max concurrent positions: 5

Circuit Breakers (fire on actual P&L)
Condition Action
Daily DD > 2% Reduce all sizes 50%, halt rest of day
Daily DD > 3% Close ALL positions, halt rest of day
Weekly DD > 7% Close ALL, halt rest of week
Peak DD > 10% Full halt, require manual deletion of trading_halted.lock file to resume
Position-Level Risk
Every position MUST have a stop loss (default: 3x ATR)

## Phase 6: Alpaca Broker Integration

Module Purpose
alpaca_client.py SDK wrapper, exponential backoff on reconnect
order_executor.py Submit bracket orders (entry + stop + profit target)
position_tracker.py Track unrealized P&L in real-time

## Phase 7: Main Loop & Orchestration

Startup (main.py)
Load config

Connect to Alpaca

Check market hours

Train/load HMM

Initialize risk manager

Main Loop (each bar close, default 5-min bars)
python
while trading_active:
fetch_new_bar_data()
compute_features()
hmm_prediction(filtered)
strategy_orchestrator() -> target_allocation
risk_manager() -> validate/modify_signal
order_executor() -> execute_trades
update_dashboard()

## Phase 8: Monitoring, Alerts & Dashboard

Component Purpose
logger.py Structured JSON logging, 30-day rotation
dashboard.py Streamlit-based UI showing: current regime (confidence & stability), portfolio equity curve & drawdown, risk status (circuit breakers), recent signal feed
alerts.py Email/webhook alerts for critical events

## Phase 9: Integration Testing & Documentation

End-to-end dry run: Verify data -> HMM -> strategy -> risk -> broker flow.
README.md: Complete setup instructions and risk management overview.
