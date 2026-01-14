# Strategy Backtest Implementation Plan

## Overview
Build a unified backtesting framework to test all 10 advanced strategies independently, compare performance, and identify the best performers for live trading.

---

## Phase 1: Core Framework (Week 1)

### 1.1 Base Architecture
**Goal:** Create reusable components that all strategies will use

**Files to Create:**
```
test/framework/
├── __init__.py
├── base_strategy.py          # Abstract base class
├── backtest_engine.py        # Core execution engine
├── performance_analyzer.py   # Unified metrics
├── data_manager.py            # Data fetching/caching
└── strategy_runner.py        # Orchestration
```

**Key Components:**

1. **BaseStrategy** (Abstract Interface)
   ```python
   class BaseStrategy:
       def generate_signal(self, market_data) -> Optional[Signal]
       def calculate_position_size(self, signal, account) -> int
       def check_exit(self, position, market_data) -> Optional[ExitReason]
       def get_required_data(self) -> List[str]  # ["OHLCV", "bid_ask", etc.]
   ```

2. **BacktestEngine** (Core Execution)
   - Bar-by-bar simulation
   - Order fill logic (market/limit orders)
   - Position tracking
   - P&L calculation
   - Slippage/commission modeling

3. **PerformanceAnalyzer** (Standardized Metrics)
   - Sharpe Ratio
   - Sortino Ratio
   - Max Drawdown
   - Win Rate
   - Profit Factor
   - Average Daily Return
   - Consistency Score (std dev of daily returns)
   - Calmar Ratio

4. **DataManager** (Data Handling)
   - Fetch from TopstepX API
   - Cache to disk
   - Align multiple instruments/timeframes
   - Handle missing data

---

## Phase 2: Strategy Implementations (Week 2-3)

### Priority Order (Easiest → Hardest)

#### **Tier 1: Start Here (Full Data Support)**

**1. Calendar Spread Arbitrage** ⭐ EASIEST
- **Data Needed:** OHLCV for front month + next month
- **Complexity:** Low
- **Implementation:** 
  - Calculate spread = Price_Front - Price_Back
  - Z-score normalization
  - Mean reversion entry/exit
- **Expected Time:** 1 day

**2. Optimal Stopping Theory** ⭐
- **Data Needed:** OHLCV + basic indicators
- **Complexity:** Low-Medium
- **Implementation:**
  - Score potential entries (momentum + mean reversion + volatility)
  - 37% rule for entry selection
  - Dynamic exit optimization
- **Expected Time:** 2 days

**3. Multi-Timeframe Convergence** ⭐
- **Data Needed:** OHLCV (1m, 5m, 15m)
- **Complexity:** Medium
- **Implementation:**
  - Calculate signals on each timeframe
  - Weight by confidence (R²)
  - Only trade when all agree
- **Expected Time:** 2 days

**4. Volatility Regime Trading** ⭐
- **Data Needed:** OHLCV + ATR/volatility
- **Complexity:** Medium
- **Implementation:**
  - Calculate intraday volatility curve
  - Compare actual vs expected vol
  - Switch strategies (mean reversion vs momentum)
- **Expected Time:** 2-3 days

**5. Volatility Surface Trading** ⭐
- **Data Needed:** OHLCV + volatility metrics
- **Complexity:** Medium
- **Implementation:**
  - Calculate vol ratio (short-term / long-term)
  - Mean reversion when ratio deviates
- **Expected Time:** 2 days

**6. Regime-Adaptive Multi-Strategy** ⭐
- **Data Needed:** OHLCV + multiple indicators
- **Complexity:** High
- **Implementation:**
  - Calculate regime indicators (Hurst, volatility, order flow)
  - HMM for regime classification
  - Switch between strategies
- **Expected Time:** 3-4 days

**7. Cross-Sectional Momentum** ⭐
- **Data Needed:** OHLCV (MES, MNQ, MYM)
- **Complexity:** Medium
- **Implementation:**
  - Calculate momentum for each instrument
  - Correlation matrix
  - Only trade when all align
- **Expected Time:** 2-3 days

#### **Tier 2: Need Data Verification**

**8. Liquidity Provision** ⚠️
- **Data Needed:** Bid/Ask volume (need to verify)
- **Complexity:** Medium-High
- **Implementation:** After verifying GatewayDepth structure

**9. Order Flow Imbalance** ⚠️
- **Data Needed:** Bid/Ask volume (need to verify)
- **Complexity:** High
- **Implementation:** After verifying GatewayDepth structure

**10. Adaptive Market Making** ⚠️
- **Data Needed:** Full order book (need to verify)
- **Complexity:** High
- **Implementation:** After verifying GatewayDepth structure

---

## Phase 3: Testing & Comparison (Week 4)

### 3.1 Individual Strategy Backtests
**For each strategy:**
1. Test on historical data (2025-01-01 to 2025-11-14)
2. Walk-forward optimization (train on first 60%, test on last 40%)
3. Parameter sensitivity analysis
4. Generate performance report

### 3.2 Strategy Comparison
**Create comparison matrix:**
- Run all strategies on same date range
- Compare metrics side-by-side
- Rank by:
  1. Sharpe Ratio (primary)
  2. Max Drawdown (risk)
  3. Win Rate (consistency)
  4. Average Daily Return (profitability)

### 3.3 Best Strategy Selection
**Criteria:**
- Sharpe > 2.0
- Max Drawdown < 10%
- Win Rate > 60%
- Positive average daily return
- Consistent (low std dev of daily returns)

---

## Phase 4: Optimization & Production (Week 5)

### 4.1 Parameter Optimization
**For top 3 strategies:**
- Grid search for optimal parameters
- Walk-forward validation
- Out-of-sample testing

### 4.2 Production Integration
**Convert best strategies to live trading:**
- Integrate with existing `main.py` structure
- Add to strategy selection in config
- Test in dry-run mode first

---

## Implementation Timeline

### Week 1: Framework
- **Day 1-2:** BaseStrategy + BacktestEngine
- **Day 3-4:** PerformanceAnalyzer + DataManager
- **Day 5:** StrategyRunner + testing framework

### Week 2: Tier 1 Strategies (7 strategies)
- **Day 1:** Calendar Spread
- **Day 2:** Optimal Stopping
- **Day 3:** Multi-Timeframe
- **Day 4:** Volatility Regime
- **Day 5:** Volatility Surface

### Week 3: Remaining Strategies
- **Day 1-2:** Regime-Adaptive
- **Day 3:** Cross-Sectional
- **Day 4-5:** Data verification + Tier 2 strategies (if data available)

### Week 4: Testing & Comparison
- **Day 1-2:** Run all backtests
- **Day 3:** Generate comparison reports
- **Day 4:** Select top performers
- **Day 5:** Parameter optimization

### Week 5: Production
- **Day 1-2:** Optimize top strategies
- **Day 3-4:** Integrate into live system
- **Day 5:** Dry-run testing

---

## File Structure

```
test/
├── framework/
│   ├── __init__.py
│   ├── base_strategy.py
│   ├── backtest_engine.py
│   ├── performance_analyzer.py
│   ├── data_manager.py
│   └── strategy_runner.py
├── strategies/
│   ├── __init__.py
│   ├── calendar_spread.py
│   ├── optimal_stopping.py
│   ├── multi_timeframe.py
│   ├── volatility_regime.py
│   ├── volatility_surface.py
│   ├── regime_adaptive.py
│   ├── cross_sectional.py
│   ├── liquidity_provision.py (if data available)
│   ├── order_flow_imbalance.py (if data available)
│   └── adaptive_mm.py (if data available)
├── comparison/
│   ├── __init__.py
│   ├── strategy_comparator.py
│   └── generate_report.py
├── configs/
│   ├── calendar_spread_config.yaml
│   ├── optimal_stopping_config.yaml
│   └── ... (config for each strategy)
└── results/
    ├── backtests/
    │   ├── calendar_spread_2025-01-01_2025-11-14.json
    │   └── ...
    ├── comparisons/
    │   └── strategy_comparison_2025.html
    └── reports/
        └── ...
```

---

## Key Design Decisions

### 1. Backtest Realism
- **Slippage:** 1 tick for market orders, 0 for limit orders (if price touched)
- **Commission:** $2.50 per contract per round trip (Topstep standard)
- **Fill Logic:**
  - Market orders: Fill at next bar's open (worst case)
  - Limit orders: Fill if price touches limit during bar

### 2. Data Requirements   
- **Primary Instruments:** MES, MNQ, MGC, GC
- **Timeframes:** 1m, 5m, 15m, 1h
- **Date Range:** 2025-01-01 to 2025-11-14 (or latest available)
- **Caching:** All data cached to disk to avoid repeated API calls

### 3. Performance Metrics Priority
1. **Sharpe Ratio** (risk-adjusted return) - PRIMARY
2. **Max Drawdown** (risk control)
3. **Win Rate** (consistency)
4. **Average Daily Return** (profitability)
5. **Consistency Score** (low std dev of daily returns)

### 4. Parameter Optimization
- **Method:** Walk-forward optimization
- **Train/Test Split:** 60/40
- **Validation:** Out-of-sample testing on held-out period
- **Avoid Overfitting:** Max 3-5 parameters per strategy

---

## Success Criteria

### For Each Strategy:
- ✅ Backtest runs without errors
- ✅ Generates performance report
- ✅ Exports trades to CSV
- ✅ Creates equity curve visualization

### For Framework:
- ✅ All strategies use same interface
- ✅ Performance metrics are comparable
- ✅ Easy to add new strategies
- ✅ Fast execution (< 5 min per strategy)

### For Selection:
- ✅ Top 3 strategies identified
- ✅ Sharpe > 2.0
- ✅ Max Drawdown < 10%
- ✅ Win Rate > 60%

---

## Next Steps

1. **Start with Framework** (Week 1)
   - Build BaseStrategy interface
   - Build BacktestEngine
   - Build PerformanceAnalyzer

2. **Implement Calendar Spread First** (Easiest, validates framework)
   - Simple spread calculation
   - Mean reversion logic
   - Quick win to validate approach

3. **Iterate and Improve**
   - Test Calendar Spread
   - Fix any framework issues
   - Then implement remaining strategies

---

## Questions to Resolve

1. **Data Verification:** Check GatewayDepth structure for bid/ask volume
2. **Commission:** Confirm Topstep commission structure
3. **Slippage:** Decide on realistic slippage model
4. **Date Range:** Confirm available historical data range
5. **Instruments:** Confirm which instruments are available for backtesting

---

## Ready to Start?

**Recommended First Steps:**
1. Build BaseStrategy interface
2. Build BacktestEngine (reuse your existing engine patterns)
3. Implement Calendar Spread (simplest strategy)
4. Test and validate framework
5. Then implement remaining strategies

This approach gives us:
- ✅ Quick validation (Calendar Spread is simple)
- ✅ Reusable framework for all strategies
- ✅ Systematic comparison
- ✅ Data-driven strategy selection

