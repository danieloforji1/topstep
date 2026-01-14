# Production-Grade Backtesting Guide

## Overview

The backtest framework now includes **realistic, production-grade analysis** to help you make informed decisions about which strategies to implement.

## What Makes These Backtests Realistic?

### ✅ Execution Realism
- **Slippage**: 1 tick per contract (configurable)
- **Commission**: $2.50 per contract round trip (Topstep standard)
- **Fill Logic**: Market orders fill at next bar (worst case)
- **Order Types**: Supports market and limit orders

### ✅ Comprehensive Analysis
- **Walk-Forward**: Tests robustness across time periods
- **Parameter Optimization**: Finds best parameters
- **Multiple Metrics**: Sharpe, Sortino, Calmar, Max DD, Win Rate
- **Trade Analysis**: Monthly breakdown, drawdown periods

### ✅ Decision Criteria
- **Minimum Requirements**: Sharpe > 2.0, Max DD < 10%, Win Rate > 60%
- **Walk-Forward Success**: >70% positive iterations
- **Stability**: Consistent performance across periods

## Quick Start

### For Quick Testing (Simple Examples)
```bash
# Calendar Spread
python run_calendar_spread.py

# Optimal Stopping
python run_optimal_stopping.py
```


### find contract date
   python3.11 examples/find_contract_data.py --symbol MES --interval 15m

### For Production Analysis (Recommended)
```bash
# Comprehensive analysis (all modes)
python run_realistic_backtest.py \
    --strategy optimal_stopping \
    --symbol MES \
    --interval 15m \
    --start-date 2025-01-01 \
    --end-date 2025-11-14 \
    --mode all
```

## Analysis Modes

### 1. Single Backtest
**Use for:** Quick validation, initial testing

```bash
--mode single
```

**Output:**
- Detailed performance report
- Trade-by-trade CSV
- Key metrics summary

### 2. Walk-Forward Analysis ⭐ RECOMMENDED
**Use for:** Testing robustness, avoiding overfitting

```bash
--mode walk-forward
```

**What it does:**
- Splits data: 60% train, 40% test
- Steps forward monthly
- Tests on out-of-sample data
- Reports average across all iterations

**Key Metrics:**
- Average Sharpe across iterations
- Success rate (positive vs negative periods)
- Performance stability

**Decision Rule:**
- ✅ **Accept** if success rate > 70%
- ⚠️ **Review** if success rate 60-70%
- ❌ **Reject** if success rate < 60%

### 3. Parameter Optimization
**Use for:** Finding best parameters

```bash
--mode optimize
```

**What it does:**
- Tests all parameter combinations
- Ranks by Sharpe ratio
- Exports full results

**Decision Rule:**
- ✅ **Accept** if best params are stable (not extreme)
- ❌ **Reject** if best params are extreme (overfitting)

## Interpreting Results

### Performance Metrics

| Metric | Good | Excellent | Reject |
|--------|------|-----------|--------|
| Sharpe Ratio | > 2.0 | > 3.0 | < 1.5 |
| Max Drawdown | < 10% | < 5% | > 15% |
| Win Rate | > 60% | > 70% | < 50% |
| Profit Factor | > 1.5 | > 2.0 | < 1.2 |

### Walk-Forward Analysis

| Success Rate | Interpretation |
|--------------|----------------|
| > 80% | Excellent - Very robust |
| 70-80% | Good - Acceptable for live |
| 60-70% | Marginal - Review carefully |
| < 60% | Poor - Reject strategy |

### Red Flags 🚩

**Reject strategy if:**
- Sharpe < 1.5
- Max Drawdown > 15%
- Walk-forward success < 60%
- Large variance across iterations
- Overfitting (extreme best parameters)

## Example Workflow

### Step 1: Initial Testing
```bash
# Quick test
python run_optimal_stopping.py
```

**Check:** Does it look promising? (Sharpe > 1.5, reasonable trades)

### Step 2: Comprehensive Analysis
```bash
# Full analysis
python run_realistic_backtest.py \
    --strategy optimal_stopping \
    --mode all
```

**Check:**
- ✅ Sharpe > 2.0?
- ✅ Max DD < 10%?
- ✅ Walk-forward success > 70%?

### Step 3: Parameter Optimization
```bash
# Find best parameters
python run_realistic_backtest.py \
    --strategy optimal_stopping \
    --mode optimize
```

**Check:**
- ✅ Best params are stable?
- ✅ Not extreme values?

### Step 4: Final Validation
```bash
# Test with optimized parameters
python run_realistic_backtest.py \
    --strategy optimal_stopping \
    --mode walk-forward
```

**Check:**
- ✅ Still performs well with optimized params?
- ✅ Success rate maintained?

### Step 5: Live Testing
- Start with **dry-run mode**
- Monitor performance vs backtest
- Adjust if needed

## Output Files

All results saved to `newtest/results/backtests/`:

```
{strategy}_report.txt          # Detailed performance report
{strategy}_trades.csv           # All trades (entry/exit/pnl)
{strategy}_walkforward.json     # Walk-forward results
{strategy}_optimization.csv     # Parameter optimization results
```

## Comparing Strategies

1. Run comprehensive backtest on all strategies
2. Compare key metrics:
   - Sharpe Ratio (primary)
   - Max Drawdown (risk)
   - Walk-forward success rate (robustness)
3. Select top 3 based on:
   - All metrics meet minimum requirements
   - Best risk-adjusted returns
   - Most consistent performance

## Best Practices

1. **Always use walk-forward analysis** - Single backtest can overfit
2. **Test multiple date ranges** - Different market conditions
3. **Optimize parameters carefully** - Avoid overfitting
4. **Compare strategies fairly** - Same date range, same assumptions
5. **Start with dry-run** - Validate live before risking capital

## Realistic Assumptions

- **Slippage**: 1 tick per contract (can adjust)
- **Commission**: $2.50 per contract (Topstep standard)
- **Fill Logic**: Conservative (next bar open for market orders)
- **Data**: Real TopstepX historical data

## Questions?

- **"Is this strategy good enough?"** → Check minimum requirements
- **"Will it work live?"** → Check walk-forward success rate
- **"What parameters to use?"** → Run optimization, pick stable ones
- **"Should I implement this?"** → Only if all criteria met

## Next Steps

1. Run comprehensive backtests on all strategies
2. Compare results side-by-side
3. Select top performers
4. Test in dry-run mode
5. Monitor live performance

