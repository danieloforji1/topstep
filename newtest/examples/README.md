# Backtest Examples - Production Grade

These examples provide **realistic, production-grade backtesting** with comprehensive analysis for making informed trading decisions.

## Features

### ✅ Realistic Execution Modeling
- **Slippage**: 1 tick for market orders (configurable)
- **Commission**: $2.50 per contract round trip (Topstep standard)
- **Fill Logic**: Realistic order execution based on bar high/low
- **Partial Fills**: Not modeled (assumes full fills)

### ✅ Comprehensive Analysis
- **Walk-Forward Analysis**: Train on one period, test on another
- **Parameter Optimization**: Grid search for best parameters
- **Multiple Date Ranges**: Test across different market conditions
- **Detailed Reporting**: Trade-by-trade, monthly, drawdown analysis

### ✅ Detailed Metrics
- **Performance**: Sharpe, Sortino, Calmar ratios
- **Risk**: Max drawdown, VaR, drawdown periods
- **Trade Analysis**: Win rate, profit factor, expectancy
- **Consistency**: Daily returns, consistency score

## Usage

### 1. Single Backtest (Quick Test)

```bash
cd newtest/examples
python run_realistic_backtest.py \
    --strategy optimal_stopping \
    --symbol MES \
    --interval 15m \
    --start-date 2025-01-01 \
    --end-date 2025-11-14 \
    --mode single
```

**Output:**
- Detailed performance report
- Trade-by-trade CSV export
- Equity curve data

### 2. Walk-Forward Analysis (Recommended)

Tests strategy robustness across multiple time periods:

```bash
python run_realistic_backtest.py \
    --strategy optimal_stopping \
    --symbol MES \
    --interval 15m \
    --start-date 2025-01-01 \
    --end-date 2025-11-14 \
    --mode walk-forward
```

**What it does:**
- Splits data into train/test periods (60/40)
- Steps forward monthly
- Tests on out-of-sample data
- Reports average performance across all iterations

**Output:**
- Average Sharpe ratio across iterations
- Success rate (positive vs negative periods)
- Performance stability metrics

### 3. Parameter Optimization

Finds best parameter combinations:

```bash
python run_realistic_backtest.py \
    --strategy optimal_stopping \
    --symbol MES \
    --interval 15m \
    --start-date 2025-01-01 \
    --end-date 2025-11-14 \
    --mode optimize
```

**What it does:**
- Tests all parameter combinations
- Ranks by Sharpe ratio
- Exports full results to CSV

**Output:**
- Top 10 parameter combinations
- Full optimization results CSV

### 4. All Analysis (Complete)

Runs all three modes:

```bash
python run_realistic_backtest.py \
    --strategy optimal_stopping \
    --symbol MES \
    --interval 15m \
    --start-date 2025-01-01 \
    --end-date 2025-11-14 \
    --mode all
```

## Output Files

All results are saved to `newtest/results/backtests/`:

- `{strategy}_report.txt` - Detailed performance report
- `{strategy}_trades.csv` - All trades with details
- `{strategy}_walkforward.json` - Walk-forward results
- `{strategy}_optimization.csv` - Parameter optimization results

## Interpreting Results

### For Live Trading Decisions

**Minimum Requirements:**
- ✅ Sharpe Ratio > 2.0 (risk-adjusted returns)
- ✅ Max Drawdown < 10% (risk control)
- ✅ Win Rate > 60% (consistency)
- ✅ Positive walk-forward results (robustness)

**Walk-Forward Success Rate:**
- >70% positive iterations = Good
- >80% positive iterations = Excellent
- <60% positive iterations = Reject strategy

**Parameter Optimization:**
- Look for stable parameters (similar performance across combinations)
- Avoid overfitting (best params shouldn't be extreme)
- Prefer parameters with consistent Sharpe > 2.0

### Red Flags

❌ **Reject if:**
- Sharpe < 1.5
- Max Drawdown > 15%
- Walk-forward success rate < 60%
- Large performance variance across iterations
- Overfitting (best params are extreme values)

## Realistic Assumptions

### Execution
- Market orders: Fill at next bar's open (worst case)
- Limit orders: Fill if price touches limit during bar
- Slippage: 1 tick per contract (configurable)
- Commission: $2.50 per contract round trip

### Data
- Uses actual TopstepX historical data
- Caches data to avoid repeated API calls
- Handles missing data gracefully

### Risk Management
- Position sizing based on risk per trade
- Stop loss and take profit enforcement
- Maximum position limits

## Example Output

```
================================================================================
COMPREHENSIVE BACKTEST REPORT: OPTIMAL_STOPPING
================================================================================

PERFORMANCE SUMMARY
--------------------------------------------------------------------------------
Initial Equity: $50,000.00
Final Equity: $62,450.00
Total Return: 24.90%
Annualized Return: 28.50%
Total P&L: $12,450.00

RISK METRICS
--------------------------------------------------------------------------------
Sharpe Ratio: 2.45
Sortino Ratio: 3.12
Calmar Ratio: 4.15
Maximum Drawdown: 6.87% ($3,435.00)

TRADE STATISTICS
--------------------------------------------------------------------------------
Total Trades: 127
Winning Trades: 82 (64.57%)
Losing Trades: 45 (35.43%)
Profit Factor: 2.18
Expectancy: $98.03 per trade

MONTHLY PERFORMANCE
--------------------------------------------------------------------------------
2025-01: $1,250.00
2025-02: $890.00
2025-03: $1,450.00
...
```

## Next Steps

1. **Run backtests** on all strategies
2. **Compare results** using walk-forward analysis
3. **Select top 3** based on:
   - Sharpe > 2.0
   - Max DD < 10%
   - Walk-forward success > 70%
4. **Optimize parameters** for selected strategies
5. **Test in dry-run** before live trading

## Notes

- Backtests use **realistic assumptions** but cannot predict future performance
- Always test in **dry-run mode** before live trading
- Monitor **live performance** vs backtest expectations
- Adjust parameters if **live results differ significantly**

