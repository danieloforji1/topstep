# Configuration Guide for MGC Backtest

## Quick Start

1. **Edit the config file**: Open `backtest_config.yaml` and modify the parameters you want to test
2. **Run the backtest**: `python run_mgc_backtest.py`
3. **Compare results**: Check the output in `test/results/` directory

## Parameter Tuning Guide

### Risk:Reward Ratio (`risk_reward`)

**What it does**: Determines how far your take profit is compared to your stop loss.

- **Lower (1.0-1.2)**: More winners, smaller wins, easier to hit TP
- **Higher (2.0-3.0)**: Fewer winners, bigger wins, harder to hit TP
- **Default**: 1.5 (balanced)

**Example**: If stop loss is $10 away, with R:R of 1.5, take profit is $15 away.

### Pivot Length (`pivot_length`)

**What it does**: How many candles on each side needed to confirm a swing high/low.

- **Lower (3-4)**: More sensitive, detects more swings (more trades, more false signals)
- **Higher (6-7)**: Less sensitive, only major swings (fewer trades, higher quality)
- **Default**: 5 (balanced)

### Risk Per Trade (`risk_per_trade`)

**What it does**: Fixed dollar amount you're willing to lose per trade.

- **Lower ($25-40)**: Smaller positions, less risk per trade
- **Higher ($75-100)**: Larger positions, more risk per trade
- **Default**: $50

**Note**: Position size is automatically calculated based on stop distance to maintain this risk.

### ATR Period (`atr_period`)

**What it does**: Number of periods used to calculate Average True Range (volatility measure).

- **Lower (3-4)**: More responsive to recent volatility
- **Higher (7-10)**: Smoother, less reactive to short-term volatility
- **Default**: 5

### ATR Multiplier (`atr_multiplier`)

**What it does**: Multiplies ATR to determine minimum stop distance.

- **Lower (3.0-4.0)**: Tighter stops, more risk of getting stopped out
- **Higher (6.0-7.0)**: Wider stops, less risk of getting stopped out
- **Default**: 5.0

**Formula**: Minimum stop = ATR × multiplier

### EMA Period (`ema_period`)

**What it does**: Period for the Exponential Moving Average used as trend filter.

- **Lower (30-40)**: More sensitive to trend changes, more trades
- **Higher (60-70)**: Less sensitive, only strong trends, fewer trades
- **Default**: 50

### Confirmation Body Ratio (`confirmation_body_ratio`)

**What it does**: Minimum percentage of candle body vs total range for confirmation.

- **Lower (0.4-0.45)**: Easier to get confirmation, more trades
- **Higher (0.55-0.6)**: Stricter confirmation, fewer but higher quality trades
- **Default**: 0.5 (50%)

## Optimization Strategy

### Step 1: Baseline
Run with default settings to establish baseline performance.

### Step 2: Test One Parameter at a Time
Change one parameter, run backtest, compare results. Keep what works better.

### Step 3: Common Optimizations

**If win rate is too low (<50%)**:
- Increase `confirmation_body_ratio` (stricter entries)
- Increase `pivot_length` (better swing quality)
- Increase `atr_multiplier` (wider stops, less false breakouts)

**If profit factor is too low (<1.3)**:
- Increase `risk_reward` (bigger wins)
- Decrease `confirmation_body_ratio` (more trades)
- Adjust `atr_multiplier` to reduce average loss size

**If drawdown is too high (>15%)**:
- Decrease `risk_per_trade` (smaller positions)
- Increase `atr_multiplier` (wider stops)
- Increase `confirmation_body_ratio` (fewer but better trades)

**If too few trades (<20 per year)**:
- Decrease `pivot_length` (more swings detected)
- Decrease `confirmation_body_ratio` (easier confirmation)
- Decrease `ema_period` (more trend changes)

## Example Configurations

### Conservative (Lower Risk, Fewer Trades)
```yaml
risk_reward: 2.0
pivot_length: 7
risk_per_trade: 40.0
atr_multiplier: 6.0
confirmation_body_ratio: 0.55
```

### Aggressive (Higher Risk, More Trades)
```yaml
risk_reward: 1.2
pivot_length: 3
risk_per_trade: 75.0
atr_multiplier: 4.0
confirmation_body_ratio: 0.45
```

### Balanced (Default)
```yaml
risk_reward: 1.5
pivot_length: 5
risk_per_trade: 50.0
atr_multiplier: 5.0
confirmation_body_ratio: 0.5
```

## Tips

1. **Use cached data**: Set `use_cached_data: true` after first run to speed up testing
2. **Compare results**: Save reports with different names to compare
3. **Test on different date ranges**: Some parameters work better in different market conditions
4. **Watch for over-optimization**: If parameters are too specific, they may not work in live trading
5. **Focus on profit factor and drawdown**: These are key metrics for strategy viability

## Output Files

After each run, check:
- `backtest_report_*.txt` - Full performance report
- `trades_*.csv` - All individual trades
- `equity_curve_*.png` - Visual equity curve (if matplotlib available)
- `monthly_returns_*.png` - Monthly P&L breakdown (if matplotlib available)

