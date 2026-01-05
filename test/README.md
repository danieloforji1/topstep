# MGC Liquidity Sweep Strategy Backtest

This directory contains a complete backtest implementation for the MGC Gold Futures Liquidity Sweep Strategy as specified in `arbts.txt`.

## Files

- **`arbts.txt`** - Strategy specification document
- **`mgc_backtest_data.py`** - Data fetcher for TopstepX API
- **`mgc_strategy_logic.py`** - Core strategy logic (swing detection, liquidity sweeps, confirmation candles)
- **`mgc_backtest_engine.py`** - Main backtest engine
- **`mgc_performance.py`** - Performance analysis and reporting
- **`run_mgc_backtest.py`** - Main script to run the backtest

## Setup

1. Ensure you have the required dependencies:
   ```bash
   pip install pandas numpy matplotlib
   ```

2. Set up your TopstepX credentials in `.env`:
   ```
   TOPSTEPX_USERNAME=your_username
   TOPSTEPX_API_KEY=your_api_key
   ```

## Usage

### Basic Usage

Run the backtest with settings from `backtest_config.yaml`:

```bash
cd test
python run_mgc_backtest.py
```

### Using Config File

Edit `backtest_config.yaml` to change parameters, then run:

```bash
python run_mgc_backtest.py
```

### Override Config with Command Line

You can override config file settings with command-line arguments:

```bash
python run_mgc_backtest.py \
    --start-date 2025-01-01 \
    --end-date 2025-11-14 \
    --risk-reward 2.0 \
    --pivot-length 7 \
    --risk-per-trade 75.0
```

### Configuration Options

Edit `backtest_config.yaml` to adjust:

- **Date Range**: `start_date`, `end_date`
- **Strategy Parameters**: 
  - `risk_reward`: Risk:Reward ratio (1.5 = take profit is 1.5x stop distance)
  - `pivot_length`: Swing detection sensitivity (3-7 typical)
- **Risk Management**:
  - `risk_per_trade`: Dollar amount to risk per trade ($50 default)
  - `atr_period`: ATR calculation period (5 default)
  - `atr_multiplier`: ATR multiplier for stop sizing (5.0 default)
- **Trend Filter**:
  - `ema_period`: EMA period for trend filter (50 default)
- **Entry/Exit**:
  - `confirmation_body_ratio`: Minimum body ratio for confirmation candle (0.5 = 50%)

### Command Line Options

All config file parameters can be overridden:
- `--start-date`, `--end-date`: Date range
- `--risk-reward`: Risk:Reward ratio
- `--pivot-length`: Pivot length
- `--risk-per-trade`: Risk per trade in dollars
- `--atr-period`, `--atr-multiplier`: ATR settings
- `--ema-period`: EMA period
- `--confirmation-body-ratio`: Confirmation candle threshold
- `--use-cached`: Use cached data
- `--cache-dir`, `--output-dir`: Directory paths
- `--config`: Path to custom config file

## Output

The backtest generates:

1. **Console Report** - Detailed performance metrics printed to console
2. **Text Report** - Saved to `test/results/backtest_report_*.txt`
3. **Trades CSV** - All trades exported to `test/results/trades_*.csv`
4. **Equity Curve Plot** - Saved to `test/results/equity_curve_*.png`
5. **Monthly Returns Plot** - Saved to `test/results/monthly_returns_*.png`

## Strategy Overview

The strategy implements:

1. **Trend Filter**: 50 EMA on 15-minute timeframe
2. **Liquidity Sweep Detection**: Identifies when price sweeps previous swing highs/lows
3. **Confirmation Candle**: Waits for strong momentum candle (>50% body) after sweep
4. **Entry**: On break of confirmation candle high/low
5. **Exit**: 
   - Take profit at 1.5x risk
   - Stop loss below/above sweep
   - Exit at opposite liquidity zone
6. **Position Sizing**: $50 risk per trade, calculated based on ATR and stop distance

## Performance Metrics

The backtest calculates:

- Total trades and win rate
- Total P&L and profit factor
- Maximum drawdown
- Average risk:reward ratio
- Expectancy per trade
- Monthly returns breakdown
- Long vs short performance

## Notes

- Data is cached to avoid repeated API calls
- The strategy uses MGC micro gold futures contract specs
- Position sizing is based on $50 fixed risk per trade
- All calculations follow the specification in `arbts.txt`

