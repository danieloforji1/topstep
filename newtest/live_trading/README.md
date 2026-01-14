# Live Trading System

Production-ready execution system for running optimized strategies on TopstepX practice account.

## Strategies Included

1. **Optimal Stopping Strategy** - Uses optimal stopping theory for entry/exit timing
2. **Multi-Timeframe Convergence** - Trades when multiple timeframes agree
3. **Liquidity Provision** - Smart market making with adverse selection protection

All strategies are fully optimized and production-ready.

## Setup

### 1. Environment Variables

Create a `.env` file in the project root:

```bash
TOPSTEPX_USERNAME=your_username
TOPSTEPX_API_KEY=your_api_key
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Required packages:
- `signalrcore` (for real-time data)
- `pandas`
- `python-dotenv`

## Usage

### Dry Run Mode (Recommended First)

Test the system without placing real orders:

```bash
cd topstep/newtest
python3.11 live_trading/live_strategy_executor.py \
    --symbol MES \
    --dry-run \
    --strategies optimal_stopping multi_timeframe liquidity_provision
```

### Live Trading Mode

**⚠️ WARNING: This will place REAL orders on your TopstepX account!**

```bash
python3.11 live_trading/live_strategy_executor.py \
    --symbol MES \
    --live \
    --strategies optimal_stopping multi_timeframe liquidity_provision
```

### Options

- `--symbol`: Trading symbol (default: MES)
- `--contract-id`: Contract ID (auto-discovered if not provided)
- `--dry-run`: Dry run mode (default: True)
- `--live`: Live trading mode (overrides dry-run)
- `--strategies`: Strategies to run (default: all three)

### Run Individual Strategies

Run only Optimal Stopping:
```bash
python3.11 live_trading/live_strategy_executor.py --symbol MES --dry-run --strategies optimal_stopping
```

Run only Multi-Timeframe:
```bash
python3.11 live_trading/live_strategy_executor.py --symbol MES --dry-run --strategies multi_timeframe
```

Run only Liquidity Provision:
```bash
python3.11 live_trading/live_strategy_executor.py --symbol MES --dry-run --strategies liquidity_provision
```

## How It Works

1. **Connection**: Connects to TopstepX API and authenticates
2. **Data Feed**: Subscribes to real-time market data via SignalR
3. **Strategy Execution**: Each strategy runs independently:
   - Monitors market data
   - Generates signals based on optimized parameters
   - Places orders when signals are generated
   - Manages exits (stop loss, take profit, trailing stops)
4. **Risk Management**: 
   - Position sizing based on risk per trade
   - Stop losses on all positions
   - Maximum position limits
5. **Monitoring**: Logs all trades, signals, and performance metrics

## Strategy Configurations

All strategies use optimized parameters from backtesting:

### Optimal Stopping
- Score threshold: 0.7
- ATR stop: 0.75x
- ATR target: 1.25x
- Trailing stops: Enabled
- Partial profit: Enabled

### Multi-Timeframe
- Convergence threshold: 0.2
- Divergence threshold: 0.2
- ATR stop: 1.0x
- ATR target: 1.5x
- Timeframes: 1m, 5m, 15m
- Trailing stops: Enabled
- Partial profit: Enabled

### Liquidity Provision
- Imbalance threshold: 0.08
- Adverse selection threshold: 0.55
- Favorable fill threshold: 0.55
- Spread target: 4 ticks
- ATR stop: 1.25x
- Trailing stops: Enabled
- Partial profit: Enabled
- Confidence scaling: Enabled

## Monitoring

### Logs

All activity is logged to:
- Console output
- `live_trading.log` file

### Key Metrics Tracked

- Entry/exit signals
- Order placement and fills
- Position P&L
- Risk metrics
- Strategy performance

## Safety Features

1. **Dry Run Mode**: Test without real orders
2. **Confirmation Required**: Live mode requires explicit confirmation
3. **Stop Losses**: All positions have stop losses
4. **Position Limits**: Maximum position size limits
5. **Risk Per Trade**: Fixed dollar risk per trade ($100 default)

## Troubleshooting

### Authentication Failed
- Check `.env` file has correct credentials
- Verify API key is valid
- Check account is active

### No Market Data
- Verify contract ID is correct
- Check SignalR connection
- Ensure market is open

### No Signals Generated
- Check historical data is loaded
- Verify strategy thresholds
- Check market conditions match strategy requirements

## Best Practices

1. **Start with Dry Run**: Always test in dry-run mode first
2. **Monitor Closely**: Watch logs and positions
3. **Start Small**: Begin with one strategy
4. **Gradual Rollout**: Add strategies one at a time
5. **Set Limits**: Use account-level risk limits
6. **Regular Review**: Review performance daily

## Performance Expectations

Based on backtesting:

- **Optimal Stopping**: Sharpe 1.5+, Win Rate 60%+
- **Multi-Timeframe**: Sharpe 1.64, Win Rate 60%+
- **Liquidity Provision**: Sharpe 1.26, Win Rate 50%+ (but larger wins)

**Note**: Live performance may differ from backtesting due to:
- Slippage
- Execution delays
- Market conditions
- Real-time data quality

## Support

For issues or questions:
1. Check logs in `live_trading.log`
2. Review strategy status in console output
3. Verify API connection and credentials

## Next Steps

1. Run in dry-run mode for 24-48 hours
2. Monitor signal generation and order execution
3. Verify risk management is working
4. Gradually enable live trading
5. Monitor performance and adjust as needed


