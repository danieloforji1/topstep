# Asian Range Breakout Strategy - Production Guide

This document explains how to run the Asian Range Breakout Strategy for MGC Gold Futures separately from the Grid Strategy.

## Overview

The Asian Range Breakout Strategy is a **completely separate** production implementation that:
- Trades **MGC (Micro Gold Futures)** - different from the Grid Strategy's MES/MNQ
- Runs on **port 8001** (Grid Strategy uses port 8000)
- Uses its own **config file** (`asian_range_config.yaml`)
- Has its own **log file** (`asian_range_trading.log`)
- Operates **independently** - no interference with Grid Strategy

## Strategy Logic

1. **Asian Session Range Calculation** (8 PM - 2 AM ET)
   - Calculates high and low during Asian session
   - Range = High - Low

2. **Order Placement** (3 AM ET - London Open)
   - Places OCO (One-Cancels-Other) orders:
     - **Buy Stop**: Asian High + 1 tick
     - **Sell Stop**: Asian Low - 1 tick

3. **Entry**
   - When price breaks above Asian High → Enter LONG
   - When price breaks below Asian Low → Enter SHORT

4. **Stop Loss**
   - Long: Below Asian Low (with buffer to avoid wicks)
   - Short: Above Asian High (with buffer to avoid wicks)

5. **Take Profit**
   - 1.5x Asian Range size (configurable)

6. **Break-Even Rule**
   - When trade reaches +1R profit (1x range size), move stop to entry price

7. **Partial Close** (12 PM ET)
   - Close 75% of position (configurable)
   - Move stop for remaining 25% to 50% profit level

8. **End of Day Exit** (5 PM ET)
   - Close any remaining positions

## Configuration

Edit `asian_range_config.yaml`:

```yaml
# Strategy Parameters
contracts_per_trade: 5        # Fixed number of contracts
tp_multiplier: 1.5            # Take profit multiplier
sl_buffer_ticks: 1           # Stop loss buffer
partial_close_percent: 0.75   # Percentage to close at 12 PM

# Risk Management
max_daily_loss: 900
trailing_drawdown_limit: 1800
max_net_notional: 5000

# Trading Mode
dry_run: true                 # Set to false for live trading
```

## Running the Strategy

### Prerequisites

1. Set environment variables:
   ```bash
   export TOPSTEPX_USERNAME=your_username
   export TOPSTEPX_API_KEY=your_api_key
   ```

2. Or use a `.env` file:
   ```
   TOPSTEPX_USERNAME=your_username
   TOPSTEPX_API_KEY=your_api_key
   ```

### Start the Strategy

```bash
python asian_range_main.py
```

### Running Both Strategies Simultaneously

You can run both strategies at the same time:

**Terminal 1 - Grid Strategy:**
```bash
python3.11 main.py
# API available at http://localhost:8000
```

**Terminal 2 - Asian Range Strategy:**
```bash
python3.11 asian_range_main.py
# API available at http://localhost:8001
```

They are **completely independent**:
- Different instruments (MES/MNQ vs MGC)
- Different ports (8000 vs 8001)
- Different config files
- Different log files
- No shared state

## API Endpoints

The Asian Range Strategy exposes the same API endpoints on port 8001:

- `GET /ops/status` - Get strategy status
- `POST /ops/flatten` - Emergency flatten positions
- `POST /ops/pause` - Pause/resume strategy
- `GET /ops/metrics` - Get metrics
- `GET /health` - Health check

Example:
```bash
curl http://localhost:8001/ops/status
```

## Monitoring

- **Log File**: `asian_range_trading.log`
- **API Dashboard**: `http://localhost:8001` (if static files are available)
- **Status Endpoint**: `http://localhost:8001/ops/status`

## Key Differences from Grid Strategy

| Feature | Grid Strategy | Asian Range Strategy |
|---------|--------------|---------------------|
| Instrument | MES/MNQ | MGC |
| Port | 8000 | 8001 |
| Config | `config.yaml` | `asian_range_config.yaml` |
| Log File | `trading.log` | `asian_range_trading.log` |
| Strategy Type | Grid with hedging | Asian Range Breakout |
| Entry Method | Limit orders on grid | Stop orders on range break |

## Safety Features

1. **Dry Run Mode**: Default is `true` - no real orders until explicitly enabled
2. **Risk Limits**: Daily loss and drawdown limits enforced
3. **Emergency Flatten**: Can be triggered via API or automatically on risk limits
4. **End of Day Exit**: All positions closed by 5 PM ET

## Troubleshooting

### Strategy not placing orders
- Check if it's 3 AM ET (London open time)
- Verify Asian range was calculated (check logs)
- Ensure `dry_run: false` if you want live trading

### No Asian range calculated
- Check if historical data is available
- Verify timezone is set correctly (ET)
- Check logs for data loading errors

### Positions not closing
- Check if it's past 5 PM ET (end of day)
- Verify API connection is active
- Check risk manager status

## Next Steps

1. Test in dry-run mode first
2. Monitor logs and API status
3. Adjust parameters in config file
4. Enable live trading when ready (`dry_run: false`)

