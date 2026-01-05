# Statistical Arbitrage Strategy - Production Guide

This document explains how to run the Statistical Arbitrage (StatArb) Strategy for GC/MGC Gold Futures separately from the Grid Strategy and Asian Range Strategy.

## Overview

The Statistical Arbitrage Strategy is a **completely separate** production implementation that:
- Trades **GC (Full Gold Futures)** and **MGC (Micro Gold Futures)** - spread trading
- Runs on **port 8002** (Grid Strategy uses port 8000, Asian Range uses port 8001)
- Uses its own **config file** (`statarb_config.yaml`)
- Has its own **log file** (`statarb_trading.log`)
- Operates **independently** - no interference with other strategies

## Strategy Logic

1. **Spread Calculation**: `spread = GC_price - beta * MGC_price`
   - Beta (hedge ratio) is calculated dynamically using linear regression
   - For GC/MGC, beta should be ~1.0 (both prices are per ounce)

2. **Z-Score Normalization**: `zscore = (spread - mean(spread)) / std(spread)`
   - Uses rolling window (default: 1440 periods = 1 day of 1-minute bars)

3. **Entry Signals**:
   - **Long Spread**: When z-score < -2.0 (spread is too low, expect it to rise)
     - Buy GC, Sell MGC
   - **Short Spread**: When z-score > +2.0 (spread is too high, expect it to fall)
     - Sell GC, Buy MGC

4. **Exit Signals**:
   - **Z-Score Exit**: When |z-score| < 0.6 (spread has reverted to mean)
   - **Stop Loss**: When spread moves against position by 3.0 standard deviations
   - **Time Stop**: Force exit after 2 hours regardless of z-score
   - **Zero Cross**: Exit if z-score crosses zero (spread crossed mean)

5. **Position Sizing**:
   - Fixed ratio: 1 GC contract : 10 MGC contracts
   - Risk-based sizing: Targets $100 risk per trade
   - Maximum: 10 GC contracts per position

## Configuration

Edit `statarb_config.yaml`:

```yaml
# Trading Instruments
instrument_a: GC   # Full Gold Futures
instrument_b: MGC  # Micro Gold Futures

# Strategy Parameters
z_entry: 2.0       # Z-score entry threshold
z_exit: 0.6        # Z-score exit threshold

# Risk Management
spread_stop_std: 3.0      # Stop loss in standard deviations
time_stop_hours: 2.0      # Maximum hold time
risk_per_trade: 100.0     # Dollar risk per trade
max_daily_loss: 900
trailing_drawdown_limit: 1800

# Trading Mode
dry_run: true      # Set to false for live trading
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
python statarb_main.py
```

### Running All Strategies Simultaneously

You can run all three strategies at the same time:

**Terminal 1 - Grid Strategy:**
```bash
python main.py
# API available at http://localhost:8000
```

**Terminal 2 - Asian Range Strategy:**
```bash
python asian_range_main.py
# API available at http://localhost:8001
```

**Terminal 3 - StatArb Strategy:**
```bash
python statarb_main.py
# API available at http://localhost:8002
```

They are **completely independent**:
- Different instruments (MES/MNQ vs MGC vs GC/MGC)
- Different ports (8000 vs 8001 vs 8002)
- Different config files
- Different log files
- No shared state

## API Endpoints

The StatArb Strategy exposes the same API endpoints on port 8002:

- `GET /ops/status` - Get strategy status
- `POST /ops/flatten` - Emergency flatten positions
- `POST /ops/pause` - Pause/resume strategy
- `GET /ops/metrics` - Get metrics
- `GET /health` - Health check

Example:
```bash
curl http://localhost:8002/ops/status
```

## Monitoring

- **Log File**: `statarb_trading.log`
- **API Dashboard**: `http://localhost:8002` (if static files are available)
- **Status Endpoint**: `http://localhost:8002/ops/status`

## Key Differences from Other Strategies

| Feature | Grid Strategy | Asian Range Strategy | StatArb Strategy |
|---------|--------------|---------------------|------------------|
| Instrument | MES/MNQ | MGC | GC/MGC |
| Port | 8000 | 8001 | 8002 |
| Config | `config.yaml` | `asian_range_config.yaml` | `statarb_config.yaml` |
| Log File | `trading.log` | `asian_range_trading.log` | `statarb_trading.log` |
| Strategy Type | Grid with hedging | Asian Range Breakout | Statistical Arbitrage |
| Entry Method | Limit orders on grid | Stop orders on range break | Z-score based spread entry |

## Safety Features

1. **Dry Run Mode**: Default is `true` - no real orders until explicitly enabled
2. **Risk Limits**: Daily loss and drawdown limits enforced
3. **Position Limits**: Maximum net exposure cap
4. **Time Stops**: Automatic exit after maximum hold time
5. **Stop Losses**: Spread-based stop loss protection

## How It Works

1. **Data Collection**: Continuously receives real-time price feeds for both GC and MGC
2. **Spread Calculation**: Calculates spread using dynamically updated beta
3. **Z-Score Monitoring**: Normalizes spread to z-score using rolling statistics
4. **Signal Generation**: Enters when z-score exceeds threshold, exits when it reverts
5. **Position Management**: Tracks both legs of spread position simultaneously
6. **Risk Management**: Monitors for stop loss, time stop, and account-level limits

## Troubleshooting

### Strategy Not Entering Trades

- Check that both GC and MGC prices are updating (check logs)
- Verify sufficient historical data is loaded (need at least 100 bars)
- Check z-score thresholds (may need to adjust `z_entry` if too strict)
- Ensure `dry_run` is set appropriately

### Orders Not Filling

- Limit orders may not fill if price moves away
- Consider using market orders for immediate execution (modify `_enter_spread_position`)
- Check order placement logs for errors

### Extreme Z-Scores

- Ensure beta calculation is working correctly (should be ~1.0 for GC/MGC)
- Check that spread history is being calculated with consistent beta
- Verify data alignment between GC and MGC

## Next Steps

1. **Start in Dry Run**: Run with `dry_run: true` to observe behavior
2. **Monitor Performance**: Watch logs and API status for several days
3. **Adjust Parameters**: Fine-tune z-entry, z-exit, stop loss based on observations
4. **Enable Live Trading**: Set `dry_run: false` when ready (with caution!)

