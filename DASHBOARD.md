# Grid Strategy Dashboard

## Overview

The Grid Strategy Dashboard is a real-time web interface for monitoring and controlling your trading strategy. It provides:

- **Real-time Metrics**: Live P&L, positions, exposure, and risk metrics
- **Visual Monitoring**: Color-coded status indicators and position tables
- **Manual Controls**: Pause/resume trading and emergency flatten
- **Auto-refresh**: Updates every 2 seconds automatically

## Accessing the Dashboard

1. Start the trading strategy:
   ```bash
   python3.11 main.py
   ```

2. Open your web browser and navigate to:
   ```
   http://localhost:8000
   ```

The dashboard will automatically load and start displaying real-time data.

## Features

### Real-time Metrics

- **Daily P&L**: Today's profit/loss (green if positive, red if negative)
- **Total P&L**: Realized + unrealized P&L
- **Net Exposure**: Current position value in dollars
- **Open Orders**: Number of active limit orders
- **Drawdown**: Current trailing drawdown
- **Current Price**: Latest market price for primary instrument

### Status Indicators

- **Running** (green): Strategy is active and trading
- **Paused** (orange): Strategy is paused (not placing new orders)
- **Stopped** (red): Strategy has stopped

### Controls

- **Pause**: Temporarily pause the strategy (stops placing new orders)
- **Resume**: Resume a paused strategy
- **Emergency Flatten**: Immediately close all positions and cancel all orders

### Positions Table

Shows all open positions with:
- Symbol (e.g., MES, MNQ)
- Quantity (number of contracts)
- Side (LONG/SHORT/FLAT) with color coding

## API Endpoints

The dashboard uses these REST API endpoints:

- `GET /ops/status` - Get current strategy status
- `POST /ops/pause` - Pause/resume strategy
- `POST /ops/flatten` - Emergency flatten positions
- `GET /ops/metrics` - Get detailed metrics

You can also access these endpoints directly via curl or any HTTP client:

```bash
# Get status
curl http://localhost:8000/ops/status

# Pause strategy
curl -X POST http://localhost:8000/ops/pause \
  -H "Content-Type: application/json" \
  -d '{"pause": true}'

# Flatten positions
curl -X POST http://localhost:8000/ops/flatten \
  -H "Content-Type: application/json" \
  -d '{"reason": "Manual flatten"}'
```

## Troubleshooting

### Dashboard not loading

1. Check that the strategy is running: `python3.11 main.py`
2. Verify the API is accessible: `curl http://localhost:8000/health`
3. Check browser console for errors (F12)

### No data showing

- Ensure the strategy has initialized successfully
- Check that you're connected to TopstepX API
- Verify account selection completed

### CORS errors

If accessing from a different domain, update CORS settings in `src/api/ops.py`:

```python
allow_origins=["http://your-domain.com"]  # Instead of ["*"]
```

## Security Note

⚠️ **Important**: The dashboard currently allows CORS from any origin (`*`). For production use, restrict this to your specific domain in `src/api/ops.py`.

## Future Enhancements

Potential additions:
- Price chart with grid levels visualization
- Trade history table
- Historical P&L charts
- Risk limit indicators
- ATR and volatility displays
- Hedge ratio visualization

