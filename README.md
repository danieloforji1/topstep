# Grid Strategy Trading System for TopstepX

A production-grade Python trading system implementing a volatility-adaptive grid strategy with cross-asset hedging for TopstepX futures trading.

## Overview

This system implements an automated intraday grid trading strategy that:
- Places limit orders in a grid pattern around the current price
- Adjusts grid spacing based on volatility (ATR-driven)
- Dynamically sizes positions based on risk
- Implements cross-asset hedging using correlated instruments (MES/MNQ)
- Enforces Topstep evaluation rules (daily loss limits, drawdown limits, no overnight positions)

## Architecture

```
src/
├── connectors/          # TopstepX API integration
│   ├── topstepx_client.py      # REST & WebSocket client
│   └── market_data_adapter.py  # Data normalization
├── strategy/            # Core strategy components
│   ├── grid_manager.py         # Grid level management
│   ├── hedge_manager.py        # Cross-asset hedging
│   ├── position_manager.py     # Position tracking
│   ├── risk_manager.py         # Risk limits enforcement
│   └── sizer.py                # Volatility-based lot sizing
├── execution/           # Order execution
│   ├── order_client.py         # Order placement with retry
│   └── fill_handler.py         # Fill processing
├── data/                # Data persistence
│   ├── timeseries_store.py     # Candle/tick storage
│   └── trade_history.py        # Trade & P&L history
├── indicators/           # Technical indicators
│   └── technical.py            # ATR, volatility, correlation
├── observability/       # Monitoring
│   └── metrics.py              # Prometheus metrics
└── api/                 # Operations API
    └── ops.py                  # Status, flatten, pause endpoints
```

## Features

- **Volatility-Adaptive Grid**: Grid spacing adjusts based on ATR
- **Dynamic Position Sizing**: Risk-based lot sizing ($150 risk per trade)
- **Cross-Asset Hedging**: Automatic hedging with correlated instruments
- **Risk Management**: Hard stops for daily loss, trailing drawdown, exposure caps
- **Real-time Monitoring**: Prometheus metrics, API endpoints for status
- **Dry-run Mode**: Test without live trading
- **Production Ready**: Logging, error handling, retry logic

## Installation

### Prerequisites

- Python 3.11+
- TopstepX account with API credentials
- Docker (optional, for containerized deployment)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd topstep
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment:
```bash
cp .env.example .env
# Edit .env with your TopstepX credentials
```

4. Configure strategy parameters:
```bash
# Edit config.yaml with your preferred settings
```

## Configuration

### Environment Variables (.env)

```bash
TOPSTEPX_USERNAME=your_username
TOPSTEPX_API_KEY=your_api_key
DRY_RUN=true  # Set to false for live trading
```

### Strategy Configuration (config.yaml)

Key parameters:
- `primary_instrument`: Primary trading instrument (default: MES)
- `hedge_instrument`: Hedge instrument (default: MNQ)
- `target_daily_PL`: Daily profit target (default: $500)
- `max_daily_loss`: Daily loss limit (default: $900)
- `grid_levels_each_side`: Number of grid levels (default: 5)
- `atr_multiplier_for_spacing`: Grid spacing multiplier (default: 0.45)
- `sizer_R_per_trade`: Risk per trade in dollars (default: $150)

See `config.yaml` for all available parameters.

## Usage

### Dry Run (Testing)

```bash
python main.py
```

The system will run in dry-run mode by default, simulating orders without executing them.

### Live Trading

1. Set `DRY_RUN=false` in `.env` or `config.yaml`
2. Ensure API credentials are correct
3. Review risk parameters
4. Run:
```bash
python main.py
```

### Docker Deployment

```bash
# Build image
docker build -t grid-strategy .

# Run container
docker run -d \
  --name grid-strategy \
  -e TOPSTEPX_USERNAME=your_username \
  -e TOPSTEPX_API_KEY=your_api_key \
  -e DRY_RUN=false \
  -p 8000:8000 \
  grid-strategy
```

## API Endpoints

The system exposes a REST API for monitoring and control:

- `GET /ops/status` - Get current strategy status
- `POST /ops/flatten` - Force flatten all positions
- `POST /ops/pause` - Pause/resume strategy
- `GET /ops/metrics` - Get strategy metrics
- `GET /health` - Health check

Example:
```bash
curl http://localhost:8000/ops/status
```

## Monitoring

### Prometheus Metrics

Metrics are exposed at `/metrics` (Prometheus format):
- `strategy_pnl_total` - Total P&L (realized, unrealized, total)
- `strategy_pnl_daily` - Daily P&L
- `strategy_drawdown` - Current drawdown
- `strategy_exposure` - Net exposure
- `strategy_orders_open` - Open orders count
- `strategy_atr` - Current ATR
- `strategy_correlation` - Hedge correlation
- `strategy_hedge_ratio` - Current hedge ratio

### Logs

Logs are written to:
- Console (stdout)
- File: `trading.log`

## Strategy Logic

### Grid Generation

1. Calculate ATR from historical candles
2. Grid spacing = ATR × multiplier (default 0.45)
3. Generate buy/sell levels around current price
4. Place limit orders at each level

### Position Sizing

- Base lot = Risk per trade ($150) / (ATR × tick value)
- Pyramiding: lot_n = base_lot × (1 + α × n)

### Hedging

- Activated when price moves >1.5× spacing from grid midpoint
- Hedge ratio = correlation × (primary_vol / hedge_vol)
- Hedge size = primary_position × hedge_ratio (opposite direction)

### Risk Management

Hard stops:
- Daily loss ≥ $900 → Emergency flatten
- Trailing drawdown ≥ $1,800 → Emergency flatten
- Exposure cap ≥ $1,200 → Stop opening new positions
- Session end → Auto-flatten (no overnight positions)

## Testing

Run unit tests:
```bash
pytest tests/
```

Run with coverage:
```bash
pytest --cov=src tests/
```

## Development

### Project Structure

Follows the architecture specified in `gridStrategy.txt`:
- Modular design with clear separation of concerns
- Each component is independently testable
- Production-ready error handling and logging

### Adding New Features

1. Follow existing module patterns
2. Add unit tests
3. Update documentation
4. Test in dry-run mode first

## Safety Features

- **Dry-run by default**: Must explicitly enable live trading
- **Multiple risk checks**: Daily loss, drawdown, exposure caps
- **Emergency flatten**: Automatic position closure on risk triggers
- **Idempotent orders**: Prevents duplicate orders
- **Retry logic**: Handles transient API errors
- **Session end protection**: Auto-flatten before market close

## TopstepX API Integration

The system uses the ProjectX Gateway API:
- Authentication: API key-based JWT tokens
- REST API: Market data, orders, positions
- WebSocket: Real-time updates (SignalR)
- Rate limiting: Exponential backoff on 429 errors

API Documentation: https://gateway.docs.projectx.com/docs/

## Troubleshooting

### Authentication Issues
- Verify API credentials in `.env`
- Check API key permissions
- Ensure account is active

### No Orders Placed
- Check dry-run mode (should see "[DRY RUN]" in logs)
- Verify contract IDs are found
- Check account has sufficient balance

### Risk Limits Triggered
- Review daily P&L
- Check drawdown from peak
- Adjust risk parameters if needed

## License

[Your License Here]

## Disclaimer

This software is for educational and research purposes. Trading futures involves substantial risk. Use at your own risk. The authors are not responsible for any trading losses.

## Support

For issues or questions:
1. Check logs in `trading.log`
2. Review API status at `/ops/status`
3. Consult `gridStrategy.txt` for detailed specifications

