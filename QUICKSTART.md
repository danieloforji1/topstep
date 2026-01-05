# Quick Start Guide

## 1. Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env

# Edit .env with your credentials
# TOPSTEPX_USERNAME=your_username
# TOPSTEPX_API_KEY=your_api_key
# DRY_RUN=true
```

## 2. Configure Strategy

Edit `config.yaml` to adjust:
- Trading instruments (MES/MNQ by default)
- Risk parameters (daily loss, drawdown limits)
- Grid parameters (levels, spacing multiplier)
- Position sizing (R per trade)

## 3. Run in Dry-Run Mode

```bash
python main.py
```

You should see:
- Authentication with TopstepX
- Contract discovery
- Historical data loading
- Grid generation
- Order placement (simulated)

## 4. Monitor

### Check Status
```bash
curl http://localhost:8000/ops/status
```

### View Metrics
```bash
curl http://localhost:8000/ops/metrics
```

### Check Logs
```bash
tail -f trading.log
```

## 5. Enable Live Trading

**WARNING**: Only enable after thorough testing!

1. Set `DRY_RUN=false` in `.env` or `config.yaml`
2. Verify all risk parameters
3. Ensure account has sufficient balance
4. Run: `python main.py`

## 6. Operations

### Pause Strategy
```bash
curl -X POST http://localhost:8000/ops/pause \
  -H "Content-Type: application/json" \
  -d '{"pause": true}'
```

### Force Flatten
```bash
curl -X POST http://localhost:8000/ops/flatten \
  -H "Content-Type: application/json" \
  -d '{"reason": "Manual intervention"}'
```

## Troubleshooting

### "Failed to authenticate"
- Check API credentials in `.env`
- Verify API key is active
- Check network connectivity

### "Could not find contract"
- Verify instrument symbols (MES, MNQ)
- Check TopstepX account has access
- Try searching contracts manually via API

### "Insufficient data for ATR"
- System needs historical candles
- Wait for data to accumulate
- Check market data connection

## Next Steps

1. Review `gridStrategy.txt` for detailed specifications
2. Test in dry-run for several sessions
3. Monitor metrics and adjust parameters
4. Gradually enable live trading with small positions

