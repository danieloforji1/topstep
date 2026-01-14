# Live Trading Quick Start Guide

## 🚀 Getting Started in 5 Minutes

### Step 1: Setup Environment

1. Ensure you have a `.env` file with your TopstepX credentials:
```bash
TOPSTEPX_USERNAME=your_username
TOPSTEPX_API_KEY=your_api_key
```

2. Verify credentials work:
```bash
python3.11 live_trading/monitor.py
```

### Step 2: Test in Dry Run Mode

Start with dry-run mode to test without placing real orders:

```bash
cd topstep/newtest
python3.11 live_trading/live_strategy_executor.py \
    --symbol MES \
    --dry-run \
    --strategies optimal_stopping
```

This will:
- Connect to TopstepX API
- Load historical data
- Generate signals (but not place orders)
- Log all activity

### Step 3: Monitor Activity

Watch the console output and check `live_trading.log` for:
- Signal generation
- Order placement (simulated in dry-run)
- Position management
- Exit conditions

### Step 4: Add More Strategies

Once comfortable, add more strategies:

```bash
python3.11 live_trading/live_strategy_executor.py \
    --symbol MES \
    --dry-run \
    --strategies optimal_stopping multi_timeframe liquidity_provision
```

### Step 5: Go Live (When Ready)

**⚠️ WARNING: This places REAL orders!**

```bash
python3.11 live_trading/live_strategy_executor.py \
    --symbol MES \
    --live \
    --strategies optimal_stopping
```

## Recommended Rollout Plan

### Week 1: Dry Run Testing
- Run all strategies in dry-run mode
- Monitor signal generation
- Verify risk management
- Check order execution logic

### Week 2: Single Strategy Live
- Enable one strategy (start with Optimal Stopping)
- Monitor closely
- Verify fills and exits
- Track performance

### Week 3: Add Second Strategy
- Add Multi-Timeframe
- Monitor both strategies
- Check for conflicts
- Adjust if needed

### Week 4: Full Deployment
- Add Liquidity Provision
- All three strategies running
- Full monitoring
- Performance tracking

## Monitoring Commands

### Check Account Status
```bash
python3.11 live_trading/monitor.py
```

### View Logs
```bash
tail -f live_trading.log
```

### Check Specific Strategy
```bash
grep "optimal_stopping" live_trading.log
```

## Safety Checklist

Before going live:

- [ ] Tested in dry-run for at least 24 hours
- [ ] Verified credentials are correct
- [ ] Checked account balance and limits
- [ ] Reviewed strategy configurations
- [ ] Understood risk per trade ($100 default)
- [ ] Set account-level stop losses if available
- [ ] Have monitoring in place
- [ ] Know how to stop the system (Ctrl+C)

## Troubleshooting

### "Failed to authenticate"
- Check `.env` file exists and has correct credentials
- Verify API key is valid
- Check account is active

### "No signals generated"
- This is normal - strategies are selective
- Check historical data loaded successfully
- Verify market is open
- Review strategy thresholds

### "No market data"
- Check SignalR connection
- Verify contract ID is correct
- Ensure market is open

## Performance Expectations

Based on backtesting:

| Strategy | Expected Sharpe | Win Rate | Notes |
|----------|----------------|----------|-------|
| Optimal Stopping | 1.5+ | 60%+ | Most consistent |
| Multi-Timeframe | 1.64 | 60%+ | Best Sharpe |
| Liquidity Provision | 1.26 | 50%+ | Largest wins |

**Remember**: Live performance may differ from backtesting!

## Next Steps

1. ✅ Complete dry-run testing
2. ✅ Review logs and signals
3. ✅ Start with one strategy
4. ✅ Gradually add more
5. ✅ Monitor and optimize

## Support

For issues:
1. Check `live_trading.log`
2. Review console output
3. Verify API connection
4. Test with monitor.py

Good luck! 🎯


