# Topstep 50k Challenge Setup Guide

## Overview
This guide explains how to use the tailored configuration for Topstep's $50,000 Trading Combine challenge.

## Challenge Requirements
- **Profit Target:** $3,000
- **Daily Loss Limit:** $1,000
- **Trailing Drawdown Limit:** $2,000 (from highest balance)
- **Max Position Size:** 5 contracts simultaneously
- **Consistency Rule:** Best day cannot exceed 50% of total profits

## Configuration File
Use `config_50k_challenge.yaml` instead of `config.yaml` for the challenge.

## Key Changes from Practice Account Config

### 1. Risk Limits (with safety buffers)
- `max_daily_loss: 900` (safety buffer below $1,000 limit)
- `trailing_drawdown_limit: 1800` (safety buffer below $2,000 limit)
- `target_daily_PL: 300` (conservative daily target)

### 2. Position Size Limits
- `max_position_size: 5` (hard limit enforced in code)
- `grid_levels_each_side: 3` (reduced from 5 to respect 5-contract limit)
- `max_lot_per_order: 5` (never exceed 5 per order)

### 3. More Conservative Settings
- `sizer_R_per_trade: 100` (reduced from $150 for safety)
- `max_net_notional: 800` (reduced exposure cap)
- `hedge_ratio_max: 1.0` (reduced from 1.25)
- `max_hedge_contracts_multiplier: 0.5` (conservative hedge sizing)

### 4. Account Selection
- `prefer_practice_account: false` **CRITICAL: Must be false for challenge**
- `dry_run: false` **CRITICAL: Must be false for live trading**

## How to Start

### Step 1: Backup Current Config
```bash
cp config.yaml config_practice_backup.yaml
```

### Step 2: Use Challenge Config
```bash
# Option 1: Rename the challenge config
cp config_50k_challenge.yaml config.yaml

# Option 2: Or modify main.py to accept config file as argument
# python main.py --config config_50k_challenge.yaml
```

### Step 3: Verify Account Settings
**CRITICAL:** Before starting, verify in `config_50k_challenge.yaml`:
- `prefer_practice_account: false` ← Must be false
- `dry_run: false` ← Must be false
- `account_id: null` ← Set to your challenge account ID if needed

### Step 4: Set Environment Variables
```bash
export TOPSTEPX_USERNAME="your_username"
export TOPSTEPX_API_KEY="your_api_key"
export DRY_RUN="false"  # Ensure this is false
```

### Step 5: Start the Strategy
```bash
python main.py
```

## Safety Features Added

### 1. Position Size Enforcement
- Code now checks total position (primary + hedge) before placing orders
- Will skip orders if they would exceed 5-contract limit
- Logs warnings when approaching limit

### 2. Enhanced Risk Checks
- More frequent position reconciliation
- Stricter hedge sizing caps
- Conservative lot sizing

### 3. Challenge-Specific Limits
- Grid levels reduced to 3 per side (from 5)
- Lower risk per trade ($100 vs $150)
- Tighter exposure caps

## Monitoring During Challenge

### Key Metrics to Watch
1. **Total Position Size:** Should never exceed 5 contracts
2. **Daily P&L:** Should stay above -$900 (safety buffer)
3. **Trailing Drawdown:** Should stay below $1,800 (safety buffer)
4. **Consistency:** Best day should not exceed 50% of total profit

### Dashboard
Access the dashboard at: `http://localhost:8000`
- Monitor positions, P&L, and risk metrics
- Check that total contracts never exceed 5

## Important Notes

⚠️ **CRITICAL WARNINGS:**
1. **Never set `prefer_practice_account: true`** during challenge
2. **Never set `dry_run: true`** during challenge
3. **Always verify account ID** matches your challenge account
4. **Monitor position size** - system will enforce 5-contract limit but verify in logs
5. **Test in practice first** - Make sure everything works before using on challenge account

## Troubleshooting

### Issue: Positions exceed 5 contracts
- **Solution:** Check logs for position size warnings
- System should automatically prevent this, but verify in TopstepX dashboard

### Issue: Daily loss limit hit
- **Solution:** System will automatically flatten and stop trading
- Review strategy parameters if this happens frequently

### Issue: Wrong account selected
- **Solution:** Set `account_id` explicitly in config or verify `prefer_practice_account: false`

## Reverting to Practice Account

After challenge (or to test):
```bash
cp config_practice_backup.yaml config.yaml
# Or use original config.yaml
```

## Success Criteria

To pass the 50k challenge:
1. ✅ Reach $3,000 profit
2. ✅ Never exceed $1,000 daily loss
3. ✅ Never exceed $2,000 trailing drawdown
4. ✅ Never exceed 5 contracts simultaneously
5. ✅ Best day ≤ 50% of total profit

Good luck! 🚀

