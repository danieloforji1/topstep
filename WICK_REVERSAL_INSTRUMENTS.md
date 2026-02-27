# Wick Reversal Strategy - Instrument Analysis

## Current Configuration: MES, MNQ, MGC

### ✅ MES (Micro E-mini S&P 500) - **EXCELLENT CHOICE**
- **Liquidity**: ⭐⭐⭐⭐⭐ Very high
- **Mean-Reversion**: ⭐⭐⭐⭐⭐ Excellent (index futures mean-revert well)
- **Volatility**: ⭐⭐⭐⭐ Good (enough for wicks, not too choppy)
- **Tick Size**: 0.25 points
- **Tick Value**: $1.25 per tick
- **Why it works**: High liquidity ensures good fills, strong mean-reversion characteristics, predictable volatility patterns

### ✅ MNQ (Micro E-mini Nasdaq) - **EXCELLENT CHOICE**
- **Liquidity**: ⭐⭐⭐⭐⭐ Very high
- **Mean-Reversion**: ⭐⭐⭐⭐ Very good (slightly more volatile than MES)
- **Volatility**: ⭐⭐⭐⭐⭐ Higher than MES (more wick opportunities)
- **Tick Size**: 0.25 points
- **Tick Value**: $0.50 per tick
- **Why it works**: Similar to MES but with higher volatility = more wick opportunities. Good correlation with MES for diversification.

### ⚠️ MGC (Micro Gold) - **GOOD BUT NOT IDEAL**
- **Liquidity**: ⭐⭐⭐ Moderate (lower than equity futures)
- **Mean-Reversion**: ⭐⭐⭐ Moderate (tends to trend more than mean-revert)
- **Volatility**: ⭐⭐⭐⭐ Good (enough for wicks)
- **Tick Size**: 0.10 points
- **Tick Value**: $1.00 per tick
- **Why it's less ideal**: 
  - Gold can trend strongly (bad for mean-reversion)
  - Lower liquidity = wider spreads, potential slippage
  - Different volatility characteristics than equity futures
  - Less correlation with MES/MNQ (diversification is good, but different behavior)

## 🎯 Better Alternatives

### 1. **MYM (Micro Dow Jones)** - **HIGHLY RECOMMENDED**
- **Liquidity**: ⭐⭐⭐⭐⭐ Very high
- **Mean-Reversion**: ⭐⭐⭐⭐⭐ Excellent
- **Volatility**: ⭐⭐⭐⭐ Good
- **Tick Size**: 1.0 points
- **Tick Value**: $0.50 per tick
- **Why it's better**: Similar characteristics to MES/MNQ, good liquidity, strong mean-reversion

### 2. **M2K (Micro Russell 2000)** - **HIGHLY RECOMMENDED**
- **Liquidity**: ⭐⭐⭐⭐ High
- **Mean-Reversion**: ⭐⭐⭐⭐⭐ Excellent (small caps mean-revert strongly)
- **Volatility**: ⭐⭐⭐⭐⭐ High (lots of wick opportunities)
- **Tick Size**: 0.10 points
- **Tick Value**: $0.50 per tick
- **Why it's better**: Strong mean-reversion characteristics, good volatility for wicks

### 3. **M6E (Micro Euro FX)** - **GOOD ALTERNATIVE**
- **Liquidity**: ⭐⭐⭐⭐ High
- **Mean-Reversion**: ⭐⭐⭐⭐ Good
- **Volatility**: ⭐⭐⭐⭐ Good
- **Tick Size**: 0.0001 (very small)
- **Tick Value**: $1.25 per tick
- **Why it's good**: Good mean-reversion, decent liquidity, different asset class

### 4. **M6B (Micro British Pound)** - **GOOD ALTERNATIVE**
- **Liquidity**: ⭐⭐⭐⭐ High
- **Mean-Reversion**: ⭐⭐⭐⭐ Good
- **Volatility**: ⭐⭐⭐⭐ Good
- **Tick Size**: 0.0001
- **Tick Value**: $0.625 per tick
- **Why it's good**: Similar to M6E, good diversification

## 📊 Recommended Configurations

### Option 1: **Pure Equity Index (Best Mean-Reversion)**
```yaml
instruments: [MES, MNQ, MYM, M2K]
```
- All equity indices = similar behavior
- Strong mean-reversion characteristics
- High liquidity across all
- Easy to manage (similar tick sizes)

### Option 2: **Current + Better Equity (Balanced)**
```yaml
instruments: [MES, MNQ, MYM]  # Replace MGC with MYM
```
- Keep your current MES/MNQ
- Replace MGC with MYM (better fit)
- Still diversified but more consistent behavior

### Option 3: **Current Mix (Keep MGC)**
```yaml
instruments: [MES, MNQ, MGC]  # Your current setup
```
- **Pros**: Diversification across asset classes
- **Cons**: MGC behaves differently (trends more)
- **Recommendation**: Monitor MGC performance separately, consider replacing if underperforming

## ⚙️ Configuration Adjustments for MGC

If you keep MGC, consider these adjustments:

```yaml
# MGC-specific considerations:
# - Smaller tick size (0.10 vs 0.25) means structure_min_move of 0.5 = 5 ticks (reasonable)
# - Use wick-relative tolerance (already set) - this adapts to each instrument
# - Consider slightly higher wick_threshold for MGC (it trends more)
```

## 🎯 Final Recommendation

**Best Setup for Wick Reversal:**
```yaml
instruments: [MES, MNQ, MYM]
```

**Why:**
1. All three are equity indices = consistent mean-reversion behavior
2. High liquidity = good fills, tight spreads
3. Similar tick sizes (0.25 for MES/MNQ, 1.0 for MYM) = easier to manage
4. Good diversification (S&P, Nasdaq, Dow) without mixing asset classes
5. All three work well with the same strategy parameters

**If you want to keep MGC:**
- Monitor its performance separately
- Consider using instrument-specific parameters if needed
- Be aware it may underperform in trending markets

