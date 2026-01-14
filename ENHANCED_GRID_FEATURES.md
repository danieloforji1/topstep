# Enhanced Grid Trading Features

This document describes the three major enhancements implemented for the grid trading strategy:

1. **Adaptive Grid Density** - Dynamic level count based on volatility regime
2. **Multi-Timeframe Alignment** - Grid alignment to support/resistance from higher timeframes
3. **Order Flow Integration** - Grid adjustment based on bid/ask imbalance and volume profile

---

## 1. Adaptive Grid Density

### Overview
The grid automatically adjusts the number of levels based on current volatility conditions. In high volatility periods, fewer levels are placed (wider spacing), while in low volatility periods, more levels are placed (tighter spacing).

### How It Works
- **Volatility Tracking**: Maintains a rolling history of volatility (ATR) readings
- **Percentile Calculation**: Calculates where current volatility sits relative to recent history
- **Dynamic Adjustment**: 
  - High volatility (high percentile) → Fewer levels (min_levels_each_side)
  - Low volatility (low percentile) → More levels (max_levels_each_side)
  - Formula: `levels = base_levels - (percentile - 0.5) * range * 2`

### Configuration
```yaml
adaptive_density: true  # Enable/disable adaptive density
base_levels_each_side: 3  # Base number of levels
min_levels_each_side: 2  # Minimum when volatility is high
max_levels_each_side: 5  # Maximum when volatility is low
```

### Benefits
- **Better Risk Management**: Fewer orders in volatile conditions reduce exposure
- **Improved Fill Rates**: More orders in calm conditions capture more opportunities
- **Automatic Adaptation**: No manual parameter tuning needed

---

## 2. Multi-Timeframe Alignment

### Overview
The grid automatically aligns to significant support and resistance levels detected from higher timeframes (1h, 4h). This ensures grid levels are placed at price zones where the market is more likely to react.

### How It Works
- **Pivot Detection**: Identifies local highs and lows (pivot points) on higher timeframes
- **Level Clustering**: Groups nearby pivot points into significant levels
- **Strength Calculation**: Each level is scored based on:
  - Number of touches
  - Recency of touches
  - Timeframe importance
- **Grid Alignment**: Adjusts grid midpoint to align with strong nearby levels

### Configuration
```yaml
enable_multi_timeframe: true  # Enable/disable MTF alignment
mtf_timeframes: ["1h", "4h"]  # Timeframes to analyze
mtf_lookback_1h: 50  # Number of 1h candles to analyze
mtf_lookback_4h: 30  # Number of 4h candles to analyze
mtf_alignment_threshold: 0.002  # Max distance to align (0.2% of price)
```

### Benefits
- **Better Entry/Exit Points**: Grid levels at significant price zones
- **Reduced False Signals**: Avoids placing orders in "dead zones"
- **Higher Win Rate**: Trades at levels where price is more likely to reverse

---

## 3. Order Flow Integration

### Overview
The grid adjusts its midpoint based on real-time order flow analysis, including bid/ask imbalance and volume profile. This helps place orders where there's actual market interest.

### How It Works
- **Bid/Ask Imbalance**: Tracks the ratio of bid size to ask size
- **Volume Imbalance**: Tracks buy volume vs sell volume
- **Combined Signal**: Weighted average of bid/ask and volume imbalances
- **Grid Adjustment**: Shifts grid midpoint slightly toward buying or selling pressure
- **Volume Profile**: Avoids placing orders in low-volume price zones

### Configuration
```yaml
enable_order_flow: true  # Enable/disable order flow analysis
order_flow_lookback: 20  # Number of snapshots to keep
order_flow_imbalance_threshold: 0.3  # Threshold for significant imbalance
order_flow_volume_threshold: 0.1  # Minimum volume to avoid price levels
```

### Benefits
- **Better Fill Probability**: Orders placed where there's actual liquidity
- **Reduced Slippage**: Avoids low-volume zones
- **Market Microstructure Edge**: Leverages order flow information

---

## Integration Details

### Data Flow

1. **Historical Data Loading**:
   - Primary timeframe (15m) for ATR/volatility
   - Higher timeframes (1h, 4h) for support/resistance
   - All stored in `TimeseriesStore`

2. **Real-Time Updates**:
   - Quote updates feed `OrderFlowAnalyzer`
   - Price updates trigger grid recalculation
   - Multi-timeframe levels updated periodically

3. **Grid Generation**:
   - Adaptive density calculates level count
   - Order flow adjusts midpoint
   - Multi-timeframe aligns to key levels
   - Volume profile filters out low-volume zones

### Performance Considerations

- **Order Flow**: Minimal overhead (simple calculations on recent snapshots)
- **Multi-Timeframe**: Moderate overhead (pivot detection on higher timeframes, runs periodically)
- **Adaptive Density**: Minimal overhead (percentile calculation from history)

All features are designed to be lightweight and not impact real-time performance.

---

## Testing on Practice Account

### Recommended Settings

For initial testing on practice account:

```yaml
# Conservative settings for testing
adaptive_density: true
base_levels_each_side: 3
min_levels_each_side: 2
max_levels_each_side: 4  # Start conservative

enable_multi_timeframe: true
mtf_alignment_threshold: 0.003  # Slightly wider threshold

enable_order_flow: true
order_flow_imbalance_threshold: 0.4  # Higher threshold (less sensitive)
```

### Monitoring

Watch for these metrics:
- **Grid level count**: Should vary based on volatility
- **Grid midpoint adjustments**: Should align to support/resistance
- **Order fill rates**: Should improve with order flow integration
- **Win rate**: Should improve with multi-timeframe alignment

### Log Messages

Look for these log messages:
- `"Adaptive levels: X (percentile: Y)"` - Shows adaptive density working
- `"MTF aligned grid mid: X"` - Shows multi-timeframe alignment
- `"Order flow adjusted mid: X (adjustment: Y)"` - Shows order flow adjustment

---

## Disabling Features

To disable any feature, set the corresponding config flag to `false`:

```yaml
adaptive_density: false  # Disable adaptive density
enable_multi_timeframe: false  # Disable MTF alignment
enable_order_flow: false  # Disable order flow
```

The system will fall back to standard grid behavior when features are disabled.

---

## Troubleshooting

### Grid not adjusting
- Check that `adaptive_density: true` in config
- Verify volatility history is being collected (check logs)
- Ensure sufficient historical data (need 20+ readings)

### MTF alignment not working
- Verify higher timeframe data is loading (check startup logs)
- Check `mtf_timeframes` includes valid intervals
- Ensure `mtf_alignment_threshold` is reasonable (0.002 = 0.2%)

### Order flow not adjusting
- Verify real-time quotes are being received
- Check that `enable_order_flow: true` in config
- Ensure quote data includes bid/ask sizes

---

## Next Steps

After testing on practice account:
1. Monitor performance metrics
2. Adjust thresholds based on results
3. Fine-tune adaptive density ranges
4. Consider adding more timeframes (e.g., daily for swing levels)

---

## Files Modified

- `src/strategy/grid_manager.py` - Enhanced with adaptive density and integration points
- `src/strategy/order_flow_analyzer.py` - New module for order flow analysis
- `src/strategy/multi_timeframe_analyzer.py` - New module for MTF analysis
- `main.py` - Integration of new analyzers
- `config.yaml` - Configuration options for new features

