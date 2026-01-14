"""
Test script to verify partial profit taking logic
"""
import sys
import os
import pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))
from framework.data_manager import DataManager
from strategies.optimal_stopping import OptimalStoppingStrategy, calculate_atr
from dotenv import load_dotenv

try:
    load_dotenv()
except:
    pass

def main():
    # Load data
    data_manager = DataManager()
    df = data_manager.fetch_bars(
        symbol="MES",
        interval="15m",
        start_date=datetime(2025, 12, 7),
        end_date=datetime(2026, 1, 6)
    )
    
    config = {
        'lookback_window': 100,
        'min_opportunities_seen': 37,
        'score_threshold': 0.7,
        'atr_period': 14,
        'atr_multiplier_stop': 0.75,
        'atr_multiplier_target': 1.25,
        'max_hold_bars': 40,
        'use_partial_profit': True,
        'partial_profit_pct': 0.5,
        'partial_profit_target_atr': 0.5
    }
    
    strategy = OptimalStoppingStrategy(config)
    
    # Simulate and check partial profit opportunities
    partial_opportunities = []
    
    for i in range(100, len(df)):
        row = df.iloc[i]
        market_data = type('MarketData', (), {
            'timestamp': row['timestamp'],
            'open': row['open'],
            'high': row['high'],
            'low': row['low'],
            'close': row['close'],
            'volume': row['volume']
        })()
        
        historical = df.iloc[:i+1]
        
        # Check if we would enter
        signal = strategy.generate_signal(market_data, historical, None)
        if signal:
            # Calculate ATR
            atr = calculate_atr(historical.iloc[i-14:i], period=14)
            if atr:
                partial_target = signal.entry_price + (atr * 0.5) if signal.direction == "LONG" else signal.entry_price - (atr * 0.5)
                distance = abs(partial_target - signal.entry_price)
                print(f"Entry: {signal.entry_price:.2f}, Partial Target: {partial_target:.2f}, Distance: {distance:.2f} points, ATR: {atr:.2f}")
                
                # Check if this target was hit in next 40 bars
                for j in range(i+1, min(i+41, len(df))):
                    next_row = df.iloc[j]
                    if signal.direction == "LONG" and next_row['high'] >= partial_target:
                        print(f"  ✓ Partial target HIT at bar {j} (price: {next_row['high']:.2f})")
                        partial_opportunities.append((i, j, signal.entry_price, partial_target))
                        break
                    elif signal.direction == "SHORT" and next_row['low'] <= partial_target:
                        print(f"  ✓ Partial target HIT at bar {j} (price: {next_row['low']:.2f})")
                        partial_opportunities.append((i, j, signal.entry_price, partial_target))
                        break
    
    print(f"\nFound {len(partial_opportunities)} partial profit opportunities")

if __name__ == "__main__":
    main()

