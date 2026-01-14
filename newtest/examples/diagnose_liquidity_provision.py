"""
Diagnostic script to understand why liquidity provision isn't generating signals
"""
import sys
import os
import logging
from datetime import datetime, timezone
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))

from framework.data_manager import DataManager
from strategies.liquidity_provision import (
    estimate_order_flow_imbalance,
    calculate_adverse_selection_probability,
    calculate_favorable_fill_probability,
    calculate_atr
)
from dotenv import load_dotenv

try:
    load_dotenv()
except Exception:
    pass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    # Fetch data
    data_manager = DataManager(cache_dir="newtest/results/cache")
    df = data_manager.fetch_bars(
        symbol="MES",
        interval="5m",
        start_date=datetime(2025, 12, 12, tzinfo=timezone.utc),
        end_date=datetime(2026, 1, 7, tzinfo=timezone.utc)
    )
    
    print(f"Loaded {len(df)} bars")
    print("\n" + "="*80)
    print("ANALYZING ORDER FLOW IMBALANCE")
    print("="*80)
    
    # Config
    imbalance_lookback = 5
    imbalance_threshold = 0.3
    adverse_selection_threshold = 0.4
    favorable_fill_threshold = 0.6
    atr_period = 14
    
    # Calculate base volatility
    if len(df) >= atr_period * 2:
        base_volatility = calculate_atr(
            df.iloc[-atr_period*2:-atr_period],
            period=atr_period
        )
    else:
        base_volatility = calculate_atr(df, period=atr_period)
    
    if base_volatility:
        print(f"Base volatility (ATR): {base_volatility:.4f}")
    else:
        print("Base volatility (ATR): None (using current ATR as fallback)")
        base_volatility = calculate_atr(df, period=atr_period) or 1.0
    print(f"\nThresholds:")
    print(f"  Imbalance threshold: {imbalance_threshold}")
    print(f"  Adverse selection threshold: {adverse_selection_threshold}")
    print(f"  Favorable fill threshold: {favorable_fill_threshold}")
    print("\n" + "="*80)
    
    # Analyze each bar
    stats = {
        'total_bars': 0,
        'insufficient_data': 0,
        'no_atr': 0,
        'adverse_too_high': 0,
        'imbalance_too_weak': 0,
        'favorable_too_low': 0,
        'signals_generated': 0,
        'imbalance_values': [],
        'adverse_p_values': [],
        'favorable_p_values': []
    }
    
    for i in range(imbalance_lookback + atr_period, len(df)):
        stats['total_bars'] += 1
        
        historical = df.iloc[:i+1]
        
        # Calculate imbalance
        imbalance = estimate_order_flow_imbalance(historical, lookback=imbalance_lookback)
        stats['imbalance_values'].append(imbalance)
        
        # Calculate ATR
        atr = calculate_atr(historical, period=atr_period)
        if atr is None or atr == 0:
            stats['no_atr'] += 1
            continue
        
        # Calculate adverse selection
        adverse_p = calculate_adverse_selection_probability(
            imbalance=imbalance,
            volatility=atr,
            base_volatility=base_volatility or atr
        )
        stats['adverse_p_values'].append(adverse_p)
        
        if adverse_p > adverse_selection_threshold:
            stats['adverse_too_high'] += 1
            continue
        
        # Check imbalance threshold
        if abs(imbalance) < imbalance_threshold:
            stats['imbalance_too_weak'] += 1
            continue
        
        # Determine order side
        if imbalance < -imbalance_threshold:
            order_side = "BID"
        elif imbalance > imbalance_threshold:
            order_side = "ASK"
        else:
            stats['imbalance_too_weak'] += 1
            continue
        
        # Calculate favorable fill probability
        favorable_p = calculate_favorable_fill_probability(
            imbalance=imbalance,
            order_side=order_side,
            adverse_p=adverse_p
        )
        stats['favorable_p_values'].append(favorable_p)
        
        if favorable_p < favorable_fill_threshold:
            stats['favorable_too_low'] += 1
            continue
        
        # Signal would be generated!
        stats['signals_generated'] += 1
        
        if stats['signals_generated'] <= 10:  # Show first 10
            print(f"\nBar {i}: Signal generated!")
            print(f"  Imbalance: {imbalance:.4f}")
            print(f"  ATR: {atr:.4f}")
            print(f"  Adverse P: {adverse_p:.4f}")
            print(f"  Favorable P: {favorable_p:.4f}")
            print(f"  Order Side: {order_side}")
    
    # Print statistics
    print("\n" + "="*80)
    print("STATISTICS")
    print("="*80)
    print(f"Total bars analyzed: {stats['total_bars']}")
    print(f"  Insufficient data: {stats['insufficient_data']}")
    print(f"  No ATR: {stats['no_atr']}")
    print(f"  Adverse selection too high: {stats['adverse_too_high']}")
    print(f"  Imbalance too weak: {stats['imbalance_too_weak']}")
    print(f"  Favorable fill too low: {stats['favorable_too_low']}")
    print(f"  Signals generated: {stats['signals_generated']}")
    
    if stats['imbalance_values']:
        print(f"\nImbalance statistics:")
        print(f"  Min: {min(stats['imbalance_values']):.4f}")
        print(f"  Max: {max(stats['imbalance_values']):.4f}")
        print(f"  Mean: {np.mean(stats['imbalance_values']):.4f}")
        print(f"  Std: {np.std(stats['imbalance_values']):.4f}")
        print(f"  Abs mean: {np.mean([abs(x) for x in stats['imbalance_values']]):.4f}")
        print(f"  Values >= {imbalance_threshold}: {sum(1 for x in stats['imbalance_values'] if abs(x) >= imbalance_threshold)}")
    
    if stats['adverse_p_values']:
        print(f"\nAdverse selection probability:")
        print(f"  Mean: {np.mean(stats['adverse_p_values']):.4f}")
        print(f"  Max: {max(stats['adverse_p_values']):.4f}")
        print(f"  Values > {adverse_selection_threshold}: {sum(1 for x in stats['adverse_p_values'] if x > adverse_selection_threshold)}")
    
    if stats['favorable_p_values']:
        print(f"\nFavorable fill probability:")
        print(f"  Mean: {np.mean(stats['favorable_p_values']):.4f}")
        print(f"  Min: {min(stats['favorable_p_values']):.4f}")
        print(f"  Values >= {favorable_fill_threshold}: {sum(1 for x in stats['favorable_p_values'] if x >= favorable_fill_threshold)}")
    
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    
    if stats['signals_generated'] == 0:
        if stats['imbalance_too_weak'] > stats['total_bars'] * 0.5:
            print("⚠️  Most bars fail due to weak imbalance. Consider:")
            print(f"   - Lowering imbalance_threshold from {imbalance_threshold} to 0.2 or 0.15")
            print("   - Improving imbalance calculation to be more sensitive")
        
        if stats['favorable_too_low'] > stats['total_bars'] * 0.3:
            print("⚠️  Many bars fail due to low favorable fill probability. Consider:")
            print(f"   - Lowering favorable_fill_threshold from {favorable_fill_threshold} to 0.5 or 0.4")
        
        if stats['adverse_too_high'] > stats['total_bars'] * 0.3:
            print("⚠️  Many bars fail due to high adverse selection. Consider:")
            print(f"   - Raising adverse_selection_threshold from {adverse_selection_threshold} to 0.5 or 0.6")


if __name__ == "__main__":
    main()

