"""
Diagnostic script to understand why Optimal Stopping strategy isn't generating trades
"""
import sys
import os
import pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))
from framework.data_manager import DataManager
from strategies.optimal_stopping import OptimalStoppingStrategy

# Try to load .env, but don't fail if it doesn't exist
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


def main():
    """Diagnose why strategy isn't generating signals"""
    # Load data
    data_manager = DataManager()
    df = data_manager.fetch_bars(
        symbol="MES",
        interval="15m",
        start_date=datetime(2025, 12, 7),
        end_date=datetime(2026, 1, 6)
    )
    
    print(f"Loaded {len(df)} bars")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print()
    
    # Test with different configurations
    configs = [
        {'score_threshold': 0.5, 'lookback_window': 100, 'min_opportunities_seen': 37},
        {'score_threshold': 0.4, 'lookback_window': 50, 'min_opportunities_seen': 18},
        {'score_threshold': 0.3, 'lookback_window': 50, 'min_opportunities_seen': 18},
    ]
    
    for config in configs:
        print("="*80)
        print(f"Testing config: {config}")
        print("="*80)
        
        strategy = OptimalStoppingStrategy(config)
        
        # Track scores and conditions
        scores = []
        opportunities = []
        best_scores = []
        signals_generated = 0
        
        # Simulate the strategy
        for i in range(config['lookback_window'], len(df)):
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
            
            # Calculate score manually
            score = strategy._calculate_entry_score(historical, i)
            scores.append(score)
            
            # Track opportunities
            strategy.opportunities_seen += 1
            strategy.opportunity_scores.append(score)
            if len(strategy.opportunity_scores) > config['lookback_window']:
                strategy.opportunity_scores.pop(0)
                strategy.opportunities_seen = len(strategy.opportunity_scores)
            
            # Update best score
            if abs(score) > abs(strategy.best_score_so_far):
                strategy.best_score_so_far = score
            best_scores.append(strategy.best_score_so_far)
            
            # Check conditions
            opportunities_pct = strategy.opportunities_seen / config['lookback_window'] if config['lookback_window'] > 0 else 0
            
            conditions = {
                'score_abs': abs(score),
                'score_threshold': config['score_threshold'],
                'meets_threshold': abs(score) >= config['score_threshold'],
                'opportunities_pct': opportunities_pct,
                'min_opportunities_pct': config['min_opportunities_seen'] / config['lookback_window'],
                'meets_opportunities': opportunities_pct >= (config['min_opportunities_seen'] / config['lookback_window']),
                'best_score_abs': abs(strategy.best_score_so_far),
                'meets_best_score': abs(score) >= abs(strategy.best_score_so_far) * 0.9,
            }
            
            opportunities.append(conditions)
            
            # Try to generate signal
            signal = strategy.generate_signal(market_data, historical, None)
            if signal:
                signals_generated += 1
                print(f"  Signal #{signals_generated} at bar {i}: {signal.direction} (score={score:.3f})")
        
        # Statistics
        scores_df = pd.DataFrame({
            'score': scores,
            'abs_score': [abs(s) for s in scores],
            'best_score': best_scores,
            'abs_best_score': [abs(s) for s in best_scores]
        })
        
        opps_df = pd.DataFrame(opportunities)
        
        print(f"\nStatistics:")
        print(f"  Total bars processed: {len(scores)}")
        print(f"  Signals generated: {signals_generated}")
        print(f"  Score stats:")
        print(f"    Mean: {scores_df['abs_score'].mean():.3f}")
        print(f"    Max: {scores_df['abs_score'].max():.3f}")
        print(f"    Min: {scores_df['abs_score'].min():.3f}")
        print(f"    Scores >= threshold ({config['score_threshold']}): {(scores_df['abs_score'] >= config['score_threshold']).sum()} ({100*(scores_df['abs_score'] >= config['score_threshold']).sum()/len(scores_df):.1f}%)")
        print(f"  Condition failures:")
        print(f"    Threshold not met: {(~opps_df['meets_threshold']).sum()} times")
        print(f"    Opportunities not met: {(~opps_df['meets_opportunities']).sum()} times")
        print(f"    Best score not met: {(~opps_df['meets_best_score']).sum()} times")
        print(f"  All conditions met: {(opps_df['meets_threshold'] & opps_df['meets_opportunities'] & opps_df['meets_best_score']).sum()} times")
        print()


if __name__ == "__main__":
    main()

