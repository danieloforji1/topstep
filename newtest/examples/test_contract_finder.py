"""
Test ContractFinder to see what contracts are available
"""
import sys
import os
import logging
from datetime import datetime, timezone

# Load environment variables
try:
    from dotenv import load_dotenv
    env_paths = [
        os.path.join(os.path.dirname(__file__), '../../.env'),
        os.path.join(os.path.dirname(__file__), '../../../.env'),
        '.env'
    ]
    for env_path in env_paths:
        if os.path.exists(env_path):
            load_dotenv(env_path)
            break
    else:
        load_dotenv()
except Exception:
    pass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

from src.connectors.topstepx_client import TopstepXClient

def test_contract_search(symbol: str):
    """Test different ways to search for contracts"""
    print(f"\n{'='*80}")
    print(f"Testing Contract Search for {symbol}")
    print(f"{'='*80}\n")
    
    client = TopstepXClient()
    if not client.authenticate():
        print("ERROR: Failed to authenticate")
        return
    
    print("✓ Authenticated successfully\n")
    
    # Test 1: Basic search
    print("Test 1: Basic search with symbol")
    print("-" * 80)
    contracts = client.search_contracts(symbol=symbol, live=False)
    print(f"Found {len(contracts)} contracts with live=False")
    for i, contract in enumerate(contracts[:10]):  # Show first 10
        print(f"  {i+1}. {contract.get('name', 'N/A')} (ID: {contract.get('contractId') or contract.get('id')})")
    print()
    
    # Test 2: Search with live=True
    print("Test 2: Search with live=True")
    print("-" * 80)
    contracts_live = client.search_contracts(symbol=symbol, live=True)
    print(f"Found {len(contracts_live)} contracts with live=True")
    for i, contract in enumerate(contracts_live[:10]):
        print(f"  {i+1}. {contract.get('name', 'N/A')} (ID: {contract.get('contractId') or contract.get('id')})")
    print()
    
    # Test 3: Search with variations - FILTER TO ONLY OUR SYMBOL
    print("Test 3: Search with different terms (FILTERED to exact symbol match)")
    print("-" * 80)
    search_terms = [symbol, symbol.upper(), symbol.lower()]
    if symbol == "MES":
        search_terms.extend(["MICRO S&P", "MICRO ES"])
    elif symbol == "MNQ":
        search_terms.extend(["MICRO NASDAQ", "MICRO NQ"])
    elif symbol == "MGC":
        search_terms.extend(["MICRO GOLD", "GOLD"])
    
    all_unique_contracts = {}
    symbol_upper = symbol.upper()
    
    for term in search_terms:
        for live in [False, True]:
            try:
                found = client.search_contracts(symbol=term, live=live)
                for contract in found:
                    contract_id = contract.get('contractId') or contract.get('id') or contract.get('contract_id') or contract.get('contractIdStr')
                    contract_name = contract.get('name', '').upper()
                    
                    # STRICT FILTER: Only include contracts that match our symbol exactly
                    # Check if contract name starts with our symbol (e.g., "MES" in "MESH6")
                    # or if contract ID contains our symbol
                    contract_id_str = str(contract_id) if contract_id else ''
                    matches_symbol = (
                        contract_name.startswith(symbol_upper) or
                        symbol_upper in contract_name or
                        symbol_upper in contract_id_str.upper()
                    )
                    
                    if contract_id and matches_symbol:
                        contract_id_str = str(contract_id)
                        if contract_id_str not in all_unique_contracts:
                            all_unique_contracts[contract_id_str] = {
                                'id': contract_id_str,
                                'name': contract.get('name', 'N/A'),
                                'description': contract.get('description', ''),
                                'found_via': f"{term} (live={live})"
                            }
            except Exception as e:
                print(f"  Error searching '{term}' (live={live}): {e}")
    
    print(f"Total unique {symbol} contracts found: {len(all_unique_contracts)}")
    for i, (contract_id, info) in enumerate(sorted(all_unique_contracts.items())):
        print(f"  {i+1}. {info['name']} (ID: {contract_id})")
        print(f"      Found via: {info['found_via']}")
    print()
    
    # Test 4: Try to get contract details
    if len(all_unique_contracts) >= 2:
        print("Test 4: Contract details for first 2 contracts")
        print("-" * 80)
        sorted_contracts = sorted(all_unique_contracts.items(), key=lambda x: x[1]['name'])
        for i, (contract_id, info) in enumerate(sorted_contracts[:2]):
            print(f"\nContract {i+1}: {info['name']}")
            print(f"  ID: {contract_id}")
            contract_details = client.get_contract_by_id(contract_id)
            if contract_details:
                print(f"  Details: {contract_details}")
            else:
                print(f"  Could not fetch details")
    
    # Test 5: Check if we can find data for these contracts
    if len(all_unique_contracts) >= 2:
        print("\nTest 5: Check data availability for first 2 contracts")
        print("-" * 80)
        sorted_contracts = sorted(all_unique_contracts.items(), key=lambda x: x[1]['name'])
        test_start = datetime(2025, 12, 7, tzinfo=timezone.utc)
        test_end = datetime(2026, 1, 6, tzinfo=timezone.utc)
        
        for i, (contract_id, info) in enumerate(sorted_contracts[:2]):
            print(f"\nContract {i+1}: {info['name']} ({contract_id})")
            for live in [False, True]:
                try:
                    bars = client.get_bars(
                        contract_id=contract_id,
                        interval="15m",
                        start_time=test_start,
                        end_time=test_end,
                        limit=10,
                        live=live
                    )
                    if bars:
                        print(f"  ✓ Has data (live={live}): {len(bars)} bars")
                        # Show date range
                        if bars:
                            first_bar = bars[0]
                            last_bar = bars[-1]
                            first_time = first_bar.get('t') or first_bar.get('timestamp')
                            last_time = last_bar.get('t') or last_bar.get('timestamp')
                            print(f"    Date range: {first_time} to {last_time}")
                    else:
                        print(f"  ✗ No data (live={live})")
                except Exception as e:
                    print(f"  ✗ Error checking data (live={live}): {e}")
    
    # Test 6: Try to find other expiration months by testing common patterns
    print("\nTest 6: Try to find other expiration months")
    print("-" * 80)
    if len(all_unique_contracts) > 0:
        # Extract the base contract ID pattern
        first_contract_id = list(all_unique_contracts.keys())[0]
        print(f"Base contract ID: {first_contract_id}")
        
        # Contract IDs seem to follow pattern: CON.F.US.{SYMBOL}.XXXX
        # Where XXXX is expiration code (e.g., H26 = March 2026)
        # Try common expiration codes
        base_pattern = f"CON.F.US.{symbol_upper}."
        expiration_codes = [
            "H26",  # March 2026 (current)
            "M26",  # June 2026
            "U26",  # September 2026
            "Z26",  # December 2026
            "H27",  # March 2027
            "M27",  # June 2027
            "G26",  # February 2026
            "J26",  # April 2026
            "K26",  # May 2026
            "N26",  # July 2026
            "Q26",  # August 2026
            "V26",  # October 2026
            "X26",  # November 2026
        ]
        
        print(f"\nTesting expiration codes for {symbol} contracts:")
        found_additional = {}
        for exp_code in expiration_codes:
            test_contract_id = base_pattern + exp_code
            if test_contract_id not in all_unique_contracts:
                # Try to get data for this contract
                for live in [False, True]:
                    try:
                        bars = client.get_bars(
                            contract_id=test_contract_id,
                            interval="15m",
                            start_time=datetime(2025, 12, 7, tzinfo=timezone.utc),
                            end_time=datetime(2026, 1, 6, tzinfo=timezone.utc),
                            limit=5,
                            live=live
                        )
                        if bars:
                            # Try to get contract name
                            contract_details = client.get_contract_by_id(test_contract_id)
                            contract_name = contract_details.get('name', f"MES{exp_code}") if contract_details else f"MES{exp_code}"
                            # Make sure this is actually the right symbol
                            if symbol_upper in contract_name or symbol_upper in test_contract_id:
                                found_additional[test_contract_id] = {
                                    'id': test_contract_id,
                                    'name': contract_name,
                                    'exp_code': exp_code,
                                    'has_data': True
                                }
                                print(f"  ✓ Found: {contract_name} ({test_contract_id}) - Has data")
                                break
                    except Exception as e:
                        pass  # Contract doesn't exist or no data
        
        if found_additional:
            print(f"\nFound {len(found_additional)} additional MES contracts!")
            all_unique_contracts.update(found_additional)
        else:
            print("  ✗ No additional MES contracts found with data")
    
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Total unique {symbol} contracts found: {len(all_unique_contracts)}")
    if len(all_unique_contracts) >= 2:
        print("✓ SUCCESS: Found at least 2 contracts for calendar spread")
        sorted_contracts = sorted(all_unique_contracts.items(), key=lambda x: x[1]['name'])
        print(f"\nRecommended contracts for calendar spread:")
        for i, (contract_id, info) in enumerate(sorted_contracts[:2]):
            print(f"  {i+1}. {info['name']} (ID: {contract_id})")
    else:
        print("✗ FAILED: Need at least 2 contracts for calendar spread")
        print("\nPossible solutions:")
        print("  1. Try a different date range (other months might have more contracts)")
        print("  2. Use a different symbol (MNQ, MGC might have more contracts)")
        print("  3. Calendar spread might not be available for this symbol/date range")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Test contract finder')
    parser.add_argument('--symbol', default='MES', help='Symbol to test (default: MES)')
    args = parser.parse_args()
    
    test_contract_search(args.symbol)
