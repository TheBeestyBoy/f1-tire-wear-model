"""
F1 RACE STRATEGY OPTIMIZER
===========================

PRESCRIPTIVE MODEL: Automatically finds optimal pit strategy using optimization.

This demonstrates the full power of dynamic systems modeling:
1. DESCRIPTIVE: Models race state evolution
2. PREDICTIVE: Simulates outcomes for any strategy
3. PRESCRIPTIVE: OPTIMIZES to find best strategy automatically

Optimization Method: Exhaustive search over feasible strategy space
- Variables: Number of stops (0-3), pit laps, tire compounds
- Objective: Minimize total race time
- Constraints: Tire rules, minimum stint lengths
"""

import numpy as np
import pandas as pd
from pathlib import Path
from dynamic_race_simulator import RaceSimulator, RACE_LAPS, BASELINE_LAP_TIME, INITIAL_WETNESS

# ============================================================================
#                          OPTIMIZATION PARAMETERS
# ============================================================================

# Tire compound options
COMPOUNDS = ['SOFT', 'MEDIUM', 'HARD', 'INTERMEDIATE']

# Strategy constraints
MIN_STINT_LENGTH = 10  # Minimum laps on a tire set
MAX_STOPS = 3          # Maximum pit stops to consider

# Optimization search granularity
LAP_INTERVAL = 5       # Test pit stops every N laps

# ============================================================================
#                          STRATEGY SPACE GENERATOR
# ============================================================================

def generate_all_strategies(race_laps, max_stops, compounds, min_stint_length, lap_interval):
    """
    Generate all feasible pit strategies.

    This creates the CONTROL SPACE we're searching over.

    Each strategy is defined by:
    - Number of pit stops
    - Lap number for each pit stop
    - Tire compound for each stint

    Returns:
        List of (strategy_dict, strategy_tuple) pairs
    """
    all_strategies = []

    # 0-stop strategies (start to finish on one compound)
    for compound in compounds:
        strategy = [(1, compound)]
        all_strategies.append({
            'name': f'0-Stop {compound}',
            'strategy': strategy
        })

    # 1-stop strategies
    for pit_lap in range(min_stint_length, race_laps - min_stint_length + 1, lap_interval):
        for start_compound in compounds:
            for second_compound in compounds:
                strategy = [(1, start_compound), (pit_lap, second_compound)]
                all_strategies.append({
                    'name': f'1-Stop L{pit_lap} {start_compound}>{second_compound}',
                    'strategy': strategy
                })

    # 2-stop strategies
    if max_stops >= 2:
        for pit1 in range(min_stint_length, race_laps - 2*min_stint_length + 1, lap_interval):
            for pit2 in range(pit1 + min_stint_length, race_laps - min_stint_length + 1, lap_interval):
                for c1 in compounds:
                    for c2 in compounds:
                        for c3 in compounds:
                            strategy = [(1, c1), (pit1, c2), (pit2, c3)]
                            all_strategies.append({
                                'name': f'2-Stop L{pit1},L{pit2} {c1}>{c2}>{c3}',
                                'strategy': strategy
                            })

    # 3-stop strategies (simplified - only include limited combinations)
    if max_stops >= 3:
        lap_step = max(lap_interval * 2, 15)  # Larger step for 3-stops to reduce search space
        for pit1 in range(min_stint_length, race_laps - 3*min_stint_length + 1, lap_step):
            for pit2 in range(pit1 + min_stint_length, race_laps - 2*min_stint_length + 1, lap_step):
                for pit3 in range(pit2 + min_stint_length, race_laps - min_stint_length + 1, lap_step):
                    # Only test most common compounds for 3-stops (reduce search space)
                    for c1 in ['INTERMEDIATE', 'MEDIUM']:
                        for c2 in ['SOFT', 'MEDIUM']:
                            for c3 in ['SOFT', 'MEDIUM']:
                                for c4 in ['SOFT', 'MEDIUM']:
                                    strategy = [(1, c1), (pit1, c2), (pit2, c3), (pit3, c4)]
                                    all_strategies.append({
                                        'name': f'3-Stop {c1}>{c2}>{c3}>{c4}',
                                        'strategy': strategy
                                    })

    return all_strategies

# ============================================================================
#                          OPTIMIZATION ENGINE
# ============================================================================

def optimize_strategy(race_laps, baseline_lap_time, initial_wetness, verbose=True):
    """
    Find optimal pit strategy using exhaustive search.

    This is the PRESCRIPTIVE component:
    - Tests all feasible strategies
    - Evaluates each using dynamic simulation
    - Returns best strategy (minimum race time)

    Args:
        race_laps: Total race laps
        baseline_lap_time: Clean lap time (seconds)
        initial_wetness: Starting track wetness (0-1)
        verbose: Print progress

    Returns:
        Dictionary with optimal strategy and results
    """
    if verbose:
        print("="*80)
        print("PRESCRIPTIVE OPTIMIZATION - FINDING OPTIMAL STRATEGY")
        print("="*80)

    # Generate strategy space
    if verbose:
        print("\n[1/3] Generating strategy space...")

    strategies = generate_all_strategies(
        race_laps=race_laps,
        max_stops=MAX_STOPS,
        compounds=COMPOUNDS,
        min_stint_length=MIN_STINT_LENGTH,
        lap_interval=LAP_INTERVAL
    )

    if verbose:
        print(f"      Generated {len(strategies)} feasible strategies")
        print(f"      Search space:")
        print(f"        - Max stops: {MAX_STOPS}")
        print(f"        - Compounds: {COMPOUNDS}")
        print(f"        - Min stint: {MIN_STINT_LENGTH} laps")
        print(f"        - Pit lap interval: {LAP_INTERVAL} laps")

    # Initialize simulator
    simulator = RaceSimulator(race_laps, baseline_lap_time, initial_wetness)

    # Evaluate all strategies
    if verbose:
        print(f"\n[2/3] Evaluating strategies (testing {len(strategies)} scenarios)...")

    results = []
    for idx, strat_dict in enumerate(strategies):
        if verbose and idx % 100 == 0:
            print(f"      Progress: {idx}/{len(strategies)} strategies tested...", end='\r')

        try:
            total_time, history = simulator.run_strategy(strat_dict['strategy'])
            results.append({
                'name': strat_dict['name'],
                'strategy': strat_dict['strategy'],
                'total_time': total_time,
                'num_stops': len(strat_dict['strategy']) - 1,
                'history': history
            })
        except Exception as e:
            # Skip invalid strategies
            if verbose:
                print(f"\n      Warning: Strategy '{strat_dict['name']}' failed: {e}")
            continue

    if verbose:
        print(f"\n      Completed: {len(results)} valid strategies evaluated")

    # Find optimal
    if verbose:
        print(f"\n[3/3] Finding optimal solution...")

    results_sorted = sorted(results, key=lambda x: x['total_time'])
    optimal = results_sorted[0]
    worst = results_sorted[-1]

    if verbose:
        print(f"\n{'='*80}")
        print("OPTIMIZATION RESULTS")
        print(f"{'='*80}\n")
        print(f"*** OPTIMAL STRATEGY FOUND ***")
        print(f"  Strategy: {optimal['name']}")
        print(f"  Total Race Time: {optimal['total_time']/60:.2f} minutes ({optimal['total_time']:.1f}s)")
        print(f"  Number of Stops: {optimal['num_stops']}")
        print(f"  Details:")
        for lap, compound in optimal['strategy']:
            if lap == 1:
                print(f"    - Start on {compound}")
            else:
                print(f"    - Lap {lap}: Pit for {compound}")

        time_saved = worst['total_time'] - optimal['total_time']
        print(f"\n  >> Time saved vs worst strategy: {time_saved:.1f}s ({time_saved/60:.2f} min)")
        print(f"  >> Total strategies evaluated: {len(results)}")

        # Show top 10
        print(f"\n{'='*80}")
        print("TOP 10 STRATEGIES")
        print(f"{'='*80}")
        for rank, result in enumerate(results_sorted[:10], 1):
            time_diff = result['total_time'] - optimal['total_time']
            print(f"{rank:2d}. {result['name']:50s} | "
                  f"{result['total_time']/60:6.2f} min | "
                  f"+{time_diff:5.1f}s")

    return {
        'optimal': optimal,
        'all_results': results_sorted,
        'search_space_size': len(strategies),
        'valid_strategies': len(results)
    }

# ============================================================================
#                          SENSITIVITY ANALYSIS
# ============================================================================

def sensitivity_analysis(optimal_strategy, race_laps, baseline_lap_time, initial_wetness):
    """
    Analyze how sensitive the optimal strategy is to parameter changes.

    Tests:
    - Different initial wetness levels
    - Different baseline lap times
    - Different race lengths
    """
    print(f"\n{'='*80}")
    print("SENSITIVITY ANALYSIS")
    print(f"{'='*80}")
    print("\nTesting how optimal strategy changes with different conditions...\n")

    simulator = RaceSimulator(race_laps, baseline_lap_time, initial_wetness)

    # Test 1: Wetness sensitivity
    print("[1] Wetness Sensitivity:")
    print("-" * 60)
    wetness_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    for wetness in wetness_levels:
        sim_temp = RaceSimulator(race_laps, baseline_lap_time, wetness)
        total_time, _ = sim_temp.run_strategy(optimal_strategy['strategy'])
        print(f"  Wetness {wetness:.1f}: Race time = {total_time/60:.2f} min")

    # Test 2: Baseline lap time sensitivity
    print(f"\n[2] Lap Time Sensitivity:")
    print("-" * 60)
    lap_times = [baseline_lap_time * 0.95, baseline_lap_time, baseline_lap_time * 1.05]
    for lap_time in lap_times:
        sim_temp = RaceSimulator(race_laps, lap_time, initial_wetness)
        total_time, _ = sim_temp.run_strategy(optimal_strategy['strategy'])
        pct = ((lap_time / baseline_lap_time) - 1) * 100
        print(f"  Baseline {lap_time:.1f}s ({pct:+.1f}%): Race time = {total_time/60:.2f} min")

    # Test 3: Race length sensitivity
    print(f"\n[3] Race Length Sensitivity:")
    print("-" * 60)
    race_lengths = [50, 60, 70, 80]
    for length in race_lengths:
        if length != race_laps:
            # Keep same stint proportions
            scale = length / race_laps
            scaled_strategy = [
                (1 if lap == 1 else max(2, int(lap * scale)), compound)
                for lap, compound in optimal_strategy['strategy']
            ]
        else:
            scaled_strategy = optimal_strategy['strategy']

        sim_temp = RaceSimulator(length, baseline_lap_time, initial_wetness)
        total_time, _ = sim_temp.run_strategy(scaled_strategy)
        print(f"  {length} laps: Race time = {total_time/60:.2f} min")

# ============================================================================
#                          MAIN
# ============================================================================

def main():
    print("\n" + "="*80)
    print("F1 RACE STRATEGY OPTIMIZER")
    print("="*80)
    print("\nPRESCRIPTIVE DYNAMIC SYSTEMS MODEL")
    print("-" * 80)
    print("This model finds the OPTIMAL race strategy by:")
    print("  1. Generating all feasible strategies (control space)")
    print("  2. Simulating each strategy (dynamic prediction)")
    print("  3. Selecting strategy with minimum race time (optimization)")
    print("="*80)

    # Run optimization
    optimization_result = optimize_strategy(
        race_laps=RACE_LAPS,
        baseline_lap_time=BASELINE_LAP_TIME,
        initial_wetness=INITIAL_WETNESS,
        verbose=True
    )

    # Sensitivity analysis
    sensitivity_analysis(
        optimal_strategy=optimization_result['optimal'],
        race_laps=RACE_LAPS,
        baseline_lap_time=BASELINE_LAP_TIME,
        initial_wetness=INITIAL_WETNESS
    )

    # Save results
    print(f"\n{'='*80}")
    print("SAVING RESULTS")
    print(f"{'='*80}")

    output_dir = Path('optimization_results')
    output_dir.mkdir(exist_ok=True)

    # Save optimal strategy details
    optimal_df = optimization_result['optimal']['history']
    optimal_file = output_dir / 'optimal_strategy_lapbylap.csv'
    optimal_df.to_csv(optimal_file, index=False)
    print(f"\n[OK] Optimal strategy lap-by-lap saved to: {optimal_file}")

    # Save top strategies
    top_strategies = []
    for rank, result in enumerate(optimization_result['all_results'][:20], 1):
        top_strategies.append({
            'Rank': rank,
            'Strategy': result['name'],
            'Total_Time_Minutes': result['total_time'] / 60,
            'Total_Time_Seconds': result['total_time'],
            'Num_Stops': result['num_stops'],
            'Time_Delta_Seconds': result['total_time'] - optimization_result['optimal']['total_time']
        })

    top_df = pd.DataFrame(top_strategies)
    top_file = output_dir / 'top_20_strategies.csv'
    top_df.to_csv(top_file, index=False)
    print(f"[OK] Top 20 strategies saved to: {top_file}")

    print(f"\n{'='*80}")
    print("OPTIMIZATION COMPLETE")
    print(f"{'='*80}")
    print(f"\nKey Findings:")
    print(f"  - Searched {optimization_result['search_space_size']} possible strategies")
    print(f"  - Evaluated {optimization_result['valid_strategies']} valid strategies")
    print(f"  - Optimal: {optimization_result['optimal']['name']}")
    print(f"  - Race Time: {optimization_result['optimal']['total_time']/60:.2f} minutes")
    print(f"\nThis demonstrates a true PRESCRIPTIVE model that recommends optimal decisions.")

if __name__ == "__main__":
    main()
