"""
F1 DYNAMIC RACE SIMULATOR
=========================

A true dynamic systems model that simulates race scenarios lap-by-lap.

This model:
1. DESCRIBES: Current race state at any time step (lap)
2. PREDICTS: Future race outcomes under different strategies
3. PRESCRIBES: Optimal pit strategy for minimum race time

KEY DIFFERENCE FROM STATIC MODEL:
- Static model: Analyzes historical data (validation)
- Dynamic model: Simulates future scenarios with controllable decisions

CONTROL PARAMETERS (User-defined):
- Pit stop laps (when to pit)
- Tire compound choices
- Push level (aggressive vs tire conservation)

STATE VARIABLES (Evolve over time):
- Tire degradation
- Fuel load
- Track wetness (drying curve)
- Cumulative race time
- Lap time performance
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
#                          CONFIGURATION
# ============================================================================

# Load calibrated parameters from static model
PARAMS = {
    'drying_rate': 0.070,
    'deg_medium': 0.000700,
    'deg_soft': 0.001000,      # Estimated (faster than medium)
    'deg_hard': 0.000400,      # Estimated (slower than medium)
    'deg_inter': 0.000030,
    'inter_optimal_wet': 0.210,
    'inter_dry_penalty': 0.048,
    'slick_wet_penalty': 0.060
}

# Race configuration
RACE_LAPS = 70
BASELINE_LAP_TIME = 75.0  # seconds (Monaco-like street circuit)
PIT_STOP_TIME_LOSS = 22.0  # seconds lost in pit lane

# Track conditions
INITIAL_WETNESS = 0.40  # Starting wetness (0=dry, 1=wet)

# Fuel effect (lighter car = faster lap)
FUEL_EFFECT_PER_LAP = 0.03  # seconds faster per lap due to fuel burn

# Tire compound properties
TIRE_COMPOUNDS = {
    'SOFT': {
        'initial_performance': 0.98,  # 2% faster than baseline initially
        'degradation_rate': PARAMS['deg_soft'],
        'optimal_wetness_range': (0.0, 0.1)
    },
    'MEDIUM': {
        'initial_performance': 1.00,  # baseline
        'degradation_rate': PARAMS['deg_medium'],
        'optimal_wetness_range': (0.0, 0.1)
    },
    'HARD': {
        'initial_performance': 1.02,  # 2% slower initially
        'degradation_rate': PARAMS['deg_hard'],
        'optimal_wetness_range': (0.0, 0.1)
    },
    'INTERMEDIATE': {
        'initial_performance': 0.95,  # 5% faster in optimal wet
        'degradation_rate': PARAMS['deg_inter'],
        'optimal_wetness_range': (0.15, 0.3)
    }
}

# ============================================================================
#                          DYNAMIC SIMULATION ENGINE
# ============================================================================

class RaceSimulator:
    """
    Dynamic race simulation engine.

    This class implements a discrete-event simulation where:
    - Time steps = Laps
    - State variables update each lap
    - Control decisions affect future state
    """

    def __init__(self, race_laps, baseline_lap_time, initial_wetness):
        self.race_laps = race_laps
        self.baseline_lap_time = baseline_lap_time
        self.initial_wetness = initial_wetness
        self.reset()

    def reset(self):
        """Initialize state variables"""
        self.current_lap = 1
        self.tire_age = 0
        self.current_compound = None
        self.cumulative_time = 0.0
        self.fuel_laps_remaining = self.race_laps
        self.history = []

    def calculate_wetness(self, lap):
        """
        Track wetness evolution (exponential drying)

        This is a state variable that changes over time based on
        environmental physics (drying rate).
        """
        return max(self.initial_wetness * np.exp(-PARAMS['drying_rate'] * (lap - 1)), 0.0)

    def calculate_tire_performance_factor(self, compound, tire_age, wetness):
        """
        Calculate tire performance based on wear and conditions.

        This implements the physics of tire degradation:
        - Degrades linearly with age
        - Penalty for wrong wetness conditions
        """
        tire_props = TIRE_COMPOUNDS[compound]

        # Base performance degradation
        base_perf = tire_props['initial_performance']
        degradation = tire_props['degradation_rate'] * tire_age

        # Wetness penalty for slick tires
        if compound in ['SOFT', 'MEDIUM', 'HARD']:
            wet_penalty = PARAMS['slick_wet_penalty'] * wetness
            performance_factor = base_perf + degradation + wet_penalty

        # Intermediate tire performance (optimal in wet)
        else:  # INTERMEDIATE
            wet_dev = abs(wetness - PARAMS['inter_optimal_wet'])
            if wetness < PARAMS['inter_optimal_wet']:
                # Too dry
                wet_penalty = PARAMS['inter_dry_penalty'] * (wet_dev / (PARAMS['inter_optimal_wet'] + 0.01))
            else:
                # Too wet
                wet_penalty = 0.05 * wet_dev

            performance_factor = base_perf + degradation + wet_penalty

        return performance_factor

    def calculate_fuel_effect(self, laps_remaining):
        """
        Fuel load effect on lap time.

        Lighter car (less fuel) = faster lap times.
        This is a state variable that decreases linearly over time.
        """
        laps_burned = self.race_laps - laps_remaining
        return -FUEL_EFFECT_PER_LAP * laps_burned  # Negative = faster

    def simulate_lap(self, compound):
        """
        Simulate one lap (time step) of the race.

        This is the core state transition function:
        State(t+1) = f(State(t), Control(t), Environment(t))
        """
        # Get environmental state
        wetness = self.calculate_wetness(self.current_lap)

        # Calculate performance factors
        tire_factor = self.calculate_tire_performance_factor(
            compound, self.tire_age, wetness
        )
        fuel_effect = self.calculate_fuel_effect(self.fuel_laps_remaining)

        # Calculate lap time (state output)
        lap_time = self.baseline_lap_time * tire_factor + fuel_effect

        # Update state variables
        self.tire_age += 1
        self.fuel_laps_remaining -= 1
        self.cumulative_time += lap_time

        # Record state history
        self.history.append({
            'lap': self.current_lap,
            'compound': compound,
            'tire_age': self.tire_age,
            'wetness': wetness,
            'lap_time': lap_time,
            'cumulative_time': self.cumulative_time,
            'tire_factor': tire_factor
        })

        self.current_lap += 1

        return lap_time

    def pit_stop(self, new_compound):
        """
        Perform a pit stop (control action).

        This is a discrete control decision that:
        - Resets tire age state variable
        - Changes compound (control parameter)
        - Adds time penalty
        """
        self.tire_age = 0
        self.cumulative_time += PIT_STOP_TIME_LOSS
        self.current_compound = new_compound

        # Record pit stop in history
        if self.history:
            self.history[-1]['pit_stop'] = True
            self.history[-1]['new_compound'] = new_compound

    def run_strategy(self, strategy):
        """
        Execute a complete race strategy.

        Strategy = sequence of control decisions over time.

        Args:
            strategy: List of (pit_lap, compound) tuples
                     e.g., [(1, 'INTERMEDIATE'), (25, 'SOFT'), (50, 'MEDIUM')]

        Returns:
            Total race time and lap-by-lap history
        """
        self.reset()

        # Sort strategy by lap number
        strategy = sorted(strategy, key=lambda x: x[0])
        strategy_idx = 0

        # Start with first compound
        if strategy:
            _, first_compound = strategy[0]
            self.current_compound = first_compound
            strategy_idx = 1

        # Simulate each lap
        for lap in range(1, self.race_laps + 1):
            # Check if pit stop scheduled
            if strategy_idx < len(strategy) and lap == strategy[strategy_idx][0]:
                _, new_compound = strategy[strategy_idx]
                self.pit_stop(new_compound)
                strategy_idx += 1

            # Simulate lap
            self.simulate_lap(self.current_compound)

        return self.cumulative_time, pd.DataFrame(self.history)

# ============================================================================
#                          SCENARIO DEFINITIONS
# ============================================================================

def define_strategies():
    """
    Define different race strategies to test.

    Each strategy is a sequence of decisions:
    - When to pit (lap number)
    - Which tire to use

    This is the CONTROL SPACE we're exploring.
    """
    strategies = {
        '1-Stop Inter>Soft': [
            (1, 'INTERMEDIATE'),
            (30, 'SOFT')
        ],

        '1-Stop Inter>Medium': [
            (1, 'INTERMEDIATE'),
            (30, 'MEDIUM')
        ],

        '2-Stop Inter>Soft>Soft': [
            (1, 'INTERMEDIATE'),
            (25, 'SOFT'),
            (50, 'SOFT')
        ],

        '2-Stop Inter>Soft>Medium': [
            (1, 'INTERMEDIATE'),
            (25, 'SOFT'),
            (50, 'MEDIUM')
        ],

        '2-Stop Inter>Medium>Soft': [
            (1, 'INTERMEDIATE'),
            (30, 'MEDIUM'),
            (55, 'SOFT')
        ],

        '1-Stop Medium Only': [
            (1, 'MEDIUM'),
            (40, 'MEDIUM')
        ],

        'Aggressive 3-Stop': [
            (1, 'INTERMEDIATE'),
            (20, 'SOFT'),
            (38, 'SOFT'),
            (56, 'SOFT')
        ],

        'Conservative 1-Stop Hard': [
            (1, 'INTERMEDIATE'),
            (35, 'HARD')
        ]
    }

    return strategies

# ============================================================================
#                          MAIN SIMULATION
# ============================================================================

def main():
    print("="*80)
    print("F1 DYNAMIC RACE SIMULATOR")
    print("="*80)
    print("\nThis is a DYNAMIC SYSTEMS MODEL that:")
    print("  1. DESCRIBES: Race state at each lap (tire wear, fuel, wetness)")
    print("  2. PREDICTS: Total race time for different strategies")
    print("  3. PRESCRIBES: Optimal pit strategy")
    print("\nRace Configuration:")
    print(f"  Total Laps: {RACE_LAPS}")
    print(f"  Baseline Lap Time: {BASELINE_LAP_TIME}s")
    print(f"  Initial Track Wetness: {INITIAL_WETNESS:.2f} (drying over time)")
    print(f"  Pit Stop Time Loss: {PIT_STOP_TIME_LOSS}s")

    # Initialize simulator
    simulator = RaceSimulator(RACE_LAPS, BASELINE_LAP_TIME, INITIAL_WETNESS)

    # Get strategies to test
    strategies = define_strategies()

    print(f"\n{'='*80}")
    print(f"SIMULATING {len(strategies)} DIFFERENT STRATEGIES")
    print(f"{'='*80}\n")

    # Run simulations
    results = {}
    for name, strategy in strategies.items():
        print(f"Running: {name}...", end=' ')
        total_time, history = simulator.run_strategy(strategy)
        results[name] = {
            'total_time': total_time,
            'history': history,
            'strategy': strategy,
            'num_stops': len(strategy) - 1  # First entry is starting compound
        }
        print(f"Total Time: {total_time/60:.2f} minutes ({total_time:.1f}s)")

    # Find optimal strategy
    best_strategy = min(results.items(), key=lambda x: x[1]['total_time'])
    worst_strategy = max(results.items(), key=lambda x: x[1]['total_time'])

    print(f"\n{'='*80}")
    print("PRESCRIPTIVE ANALYSIS - OPTIMAL STRATEGY")
    print(f"{'='*80}\n")
    print(f"*** BEST STRATEGY: {best_strategy[0]}")
    print(f"   Total Race Time: {best_strategy[1]['total_time']/60:.2f} minutes")
    print(f"   Number of Pit Stops: {best_strategy[1]['num_stops']}")
    print(f"   Strategy:")
    for lap, compound in best_strategy[1]['strategy']:
        if lap == 1:
            print(f"      Start on {compound}")
        else:
            print(f"      Lap {lap}: Pit for {compound}")

    time_saved = worst_strategy[1]['total_time'] - best_strategy[1]['total_time']
    print(f"\n   >> Time saved vs worst strategy: {time_saved:.1f}s ({time_saved/60:.2f} min)")

    print(f"\n*** STRATEGY COMPARISON (Sorted by Performance):")
    print(f"{'='*80}")
    sorted_results = sorted(results.items(), key=lambda x: x[1]['total_time'])

    for rank, (name, data) in enumerate(sorted_results, 1):
        time_diff = data['total_time'] - best_strategy[1]['total_time']
        print(f"{rank}. {name:30s} | "
              f"{data['total_time']/60:6.2f} min | "
              f"+{time_diff:5.1f}s | "
              f"{data['num_stops']} stops")

    # Create visualizations
    create_visualizations(results, best_strategy[0])

    print(f"\n{'='*80}")
    print("SIMULATION COMPLETE")
    print(f"{'='*80}")
    print(f"\nOutputs saved to 'dynamic_simulation_results/' directory")

# ============================================================================
#                          VISUALIZATION
# ============================================================================

def create_visualizations(results, best_strategy_name):
    """Create comprehensive visualizations of simulation results"""

    output_dir = Path('dynamic_simulation_results')
    output_dir.mkdir(exist_ok=True)

    # Professional color scheme
    BLUE_DARK = '#0A2647'
    BLUE_PRIMARY = '#144272'
    WHITE = '#FFFFFF'

    colors = ['#FF6B35', '#4ECDC4', '#95E1D3', '#F38181', '#AA96DA',
              '#FCBAD3', '#A8D8EA', '#FFCB77']

    # ========================================================================
    # GRAPH 1: Lap Time Evolution for All Strategies
    # ========================================================================
    fig1, ax1 = plt.subplots(figsize=(16, 9), facecolor=BLUE_DARK)
    ax1.set_facecolor(BLUE_PRIMARY)

    for idx, (name, data) in enumerate(results.items()):
        history = data['history']
        linestyle = '-' if name == best_strategy_name else '--'
        linewidth = 3 if name == best_strategy_name else 1.5
        alpha = 1.0 if name == best_strategy_name else 0.6

        ax1.plot(history['lap'], history['lap_time'],
                label=name, color=colors[idx % len(colors)],
                linestyle=linestyle, linewidth=linewidth, alpha=alpha)

    ax1.set_xlabel('Lap Number', fontweight='bold', fontsize=14, color=WHITE)
    ax1.set_ylabel('Lap Time (seconds)', fontweight='bold', fontsize=14, color=WHITE)
    ax1.set_title('Dynamic Simulation: Lap Time Evolution by Strategy\n(Bold line = Optimal Strategy)',
                 fontweight='bold', fontsize=18, color=WHITE, pad=20)
    ax1.legend(fontsize=10, loc='upper left', framealpha=0.9,
              facecolor=BLUE_PRIMARY, edgecolor=WHITE, labelcolor=WHITE)
    ax1.grid(True, alpha=0.3, color=WHITE)
    ax1.tick_params(colors=WHITE, labelsize=11)

    plt.tight_layout()
    plt.savefig(output_dir / 'strategy_comparison_lap_times.png',
                dpi=300, bbox_inches='tight', facecolor=BLUE_DARK)
    print(f"\n[OK] Saved: strategy_comparison_lap_times.png")
    plt.close()

    # ========================================================================
    # GRAPH 2: Cumulative Race Time (Race Position Evolution)
    # ========================================================================
    fig2, ax2 = plt.subplots(figsize=(16, 9), facecolor=BLUE_DARK)
    ax2.set_facecolor(BLUE_PRIMARY)

    for idx, (name, data) in enumerate(results.items()):
        history = data['history']
        linestyle = '-' if name == best_strategy_name else '--'
        linewidth = 3 if name == best_strategy_name else 1.5
        alpha = 1.0 if name == best_strategy_name else 0.6

        ax2.plot(history['lap'], history['cumulative_time'] / 60,
                label=name, color=colors[idx % len(colors)],
                linestyle=linestyle, linewidth=linewidth, alpha=alpha)

    ax2.set_xlabel('Lap Number', fontweight='bold', fontsize=14, color=WHITE)
    ax2.set_ylabel('Cumulative Race Time (minutes)', fontweight='bold', fontsize=14, color=WHITE)
    ax2.set_title('Dynamic Simulation: Race Time Progression\n(Shows when different strategies gain/lose time)',
                 fontweight='bold', fontsize=18, color=WHITE, pad=20)
    ax2.legend(fontsize=10, loc='upper left', framealpha=0.9,
              facecolor=BLUE_PRIMARY, edgecolor=WHITE, labelcolor=WHITE)
    ax2.grid(True, alpha=0.3, color=WHITE)
    ax2.tick_params(colors=WHITE, labelsize=11)

    plt.tight_layout()
    plt.savefig(output_dir / 'strategy_comparison_cumulative.png',
                dpi=300, bbox_inches='tight', facecolor=BLUE_DARK)
    print(f"[OK] Saved: strategy_comparison_cumulative.png")
    plt.close()

    # ========================================================================
    # GRAPH 3: Optimal Strategy Detail
    # ========================================================================
    best_history = results[best_strategy_name]['history']

    fig3, (ax3a, ax3b) = plt.subplots(2, 1, figsize=(16, 12), facecolor=BLUE_DARK)

    # Subplot A: Lap times with tire compound markers
    ax3a.set_facecolor(BLUE_PRIMARY)

    compound_colors = {
        'SOFT': '#FF1E1E',
        'MEDIUM': '#FFD700',
        'HARD': '#E8E8E8',
        'INTERMEDIATE': '#00FF00'
    }

    for compound in best_history['compound'].unique():
        mask = best_history['compound'] == compound
        data = best_history[mask]
        ax3a.plot(data['lap'], data['lap_time'],
                 color=compound_colors.get(compound, '#FFFFFF'),
                 linewidth=3, label=compound, marker='o', markersize=4)

    ax3a.set_xlabel('Lap Number', fontweight='bold', fontsize=13, color=WHITE)
    ax3a.set_ylabel('Lap Time (seconds)', fontweight='bold', fontsize=13, color=WHITE)
    ax3a.set_title(f'Optimal Strategy Detail: {best_strategy_name}',
                  fontweight='bold', fontsize=16, color=WHITE, pad=15)
    ax3a.legend(fontsize=11, framealpha=0.9, facecolor=BLUE_PRIMARY,
               edgecolor=WHITE, labelcolor=WHITE)
    ax3a.grid(True, alpha=0.3, color=WHITE)
    ax3a.tick_params(colors=WHITE, labelsize=11)

    # Subplot B: State variables evolution
    ax3b.set_facecolor(BLUE_PRIMARY)

    ax3b_twin = ax3b.twinx()

    ln1 = ax3b.plot(best_history['lap'], best_history['tire_age'],
                   color='#FF6B35', linewidth=2.5, label='Tire Age', marker='.')
    ln2 = ax3b_twin.plot(best_history['lap'], best_history['wetness'],
                        color='#4ECDC4', linewidth=2.5, label='Track Wetness', marker='.')

    ax3b.set_xlabel('Lap Number', fontweight='bold', fontsize=13, color=WHITE)
    ax3b.set_ylabel('Tire Age (laps)', fontweight='bold', fontsize=13, color=WHITE)
    ax3b_twin.set_ylabel('Track Wetness', fontweight='bold', fontsize=13, color=WHITE)
    ax3b.set_title('State Variables Evolution (Optimal Strategy)',
                  fontweight='bold', fontsize=16, color=WHITE, pad=15)

    lns = ln1 + ln2
    labs = [l.get_label() for l in lns]
    ax3b.legend(lns, labs, fontsize=11, framealpha=0.9, facecolor=BLUE_PRIMARY,
               edgecolor=WHITE, labelcolor=WHITE)

    ax3b.grid(True, alpha=0.3, color=WHITE)
    ax3b.tick_params(colors=WHITE, labelsize=11)
    ax3b_twin.tick_params(colors=WHITE, labelsize=11)
    ax3b.spines['left'].set_color(WHITE)
    ax3b.spines['right'].set_color(WHITE)
    ax3b.spines['top'].set_color(WHITE)
    ax3b.spines['bottom'].set_color(WHITE)
    ax3b_twin.spines['left'].set_color(WHITE)
    ax3b_twin.spines['right'].set_color(WHITE)
    ax3b_twin.spines['top'].set_color(WHITE)
    ax3b_twin.spines['bottom'].set_color(WHITE)

    plt.tight_layout()
    plt.savefig(output_dir / 'optimal_strategy_detail.png',
                dpi=300, bbox_inches='tight', facecolor=BLUE_DARK)
    print(f"[OK] Saved: optimal_strategy_detail.png")
    plt.close()

    # ========================================================================
    # GRAPH 4: Strategy Performance Bar Chart
    # ========================================================================
    fig4, ax4 = plt.subplots(figsize=(14, 10), facecolor=BLUE_DARK)
    ax4.set_facecolor(BLUE_PRIMARY)

    sorted_results = sorted(results.items(), key=lambda x: x[1]['total_time'])
    names = [name for name, _ in sorted_results]
    times = [data['total_time']/60 for _, data in sorted_results]
    bar_colors = [colors[0] if name == best_strategy_name else colors[i % len(colors)]
                  for i, (name, _) in enumerate(sorted_results)]

    bars = ax4.barh(names, times, color=bar_colors, edgecolor=WHITE, linewidth=1.5)

    # Add time labels
    for idx, (bar, time) in enumerate(zip(bars, times)):
        ax4.text(time + 0.1, bar.get_y() + bar.get_height()/2,
                f'{time:.2f} min',
                va='center', fontsize=11, color=WHITE, fontweight='bold')

    ax4.set_xlabel('Total Race Time (minutes)', fontweight='bold', fontsize=14, color=WHITE)
    ax4.set_title('Strategy Performance Comparison\n(Lower is Better)',
                 fontweight='bold', fontsize=18, color=WHITE, pad=20)
    ax4.grid(True, alpha=0.3, color=WHITE, axis='x')
    ax4.tick_params(colors=WHITE, labelsize=11)
    ax4.spines['left'].set_color(WHITE)
    ax4.spines['right'].set_color(WHITE)
    ax4.spines['top'].set_color(WHITE)
    ax4.spines['bottom'].set_color(WHITE)

    plt.tight_layout()
    plt.savefig(output_dir / 'strategy_performance_comparison.png',
                dpi=300, bbox_inches='tight', facecolor=BLUE_DARK)
    print(f"[OK] Saved: strategy_performance_comparison.png")
    plt.close()

if __name__ == "__main__":
    main()
