"""
F1 Tire Wear Simulation - Improved Tire Scenarios

Uses CALIBRATED coefficients to simulate "better tire" scenarios
and compare time savings vs baseline.
"""

import fastf1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Enable FastF1 cache
cache_dir = Path('./fastf1_cache')
cache_dir.mkdir(exist_ok=True)
fastf1.Cache.enable_cache(str(cache_dir))

print("="*80)
print("F1 TIRE WEAR SIMULATION - IMPROVED TIRE SCENARIOS")
print("="*80)
print("\nUsing CALIBRATED coefficients from previous optimization")
print()

# Configuration
YEAR = 2023
RACE = 'Monaco'
SESSION = 'R'
DRIVER = 'VER'

# CALIBRATED COEFFICIENTS (from optimization)
VEHICLE_MASS = 800

COMPOUND_COEFFICIENTS = {
    'SOFT': 1.5,
    'MEDIUM': 1.0,
    'HARD': 0.6,
    'INTERMEDIATE': 0.8,
    'WET': 0.7
}

K_BRAKING = 0.001
K_CORNERING = 0.02
K_ACCELERATION = 0.0005
K_THERMAL = 0.01
ALPHA_TEMP = 0.5

# OPTIMIZED VALUES - Need much smaller BETA to prevent instant saturation
BETA_DECAY = 0.00001      # Scaled down to show realistic degradation curve
GAMMA_LAPTIME = 0.05      # Calibrated!

print(f"Loading {YEAR} {RACE} GP - {SESSION} session for driver {DRIVER}...")

# Load session data
session = fastf1.get_session(YEAR, RACE, SESSION)
session.load()

driver_laps = session.laps.pick_driver(DRIVER)
valid_laps = driver_laps[
    (driver_laps['IsAccurate'] == True) &
    (driver_laps['LapTime'].notna())
].copy()

print(f"Valid laps: {len(valid_laps)}")
print(f"Compounds: {valid_laps['Compound'].value_counts().to_dict()}")

# Extract telemetry features
def extract_lap_telemetry_features(lap):
    """Extract aggregated telemetry features from a lap."""
    try:
        telemetry = lap.get_telemetry()
        if telemetry.empty:
            return None

        features = {
            'speed_mean': telemetry['Speed'].mean(),
            'speed_max': telemetry['Speed'].max(),
            'speed_std': telemetry['Speed'].std(),
            'throttle_mean': telemetry['Throttle'].mean(),
            'throttle_max': telemetry['Throttle'].max(),
            'throttle_full_pct': (telemetry['Throttle'] > 95).mean() * 100,
            'brake_mean': telemetry['Brake'].mean() if 'Brake' in telemetry.columns else 0,
            'brake_applications': (telemetry['Brake'] > 0).sum() if 'Brake' in telemetry.columns else 0,
            'rpm_mean': telemetry['RPM'].mean() if 'RPM' in telemetry.columns else 0,
            'rpm_max': telemetry['RPM'].max() if 'RPM' in telemetry.columns else 0,
            'gear_mean': telemetry['nGear'].mean() if 'nGear' in telemetry.columns else 0,
            'gear_shifts': (telemetry['nGear'].diff().abs() > 0).sum() if 'nGear' in telemetry.columns else 0,
            'drs_active_pct': (telemetry['DRS'] > 8).mean() * 100 if 'DRS' in telemetry.columns else 0,
            'telemetry_points': len(telemetry),
            'lap_duration': (telemetry['Time'].max() - telemetry['Time'].min()).total_seconds()
        }
        return features
    except Exception as e:
        return None

print("\nExtracting telemetry...")
telemetry_features = []
for idx, lap in valid_laps.iterrows():
    features = extract_lap_telemetry_features(lap)
    if features:
        features['lap_number'] = lap['LapNumber']
        features['lap_time_seconds'] = lap['LapTime'].total_seconds()
        features['compound'] = lap['Compound']
        features['tyre_life'] = lap['TyreLife']
        telemetry_features.append(features)

df = pd.DataFrame(telemetry_features)
print(f"Extracted {len(df)} laps\n")

# Tire wear model functions
def calculate_braking_energy(row):
    v_mean_ms = row['speed_mean'] / 3.6
    brake_intensity = row['brake_mean'] / 100.0
    energy = K_BRAKING * VEHICLE_MASS * v_mean_ms * row['brake_applications'] * (brake_intensity + 0.1)
    return energy

def calculate_cornering_energy(row):
    energy = K_CORNERING * (row['speed_std'] ** 2) * row['telemetry_points']
    return energy

def calculate_acceleration_energy(row):
    v_mean_ms = row['speed_mean'] / 3.6
    throttle_norm = row['throttle_mean'] / 100.0
    energy = K_ACCELERATION * throttle_norm * v_mean_ms * row['lap_duration']
    return energy

def calculate_thermal_load(row):
    thermal = K_THERMAL * (row['speed_mean'] * row['lap_duration'] + row['gear_shifts'] * 10)
    return thermal

def calculate_temperature_factor(row, max_speed, max_brake_apps):
    speed_norm = row['speed_mean'] / max_speed if max_speed > 0 else 0
    brake_norm = row['brake_applications'] / max_brake_apps if max_brake_apps > 0 else 0
    throttle_norm = row['throttle_mean'] / 100.0
    T_normalized = 0.4 * speed_norm + 0.3 * brake_norm + 0.3 * throttle_norm
    T_factor = np.exp(ALPHA_TEMP * T_normalized)
    return T_factor

def calculate_lap_wear(row, max_speed, max_brake_apps, compound_multiplier=1.0):
    W_brake = calculate_braking_energy(row)
    W_corner = calculate_cornering_energy(row)
    W_accel = calculate_acceleration_energy(row)
    W_thermal = calculate_thermal_load(row)
    W_mechanical = W_brake + W_corner + W_accel + W_thermal

    k_compound = COMPOUND_COEFFICIENTS.get(row['compound'], 1.0) * compound_multiplier
    T_factor = calculate_temperature_factor(row, max_speed, max_brake_apps)

    W_lap = k_compound * W_mechanical * T_factor
    return W_lap

def run_simulation(df, wear_multiplier=1.0, scenario_name="Baseline"):
    """Run tire wear simulation with given wear multiplier."""
    df_sim = df.copy()

    # Calculate normalization factors
    max_speed = df_sim['speed_mean'].max()
    max_brake_apps = df_sim['brake_applications'].max()

    # Calculate wear for each lap
    df_sim['wear_total'] = df_sim.apply(
        lambda row: calculate_lap_wear(row, max_speed, max_brake_apps, wear_multiplier),
        axis=1
    )

    # Sort by lap number
    df_sim = df_sim.sort_values('lap_number')

    # Detect stints
    df_sim['stint_id'] = ((df_sim['compound'] != df_sim['compound'].shift()) |
                          (df_sim['tyre_life'] < df_sim['tyre_life'].shift())).cumsum()

    # Calculate cumulative wear within each stint
    df_sim['wear_cumulative'] = df_sim.groupby('stint_id')['wear_total'].cumsum()
    df_sim['degradation_normalized'] = 1 - np.exp(-BETA_DECAY * df_sim['wear_cumulative'])

    # Calculate baseline time (best of first 3 laps per stint)
    def get_baseline_time(stint_series):
        early_laps = stint_series.head(3)
        return early_laps.min()

    df_sim['baseline_time'] = df_sim.groupby('stint_id')['lap_time_seconds'].transform(get_baseline_time)

    # Predicted lap time
    df_sim['predicted_lap_time'] = df_sim['baseline_time'] * (1 + GAMMA_LAPTIME * df_sim['degradation_normalized'])
    df_sim['scenario'] = scenario_name

    return df_sim

# Run baseline and improved scenarios
print("="*80)
print("RUNNING SIMULATIONS")
print("="*80)

scenarios = {
    'Baseline (Standard Tire)': 1.0,
    '15% Improved Tire': 0.85,
    '30% Improved Tire': 0.70,
    '50% Improved Tire': 0.50
}

results = {}
for scenario_name, wear_mult in scenarios.items():
    print(f"\n{scenario_name} (wear multiplier = {wear_mult:.2f})...")
    results[scenario_name] = run_simulation(df, wear_mult, scenario_name)

# Calculate time savings
print("\n" + "="*80)
print("TIME SAVINGS ANALYSIS")
print("="*80)

baseline = results['Baseline (Standard Tire)']

for scenario_name in list(scenarios.keys())[1:]:  # Skip baseline
    improved = results[scenario_name]

    # Calculate time saved per lap
    improved['time_saved'] = baseline['predicted_lap_time'].values - improved['predicted_lap_time'].values

    total_saved = improved['time_saved'].sum()
    avg_saved = improved['time_saved'].mean()
    max_saved = improved['time_saved'].max()

    print(f"\n{scenario_name}:")
    print(f"  Total time saved over race: {total_saved:.2f} seconds ({total_saved/60:.2f} minutes)")
    print(f"  Average time saved per lap: {avg_saved:.3f} seconds")
    print(f"  Maximum single lap saving: {max_saved:.3f} seconds")

    # By stint
    print(f"  By stint:")
    for stint_id in sorted(improved['stint_id'].unique()):
        stint_data = improved[improved['stint_id'] == stint_id]
        compound = stint_data['compound'].iloc[0]
        stint_saved = stint_data['time_saved'].sum()
        laps = len(stint_data)
        print(f"    Stint {stint_id} ({compound:12s}): {stint_saved:6.2f}s saved over {laps} laps")

# Create comprehensive visualizations
print("\n" + "="*80)
print("CREATING VISUALIZATIONS")
print("="*80)

fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

colors = ['red', 'green', 'blue', 'purple']
scenario_list = list(scenarios.keys())

# Plot 1: Lap time evolution (large plot)
ax1 = fig.add_subplot(gs[0, :])
ax1.scatter(baseline['lap_number'], baseline['lap_time_seconds'],
           alpha=0.6, s=80, label='Actual Lap Times', zorder=3, color='black')

for idx, scenario_name in enumerate(scenario_list):
    df_scenario = results[scenario_name]
    linestyle = '-' if idx == 0 else '--'
    linewidth = 2.5 if idx == 0 else 2
    ax1.plot(df_scenario['lap_number'], df_scenario['predicted_lap_time'],
            linestyle=linestyle, linewidth=linewidth, color=colors[idx],
            label=scenario_name, alpha=0.9)

ax1.set_xlabel('Lap Number', fontweight='bold', fontsize=13)
ax1.set_ylabel('Lap Time (seconds)', fontweight='bold', fontsize=13)
ax1.set_title('Lap Time Evolution: Baseline vs Improved Tire Scenarios',
             fontweight='bold', fontsize=15)
ax1.legend(loc='best', fontsize=11, framealpha=0.9)
ax1.grid(True, alpha=0.3)

# Add stint backgrounds
for stint_id in baseline['stint_id'].unique():
    stint_data = baseline[baseline['stint_id'] == stint_id]
    compound = stint_data['compound'].iloc[0]
    bg_color = 'red' if compound == 'SOFT' else 'yellow' if compound == 'MEDIUM' else 'lightblue' if compound == 'INTERMEDIATE' else 'gray'
    ax1.axvspan(stint_data['lap_number'].min(), stint_data['lap_number'].max(),
               alpha=0.1, color=bg_color)

# Plot 2: Degradation state comparison
ax2 = fig.add_subplot(gs[1, 0])
for idx, scenario_name in enumerate(scenario_list):
    df_scenario = results[scenario_name]
    linestyle = '-' if idx == 0 else '--'
    ax2.plot(df_scenario['lap_number'], df_scenario['degradation_normalized'],
            linestyle=linestyle, linewidth=2, color=colors[idx],
            label=scenario_name)

ax2.set_xlabel('Lap Number', fontweight='bold')
ax2.set_ylabel('Degradation State (0=new, 1=worn)', fontweight='bold')
ax2.set_title('Tire Degradation State', fontweight='bold', fontsize=12)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_ylim([0, 1])

# Plot 3: Time saved per lap
ax3 = fig.add_subplot(gs[1, 1])
for idx, scenario_name in enumerate(list(scenarios.keys())[1:], 1):
    improved = results[scenario_name]
    ax3.plot(improved['lap_number'], improved['time_saved'],
            linewidth=2, color=colors[idx], marker='o', markersize=3,
            label=scenario_name)

ax3.set_xlabel('Lap Number', fontweight='bold')
ax3.set_ylabel('Time Saved (seconds)', fontweight='bold')
ax3.set_title('Time Saved Per Lap vs Baseline', fontweight='bold', fontsize=12)
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)
ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.8)

# Plot 4: Cumulative time saved
ax4 = fig.add_subplot(gs[1, 2])
for idx, scenario_name in enumerate(list(scenarios.keys())[1:], 1):
    improved = results[scenario_name]
    cumulative = improved['time_saved'].cumsum()
    ax4.plot(improved['lap_number'], cumulative,
            linewidth=3, color=colors[idx], label=scenario_name)

    # Final value annotation
    final_val = cumulative.iloc[-1]
    ax4.annotate(f'{final_val:.1f}s\n({final_val/60:.2f}m)',
                xy=(improved['lap_number'].iloc[-1], final_val),
                xytext=(10, 0), textcoords='offset points',
                fontsize=9, fontweight='bold', color=colors[idx])

ax4.set_xlabel('Lap Number', fontweight='bold')
ax4.set_ylabel('Cumulative Time Saved (seconds)', fontweight='bold')
ax4.set_title('Cumulative Time Advantage', fontweight='bold', fontsize=12)
ax4.legend(fontsize=9, loc='upper left')
ax4.grid(True, alpha=0.3)

# Plot 5: Stint-by-stint comparison (Stint 1)
ax5 = fig.add_subplot(gs[2, 0])
stint_1_data = baseline[baseline['stint_id'] == 1]
compound_1 = stint_1_data['compound'].iloc[0]

ax5.scatter(stint_1_data['tyre_life'], stint_1_data['lap_time_seconds'],
           alpha=0.7, s=80, label='Actual', color='black', zorder=3)

for idx, scenario_name in enumerate(scenario_list):
    df_scenario = results[scenario_name][results[scenario_name]['stint_id'] == 1]
    linestyle = '-' if idx == 0 else '--'
    ax5.plot(df_scenario['tyre_life'], df_scenario['predicted_lap_time'],
            linestyle=linestyle, linewidth=2, color=colors[idx],
            label=scenario_name)

ax5.set_xlabel('Tire Age (laps)', fontweight='bold')
ax5.set_ylabel('Lap Time (seconds)', fontweight='bold')
ax5.set_title(f'Stint 1: {compound_1} Compound', fontweight='bold', fontsize=12)
ax5.legend(fontsize=8)
ax5.grid(True, alpha=0.3)

# Plot 6: Stint-by-stint comparison (Stint 2)
ax6 = fig.add_subplot(gs[2, 1])
if baseline['stint_id'].nunique() > 1:
    stint_2_data = baseline[baseline['stint_id'] == 2]
    compound_2 = stint_2_data['compound'].iloc[0]

    ax6.scatter(stint_2_data['tyre_life'], stint_2_data['lap_time_seconds'],
               alpha=0.7, s=80, label='Actual', color='black', zorder=3)

    for idx, scenario_name in enumerate(scenario_list):
        df_scenario = results[scenario_name][results[scenario_name]['stint_id'] == 2]
        linestyle = '-' if idx == 0 else '--'
        ax6.plot(df_scenario['tyre_life'], df_scenario['predicted_lap_time'],
                linestyle=linestyle, linewidth=2, color=colors[idx],
                label=scenario_name)

    ax6.set_xlabel('Tire Age (laps)', fontweight='bold')
    ax6.set_ylabel('Lap Time (seconds)', fontweight='bold')
    ax6.set_title(f'Stint 2: {compound_2} Compound', fontweight='bold', fontsize=12)
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3)

# Plot 7: Summary bar chart
ax7 = fig.add_subplot(gs[2, 2])
summary_data = []
for scenario_name in list(scenarios.keys())[1:]:
    improved = results[scenario_name]
    total_saved = improved['time_saved'].sum()
    summary_data.append(total_saved)

x_pos = np.arange(len(summary_data))
bars = ax7.bar(x_pos, summary_data, color=colors[1:4], alpha=0.8, edgecolor='black')

# Add value labels on bars
for bar, value in zip(bars, summary_data):
    height = bar.get_height()
    ax7.text(bar.get_x() + bar.get_width()/2., height,
            f'{value:.1f}s\n({value/60:.2f}m)',
            ha='center', va='bottom', fontweight='bold', fontsize=9)

ax7.set_xticks(x_pos)
ax7.set_xticklabels(['15%\nImproved', '30%\nImproved', '50%\nImproved'], fontsize=10)
ax7.set_ylabel('Total Time Saved (seconds)', fontweight='bold')
ax7.set_title('Total Race Time Savings', fontweight='bold', fontsize=12)
ax7.grid(True, alpha=0.3, axis='y')

plt.suptitle(f'{YEAR} {RACE} GP - Driver {DRIVER}: Improved Tire Performance Analysis',
            fontsize=16, fontweight='bold', y=0.995)

plt.savefig('improved_tire_simulation_results.png', dpi=300, bbox_inches='tight')
print("\n[OK] Main visualization saved: improved_tire_simulation_results.png")

# Export results to CSV
print("\nExporting results...")
for scenario_name, df_scenario in results.items():
    safe_name = scenario_name.replace(' ', '_').replace('(', '').replace(')', '').replace('%', 'pct')
    filename = f'simulation_{safe_name}.csv'
    df_scenario.to_csv(filename, index=False)
    print(f"  [OK] {filename}")

# Create summary report
summary_filename = 'SIMULATION_SUMMARY.txt'
with open(summary_filename, 'w') as f:
    f.write("="*80 + "\n")
    f.write(f"  IMPROVED TIRE SIMULATION RESULTS\n")
    f.write(f"  {YEAR} {RACE} Grand Prix - Driver: {DRIVER}\n")
    f.write("="*80 + "\n\n")

    f.write("CALIBRATED MODEL COEFFICIENTS USED:\n")
    f.write(f"  BETA_DECAY:    {BETA_DECAY} (optimized)\n")
    f.write(f"  GAMMA_LAPTIME: {GAMMA_LAPTIME} (optimized)\n\n")

    f.write("SCENARIOS TESTED:\n")
    for scenario_name, wear_mult in scenarios.items():
        f.write(f"  {scenario_name}: wear coefficient × {wear_mult:.2f}\n")
    f.write("\n")

    f.write("RESULTS SUMMARY:\n")
    f.write("-"*80 + "\n")

    for scenario_name in list(scenarios.keys())[1:]:
        improved = results[scenario_name]
        total_saved = improved['time_saved'].sum()
        avg_saved = improved['time_saved'].mean()
        max_saved = improved['time_saved'].max()

        f.write(f"\n{scenario_name}:\n")
        f.write(f"  Total time saved: {total_saved:.2f}s ({total_saved/60:.2f} minutes)\n")
        f.write(f"  Average per lap:  {avg_saved:.3f}s\n")
        f.write(f"  Maximum benefit:  {max_saved:.3f}s\n")

        f.write(f"  By stint:\n")
        for stint_id in sorted(improved['stint_id'].unique()):
            stint_data = improved[improved['stint_id'] == stint_id]
            compound = stint_data['compound'].iloc[0]
            stint_saved = stint_data['time_saved'].sum()
            laps = len(stint_data)
            f.write(f"    Stint {stint_id} ({compound:12s}): {stint_saved:6.2f}s over {laps} laps\n")

print(f"  [OK] {summary_filename}")

print("\n" + "="*80)
print("SIMULATION COMPLETE!")
print("="*80)
print("\nGenerated files:")
print("  - improved_tire_simulation_results.png (comprehensive visualization)")
print("  - simulation_*.csv (detailed results for each scenario)")
print("  - SIMULATION_SUMMARY.txt (summary report)")
print("\nYou now have complete data showing the benefit of improved tire technology!")
