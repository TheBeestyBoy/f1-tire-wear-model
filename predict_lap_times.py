"""
F1 Lap Time Prediction - Calibrated Model (RMSE = 4.18s)

This script uses the calibrated parameters to predict lap times for any race.
Simply change the YEAR, RACE, DRIVER variables to analyze different sessions.
"""

import fastf1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Enable FastF1 cache
cache_dir = Path('./fastf1_cache')
cache_dir.mkdir(exist_ok=True)
fastf1.Cache.enable_cache(str(cache_dir))

# ============================================================================
#                          CONFIGURATION
# ============================================================================

YEAR = 2023
RACE = 'Monaco'
SESSION = 'R'  # 'R' for Race, 'Q' for Qualifying, 'FP1', 'FP2', 'FP3' for Practice
DRIVER = 'VER'  # Three-letter driver code

# CALIBRATED OPTIMAL PARAMETERS (RMSE = 4.18s)
# From quick_calibration.py - validated and reproducible
OPTIMAL_PARAMS = {
    'drying_rate': 0.070,
    'deg_medium': 0.000700,
    'deg_inter': 0.000030,
    'inter_optimal_wet': 0.210,
    'inter_dry_penalty': 0.048,
    'slick_wet_penalty': 0.060
}

print("="*80)
print("F1 LAP TIME PREDICTION - CALIBRATED MODEL")
print("="*80)
print(f"\nSession: {YEAR} {RACE} - {SESSION}")
print(f"Driver: {DRIVER}")
print(f"\nModel RMSE: 4.18 seconds (calibrated on Monaco 2023)")

# ============================================================================
#                          LOAD DATA
# ============================================================================

print(f"\n[1/4] Loading session data...")
session = fastf1.get_session(YEAR, RACE, SESSION)
session.load()

driver_laps = session.laps.pick_driver(DRIVER)
valid_laps = driver_laps[
    (driver_laps['IsAccurate'] == True) &
    (driver_laps['LapTime'].notna())
].copy()

print(f"[OK] Loaded {len(valid_laps)} valid laps")

# Extract telemetry features
def extract_features(lap):
    try:
        telemetry = lap.get_telemetry()
        if telemetry.empty:
            return None
        return {
            'speed_mean': telemetry['Speed'].mean(),
            'telemetry_points': len(telemetry),
            'lap_duration': (telemetry['Time'].max() - telemetry['Time'].min()).total_seconds()
        }
    except:
        return None

print(f"[2/4] Extracting features...")
weather_data = session.weather_data
features = []

for idx, lap in valid_laps.iterrows():
    feat = extract_features(lap)
    if feat:
        feat['lap_number'] = lap['LapNumber']
        feat['lap_time_seconds'] = lap['LapTime'].total_seconds()
        feat['compound'] = lap['Compound']
        feat['tyre_life'] = lap['TyreLife']

        # Add weather
        lap_start_time = lap['Time']
        lap_end_time = lap_start_time + lap['LapTime']
        lap_weather = weather_data[
            (weather_data['Time'] >= lap_start_time) &
            (weather_data['Time'] <= lap_end_time)
        ]

        if not lap_weather.empty:
            feat['track_temp'] = lap_weather['TrackTemp'].mean()
            feat['humidity'] = lap_weather['Humidity'].mean()
            feat['rainfall'] = lap_weather['Rainfall'].mean()
        else:
            feat['track_temp'] = 35
            feat['humidity'] = 50
            feat['rainfall'] = 0

        features.append(feat)
        print(f"  Processing lap {int(lap['LapNumber'])}", end='\r')

df = pd.DataFrame(features)
print(f"\n[OK] Extracted {len(df)} laps")

# Fill missing weather
for col in ['track_temp', 'humidity', 'rainfall']:
    df[col] = df[col].interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')

# Add stint detection
df = df.sort_values('lap_number')
df['stint_id'] = ((df['compound'] != df['compound'].shift()) |
                  (df['tyre_life'] < df['tyre_life'].shift())).cumsum()

# ============================================================================
#                          PREDICTION MODEL
# ============================================================================

print(f"\n[3/4] Running predictions...")

# Estimate track wetness
humidity_norm = df['humidity'] / 100.0
rainfall_norm = df['rainfall'] / (df['rainfall'].max() + 0.01)
track_temp_norm = 1.0 - (df['track_temp'] - df['track_temp'].min()) / (df['track_temp'].max() - df['track_temp'].min() + 0.01)

wetness = (0.5 * rainfall_norm + 0.3 * humidity_norm + 0.2 * track_temp_norm)

# Apply drying per stint (single drying rate)
df['wetness'] = 0.0
for stint in df['stint_id'].unique():
    mask = df['stint_id'] == stint
    stint_indices = df[mask].index

    initial_wet = wetness.loc[stint_indices[0]]
    laps = np.arange(len(stint_indices))
    df.loc[stint_indices, 'wetness'] = np.maximum(
        initial_wet * np.exp(-OPTIMAL_PARAMS['drying_rate'] * laps), 0
    )

# Baseline: best lap in first 10 of stint
def get_baseline(group):
    return group.head(10).min()
df['baseline'] = df.groupby('stint_id')['lap_time_seconds'].transform(get_baseline)

# Tire performance factor (from quick_calibration.py)
def calc_factor(row):
    if row['compound'] == 'INTERMEDIATE':
        # INTERS: penalty when too dry OR too wet
        wet_dev = abs(row['wetness'] - OPTIMAL_PARAMS['inter_optimal_wet'])

        if row['wetness'] < OPTIMAL_PARAMS['inter_optimal_wet']:
            # Too dry - penalty
            perf = 1.0 + OPTIMAL_PARAMS['inter_dry_penalty'] * (wet_dev / (OPTIMAL_PARAMS['inter_optimal_wet'] + 0.01))
        else:
            # Too wet - fixed penalty
            perf = 1.0 + 0.05 * wet_dev

        # Simple degradation
        deg = 1.0 + OPTIMAL_PARAMS['deg_inter'] * row['tyre_life']
        return perf * deg
    else:
        # MEDIUM: standard degradation + wet penalty
        wet_pen = 1.0 + OPTIMAL_PARAMS['slick_wet_penalty'] * row['wetness']
        deg = 1.0 + OPTIMAL_PARAMS['deg_medium'] * row['tyre_life']
        return wet_pen * deg

df['tire_factor'] = df.apply(calc_factor, axis=1)
df['predicted_lap_time'] = df['baseline'] * df['tire_factor']
df['residual'] = df['lap_time_seconds'] - df['predicted_lap_time']

print(f"[OK] Predictions complete")

# ============================================================================
#                          RESULTS
# ============================================================================

print(f"\n[4/4] Analyzing results...")

rmse = np.sqrt((df['residual']**2).mean())
mae = df['residual'].abs().mean()
max_error = df['residual'].abs().max()

print("\n" + "="*80)
print("PREDICTION RESULTS")
print("="*80)

print(f"\nOverall Performance:")
print(f"  RMSE:      {rmse:.4f} seconds")
print(f"  MAE:       {mae:.4f} seconds")
print(f"  Max Error: {max_error:.4f} seconds")

print(f"\nBy Stint:")
for stint in sorted(df['stint_id'].unique()):
    stint_df = df[df['stint_id'] == stint]
    compound = stint_df['compound'].iloc[0]
    lap_range = f"{int(stint_df['lap_number'].min())}-{int(stint_df['lap_number'].max())}"
    rmse_s = np.sqrt((stint_df['residual']**2).mean())
    mae_s = stint_df['residual'].abs().mean()
    avg_wet = stint_df['wetness'].mean()
    print(f"  Stint {stint} ({compound:12s}) Laps {lap_range:8s}: "
          f"RMSE={rmse_s:5.3f}s, MAE={mae_s:5.3f}s, Wetness={avg_wet:.3f}")

# Sample predictions
print(f"\n" + "="*80)
print("SAMPLE PREDICTIONS")
print("="*80)
print(f"\nLap | Compound | Actual | Predicted | Error  | Wetness")
print("-" * 65)
for idx in range(0, len(df), max(1, len(df)//10)):
    row = df.iloc[idx]
    print(f"{row['lap_number']:3.0f} | {row['compound']:8s} | "
          f"{row['lap_time_seconds']:6.2f} | {row['predicted_lap_time']:9.2f} | "
          f"{row['residual']:6.2f} | {row['wetness']:7.3f}")

# Save results
output_file = f'predictions_{YEAR}_{RACE}_{DRIVER}.csv'
df.to_csv(output_file, index=False)
print(f"\n[OK] Results saved to: {output_file}")

# ============================================================================
#                          VISUALIZATION
# ============================================================================

print(f"\n[5/5] Creating visualizations...")

# Set up professional blue theme for presentations
plt.style.use('seaborn-v0_8-darkgrid')
BLUE_DARK = '#0A2647'      # Deep navy blue for backgrounds
BLUE_PRIMARY = '#144272'   # Primary blue
BLUE_LIGHT = '#2C74B3'     # Light blue for accents
BLUE_ACCENT = '#4A90E2'    # Bright accent blue
ORANGE_ACCENT = '#FF6B35'  # Warm orange for contrast
WHITE = '#FFFFFF'
GRAY_LIGHT = '#E8E8E8'

# Create output directory for individual graphs
output_dir = Path('presentation_graphs')
output_dir.mkdir(exist_ok=True)

# ============================================================================
# GRAPH 1: Actual vs Predicted Lap Times (with Tire Upgrade Scenarios)
# ============================================================================

# Define tire upgrade scenarios (percentage improvement in tire performance)
upgrade_scenarios = [
    {'percent': 15, 'color': '#00FF88', 'label': '15% Tire Upgrade'},
    {'percent': 25, 'color': '#FFD700', 'label': '25% Tire Upgrade'},
    {'percent': 50, 'color': '#FF1E9D', 'label': '50% Tire Upgrade'}
]

for scenario in upgrade_scenarios:
    upgrade_percent = scenario['percent']
    upgrade_color = scenario['color']
    upgrade_label = scenario['label']

    # Calculate upgraded tire performance
    # Reduce tire degradation factor by the upgrade percentage
    df['tire_factor_upgraded'] = 1 + (df['tire_factor'] - 1) * (1 - upgrade_percent / 100)
    df['predicted_upgraded'] = df['baseline'] * df['tire_factor_upgraded']

    # Calculate time savings
    total_time_saved = (df['predicted_lap_time'] - df['predicted_upgraded']).sum()
    avg_lap_improvement = (df['predicted_lap_time'] - df['predicted_upgraded']).mean()

    # Create figure
    fig1, ax1 = plt.subplots(figsize=(14, 8), facecolor=BLUE_DARK)
    ax1.set_facecolor(BLUE_PRIMARY)

    # Plot actual lap times
    ax1.scatter(df['lap_number'], df['lap_time_seconds'],
               alpha=0.7, s=100, label='Actual Lap Times',
               c=BLUE_ACCENT, edgecolors=WHITE, linewidths=1.5, zorder=4)

    # Plot baseline prediction (original)
    ax1.plot(df['lap_number'], df['predicted_lap_time'],
            color=ORANGE_ACCENT, linewidth=3, label='Baseline Prediction',
            alpha=0.9, zorder=3, linestyle='--')

    # Plot upgraded prediction
    ax1.plot(df['lap_number'], df['predicted_upgraded'],
            color=upgrade_color, linewidth=4, label=f'{upgrade_label} Prediction',
            alpha=0.95, zorder=2)

    # Fill area between baseline and upgraded to show improvement
    ax1.fill_between(df['lap_number'],
                     df['predicted_lap_time'],
                     df['predicted_upgraded'],
                     alpha=0.3, color=upgrade_color, zorder=1,
                     label=f'Time Saved per Lap')

    # Styling
    ax1.set_xlabel('Lap Number', fontweight='bold', fontsize=16, color=WHITE)
    ax1.set_ylabel('Lap Time (seconds)', fontweight='bold', fontsize=16, color=WHITE)
    ax1.set_title(f'{YEAR} {RACE} Grand Prix - {DRIVER}\n{upgrade_label} Performance Analysis',
                 fontweight='bold', fontsize=20, color=WHITE, pad=20)
    ax1.legend(fontsize=13, loc='upper right', framealpha=0.95, facecolor=BLUE_PRIMARY,
              edgecolor=WHITE, labelcolor=WHITE)
    ax1.grid(True, alpha=0.25, color=WHITE, linestyle='--', linewidth=0.8)
    ax1.tick_params(colors=WHITE, labelsize=12)

    # Add statistics box with upgrade benefits
    stats_text = (f'Total Laps: {len(df)}\n'
                  f'Total Time Saved: {total_time_saved:.2f}s\n'
                  f'Avg Lap Improvement: {avg_lap_improvement:.3f}s\n'
                  f'Tire Degradation: -{upgrade_percent}%')
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
            fontsize=12, verticalalignment='top', color=WHITE, family='monospace',
            bbox=dict(boxstyle='round', facecolor=BLUE_PRIMARY, alpha=0.9, edgecolor=WHITE))

    # Add tire degradation annotations every 5 laps along the top
    y_min, y_max = ax1.get_ylim()
    y_range = y_max - y_min

    # Position annotations in the middle-upper area to avoid overlapping with corner text boxes
    y_annotation = y_max - (y_range * 0.15)  # Position at 85% height (leaves room for title and margins)

    # Get lap numbers at intervals of 5
    annotation_laps = range(5, int(df['lap_number'].max()) + 1, 5)

    # Calculate x-axis range to determine which laps are in the left/right corners
    x_min, x_max = df['lap_number'].min(), df['lap_number'].max()
    x_range = x_max - x_min
    left_margin = x_min + (x_range * 0.25)  # Skip first 25% (left stats box)
    right_margin = x_max - (x_range * 0.25)  # Skip last 25% (right legend box)

    for lap_num in annotation_laps:
        # Skip laps in the corner areas where text boxes are
        if lap_num < left_margin or lap_num > right_margin:
            continue

        # Find the closest lap in the dataframe
        lap_data = df[df['lap_number'] == lap_num]
        if not lap_data.empty:
            # Calculate degradation percentage at this lap
            # Degradation = (tire_factor - 1) * 100, adjusted for upgrade
            baseline_deg = (lap_data['tire_factor'].iloc[0] - 1) * 100
            upgraded_deg = (lap_data['tire_factor_upgraded'].iloc[0] - 1) * 100

            # Annotate with arrow pointing down to the upgraded prediction line
            annotation_text = f'{upgraded_deg:.1f}%'
            ax1.annotate(annotation_text,
                        xy=(lap_num, lap_data['predicted_upgraded'].iloc[0]),
                        xytext=(lap_num, y_annotation),
                        ha='center', va='bottom',
                        fontsize=10, color=upgrade_color, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor=BLUE_DARK,
                                 edgecolor=upgrade_color, linewidth=1.5, alpha=0.85),
                        arrowprops=dict(arrowstyle='->', color=upgrade_color,
                                       linewidth=1.5, alpha=0.7))

    plt.tight_layout()
    graph1_file = output_dir / f'1a_lap_times_{upgrade_percent}pct_upgrade_{YEAR}_{RACE}_{DRIVER}.png'
    plt.savefig(graph1_file, dpi=300, bbox_inches='tight', facecolor=BLUE_DARK)
    print(f"[OK] Graph 1 ({upgrade_percent}% upgrade) saved: {graph1_file}")
    plt.close()

# Also create the baseline comparison without upgrades
fig1, ax1 = plt.subplots(figsize=(14, 8), facecolor=BLUE_DARK)
ax1.set_facecolor(BLUE_PRIMARY)

# Plot actual lap times with gradient effect
ax1.scatter(df['lap_number'], df['lap_time_seconds'],
           alpha=0.8, s=120, label='Actual Lap Times',
           c=BLUE_ACCENT, edgecolors=WHITE, linewidths=1.5, zorder=3)

# Plot predicted lap times
ax1.plot(df['lap_number'], df['predicted_lap_time'],
        color=ORANGE_ACCENT, linewidth=3.5, label='Predicted Lap Times',
        alpha=0.95, zorder=2)

# Fill area between for visual effect
ax1.fill_between(df['lap_number'],
                 df['predicted_lap_time'],
                 df['lap_time_seconds'],
                 alpha=0.2, color=GRAY_LIGHT, zorder=1)

# Styling
ax1.set_xlabel('Lap Number', fontweight='bold', fontsize=16, color=WHITE)
ax1.set_ylabel('Lap Time (seconds)', fontweight='bold', fontsize=16, color=WHITE)
ax1.set_title(f'{YEAR} {RACE} Grand Prix - {DRIVER}\nBaseline Lap Time Prediction Model (RMSE = {rmse:.2f}s)',
             fontweight='bold', fontsize=20, color=WHITE, pad=20)
ax1.legend(fontsize=14, loc='upper right', framealpha=0.95, facecolor=BLUE_PRIMARY,
          edgecolor=WHITE, labelcolor=WHITE)
ax1.grid(True, alpha=0.25, color=WHITE, linestyle='--', linewidth=0.8)
ax1.tick_params(colors=WHITE, labelsize=12)

# Add annotations
ax1.text(0.02, 0.98, f'Total Laps: {len(df)}', transform=ax1.transAxes,
        fontsize=13, verticalalignment='top', color=WHITE,
        bbox=dict(boxstyle='round', facecolor=BLUE_PRIMARY, alpha=0.9, edgecolor=WHITE))

plt.tight_layout()
graph1_file = output_dir / f'1_lap_times_baseline_{YEAR}_{RACE}_{DRIVER}.png'
plt.savefig(graph1_file, dpi=300, bbox_inches='tight', facecolor=BLUE_DARK)
print(f"[OK] Graph 1 (baseline) saved: {graph1_file}")
plt.close()

# ============================================================================
# GRAPH 2: Prediction Residuals by Compound
# ============================================================================
fig2, ax2 = plt.subplots(figsize=(14, 8), facecolor=BLUE_DARK)
ax2.set_facecolor(BLUE_PRIMARY)

# Color mapping for tire compounds
compound_colors = {
    'SOFT': '#FF1E1E',
    'MEDIUM': '#FFD700',
    'HARD': '#E8E8E8',
    'INTERMEDIATE': '#00FF00',
    'WET': '#0066FF'
}

# Plot residuals with compound-specific colors
for compound in df['compound'].unique():
    compound_data = df[df['compound'] == compound]
    ax2.scatter(compound_data['lap_number'], compound_data['residual'],
               alpha=0.75, s=100, label=compound,
               c=compound_colors.get(compound, BLUE_ACCENT),
               edgecolors=WHITE, linewidths=1.2, zorder=3)

# Reference lines
ax2.axhline(0, color=ORANGE_ACCENT, linestyle='-', linewidth=3, alpha=0.9, zorder=2, label='Perfect Prediction')
ax2.axhline(3, color='#00FF88', linestyle='--', linewidth=2, alpha=0.7, zorder=1)
ax2.axhline(-3, color='#00FF88', linestyle='--', linewidth=2, alpha=0.7, zorder=1)
ax2.fill_between(df['lap_number'], -3, 3, alpha=0.15, color='#00FF88', zorder=0)

# Styling
ax2.set_xlabel('Lap Number', fontweight='bold', fontsize=16, color=WHITE)
ax2.set_ylabel('Prediction Error (seconds)', fontweight='bold', fontsize=16, color=WHITE)
ax2.set_title('Model Prediction Accuracy by Tire Compound',
             fontweight='bold', fontsize=20, color=WHITE, pad=20)
ax2.legend(fontsize=13, loc='best', framealpha=0.95, facecolor=BLUE_PRIMARY,
          edgecolor=WHITE, labelcolor=WHITE, ncol=2)
ax2.grid(True, alpha=0.25, color=WHITE, linestyle='--', linewidth=0.8)
ax2.tick_params(colors=WHITE, labelsize=12)

# Add statistics box
stats_text = f'MAE: {mae:.2f}s\nMax Error: {max_error:.2f}s\nWithin ±3s: {(df["residual"].abs() <= 3).sum()}/{len(df)} laps'
ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes,
        fontsize=12, verticalalignment='top', color=WHITE, family='monospace',
        bbox=dict(boxstyle='round', facecolor=BLUE_PRIMARY, alpha=0.9, edgecolor=WHITE))

plt.tight_layout()
graph2_file = output_dir / f'2_residuals_{YEAR}_{RACE}_{DRIVER}.png'
plt.savefig(graph2_file, dpi=300, bbox_inches='tight', facecolor=BLUE_DARK)
print(f"[OK] Graph 2 saved: {graph2_file}")
plt.close()

# ============================================================================
# GRAPH 3: Track Wetness Evolution
# ============================================================================
fig3, ax3 = plt.subplots(figsize=(14, 8), facecolor=BLUE_DARK)
ax3.set_facecolor(BLUE_PRIMARY)

# Create gradient effect for wetness
ax3.plot(df['lap_number'], df['wetness'],
        color=BLUE_ACCENT, linewidth=4, label='Track Wetness', alpha=0.95, zorder=3)
ax3.fill_between(df['lap_number'], 0, df['wetness'],
                alpha=0.4, color=BLUE_LIGHT, zorder=2)

# Add optimal intermediate line
ax3.axhline(OPTIMAL_PARAMS['inter_optimal_wet'],
           color=ORANGE_ACCENT, linestyle='--', linewidth=2.5, alpha=0.9,
           label=f'Optimal for Intermediates ({OPTIMAL_PARAMS["inter_optimal_wet"]:.2f})', zorder=1)

# Shade optimal zone
ax3.fill_between(df['lap_number'],
                OPTIMAL_PARAMS['inter_optimal_wet'] * 0.9,
                OPTIMAL_PARAMS['inter_optimal_wet'] * 1.1,
                alpha=0.2, color=ORANGE_ACCENT, zorder=0)

# Styling
ax3.set_xlabel('Lap Number', fontweight='bold', fontsize=16, color=WHITE)
ax3.set_ylabel('Track Wetness Index (0=Dry, 1=Wet)', fontweight='bold', fontsize=16, color=WHITE)
ax3.set_title('Track Condition Evolution Throughout Race',
             fontweight='bold', fontsize=20, color=WHITE, pad=20)
ax3.legend(fontsize=13, loc='upper right', framealpha=0.95, facecolor=BLUE_PRIMARY,
          edgecolor=WHITE, labelcolor=WHITE)
ax3.grid(True, alpha=0.25, color=WHITE, linestyle='--', linewidth=0.8)
ax3.tick_params(colors=WHITE, labelsize=12)
ax3.set_ylim(-0.05, max(df['wetness'].max() * 1.1, 0.5))

# Add drying rate annotation
drying_text = f'Drying Rate: {OPTIMAL_PARAMS["drying_rate"]:.3f} per lap'
ax3.text(0.02, 0.98, drying_text, transform=ax3.transAxes,
        fontsize=13, verticalalignment='top', color=WHITE,
        bbox=dict(boxstyle='round', facecolor=BLUE_PRIMARY, alpha=0.9, edgecolor=WHITE))

plt.tight_layout()
graph3_file = output_dir / f'3_wetness_{YEAR}_{RACE}_{DRIVER}.png'
plt.savefig(graph3_file, dpi=300, bbox_inches='tight', facecolor=BLUE_DARK)
print(f"[OK] Graph 3 saved: {graph3_file}")
plt.close()

# ============================================================================
# GRAPH 4: Error Distribution Histogram
# ============================================================================
fig4, ax4 = plt.subplots(figsize=(14, 8), facecolor=BLUE_DARK)
ax4.set_facecolor(BLUE_PRIMARY)

# Create histogram with blue gradient
n, bins, patches = ax4.hist(df['residual'], bins=30, alpha=0.85,
                            edgecolor=WHITE, linewidth=1.5, color=BLUE_LIGHT)

# Color bars based on distance from zero
for i, patch in enumerate(patches):
    bin_center = (bins[i] + bins[i+1]) / 2
    intensity = 1 - min(abs(bin_center) / max(abs(df['residual'])), 1)
    patch.set_facecolor(plt.cm.Blues(0.4 + intensity * 0.5))

# Reference lines
ax4.axvline(0, color=ORANGE_ACCENT, linestyle='-', linewidth=3.5,
           alpha=0.95, label='Perfect Prediction', zorder=3)
ax4.axvline(rmse, color='#00FF88', linestyle='--', linewidth=2.5,
           label=f'±RMSE ({rmse:.2f}s)', alpha=0.9, zorder=2)
ax4.axvline(-rmse, color='#00FF88', linestyle='--', linewidth=2.5, alpha=0.9, zorder=2)

# Styling
ax4.set_xlabel('Prediction Error (seconds)', fontweight='bold', fontsize=16, color=WHITE)
ax4.set_ylabel('Frequency (Number of Laps)', fontweight='bold', fontsize=16, color=WHITE)
ax4.set_title('Model Error Distribution',
             fontweight='bold', fontsize=20, color=WHITE, pad=20)
ax4.legend(fontsize=13, loc='upper right', framealpha=0.95, facecolor=BLUE_PRIMARY,
          edgecolor=WHITE, labelcolor=WHITE)
ax4.grid(True, alpha=0.25, color=WHITE, linestyle='--', linewidth=0.8, axis='y')
ax4.tick_params(colors=WHITE, labelsize=12)

# Add statistics box
stats_text = f'Mean Error: {df["residual"].mean():.2f}s\nStd Dev: {df["residual"].std():.2f}s\nRMSE: {rmse:.2f}s\nMAE: {mae:.2f}s'
ax4.text(0.02, 0.98, stats_text, transform=ax4.transAxes,
        fontsize=12, verticalalignment='top', color=WHITE, family='monospace',
        bbox=dict(boxstyle='round', facecolor=BLUE_PRIMARY, alpha=0.9, edgecolor=WHITE))

plt.tight_layout()
graph4_file = output_dir / f'4_error_distribution_{YEAR}_{RACE}_{DRIVER}.png'
plt.savefig(graph4_file, dpi=300, bbox_inches='tight', facecolor=BLUE_DARK)
print(f"[OK] Graph 4 saved: {graph4_file}")
plt.close()

# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*80)
print("PREDICTION COMPLETE")
print("="*80)
if rmse < 3:
    print(f"\nModel Status: EXCELLENT (RMSE = {rmse:.3f}s)")
elif rmse < 5:
    print(f"\nModel Status: GOOD (RMSE = {rmse:.3f}s)")
else:
    print(f"\nModel Status: OK (RMSE = {rmse:.3f}s)")
print(f"\nNote: Model was calibrated on Monaco 2023. Performance may vary on other tracks.")
print(f"\nPresentation graphs saved to: {output_dir}/")
print("  - 1_lap_times_baseline_*.png (Baseline Prediction)")
print("  - 1a_lap_times_15pct_upgrade_*.png (15% Tire Upgrade Scenario)")
print("  - 1a_lap_times_25pct_upgrade_*.png (25% Tire Upgrade Scenario)")
print("  - 1a_lap_times_50pct_upgrade_*.png (50% Tire Upgrade Scenario)")
print("  - 2_residuals_*.png (Prediction Accuracy)")
print("  - 3_wetness_*.png (Track Conditions)")
print("  - 4_error_distribution_*.png (Statistical Analysis)")
