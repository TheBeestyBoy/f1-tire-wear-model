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

print(f"\n[5/5] Creating visualization...")

fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Plot 1: Actual vs Predicted
ax1 = axes[0, 0]
ax1.scatter(df['lap_number'], df['lap_time_seconds'], alpha=0.6, s=70, label='Actual', c='blue')
ax1.plot(df['lap_number'], df['predicted_lap_time'], 'r-', linewidth=2.5, label='Predicted')
ax1.set_xlabel('Lap Number', fontweight='bold', fontsize=12)
ax1.set_ylabel('Lap Time (seconds)', fontweight='bold', fontsize=12)
ax1.set_title(f'{YEAR} {RACE} - {DRIVER} | RMSE = {rmse:.3f}s', fontweight='bold', fontsize=14)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# Plot 2: Residuals
ax2 = axes[0, 1]
colors = df['compound'].map({
    'SOFT': 'red', 'MEDIUM': 'orange', 'HARD': 'white',
    'INTERMEDIATE': 'cyan', 'WET': 'blue'
})
ax2.scatter(df['lap_number'], df['residual'], alpha=0.6, s=55, c=colors, edgecolors='black', linewidths=0.5)
ax2.axhline(0, color='red', linestyle='--', linewidth=2)
ax2.axhline(3, color='green', linestyle=':', linewidth=1, label='±3s')
ax2.axhline(-3, color='green', linestyle=':', linewidth=1)
ax2.set_xlabel('Lap Number', fontweight='bold', fontsize=12)
ax2.set_ylabel('Prediction Error (seconds)', fontweight='bold', fontsize=12)
ax2.set_title('Prediction Residuals', fontweight='bold', fontsize=14)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

# Plot 3: Track Wetness
ax3 = axes[1, 0]
ax3.plot(df['lap_number'], df['wetness'], 'b-', linewidth=2.5)
ax3.fill_between(df['lap_number'], 0, df['wetness'], alpha=0.3, color='blue')
ax3.axhline(OPTIMAL_PARAMS['inter_optimal_wet'], color='red', linestyle='--',
           linewidth=1.5, label=f'INTER optimal ({OPTIMAL_PARAMS["inter_optimal_wet"]:.2f})')
ax3.set_xlabel('Lap Number', fontweight='bold', fontsize=12)
ax3.set_ylabel('Track Wetness (0=dry, 1=wet)', fontweight='bold', fontsize=12)
ax3.set_title('Track Condition Evolution', fontweight='bold', fontsize=14)
ax3.legend(fontsize=11)
ax3.grid(True, alpha=0.3)

# Plot 4: Error Distribution
ax4 = axes[1, 1]
ax4.hist(df['residual'], bins=25, alpha=0.7, edgecolor='black', color='skyblue')
ax4.axvline(0, color='red', linestyle='--', linewidth=2.5, label='Zero error')
ax4.axvline(rmse, color='orange', linestyle=':', linewidth=2, label=f'RMSE ({rmse:.2f}s)')
ax4.axvline(-rmse, color='orange', linestyle=':', linewidth=2)
ax4.set_xlabel('Prediction Error (seconds)', fontweight='bold', fontsize=12)
ax4.set_ylabel('Frequency', fontweight='bold', fontsize=12)
ax4.set_title('Error Distribution', fontweight='bold', fontsize=14)
ax4.legend(fontsize=11)
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plot_file = f'predictions_{YEAR}_{RACE}_{DRIVER}.png'
plt.savefig(plot_file, dpi=300, bbox_inches='tight')
print(f"[OK] Visualization saved to: {plot_file}")

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

plt.show()
