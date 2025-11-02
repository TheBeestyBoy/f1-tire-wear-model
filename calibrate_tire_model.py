"""
F1 Tire Wear Mathematical Model - Calibration Script

This script runs the tire wear model iteratively and tunes coefficients
to match actual lap time data as closely as possible.
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

# Configuration
YEAR = 2023
RACE = 'Monaco'
SESSION = 'R'
DRIVER = 'VER'

print("="*80)
print("F1 TIRE WEAR MODEL - CALIBRATION PROCESS")
print("="*80)
print(f"\nLoading: {YEAR} {RACE} - {SESSION} - Driver {DRIVER}")

# Load session data
session = fastf1.get_session(YEAR, RACE, SESSION)
session.load()
print(f"[OK] Session loaded")

# Get driver laps
driver_laps = session.laps.pick_driver(DRIVER)
valid_laps = driver_laps[
    (driver_laps['IsAccurate'] == True) &
    (driver_laps['LapTime'].notna())
].copy()

print(f"[OK] Valid laps: {len(valid_laps)}")
print(f"[OK] Compounds: {valid_laps['Compound'].value_counts().to_dict()}")

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

print("\n" + "="*80)
print("EXTRACTING TELEMETRY DATA...")
print("="*80)

telemetry_features = []
for idx, lap in valid_laps.iterrows():
    features = extract_lap_telemetry_features(lap)
    if features:
        features['lap_number'] = lap['LapNumber']
        features['lap_time_seconds'] = lap['LapTime'].total_seconds()
        features['compound'] = lap['Compound']
        features['tyre_life'] = lap['TyreLife']
        telemetry_features.append(features)
        print(f"  Lap {int(lap['LapNumber']):2d}: {features['lap_time_seconds']:.2f}s - {lap['Compound']:6s} (life: {int(lap['TyreLife']):2d})", end='\r')

df = pd.DataFrame(telemetry_features)
print(f"\n[OK] Extracted features for {len(df)} laps")

# Show actual data statistics
print("\n" + "="*80)
print("ACTUAL DATA ANALYSIS")
print("="*80)
print("\nLap time statistics:")
print(f"  Mean: {df['lap_time_seconds'].mean():.3f}s")
print(f"  Std:  {df['lap_time_seconds'].std():.3f}s")
print(f"  Min:  {df['lap_time_seconds'].min():.3f}s")
print(f"  Max:  {df['lap_time_seconds'].max():.3f}s")

# Show telemetry ranges
print("\nTelemetry ranges:")
print(f"  Speed:    {df['speed_mean'].min():.1f} - {df['speed_mean'].max():.1f} km/h")
print(f"  Throttle: {df['throttle_mean'].min():.1f} - {df['throttle_mean'].max():.1f} %")
print(f"  Braking:  {df['brake_applications'].min():.0f} - {df['brake_applications'].max():.0f} applications")
print(f"  Gear shifts: {df['gear_shifts'].min():.0f} - {df['gear_shifts'].max():.0f} per lap")


# ============================================================================
#                          MATHEMATICAL MODEL
# ============================================================================

class TireWearModel:
    """Mathematical tire wear model with tunable coefficients."""

    def __init__(self):
        # Physical constants
        self.VEHICLE_MASS = 800  # kg

        # Compound coefficients
        self.COMPOUND_COEFFICIENTS = {
            'SOFT': 1.5,
            'MEDIUM': 1.0,
            'HARD': 0.6,
            'INTERMEDIATE': 0.8,
            'WET': 0.7
        }

        # Energy weighting factors (THESE WILL BE TUNED)
        self.K_BRAKING = 0.001
        self.K_CORNERING = 0.02
        self.K_ACCELERATION = 0.0005
        self.K_THERMAL = 0.01

        # Temperature effect
        self.ALPHA_TEMP = 0.5

        # Degradation parameters (THESE WILL BE TUNED)
        self.BETA_DECAY = 0.03
        self.GAMMA_LAPTIME = 0.10

    def calculate_braking_energy(self, row):
        v_mean_ms = row['speed_mean'] / 3.6
        brake_intensity = row['brake_mean'] / 100.0
        energy = self.K_BRAKING * self.VEHICLE_MASS * v_mean_ms * row['brake_applications'] * (brake_intensity + 0.1)
        return energy

    def calculate_cornering_energy(self, row):
        energy = self.K_CORNERING * (row['speed_std'] ** 2) * row['telemetry_points']
        return energy

    def calculate_acceleration_energy(self, row):
        v_mean_ms = row['speed_mean'] / 3.6
        throttle_norm = row['throttle_mean'] / 100.0
        energy = self.K_ACCELERATION * throttle_norm * v_mean_ms * row['lap_duration']
        return energy

    def calculate_thermal_load(self, row):
        thermal = self.K_THERMAL * (row['speed_mean'] * row['lap_duration'] + row['gear_shifts'] * 10)
        return thermal

    def calculate_temperature_factor(self, row, max_speed, max_brake_apps):
        speed_norm = row['speed_mean'] / max_speed if max_speed > 0 else 0
        brake_norm = row['brake_applications'] / max_brake_apps if max_brake_apps > 0 else 0
        throttle_norm = row['throttle_mean'] / 100.0

        T_normalized = 0.4 * speed_norm + 0.3 * brake_norm + 0.3 * throttle_norm
        T_factor = np.exp(self.ALPHA_TEMP * T_normalized)
        return T_factor

    def calculate_lap_wear(self, row, max_speed, max_brake_apps):
        W_brake = self.calculate_braking_energy(row)
        W_corner = self.calculate_cornering_energy(row)
        W_accel = self.calculate_acceleration_energy(row)
        W_thermal = self.calculate_thermal_load(row)

        W_mechanical = W_brake + W_corner + W_accel + W_thermal
        k_compound = self.COMPOUND_COEFFICIENTS.get(row['compound'], 1.0)
        T_factor = self.calculate_temperature_factor(row, max_speed, max_brake_apps)

        W_lap = k_compound * W_mechanical * T_factor
        return W_lap, W_brake, W_corner, W_accel, W_thermal, T_factor

    def fit_and_predict(self, df):
        """Run the full model pipeline."""
        df = df.copy()

        # Calculate normalization factors
        max_speed = df['speed_mean'].max()
        max_brake_apps = df['brake_applications'].max()

        # Calculate wear for each lap
        wear_results = df.apply(lambda row: self.calculate_lap_wear(row, max_speed, max_brake_apps), axis=1)
        df['wear_total'] = [w[0] for w in wear_results]
        df['wear_braking'] = [w[1] for w in wear_results]
        df['wear_cornering'] = [w[2] for w in wear_results]
        df['wear_acceleration'] = [w[3] for w in wear_results]
        df['wear_thermal'] = [w[4] for w in wear_results]
        df['temp_factor'] = [w[5] for w in wear_results]

        # Sort by lap number
        df = df.sort_values('lap_number')

        # Detect stints
        df['stint_id'] = ((df['compound'] != df['compound'].shift()) |
                          (df['tyre_life'] < df['tyre_life'].shift())).cumsum()

        # Calculate cumulative wear within each stint
        df['wear_cumulative'] = df.groupby('stint_id')['wear_total'].cumsum()
        df['degradation_normalized'] = 1 - np.exp(-self.BETA_DECAY * df['wear_cumulative'])

        # Calculate baseline time (best of first 3 laps per stint)
        def get_baseline_time(stint_series):
            early_laps = stint_series.head(3)
            return early_laps.min()

        df['baseline_time'] = df.groupby('stint_id')['lap_time_seconds'].transform(get_baseline_time)

        # Predicted lap time
        df['predicted_lap_time'] = df['baseline_time'] * (1 + self.GAMMA_LAPTIME * df['degradation_normalized'])
        df['lap_time_residual'] = df['lap_time_seconds'] - df['predicted_lap_time']

        return df

    def evaluate(self, df):
        """Calculate model performance metrics."""
        rmse = np.sqrt(np.mean(df['lap_time_residual'] ** 2))
        mae = df['lap_time_residual'].abs().mean()
        ss_res = np.sum(df['lap_time_residual'] ** 2)
        ss_tot = np.sum((df['lap_time_seconds'] - df['lap_time_seconds'].mean()) ** 2)
        r_squared = 1 - (ss_res / ss_tot)

        return {
            'rmse': rmse,
            'mae': mae,
            'r_squared': r_squared,
            'mean_residual': df['lap_time_residual'].mean(),
            'std_residual': df['lap_time_residual'].std()
        }


# ============================================================================
#                          CALIBRATION LOOP
# ============================================================================

print("\n" + "="*80)
print("CALIBRATION ITERATION 1 - INITIAL COEFFICIENTS")
print("="*80)

model = TireWearModel()
print("\nInitial coefficients:")
print(f"  K_BRAKING:      {model.K_BRAKING}")
print(f"  K_CORNERING:    {model.K_CORNERING}")
print(f"  K_ACCELERATION: {model.K_ACCELERATION}")
print(f"  K_THERMAL:      {model.K_THERMAL}")
print(f"  BETA_DECAY:     {model.BETA_DECAY}")
print(f"  GAMMA_LAPTIME:  {model.GAMMA_LAPTIME}")

# Run model
df_result = model.fit_and_predict(df)
metrics = model.evaluate(df_result)

print("\nModel Performance:")
print(f"  RMSE:          {metrics['rmse']:.4f} seconds")
print(f"  MAE:           {metrics['mae']:.4f} seconds")
print(f"  R²:            {metrics['r_squared']:.4f}")
print(f"  Mean residual: {metrics['mean_residual']:.4f} seconds")
print(f"  Std residual:  {metrics['std_residual']:.4f} seconds")

# Analyze residuals by stint
print("\nResiduals by stint:")
for stint_id in sorted(df_result['stint_id'].unique()):
    stint_df = df_result[df_result['stint_id'] == stint_id]
    compound = stint_df['compound'].iloc[0]
    rmse_stint = np.sqrt(np.mean(stint_df['lap_time_residual'] ** 2))
    print(f"  Stint {stint_id} ({compound:6s}): RMSE = {rmse_stint:.4f}s")

# Store iteration results
iterations = [{
    'iteration': 1,
    'rmse': metrics['rmse'],
    'mae': metrics['mae'],
    'r_squared': metrics['r_squared'],
    'K_BRAKING': model.K_BRAKING,
    'K_CORNERING': model.K_CORNERING,
    'K_ACCELERATION': model.K_ACCELERATION,
    'K_THERMAL': model.K_THERMAL,
    'BETA_DECAY': model.BETA_DECAY,
    'GAMMA_LAPTIME': model.GAMMA_LAPTIME
}]

best_rmse = metrics['rmse']
best_model_params = {
    'K_BRAKING': model.K_BRAKING,
    'K_CORNERING': model.K_CORNERING,
    'K_ACCELERATION': model.K_ACCELERATION,
    'K_THERMAL': model.K_THERMAL,
    'BETA_DECAY': model.BETA_DECAY,
    'GAMMA_LAPTIME': model.GAMMA_LAPTIME
}

# ============================================================================
#                       AUTOMATED CALIBRATION
# ============================================================================

print("\n" + "="*80)
print("AUTOMATED CALIBRATION - TUNING COEFFICIENTS")
print("="*80)

# Strategy: Grid search on key parameters
# Focus on BETA_DECAY and GAMMA_LAPTIME first (most direct impact on lap time prediction)

BETA_VALUES = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10]
GAMMA_VALUES = [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20]

iteration_num = 2

print(f"\nTuning BETA_DECAY and GAMMA_LAPTIME...")
print(f"Testing {len(BETA_VALUES)} × {len(GAMMA_VALUES)} = {len(BETA_VALUES)*len(GAMMA_VALUES)} combinations\n")

for beta in BETA_VALUES:
    for gamma in GAMMA_VALUES:
        model.BETA_DECAY = beta
        model.GAMMA_LAPTIME = gamma

        df_result = model.fit_and_predict(df)
        metrics = model.evaluate(df_result)

        print(f"  beta={beta:.3f}, gamma={gamma:.3f} -> RMSE={metrics['rmse']:.4f}, R²={metrics['r_squared']:.4f}", end='\r')

        iterations.append({
            'iteration': iteration_num,
            'rmse': metrics['rmse'],
            'mae': metrics['mae'],
            'r_squared': metrics['r_squared'],
            'K_BRAKING': model.K_BRAKING,
            'K_CORNERING': model.K_CORNERING,
            'K_ACCELERATION': model.K_ACCELERATION,
            'K_THERMAL': model.K_THERMAL,
            'BETA_DECAY': model.BETA_DECAY,
            'GAMMA_LAPTIME': model.GAMMA_LAPTIME
        })

        if metrics['rmse'] < best_rmse:
            best_rmse = metrics['rmse']
            best_model_params = {
                'K_BRAKING': model.K_BRAKING,
                'K_CORNERING': model.K_CORNERING,
                'K_ACCELERATION': model.K_ACCELERATION,
                'K_THERMAL': model.K_THERMAL,
                'BETA_DECAY': model.BETA_DECAY,
                'GAMMA_LAPTIME': model.GAMMA_LAPTIME
            }

        iteration_num += 1

print("\n\n[OK] Grid search complete")

# Apply best parameters
model.K_BRAKING = best_model_params['K_BRAKING']
model.K_CORNERING = best_model_params['K_CORNERING']
model.K_ACCELERATION = best_model_params['K_ACCELERATION']
model.K_THERMAL = best_model_params['K_THERMAL']
model.BETA_DECAY = best_model_params['BETA_DECAY']
model.GAMMA_LAPTIME = best_model_params['GAMMA_LAPTIME']

# ============================================================================
#                          FINAL MODEL EVALUATION
# ============================================================================

print("\n" + "="*80)
print("FINAL CALIBRATED MODEL")
print("="*80)

print("\nOptimal coefficients:")
for param, value in best_model_params.items():
    print(f"  {param:15s}: {value}")

df_final = model.fit_and_predict(df)
metrics_final = model.evaluate(df_final)

print("\nFinal Model Performance:")
print(f"  RMSE:          {metrics_final['rmse']:.4f} seconds")
print(f"  MAE:           {metrics_final['mae']:.4f} seconds")
print(f"  R²:            {metrics_final['r_squared']:.4f}")
print(f"  Mean residual: {metrics_final['mean_residual']:.4f} seconds")
print(f"  Std residual:  {metrics_final['std_residual']:.4f} seconds")

improvement = ((iterations[0]['rmse'] - metrics_final['rmse']) / iterations[0]['rmse']) * 100
print(f"\nImprovement: {improvement:.1f}% reduction in RMSE")

# Show prediction samples
print("\n" + "="*80)
print("SAMPLE PREDICTIONS")
print("="*80)
print("\nLap | Compound | Actual | Predicted | Residual | Deg. State")
print("-" * 65)
for idx in range(0, len(df_final), max(1, len(df_final)//10)):
    row = df_final.iloc[idx]
    print(f"{row['lap_number']:3.0f} | {row['compound']:8s} | {row['lap_time_seconds']:6.2f} | "
          f"{row['predicted_lap_time']:9.2f} | {row['lap_time_residual']:8.3f} | {row['degradation_normalized']:10.3f}")

# Save results
df_final.to_csv('calibrated_model_results.csv', index=False)

iterations_df = pd.DataFrame(iterations)
iterations_df.to_csv('calibration_history.csv', index=False)

print("\n" + "="*80)
print("CALIBRATION COMPLETE")
print("="*80)
print(f"\n[OK] Results saved to: calibrated_model_results.csv")
print(f"[OK] Calibration history: calibration_history.csv")
print(f"\nBest RMSE: {best_rmse:.4f} seconds")
print(f"This means predictions are typically within ±{best_rmse:.3f}s of actual lap times")

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Actual vs Predicted
ax1 = axes[0, 0]
ax1.scatter(df_final['lap_number'], df_final['lap_time_seconds'], alpha=0.6, label='Actual', s=80)
ax1.plot(df_final['lap_number'], df_final['predicted_lap_time'], 'r-', label='Predicted', linewidth=2)
ax1.set_xlabel('Lap Number', fontweight='bold', fontsize=12)
ax1.set_ylabel('Lap Time (seconds)', fontweight='bold', fontsize=12)
ax1.set_title('Actual vs Predicted Lap Times (Calibrated Model)', fontweight='bold', fontsize=14)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# Plot 2: Residuals
ax2 = axes[0, 1]
ax2.scatter(df_final['lap_number'], df_final['lap_time_residual'], alpha=0.6, s=60)
ax2.axhline(y=0, color='red', linestyle='--', linewidth=2)
ax2.axhline(y=metrics_final['rmse'], color='orange', linestyle=':', linewidth=1, label=f'±RMSE ({metrics_final["rmse"]:.3f}s)')
ax2.axhline(y=-metrics_final['rmse'], color='orange', linestyle=':', linewidth=1)
ax2.set_xlabel('Lap Number', fontweight='bold', fontsize=12)
ax2.set_ylabel('Residual (seconds)', fontweight='bold', fontsize=12)
ax2.set_title('Prediction Residuals', fontweight='bold', fontsize=14)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

# Plot 3: Calibration convergence
ax3 = axes[1, 0]
ax3.plot(iterations_df['iteration'], iterations_df['rmse'], 'b-', linewidth=2, marker='o', markersize=3)
ax3.axhline(y=best_rmse, color='green', linestyle='--', linewidth=2, label=f'Best: {best_rmse:.4f}s')
ax3.set_xlabel('Iteration', fontweight='bold', fontsize=12)
ax3.set_ylabel('RMSE (seconds)', fontweight='bold', fontsize=12)
ax3.set_title('Calibration Progress', fontweight='bold', fontsize=14)
ax3.legend(fontsize=11)
ax3.grid(True, alpha=0.3)

# Plot 4: R² progression
ax4 = axes[1, 1]
ax4.plot(iterations_df['iteration'], iterations_df['r_squared'], 'g-', linewidth=2, marker='o', markersize=3)
ax4.axhline(y=metrics_final['r_squared'], color='green', linestyle='--', linewidth=2,
            label=f'Final: {metrics_final["r_squared"]:.4f}')
ax4.set_xlabel('Iteration', fontweight='bold', fontsize=12)
ax4.set_ylabel('R² Score', fontweight='bold', fontsize=12)
ax4.set_title('Model Fit Quality (R²)', fontweight='bold', fontsize=14)
ax4.legend(fontsize=11)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('model_calibration_results.png', dpi=300, bbox_inches='tight')
print(f"[OK] Visualization saved: model_calibration_results.png")

plt.show()

print("\n" + "="*80)
print("READY TO USE CALIBRATED MODEL")
print("="*80)
print("\nYou can now use these optimized coefficients in the main notebook.")
print("The model has been tuned specifically for this driver/track combination.")
