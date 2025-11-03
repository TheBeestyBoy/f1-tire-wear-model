"""
Download HIGH-QUALITY F1 Data (2022-2024) - Validated Telemetry Only

Focuses on 2022-2024 seasons where FastF1 has complete, reliable telemetry data.
Validates data availability before processing to avoid missing telemetry issues.
"""

import fastf1
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import pickle
from datetime import datetime
warnings.filterwarnings('ignore')

# Reduce FastF1 verbosity
fastf1.set_log_level('WARNING')

# Setup cache
cache_dir = Path('../fastf1_cache')
cache_dir.mkdir(exist_ok=True)
fastf1.Cache.enable_cache(str(cache_dir))

# Output directory
output_dir = Path('./data')
output_dir.mkdir(exist_ok=True)

print("="*80)
print("DOWNLOADING HIGH-QUALITY F1 DATA (2022-2024 ONLY)")
print("="*80)

# Focus on 2022-2024 (reliable telemetry era)
# These years have complete car telemetry data from FastF1
YEARS = [2022, 2023, 2024]
SESSION_TYPE = 'R'  # Race sessions only

print(f"\nTarget years: {YEARS} (reliable telemetry)")
print(f"Session type: Race")
print(f"Estimated: ~60 races × 20 drivers × 55 laps = ~66,000 laps")

# ============================================================================
#                          DATA COLLECTION WITH VALIDATION
# ============================================================================

all_data = []
failed_sessions = []
successful_sessions = []
skipped_no_telemetry = []

total_laps = 0
total_races = 0

print("\n" + "="*80)
print("DOWNLOADING RACE DATA")
print("="*80)

for year in YEARS:
    print(f"\n{'='*80}")
    print(f"YEAR: {year}")
    print(f"{'='*80}")

    try:
        # Get all race events for this year
        schedule = fastf1.get_event_schedule(year)

        # Filter to actual races (exclude testing)
        race_events = schedule[schedule['EventFormat'] != 'testing']

        print(f"Found {len(race_events)} events in {year}\n")

        for idx, event in race_events.iterrows():
            race_name = event['EventName']
            round_num = event['RoundNumber']

            print(f"  [{year} R{round_num:2d}] {race_name:30s}...", end=' ', flush=True)

            try:
                # Load session
                session = fastf1.get_session(year, round_num, SESSION_TYPE)
                session.load()

                # VALIDATION: Check if telemetry is available
                if session.laps.empty:
                    skipped_no_telemetry.append(f"{year} {race_name}: No lap data")
                    print(f"✗ SKIP (no lap data)")
                    continue

                # Try to get telemetry from first lap to validate
                try:
                    test_lap = session.laps.iloc[0]
                    test_telemetry = test_lap.get_telemetry()
                    if test_telemetry.empty or 'Speed' not in test_telemetry.columns:
                        skipped_no_telemetry.append(f"{year} {race_name}: No telemetry")
                        print(f"✗ SKIP (no telemetry)")
                        continue
                except:
                    skipped_no_telemetry.append(f"{year} {race_name}: Telemetry error")
                    print(f"✗ SKIP (telemetry error)")
                    continue

                # Get all drivers who completed the race
                drivers = session.laps['Driver'].unique()
                race_laps = 0

                for driver in drivers:
                    driver_laps = session.laps.pick_driver(driver)
                    valid_laps = driver_laps[
                        (driver_laps['IsAccurate'] == True) &
                        (driver_laps['LapTime'].notna())
                    ].copy()

                    if len(valid_laps) < 5:  # Skip drivers with very few laps
                        continue

                    # Extract features for each lap
                    for lap_idx, lap in valid_laps.iterrows():
                        try:
                            # Get telemetry
                            telemetry = lap.get_telemetry()
                            if telemetry.empty:
                                continue

                            # Validate critical telemetry columns
                            required_cols = ['Speed', 'Throttle', 'Brake', 'RPM', 'nGear']
                            if not all(col in telemetry.columns for col in required_cols):
                                continue

                            # Basic lap data
                            lap_data = {
                                'year': year,
                                'race': race_name,
                                'round': round_num,
                                'driver': driver,
                                'lap_number': lap['LapNumber'],
                                'lap_time_seconds': lap['LapTime'].total_seconds(),

                                # Tire data
                                'compound': lap['Compound'],
                                'tyre_life': lap['TyreLife'],
                                'is_fresh_tyre': lap['FreshTyre'],

                                # Telemetry aggregates
                                'speed_mean': telemetry['Speed'].mean(),
                                'speed_max': telemetry['Speed'].max(),
                                'speed_std': telemetry['Speed'].std(),
                                'throttle_mean': telemetry['Throttle'].mean(),
                                'brake_mean': telemetry['Brake'].mean(),
                                'rpm_mean': telemetry['RPM'].mean(),
                                'n_gear_mean': telemetry['nGear'].mean(),
                            }

                            # Optional: DRS (may not always be available)
                            if 'DRS' in telemetry.columns:
                                lap_data['drs_usage'] = (telemetry['DRS'] > 0).sum() / len(telemetry) if len(telemetry) > 0 else 0
                            else:
                                lap_data['drs_usage'] = 0

                            # Braking stats
                            brake_points = telemetry[telemetry['Brake'] > 0]
                            if len(brake_points) > 0:
                                brake_changes = telemetry['Brake'].diff()
                                lap_data['n_braking_zones'] = max(1, (brake_changes > 50).sum())
                                lap_data['brake_intensity_mean'] = brake_points['Brake'].mean()
                            else:
                                lap_data['n_braking_zones'] = 0
                                lap_data['brake_intensity_mean'] = 0

                            # Cornering (speed variance)
                            lap_data['cornering_intensity'] = telemetry['Speed'].std()

                            # Weather data
                            if not session.weather_data.empty:
                                lap_start_time = lap['Time']
                                lap_end_time = lap_start_time + lap['LapTime']
                                lap_weather = session.weather_data[
                                    (session.weather_data['Time'] >= lap_start_time) &
                                    (session.weather_data['Time'] <= lap_end_time)
                                ]

                                if not lap_weather.empty:
                                    lap_data['track_temp'] = lap_weather['TrackTemp'].mean()
                                    lap_data['air_temp'] = lap_weather['AirTemp'].mean()
                                    lap_data['humidity'] = lap_weather['Humidity'].mean()
                                    lap_data['pressure'] = lap_weather['Pressure'].mean()
                                    lap_data['rainfall'] = lap_weather['Rainfall'].mean() if 'Rainfall' in lap_weather.columns else 0
                                    lap_data['wind_speed'] = lap_weather['WindSpeed'].mean() if 'WindSpeed' in lap_weather.columns else 0
                                else:
                                    lap_data['track_temp'] = np.nan
                                    lap_data['air_temp'] = np.nan
                                    lap_data['humidity'] = np.nan
                                    lap_data['pressure'] = np.nan
                                    lap_data['rainfall'] = 0
                                    lap_data['wind_speed'] = 0
                            else:
                                lap_data['track_temp'] = np.nan
                                lap_data['air_temp'] = np.nan
                                lap_data['humidity'] = np.nan
                                lap_data['pressure'] = np.nan
                                lap_data['rainfall'] = 0
                                lap_data['wind_speed'] = 0

                            # Sector times
                            lap_data['sector1_time'] = lap['Sector1Time'].total_seconds() if pd.notna(lap['Sector1Time']) else np.nan
                            lap_data['sector2_time'] = lap['Sector2Time'].total_seconds() if pd.notna(lap['Sector2Time']) else np.nan
                            lap_data['sector3_time'] = lap['Sector3Time'].total_seconds() if pd.notna(lap['Sector3Time']) else np.nan

                            all_data.append(lap_data)
                            race_laps += 1

                        except Exception as e:
                            continue  # Skip individual lap errors

                total_laps += race_laps
                total_races += 1
                successful_sessions.append(f"{year} {race_name}")
                print(f"✓ {race_laps:4d} laps")

            except Exception as e:
                failed_sessions.append(f"{year} {race_name}: {str(e)}")
                print(f"✗ FAILED: {str(e)[:40]}")
                continue

    except Exception as e:
        print(f"\n  ERROR loading {year} schedule: {str(e)}")
        continue

# ============================================================================
#                          SAVE RAW DATA
# ============================================================================

print("\n" + "="*80)
print("SAVING DATA")
print("="*80)

if len(all_data) == 0:
    print("\n[ERROR] No data collected!")
    print("\nTroubleshooting:")
    print("  - FastF1 may not have telemetry for the selected years/races")
    print("  - Try using more recent races (2023-2024)")
    print("  - Check FastF1 documentation for data availability")
    exit(1)

df = pd.DataFrame(all_data)

print(f"\n[OK] Collected {len(df):,} laps from {total_races} races")
print(f"     Years: {df['year'].min()} - {df['year'].max()}")
print(f"     Unique races: {df['race'].nunique()}")
print(f"     Unique drivers: {df['driver'].nunique()}")

# Save raw data
raw_file = output_dir / 'raw_f1_data.csv'
df.to_csv(raw_file, index=False)
print(f"\n[OK] Saved raw data: {raw_file}")

# Save metadata
metadata = {
    'download_date': datetime.now().isoformat(),
    'total_laps': len(df),
    'total_races': total_races,
    'years': YEARS,
    'successful_sessions': successful_sessions,
    'failed_sessions': failed_sessions,
    'skipped_no_telemetry': skipped_no_telemetry,
    'columns': list(df.columns),
    'shape': df.shape
}

metadata_file = output_dir / 'data_metadata.pkl'
with open(metadata_file, 'wb') as f:
    pickle.dump(metadata, f)
print(f"[OK] Saved metadata: {metadata_file}")

# ============================================================================
#                          DATA ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("DATA ANALYSIS")
print("="*80)

print(f"\nDataset Shape: {df.shape}")
print(f"  Rows (laps): {df.shape[0]:,}")
print(f"  Columns (features): {df.shape[1]}")

print(f"\nFeatures available:")
for i, col in enumerate(df.columns, 1):
    missing_pct = (df[col].isna().sum() / len(df)) * 100
    dtype = df[col].dtype
    print(f"  {i:2d}. {col:25s} ({str(dtype):10s}) - {missing_pct:5.1f}% missing")

print(f"\nLap time statistics:")
print(f"  Mean: {df['lap_time_seconds'].mean():.2f}s")
print(f"  Std:  {df['lap_time_seconds'].std():.2f}s")
print(f"  Min:  {df['lap_time_seconds'].min():.2f}s")
print(f"  Max:  {df['lap_time_seconds'].max():.2f}s")

print(f"\nCompound distribution:")
compound_counts = df['compound'].value_counts()
for compound, count in compound_counts.items():
    pct = (count / len(df)) * 100
    print(f"  {compound:15s}: {count:6,} laps ({pct:5.1f}%)")

print(f"\nRaces per year:")
year_counts = df.groupby('year')['race'].nunique().sort_index()
for year, count in year_counts.items():
    print(f"  {year}: {count:2d} races")

print(f"\nLaps per year:")
lap_counts = df.groupby('year').size().sort_index()
for year, count in lap_counts.items():
    print(f"  {year}: {count:6,} laps")

# ============================================================================
#                          PREPROCESSING FOR MODEL
# ============================================================================

print("\n" + "="*80)
print("PREPROCESSING FOR PYTORCH MODEL")
print("="*80)

# Sort by year, round, driver, lap_number
df = df.sort_values(['year', 'round', 'driver', 'lap_number']).reset_index(drop=True)

# Create stint detection (tire change)
df['stint_id'] = df.groupby(['year', 'round', 'driver']).apply(
    lambda x: ((x['compound'] != x['compound'].shift()) |
               (x['tyre_life'] < x['tyre_life'].shift())).cumsum()
).reset_index(level=[0,1,2], drop=True)

# Add previous lap time (KEY FEATURE for sequential prediction)
df['prev_lap_time'] = df.groupby(['year', 'round', 'driver'])['lap_time_seconds'].shift(1)

# Lap number in stint
df['lap_in_stint'] = df.groupby(['year', 'round', 'driver', 'stint_id']).cumcount() + 1

# Fill missing weather with interpolation
weather_cols = ['track_temp', 'air_temp', 'humidity', 'pressure', 'rainfall', 'wind_speed']
for col in weather_cols:
    if col in df.columns:
        df[col] = df.groupby(['year', 'round'])[col].transform(
            lambda x: x.interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')
        )

# Fill remaining NaNs with median
for col in weather_cols:
    if col in df.columns and df[col].isna().any():
        df[col] = df[col].fillna(df[col].median())

# Encode compound as numeric
compound_map = {'SOFT': 0, 'MEDIUM': 1, 'HARD': 2, 'INTERMEDIATE': 3, 'WET': 4}
df['compound_encoded'] = df['compound'].map(compound_map)

# Remove laps with missing critical data
df_clean = df.dropna(subset=['prev_lap_time', 'lap_time_seconds', 'compound_encoded']).copy()

print(f"\n[OK] Preprocessed data:")
print(f"     Original: {len(df):,} laps")
print(f"     Clean:    {len(df_clean):,} laps ({len(df_clean)/len(df)*100:.1f}%)")
print(f"     Removed:  {len(df) - len(df_clean):,} laps (missing critical data)")

# Save preprocessed data
clean_file = output_dir / 'preprocessed_f1_data.csv'
df_clean.to_csv(clean_file, index=False)
print(f"\n[OK] Saved preprocessed data: {clean_file}")

# ============================================================================
#                          MODEL INPUT/OUTPUT ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("PYTORCH MODEL SPECIFICATIONS")
print("="*80)

# Features for the model (excluding metadata and target)
feature_cols = [
    'prev_lap_time',        # CRITICAL: previous lap time
    'lap_number',           # Lap number (for fuel load proxy)
    'tyre_life',            # Tire age
    'compound_encoded',     # Tire type
    'lap_in_stint',         # Laps on current tires
    'speed_mean',
    'speed_max',
    'speed_std',
    'throttle_mean',
    'brake_mean',
    'rpm_mean',
    'n_gear_mean',
    'drs_usage',
    'n_braking_zones',
    'brake_intensity_mean',
    'cornering_intensity',
    'track_temp',
    'air_temp',
    'humidity',
    'pressure',
    'rainfall',
    'wind_speed',
    'sector1_time',
    'sector2_time',
    'sector3_time',
]

# Check which features are valid
valid_features = []
for col in feature_cols:
    if col in df_clean.columns:
        missing_pct = (df_clean[col].isna().sum() / len(df_clean)) * 100
        if missing_pct < 50:  # Keep features with < 50% missing
            valid_features.append(col)

print(f"\n[MODEL INPUT FEATURES] ({len(valid_features)} features)")
for i, feat in enumerate(valid_features, 1):
    missing_pct = (df_clean[feat].isna().sum() / len(df_clean)) * 100
    print(f"  {i:2d}. {feat:25s} - {missing_pct:5.1f}% missing")

print(f"\n[MODEL OUTPUT]")
print(f"  1. lap_time_seconds (regression target)")

print(f"\n[DATASET SPLIT RECOMMENDATION]")
total_samples = len(df_clean)
train_size = int(total_samples * 0.7)
val_size = int(total_samples * 0.15)
test_size = total_samples - train_size - val_size

print(f"  Total samples: {total_samples:,} laps")
print(f"  Suggested split:")
print(f"    - Train: {train_size:,} laps (70%)")
print(f"    - Val:   {val_size:,} laps (15%)")
print(f"    - Test:  {test_size:,} laps (15%)")

print(f"\n[PYTORCH MODEL ARCHITECTURE RECOMMENDATION]")
n_inputs = len(valid_features)
print(f"  Input size:  {n_inputs} features")
print(f"  Output size: 1 (lap_time_seconds)")
print(f"\n  Suggested network architecture:")
print(f"    Layer 1: Input({n_inputs}) -> Linear({n_inputs}, 128) -> ReLU -> BatchNorm -> Dropout(0.2)")
print(f"    Layer 2: Linear(128, 64) -> ReLU -> BatchNorm -> Dropout(0.2)")
print(f"    Layer 3: Linear(64, 32) -> ReLU -> BatchNorm -> Dropout(0.1)")
print(f"    Layer 4: Linear(32, 16) -> ReLU")
print(f"    Output:  Linear(16, 1)")

# Calculate parameters
params = n_inputs*128 + 128 + 128*64 + 64 + 64*32 + 32 + 32*16 + 16 + 16*1 + 1
print(f"\n  Total parameters: ~{params:,}")

# Determine if dataset size supports this architecture
samples_per_param = total_samples / params

print(f"\n  Dataset analysis:")
print(f"    Samples: {total_samples:,}")
print(f"    Parameters: ~{params:,}")
print(f"    Samples/Parameter ratio: {samples_per_param:.1f}")

if samples_per_param > 10:
    print(f"    ✓ EXCELLENT - Large dataset, low overfitting risk")
    recommended_arch = "4 hidden layers (128, 64, 32, 16)"
elif samples_per_param > 5:
    print(f"    ✓ GOOD - Adequate dataset size")
    recommended_arch = "3 hidden layers (128, 64, 32)"
else:
    print(f"    ⚠ MODERATE - Consider simpler architecture")
    recommended_arch = "2 hidden layers (64, 32)"

print(f"    Recommended: {recommended_arch}")

# Save feature list for model training
feature_file = output_dir / 'model_features.txt'
with open(feature_file, 'w') as f:
    f.write('\n'.join(valid_features))
print(f"\n[OK] Saved feature list: {feature_file}")

print("\n" + "="*80)
print("DATA DOWNLOAD COMPLETE")
print("="*80)
print(f"\nFiles created:")
print(f"  1. {raw_file.name} ({df.shape[0]:,} laps)")
print(f"  2. {clean_file.name} ({df_clean.shape[0]:,} laps)")
print(f"  3. {metadata_file.name}")
print(f"  4. {feature_file.name}")

if skipped_no_telemetry:
    print(f"\n[INFO] {len(skipped_no_telemetry)} sessions skipped (no telemetry):")
    for session in skipped_no_telemetry[:5]:
        print(f"  - {session}")
    if len(skipped_no_telemetry) > 5:
        print(f"  ... and {len(skipped_no_telemetry)-5} more")

if failed_sessions:
    print(f"\n[WARNING] {len(failed_sessions)} sessions failed:")
    for session in failed_sessions[:5]:
        print(f"  - {session}")
    if len(failed_sessions) > 5:
        print(f"  ... and {len(failed_sessions)-5} more")

print(f"\n{'='*80}")
print("READY FOR PYTORCH MODEL TRAINING!")
print(f"{'='*80}")
print(f"\nNext step: Run PyTorch training script")
