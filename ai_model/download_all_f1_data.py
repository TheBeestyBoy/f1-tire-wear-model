"""
Download ALL available FastF1 data for AI model training

This script downloads all available F1 race data from FastF1 and prepares it
for PyTorch model training. It extracts all telemetry, weather, and lap data.
"""

import fastf1
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import pickle
from datetime import datetime
warnings.filterwarnings('ignore')

# Setup cache
cache_dir = Path('../fastf1_cache')
cache_dir.mkdir(exist_ok=True)
fastf1.Cache.enable_cache(str(cache_dir))

# Output directory
output_dir = Path('./data')
output_dir.mkdir(exist_ok=True)

print("="*80)
print("DOWNLOADING ALL FASTF1 DATA FOR AI MODEL TRAINING")
print("="*80)

# FastF1 has data from 2018 onwards (reliable telemetry)
YEARS = list(range(2018, 2025))  # 2018-2024
SESSION_TYPE = 'R'  # Race sessions only (most complete data)

print(f"\nTarget years: {YEARS}")
print(f"Session type: Race")

# ============================================================================
#                          DATA COLLECTION
# ============================================================================

all_data = []
failed_sessions = []
successful_sessions = []

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
        race_events = schedule[schedule['EventFormat'] != 'testing']

        print(f"Found {len(race_events)} events in {year}")

        for idx, event in race_events.iterrows():
            race_name = event['EventName']
            round_num = event['RoundNumber']

            try:
                print(f"\n  [{year} Round {round_num}] {race_name}...", end=' ')

                # Load session
                session = fastf1.get_session(year, round_num, SESSION_TYPE)
                session.load()

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
                                'drs_usage': (telemetry['DRS'] > 0).sum() / len(telemetry),
                            }

                            # Braking stats
                            brake_points = telemetry[telemetry['Brake'] > 0]
                            if len(brake_points) > 0:
                                lap_data['n_braking_zones'] = (telemetry['Brake'].diff() > 50).sum()
                                lap_data['brake_intensity_mean'] = brake_points['Brake'].mean()
                            else:
                                lap_data['n_braking_zones'] = 0
                                lap_data['brake_intensity_mean'] = 0

                            # Cornering (speed variance)
                            lap_data['cornering_intensity'] = telemetry['Speed'].std()

                            # Weather data
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
                                lap_data['rainfall'] = lap_weather['Rainfall'].mean()
                                lap_data['wind_speed'] = lap_weather['WindSpeed'].mean()
                            else:
                                lap_data['track_temp'] = np.nan
                                lap_data['air_temp'] = np.nan
                                lap_data['humidity'] = np.nan
                                lap_data['pressure'] = np.nan
                                lap_data['rainfall'] = np.nan
                                lap_data['wind_speed'] = np.nan

                            # Track position
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
                print(f"✓ {race_laps} laps")

            except Exception as e:
                failed_sessions.append(f"{year} {race_name}: {str(e)}")
                print(f"✗ FAILED: {str(e)[:50]}")
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
    exit(1)

df = pd.DataFrame(all_data)

print(f"\n[OK] Collected {len(df)} laps from {total_races} races")
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
    print(f"  {i:2d}. {col:25s} ({dtype:8s}) - {missing_pct:5.1f}% missing")

print(f"\nLap time statistics:")
print(f"  Mean: {df['lap_time_seconds'].mean():.2f}s")
print(f"  Std:  {df['lap_time_seconds'].std():.2f}s")
print(f"  Min:  {df['lap_time_seconds'].min():.2f}s")
print(f"  Max:  {df['lap_time_seconds'].max():.2f}s")

print(f"\nCompound distribution:")
print(df['compound'].value_counts())

print(f"\nRaces per year:")
print(df.groupby('year')['race'].nunique().sort_index())

print(f"\nLaps per year:")
print(df.groupby('year').size().sort_index())

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

# Add previous lap time (key feature for sequential model)
df['prev_lap_time'] = df.groupby(['year', 'round', 'driver'])['lap_time_seconds'].shift(1)

# Lap number in stint
df['lap_in_stint'] = df.groupby(['year', 'round', 'driver', 'stint_id']).cumcount() + 1

# Fill missing weather with interpolation
weather_cols = ['track_temp', 'air_temp', 'humidity', 'pressure', 'rainfall', 'wind_speed']
for col in weather_cols:
    df[col] = df.groupby(['year', 'round'])[col].transform(
        lambda x: x.interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')
    )

# Encode compound as numeric
compound_map = {'SOFT': 0, 'MEDIUM': 1, 'HARD': 2, 'INTERMEDIATE': 3, 'WET': 4}
df['compound_encoded'] = df['compound'].map(compound_map)

# Remove laps with missing critical data
df_clean = df.dropna(subset=['prev_lap_time', 'lap_time_seconds']).copy()

print(f"\n[OK] Preprocessed data:")
print(f"     Original: {len(df):,} laps")
print(f"     Clean:    {len(df_clean):,} laps ({len(df_clean)/len(df)*100:.1f}%)")
print(f"     Removed:  {len(df) - len(df_clean):,} laps (missing prev_lap_time)")

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

# Check which features have too much missing data
valid_features = []
for col in feature_cols:
    if col in df_clean.columns:
        missing_pct = (df_clean[col].isna().sum() / len(df_clean)) * 100
        if missing_pct < 50:  # Keep features with < 50% missing
            valid_features.append(col)
        else:
            print(f"  [SKIP] {col} - {missing_pct:.1f}% missing (too high)")

print(f"\n[MODEL INPUT FEATURES] ({len(valid_features)} features)")
for i, col in enumerate(valid_features, 1):
    missing_pct = (df_clean[col].isna().sum() / len(df_clean)) * 100
    print(f"  {i:2d}. {col:25s} - {missing_pct:5.1f}% missing")

print(f"\n[MODEL OUTPUT]")
print(f"  1. lap_time_seconds (regression target)")

print(f"\n[DATASET SPLIT RECOMMENDATION]")
print(f"  Total samples: {len(df_clean):,} laps")
print(f"  Suggested train/val/test split:")
print(f"    - Train: {int(len(df_clean)*0.7):,} laps (70%)")
print(f"    - Val:   {int(len(df_clean)*0.15):,} laps (15%)")
print(f"    - Test:  {int(len(df_clean)*0.15):,} laps (15%)")

print(f"\n[PYTORCH MODEL ARCHITECTURE RECOMMENDATION]")
print(f"  Input size:  {len(valid_features)} features")
print(f"  Output size: 1 (lap_time_seconds)")
print(f"  Suggested network:")
print(f"    - Input layer: {len(valid_features)} neurons")
print(f"    - Hidden 1: 128 neurons + ReLU + Dropout(0.2)")
print(f"    - Hidden 2: 64 neurons + ReLU + Dropout(0.2)")
print(f"    - Hidden 3: 32 neurons + ReLU")
print(f"    - Output: 1 neuron (linear)")
print(f"  Total parameters: ~{(len(valid_features)*128 + 128*64 + 64*32 + 32*1):,}")

print(f"\n  Dataset size: {len(df_clean):,} samples")
if len(df_clean) > 100000:
    print(f"  ✓ Large dataset - can support deep network without overfitting")
elif len(df_clean) > 50000:
    print(f"  ✓ Medium dataset - good for 2-3 hidden layers")
else:
    print(f"  ⚠ Small dataset - consider simpler network (1-2 hidden layers)")

# Save feature list for model training
feature_file = output_dir / 'model_features.txt'
with open(feature_file, 'w') as f:
    f.write('\n'.join(valid_features))
print(f"\n[OK] Saved feature list: {feature_file}")

print("\n" + "="*80)
print("DATA DOWNLOAD COMPLETE")
print("="*80)
print(f"\nFiles created:")
print(f"  1. {raw_file}")
print(f"  2. {clean_file}")
print(f"  3. {metadata_file}")
print(f"  4. {feature_file}")

if failed_sessions:
    print(f"\n[WARNING] {len(failed_sessions)} sessions failed to download:")
    for session in failed_sessions[:10]:  # Show first 10
        print(f"  - {session}")
    if len(failed_sessions) > 10:
        print(f"  ... and {len(failed_sessions)-10} more")

print(f"\nReady for PyTorch model training!")
