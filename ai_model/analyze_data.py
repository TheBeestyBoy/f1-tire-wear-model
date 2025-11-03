"""Quick data analysis to determine PyTorch model architecture"""

import pandas as pd
import numpy as np
from pathlib import Path

print("="*80)
print("ANALYZING DOWNLOADED F1 DATA")
print("="*80)

# Load raw data
data_file = Path('./data/raw_f1_data.csv')
df = pd.read_csv(data_file)

print(f"\n[OK] Loaded {len(df):,} laps")
print(f"\nDataset Shape: {df.shape}")
print(f"  Rows (laps): {df.shape[0]:,}")
print(f"  Columns: {df.shape[1]}")

# Show columns and missing data
print(f"\nFeatures:")
for i, col in enumerate(df.columns, 1):
    missing_pct = (df[col].isna().sum() / len(df)) * 100
    dtype = str(df[col].dtype)
    print(f"  {i:2d}. {col:25s} ({dtype:10s}) - {missing_pct:5.1f}% missing")

# Basic stats
print(f"\nLap time statistics:")
print(f"  Mean: {df['lap_time_seconds'].mean():.2f}s")
print(f"  Std:  {df['lap_time_seconds'].std():.2f}s")
print(f"  Min:  {df['lap_time_seconds'].min():.2f}s")
print(f"  Max:  {df['lap_time_seconds'].max():.2f}s")

print(f"\nYears:")
print(df['year'].value_counts().sort_index())

print(f"\nCompounds:")
print(df['compound'].value_counts())

# Preprocessing
print(f"\n{'='*80}")
print("PREPROCESSING")
print(f"{'='*80}")

df = df.sort_values(['year', 'race', 'driver', 'lap_number']).reset_index(drop=True)

# Stint detection
df['stint_id'] = df.groupby(['year', 'race', 'driver']).apply(
    lambda x: ((x['compound'] != x['compound'].shift()) |
               (x['tyre_life'] < x['tyre_life'].shift())).cumsum()
).reset_index(level=[0,1,2], drop=True)

# Previous lap time (CRITICAL)
df['prev_lap_time'] = df.groupby(['year', 'race', 'driver'])['lap_time_seconds'].shift(1)

# Lap in stint
df['lap_in_stint'] = df.groupby(['year', 'race', 'driver', 'stint_id']).cumcount() + 1

# Fill weather
weather_cols = ['track_temp', 'air_temp', 'humidity', 'pressure', 'rainfall', 'wind_speed']
for col in weather_cols:
    if col in df.columns:
        df[col] = df.groupby(['year', 'race'])[col].transform(
            lambda x: x.interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')
        )
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

# Encode compound
compound_map = {'SOFT': 0, 'MEDIUM': 1, 'HARD': 2, 'INTERMEDIATE': 3, 'WET': 4}
df['compound_encoded'] = df['compound'].map(compound_map)

# Clean
df_clean = df.dropna(subset=['prev_lap_time', 'lap_time_seconds', 'compound_encoded']).copy()

print(f"\n[OK] Preprocessed:")
print(f"  Original: {len(df):,} laps")
print(f"  Clean:    {len(df_clean):,} laps ({len(df_clean)/len(df)*100:.1f}%)")
print(f"  Removed:  {len(df) - len(df_clean):,} laps")

# Save
clean_file = Path('./data/preprocessed_f1_data.csv')
df_clean.to_csv(clean_file, index=False)
print(f"\n[OK] Saved: {clean_file}")

# Model specs
print(f"\n{'='*80}")
print("PYTORCH MODEL SPECIFICATIONS")
print(f"{'='*80}")

feature_cols = [
    'prev_lap_time', 'lap_number', 'tyre_life', 'compound_encoded', 'lap_in_stint',
    'speed_mean', 'speed_max', 'speed_std', 'throttle_mean', 'brake_mean',
    'rpm_mean', 'n_gear_mean', 'drs_usage', 'n_braking_zones', 'brake_intensity_mean',
    'cornering_intensity', 'track_temp', 'air_temp', 'humidity', 'pressure',
    'rainfall', 'wind_speed', 'sector1_time', 'sector2_time', 'sector3_time',
]

valid_features = []
for col in feature_cols:
    if col in df_clean.columns:
        missing_pct = (df_clean[col].isna().sum() / len(df_clean)) * 100
        if missing_pct < 50:
            valid_features.append(col)

print(f"\n[INPUT FEATURES] ({len(valid_features)} features):")
for i, feat in enumerate(valid_features, 1):
    missing_pct = (df_clean[feat].isna().sum() / len(df_clean)) * 100
    print(f"  {i:2d}. {feat:25s} - {missing_pct:5.1f}% missing")

print(f"\n[OUTPUT]: lap_time_seconds")

total_samples = len(df_clean)
train_size = int(total_samples * 0.7)
val_size = int(total_samples * 0.15)
test_size = total_samples - train_size - val_size

print(f"\n[DATASET SPLIT]:")
print(f"  Total:  {total_samples:,} laps")
print(f"  Train:  {train_size:,} laps (70%)")
print(f"  Val:    {val_size:,} laps (15%)")
print(f"  Test:   {test_size:,} laps (15%)")

n_inputs = len(valid_features)
params = n_inputs*128 + 128*64 + 64*32 + 32*16 + 16*1
ratio = total_samples / params

print(f"\n[MODEL ARCHITECTURE]:")
print(f"  Input:   {n_inputs} features")
print(f"  Hidden:  128 -> 64 -> 32 -> 16")
print(f"  Output:  1 (lap_time)")
print(f"  Params:  ~{params:,}")
print(f"  Sample/Param ratio: {ratio:.1f}")

if ratio > 10:
    print(f"  ✓ EXCELLENT - Very low overfitting risk!")
elif ratio > 5:
    print(f"  ✓ GOOD - Adequate dataset size")
else:
    print(f"  ⚠ MODERATE - Consider simpler model")

# Save feature list
feature_file = Path('./data/model_features.txt')
with open(feature_file, 'w') as f:
    f.write('\n'.join(valid_features))
print(f"\n[OK] Saved features: {feature_file}")

print(f"\n{'='*80}")
print("READY FOR MODEL TRAINING!")
print(f"{'='*80}")
