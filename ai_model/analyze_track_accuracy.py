"""
Analyze Model Accuracy Per Track

Tests the AI model on each race circuit to determine which tracks
it predicts accurately and calculates average lap times per track.
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import json

print("="*80)
print("F1 AI MODEL - PER-TRACK ACCURACY ANALYSIS")
print("="*80)

# ============================================================================
#                          MODEL DEFINITION
# ============================================================================

class F1LapTimePredictor(nn.Module):
    def __init__(self, input_size=25):
        super(F1LapTimePredictor, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),

            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.1),

            nn.Linear(32, 16),
            nn.ReLU(),

            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.network(x)

# ============================================================================
#                          LOAD MODEL
# ============================================================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice: {device}")

model_dir = Path('./models')

# Load model
checkpoint = torch.load(model_dir / 'best_model.pth', map_location=device, weights_only=False)
feature_names = checkpoint['feature_names']

model = F1LapTimePredictor(input_size=len(feature_names)).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load scalers
with open(model_dir / 'scaler_X.pkl', 'rb') as f:
    scaler_X = pickle.load(f)
with open(model_dir / 'scaler_y.pkl', 'rb') as f:
    scaler_y = pickle.load(f)

print(f"[OK] Model loaded")

# ============================================================================
#                          LOAD DATA
# ============================================================================

data_file = Path('./data/preprocessed_f1_data.csv')
df = pd.read_csv(data_file)

print(f"[OK] Loaded {len(df):,} laps")
print(f"\nColumns: {df.columns.tolist()}")

# ============================================================================
#                          ANALYZE TRACKS
# ============================================================================

print(f"\n{'='*80}")
print("TRACK STATISTICS")
print(f"{'='*80}")

# Get unique tracks
tracks = df['race'].unique()
print(f"\nTotal tracks: {len(tracks)}")

track_stats = []

for track in sorted(tracks):
    track_data = df[df['race'] == track].copy()

    # Calculate statistics
    stats = {
        'track': track,
        'total_laps': len(track_data),
        'avg_lap_time': track_data['lap_time_seconds'].mean(),
        'min_lap_time': track_data['lap_time_seconds'].min(),
        'max_lap_time': track_data['lap_time_seconds'].max(),
        'std_lap_time': track_data['lap_time_seconds'].std(),
        'years': sorted(track_data['year'].unique()),
        'drivers': len(track_data['driver'].unique()),
        'compounds': track_data['compound'].unique().tolist(),
    }

    track_stats.append(stats)

# Create DataFrame
track_df = pd.DataFrame(track_stats)
track_df = track_df.sort_values('avg_lap_time')

print(f"\n{'Track':<25s} | {'Laps':>6s} | {'Avg Time':>10s} | {'Range':>15s} | {'Drivers':>7s}")
print("-" * 80)

for _, row in track_df.iterrows():
    range_str = f"{row['min_lap_time']:.1f}-{row['max_lap_time']:.1f}s"
    print(f"{row['track']:<25s} | {row['total_laps']:>6d} | {row['avg_lap_time']:>10.2f}s | {range_str:>15s} | {row['drivers']:>7d}")

# ============================================================================
#                          TEST MODEL PER TRACK
# ============================================================================

print(f"\n{'='*80}")
print("MODEL ACCURACY PER TRACK")
print(f"{'='*80}")

track_accuracy = []

for track in sorted(tracks):
    track_data = df[df['race'] == track].copy()

    # Skip if too few samples
    if len(track_data) < 100:
        continue

    # Prepare features
    X = track_data[feature_names].values
    y = track_data['lap_time_seconds'].values.reshape(-1, 1)

    # Handle NaNs
    X = np.nan_to_num(X, nan=0.0)
    y = np.nan_to_num(y, nan=0.0)

    # Scale
    X_scaled = scaler_X.transform(X)
    y_scaled = scaler_y.transform(y)

    # Predict
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_scaled).to(device)
        predictions_scaled = model(X_tensor).cpu().numpy()

    # Unscale
    predictions = scaler_y.inverse_transform(predictions_scaled).flatten()
    actual = y.flatten()

    # Calculate metrics
    errors = predictions - actual
    rmse = np.sqrt(np.mean(errors ** 2))
    mae = np.mean(np.abs(errors))
    max_error = np.max(np.abs(errors))

    # R² score
    ss_res = np.sum((actual - predictions) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    track_accuracy.append({
        'track': track,
        'laps': len(track_data),
        'rmse': rmse,
        'mae': mae,
        'max_error': max_error,
        'r2': r2,
        'avg_lap_time': track_data['lap_time_seconds'].mean(),
    })

# Sort by RMSE (best first)
accuracy_df = pd.DataFrame(track_accuracy)
accuracy_df = accuracy_df.sort_values('rmse')

print(f"\n{'Track':<25s} | {'Laps':>6s} | {'RMSE':>8s} | {'MAE':>8s} | {'R²':>8s} | {'Accuracy'}")
print("-" * 90)

for _, row in accuracy_df.iterrows():
    # Determine accuracy rating
    if row['rmse'] < 1.0:
        rating = "EXCELLENT"
    elif row['rmse'] < 2.0:
        rating = "VERY GOOD"
    elif row['rmse'] < 3.0:
        rating = "GOOD"
    elif row['rmse'] < 5.0:
        rating = "OK"
    else:
        rating = "POOR"

    print(f"{row['track']:<25s} | {row['laps']:>6d} | {row['rmse']:>8.2f}s | {row['mae']:>8.2f}s | {row['r2']:>8.4f} | {rating}")

# ============================================================================
#                          SUMMARY STATISTICS
# ============================================================================

print(f"\n{'='*80}")
print("SUMMARY")
print(f"{'='*80}")

print(f"\nOverall Model Performance:")
print(f"  Average RMSE across tracks: {accuracy_df['rmse'].mean():.2f}s")
print(f"  Median RMSE: {accuracy_df['rmse'].median():.2f}s")
print(f"  Best track: {accuracy_df.iloc[0]['track']} (RMSE: {accuracy_df.iloc[0]['rmse']:.2f}s)")
print(f"  Worst track: {accuracy_df.iloc[-1]['track']} (RMSE: {accuracy_df.iloc[-1]['rmse']:.2f}s)")

# Categorize tracks
excellent = accuracy_df[accuracy_df['rmse'] < 1.0]
very_good = accuracy_df[(accuracy_df['rmse'] >= 1.0) & (accuracy_df['rmse'] < 2.0)]
good = accuracy_df[(accuracy_df['rmse'] >= 2.0) & (accuracy_df['rmse'] < 3.0)]
ok = accuracy_df[(accuracy_df['rmse'] >= 3.0) & (accuracy_df['rmse'] < 5.0)]
poor = accuracy_df[accuracy_df['rmse'] >= 5.0]

print(f"\nAccuracy Distribution:")
print(f"  EXCELLENT (RMSE < 1.0s): {len(excellent)} tracks")
print(f"  VERY GOOD (1.0-2.0s):    {len(very_good)} tracks")
print(f"  GOOD (2.0-3.0s):         {len(good)} tracks")
print(f"  OK (3.0-5.0s):           {len(ok)} tracks")
print(f"  POOR (> 5.0s):           {len(poor)} tracks")

# ============================================================================
#                          SAVE RESULTS
# ============================================================================

output_dir = Path('./outputs')
output_dir.mkdir(exist_ok=True)

# Save track statistics
track_df.to_csv(output_dir / 'track_statistics.csv', index=False)
print(f"\n[OK] Saved track statistics: {output_dir / 'track_statistics.csv'}")

# Save accuracy results
accuracy_df.to_csv(output_dir / 'track_accuracy.csv', index=False)
print(f"[OK] Saved accuracy results: {output_dir / 'track_accuracy.csv'}")

# Save as JSON for frontend
track_data_for_ui = []
for _, row in track_df.iterrows():
    # Find accuracy for this track
    acc = accuracy_df[accuracy_df['track'] == row['track']]

    track_data_for_ui.append({
        'name': row['track'],
        'avg_lap_time': float(row['avg_lap_time']),
        'min_lap_time': float(row['min_lap_time']),
        'max_lap_time': float(row['max_lap_time']),
        'total_laps': int(row['total_laps']),
        'rmse': float(acc.iloc[0]['rmse']) if len(acc) > 0 else None,
        'mae': float(acc.iloc[0]['mae']) if len(acc) > 0 else None,
        'r2': float(acc.iloc[0]['r2']) if len(acc) > 0 else None,
    })

with open(output_dir / 'track_data.json', 'w') as f:
    json.dump(track_data_for_ui, f, indent=2)

print(f"[OK] Saved track data for UI: {output_dir / 'track_data.json'}")

print(f"\n{'='*80}")
print("ANALYSIS COMPLETE!")
print(f"{'='*80}")
