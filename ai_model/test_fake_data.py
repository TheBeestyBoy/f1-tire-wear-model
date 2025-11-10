"""
Test the AI model with synthetic/artificial data to evaluate "Game Mode" capabilities.

This script:
1. Generates fake scenarios with user-defined conditions
2. Tests if the model produces physically reasonable predictions
3. Validates that the model can handle synthetic inputs
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import sys

print("="*80)
print("F1 AI MODEL - SYNTHETIC DATA TESTING (GAME MODE)")
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
#                          LOAD MODEL AND SCALERS
# ============================================================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice: {device}")

model_dir = Path('./models')

# Check if model exists
if not (model_dir / 'best_model.pth').exists():
    print(f"\n[ERROR] Model not found at {model_dir / 'best_model.pth'}")
    print(f"Please train the model first: cd ai_model && python train_model.py")
    sys.exit(1)

# Load model
print(f"\nLoading model...")
checkpoint = torch.load(model_dir / 'best_model.pth', map_location=device, weights_only=False)
feature_names = checkpoint['feature_names']

model = F1LapTimePredictor(input_size=len(feature_names)).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"[OK] Model loaded (trained for {checkpoint['epoch']+1} epochs)")
print(f"[OK] Using {len(feature_names)} features")
print(f"\nFeature list:")
for i, feat in enumerate(feature_names, 1):
    print(f"  {i:2d}. {feat}")

# Load scalers
with open(model_dir / 'scaler_X.pkl', 'rb') as f:
    scaler_X = pickle.load(f)
with open(model_dir / 'scaler_y.pkl', 'rb') as f:
    scaler_y = pickle.load(f)

print(f"\n[OK] Scalers loaded")

# ============================================================================
#                          SYNTHETIC DATA GENERATOR
# ============================================================================

def generate_scenario(scenario_name, conditions):
    """
    Generate synthetic lap data for a given scenario.

    Args:
        scenario_name: Name of the scenario
        conditions: Dictionary with environmental/tire parameters

    Returns:
        Dictionary with all 25 features
    """
    # Base values (typical Monaco lap)
    base = {
        'prev_lap_time': 75.0,
        'lap_number': 30,
        'tyre_life': conditions.get('tire_age', 15),
        'compound_encoded': conditions.get('compound_encoded', 1),  # 0=SOFT, 1=MEDIUM, 2=HARD, 3=INTER, 4=WET
        'lap_in_stint': conditions.get('tire_age', 15),

        # Speed metrics (Monaco-typical)
        'speed_mean': 150.0,
        'speed_max': 280.0,
        'speed_std': 45.0,

        # Driving metrics
        'throttle_mean': conditions.get('throttle', 65.0),
        'brake_mean': conditions.get('brake', 25.0),
        'rpm_mean': 11000.0,
        'n_gear_mean': 4.5,
        'drs_usage': 0.15,
        'n_braking_zones': 15,
        'brake_intensity_mean': 3.5,
        'cornering_intensity': 2.8,

        # Weather
        'track_temp': conditions.get('track_temp', 30.0),
        'air_temp': conditions.get('air_temp', 25.0),
        'humidity': conditions.get('humidity', 50.0),
        'pressure': 1013.0,
        'rainfall': conditions.get('rainfall', 0.0),
        'wind_speed': 2.5,

        # Sector times (Monaco typical)
        'sector1_time': 25.0,
        'sector2_time': 25.0,
        'sector3_time': 25.0,
    }

    # Apply condition modifiers
    # Wet conditions slow the car down
    if conditions.get('rainfall', 0) > 0:
        base['speed_mean'] *= 0.85
        base['speed_max'] *= 0.90
        base['throttle_mean'] *= 0.80
        base['brake_mean'] *= 1.2

    # Older tires degrade performance
    tire_age = conditions.get('tire_age', 15)
    if tire_age > 20:
        base['speed_mean'] *= 0.98
        base['throttle_mean'] *= 0.95

    # Adjust prev_lap_time based on conditions for consistency
    # This helps the model see reasonable input patterns
    lap_time_modifier = 1.0
    if conditions.get('rainfall', 0) > 5:
        lap_time_modifier += 0.15  # Wet = slower
    if tire_age > 30:
        lap_time_modifier += 0.05  # Old tires = slower
    if conditions.get('track_temp', 30) < 15:
        lap_time_modifier += 0.03  # Cold = slower

    base['prev_lap_time'] = 75.0 * lap_time_modifier

    return base

# ============================================================================
#                          TEST SCENARIOS
# ============================================================================

print(f"\n{'='*80}")
print("GENERATING SYNTHETIC TEST SCENARIOS")
print(f"{'='*80}")

scenarios = [
    # Baseline scenario
    {
        'name': 'Baseline (Dry, Medium, Fresh)',
        'conditions': {
            'tire_age': 5,
            'compound_encoded': 1,  # MEDIUM
            'track_temp': 30.0,
            'air_temp': 25.0,
            'humidity': 50.0,
            'rainfall': 0.0,
            'throttle': 65.0,
            'brake': 25.0,
        },
        'expected_behavior': 'Fast lap time (70-80s)'
    },

    # Old tires
    {
        'name': 'Old Tires (35 laps old)',
        'conditions': {
            'tire_age': 35,
            'compound_encoded': 1,  # MEDIUM
            'track_temp': 30.0,
            'air_temp': 25.0,
            'humidity': 50.0,
            'rainfall': 0.0,
            'throttle': 62.0,
            'brake': 26.0,
        },
        'expected_behavior': 'Slower than baseline (degradation)'
    },

    # Wet conditions
    {
        'name': 'Wet Track (Rain)',
        'conditions': {
            'tire_age': 5,
            'compound_encoded': 3,  # INTERMEDIATE
            'track_temp': 20.0,
            'air_temp': 18.0,
            'humidity': 85.0,
            'rainfall': 10.0,
            'throttle': 55.0,
            'brake': 30.0,
        },
        'expected_behavior': 'Much slower (wet conditions)'
    },

    # Hot dry conditions
    {
        'name': 'Hot Dry Track',
        'conditions': {
            'tire_age': 10,
            'compound_encoded': 2,  # HARD
            'track_temp': 45.0,
            'air_temp': 35.0,
            'humidity': 30.0,
            'rainfall': 0.0,
            'throttle': 70.0,
            'brake': 23.0,
        },
        'expected_behavior': 'Fast but hot tires degrade faster'
    },

    # Cold conditions
    {
        'name': 'Cold Track',
        'conditions': {
            'tire_age': 5,
            'compound_encoded': 0,  # SOFT
            'track_temp': 15.0,
            'air_temp': 12.0,
            'humidity': 60.0,
            'rainfall': 0.0,
            'throttle': 63.0,
            'brake': 27.0,
        },
        'expected_behavior': 'Slower (less grip in cold)'
    },

    # Very old tires + wet
    {
        'name': 'Old Tires + Wet (Worst Case)',
        'conditions': {
            'tire_age': 40,
            'compound_encoded': 1,  # MEDIUM (wrong compound!)
            'track_temp': 18.0,
            'air_temp': 16.0,
            'humidity': 90.0,
            'rainfall': 15.0,
            'throttle': 50.0,
            'brake': 35.0,
        },
        'expected_behavior': 'Very slow (multiple negative factors)'
    },

    # Fresh soft tires, optimal conditions
    {
        'name': 'Optimal (Fresh Softs, Perfect Weather)',
        'conditions': {
            'tire_age': 1,
            'compound_encoded': 0,  # SOFT
            'track_temp': 28.0,
            'air_temp': 24.0,
            'humidity': 45.0,
            'rainfall': 0.0,
            'throttle': 70.0,
            'brake': 22.0,
        },
        'expected_behavior': 'Fastest possible lap time'
    },
]

print(f"\n[OK] Created {len(scenarios)} test scenarios\n")

# ============================================================================
#                          RUN PREDICTIONS
# ============================================================================

print(f"{'='*80}")
print("RUNNING PREDICTIONS")
print(f"{'='*80}\n")

results = []

for scenario in scenarios:
    # Generate features
    features = generate_scenario(scenario['name'], scenario['conditions'])

    # Convert to array in correct order
    X = np.array([features[name] for name in feature_names]).reshape(1, -1)

    # Scale features
    X_scaled = scaler_X.transform(X)

    # Predict
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_scaled).to(device)
        prediction_scaled = model(X_tensor).cpu().numpy()

    # Unscale prediction
    prediction = scaler_y.inverse_transform(prediction_scaled)[0, 0]

    # Store result
    results.append({
        'scenario': scenario['name'],
        'predicted_time': prediction,
        'expected_behavior': scenario['expected_behavior'],
        'tire_age': scenario['conditions'].get('tire_age', 0),
        'compound': ['SOFT', 'MEDIUM', 'HARD', 'INTER', 'WET'][scenario['conditions'].get('compound_encoded', 1)],
        'track_temp': scenario['conditions'].get('track_temp', 30),
        'rainfall': scenario['conditions'].get('rainfall', 0),
    })

    print(f"Scenario: {scenario['name']}")
    print(f"  Conditions:")
    print(f"    - Tire: {results[-1]['compound']}, {results[-1]['tire_age']} laps old")
    print(f"    - Track Temp: {results[-1]['track_temp']:.1f}°C")
    print(f"    - Rainfall: {results[-1]['rainfall']:.1f} mm")
    print(f"  Predicted Lap Time: {prediction:.2f}s")
    print(f"  Expected: {scenario['expected_behavior']}")
    print()

# ============================================================================
#                          EVALUATE PHYSICAL REASONABLENESS
# ============================================================================

print(f"{'='*80}")
print("PHYSICAL REASONABLENESS ANALYSIS")
print(f"{'='*80}\n")

# Sort by predicted time
results_sorted = sorted(results, key=lambda x: x['predicted_time'])

print("Scenarios ranked by speed (fastest to slowest):")
print(f"{'Rank':>4s} | {'Time':>8s} | {'Scenario':<50s}")
print("-" * 70)
for i, res in enumerate(results_sorted, 1):
    print(f"{i:4d} | {res['predicted_time']:8.2f} | {res['scenario']}")

print("\n" + "="*80)
print("VALIDATION CHECKS")
print("="*80 + "\n")

# Check 1: Wet should be slower than dry
baseline = next(r for r in results if 'Baseline' in r['scenario'])
wet = next(r for r in results if 'Wet Track' in r['scenario'])

check1 = wet['predicted_time'] > baseline['predicted_time']
print(f"Check 1: Wet slower than dry?")
print(f"  Baseline (dry):  {baseline['predicted_time']:.2f}s")
print(f"  Wet:            {wet['predicted_time']:.2f}s")
print(f"  Difference:     {wet['predicted_time'] - baseline['predicted_time']:+.2f}s")
print(f"  RESULT: {'PASS' if check1 else 'FAIL'}\n")

# Check 2: Old tires slower than fresh
fresh = next(r for r in results if 'Baseline' in r['scenario'])
old = next(r for r in results if 'Old Tires' in r['scenario'])

check2 = old['predicted_time'] > fresh['predicted_time']
print(f"Check 2: Old tires slower than fresh?")
print(f"  Fresh (5 laps):  {fresh['predicted_time']:.2f}s")
print(f"  Old (35 laps):   {old['predicted_time']:.2f}s")
print(f"  Difference:      {old['predicted_time'] - fresh['predicted_time']:+.2f}s")
print(f"  RESULT: {'PASS' if check2 else 'FAIL'}\n")

# Check 3: Optimal should be fastest
optimal = next(r for r in results if 'Optimal' in r['scenario'])
check3 = optimal['predicted_time'] == min(r['predicted_time'] for r in results)

print(f"Check 3: Optimal conditions produce fastest time?")
print(f"  Optimal time:   {optimal['predicted_time']:.2f}s")
print(f"  Fastest time:   {min(r['predicted_time'] for r in results):.2f}s")
print(f"  RESULT: {'PASS' if check3 else 'FAIL'}\n")

# Check 4: Worst case should be slowest
worst = next(r for r in results if 'Worst Case' in r['scenario'])
check4 = worst['predicted_time'] == max(r['predicted_time'] for r in results)

print(f"Check 4: Worst conditions produce slowest time?")
print(f"  Worst time:     {worst['predicted_time']:.2f}s")
print(f"  Slowest time:   {max(r['predicted_time'] for r in results):.2f}s")
print(f"  RESULT: {'PASS' if check4 else 'FAIL'}\n")

# Overall summary
all_checks = [check1, check2, check3, check4]
passed = sum(all_checks)

print("="*80)
print(f"OVERALL RESULT: {passed}/{len(all_checks)} checks passed")
print("="*80 + "\n")

if passed == len(all_checks):
    print("EXCELLENT! The model shows physically reasonable behavior on synthetic data.")
    print("  The model is ready for 'Game Mode' scenario simulation.")
elif passed >= len(all_checks) * 0.75:
    print("GOOD. Most checks passed, but some improvements needed.")
    print("  The model may need fine-tuning for extreme scenarios.")
else:
    print("WARNING. The model struggles with synthetic data.")
    print("  Consider retraining with augmented data or adjusting feature scaling.")

# ============================================================================
#                          SAVE RESULTS
# ============================================================================

print(f"\n{'='*80}")
print("SAVING RESULTS")
print(f"{'='*80}")

output_dir = Path('./outputs')
output_dir.mkdir(exist_ok=True)

# Save as CSV
results_df = pd.DataFrame(results)
csv_file = output_dir / 'synthetic_test_results.csv'
results_df.to_csv(csv_file, index=False)
print(f"\n[OK] Saved results: {csv_file}")

# Save validation summary
summary_file = output_dir / 'synthetic_test_summary.txt'
with open(summary_file, 'w') as f:
    f.write("F1 AI MODEL - SYNTHETIC DATA TEST SUMMARY\n")
    f.write("="*80 + "\n\n")
    f.write("VALIDATION CHECKS:\n")
    f.write(f"  1. Wet slower than dry:        {'PASS' if check1 else 'FAIL'}\n")
    f.write(f"  2. Old tires slower than fresh: {'PASS' if check2 else 'FAIL'}\n")
    f.write(f"  3. Optimal = fastest:           {'PASS' if check3 else 'FAIL'}\n")
    f.write(f"  4. Worst = slowest:             {'PASS' if check4 else 'FAIL'}\n")
    f.write(f"\n  OVERALL: {passed}/{len(all_checks)} checks passed\n\n")

    f.write("PREDICTED LAP TIMES:\n")
    for res in results_sorted:
        f.write(f"  {res['predicted_time']:6.2f}s - {res['scenario']}\n")

print(f"[OK] Saved summary: {summary_file}")

print(f"\n{'='*80}")
print("TEST COMPLETE!")
print(f"{'='*80}")
