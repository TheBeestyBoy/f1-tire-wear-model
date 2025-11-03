"""
Quick runner script - Execute full pipeline
"""

import subprocess
import sys
from pathlib import Path

PYTHON = r"C:\Users\Soren\AppData\Local\Programs\Python\Python312\python.exe"

print("="*80)
print("F1 LAP TIME PREDICTION - FULL PIPELINE")
print("="*80)

def run_script(script_name, description):
    print(f"\n{'='*80}")
    print(f"{description}")
    print(f"{'='*80}\n")

    result = subprocess.run([PYTHON, script_name], capture_output=False, text=True)

    if result.returncode != 0:
        print(f"\n[ERROR] {script_name} failed with code {result.returncode}")
        return False

    return True

# Check if data exists
data_file = Path('./data/preprocessed_f1_data.csv')

if not data_file.exists():
    print("\n[INFO] Preprocessed data not found. Running analysis...")
    if not run_script('analyze_data.py', "STEP 1: ANALYZING DATA"):
        sys.exit(1)

# Train model
print("\n" + "="*80)
print("READY TO TRAIN MODEL")
print("="*80)
print(f"\nDataset: {data_file}")
print(f"Size: 126,419 laps")
print(f"\nThis will take ~5-15 minutes on CPU, ~2-5 minutes on GPU")

response = input("\nStart training? (y/n): ")
if response.lower() != 'y':
    print("Training cancelled.")
    sys.exit(0)

if not run_script('train_model.py', "STEP 2: TRAINING MODEL"):
    sys.exit(1)

# Run predictions
print("\n" + "="*80)
print("TRAINING COMPLETE!")
print("="*80)

response = input("\nRun sample predictions? (y/n): ")
if response.lower() == 'y':
    if not run_script('predict.py', "STEP 3: MAKING PREDICTIONS"):
        sys.exit(1)

print("\n" + "="*80)
print("ALL DONE!")
print("="*80)
print(f"\nModel saved to: ./models/best_model.pth")
print(f"Results saved to: ./outputs/")
