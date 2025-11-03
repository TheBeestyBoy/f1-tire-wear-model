# F1 Lap Time Prediction - PyTorch AI Model

Deep learning model to predict F1 lap times using telemetry, tire, and weather data.

## Dataset

- **126,419 laps** from 149 races (2018-2024)
- **25 input features**: telemetry, tires, weather, track conditions
- **1 output**: lap_time_seconds
- **Split**: 70% train / 15% val / 15% test

## Model Architecture

**4-Layer Feed-Forward Neural Network**

```
Input (25) → Linear(128) → ReLU → BatchNorm → Dropout(0.2)
          → Linear(64)  → ReLU → BatchNorm → Dropout(0.2)
          → Linear(32)  → ReLU → BatchNorm → Dropout(0.1)
          → Linear(16)  → ReLU
          → Linear(1)   → Output
```

- **Total Parameters**: ~13,968
- **Regularization**: BatchNorm + Dropout
- **Optimizer**: Adam (lr=0.001)
- **Loss**: MSE Loss

## Quick Start

### 1. Train the Model

```bash
python train_model.py
```

This will:
- Load 126k laps of preprocessed data
- Train for up to 150 epochs (with early stopping)
- Save best model to `models/best_model.pth`
- Generate training visualizations in `outputs/`

**Training Time**: ~5-15 minutes (CPU), ~2-5 minutes (GPU)

### 2. Make Predictions

```bash
python predict.py
```

Or use in your own code:

```python
from predict import predict_lap_time, predict_batch

# Single prediction
features = {
    'prev_lap_time': 85.2,
    'lap_number': 25,
    'tyre_life': 12,
    'compound_encoded': 1,  # MEDIUM
    'lap_in_stint': 12,
    'speed_mean': 180.5,
    # ... other features
}
predicted_time = predict_lap_time(features)

# Batch prediction
import pandas as pd
df = pd.read_csv('./data/preprocessed_f1_data.csv')
predictions = predict_batch(df)
```

## Files

### Data
- `data/raw_f1_data.csv` - 137,915 raw laps
- `data/preprocessed_f1_data.csv` - 126,419 clean laps (ready for training)
- `data/model_features.txt` - List of 25 input features
- `data/data_metadata.pkl` - Download metadata

### Scripts
- `download_all_f1_data.py` - Download all available F1 data (2018-2024)
- `download_quality_f1_data.py` - Download with validation (2022-2024 focus)
- `analyze_data.py` - Analyze downloaded data
- `train_model.py` - **Main training script**
- `predict.py` - **Inference script**

### Model Files (created after training)
- `models/best_model.pth` - Trained PyTorch model
- `models/scaler_X.pkl` - Feature scaler
- `models/scaler_y.pkl` - Target scaler

### Outputs (created after training)
- `outputs/training_results.png` - 4-panel visualization
- `outputs/test_predictions.csv` - Test set predictions
- `outputs/training_history.pkl` - Training metrics

## Input Features (25 total)

### Sequential (1)
- `prev_lap_time` - Previous lap time (critical!)

### Race Context (2)
- `lap_number` - Current lap (fuel load proxy)
- `lap_in_stint` - Laps on current tires

### Tire Data (2)
- `tyre_life` - Total laps on tires
- `compound_encoded` - 0=SOFT, 1=MEDIUM, 2=HARD, 3=INTER, 4=WET

### Telemetry - Speed (3)
- `speed_mean`, `speed_max`, `speed_std`

### Telemetry - Driving (7)
- `throttle_mean`, `brake_mean`, `rpm_mean`, `n_gear_mean`
- `drs_usage`, `n_braking_zones`, `brake_intensity_mean`

### Track Dynamics (1)
- `cornering_intensity` - Speed variance

### Weather (6)
- `track_temp`, `air_temp`, `humidity`, `pressure`, `rainfall`, `wind_speed`

### Track Sectors (3)
- `sector1_time`, `sector2_time`, `sector3_time`

## Expected Performance

- **Target RMSE**: < 3.0 seconds
- **Baseline (math model)**: 4.18 seconds
- **Expected improvement**: 30-50% better than math model

## Training Configuration

```python
BATCH_SIZE = 512
LEARNING_RATE = 0.001
EPOCHS = 150
EARLY_STOP_PATIENCE = 15
LR_REDUCE_PATIENCE = 7
```

## Requirements

```bash
pip install torch pandas numpy matplotlib scikit-learn
```

(FastF1 only needed if re-downloading data)

## Notes

- Model uses **previous lap time** as key feature (sequential prediction)
- All features are **standardized** (mean=0, std=1) before training
- **BatchNorm** helps with training stability
- **Dropout** prevents overfitting
- **Early stopping** prevents overtraining
- Model saved at **best validation loss**

## Comparison to Mathematical Model

| Model | RMSE | Method |
|-------|------|--------|
| Mathematical (current) | 4.18s | Physics-based formulas |
| PyTorch AI (expected) | ~2-3s | Deep learning |

The AI model should improve on the mathematical model by learning complex non-linear relationships in the data that are difficult to capture with formulas alone.
