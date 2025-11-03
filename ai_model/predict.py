"""
PyTorch F1 Lap Time Prediction - Inference Script

Use trained model to predict lap times for new data.
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
import pickle

print("="*80)
print("F1 LAP TIME PREDICTION - INFERENCE")
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

# Load model
print(f"\nLoading model...")
checkpoint = torch.load(model_dir / 'best_model.pth', map_location=device)
feature_names = checkpoint['feature_names']

model = F1LapTimePredictor(input_size=len(feature_names)).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"[OK] Model loaded (trained for {checkpoint['epoch']+1} epochs)")
print(f"[OK] Using {len(feature_names)} features")

# Load scalers
with open(model_dir / 'scaler_X.pkl', 'rb') as f:
    scaler_X = pickle.load(f)
with open(model_dir / 'scaler_y.pkl', 'rb') as f:
    scaler_y = pickle.load(f)

print(f"[OK] Scalers loaded")

# ============================================================================
#                          PREDICTION FUNCTION
# ============================================================================

def predict_lap_time(features_dict):
    """
    Predict lap time from feature dictionary.

    Args:
        features_dict: Dictionary with all required features

    Returns:
        Predicted lap time in seconds
    """
    # Convert to array in correct order
    features = np.array([features_dict[name] for name in feature_names]).reshape(1, -1)

    # Scale features
    features_scaled = scaler_X.transform(features)

    # Predict
    with torch.no_grad():
        features_tensor = torch.FloatTensor(features_scaled).to(device)
        prediction_scaled = model(features_tensor).cpu().numpy()

    # Unscale prediction
    prediction = scaler_y.inverse_transform(prediction_scaled)

    return prediction[0, 0]

def predict_batch(df):
    """
    Predict lap times for a DataFrame of features.

    Args:
        df: DataFrame with all required features

    Returns:
        Array of predicted lap times
    """
    # Extract features in correct order
    X = df[feature_names].values

    # Handle NaNs
    X = np.nan_to_num(X, nan=0.0)

    # Scale features
    X_scaled = scaler_X.transform(X)

    # Predict in batches
    batch_size = 1024
    predictions = []

    model.eval()
    with torch.no_grad():
        for i in range(0, len(X_scaled), batch_size):
            batch = X_scaled[i:i+batch_size]
            batch_tensor = torch.FloatTensor(batch).to(device)
            batch_pred_scaled = model(batch_tensor).cpu().numpy()
            predictions.append(batch_pred_scaled)

    predictions = np.vstack(predictions)

    # Unscale predictions
    predictions_unscaled = scaler_y.inverse_transform(predictions)

    return predictions_unscaled.flatten()

# ============================================================================
#                          EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print(f"\n{'='*80}")
    print("EXAMPLE: PREDICT ON TEST DATA")
    print(f"{'='*80}")

    # Load preprocessed data
    data_file = Path('./data/preprocessed_f1_data.csv')
    df = pd.read_csv(data_file)

    # Take last 100 laps as example
    test_df = df.tail(100).copy()

    print(f"\nPredicting on {len(test_df)} laps...")

    # Make predictions
    predictions = predict_batch(test_df)
    actual = test_df['lap_time_seconds'].values

    # Calculate error
    errors = predictions - actual
    rmse = np.sqrt(np.mean(errors ** 2))
    mae = np.mean(np.abs(errors))

    print(f"\n[OK] Predictions complete")
    print(f"\nPerformance on sample:")
    print(f"  RMSE: {rmse:.4f} seconds")
    print(f"  MAE:  {mae:.4f} seconds")

    # Show some examples
    print(f"\nSample predictions:")
    print(f"{'Actual':>8s} | {'Predicted':>10s} | {'Error':>8s}")
    print("-" * 35)
    for i in range(min(10, len(test_df))):
        print(f"{actual[i]:8.2f} | {predictions[i]:10.2f} | {errors[i]:8.2f}")

    # Save predictions
    output_file = Path('./outputs/sample_predictions.csv')
    output_file.parent.mkdir(exist_ok=True)

    result_df = test_df[['year', 'race', 'driver', 'lap_number', 'compound', 'lap_time_seconds']].copy()
    result_df['predicted_lap_time'] = predictions
    result_df['prediction_error'] = errors

    result_df.to_csv(output_file, index=False)
    print(f"\n[OK] Saved sample predictions: {output_file}")

    print(f"\n{'='*80}")
    print("READY FOR INFERENCE!")
    print(f"{'='*80}")
    print(f"\nUsage:")
    print(f"  from predict import predict_lap_time, predict_batch")
    print(f"  prediction = predict_lap_time(features_dict)")
    print(f"  predictions = predict_batch(dataframe)")
