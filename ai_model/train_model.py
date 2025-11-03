"""
PyTorch F1 Lap Time Prediction Model - Training Script

Trains a 4-layer feed-forward neural network to predict lap times based on
telemetry, tire, and weather data from 126,419 laps across 2018-2024 seasons.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pickle
import time

print("="*80)
print("F1 LAP TIME PREDICTION - PYTORCH MODEL TRAINING")
print("="*80)

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ============================================================================
#                          HYPERPARAMETERS
# ============================================================================

BATCH_SIZE = 512
LEARNING_RATE = 0.001
EPOCHS = 150
EARLY_STOP_PATIENCE = 15
LR_REDUCE_PATIENCE = 7

print(f"\nHyperparameters:")
print(f"  Batch Size: {BATCH_SIZE}")
print(f"  Learning Rate: {LEARNING_RATE}")
print(f"  Max Epochs: {EPOCHS}")
print(f"  Early Stop Patience: {EARLY_STOP_PATIENCE}")

# ============================================================================
#                          LOAD DATA
# ============================================================================

print(f"\n{'='*80}")
print("LOADING DATA")
print(f"{'='*80}")

data_file = Path('./data/preprocessed_f1_data.csv')
feature_file = Path('./data/model_features.txt')

print(f"\nLoading: {data_file}")
df = pd.read_csv(data_file)
print(f"[OK] Loaded {len(df):,} laps")

# Load feature list
with open(feature_file, 'r') as f:
    feature_names = [line.strip() for line in f.readlines()]

print(f"[OK] Using {len(feature_names)} features")

# Prepare features and target
X = df[feature_names].values
y = df['lap_time_seconds'].values.reshape(-1, 1)

print(f"\nData shapes:")
print(f"  X (features): {X.shape}")
print(f"  y (target):   {y.shape}")

# Handle any remaining NaNs
X = np.nan_to_num(X, nan=0.0)
y = np.nan_to_num(y, nan=0.0)

# Standardize features (critical for neural networks)
print(f"\nStandardizing features...")
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

print(f"[OK] Features standardized (mean=0, std=1)")

# Save scalers for inference
scaler_dir = Path('./models')
scaler_dir.mkdir(exist_ok=True)

with open(scaler_dir / 'scaler_X.pkl', 'wb') as f:
    pickle.dump(scaler_X, f)
with open(scaler_dir / 'scaler_y.pkl', 'wb') as f:
    pickle.dump(scaler_y, f)

print(f"[OK] Saved scalers to {scaler_dir}")

# ============================================================================
#                          PYTORCH DATASET
# ============================================================================

class F1Dataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Create dataset
dataset = F1Dataset(X_scaled, y_scaled)

# Split: 70% train, 15% val, 15% test
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    dataset, [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42)
)

print(f"\nDataset split:")
print(f"  Train: {len(train_dataset):,} samples (70%)")
print(f"  Val:   {len(val_dataset):,} samples (15%)")
print(f"  Test:  {len(test_dataset):,} samples (15%)")

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"[OK] Created data loaders")

# ============================================================================
#                          MODEL ARCHITECTURE
# ============================================================================

class F1LapTimePredictor(nn.Module):
    def __init__(self, input_size=25):
        super(F1LapTimePredictor, self).__init__()

        self.network = nn.Sequential(
            # Layer 1: 25 -> 128
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),

            # Layer 2: 128 -> 64
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),

            # Layer 3: 64 -> 32
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.1),

            # Layer 4: 32 -> 16
            nn.Linear(32, 16),
            nn.ReLU(),

            # Output: 16 -> 1
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.network(x)

# Initialize model
model = F1LapTimePredictor(input_size=len(feature_names)).to(device)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"\n{'='*80}")
print("MODEL ARCHITECTURE")
print(f"{'='*80}")
print(model)
print(f"\nTotal parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# ============================================================================
#                          TRAINING SETUP
# ============================================================================

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=LR_REDUCE_PATIENCE, verbose=True
)

print(f"\nLoss Function: MSE Loss")
print(f"Optimizer: Adam (lr={LEARNING_RATE})")
print(f"LR Scheduler: ReduceLROnPlateau (patience={LR_REDUCE_PATIENCE})")

# ============================================================================
#                          TRAINING LOOP
# ============================================================================

print(f"\n{'='*80}")
print("TRAINING")
print(f"{'='*80}\n")

train_losses = []
val_losses = []
best_val_loss = float('inf')
patience_counter = 0
start_time = time.time()

for epoch in range(EPOCHS):
    # Training phase
    model.train()
    train_loss = 0.0

    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        # Forward pass
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * batch_X.size(0)

    train_loss /= len(train_loader.dataset)
    train_losses.append(train_loss)

    # Validation phase
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            val_loss += loss.item() * batch_X.size(0)

    val_loss /= len(val_loader.dataset)
    val_losses.append(val_loss)

    # Learning rate scheduling
    scheduler.step(val_loss)

    # Convert scaled loss to actual RMSE (in seconds)
    train_rmse_scaled = np.sqrt(train_loss)
    val_rmse_scaled = np.sqrt(val_loss)

    # Approximate unscaled RMSE (rough estimate)
    # This is approximate because we're in scaled space
    approx_rmse = val_rmse_scaled * scaler_y.scale_[0]

    # Print progress
    print(f"Epoch [{epoch+1:3d}/{EPOCHS}] | "
          f"Train Loss: {train_loss:.6f} | "
          f"Val Loss: {val_loss:.6f} | "
          f"Val RMSE: ~{approx_rmse:.2f}s")

    # Early stopping check
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0

        # Save best model
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'feature_names': feature_names,
        }, scaler_dir / 'best_model.pth')

        print(f"         → New best model saved! (Val Loss: {val_loss:.6f})")
    else:
        patience_counter += 1
        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break

training_time = time.time() - start_time

print(f"\n{'='*80}")
print("TRAINING COMPLETE")
print(f"{'='*80}")
print(f"Training time: {training_time/60:.1f} minutes")
print(f"Best validation loss: {best_val_loss:.6f}")

# ============================================================================
#                          FINAL EVALUATION
# ============================================================================

print(f"\n{'='*80}")
print("FINAL EVALUATION ON TEST SET")
print(f"{'='*80}")

# Load best model
checkpoint = torch.load(scaler_dir / 'best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Evaluate on test set
test_predictions = []
test_targets = []

with torch.no_grad():
    for batch_X, batch_y in test_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        outputs = model(batch_X)

        test_predictions.append(outputs.cpu().numpy())
        test_targets.append(batch_y.cpu().numpy())

test_predictions = np.vstack(test_predictions)
test_targets = np.vstack(test_targets)

# Unscale predictions and targets
test_predictions_unscaled = scaler_y.inverse_transform(test_predictions)
test_targets_unscaled = scaler_y.inverse_transform(test_targets)

# Calculate metrics
mse = np.mean((test_predictions_unscaled - test_targets_unscaled) ** 2)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(test_predictions_unscaled - test_targets_unscaled))
max_error = np.max(np.abs(test_predictions_unscaled - test_targets_unscaled))

# R² score
ss_res = np.sum((test_targets_unscaled - test_predictions_unscaled) ** 2)
ss_tot = np.sum((test_targets_unscaled - np.mean(test_targets_unscaled)) ** 2)
r2 = 1 - (ss_res / ss_tot)

print(f"\nTest Set Performance:")
print(f"  RMSE:      {rmse:.4f} seconds")
print(f"  MAE:       {mae:.4f} seconds")
print(f"  Max Error: {max_error:.4f} seconds")
print(f"  R² Score:  {r2:.4f}")

if rmse < 2.0:
    print(f"\n  *** EXCELLENT! RMSE < 2.0s ***")
elif rmse < 3.0:
    print(f"\n  *** VERY GOOD! RMSE < 3.0s ***")
elif rmse < 5.0:
    print(f"\n  *** GOOD! RMSE < 5.0s ***")
else:
    print(f"\n  Model performance needs improvement")

# Save test predictions
results_df = pd.DataFrame({
    'actual': test_targets_unscaled.flatten(),
    'predicted': test_predictions_unscaled.flatten(),
    'error': (test_predictions_unscaled - test_targets_unscaled).flatten()
})

results_file = Path('./outputs/test_predictions.csv')
results_file.parent.mkdir(exist_ok=True)
results_df.to_csv(results_file, index=False)
print(f"\n[OK] Saved test predictions: {results_file}")

# ============================================================================
#                          VISUALIZATIONS
# ============================================================================

print(f"\n{'='*80}")
print("CREATING VISUALIZATIONS")
print(f"{'='*80}")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Training curves
ax1 = axes[0, 0]
epochs_range = range(1, len(train_losses) + 1)
ax1.plot(epochs_range, train_losses, 'b-', label='Train Loss', linewidth=2)
ax1.plot(epochs_range, val_losses, 'r-', label='Val Loss', linewidth=2)
ax1.set_xlabel('Epoch', fontweight='bold')
ax1.set_ylabel('Loss (MSE)', fontweight='bold')
ax1.set_title('Training and Validation Loss', fontweight='bold', fontsize=13)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Actual vs Predicted
ax2 = axes[0, 1]
sample_size = min(5000, len(test_targets_unscaled))
indices = np.random.choice(len(test_targets_unscaled), sample_size, replace=False)
ax2.scatter(test_targets_unscaled[indices], test_predictions_unscaled[indices],
           alpha=0.3, s=10, c='blue')
ax2.plot([test_targets_unscaled.min(), test_targets_unscaled.max()],
         [test_targets_unscaled.min(), test_targets_unscaled.max()],
         'r--', linewidth=2, label='Perfect prediction')
ax2.set_xlabel('Actual Lap Time (s)', fontweight='bold')
ax2.set_ylabel('Predicted Lap Time (s)', fontweight='bold')
ax2.set_title(f'Actual vs Predicted (RMSE={rmse:.2f}s)', fontweight='bold', fontsize=13)
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Error distribution
ax3 = axes[1, 0]
errors = (test_predictions_unscaled - test_targets_unscaled).flatten()
ax3.hist(errors, bins=50, alpha=0.7, edgecolor='black', color='skyblue')
ax3.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero error')
ax3.axvline(rmse, color='orange', linestyle=':', linewidth=2, label=f'RMSE ({rmse:.2f}s)')
ax3.axvline(-rmse, color='orange', linestyle=':', linewidth=2)
ax3.set_xlabel('Prediction Error (s)', fontweight='bold')
ax3.set_ylabel('Frequency', fontweight='bold')
ax3.set_title('Error Distribution', fontweight='bold', fontsize=13)
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')

# Plot 4: Residuals
ax4 = axes[1, 1]
ax4.scatter(test_predictions_unscaled[indices], errors[indices],
           alpha=0.3, s=10, c='blue')
ax4.axhline(0, color='red', linestyle='--', linewidth=2)
ax4.axhline(3, color='green', linestyle=':', linewidth=1, alpha=0.5)
ax4.axhline(-3, color='green', linestyle=':', linewidth=1, alpha=0.5)
ax4.set_xlabel('Predicted Lap Time (s)', fontweight='bold')
ax4.set_ylabel('Residual Error (s)', fontweight='bold')
ax4.set_title('Residual Plot', fontweight='bold', fontsize=13)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plot_file = Path('./outputs/training_results.png')
plt.savefig(plot_file, dpi=300, bbox_inches='tight')
print(f"[OK] Saved visualization: {plot_file}")

# Save training history
history = {
    'train_losses': train_losses,
    'val_losses': val_losses,
    'test_rmse': rmse,
    'test_mae': mae,
    'test_r2': r2,
    'training_time': training_time,
    'total_epochs': len(train_losses)
}

history_file = Path('./outputs/training_history.pkl')
with open(history_file, 'wb') as f:
    pickle.dump(history, f)
print(f"[OK] Saved training history: {history_file}")

print(f"\n{'='*80}")
print("ALL DONE!")
print(f"{'='*80}")
print(f"\nModel saved to: {scaler_dir / 'best_model.pth'}")
print(f"Scalers saved to: {scaler_dir}")
print(f"Results saved to: ./outputs/")
print(f"\nFinal Test RMSE: {rmse:.4f} seconds")
print(f"\nCompare to mathematical model RMSE: 4.18 seconds")
if rmse < 4.18:
    improvement = ((4.18 - rmse) / 4.18) * 100
    print(f"Improvement: {improvement:.1f}% better!")

plt.show()
