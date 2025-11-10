"""
Game Mode Simulator - User-configurable F1 lap time predictions

Allows users to specify environmental parameters and get simulated lap times.
This module bridges the AI model with user-friendly configuration options.
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import json
from typing import Dict, List, Tuple, Optional
import sys

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
#                          GAME SIMULATOR CLASS
# ============================================================================

class F1GameSimulator:
    """
    Game Mode simulator for F1 lap time predictions.

    Allows user-configured scenarios with:
    - Weather conditions (temperature, rainfall)
    - Tire parameters (compound, age)
    - Fuel load
    - Lap number
    """

    def __init__(self, model_dir='./models'):
        self.model_dir = Path(model_dir)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load model
        checkpoint = torch.load(
            self.model_dir / 'best_model.pth',
            map_location=self.device,
            weights_only=False
        )
        self.feature_names = checkpoint['feature_names']

        self.model = F1LapTimePredictor(input_size=len(self.feature_names)).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        # Load scalers
        with open(self.model_dir / 'scaler_X.pkl', 'rb') as f:
            self.scaler_X = pickle.load(f)
        with open(self.model_dir / 'scaler_y.pkl', 'rb') as f:
            self.scaler_y = pickle.load(f)

        # Load track data
        track_data_file = Path(model_dir).parent / 'outputs' / 'track_data.json'
        if track_data_file.exists():
            with open(track_data_file, 'r') as f:
                self.track_data = json.load(f)
        else:
            self.track_data = []

        # Compound mapping
        self.compound_map = {
            'SOFT': 0,
            'MEDIUM': 1,
            'HARD': 2,
            'INTERMEDIATE': 3,
            'WET': 4
        }

        # Weather presets
        self.weather_presets = {
            'DRY': {'rainfall': 0.0, 'humidity': 45.0, 'track_temp': 30.0, 'air_temp': 25.0},
            'DRIZZLE': {'rainfall': 2.0, 'humidity': 70.0, 'track_temp': 22.0, 'air_temp': 20.0},
            'RAIN': {'rainfall': 10.0, 'humidity': 85.0, 'track_temp': 18.0, 'air_temp': 16.0},
            'HEAVY_RAIN': {'rainfall': 20.0, 'humidity': 95.0, 'track_temp': 15.0, 'air_temp': 14.0},
            'HOT': {'rainfall': 0.0, 'humidity': 30.0, 'track_temp': 45.0, 'air_temp': 35.0},
            'COLD': {'rainfall': 0.0, 'humidity': 60.0, 'track_temp': 15.0, 'air_temp': 12.0},
        }

    def get_tracks(self) -> List[Dict]:
        """
        Get list of all available tracks with their statistics.

        Returns:
            List of track dictionaries with name, avg_lap_time, rmse, etc.
        """
        return self.track_data

    def _get_track_baseline(self, track_name: Optional[str]) -> float:
        """
        Get baseline lap time for a specific track.

        Args:
            track_name: Name of the track (e.g., 'Monaco Grand Prix')

        Returns:
            Baseline lap time in seconds (defaults to 75.0 for Monaco if not found)
        """
        if not track_name or not self.track_data:
            return 75.0  # Default Monaco baseline

        # Find track in data
        for track in self.track_data:
            if track['name'] == track_name:
                return float(track['avg_lap_time'])

        # Track not found, return default
        return 75.0

    def _build_features(self, config: Dict) -> np.ndarray:
        """
        Build feature vector from user configuration.

        Args:
            config: Dictionary with user parameters

        Returns:
            Feature array (1, 25)
        """
        # Get track-specific baseline lap time
        track_name = config.get('track', None)
        baseline_lap_time = self._get_track_baseline(track_name)

        # Base values
        features = {
            'prev_lap_time': baseline_lap_time,
            'lap_number': config.get('lap_number', 30),
            'tyre_life': config.get('tire_age', 15),
            'compound_encoded': self.compound_map[config.get('compound', 'MEDIUM')],
            'lap_in_stint': config.get('tire_age', 15),

            # Speed metrics
            'speed_mean': 150.0,
            'speed_max': 280.0,
            'speed_std': 45.0,

            # Driving metrics
            'throttle_mean': 65.0,
            'brake_mean': 25.0,
            'rpm_mean': 11000.0,
            'n_gear_mean': 4.5,
            'drs_usage': 0.15,
            'n_braking_zones': 15,
            'brake_intensity_mean': 3.5,
            'cornering_intensity': 2.8,

            # Weather
            'track_temp': config.get('track_temp', 30.0),
            'air_temp': config.get('air_temp', 25.0),
            'humidity': config.get('humidity', 50.0),
            'pressure': config.get('pressure', 1013.0),
            'rainfall': config.get('rainfall', 0.0),
            'wind_speed': config.get('wind_speed', 2.5),

            # Sector times
            'sector1_time': 25.0,
            'sector2_time': 25.0,
            'sector3_time': 25.0,
        }

        # Apply physical modifiers based on conditions
        rainfall = features['rainfall']
        tire_age = features['tyre_life']
        track_temp = features['track_temp']

        # Wet conditions reduce speed
        if rainfall > 0:
            wet_factor = min(0.85, 1.0 - (rainfall / 50))
            features['speed_mean'] *= wet_factor
            features['speed_max'] *= (wet_factor + 0.05)
            features['throttle_mean'] *= (wet_factor - 0.05)
            features['brake_mean'] *= 1.2

        # Old tires degrade performance
        if tire_age > 20:
            age_factor = 1.0 - ((tire_age - 20) * 0.001)
            features['speed_mean'] *= age_factor
            features['throttle_mean'] *= age_factor

        # Temperature effects
        if track_temp < 15:
            features['speed_mean'] *= 0.97
        elif track_temp > 40:
            features['speed_mean'] *= 0.99

        # Adjust prev_lap_time for consistency
        lap_time_modifier = 1.0
        if rainfall > 5:
            lap_time_modifier += min(0.3, rainfall / 50)
        if tire_age > 30:
            lap_time_modifier += (tire_age - 30) * 0.001
        if track_temp < 15:
            lap_time_modifier += 0.03

        features['prev_lap_time'] *= lap_time_modifier

        # Fuel load effect (optional)
        if 'fuel_load' in config:
            fuel_factor = config['fuel_load']  # 0.0 to 1.0
            features['speed_mean'] *= (0.98 + 0.02 * (1 - fuel_factor))

        # Convert to array in correct order
        X = np.array([features[name] for name in self.feature_names]).reshape(1, -1)
        return X

    def simulate_lap(self, **config) -> float:
        """
        Simulate a single lap with given configuration.

        Args:
            **config: User configuration parameters
                - track: Track name (e.g., 'Monaco Grand Prix') (default: Monaco baseline)
                - compound: 'SOFT', 'MEDIUM', 'HARD', 'INTERMEDIATE', 'WET'
                - tire_age: Tire age in laps (default: 15)
                - track_temp: Track temperature in C (default: 30)
                - air_temp: Air temperature in C (default: 25)
                - rainfall: Rainfall in mm (default: 0)
                - humidity: Humidity percentage (default: 50)
                - lap_number: Current lap number (default: 30)
                - fuel_load: Fuel load 0.0-1.0 (default: 0.5)

        Returns:
            Predicted lap time in seconds
        """
        # Build features
        X = self._build_features(config)

        # Scale features
        X_scaled = self.scaler_X.transform(X)

        # Predict
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)
            prediction_scaled = self.model(X_tensor).cpu().numpy()

        # Unscale prediction
        prediction = self.scaler_y.inverse_transform(prediction_scaled)[0, 0]

        # Add slight random noise for simulation variability
        if config.get('add_noise', False):
            noise = np.random.normal(0, 0.2)  # +/- 0.2s standard deviation
            prediction += noise

        return float(prediction)

    def simulate_stint(self, laps: int, **config) -> List[Tuple[int, float]]:
        """
        Simulate a tire stint (multiple laps with degrading tires).

        Args:
            laps: Number of laps to simulate
            **config: User configuration (same as simulate_lap)

        Returns:
            List of (lap_number, lap_time) tuples
        """
        results = []
        start_lap = config.get('lap_number', 1)
        start_tire_age = config.get('tire_age', 0)

        for i in range(laps):
            lap_config = config.copy()
            lap_config['lap_number'] = start_lap + i
            lap_config['tire_age'] = start_tire_age + i

            # Fuel load decreases over stint
            if 'fuel_load' in lap_config:
                fuel_progress = i / max(laps, 1)
                lap_config['fuel_load'] = max(0.0, lap_config['fuel_load'] * (1 - fuel_progress * 0.5))

            lap_time = self.simulate_lap(**lap_config)
            results.append((start_lap + i, lap_time))

        return results

    def simulate_with_weather_preset(self, weather: str, **other_config) -> float:
        """
        Simulate lap with a weather preset.

        Args:
            weather: Weather preset ('DRY', 'DRIZZLE', 'RAIN', 'HEAVY_RAIN', 'HOT', 'COLD')
            **other_config: Additional configuration parameters

        Returns:
            Predicted lap time in seconds
        """
        if weather not in self.weather_presets:
            raise ValueError(f"Unknown weather preset: {weather}. Available: {list(self.weather_presets.keys())}")

        config = self.weather_presets[weather].copy()
        config.update(other_config)

        return self.simulate_lap(**config)

# ============================================================================
#                          EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("F1 GAME MODE SIMULATOR - Example Usage")
    print("="*80)

    # Initialize simulator
    simulator = F1GameSimulator()
    print("\n[OK] Simulator initialized")

    # Example 1: Simple lap simulation
    print("\n" + "="*80)
    print("EXAMPLE 1: Simple Lap Simulation")
    print("="*80)

    lap_time = simulator.simulate_lap(
        compound='MEDIUM',
        tire_age=10,
        track_temp=30.0,
        rainfall=0.0
    )

    print(f"\nScenario: MEDIUM tires (10 laps old), dry conditions")
    print(f"Predicted Lap Time: {lap_time:.2f}s")

    # Example 2: Weather presets
    print("\n" + "="*80)
    print("EXAMPLE 2: Weather Presets")
    print("="*80)

    for weather in ['DRY', 'DRIZZLE', 'RAIN']:
        lap_time = simulator.simulate_with_weather_preset(
            weather=weather,
            compound='INTERMEDIATE' if weather != 'DRY' else 'SOFT',
            tire_age=5
        )
        print(f"\n{weather:12s}: {lap_time:.2f}s")

    # Example 3: Stint simulation
    print("\n" + "="*80)
    print("EXAMPLE 3: Tire Stint Simulation (20 laps)")
    print("="*80)

    stint_results = simulator.simulate_stint(
        laps=20,
        compound='SOFT',
        tire_age=0,
        lap_number=1,
        track_temp=28.0,
        rainfall=0.0,
        fuel_load=1.0
    )

    print(f"\n{'Lap':>4s} | {'Time':>8s} | {'Degradation'}")
    print("-" * 35)
    baseline = stint_results[0][1]
    for lap_num, lap_time in stint_results[::5]:  # Every 5 laps
        deg = lap_time - baseline
        print(f"{lap_num:4d} | {lap_time:8.2f}s | {deg:+.2f}s")

    # Example 4: Compare compounds
    print("\n" + "="*80)
    print("EXAMPLE 4: Compound Comparison")
    print("="*80)

    print(f"\n{'Compound':15s} | {'Fresh (1 lap)':>14s} | {'Worn (30 laps)':>15s} | {'Degradation'}")
    print("-" * 70)

    for compound in ['SOFT', 'MEDIUM', 'HARD']:
        fresh = simulator.simulate_lap(compound=compound, tire_age=1)
        worn = simulator.simulate_lap(compound=compound, tire_age=30)
        deg = worn - fresh
        print(f"{compound:15s} | {fresh:14.2f}s | {worn:15.2f}s | {deg:+.2f}s")

    print("\n" + "="*80)
    print("GAME SIMULATOR READY!")
    print("="*80)
    print("\nUsage:")
    print("  from game_simulator import F1GameSimulator")
    print("  simulator = F1GameSimulator()")
    print("  lap_time = simulator.simulate_lap(compound='SOFT', tire_age=10)")
    print("  stint = simulator.simulate_stint(laps=20, compound='MEDIUM')")
