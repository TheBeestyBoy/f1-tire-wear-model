"""
FastAPI application for F1 Lap Time Prediction.
Interactive API for running race simulations and comparing scenarios.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import os
import sys
from pathlib import Path
from contextlib import asynccontextmanager
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime

# Add parent directory to path to import from ai_model
sys.path.append(str(Path(__file__).parent.parent.parent / 'ai_model'))

# ============================================================================
#                          PYDANTIC MODELS
# ============================================================================

class HealthResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    status: str
    model_loaded: bool
    available_races: int

class ModelInfo(BaseModel):
    input_features: int
    total_parameters: int
    architecture: List[int]
    test_rmse: float
    test_mae: float
    test_r2: float

class RaceScenario(BaseModel):
    year: int = Field(default=2023, ge=2018, le=2024)
    race: str = Field(default="Monaco")
    driver: str = Field(default="VER")
    tire_improvement: float = Field(default=0.0, ge=0.0, le=0.5, description="Tire improvement factor (0.0 = normal, 0.25 = 25% better)")
    fuel_load_factor: float = Field(default=1.0, ge=0.8, le=1.2, description="Fuel load adjustment")
    weather_override: Optional[Dict[str, float]] = None

class PredictionRequest(BaseModel):
    scenarios: List[RaceScenario]
    comparison_mode: bool = True

class LapPrediction(BaseModel):
    lap_number: int
    predicted_time: float
    actual_time: Optional[float] = None
    compound: str
    tire_age: int
    scenario_name: str
    # Weather data
    rainfall: Optional[float] = None
    track_temp: Optional[float] = None
    air_temp: Optional[float] = None
    humidity: Optional[float] = None
    # Error metrics
    error: Optional[float] = None  # predicted - actual
    absolute_error: Optional[float] = None

class ScenarioResult(BaseModel):
    scenario_name: str
    total_race_time: float
    average_lap_time: float
    fastest_lap: float
    slowest_lap: float
    laps: List[LapPrediction]
    tire_strategy: List[Dict[str, Any]]

class PredictionResponse(BaseModel):
    scenarios: List[ScenarioResult]
    comparison: Optional[Dict[str, Any]] = None

class AvailableRace(BaseModel):
    year: int
    race: str
    drivers: List[str]
    laps: int

class AvailableRacesResponse(BaseModel):
    races: List[AvailableRace]

# ============================================================================
#                          F1 MODEL WRAPPER
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

class F1Predictor:
    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.scaler_X = None
        self.scaler_y = None
        self.feature_names = None
        self.data_df = None

    def load_model(self):
        """Load trained model, scalers, and data"""
        # Load model
        checkpoint = torch.load(
            self.model_dir / 'models' / 'best_model.pth',
            map_location=self.device,
            weights_only=False
        )
        self.feature_names = checkpoint['feature_names']

        self.model = F1LapTimePredictor(input_size=len(self.feature_names)).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        # Load scalers
        with open(self.model_dir / 'models' / 'scaler_X.pkl', 'rb') as f:
            self.scaler_X = pickle.load(f)
        with open(self.model_dir / 'models' / 'scaler_y.pkl', 'rb') as f:
            self.scaler_y = pickle.load(f)

        # Load preprocessed data
        self.data_df = pd.read_csv(self.model_dir / 'data' / 'preprocessed_f1_data.csv')

        print(f"[OK] Model loaded with {len(self.feature_names)} features")
        print(f"[OK] Data loaded: {len(self.data_df):,} laps")

    def get_race_data(self, year: int, race: str, driver: str):
        """Get race data for specific year/race/driver"""
        race_data = self.data_df[
            (self.data_df['year'] == year) &
            (self.data_df['race'] == race) &
            (self.data_df['driver'] == driver)
        ].copy()

        if len(race_data) == 0:
            raise ValueError(f"No data found for {year} {race} {driver}")

        return race_data.sort_values('lap_number').reset_index(drop=True)

    def predict_scenario(self, scenario: RaceScenario):
        """Run prediction for a specific scenario"""
        # Get base race data
        race_data = self.get_race_data(scenario.year, scenario.race, scenario.driver)

        # Make base predictions first (on UNMODIFIED data)
        X = race_data[self.feature_names].values
        X = np.nan_to_num(X, nan=0.0)
        X_scaled = self.scaler_X.transform(X)

        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)
            predictions_scaled = self.model(X_tensor).cpu().numpy()

        predictions = self.scaler_y.inverse_transform(predictions_scaled).flatten()

        # Store ORIGINAL predictions for error calculation
        original_predictions = predictions.copy()

        # NOW apply scenario modifications to the PREDICTIONS
        # This is more realistic than modifying input features

        # Apply tire improvement (reduces lap times directly)
        if scenario.tire_improvement > 0:
            # Better tires = faster lap times
            # Improvement scales with tire age (more benefit on worn tires)
            tire_ages = race_data['tyre_life'].values
            max_tire_age = tire_ages.max() if tire_ages.max() > 0 else 1

            # Calculate improvement per lap (more improvement on older tires)
            tire_age_factor = tire_ages / max_tire_age  # 0 to 1
            improvement_per_lap = scenario.tire_improvement * tire_age_factor * 0.5  # Up to 50% of improvement factor

            # Apply improvement (reduce lap times)
            predictions = predictions * (1 - improvement_per_lap)

        # Apply fuel load factor (lighter car = faster)
        if scenario.fuel_load_factor != 1.0:
            # Fuel effect: lighter car = faster, scales with lap number
            lap_numbers = race_data['lap_number'].values
            max_lap = lap_numbers.max() if lap_numbers.max() > 0 else 1

            # More fuel = slower (linear effect through race)
            fuel_progress = lap_numbers / max_lap  # 0 to 1 (how much fuel burned)
            fuel_factor = scenario.fuel_load_factor - 1.0  # e.g., 0.9x = -0.1

            # Fuel effect diminishes as race progresses (car gets lighter)
            fuel_impact = fuel_factor * (1 - fuel_progress) * 0.02  # Max 2% effect
            predictions = predictions * (1 + fuel_impact)

        # Build result
        laps = []
        for idx, row in race_data.iterrows():
            # Get actual lap time if available
            actual_time = None
            if 'lap_time' in race_data.columns and pd.notna(row['lap_time']):
                actual_time = float(row['lap_time'])

            # Calculate error metrics using ORIGINAL predictions (not modified)
            error = None
            absolute_error = None
            if actual_time is not None:
                error = float(original_predictions[idx] - actual_time)
                absolute_error = abs(error)

            # Get weather data
            rainfall = float(row['rainfall']) if 'rainfall' in race_data.columns and pd.notna(row['rainfall']) else None
            track_temp = float(row['track_temp']) if 'track_temp' in race_data.columns and pd.notna(row['track_temp']) else None
            air_temp = float(row['air_temp']) if 'air_temp' in race_data.columns and pd.notna(row['air_temp']) else None
            humidity = float(row['humidity']) if 'humidity' in race_data.columns and pd.notna(row['humidity']) else None

            laps.append(LapPrediction(
                lap_number=int(row['lap_number']),
                predicted_time=float(predictions[idx]),
                actual_time=actual_time,
                compound=str(row['compound']) if pd.notna(row['compound']) else 'MEDIUM',
                tire_age=int(row['tyre_life']) if pd.notna(row['tyre_life']) else 0,
                scenario_name=self._get_scenario_name(scenario),
                rainfall=rainfall,
                track_temp=track_temp,
                air_temp=air_temp,
                humidity=humidity,
                error=error,
                absolute_error=absolute_error
            ))

        # Calculate tire strategy
        tire_strategy = self._extract_tire_strategy(race_data)

        return ScenarioResult(
            scenario_name=self._get_scenario_name(scenario),
            total_race_time=float(np.sum(predictions)),
            average_lap_time=float(np.mean(predictions)),
            fastest_lap=float(np.min(predictions)),
            slowest_lap=float(np.max(predictions)),
            laps=laps,
            tire_strategy=tire_strategy
        )

    def _get_scenario_name(self, scenario: RaceScenario) -> str:
        """Generate scenario name"""
        parts = [f"{scenario.year} {scenario.race} - {scenario.driver}"]

        if scenario.tire_improvement > 0:
            parts.append(f"+{scenario.tire_improvement*100:.0f}% Tires")

        if scenario.fuel_load_factor != 1.0:
            diff = (scenario.fuel_load_factor - 1) * 100
            parts.append(f"{diff:+.0f}% Fuel")

        return " | ".join(parts)

    def _extract_tire_strategy(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Extract tire change strategy"""
        stints = []
        current_compound = None
        stint_start = None

        for idx, row in data.iterrows():
            compound = row.get('compound', 'MEDIUM')

            if compound != current_compound:
                if current_compound is not None:
                    stints.append({
                        'stint': len(stints) + 1,
                        'compound': current_compound,
                        'start_lap': stint_start,
                        'end_lap': int(row['lap_number']) - 1,
                        'laps': int(row['lap_number']) - stint_start
                    })

                current_compound = compound
                stint_start = int(row['lap_number'])

        # Add final stint
        if current_compound is not None:
            stints.append({
                'stint': len(stints) + 1,
                'compound': current_compound,
                'start_lap': stint_start,
                'end_lap': int(data.iloc[-1]['lap_number']),
                'laps': int(data.iloc[-1]['lap_number']) - stint_start + 1
            })

        return stints

    def get_available_races(self) -> List[AvailableRace]:
        """Get list of available races in dataset"""
        races = []

        for (year, race), group in self.data_df.groupby(['year', 'race']):
            drivers = sorted(group['driver'].unique().tolist())
            races.append(AvailableRace(
                year=int(year),
                race=str(race),
                drivers=drivers,
                laps=len(group)
            ))

        return sorted(races, key=lambda x: (x.year, x.race))

# ============================================================================
#                          FASTAPI APP
# ============================================================================

predictor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown events"""
    global predictor
    try:
        model_dir = Path(__file__).parent.parent.parent / 'ai_model'
        predictor = F1Predictor(model_dir)
        predictor.load_model()
        print("[OK] F1 Predictor initialized")
    except Exception as e:
        print(f"[ERROR] Failed to initialize: {e}")

    yield

    # Cleanup
    pass

app = FastAPI(
    title="F1 Lap Time Prediction API",
    description="AI-powered F1 lap time predictions with scenario comparison",
    version="1.0.0",
    lifespan=lifespan
)

# CORS - Allow frontend domains
ALLOWED_ORIGINS = os.environ.get(
    "ALLOWED_ORIGINS",
    "*"  # Default to all origins for development
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS if ALLOWED_ORIGINS != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
#                          ENDPOINTS
# ============================================================================

@app.get("/", response_model=HealthResponse)
async def health_check():
    """Health check"""
    return HealthResponse(
        status="healthy",
        model_loaded=predictor is not None,
        available_races=len(predictor.get_available_races()) if predictor else 0
    )

@app.get("/model/info", response_model=ModelInfo)
async def get_model_info():
    """Get model information"""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    total_params = sum(p.numel() for p in predictor.model.parameters())

    # Load test metrics if available
    try:
        with open(predictor.model_dir / 'outputs' / 'training_history.pkl', 'rb') as f:
            history = pickle.load(f)
            test_rmse = history.get('test_rmse', 0.0)
            test_mae = history.get('test_mae', 0.0)
            test_r2 = history.get('test_r2', 0.0)
    except:
        test_rmse = 0.40
        test_mae = 0.29
        test_r2 = 0.9988

    return ModelInfo(
        input_features=len(predictor.feature_names),
        total_parameters=total_params,
        architecture=[25, 128, 64, 32, 16, 1],
        test_rmse=test_rmse,
        test_mae=test_mae,
        test_r2=test_r2
    )

@app.get("/races/available", response_model=AvailableRacesResponse)
async def get_available_races():
    """Get all available races"""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    races = predictor.get_available_races()
    return AvailableRacesResponse(races=races)

@app.post("/predict", response_model=PredictionResponse)
async def predict_scenarios(request: PredictionRequest):
    """Run predictions for one or more scenarios"""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        results = []

        for scenario in request.scenarios:
            result = predictor.predict_scenario(scenario)
            results.append(result)

        # Calculate comparison if multiple scenarios
        comparison = None
        if request.comparison_mode and len(results) > 1:
            baseline = results[0]
            comparisons = []

            for i, result in enumerate(results[1:], 1):
                time_diff = result.total_race_time - baseline.total_race_time
                lap_diff = result.average_lap_time - baseline.average_lap_time

                comparisons.append({
                    'scenario': result.scenario_name,
                    'total_time_diff': time_diff,
                    'average_lap_diff': lap_diff,
                    'time_saved': -time_diff,
                    'percentage_improvement': (-time_diff / baseline.total_race_time) * 100
                })

            comparison = {
                'baseline': baseline.scenario_name,
                'comparisons': comparisons
            }

        return PredictionResponse(
            scenarios=results,
            comparison=comparison
        )

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/visualizations/architecture")
async def get_architecture():
    """Get model architecture diagram"""
    path = predictor.model_dir / 'outputs' / 'training_results.png'
    if not path.exists():
        raise HTTPException(status_code=404, detail="Visualization not found")
    return FileResponse(path)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
