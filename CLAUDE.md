# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

F1 Tire Wear Model - A comprehensive F1 lap time prediction system with THREE parallel implementations:

1. **Physics-Based Mathematical Model** - Formula-driven tire degradation (RMSE ~4.18s on Monaco 2023)
2. **PyTorch AI Model** - Neural network trained on 126K laps from 2018-2024 (RMSE ~0.40s)
3. **Dynamic Systems Model** - Discrete-time race simulator with strategy optimization

All three models use FastF1 telemetry data and support scenario analysis, but serve different purposes and have different accuracy characteristics.

## Development Commands

### Physics-Based Model (Mathematical Formulas)
```bash
python predict_lap_times.py
```
- Uses calibrated parameters from `quick_optimal_params.csv`
- Edit YEAR, RACE, SESSION, DRIVER variables to analyze different sessions
- Best for: Understanding tire degradation physics, track-specific calibration
- Output: CSV + 4-panel PNG visualization

```bash
python predict_FINAL.py
```
- Alternative with 6-panel visualization
- Same calibrated parameters (RMSE = 4.18s)

### Dynamic Systems Model (Race Strategy Simulator)
```bash
python dynamic_race_simulator.py
```
- Simulates 8 predefined pit strategies for 70-lap race
- Models state evolution: tire age, fuel load, track wetness, tire wear
- Best for: What-if scenario analysis, strategy comparison
- Output: 4 PNG visualizations in `dynamic_simulation_results/`

```bash
python strategy_optimizer.py
```
- Exhaustive search over 3,220+ feasible strategies
- Finds globally optimal pit strategy
- Includes sensitivity analysis
- Output: CSV files with top 20 strategies + optimal lap-by-lap data

### AI Model (PyTorch Neural Network)

**Train model from scratch:**
```bash
cd ai_model
python run_all.py
```
- Downloads all F1 data (2018-2024, ~126K laps)
- Preprocesses features (25 input features)
- Trains PyTorch model (128→64→32→16→1 architecture)
- Takes ~30-60 minutes depending on hardware
- Output: `models/best_model.pth`, `outputs/training_results.png`

**Individual AI workflow steps:**
```bash
# 1. Download data only
python download_all_f1_data.py

# 2. Preprocess data
python analyze_data.py

# 3. Train model
python train_model.py

# 4. Run predictions
python predict.py
```

### Full-Stack Web Application

**Backend (FastAPI + PyTorch):**
```bash
cd backend/app
python f1_main.py
```
- Serves PyTorch AI model at http://localhost:8000
- Supports scenario comparison via REST API
- Requires trained model in `ai_model/models/best_model.pth`

**Frontend (React + Material-UI):**
```bash
cd frontend
npm install
npm start
```
- Interactive UI at http://localhost:3000
- Configure race scenarios, compare strategies
- Real-time predictions with visualization

### Dependencies
```bash
# Physics and Dynamic Models
pip install fastf1 pandas numpy matplotlib seaborn

# AI Model (additional)
pip install torch scikit-learn tqdm

# Backend API (additional)
pip install fastapi uvicorn python-multipart pydantic

# Frontend
cd frontend && npm install
```

## Architecture

### Three Parallel Model Architectures

**1. Physics-Based Model** (`predict_lap_times.py`, `predict_FINAL.py`)
- Data Flow: FastF1 telemetry → Track wetness estimation → Tire performance factors → Lap time predictions
- State: Wetness per lap, tire age per stint, baseline lap time per stint
- Key Innovation: INTERMEDIATE tires improve as track dries toward optimal wetness (0.21)
- Use Case: Transparent, formula-driven predictions for specific races

**2. Dynamic Systems Model** (`dynamic_race_simulator.py`, `strategy_optimizer.py`)
- Data Flow: Initial conditions → Lap-by-lap state transitions → Strategy outcomes
- State Variables: `[tire_age, fuel_remaining, cumulative_wear, track_wetness]`
- Control Inputs: `[pit_decision, compound_choice]`
- Discrete-time simulation: `x(t+1) = f(x(t), u(t))`
- Use Case: Race strategy optimization, what-if scenario analysis

**3. AI Model** (`ai_model/` directory)
- Data Flow: Raw telemetry (126K laps) → Feature engineering (25 features) → PyTorch NN → Lap time prediction
- Architecture: Input(25) → Dense(128) → Dense(64) → Dense(32) → Dense(16) → Output(1)
- Training: Adam optimizer, MSE loss, 80/10/10 train/val/test split
- Use Case: Maximum accuracy predictions, web application backend

### Full-Stack Web Architecture

```
User Browser (React + MUI)
    ↓ HTTP POST /predict
FastAPI Backend (f1_main.py)
    ↓ torch.load()
PyTorch Model (best_model.pth)
    ↓ predictions
Preprocessed Data (preprocessed_f1_data.csv)
```

**Key Components:**
- `F1LapTimePredictor` (PyTorch nn.Module): Neural network definition
- `F1Predictor` class: Model wrapper with scenario simulation logic
- `backend/app/f1_main.py`: FastAPI endpoints + CORS middleware
- `frontend/src/F1App.tsx`: React UI with scenario configuration
- Shared data: `ai_model/data/preprocessed_f1_data.csv` (126,419 laps)

### Physics Model Components

**Track Wetness Estimation** (predict_lap_times.py:123-145)
- Normalized: `0.5×rainfall + 0.3×humidity + 0.2×(1 - track_temp/50)`
- Exponential drying per stint: `wetness(lap) = initial × exp(-drying_rate × laps)`
- Uses 5-lap rolling average to smooth rainfall spikes

**Tire Performance Factors** (predict_lap_times.py:153-173)
- **INTERMEDIATE**: Optimal at wetness=0.21, penalized when too dry OR too wet
  - Breakthrough insight: Lap times IMPROVE as track dries toward optimal
- **Slick (MEDIUM/SOFT/HARD)**: Linear degradation with tire age + wet penalty
- Baseline = best lap in first 10 laps of each stint

**Stint Detection** (predict_lap_times.py:118-120)
- Auto-detects tire changes: compound switch OR tire age reset
- Each stint: independent baseline + drying curve

### Calibrated Parameters

From `quick_optimal_params.csv` (Monaco 2023, RMSE=4.17s):
```python
drying_rate = 0.070           # Track drying per lap (all compounds)
deg_medium = 0.000700         # MEDIUM tire degradation rate
deg_inter = 0.000030          # INTERMEDIATE degradation (minimal)
inter_optimal_wet = 0.210     # Optimal wetness for INTERS
inter_dry_penalty = 0.048     # INTER penalty when too dry
slick_wet_penalty = 0.060     # MEDIUM wet penalty
```

**IMPORTANT**: Parameters are Monaco-specific. Other tracks require re-calibration.

## Key Files & Directories

### Physics-Based Model
- `predict_lap_times.py` - Main prediction script (4-panel viz)
- `predict_FINAL.py` - Alternative with 6-panel visualization
- `quick_optimal_params.csv` - Calibrated coefficients (Monaco 2023)
- `f1_tire_wear_mathematical_model.ipynb` - Original research notebook
- `calibrate_tire_model.py` - Parameter calibration script
- `quick_calibration.py` - Fast weather-based calibration
- `final_optimization.py` - INTERMEDIATE-focused optimization

### Dynamic Systems Model
- `dynamic_race_simulator.py` - Discrete-time race simulator (8 strategies)
- `strategy_optimizer.py` - Exhaustive search optimizer (3,220 strategies)
- `dynamic_simulation_results/` - Output directory for visualizations
- `optimization_results/` - Optimal strategy data + top 20 ranking

### AI Model
- `ai_model/download_all_f1_data.py` - Download 2018-2024 telemetry
- `ai_model/analyze_data.py` - Feature engineering + preprocessing
- `ai_model/train_model.py` - PyTorch training script
- `ai_model/predict.py` - Run predictions with trained model
- `ai_model/run_all.py` - Complete pipeline (download → train → predict)
- `ai_model/models/best_model.pth` - Trained neural network
- `ai_model/data/preprocessed_f1_data.csv` - 126K preprocessed laps

### Full-Stack Application
- `backend/app/f1_main.py` - FastAPI server (PyTorch inference)
- `frontend/src/F1App.tsx` - React application (main component)
- `frontend/src/index_f1.tsx` - Entry point with Material-UI theme
- `frontend/package.json` - Frontend dependencies

### Documentation
- `README.md` - Main project documentation
- `HOW_TO_USE.md` - Physics model user guide
- `F1_WEBAPP_README.md` - Web application setup guide
- `DYNAMIC_MODEL_DOCUMENTATION.md` - Dynamic systems model explanation
- `QUICKSTART.md` - Quick start guide
- `DEPLOYMENT.md` - Deployment instructions

### Data & Cache
- `fastf1_cache/` - FastF1 telemetry cache (auto-created, gitignored)

## Model Comparison & When to Use Each

| Aspect | Physics Model | Dynamic Systems | AI Model |
|--------|--------------|-----------------|----------|
| **Accuracy** | RMSE ~4.18s | N/A (simulator) | RMSE ~0.40s |
| **Transparency** | Full formula visibility | State equations visible | Black box |
| **Training Data** | Monaco 2023 only | Uses physics params | 126K laps (2018-2024) |
| **Generalization** | Monaco-specific | Configurable track | All tracks in dataset |
| **Speed** | ~10s per race | ~1s per strategy | <100ms per race |
| **Use Case** | Understand physics | Optimize strategy | Maximum accuracy |
| **Scenario Testing** | Manual parameter edits | Built-in (3,220 options) | API-based (web UI) |
| **Real-time** | No | Yes | Yes |

**Recommendation:**
- Use **Physics Model** for educational purposes, understanding tire degradation mechanics
- Use **Dynamic Systems** for race strategy planning, what-if analysis, optimization
- Use **AI Model** for web application, maximum prediction accuracy, production deployment

## Limitations & Important Context

### Physics Model Limitations
- **Track-specific**: Calibrated on Monaco 2023 (street circuit, low-speed)
- **DO NOT expect same accuracy** on high-speed tracks (Monza, Spa, Silverstone)
- **Weather-dependent**: Requires FastF1 weather data; best for wet-to-dry transitions
- **Not modeled**: Safety cars, fuel load reduction, track evolution, traffic, pit outlaps

### AI Model Limitations
- **Black box**: Cannot interpret why predictions are made
- **Data dependency**: Accuracy tied to training data quality (2018-2024)
- **Missing features**: Some older races have incomplete telemetry
- **Overfitting risk**: May not generalize to unprecedented conditions

### Dynamic Systems Limitations
- **Simplified physics**: Uses calibrated parameters from physics model
- **Deterministic**: No stochastic elements (safety car probability, tire failure risk)
- **Single car**: Does not model competitor strategies or overtaking
- **Fixed conditions**: Weather/track evolution not dynamically modeled during simulation

### General Limitations
- FastF1 data derived from timing/GPS (not official FIA ECU data)
- Some fields estimated/interpolated
- Not all sessions have complete weather coverage
- Pit stop times are estimates (varies by team/track)

## Mathematical Formulas (Physics Model)

### Original Research Model
Energy Dissipation → Temperature Effect → Total Wear → Cumulative Degradation → Lap Time Impact

```python
# Energy components
W_brake = K_brake × m × v_mean × N_brake × brake_intensity
W_corner = K_corner × σ_speed² × N_telemetry
W_accel = K_accel × throttle × v_mean × t_lap
W_thermal = K_thermal × (v_mean × t_lap + gear_shifts × 10)

# Temperature effect
T_normalized = 0.4×(v/v_max) + 0.3×(N_brake/N_max) + 0.3×(throttle/100)
T_factor = exp(α × T_normalized)

# Total wear per lap
W_lap = k_compound × (W_brake + W_corner + W_accel + W_thermal) × T_factor

# Cumulative degradation
D_cumulative(n) = Σ W_lap(i) for i=1 to n
D_normalized(n) = 1 - exp(-β × D_cumulative)

# Lap time prediction
t_predicted(n) = t_baseline × (1 + γ × D_normalized(n))
```

### Production Model (Simplified)
The production scripts (`predict_lap_times.py`, `predict_FINAL.py`) use a simpler wetness-based approach:

```python
# Track wetness evolution
wetness(lap) = initial_wetness × exp(-drying_rate × laps_in_stint)

# Tire performance factor
if compound == 'INTERMEDIATE':
    if wetness < optimal:
        factor = 1.0 - dry_benefit × (optimal - wetness)  # Faster as dries
    else:
        factor = 1.0 + wet_penalty × (wetness - optimal)   # Slower when wet
else:  # Slicks
    factor = (1.0 + wet_penalty × wetness) × (1.0 + deg_rate × tire_age)

# Lap time prediction
lap_time = baseline_time × tire_factor
```

### Dynamic Systems State Equations

```python
# State vector
x(t) = [tire_age, fuel_remaining, cumulative_wear, wetness]

# State transitions (discrete-time)
tire_age(t+1) = tire_age(t) + 1        if no pit stop
              = 0                       if pit stop

fuel_remaining(t+1) = fuel_remaining(t) - 1
wetness(t+1) = wetness(t) × exp(-drying_rate)
cumulative_wear(t+1) = cumulative_wear(t) + deg_rate × tire_age(t)

# Output (lap time)
lap_time(t) = baseline × tire_factor(x(t)) + fuel_effect(x(t)) + pit_penalty(u(t))

# Objective function
minimize: J = Σ lap_time(t) for t=1 to T
```

## Common Development Tasks

### Modify Race Configuration (Physics Model)
Edit constants in `predict_lap_times.py`:
```python
YEAR = 2024          # Any year with FastF1 data (2018-2024)
RACE = 'Silverstone' # Race name (Monaco, Silverstone, Spa, etc.)
SESSION = 'R'        # 'R'=Race, 'Q'=Qualifying, 'FP1/2/3'=Practice
DRIVER = 'HAM'       # Three-letter code (VER, HAM, LEC, SAI, etc.)
```

### Modify Dynamic Simulation Parameters
Edit constants in `dynamic_race_simulator.py`:
```python
RACE_LAPS = 70              # Change to 50, 60, 80, etc.
BASELINE_LAP_TIME = 75.0    # Adjust for different tracks
INITIAL_WETNESS = 0.40      # 0.0 = dry, 1.0 = fully wet
PIT_STOP_TIME_LOSS = 22.0   # Track-specific pit lane time
```

Add custom strategies in `define_strategies()`:
```python
strategies = {
    'Custom Strategy': [
        (1, 'SOFT'),      # Start on SOFT
        (20, 'MEDIUM'),   # Pit lap 20 for MEDIUM
        (45, 'SOFT')      # Pit lap 45 for SOFT
    ]
}
```

### Interpret Accuracy Metrics
- **RMSE < 1s**: Excellent (AI model typical)
- **RMSE 1-3s**: Very good (physics model best case)
- **RMSE 3-5s**: Good (physics model production)
- **RMSE 5-10s**: OK (needs track-specific calibration)
- **RMSE > 10s**: Poor (wrong parameters or track conditions)

### Troubleshooting

**Physics model high errors:**
1. Check stint-specific RMSE (printed in output)
2. Examine wetness values (visualization panel 3)
3. Verify tire compound detection (sample predictions table)
4. Look for outlier laps (safety car, pit stops, incidents)
5. Consider re-calibrating for the specific track

**AI model not loading:**
1. Ensure `ai_model/models/best_model.pth` exists
2. Run `cd ai_model && python train_model.py` to train
3. Check `ai_model/data/preprocessed_f1_data.csv` exists
4. Verify PyTorch installation: `python -c "import torch; print(torch.__version__)"`

**Frontend API errors:**
1. Verify backend running: `curl http://localhost:8000`
2. Check CORS settings in `backend/app/f1_main.py`
3. Ensure API_BASE in `F1App.tsx` matches backend URL
4. Check browser console for detailed error messages

**FastF1 data download issues:**
1. First run downloads ~100-500MB (slow, be patient)
2. Subsequent runs use cache in `fastf1_cache/`
3. Clear cache if corrupted: `rm -rf fastf1_cache`
4. Some older races may have incomplete data (expected)

## Important Implementation Notes

### FastF1 Data Handling
- First run on new race downloads ~100-500MB telemetry (slow, be patient)
- Subsequent runs use cache in `fastf1_cache/` (fast)
- Cache is auto-created and gitignored
- Clear cache if corrupted: `rm -rf fastf1_cache`
- Some sessions may have incomplete weather data (model uses defaults)

### Cross-Platform Compatibility
- Scripts use `pathlib.Path` for cross-platform file paths
- Tested on Windows (primary development environment)
- Should work on macOS/Linux without modification
- Frontend uses standard Node.js tooling (cross-platform)

### AI Model Training Considerations
- Training takes 30-60 minutes on CPU, 5-10 minutes on GPU
- Model size: ~13,968 parameters (~54KB file)
- Best results on 2022-2024 data (most complete telemetry)
- Early stopping used to prevent overfitting (patience=10 epochs)
- Scalers saved separately (`scaler_X.pkl`, `scaler_y.pkl`)

### Web Application Architecture
- Backend loads model once at startup (2-3s initialization)
- Inference is fast (<100ms per race prediction)
- CORS configured for development (allow all origins)
- Production deployment should restrict CORS origins
- Frontend connects to backend via Axios HTTP client

## Educational Context

Academic project for **STG-390: Dynamic Systems**. Demonstrates:

**Mathematical Modeling:**
- Physics-based formulas (tire wear, energy dissipation)
- State-space representation (dynamic systems)
- Discrete-time simulation (lap-by-lap evolution)

**Optimization Techniques:**
- Parameter calibration (57 iterations, 28% improvement)
- Exhaustive search (3,220 strategies evaluated)
- Constraint satisfaction (pit stop rules)

**Machine Learning:**
- Deep learning with PyTorch
- Neural network architecture design
- Training/validation/testing pipeline

**Software Engineering:**
- Full-stack web development (React + FastAPI)
- REST API design
- Data visualization (matplotlib, seaborn, recharts)

**Statistical Validation:**
- RMSE, MAE, R² metrics
- Train/val/test split methodology
- Model comparison and selection
