# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

F1 Tire Wear Mathematical Model - A physics-based (non-ML) tire degradation model that predicts lap time changes using real F1 telemetry from FastF1. Calibrated on 2023 Monaco GP data with RMSE ~4.18 seconds.

**Core Concept**: 5 interacting mathematical formulas calculate tire wear from telemetry (speed, braking, throttle), then predict lap time degradation based on cumulative wear and track conditions.

## Development Commands

### Run Predictions (Main Use Case)
```bash
python predict_lap_times.py
```
- Uses calibrated parameters from `quick_optimal_params.csv`
- Edit YEAR, RACE, SESSION, DRIVER variables in the script to analyze different sessions
- Outputs: CSV with predictions + 4-panel PNG visualization
- Default: 2023 Monaco Race, Driver VER (Verstappen)

### Alternative Prediction Script
```bash
python predict_FINAL.py
```
- Same functionality as `predict_lap_times.py` with slightly different visualization (6-panel)
- Uses identical calibrated parameters (RMSE = 4.18s)

### Dependencies
```bash
pip install fastf1 pandas numpy matplotlib seaborn
```

## Architecture

### Data Flow
1. **Load**: FastF1 fetches FIA telemetry + weather data (cached in `fastf1_cache/`)
2. **Extract**: Per-lap features (speed, braking, throttle, gear shifts, tire compound/age)
3. **Model Track Conditions**: Estimate wetness from rainfall/humidity/track temp, apply exponential drying per stint
4. **Calculate Tire Performance**: Compound-specific degradation + wetness penalties
5. **Predict**: `lap_time = baseline × tire_factor`

### Model Components

**Track Wetness Estimation** (predict_lap_times.py:123-145)
- Normalized from rainfall (50%), humidity (30%), track temp (20%)
- Exponential drying per stint: `wetness(lap) = initial × exp(-drying_rate × laps)`
- CRITICAL: Uses 5-lap rolling average to smooth rainfall spikes (track surface != rainfall sensor)

**Tire Performance Factors** (predict_lap_times.py:153-173)
- **INTERMEDIATE**: Penalty when too dry OR too wet (optimal wetness = 0.21)
  - Key insight: Lap times improve as track dries toward optimal point
- **Slick (MEDIUM/SOFT/HARD)**: Linear degradation + wet penalty
- Baseline = best lap in first 10 laps of stint

**Stint Detection** (predict_lap_times.py:118-120)
- Auto-detects tire changes via compound switch or tire age reset
- Each stint has independent baseline + drying curve

### Calibrated Parameters (quick_optimal_params.csv)

```python
{
    'drying_rate': 0.070,          # Track drying speed (all compounds)
    'deg_medium': 0.000700,        # MEDIUM degradation per lap
    'deg_inter': 0.000030,         # INTERMEDIATE degradation (minimal)
    'inter_optimal_wet': 0.210,    # Optimal wetness for INTERS
    'inter_dry_penalty': 0.048,    # Penalty when too dry
    'slick_wet_penalty': 0.060     # MEDIUM wet penalty
}
```

**DO NOT** modify these without re-calibration (see Limitations below).

## Key Files

- `predict_lap_times.py` - Production prediction script (recommended)
- `predict_FINAL.py` - Alternative with extended visualization
- `quick_optimal_params.csv` - Calibrated model parameters (RMSE = 4.17s)
- `f1_tire_wear_mathematical_model.ipynb` - Original research notebook
- `HOW_TO_USE.md` - Detailed user guide with calibration history
- `fastf1_cache/` - Telemetry cache (auto-created, excluded from git)

## Limitations & Important Context

### Track-Specific Calibration
- **Current calibration**: Monaco 2023 only (street circuit, low-speed corners)
- **DO NOT expect same accuracy** on high-speed tracks (Monza, Spa, Silverstone)
- To use on different tracks: Re-run calibration with target track data

### Weather Dependency
- Model requires FastF1 weather data (rainfall, humidity, track temp)
- Accuracy degrades if weather data missing (uses defaults)
- Best for wet-to-dry transitions (Monaco 2023 had rain → INTERMEDIATE stint)

### Race Conditions Not Modeled
- Safety car periods (artificially fast/slow laps)
- Fuel load reduction (lighter car = faster laps independent of tires)
- Track evolution (rubber buildup improves grip)
- Traffic/DRS/slipstream effects
- Pit stop outlaps (cold tires)

### Telemetry Limitations
- FastF1 data is derived from timing/GPS (not official FIA ECU data)
- Some fields may be estimated/interpolated
- Not all sessions have complete weather coverage

## Mathematical Model (Reference)

**Energy Dissipation** → **Temperature Effect** → **Total Wear** → **Cumulative Degradation** → **Lap Time Impact**

Key formulas:
```
W_brake = K_brake × m × v_mean × N_brake × brake_intensity
W_corner = K_corner × σ_speed² × N_telemetry
W_accel = K_accel × throttle × v_mean × t_lap
W_thermal = K_thermal × (v_mean × t_lap + gear_shifts × 10)

T_factor = exp(α × T_normalized)
W_lap = k_compound × (W_brake + W_corner + W_accel + W_thermal) × T_factor

D_cumulative(n) = Σ W_lap(i)
D_normalized(n) = 1 - exp(-β × D_cumulative)

t_predicted(n) = t_baseline × (1 + γ × D_normalized(n))
```

(Implementation differs slightly in production scripts - uses simpler wetness-based model)

## Common Development Tasks

### Change Target Race/Driver
Edit in `predict_lap_times.py`:
```python
YEAR = 2024          # Any year with FastF1 data
RACE = 'Silverstone' # Race name (see FastF1 docs for list)
SESSION = 'R'        # 'R'=Race, 'Q'=Qualifying, 'FP1/2/3'=Practice
DRIVER = 'HAM'       # Three-letter code (HAM, LEC, SAI, etc.)
```

### Interpret Results
- **RMSE < 3s**: Excellent (theoretical best achieved in calibration)
- **RMSE 3-5s**: Good (current production performance)
- **RMSE 5-10s**: OK (needs re-calibration for track)
- **RMSE > 10s**: Poor (wrong track/conditions)

### Debug High Errors
1. Check stint-specific RMSE (printed in output)
2. Examine wetness values (visualization panel 3)
3. Verify tire compound detection (sample predictions table)
4. Look for outlier laps (safety car, pit stops)

## FastF1 Notes

- First run on new track downloads ~100MB telemetry (slow)
- Subsequent runs use cache (fast)
- Cache location: `./fastf1_cache/` (excluded from git)
- Clear cache if data seems corrupted: `rm -rf fastf1_cache`

## Windows-Specific
- Scripts tested on Windows (uses `Path` for cross-platform compatibility)
- Output uses `\r` for in-place lap counter updates
- No bash-specific commands used

## Educational Context

Academic project for STG-390: Dynamic Systems. Demonstrates:
- Physics-based modeling without AI/ML
- Real-world data integration
- Iterative parameter optimization
- Statistical validation (RMSE, MAE, R²)
- Transparent, reproducible results
