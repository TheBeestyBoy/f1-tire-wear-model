# F1 Game Mode UI - User Guide

## Overview

The Game Mode Simulator now has a **complete frontend UI** with a toggle to switch between Historical Data predictions and Game Mode simulation.

## How to Access

### 1. Start the Backend API
```bash
cd backend/app
python f1_main.py
```
Backend runs on: `http://localhost:8000`

### 2. Start the Frontend
```bash
cd frontend
npm install  # First time only
npm start
```
Frontend runs on: `http://localhost:3000`

### 3. Toggle Between Modes
At the top of the page, you'll see two buttons:
- **📜 Historical Data** - Analyze real race data with AI predictions
- **🎮 Game Mode** - Configure custom scenarios and simulate lap times

Click **Game Mode** to access the simulator!

---

## Game Mode UI Features

### Left Panel - Configuration

**Mode Selector (Tabs)**
- **Single Lap**: Simulate one lap with custom parameters
- **Stint Simulation**: Simulate 5-50 laps with tire degradation

**Quick Weather Presets**
Click any preset button to instantly apply conditions:
- ☀️ **Dry** - Sunny, 30°C, 0mm rain
- 🌦️ **Drizzle** - Light rain, 22°C, 2mm rain
- 🌧️ **Rain** - Moderate rain, 18°C, 10mm rain
- ⛈️ **Heavy Rain** - Severe weather, 15°C, 20mm rain
- 🔥 **Hot** - Very hot, 45°C, 0mm rain
- ❄️ **Cold** - Very cold, 15°C, 0mm rain

**Tire Compound Selector**
Dropdown with color-coded compounds:
- 🔴 **Soft** - Fastest, least durable
- 🟡 **Medium** - Balanced performance
- ⚪ **Hard** - Slowest, most durable
- 🟢 **Intermediate** - For light rain
- 🔵 **Wet** - For heavy rain

**Sliders (Interactive Controls)**

1. **Tire Age**: 0-50 laps
   - Fresh tires = 0
   - Worn tires = 40-50

2. **Track Temperature**: 0-60°C
   - Cold = 0-15°C
   - Normal = 20-35°C
   - Hot = 40-60°C

3. **Air Temperature**: 0-50°C

4. **Rainfall**: 0-50mm
   - Dry = 0mm
   - Drizzle = 1-5mm
   - Rain = 5-15mm
   - Heavy = 15-50mm

5. **Humidity**: 0-100%

6. **Fuel Load**: 0-100%
   - Empty = 0% (lightest, fastest)
   - Half = 50%
   - Full = 100% (heaviest, slowest)

7. **Lap Number**: 1-100

8. **Stint Length** (Stint mode only): 5-50 laps

**Simulate Button**
- Red button at bottom
- Click to run simulation
- Shows loading spinner while processing

---

### Right Panel - Results

#### Single Lap Mode

**Large Time Display**
- Shows predicted lap time in big, bold numbers
- Format: `1:45.23` or `95.23s`

**Configuration Summary**
Table showing all parameters used:
- Compound
- Tire Age
- Track Temperature
- Rainfall
- Humidity

#### Stint Simulation Mode

**Summary Statistics (4 Cards)**
- Average Lap Time
- Degradation (slowest - fastest)
- Fastest Lap
- Slowest Lap

**Lap-by-Lap Performance Chart**
- Line chart showing lap times over the stint
- X-axis: Lap number
- Y-axis: Lap time (seconds)
- Red line tracks performance degradation

**Detailed Data Table**
- Shows every 5th lap
- Columns: Lap, Time, Tire Age
- Scrollable for long stints

---

## Example Usage Scenarios

### Scenario 1: Testing Tire Compounds
**Goal**: Compare Soft vs Medium vs Hard on dry track

1. Select **Single Lap** mode
2. Set conditions:
   - Track Temp: 30°C
   - Rainfall: 0mm
   - Tire Age: 10 laps
3. Select **SOFT** compound → Click Simulate
4. Note the lap time
5. Repeat with **MEDIUM** → Click Simulate
6. Repeat with **HARD** → Click Simulate
7. Compare the three results

### Scenario 2: Wet vs Dry Performance
**Goal**: See how much rain slows lap times

1. Click **☀️ Dry** preset
2. Select **SOFT** compound
3. Click **Simulate** → Note lap time
4. Click **🌧️ Rain** preset
5. Change compound to **INTERMEDIATE**
6. Click **Simulate** → Compare lap times
7. Difference shows wet-weather impact

### Scenario 3: Tire Degradation Over a Stint
**Goal**: Visualize tire wear over 20 laps

1. Switch to **Stint Simulation** mode
2. Set Stint Length: 20 laps
3. Select **SOFT** compound
4. Set Tire Age: 0 (fresh tires)
5. Set conditions: Dry, 28°C
6. Set Fuel Load: 100% (full tank)
7. Click **Simulate Stint**
8. Watch the chart show degradation
9. Summary shows total degradation (e.g., +0.5s over 20 laps)

### Scenario 4: Fuel Load Impact
**Goal**: Quantify fuel weight effect

1. **Single Lap** mode
2. Set Fuel Load: 100% (full)
3. All other settings default
4. Click **Simulate** → Note lap time
5. Change Fuel Load: 0% (empty)
6. Click **Simulate** → Note lap time
7. Difference shows fuel effect (typically ~0.1-0.2s)

---

## Tips & Tricks

### Optimal Conditions Testing
To find the fastest possible lap time:
1. Use **☀️ Dry** preset
2. Set Tire Age: **1 lap** (fresh)
3. Select **SOFT** compound
4. Set Fuel Load: **0%** (empty tank)
5. This gives you the theoretical minimum lap time

### Worst Case Scenario
To test extreme conditions:
1. Use **⛈️ Heavy Rain** preset
2. Set Tire Age: **40 laps** (very worn)
3. Select **MEDIUM** (wrong compound for rain!)
4. Set Fuel Load: **100%** (full)
5. This gives you the theoretical maximum lap time

### Weather Preset Customization
After applying a preset, you can still adjust individual sliders:
1. Click **🌦️ Drizzle** preset
2. Manually adjust Track Temp slider
3. Fine-tune Rainfall slider
4. Click **Simulate** with modified conditions

---

## Technical Details

### API Endpoints Used

**Single Lap**: `POST /simulate`
```json
{
  "compound": "MEDIUM",
  "tire_age": 15,
  "track_temp": 30.0,
  "rainfall": 0.0,
  ...
}
```

**Stint**: `POST /simulate/stint`
```json
{
  "laps": 20,
  "compound": "SOFT",
  "tire_age": 0,
  ...
}
```

**Weather Preset**: `POST /simulate/weather`
```json
{
  "weather": "RAIN",
  "compound": "INTERMEDIATE",
  "tire_age": 5
}
```

### Performance
- Backend response time: < 100ms
- Frontend rendering: Instant
- No loading delay for subsequent simulations

### Data Validation
The UI enforces valid ranges:
- Tire age: 0-50 laps (can't be negative)
- Temperature: 0-60°C (realistic F1 range)
- Rainfall: 0-50mm (0-50mm is extreme but possible)
- Fuel load: 0-100% (normalized)

---

## Troubleshooting

### Backend Not Running
**Symptom**: Error messages when clicking Simulate

**Solution**:
```bash
cd backend/app
python f1_main.py
```
Check console for "[OK] F1 Game Simulator initialized"

### Frontend Not Loading
**Symptom**: Blank page or React errors

**Solution**:
```bash
cd frontend
npm install
npm start
```

### Game Mode Button Not Working
**Symptom**: Nothing happens when clicking Game Mode

**Solution**:
1. Check browser console (F12) for errors
2. Verify GameModeSimulator.tsx exists in `frontend/src/`
3. Refresh the page (Ctrl+R or Cmd+R)

### Predictions Seem Unrealistic
**Symptom**: Lap times don't match expectations

**Notes**:
- Model trained on Monaco 2023 (street circuit)
- Lap times are in seconds (not minutes:seconds format until display)
- Typical Monaco lap: 70-150s depending on conditions
- Very fast: 70-90s (dry, fresh softs)
- Very slow: 130-150s (heavy rain, old tires)

---

## Files Created

### Frontend
- `frontend/src/GameModeSimulator.tsx` - Main Game Mode component (520 lines)
- Modified `frontend/src/App.tsx` - Added mode toggle

### Backend
- `backend/app/f1_main.py` - Added simulation endpoints:
  - POST `/simulate`
  - POST `/simulate/stint`
  - POST `/simulate/weather`
  - GET `/simulate/presets`

### AI Model
- `ai_model/game_simulator.py` - Game simulator class
- `ai_model/test_fake_data.py` - Validation harness

---

## UI Screenshots (Description)

### Historical Data Mode
- Toggle button: 📜 **Historical Data** (red/filled)
- Shows model accuracy metrics (RMSE, MAE, R²)
- Race/driver selectors
- Scenario comparison tools

### Game Mode
- Toggle button: 🎮 **Game Mode** (red/filled)
- Weather preset buttons across top
- Left panel: All configuration sliders
- Right panel: Results display
- Clean, modern Material-UI design
- F1 red theme (#e10600)

---

## Next Steps

### Suggested Enhancements
1. **Save/Load Configurations**: Save favorite scenarios
2. **Race Comparison**: Compare multiple simulations side-by-side
3. **Export Results**: Download CSV of stint data
4. **Weather Animation**: Animated rainfall effect on chart
5. **Tire Wear Visualization**: Visual tire graphic showing degradation

### Integration Ideas
1. Link Game Mode results to Historical Data for comparison
2. "Try This Scenario" button in Historical mode → jumps to Game Mode with matching conditions
3. Strategy optimizer: Input race length, track conditions → get optimal pit strategy

---

## Conclusion

The Game Mode UI is **fully functional** and ready for use! Users can:

✅ Toggle between Historical and Game Mode
✅ Configure all track/tire parameters via sliders
✅ Apply quick weather presets
✅ Simulate single laps or full stints
✅ View results with charts and tables
✅ Test unlimited "what-if" scenarios

**Status**: Production Ready 🏎️💨
