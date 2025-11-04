# F1 Tire Wear AI - Full Stack Web Application

Interactive web application for F1 lap time prediction with scenario comparison using PyTorch AI model.

## 🚀 Features

- **Interactive Configuration**: Select year, race, driver, and adjust parameters
- **Scenario Comparison**: Compare baseline vs improved tires, fuel load, weather
- **Real-Time Predictions**: AI predictions in < 1 second
- **Visual Analytics**: Interactive charts showing lap progression and comparisons
- **126K+ Training Laps**: Model trained on 7 years of F1 data (2018-2024)
- **0.40s RMSE**: Extremely accurate predictions

## 📁 Project Structure

```
f1-tire-wear-model/
├── backend/
│   └── app/
│       └── f1_main.py           # FastAPI backend server
├── frontend/
│   ├── src/
│   │   ├── F1App.tsx            # Main React application
│   │   └── index_f1.tsx         # Entry point with MUI theme
│   └── package.json             # Updated dependencies
└── ai_model/
    ├── models/
    │   └── best_model.pth       # Trained PyTorch model
    └── data/
        └── preprocessed_f1_data.csv  # 126K laps dataset
```

## 🔧 Setup Instructions

### Backend Setup

1. **Install Python dependencies:**
   ```bash
   cd backend
   pip install fastapi uvicorn torch pandas numpy scikit-learn
   ```

2. **Start the API server:**
   ```bash
   cd backend/app
   python f1_main.py
   ```

   Server will run on: `http://localhost:8000`

3. **Verify it's working:**
   - Open browser: `http://localhost:8000`
   - Should see: `{"status":"healthy","model_loaded":true,"available_races":149}`

### Frontend Setup

1. **Install dependencies:**
   ```bash
   cd frontend
   npm install
   ```

   This will install:
   - React + TypeScript
   - Material-UI (MUI) components
   - Recharts for data visualization
   - Axios for API calls

2. **Update entry point** (replace index.tsx):
   ```bash
   # Windows
   copy src\index_f1.tsx src\index.tsx

   # Or manually rename index_f1.tsx to index.tsx
   ```

3. **Start development server:**
   ```bash
   npm start
   ```

   Frontend will run on: `http://localhost:3000`

## 🎮 How to Use

### 1. **Base Scenario Configuration**
   - Select **Year** (2018-2024)
   - Select **Race** (Monaco, Silverstone, etc.)
   - Select **Driver** (VER, HAM, LEC, etc.)
   - Adjust **Tire Improvement** (0-50%)
   - Adjust **Fuel Load Factor** (0.8x-1.2x)

### 2. **Run Single Prediction**
   - Click **"Run Single"** to predict lap times for base scenario
   - View results:
     - Total race time
     - Average lap time
     - Fastest/slowest laps
     - Lap-by-lap progression chart
     - Tire strategy breakdown

### 3. **Compare Scenarios**
   - Switch to **"Compare"** tab
   - Configure comparison scenario (e.g., +25% better tires)
   - Click **"Compare"**
   - View:
     - Side-by-side comparison cards
     - Time savings analysis
     - Percentage improvement
     - Dual-line lap progression chart
     - Bar chart comparison

### 4. **Example Scenarios to Try**

**Scenario 1: Normal vs Improved Tires**
- Base: 2023 Monaco, VER, 0% tire improvement
- Compare: 2023 Monaco, VER, 25% tire improvement
- **See**: How much faster lap times would be with better tires

**Scenario 2: Fuel Load Impact**
- Base: 2023 Monaco, VER, 1.0x fuel
- Compare: 2023 Monaco, VER, 0.9x fuel (lighter car)
- **See**: Impact of fuel weight on lap times

**Scenario 3: Extreme Tire Tech**
- Base: 2023 Monaco, VER, 0% improvement
- Compare: 2023 Monaco, VER, 50% improvement
- **See**: Theoretical maximum tire performance gain

## 📊 API Endpoints

### GET `/`
Health check
```json
{
  "status": "healthy",
  "model_loaded": true,
  "available_races": 149
}
```

### GET `/model/info`
Model information
```json
{
  "input_features": 25,
  "total_parameters": 13968,
  "architecture": [25, 128, 64, 32, 16, 1],
  "test_rmse": 0.3979,
  "test_mae": 0.2913,
  "test_r2": 0.9988
}
```

### GET `/races/available`
List all available races
```json
{
  "races": [
    {
      "year": 2023,
      "race": "Monaco",
      "drivers": ["VER", "HAM", "LEC", ...],
      "laps": 1127
    },
    ...
  ]
}
```

### POST `/predict`
Run predictions
```json
{
  "scenarios": [
    {
      "year": 2023,
      "race": "Monaco",
      "driver": "VER",
      "tire_improvement": 0.0,
      "fuel_load_factor": 1.0,
      "weather_override": null
    }
  ],
  "comparison_mode": true
}
```

Response:
```json
{
  "scenarios": [
    {
      "scenario_name": "2023 Monaco - VER",
      "total_race_time": 6843.21,
      "average_lap_time": 91.24,
      "fastest_lap": 84.56,
      "slowest_lap": 103.45,
      "laps": [...],
      "tire_strategy": [...]
    }
  ],
  "comparison": {
    "baseline": "2023 Monaco - VER",
    "comparisons": [
      {
        "scenario": "+25% Tires",
        "time_saved": 45.23,
        "percentage_improvement": 0.66
      }
    ]
  }
}
```

## 🎨 Frontend Components

### **ScenarioConfig**
- Race/driver selection dropdowns
- Tire improvement slider (0-50%)
- Fuel load slider (0.8x-1.2x)
- Weather override controls (future)

### **ResultCard**
- Summary statistics
- Total race time
- Average, fastest, slowest laps
- Tire strategy info

### **Lap Times Chart** (Line Chart)
- X-axis: Lap number
- Y-axis: Lap time (seconds)
- Multiple lines for comparison scenarios
- Interactive tooltips

### **Comparison Bar Chart**
- Side-by-side metric comparison
- Total race time
- Average lap time
- Visual difference highlighting

## 🔧 Customization

### Add New Features

**Backend (f1_main.py):**
```python
# Add weather override support
if scenario.weather_override:
    for key, value in scenario.weather_override.items():
        if key in modified_data.columns:
            modified_data[key] = value
```

**Frontend (F1App.tsx):**
```tsx
// Add weather controls
<Slider
  label="Track Temperature"
  value={scenario.weather_override?.track_temp || 25}
  onChange={(v) => setWeatherOverride('track_temp', v)}
  min={10}
  max={50}
/>
```

### Styling

Modify theme in `index_f1.tsx`:
```tsx
const theme = createTheme({
  palette: {
    primary: { main: '#e10600' },  // F1 Red
    secondary: { main: '#15151e' }, // F1 Black
  },
});
```

## 🐛 Troubleshooting

**Backend won't start:**
```bash
# Check if model exists
ls ai_model/models/best_model.pth

# If missing, train model first:
cd ai_model
python train_model.py
```

**Frontend API errors:**
- Ensure backend is running on port 8000
- Check CORS settings in `f1_main.py` (currently allows all origins)
- Verify API_BASE in `F1App.tsx` matches backend URL

**No races available:**
- Backend needs preprocessed data
- Run: `cd ai_model && python analyze_data.py`

**Slow predictions:**
- First prediction loads model (2-3s)
- Subsequent predictions are fast (<100ms)
- GPU recommended for best performance

## 📦 Deployment

### Backend (Production)

```bash
# Use Gunicorn with Uvicorn workers
pip install gunicorn
gunicorn -w 4 -k uvicorn.workers.UvicornWorker app.f1_main:app --bind 0.0.0.0:8000
```

### Frontend (Production Build)

```bash
cd frontend
npm run build

# Serve with any static server
npx serve -s build -l 3000
```

## 🎯 Performance

- **Backend Response Time**: < 100ms per prediction
- **Frontend Load Time**: < 2s initial load
- **Model Inference**: < 50ms for full race prediction
- **Memory Usage**: ~500MB (model + data loaded)

## 🚧 Future Enhancements

- [ ] Real-time race monitoring
- [ ] Custom tire strategy planner
- [ ] Weather scenario generator
- [ ] Multi-race comparison
- [ ] Export results to CSV/PDF
- [ ] 3D track visualizations
- [ ] Pit stop strategy optimizer
- [ ] Driver comparison mode

## 📝 Notes

- Model trained on 2018-2024 data (126,419 laps)
- Best accuracy on 2022-2024 races (complete telemetry)
- 2018-2021 has some missing data but still usable
- Monaco 2023 is the calibration baseline

## 🎓 Academic Context

Built for STG-390: Dynamic Systems project demonstrating:
- Full-stack web development
- PyTorch deep learning integration
- Real-time data visualization
- Interactive scenario modeling
- REST API design

---

**🏎️ Ready to race! Start the backend and frontend, then open http://localhost:3000**
