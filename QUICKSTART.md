# 🏎️ F1 Tire Wear AI - Quick Start Guide

## 🚀 Get Running in 3 Steps

### Step 1: Setup Frontend (ONE TIME ONLY)
```bash
# Double-click this file:
setup_frontend.bat

# Or run manually:
cd frontend
npm install
```

### Step 2: Start Backend
```bash
# Double-click this file in one terminal:
start_backend.bat

# Wait for: "Uvicorn running on http://0.0.0.0:8000"
```

### Step 3: Start Frontend
```bash
# Double-click this file in another terminal:
start_frontend.bat

# Wait for browser to open at http://localhost:3000
```

---

## 🎮 Using the Application

### Basic Usage

1. **Select Race**
   - Choose Year (2018-2024)
   - Choose Race (Monaco, Silverstone, etc.)
   - Choose Driver (VER, HAM, LEC, etc.)

2. **Run Prediction**
   - Click **"Run Single"**
   - See lap-by-lap predictions
   - View total race time and strategy

### Comparison Mode

1. **Configure Base Scenario** (Tab 1)
   - 2023 Monaco, VER
   - Tire Improvement: 0%
   - Fuel Load: 1.0x

2. **Configure Comparison** (Tab 2)
   - 2023 Monaco, VER
   - Tire Improvement: 25% ← CHANGE THIS
   - Fuel Load: 1.0x

3. **Click "Compare"**
   - See time savings
   - View lap progression charts
   - Analyze performance difference

---

## 📊 Example Scenarios

### Scenario 1: Better Tires
- **Question**: How much faster with 25% better tires?
- **Setup**: Base (0%) vs Compare (25% tire improvement)
- **Expected**: ~30-60 seconds faster total race time

### Scenario 2: Lighter Car
- **Question**: Impact of less fuel weight?
- **Setup**: Base (1.0x fuel) vs Compare (0.9x fuel)
- **Expected**: Slight improvement in later laps

### Scenario 3: Extreme Tech
- **Question**: Maximum tire performance gain?
- **Setup**: Base (0%) vs Compare (50% tire improvement)
- **Expected**: Dramatic time savings (1-2 minutes)

---

## 🎯 What You'll See

### Result Cards
- Total Race Time: `1:54:32.45`
- Average Lap Time: `91.234s`
- Fastest Lap: `84.567s`
- Tire Strategy: `3 stints`

### Charts
1. **Lap Times Progression**
   - Line chart showing each lap
   - Multiple scenarios overlay
   - See degradation patterns

2. **Performance Comparison**
   - Bar chart of total/average times
   - Visual difference highlighting
   - Easy to see improvements

### Comparison Analysis
```
📊 Comparison Analysis

+25% Tires
  Time Saved: 45.23s total (0.66% improvement)
  Per Lap: 0.602s faster
```

---

## 🐛 Troubleshooting

**Backend won't start**
- Ensure PyTorch model exists: `ai_model/models/best_model.pth`
- If missing, run: `cd ai_model && python train_model.py`

**Frontend shows error**
- Make sure backend is running first
- Check backend shows: `http://localhost:8000`
- Try refreshing the page

**No races in dropdown**
- Backend needs data: `ai_model/data/preprocessed_f1_data.csv`
- If missing, run: `cd ai_model && python analyze_data.py`

**Prediction fails**
- Check selected race/driver combination exists
- Try Monaco 2023 with VER (guaranteed to work)

---

## 📱 Application Features

✅ **Interactive Configuration**
- Easy-to-use dropdowns and sliders
- Real-time parameter adjustment
- Multiple scenario support

✅ **Fast Predictions**
- < 100ms prediction time
- Full race in < 1 second
- Powered by PyTorch + GPU

✅ **Visual Analytics**
- Interactive charts (hover for details)
- Multiple visualization types
- Export-ready graphics

✅ **Scenario Comparison**
- Side-by-side analysis
- Time savings calculation
- Percentage improvements

✅ **126K Training Laps**
- 7 years of F1 data
- 149 races available
- 40 different drivers

---

## 🎓 Model Performance

- **RMSE**: 0.40 seconds (excellent!)
- **MAE**: 0.29 seconds
- **R² Score**: 0.9988 (99.88% accuracy)
- **Improvement**: 90.5% better than math model

---

## 🔥 Pro Tips

1. **Monaco 2023 - VER**: Best calibrated race
2. **Compare Mode**: Always more interesting than single
3. **Tire Slider**: Most impactful parameter
4. **Fuel Slider**: Subtle but realistic effects
5. **Chart Hover**: See exact lap times
6. **Multiple Runs**: Try different drivers same race

---

## 📚 Full Documentation

See `F1_WEBAPP_README.md` for:
- Detailed API documentation
- Architecture overview
- Deployment instructions
- Customization guide

---

## ⚡ Quick Commands

```bash
# Backend
cd backend/app
python f1_main.py

# Frontend
cd frontend
npm start

# Install backend deps
pip install fastapi uvicorn torch pandas numpy scikit-learn

# Install frontend deps
cd frontend
npm install
```

---

**🏁 Ready to race! Enjoy exploring F1 lap time predictions!**
