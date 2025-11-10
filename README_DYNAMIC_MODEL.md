# F1 Tire Wear - Dynamic Systems Model

## Quick Start Guide for Resubmission

### What Changed?
**Before**: Static validation models that analyzed historical race data
**Now**: Dynamic simulation framework that predicts and optimizes future race strategies

---

## How to Run (Show This to Professor)

### 1. Dynamic Scenario Simulation
```bash
python dynamic_race_simulator.py
```
**Demonstrates**: DESCRIPTIVE + PREDICTIVE model
- Simulates 8 different pit strategies over 70 laps
- Shows how race unfolds lap-by-lap under different decisions
- Identifies best strategy from tested scenarios
- **Time**: ~5 seconds

**Output**:
- 4 graphs showing strategy comparisons
- Shows how state variables (tire wear, wetness) evolve over time

---

### 2. Prescriptive Strategy Optimizer
```bash
python strategy_optimizer.py
```
**Demonstrates**: PRESCRIPTIVE model
- Automatically finds optimal pit strategy
- Evaluates 3,220 different scenarios
- Recommends best decision for minimum race time
- Includes sensitivity analysis
- **Time**: ~30 seconds

**Output**:
```
*** OPTIMAL STRATEGY FOUND ***
  Strategy: 0-Stop INTERMEDIATE
  Total Race Time: 84.71 minutes
  Time saved vs worst: 311.1s (5.18 min)
```

---

## Key Files to Review

### NEW: Dynamic Model Files

1. **`dynamic_race_simulator.py`** (400+ lines)
   - Discrete-event simulation engine
   - `RaceSimulator` class implements state transitions
   - Simulates lap-by-lap race evolution
   - **Lines 79-200**: Core dynamic model implementation

2. **`strategy_optimizer.py`** (350+ lines)
   - Prescriptive optimization framework
   - Exhaustive search over strategy space
   - **Lines 136-205**: Optimization algorithm

3. **`DYNAMIC_MODEL_DOCUMENTATION.md`** (Comprehensive explanation)
   - Explains static vs dynamic difference
   - Shows how feedback was addressed
   - Mathematical formulation
   - Usage instructions

### OLD: Static Model Files (For Comparison)

4. **`predict_lap_times.py`** (Static validation)
   - Shows what we had before
   - Only validates accuracy on historical data
   - No controllable decisions or scenarios

---

## How This Addresses Feedback

### Professor Said:
> "You have two static models... You need to define a DYNAMIC MODEL that describes what occurs now, predicts what will happen in the future, and prescribes potential solutions."

### How We Fixed It:

#### ✅ 1. "Describes what occurs NOW"
**`RaceSimulator.simulate_lap()` method**:
```python
# At any lap t, we can query exact system state
State(lap=35) = {
    tire_age: 10 laps,
    wetness: 0.18,
    compound: 'SOFT',
    lap_time: 75.8s,
    cumulative_time: 2641.2s
}
```

#### ✅ 2. "Predicts what will happen in the FUTURE"
**`run_strategy()` method**:
- Tests multiple "what-if" scenarios
- Example: "What if we pit on lap 25 vs lap 30?"
- Predicts all 70 future laps for any strategy

#### ✅ 3. "Prescribes potential solutions for the FUTURE"
**`optimize_strategy()` function**:
- Searches 3,220 possible strategies
- Finds optimal: "Use 0-stop INTERMEDIATE strategy"
- Recommends actionable decision (not just analysis)

#### ✅ 4. "Vary factors AS A FUNCTION OF TIME"
**Time-variant state variables**:
- Tire age: +1 per lap
- Wetness: exponential decay per lap
- Fuel: decreases each lap
- Lap time: changes based on state

#### ✅ 5. "Run simulations with multiple scenarios"
**Completed**:
- Scenario testing: 8 strategies
- Optimization: 3,220 strategies
- Sensitivity analysis: 12 variations

---

## Visual Proof (Graphs Generated)

### Dynamic Simulation Outputs
Located in `dynamic_simulation_results/`:

1. **`strategy_comparison_lap_times.png`**
   - Shows lap time evolution over 70 laps
   - Compares 8 different strategies
   - Demonstrates TIME-VARYING behavior

2. **`strategy_comparison_cumulative.png`**
   - Race time progression
   - Shows when strategies gain/lose time
   - Illustrates strategic decisions matter

3. **`optimal_strategy_detail.png`**
   - Detailed view of best strategy
   - Shows state variables evolving lap-by-lap
   - Proves model is dynamic (not static)

4. **`strategy_performance_comparison.png`**
   - Bar chart ranking all strategies
   - Quantifies value of optimization

---

## Technical Summary for Report

### System Type
**Discrete-time dynamic system with control inputs**

### Mathematical Formulation
```
State:    x(t) = [tire_age, fuel, wetness, cumulative_wear]
Control:  u(t) = [pit_decision, compound_choice]
Dynamics: x(t+1) = f(x(t), u(t), θ)
Output:   y(t) = lap_time(x(t), u(t))
Objective: minimize Σ y(t) + Σ pit_penalties
```

### Model Components
1. **DESCRIPTIVE**: State-space representation of race system
2. **PREDICTIVE**: Scenario simulation for any strategy
3. **PRESCRIPTIVE**: Optimization to find best strategy

### Validation
- Physics model calibrated on Monaco 2023 (RMSE = 4.18s)
- Dynamic model verified: state transitions, conservation laws
- Sensitivity analysis: robust to parameter variations

---

## What to Say to Professor

> "I revised the project based on your feedback. I realized I was only validating my physics models on historical data, which is a static analysis. I've now built a complete dynamic systems framework:
>
> **1. Dynamic Simulation** (`dynamic_race_simulator.py`): Simulates how the race unfolds lap-by-lap under different pit strategies. State variables (tire wear, fuel, track wetness) evolve over time based on control decisions.
>
> **2. Prescriptive Optimization** (`strategy_optimizer.py`): Automatically finds the optimal pit strategy by searching over 3,220 scenarios and recommending the best decision.
>
> The key difference: my original model analyzed PAST races to validate accuracy. The new model PREDICTS FUTURE race outcomes for different strategies and PRESCRIBES which strategy to use. It's now a true dynamic model with time-varying states, control inputs, and optimization."

---

## Dependencies
All dependencies already installed from original project:
```bash
pip install fastf1 pandas numpy matplotlib seaborn
```

**Note**: FastF1 is NOT required for dynamic model (only for static validation)
The simulator and optimizer run standalone using calibrated parameters.

---

## File Organization

```
f1-tire-wear-model/
├── dynamic_race_simulator.py        ← NEW: Main simulator
├── strategy_optimizer.py            ← NEW: Prescriptive optimizer
├── DYNAMIC_MODEL_DOCUMENTATION.md   ← NEW: Technical documentation
├── README_DYNAMIC_MODEL.md          ← NEW: This file
│
├── dynamic_simulation_results/      ← NEW: Scenario graphs (created on run)
├── optimization_results/            ← NEW: Optimizer CSVs (created on run)
│
├── predict_lap_times.py             ← OLD: Static validation
├── quick_optimal_params.csv         ← Calibrated physics parameters
└── CLAUDE.md                        ← Original project instructions
```

---

## Differences at a Glance

| Feature | Static Model (Old) | Dynamic Model (New) |
|---------|-------------------|---------------------|
| **Input** | Historical telemetry | Strategy definition |
| **Output** | Prediction RMSE | Optimal strategy recommendation |
| **Time Direction** | Backward (validation) | Forward (prediction) |
| **Scenarios** | 1 (actual race) | 3,220 (what-if) |
| **Control** | None | Pit timing, tire choice |
| **Prescriptive** | ❌ No | ✅ **YES** |
| **Dynamic** | ❌ No | ✅ **YES** |

---

## Questions?

**Q: Do I still need the static model?**
A: Yes, it validates the physics. The dynamic model USES the calibrated physics parameters from static validation.

**Q: Which file should I show the professor?**
A: Run both:
1. `python dynamic_race_simulator.py` (scenario testing)
2. `python strategy_optimizer.py` (prescriptive optimization)

Then show the graphs and explain using `DYNAMIC_MODEL_DOCUMENTATION.md`

**Q: Is this really a dynamic model now?**
A: **YES**. It has:
- ✅ State variables that evolve over time
- ✅ Control inputs (pit decisions)
- ✅ Discrete-time stepping (lap by lap)
- ✅ Multiple scenario simulations
- ✅ Prescriptive optimization

---

## Contact
For STG-390 resubmission. Demonstrates complete dynamic systems modeling:
- Descriptive (state-space)
- Predictive (scenario simulation)
- Prescriptive (strategy optimization)
