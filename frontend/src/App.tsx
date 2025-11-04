import React, { useState, useEffect } from 'react';
import {
  Container,
  Typography,
  Box,
  Paper,
  Grid,
  Button,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Slider,
  Card,
  CardContent,
  Tabs,
  Tab,
  CircularProgress,
  Alert,
  Chip,
  Divider,
} from '@mui/material';
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  ComposedChart,
} from 'recharts';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import CompareArrowsIcon from '@mui/icons-material/CompareArrows';
import TrendingDownIcon from '@mui/icons-material/TrendingDown';
import SpeedIcon from '@mui/icons-material/Speed';

// ============================================================================
//                          TYPES
// ============================================================================

interface RaceScenario {
  year: number;
  race: string;
  driver: string;
  tire_improvement: number;
  fuel_load_factor: number;
  weather_override: Record<string, number> | null;
}

interface LapPrediction {
  lap_number: number;
  predicted_time: number;
  actual_time?: number | null;
  compound: string;
  tire_age: number;
  scenario_name: string;
  // Weather
  rainfall?: number | null;
  track_temp?: number | null;
  air_temp?: number | null;
  humidity?: number | null;
  // Error metrics
  error?: number | null;
  absolute_error?: number | null;
}

interface TireStint {
  stint: number;
  compound: string;
  start_lap: number;
  end_lap: number;
  laps: number;
}

interface ScenarioResult {
  scenario_name: string;
  total_race_time: number;
  average_lap_time: number;
  fastest_lap: number;
  slowest_lap: number;
  laps: LapPrediction[];
  tire_strategy: TireStint[];
}

interface AvailableRace {
  year: number;
  race: string;
  drivers: string[];
  laps: number;
}

interface ModelInfo {
  input_features: number;
  total_parameters: number;
  architecture: number[];
  test_rmse: number;
  test_mae: number;
  test_r2: number;
}

// ============================================================================
//                          API SERVICE
// ============================================================================

const API_BASE = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const api = {
  async getHealth() {
    const res = await fetch(`${API_BASE}/`);
    return res.json();
  },

  async getModelInfo(): Promise<ModelInfo> {
    const res = await fetch(`${API_BASE}/model/info`);
    return res.json();
  },

  async getAvailableRaces(): Promise<AvailableRace[]> {
    const res = await fetch(`${API_BASE}/races/available`);
    const data = await res.json();
    return data.races;
  },

  async predict(scenarios: RaceScenario[], comparison_mode: boolean = true) {
    const res = await fetch(`${API_BASE}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ scenarios, comparison_mode }),
    });
    return res.json();
  },
};

// ============================================================================
//                          MAIN COMPONENT
// ============================================================================

export default function F1App() {
  const [tabIndex, setTabIndex] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Model state
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [availableRaces, setAvailableRaces] = useState<AvailableRace[]>([]);

  // Scenario configuration
  const [baseScenario, setBaseScenario] = useState<RaceScenario>({
    year: 2023,
    race: 'Monaco',
    driver: 'VER',
    tire_improvement: 0,
    fuel_load_factor: 1.0,
    weather_override: null,
  });

  const [compareScenario, setCompareScenario] = useState<RaceScenario>({
    year: 2023,
    race: 'Monaco',
    driver: 'VER',
    tire_improvement: 0.25,
    fuel_load_factor: 1.0,
    weather_override: null,
  });

  // Results
  const [results, setResults] = useState<ScenarioResult[]>([]);
  const [comparison, setComparison] = useState<any>(null);

  // Available drivers for selected race
  const [availableDrivers, setAvailableDrivers] = useState<string[]>([]);

  // ============================================================================
  //                          EFFECTS
  // ============================================================================

  useEffect(() => {
    loadInitialData();
  }, []);

  useEffect(() => {
    // Update available drivers when race changes
    const race = availableRaces.find(
      (r) => r.year === baseScenario.year && r.race === baseScenario.race
    );
    if (race) {
      setAvailableDrivers(race.drivers);
      if (!race.drivers.includes(baseScenario.driver)) {
        setBaseScenario((prev) => ({ ...prev, driver: race.drivers[0] }));
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [baseScenario.year, baseScenario.race, availableRaces]);

  useEffect(() => {
    // Auto-fill comparison scenario with same race/driver when base scenario changes
    setCompareScenario((prev) => ({
      ...prev,
      year: baseScenario.year,
      race: baseScenario.race,
      driver: baseScenario.driver,
    }));
  }, [baseScenario.year, baseScenario.race, baseScenario.driver]);

  // ============================================================================
  //                          FUNCTIONS
  // ============================================================================

  const loadInitialData = async () => {
    try {
      setLoading(true);
      const [info, races] = await Promise.all([
        api.getModelInfo(),
        api.getAvailableRaces(),
      ]);
      setModelInfo(info);
      setAvailableRaces(races);
      setError(null);
    } catch (err) {
      setError('Failed to connect to backend. Make sure the API is running on port 8000.');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const runPrediction = async (compareMode: boolean) => {
    try {
      setLoading(true);
      setError(null);

      // Validate selections
      if (!baseScenario.race || !baseScenario.driver) {
        setError('Please select a race and driver first!');
        setLoading(false);
        return;
      }

      if (compareMode && (!compareScenario.race || !compareScenario.driver)) {
        setError('Please configure the comparison scenario!');
        setLoading(false);
        return;
      }

      const scenarios = compareMode
        ? [baseScenario, compareScenario]
        : [baseScenario];

      const data = await api.predict(scenarios, compareMode);

      if (!data || !data.scenarios || data.scenarios.length === 0) {
        setError('No prediction data returned. Check if the race/driver combination exists.');
        setLoading(false);
        return;
      }

      setResults(data.scenarios);
      setComparison(data.comparison);
    } catch (err: any) {
      setError(`Prediction failed: ${err?.message || 'Unknown error'}. Check console for details.`);
      console.error('Prediction error:', err);
    } finally {
      setLoading(false);
    }
  };

  // ============================================================================
  //                          CHART DATA
  // ============================================================================

  const getLapTimeChartData = () => {
    if (!results || results.length === 0) return [];

    try {
      const data: any[] = [];
      const maxLaps = Math.max(...results.map((r) => r.laps?.length || 0));

      for (let i = 0; i < maxLaps; i++) {
        const point: any = { lap: i + 1 };

        // Track stint changes for baseline scenario
        let currentCompound = '';
        let tireAge = 0;
        let stintNumber = 1;

        results.forEach((result, resultIdx) => {
          if (result.laps && i < result.laps.length) {
            const lap = result.laps[i];
            point[result.scenario_name] = lap.predicted_time;

            // Store tire info from baseline scenario (first result)
            if (resultIdx === 0) {
              currentCompound = lap.compound;
              tireAge = lap.tire_age;

              // Calculate stint number by counting compound changes
              stintNumber = 1;
              for (let j = 0; j < i; j++) {
                if (result.laps[j].compound !== result.laps[j > 0 ? j - 1 : 0].compound) {
                  stintNumber++;
                }
              }

              point['compound'] = currentCompound;
              point['tire_age'] = tireAge;
              point['stint'] = stintNumber;
            }

            // Add actual lap time (only for baseline scenario, index 0)
            if (resultIdx === 0 && lap.actual_time !== null && lap.actual_time !== undefined) {
              point['actual'] = lap.actual_time;
            }
          }
        });

        data.push(point);
      }

      return data;
    } catch (err) {
      console.error('Error generating lap time chart data:', err);
      return [];
    }
  };

  const getComparisonChartData = () => {
    if (!results || results.length === 0) return [];

    try {
      return results.map((result) => ({
        scenario: result.scenario_name?.split('|')[1]?.trim() || 'Baseline',
        total_time: result.total_race_time || 0,
        avg_lap: result.average_lap_time || 0,
        fastest: result.fastest_lap || 0,
      }));
    } catch (err) {
      console.error('Error generating comparison chart data:', err);
      return [];
    }
  };

  const getWeatherChartData = () => {
    if (!results || results.length === 0) return [];

    try {
      const baselineLaps = results[0]?.laps || [];
      return baselineLaps.map((lap) => ({
        lap: lap.lap_number,
        rainfall: lap.rainfall || 0,
        track_temp: lap.track_temp || 0,
        air_temp: lap.air_temp || 0,
        humidity: lap.humidity || 0,
      }));
    } catch (err) {
      console.error('Error generating weather chart data:', err);
      return [];
    }
  };

  const getErrorDistributionData = () => {
    if (!results || results.length === 0) return [];

    try {
      const baselineLaps = results[0]?.laps || [];
      const errors = baselineLaps
        .filter((lap) => lap.error !== null && lap.error !== undefined)
        .map((lap) => lap.error!);

      console.log('Error Distribution - Total laps:', baselineLaps.length);
      console.log('Error Distribution - Laps with errors:', errors.length);
      console.log('Error Distribution - Sample errors:', errors.slice(0, 5));

      // Create histogram bins
      if (errors.length === 0) {
        console.warn('No error data available for distribution chart');
        return [];
      }

      const bins = 20;
      const min = Math.min(...errors);
      const max = Math.max(...errors);
      const binSize = (max - min) / bins;

      const histogram: Record<string, number> = {};
      for (let i = 0; i < bins; i++) {
        const binStart = min + i * binSize;
        const binLabel = binStart.toFixed(2);
        histogram[binLabel] = 0;
      }

      errors.forEach((error) => {
        const binIndex = Math.min(Math.floor((error - min) / binSize), bins - 1);
        const binStart = min + binIndex * binSize;
        const binLabel = binStart.toFixed(2);
        histogram[binLabel]++;
      });

      const result = Object.entries(histogram).map(([error, count]) => ({
        error: parseFloat(error),
        count,
      }));

      console.log('Error Distribution - Histogram bins:', result.length);
      return result;
    } catch (err) {
      console.error('Error generating error distribution data:', err);
      return [];
    }
  };

  const getResidualsData = () => {
    if (!results || results.length === 0) return [];

    try {
      const baselineLaps = results[0]?.laps || [];
      const residualsData = baselineLaps
        .filter((lap) => lap.error !== null && lap.error !== undefined)
        .map((lap) => ({
          lap: lap.lap_number,
          error: lap.error!,
        }));

      console.log('Residuals - Total laps:', baselineLaps.length);
      console.log('Residuals - Data points:', residualsData.length);
      console.log('Residuals - Sample data:', residualsData.slice(0, 5));

      return residualsData;
    } catch (err) {
      console.error('Error generating residuals data:', err);
      return [];
    }
  };

  const getTireDegradationData = () => {
    if (!results || results.length === 0) return [];

    try {
      const baselineLaps = results[0]?.laps || [];
      return baselineLaps.map((lap) => ({
        tire_age: lap.tire_age,
        lap_time: lap.actual_time || lap.predicted_time,
        compound: lap.compound,
      }));
    } catch (err) {
      console.error('Error generating tire degradation data:', err);
      return [];
    }
  };

  const getCompoundPerformanceData = () => {
    if (!results || results.length === 0) return [];

    try {
      const baselineLaps = results[0]?.laps || [];
      const compoundStats: Record<string, { sum: number; count: number }> = {};

      baselineLaps.forEach((lap) => {
        const time = lap.actual_time || lap.predicted_time;
        if (!compoundStats[lap.compound]) {
          compoundStats[lap.compound] = { sum: 0, count: 0 };
        }
        compoundStats[lap.compound].sum += time;
        compoundStats[lap.compound].count++;
      });

      return Object.entries(compoundStats).map(([compound, stats]) => ({
        compound,
        avg_lap_time: stats.sum / stats.count,
        laps: stats.count,
      }));
    } catch (err) {
      console.error('Error generating compound performance data:', err);
      return [];
    }
  };

  // ============================================================================
  //                          RENDER
  // ============================================================================

  if (!modelInfo) {
    return (
      <Container maxWidth="lg" sx={{ mt: 4 }}>
        <Box display="flex" justifyContent="center" alignItems="center" minHeight="50vh">
          <CircularProgress />
        </Box>
      </Container>
    );
  }

  return (
    <Container maxWidth="xl" sx={{ py: 4 }}>
      {/* Header */}
      <Box mb={4}>
        <Typography variant="h3" gutterBottom fontWeight="bold">
          🏎️ F1 Tire Wear AI Predictor
        </Typography>
        <Typography variant="subtitle1" color="text.secondary">
          AI-Powered Lap Time Predictions with Scenario Comparison
        </Typography>

        <Box mt={2} display="flex" gap={1} flexWrap="wrap">
          <Chip icon={<SpeedIcon />} label={`RMSE: ${modelInfo.test_rmse.toFixed(3)}s`} color="success" />
          <Chip label={`MAE: ${modelInfo.test_mae.toFixed(3)}s`} />
          <Chip label={`R²: ${modelInfo.test_r2.toFixed(4)}`} />
          <Chip label={`${availableRaces.length} Races Available`} variant="outlined" />
        </Box>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 3 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      {/* Main Content */}
      <Grid container spacing={3}>
        {/* Left Panel - Configuration */}
        <Grid item xs={12} md={4}>
          <Paper elevation={3} sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom fontWeight="bold">
              Scenario Configuration
            </Typography>

            <Tabs value={tabIndex} onChange={(_, v) => setTabIndex(v)} sx={{ mb: 2 }}>
              <Tab label="Base Scenario" />
              <Tab label="Compare" />
            </Tabs>

            {tabIndex === 0 && (
              <ScenarioConfig
                scenario={baseScenario}
                setScenario={setBaseScenario}
                availableRaces={availableRaces}
                availableDrivers={availableDrivers}
                title="Baseline"
              />
            )}

            {tabIndex === 1 && (
              <ScenarioConfig
                scenario={compareScenario}
                setScenario={setCompareScenario}
                availableRaces={availableRaces}
                availableDrivers={availableDrivers}
                title="Comparison"
              />
            )}

            <Divider sx={{ my: 3 }} />

            <Box display="flex" gap={2}>
              {tabIndex === 0 ? (
                <Button
                  fullWidth
                  variant="contained"
                  startIcon={<PlayArrowIcon />}
                  onClick={() => runPrediction(false)}
                  disabled={loading}
                >
                  Run Prediction
                </Button>
              ) : (
                <Button
                  fullWidth
                  variant="contained"
                  color="secondary"
                  startIcon={<CompareArrowsIcon />}
                  onClick={() => runPrediction(true)}
                  disabled={loading}
                >
                  Compare Scenarios
                </Button>
              )}
            </Box>
          </Paper>
        </Grid>

        {/* Right Panel - Results */}
        <Grid item xs={12} md={8}>
          {loading ? (
            <Box display="flex" justifyContent="center" alignItems="center" minHeight="400px">
              <CircularProgress size={60} />
            </Box>
          ) : results.length > 0 ? (
            <>
              {/* Summary Cards */}
              <Grid container spacing={2} sx={{ mb: 3 }}>
                {results.map((result, idx) => (
                  <Grid item xs={12} sm={6} key={idx}>
                    <ResultCard result={result} isBaseline={idx === 0} />
                  </Grid>
                ))}
              </Grid>

              {/* Comparison */}
              {comparison && (
                <Paper elevation={2} sx={{ p: 3, mb: 3, bgcolor: 'success.light' }}>
                  <Typography variant="h6" gutterBottom>
                    📊 Comparison Analysis
                  </Typography>
                  {comparison.comparisons.map((comp: any, idx: number) => (
                    <Box key={idx} sx={{ mt: 2 }}>
                      <Typography variant="body1" fontWeight="bold">
                        {comp.scenario}
                      </Typography>
                      <Typography variant="body2">
                        Time Saved: <strong>{Math.abs(comp.time_saved).toFixed(2)}s</strong> total
                        ({comp.percentage_improvement.toFixed(2)}% improvement)
                      </Typography>
                      <Typography variant="body2">
                        Per Lap: <strong>{Math.abs(comp.average_lap_diff).toFixed(3)}s</strong> faster
                      </Typography>
                    </Box>
                  ))}
                </Paper>
              )}

              {/* Lap Times Chart */}
              <Paper elevation={2} sx={{ p: 3, mb: 3 }}>
                <Typography variant="h6" gutterBottom>
                  Lap Times Progression
                </Typography>
                <ResponsiveContainer width="100%" height={300}>
                  <ComposedChart data={getLapTimeChartData()}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="lap" label={{ value: 'Lap Number', position: 'insideBottom', offset: -5 }} />
                    <YAxis label={{ value: 'Lap Time (s)', angle: -90, position: 'insideLeft' }} />
                    <Tooltip content={<CustomTooltip />} />
                    <Legend />

                    {/* Actual lap times as scatter dots */}
                    <Scatter
                      name="Actual Lap Times"
                      dataKey="actual"
                      fill="#4caf50"
                      shape="circle"
                    />

                    {/* Predicted lap times as lines */}
                    {results.map((result, idx) => (
                      <Line
                        key={idx}
                        name={result.scenario_name.split('|')[1]?.trim() || 'Baseline'}
                        type="monotone"
                        dataKey={result.scenario_name}
                        stroke={idx === 0 ? '#1976d2' : '#f50057'}
                        strokeWidth={2}
                        dot={false}
                      />
                    ))}
                  </ComposedChart>
                </ResponsiveContainer>
              </Paper>

              {/* Comparison Bar Chart */}
              {results.length > 1 && (
                <Paper elevation={2} sx={{ p: 3, mb: 3 }}>
                  <Typography variant="h6" gutterBottom>
                    Performance Comparison
                  </Typography>
                  <ResponsiveContainer width="100%" height={250}>
                    <BarChart data={getComparisonChartData()}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="scenario" />
                      <YAxis />
                      <Tooltip />
                      <Legend />
                      <Bar dataKey="total_time" fill="#1976d2" name="Total Race Time (s)" />
                      <Bar dataKey="avg_lap" fill="#f50057" name="Avg Lap Time (s)" />
                    </BarChart>
                  </ResponsiveContainer>
                </Paper>
              )}

              {/* Additional Analytics - 2x2 Grid */}
              <Grid container spacing={3}>
                {/* Weather Conditions */}
                <Grid item xs={12} md={6}>
                  <Paper elevation={2} sx={{ p: 3 }}>
                    <Typography variant="h6" gutterBottom>
                      Weather & Track Conditions
                    </Typography>
                    <ResponsiveContainer width="100%" height={250}>
                      <LineChart data={getWeatherChartData()}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="lap" label={{ value: 'Lap', position: 'insideBottom', offset: -5 }} />
                        <YAxis />
                        <Tooltip />
                        <Legend />
                        <Line type="monotone" dataKey="rainfall" stroke="#2196f3" name="Rainfall" />
                        <Line type="monotone" dataKey="track_temp" stroke="#ff5722" name="Track Temp (°C)" />
                        <Line type="monotone" dataKey="air_temp" stroke="#ff9800" name="Air Temp (°C)" />
                        <Line type="monotone" dataKey="humidity" stroke="#4caf50" name="Humidity (%)" />
                      </LineChart>
                    </ResponsiveContainer>
                  </Paper>
                </Grid>

                {/* Error Distribution */}
                <Grid item xs={12} md={6}>
                  <Paper elevation={2} sx={{ p: 3 }}>
                    <Typography variant="h6" gutterBottom>
                      Prediction Error Distribution
                    </Typography>
                    <ResponsiveContainer width="100%" height={250}>
                      <BarChart data={getErrorDistributionData()}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis
                          dataKey="error"
                          label={{ value: 'Error (s)', position: 'insideBottom', offset: -5 }}
                          tickFormatter={(val) => val.toFixed(1)}
                        />
                        <YAxis label={{ value: 'Frequency', angle: -90, position: 'insideLeft' }} />
                        <Tooltip />
                        <Bar dataKey="count" fill="#9c27b0" name="Frequency" />
                      </BarChart>
                    </ResponsiveContainer>
                  </Paper>
                </Grid>

                {/* Residuals Plot */}
                <Grid item xs={12} md={6}>
                  <Paper elevation={2} sx={{ p: 3 }}>
                    <Typography variant="h6" gutterBottom>
                      Residuals Plot
                    </Typography>
                    <ResponsiveContainer width="100%" height={250}>
                      <ScatterChart>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis
                          dataKey="lap"
                          type="number"
                          label={{ value: 'Lap Number', position: 'insideBottom', offset: -5 }}
                        />
                        <YAxis
                          dataKey="error"
                          label={{ value: 'Residual Error (s)', angle: -90, position: 'insideLeft' }}
                        />
                        <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                        <Scatter name="Prediction Error" data={getResidualsData()} fill="#e91e63" />
                      </ScatterChart>
                    </ResponsiveContainer>
                  </Paper>
                </Grid>

                {/* Tire Degradation */}
                <Grid item xs={12} md={6}>
                  <Paper elevation={2} sx={{ p: 3 }}>
                    <Typography variant="h6" gutterBottom>
                      Tire Degradation
                    </Typography>
                    <ResponsiveContainer width="100%" height={250}>
                      <ScatterChart>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis
                          dataKey="tire_age"
                          type="number"
                          label={{ value: 'Tire Age (laps)', position: 'insideBottom', offset: -5 }}
                        />
                        <YAxis
                          dataKey="lap_time"
                          label={{ value: 'Lap Time (s)', angle: -90, position: 'insideLeft' }}
                        />
                        <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                        <Scatter name="Lap Time vs Tire Age" data={getTireDegradationData()} fill="#3f51b5" />
                      </ScatterChart>
                    </ResponsiveContainer>
                  </Paper>
                </Grid>

                {/* Compound Performance */}
                <Grid item xs={12}>
                  <Paper elevation={2} sx={{ p: 3 }}>
                    <Typography variant="h6" gutterBottom>
                      Average Performance by Compound
                    </Typography>
                    <ResponsiveContainer width="100%" height={250}>
                      <BarChart data={getCompoundPerformanceData()}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="compound" />
                        <YAxis label={{ value: 'Avg Lap Time (s)', angle: -90, position: 'insideLeft' }} />
                        <Tooltip />
                        <Legend />
                        <Bar dataKey="avg_lap_time" fill="#00bcd4" name="Avg Lap Time (s)" />
                        <Bar dataKey="laps" fill="#8bc34a" name="Number of Laps" />
                      </BarChart>
                    </ResponsiveContainer>
                  </Paper>
                </Grid>
              </Grid>
            </>
          ) : (
            <Paper elevation={2} sx={{ p: 5, textAlign: 'center' }}>
              <TrendingDownIcon sx={{ fontSize: 80, color: 'text.secondary', mb: 2 }} />
              <Typography variant="h6" color="text.secondary">
                Configure a scenario and click "Run" to see predictions
              </Typography>
            </Paper>
          )}
        </Grid>
      </Grid>
    </Container>
  );
}

// ============================================================================
//                          SUB-COMPONENTS
// ============================================================================

function CustomTooltip({ active, payload, label }: any) {
  if (!active || !payload || payload.length === 0) return null;

  const data = payload[0].payload;

  // Tire compound colors
  const compoundColors: Record<string, string> = {
    SOFT: '#ff0000',
    MEDIUM: '#ffff00',
    HARD: '#ffffff',
    INTERMEDIATE: '#00ff00',
    WET: '#0000ff',
  };

  const compound = data.compound || 'UNKNOWN';
  const tireColor = compoundColors[compound] || '#cccccc';

  return (
    <Box
      sx={{
        bgcolor: 'background.paper',
        border: '1px solid #ccc',
        borderRadius: 1,
        p: 1.5,
        boxShadow: 2,
      }}
    >
      <Typography variant="subtitle2" fontWeight="bold" gutterBottom>
        Lap {label}
      </Typography>

      {/* Tire Information */}
      <Box display="flex" alignItems="center" gap={1} mb={1}>
        <Box
          sx={{
            width: 16,
            height: 16,
            borderRadius: '50%',
            bgcolor: tireColor,
            border: '2px solid black',
          }}
        />
        <Typography variant="body2">
          <strong>{compound}</strong> Compound
        </Typography>
      </Box>

      <Typography variant="body2" sx={{ mb: 0.5 }}>
        Tire Age: <strong>{data.tire_age || 0} laps</strong>
      </Typography>

      <Typography variant="body2" sx={{ mb: 1 }}>
        Stint: <strong>#{data.stint || 1}</strong>
      </Typography>

      <Divider sx={{ my: 1 }} />

      {/* Lap Times */}
      {payload.map((entry: any, index: number) => {
        if (entry.dataKey === 'actual' && entry.value) {
          return (
            <Typography key={index} variant="body2" sx={{ color: entry.fill }}>
              <strong>Actual:</strong> {entry.value.toFixed(3)}s
            </Typography>
          );
        } else if (entry.dataKey !== 'actual' && entry.value) {
          const scenarioName = entry.name || entry.dataKey.split('|')[1]?.trim() || 'Predicted';
          return (
            <Typography key={index} variant="body2" sx={{ color: entry.stroke }}>
              <strong>{scenarioName}:</strong> {entry.value.toFixed(3)}s
            </Typography>
          );
        }
        return null;
      })}
    </Box>
  );
}

function ScenarioConfig({
  scenario,
  setScenario,
  availableRaces,
  availableDrivers,
  title,
}: {
  scenario: RaceScenario;
  setScenario: (s: RaceScenario) => void;
  availableRaces: AvailableRace[];
  availableDrivers: string[];
  title: string;
}) {
  const years = [...new Set(availableRaces.map((r) => r.year))].sort((a, b) => b - a);
  const races = availableRaces.filter((r) => r.year === scenario.year);

  return (
    <Box>
      <Typography variant="subtitle2" gutterBottom color="primary">
        {title}
      </Typography>

      <FormControl fullWidth sx={{ mb: 2 }}>
        <InputLabel>Year</InputLabel>
        <Select
          value={scenario.year}
          label="Year"
          onChange={(e) => setScenario({ ...scenario, year: Number(e.target.value) })}
        >
          {years.map((year) => (
            <MenuItem key={year} value={year}>
              {year}
            </MenuItem>
          ))}
        </Select>
      </FormControl>

      <FormControl fullWidth sx={{ mb: 2 }}>
        <InputLabel>Race</InputLabel>
        <Select
          value={scenario.race}
          label="Race"
          onChange={(e) => setScenario({ ...scenario, race: e.target.value })}
        >
          {races.map((race) => (
            <MenuItem key={race.race} value={race.race}>
              {race.race}
            </MenuItem>
          ))}
        </Select>
      </FormControl>

      <FormControl fullWidth sx={{ mb: 3 }}>
        <InputLabel>Driver</InputLabel>
        <Select
          value={scenario.driver}
          label="Driver"
          onChange={(e) => setScenario({ ...scenario, driver: e.target.value })}
        >
          {availableDrivers.map((driver) => (
            <MenuItem key={driver} value={driver}>
              {driver}
            </MenuItem>
          ))}
        </Select>
      </FormControl>

      <Typography gutterBottom>
        Tire Improvement: {(scenario.tire_improvement * 100).toFixed(0)}%
      </Typography>
      <Slider
        value={scenario.tire_improvement}
        onChange={(_, v) => setScenario({ ...scenario, tire_improvement: v as number })}
        min={0}
        max={0.5}
        step={0.05}
        marks
        valueLabelDisplay="auto"
        valueLabelFormat={(v) => `${(v * 100).toFixed(0)}%`}
        sx={{ mb: 3 }}
      />

      <Typography gutterBottom>
        Fuel Load Factor: {scenario.fuel_load_factor.toFixed(2)}x
      </Typography>
      <Slider
        value={scenario.fuel_load_factor}
        onChange={(_, v) => setScenario({ ...scenario, fuel_load_factor: v as number })}
        min={0.8}
        max={1.2}
        step={0.05}
        marks
        valueLabelDisplay="auto"
        sx={{ mb: 2 }}
      />
    </Box>
  );
}

function ResultCard({ result, isBaseline }: { result: ScenarioResult; isBaseline: boolean }) {
  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = (seconds % 60).toFixed(2);
    return `${mins}:${secs.padStart(5, '0')}`;
  };

  return (
    <Card elevation={isBaseline ? 1 : 3} sx={{ bgcolor: isBaseline ? 'grey.100' : 'primary.light' }}>
      <CardContent>
        <Typography variant="h6" gutterBottom noWrap>
          {result.scenario_name.split('|')[1]?.trim() || 'Baseline'}
        </Typography>
        <Divider sx={{ my: 1 }} />
        <Typography variant="body2">
          <strong>Total Time:</strong> {formatTime(result.total_race_time)}
        </Typography>
        <Typography variant="body2">
          <strong>Avg Lap:</strong> {result.average_lap_time.toFixed(3)}s
        </Typography>
        <Typography variant="body2">
          <strong>Fastest:</strong> {result.fastest_lap.toFixed(3)}s
        </Typography>
        <Typography variant="body2">
          <strong>Slowest:</strong> {result.slowest_lap.toFixed(3)}s
        </Typography>
        <Typography variant="body2" sx={{ mt: 1 }}>
          <strong>Stints:</strong> {result.tire_strategy.length}
        </Typography>
      </CardContent>
    </Card>
  );
}
