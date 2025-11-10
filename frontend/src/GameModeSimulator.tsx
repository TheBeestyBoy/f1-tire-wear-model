import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Slider,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Button,
  Grid,
  Chip,
  Stack,
  Paper,
  Alert,
  CircularProgress,
  Tabs,
  Tab,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
} from '@mui/material';
import {
  Speed as SpeedIcon,
  Thermostat as ThermostatIcon,
  WaterDrop as WaterDropIcon,
  Air as AirIcon,
  Timer as TimerIcon,
} from '@mui/icons-material';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import axios from 'axios';

const API_BASE = 'http://localhost:8000';

interface SimulateResponse {
  predicted_lap_time: number;
  configuration: any;
  message: string;
}

interface StintLap {
  lap_number: number;
  lap_time: number;
  tire_age: number;
}

interface StintResponse {
  laps: StintLap[];
  summary: {
    total_laps: number;
    total_time: number;
    average_lap_time: number;
    fastest_lap: number;
    slowest_lap: number;
    degradation: number;
  };
}

interface TrackData {
  name: string;
  avg_lap_time: number;
  min_lap_time: number;
  max_lap_time: number;
  total_laps: number;
  rmse: number | null;
  mae: number | null;
  r2: number | null;
}

const WEATHER_PRESETS = {
  DRY: { label: 'Dry', color: '#FFD700', icon: '☀️' },
  DRIZZLE: { label: 'Drizzle', color: '#87CEEB', icon: '🌦️' },
  RAIN: { label: 'Rain', color: '#4682B4', icon: '🌧️' },
  HEAVY_RAIN: { label: 'Heavy Rain', color: '#191970', icon: '⛈️' },
  HOT: { label: 'Hot', color: '#FF4500', icon: '🔥' },
  COLD: { label: 'Cold', color: '#00CED1', icon: '❄️' },
};

const TIRE_COMPOUNDS = {
  SOFT: { label: 'Soft', color: '#FF0000' },
  MEDIUM: { label: 'Medium', color: '#FFD700' },
  HARD: { label: 'Hard', color: '#FFFFFF' },
  INTERMEDIATE: { label: 'Intermediate', color: '#00FF00' },
  WET: { label: 'Wet', color: '#0000FF' },
};

export const GameModeSimulator: React.FC = () => {
  const [mode, setMode] = useState<'single' | 'stint'>('single');

  // Track state
  const [tracks, setTracks] = useState<TrackData[]>([]);
  const [selectedTrack, setSelectedTrack] = useState<string>('Monaco Grand Prix');
  const [tracksLoading, setTracksLoading] = useState(true);

  // Configuration state
  const [compound, setCompound] = useState<string>('MEDIUM');
  const [tireAge, setTireAge] = useState<number>(15);
  const [trackTemp, setTrackTemp] = useState<number>(30);
  const [airTemp, setAirTemp] = useState<number>(25);
  const [rainfall, setRainfall] = useState<number>(0);
  const [humidity, setHumidity] = useState<number>(50);
  const [lapNumber, setLapNumber] = useState<number>(30);
  const [fuelLoad, setFuelLoad] = useState<number>(0.5);
  const [stintLaps, setStintLaps] = useState<number>(20);

  // Results state
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [singleLapResult, setSingleLapResult] = useState<SimulateResponse | null>(null);
  const [stintResult, setStintResult] = useState<StintResponse | null>(null);

  // Load tracks on mount
  useEffect(() => {
    const loadTracks = async () => {
      try {
        const response = await axios.get(`${API_BASE}/simulate/tracks`);
        setTracks(response.data.tracks);
        setTracksLoading(false);
      } catch (err) {
        console.error('Failed to load tracks:', err);
        setTracksLoading(false);
      }
    };

    loadTracks();
  }, []);

  const handleWeatherPreset = async (preset: string) => {
    setLoading(true);
    setError(null);

    try {
      const response = await axios.post(`${API_BASE}/simulate/weather`, {
        weather: preset,
        compound,
        tire_age: tireAge,
        lap_number: lapNumber,
      });

      // Update sliders with preset values
      const conditions = response.data.weather_conditions;
      setTrackTemp(conditions.track_temp);
      setAirTemp(conditions.air_temp);
      setRainfall(conditions.rainfall);
      setHumidity(conditions.humidity);

      // Show result
      setSingleLapResult({
        predicted_lap_time: response.data.predicted_lap_time,
        configuration: response.data.configuration,
        message: `Applied ${preset} preset`,
      });
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to apply weather preset');
    } finally {
      setLoading(false);
    }
  };

  const handleSimulateLap = async () => {
    setLoading(true);
    setError(null);
    setSingleLapResult(null);

    try {
      const response = await axios.post<SimulateResponse>(`${API_BASE}/simulate`, {
        track: selectedTrack,
        compound,
        tire_age: tireAge,
        track_temp: trackTemp,
        air_temp: airTemp,
        rainfall,
        humidity,
        lap_number: lapNumber,
        fuel_load: fuelLoad,
        add_noise: false,
      });

      setSingleLapResult(response.data);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to simulate lap');
    } finally {
      setLoading(false);
    }
  };

  const handleSimulateStint = async () => {
    setLoading(true);
    setError(null);
    setStintResult(null);

    try {
      const response = await axios.post<StintResponse>(`${API_BASE}/simulate/stint`, {
        laps: stintLaps,
        track: selectedTrack,
        compound,
        tire_age: tireAge,
        track_temp: trackTemp,
        air_temp: airTemp,
        rainfall,
        humidity,
        lap_number: lapNumber,
        fuel_load: fuelLoad,
      });

      setStintResult(response.data);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to simulate stint');
    } finally {
      setLoading(false);
    }
  };

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = (seconds % 60).toFixed(2);
    return mins > 0 ? `${mins}:${secs.padStart(5, '0')}` : `${secs}s`;
  };

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom sx={{ fontWeight: 'bold', color: '#e10600' }}>
        🎮 Game Mode Simulator
      </Typography>
      <Typography variant="body1" color="text.secondary" gutterBottom sx={{ mb: 3 }}>
        Configure track conditions and tire parameters to simulate lap times
      </Typography>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      <Grid container spacing={3}>
        {/* Left Column - Configuration */}
        <Grid item xs={12} md={6}>
          <Card elevation={3}>
            <CardContent>
              <Typography variant="h6" gutterBottom sx={{ fontWeight: 'bold' }}>
                Configuration
              </Typography>

              {/* Mode Toggle */}
              <Tabs value={mode} onChange={(e, v) => setMode(v)} sx={{ mb: 3 }}>
                <Tab label="Single Lap" value="single" />
                <Tab label="Stint Simulation" value="stint" />
              </Tabs>

              {/* Weather Presets */}
              <Typography variant="subtitle2" gutterBottom sx={{ mt: 2 }}>
                Quick Weather Presets
              </Typography>
              <Stack direction="row" spacing={1} flexWrap="wrap" sx={{ mb: 3 }}>
                {Object.entries(WEATHER_PRESETS).map(([key, preset]) => (
                  <Chip
                    key={key}
                    label={`${preset.icon} ${preset.label}`}
                    onClick={() => handleWeatherPreset(key)}
                    sx={{
                      backgroundColor: preset.color,
                      color: key === 'HARD' ? '#000' : '#fff',
                      fontWeight: 'bold',
                      mb: 1,
                    }}
                  />
                ))}
              </Stack>

              {/* Track Selector */}
              <FormControl fullWidth sx={{ mb: 2 }}>
                <InputLabel>Track</InputLabel>
                <Select
                  value={selectedTrack}
                  onChange={(e) => setSelectedTrack(e.target.value)}
                  disabled={tracksLoading}
                >
                  {tracks.map((track) => (
                    <MenuItem key={track.name} value={track.name}>
                      <Box sx={{ display: 'flex', flexDirection: 'column' }}>
                        <Typography variant="body2">{track.name}</Typography>
                        <Typography variant="caption" color="text.secondary">
                          Avg: {track.avg_lap_time.toFixed(2)}s
                          {track.rmse !== null && ` | RMSE: ${track.rmse.toFixed(2)}s`}
                        </Typography>
                      </Box>
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>

              {/* Track Info */}
              {selectedTrack && tracks.length > 0 && (() => {
                const track = tracks.find(t => t.name === selectedTrack);
                return track ? (
                  <Alert severity="info" sx={{ mb: 2 }}>
                    <Typography variant="body2">
                      <strong>{track.name}</strong>
                    </Typography>
                    <Typography variant="caption">
                      Average lap time: {track.avg_lap_time.toFixed(2)}s |
                      Range: {track.min_lap_time.toFixed(2)}s - {track.max_lap_time.toFixed(2)}s
                      {track.rmse !== null && (
                        <> | Model RMSE: {track.rmse.toFixed(2)}s ({track.rmse < 1 ? 'EXCELLENT' : track.rmse < 2 ? 'VERY GOOD' : 'GOOD'})</>
                      )}
                    </Typography>
                  </Alert>
                ) : null;
              })()}

              {/* Tire Compound */}
              <FormControl fullWidth sx={{ mb: 2 }}>
                <InputLabel>Tire Compound</InputLabel>
                <Select value={compound} onChange={(e) => setCompound(e.target.value)}>
                  {Object.entries(TIRE_COMPOUNDS).map(([key, tire]) => (
                    <MenuItem key={key} value={key}>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <Box
                          sx={{
                            width: 16,
                            height: 16,
                            borderRadius: '50%',
                            backgroundColor: tire.color,
                            border: key === 'HARD' ? '1px solid #000' : 'none',
                          }}
                        />
                        {tire.label}
                      </Box>
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>

              {/* Tire Age */}
              <Box sx={{ mb: 3 }}>
                <Typography gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <SpeedIcon fontSize="small" />
                  Tire Age: {tireAge} laps
                </Typography>
                <Slider
                  value={tireAge}
                  onChange={(e, v) => setTireAge(v as number)}
                  min={0}
                  max={50}
                  marks={[
                    { value: 0, label: 'Fresh' },
                    { value: 25, label: '25' },
                    { value: 50, label: 'Worn' },
                  ]}
                  valueLabelDisplay="auto"
                />
              </Box>

              {/* Track Temperature */}
              <Box sx={{ mb: 3 }}>
                <Typography gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <ThermostatIcon fontSize="small" />
                  Track Temperature: {trackTemp}°C
                </Typography>
                <Slider
                  value={trackTemp}
                  onChange={(e, v) => setTrackTemp(v as number)}
                  min={0}
                  max={60}
                  marks={[
                    { value: 0, label: '0°C' },
                    { value: 30, label: '30°C' },
                    { value: 60, label: '60°C' },
                  ]}
                  valueLabelDisplay="auto"
                />
              </Box>

              {/* Air Temperature */}
              <Box sx={{ mb: 3 }}>
                <Typography gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <AirIcon fontSize="small" />
                  Air Temperature: {airTemp}°C
                </Typography>
                <Slider
                  value={airTemp}
                  onChange={(e, v) => setAirTemp(v as number)}
                  min={0}
                  max={50}
                  marks={[
                    { value: 0, label: '0°C' },
                    { value: 25, label: '25°C' },
                    { value: 50, label: '50°C' },
                  ]}
                  valueLabelDisplay="auto"
                />
              </Box>

              {/* Rainfall */}
              <Box sx={{ mb: 3 }}>
                <Typography gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <WaterDropIcon fontSize="small" />
                  Rainfall: {rainfall}mm
                </Typography>
                <Slider
                  value={rainfall}
                  onChange={(e, v) => setRainfall(v as number)}
                  min={0}
                  max={50}
                  marks={[
                    { value: 0, label: 'Dry' },
                    { value: 10, label: 'Rain' },
                    { value: 50, label: 'Heavy' },
                  ]}
                  valueLabelDisplay="auto"
                />
              </Box>

              {/* Humidity */}
              <Box sx={{ mb: 3 }}>
                <Typography gutterBottom>Humidity: {humidity}%</Typography>
                <Slider
                  value={humidity}
                  onChange={(e, v) => setHumidity(v as number)}
                  min={0}
                  max={100}
                  valueLabelDisplay="auto"
                />
              </Box>

              {/* Fuel Load */}
              <Box sx={{ mb: 3 }}>
                <Typography gutterBottom>Fuel Load: {(fuelLoad * 100).toFixed(0)}%</Typography>
                <Slider
                  value={fuelLoad}
                  onChange={(e, v) => setFuelLoad(v as number)}
                  min={0}
                  max={1}
                  step={0.05}
                  marks={[
                    { value: 0, label: 'Empty' },
                    { value: 0.5, label: '50%' },
                    { value: 1, label: 'Full' },
                  ]}
                  valueLabelDisplay="auto"
                  valueLabelFormat={(v) => `${(v * 100).toFixed(0)}%`}
                />
              </Box>

              {/* Lap Number */}
              <Box sx={{ mb: 3 }}>
                <Typography gutterBottom>Lap Number: {lapNumber}</Typography>
                <Slider
                  value={lapNumber}
                  onChange={(e, v) => setLapNumber(v as number)}
                  min={1}
                  max={100}
                  valueLabelDisplay="auto"
                />
              </Box>

              {/* Stint Laps (only in stint mode) */}
              {mode === 'stint' && (
                <Box sx={{ mb: 3 }}>
                  <Typography gutterBottom>Stint Length: {stintLaps} laps</Typography>
                  <Slider
                    value={stintLaps}
                    onChange={(e, v) => setStintLaps(v as number)}
                    min={5}
                    max={50}
                    marks={[
                      { value: 5, label: '5' },
                      { value: 20, label: '20' },
                      { value: 50, label: '50' },
                    ]}
                    valueLabelDisplay="auto"
                  />
                </Box>
              )}

              {/* Simulate Button */}
              <Button
                variant="contained"
                size="large"
                fullWidth
                onClick={mode === 'single' ? handleSimulateLap : handleSimulateStint}
                disabled={loading}
                sx={{
                  backgroundColor: '#e10600',
                  '&:hover': { backgroundColor: '#b00500' },
                  fontWeight: 'bold',
                }}
              >
                {loading ? (
                  <CircularProgress size={24} sx={{ color: 'white' }} />
                ) : (
                  `Simulate ${mode === 'single' ? 'Lap' : 'Stint'}`
                )}
              </Button>
            </CardContent>
          </Card>
        </Grid>

        {/* Right Column - Results */}
        <Grid item xs={12} md={6}>
          {mode === 'single' && singleLapResult && (
            <Card elevation={3}>
              <CardContent>
                <Typography variant="h6" gutterBottom sx={{ fontWeight: 'bold' }}>
                  Single Lap Result
                </Typography>

                <Paper elevation={2} sx={{ p: 3, backgroundColor: '#f5f5f5', textAlign: 'center', mb: 2 }}>
                  <Typography variant="h3" sx={{ fontWeight: 'bold', color: '#e10600' }}>
                    {formatTime(singleLapResult.predicted_lap_time)}
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    Predicted Lap Time
                  </Typography>
                </Paper>

                <Alert severity="info" sx={{ mb: 2 }}>
                  {singleLapResult.message}
                </Alert>

                <Typography variant="subtitle2" gutterBottom>
                  Configuration:
                </Typography>
                <TableContainer component={Paper} variant="outlined">
                  <Table size="small">
                    <TableBody>
                      <TableRow>
                        <TableCell>Compound</TableCell>
                        <TableCell align="right">{compound}</TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell>Tire Age</TableCell>
                        <TableCell align="right">{tireAge} laps</TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell>Track Temp</TableCell>
                        <TableCell align="right">{trackTemp}°C</TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell>Rainfall</TableCell>
                        <TableCell align="right">{rainfall}mm</TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell>Humidity</TableCell>
                        <TableCell align="right">{humidity}%</TableCell>
                      </TableRow>
                    </TableBody>
                  </Table>
                </TableContainer>
              </CardContent>
            </Card>
          )}

          {mode === 'stint' && stintResult && (
            <Card elevation={3}>
              <CardContent>
                <Typography variant="h6" gutterBottom sx={{ fontWeight: 'bold' }}>
                  Stint Simulation Results
                </Typography>

                {/* Summary Stats */}
                <Grid container spacing={2} sx={{ mb: 3 }}>
                  <Grid item xs={6}>
                    <Paper sx={{ p: 2, textAlign: 'center' }}>
                      <Typography variant="h6" sx={{ fontWeight: 'bold' }}>
                        {formatTime(stintResult.summary.average_lap_time)}
                      </Typography>
                      <Typography variant="caption">Avg Lap Time</Typography>
                    </Paper>
                  </Grid>
                  <Grid item xs={6}>
                    <Paper sx={{ p: 2, textAlign: 'center' }}>
                      <Typography variant="h6" sx={{ fontWeight: 'bold' }}>
                        {stintResult.summary.degradation.toFixed(2)}s
                      </Typography>
                      <Typography variant="caption">Degradation</Typography>
                    </Paper>
                  </Grid>
                  <Grid item xs={6}>
                    <Paper sx={{ p: 2, textAlign: 'center' }}>
                      <Typography variant="h6" sx={{ fontWeight: 'bold', color: 'green' }}>
                        {formatTime(stintResult.summary.fastest_lap)}
                      </Typography>
                      <Typography variant="caption">Fastest Lap</Typography>
                    </Paper>
                  </Grid>
                  <Grid item xs={6}>
                    <Paper sx={{ p: 2, textAlign: 'center' }}>
                      <Typography variant="h6" sx={{ fontWeight: 'bold', color: 'red' }}>
                        {formatTime(stintResult.summary.slowest_lap)}
                      </Typography>
                      <Typography variant="caption">Slowest Lap</Typography>
                    </Paper>
                  </Grid>
                </Grid>

                {/* Lap Times Chart */}
                <Typography variant="subtitle2" gutterBottom>
                  Lap-by-Lap Performance
                </Typography>
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={stintResult.laps}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="lap_number" label={{ value: 'Lap Number', position: 'insideBottom', offset: -5 }} />
                    <YAxis label={{ value: 'Lap Time (s)', angle: -90, position: 'insideLeft' }} />
                    <Tooltip formatter={(value: number) => formatTime(value)} />
                    <Legend />
                    <Line type="monotone" dataKey="lap_time" stroke="#e10600" strokeWidth={2} name="Lap Time" />
                  </LineChart>
                </ResponsiveContainer>

                {/* Lap Data Table */}
                <Typography variant="subtitle2" gutterBottom sx={{ mt: 2 }}>
                  Detailed Data (showing every 5 laps)
                </Typography>
                <TableContainer component={Paper} variant="outlined" sx={{ maxHeight: 300 }}>
                  <Table size="small" stickyHeader>
                    <TableHead>
                      <TableRow>
                        <TableCell>Lap</TableCell>
                        <TableCell align="right">Time</TableCell>
                        <TableCell align="right">Tire Age</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {stintResult.laps.filter((_, idx) => idx % 5 === 0).map((lap) => (
                        <TableRow key={lap.lap_number}>
                          <TableCell>{lap.lap_number}</TableCell>
                          <TableCell align="right">{formatTime(lap.lap_time)}</TableCell>
                          <TableCell align="right">{lap.tire_age} laps</TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              </CardContent>
            </Card>
          )}

          {!singleLapResult && !stintResult && (
            <Card elevation={3}>
              <CardContent sx={{ textAlign: 'center', py: 8 }}>
                <TimerIcon sx={{ fontSize: 64, color: '#ccc', mb: 2 }} />
                <Typography variant="h6" color="text.secondary">
                  Configure parameters and click Simulate
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Results will appear here
                </Typography>
              </CardContent>
            </Card>
          )}
        </Grid>
      </Grid>
    </Box>
  );
};
