import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export interface GenerateRequest {
  seed_text: string;
  num_words: number;
  temperature: number;
  use_beam_search?: boolean;
  beam_width?: number;
  length_penalty?: number;
  repetition_penalty?: number;
  beam_temperature?: number;
  add_punctuation?: boolean;
  validate_grammar?: boolean;
}

export interface GenerateResponse {
  generated_text: string;
  seed_text: string;
  num_words: number;
  temperature: number;
  use_beam_search: boolean;
  beam_width?: number;
  length_penalty?: number;
  repetition_penalty?: number;
  beam_temperature?: number;
  add_punctuation: boolean;
  validate_grammar: boolean;
}

export interface ModelInfo {
  vocab_size: number;
  sequence_length: number;
  embedding_dim: number;
  lstm_units: number;
  num_layers: number;
  total_neurons: number;
}

export interface HealthResponse {
  status: string;
  model_loaded: boolean;
}

export interface TestMetrics {
  test_loss: number;
  test_accuracy: number;
  perplexity: number;
  samples_tested: number;
  r_squared?: number;
}

export interface AvailableModel {
  name: string;
  display_name: string;
  model_path: string;
  tokenizer_path: string;
  config_path: string;
}

export interface AvailableModelsResponse {
  models: AvailableModel[];
  current_model: string | null;
}

export interface SwitchModelRequest {
  model_name: string;
}

export interface SwitchModelResponse {
  success: boolean;
  message: string;
  model_name: string;
}

export const healthCheck = async (): Promise<HealthResponse> => {
  const response = await api.get<HealthResponse>('/');
  return response.data;
};

export const testModel = async (): Promise<TestMetrics> => {
  const response = await api.get<TestMetrics>('/model/test');
  return response.data;
};

export const generateText = async (request: GenerateRequest): Promise<GenerateResponse> => {
  const response = await api.post<GenerateResponse>('/generate', request);
  return response.data;
};

export const getModelInfo = async (): Promise<ModelInfo> => {
  const response = await api.get<ModelInfo>('/model/info');
  return response.data;
};

export const getArchitectureImage = (): string => {
  return `${API_BASE_URL}/visualizations/architecture`;
};

export const getTrainingHistoryImage = (): string => {
  return `${API_BASE_URL}/visualizations/training`;
};

export const getAvailableModels = async (): Promise<AvailableModelsResponse> => {
  const response = await api.get<AvailableModelsResponse>('/models/available');
  return response.data;
};

export const switchModel = async (modelName: string): Promise<SwitchModelResponse> => {
  const response = await api.post<SwitchModelResponse>('/models/switch', {
    model_name: modelName,
  });
  return response.data;
};
