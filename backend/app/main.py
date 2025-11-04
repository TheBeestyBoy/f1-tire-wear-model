"""
FastAPI application for RNN text generation.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from text_generator import TextGenerator
from models import (
    GenerateRequest,
    GenerateResponse,
    ModelInfo,
    HealthResponse,
    TestMetrics,
    AvailableModel,
    AvailableModelsResponse,
    SwitchModelRequest,
    SwitchModelResponse
)
import os
from pathlib import Path
from contextlib import asynccontextmanager

# Global generator instance and current model
generator = None
current_model_name = None
MODEL_DIR = "saved_models"
DEFAULT_MODEL_NAME = "model"

def get_available_models():
    """Scan for available models in the saved_models directory."""
    models = []
    if not os.path.exists(MODEL_DIR):
        return models

    # Find all .pt files
    for file in os.listdir(MODEL_DIR):
        if file.endswith('.pt'):
            model_name = file.replace('.pt', '')
            tokenizer_name = file.replace('.pt', '.pkl')

            # Handle special case for tokenizer names
            if model_name.startswith('model_optimal'):
                tokenizer_name = f"tokenizer_optimal_{model_name.split('_')[-1]}.pkl"
            else:
                tokenizer_name = tokenizer_name.replace('model', 'tokenizer')

            model_path = os.path.join(MODEL_DIR, file)
            tokenizer_path = os.path.join(MODEL_DIR, tokenizer_name)
            config_path = model_path.replace('.pt', '_config.json')

            # Only include if both model and tokenizer exist
            if os.path.exists(tokenizer_path):
                # Create display name
                display_name = model_name.replace('_', ' ').title()

                models.append(AvailableModel(
                    name=model_name,
                    display_name=display_name,
                    model_path=model_path,
                    tokenizer_path=tokenizer_path,
                    config_path=config_path
                ))

    return models

def load_model_by_name(model_name: str):
    """Load a model by its name."""
    global generator, current_model_name

    available_models = get_available_models()
    model_info = next((m for m in available_models if m.name == model_name), None)

    if model_info is None:
        raise ValueError(f"Model '{model_name}' not found")

    if not os.path.exists(model_info.model_path):
        raise ValueError(f"Model file not found: {model_info.model_path}")

    if not os.path.exists(model_info.tokenizer_path):
        raise ValueError(f"Tokenizer file not found: {model_info.tokenizer_path}")

    # Load the model
    generator = TextGenerator()
    generator.load_model(model_info.model_path, model_info.tokenizer_path)
    current_model_name = model_name

    print(f"[OK] Loaded model: {model_name}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown."""
    # Startup
    global generator, current_model_name
    try:
        # Try to load default model on startup
        available_models = get_available_models()
        if available_models:
            # Try to load default model, or first available model
            default_model = next((m for m in available_models if m.name == DEFAULT_MODEL_NAME), None)
            if default_model:
                load_model_by_name(DEFAULT_MODEL_NAME)
            else:
                # Load first available model
                load_model_by_name(available_models[0].name)
        else:
            print("[WARNING] No models found. Train a model first.")
    except Exception as e:
        print(f"[ERROR] Error loading model: {e}")

    yield

    # Shutdown (cleanup if needed)
    pass

# Initialize FastAPI app
app = FastAPI(
    title="RNN Text Generator API",
    description="Generate text using LSTM neural networks",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware for React frontend
# Get allowed origins from environment variable, default to all for development
allowed_origins = os.getenv("ALLOWED_ORIGINS", "*").split(",")
if allowed_origins == ["*"]:
    allowed_origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=generator is not None
    )

@app.get("/model/info", response_model=ModelInfo)
async def get_model_info():
    """Get model information."""
    if generator is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Calculate total trainable parameters (neurons) in the model
    total_neurons = sum(p.numel() for p in generator.model.parameters() if p.requires_grad)

    return ModelInfo(
        vocab_size=generator.vocab_size,
        sequence_length=generator.sequence_length,
        embedding_dim=generator.embedding_dim,
        lstm_units=generator.lstm_units,
        num_layers=generator.num_layers,
        total_neurons=total_neurons
    )

@app.post("/generate", response_model=GenerateResponse)
async def generate_text(request: GenerateRequest):
    """Generate text from seed using sampling or beam search."""
    if generator is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        generated = generator.generate_text(
            seed_text=request.seed_text,
            num_words=request.num_words,
            temperature=request.temperature,
            use_beam_search=request.use_beam_search,
            beam_width=request.beam_width,
            length_penalty=request.length_penalty,
            repetition_penalty=request.repetition_penalty,
            beam_temperature=request.beam_temperature,
            add_punctuation=request.add_punctuation,
            validate_grammar=request.validate_grammar
        )

        return GenerateResponse(
            generated_text=generated,
            seed_text=request.seed_text,
            num_words=request.num_words,
            temperature=request.temperature,
            use_beam_search=request.use_beam_search,
            beam_width=request.beam_width if request.use_beam_search else None,
            length_penalty=request.length_penalty if request.use_beam_search else None,
            repetition_penalty=request.repetition_penalty if request.use_beam_search else None,
            beam_temperature=request.beam_temperature if request.use_beam_search else None,
            add_punctuation=request.add_punctuation,
            validate_grammar=request.validate_grammar
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/visualizations/architecture")
async def get_architecture():
    """Get model architecture diagram."""
    path = "visualizations/model_architecture.png"
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Architecture diagram not found")
    return FileResponse(path)

@app.get("/visualizations/training")
async def get_training_history():
    """Get training history plot."""
    path = "visualizations/training_history.png"
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Training history not found")
    return FileResponse(path)

@app.get("/model/test", response_model=TestMetrics)
async def test_model(use_beam_search: bool = True, beam_width: int = 5):
    """Test model and return metrics using beam search or greedy decoding.

    Args:
        use_beam_search: If True, uses beam search for predictions (default: True)
        beam_width: Number of beams for beam search (default: 5)
    """
    if generator is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Try to find available test data files
        data_dir = "data"
        possible_files = [
            "training_text.txt",
            "kjv.txt",
            "web.txt",
            "net.txt",
            "asv.txt"
        ]

        test_data_path = None
        for filename in possible_files:
            path = os.path.join(data_dir, filename)
            if os.path.exists(path):
                test_data_path = path
                break

        if test_data_path is None:
            # If no specific file found, try to find any .txt file in data directory
            if os.path.exists(data_dir):
                txt_files = [f for f in os.listdir(data_dir) if f.endswith('.txt')]
                if txt_files:
                    test_data_path = os.path.join(data_dir, txt_files[0])

        if test_data_path is None or not os.path.exists(test_data_path):
            raise HTTPException(status_code=404, detail="No test data found. Please ensure data files exist in the 'data' directory.")

        # Load and prepare test data
        with open(test_data_path, 'r', encoding='utf-8') as f:
            text = f.read()

        # Use a portion of data for testing (last 10%)
        words = text.split()
        test_size = len(words) // 10
        test_text = ' '.join(words[-test_size:])

        # Prepare sequences
        X, y, _ = generator.prepare_sequences(test_text)

        # Evaluate model with beam search
        metrics = generator.evaluate_model(X, y, use_beam_search=use_beam_search, beam_width=beam_width)

        return TestMetrics(
            test_loss=metrics['test_loss'],
            test_accuracy=metrics['test_accuracy'],
            perplexity=metrics['perplexity'],
            samples_tested=metrics['samples_tested'],
            r_squared=metrics['r_squared']
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models/available", response_model=AvailableModelsResponse)
async def list_available_models():
    """List all available models."""
    try:
        models = get_available_models()
        return AvailableModelsResponse(
            models=models,
            current_model=current_model_name
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/models/switch", response_model=SwitchModelResponse)
async def switch_model(request: SwitchModelRequest):
    """Switch to a different model."""
    try:
        load_model_by_name(request.model_name)
        return SwitchModelResponse(
            success=True,
            message=f"Successfully switched to model: {request.model_name}",
            model_name=request.model_name
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
