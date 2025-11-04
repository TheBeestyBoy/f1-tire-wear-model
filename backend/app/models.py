"""
Pydantic models for FastAPI request/response validation.
"""
from pydantic import BaseModel, Field
from typing import Optional

class GenerateRequest(BaseModel):
    """Request model for text generation."""
    seed_text: str = Field(..., min_length=1, description="Starting text")
    num_words: int = Field(50, ge=10, le=200, description="Number of words to generate")
    temperature: float = Field(1.0, ge=0.1, le=5.0, description="Sampling temperature (ignored if use_beam_search=True)")
    use_beam_search: bool = Field(False, description="Use beam search instead of sampling")
    beam_width: int = Field(5, ge=1, le=10, description="Number of beams for beam search")
    length_penalty: float = Field(1.0, ge=0.1, le=2.0, description="Length penalty for beam search")
    repetition_penalty: float = Field(1.2, ge=1.0, le=3.0, description="Repetition penalty for beam search (reduces loops)")
    beam_temperature: float = Field(0.0, ge=0.0, le=2.0, description="Randomness in beam search (0.0=deterministic, 0.5-1.0=varied)")
    add_punctuation: bool = Field(False, description="Add punctuation and capitalization post-processing")
    validate_grammar: bool = Field(False, description="Validate grammar during generation (slower but more grammatical)")

class GenerateResponse(BaseModel):
    """Response model for text generation."""
    generated_text: str
    seed_text: str
    num_words: int
    temperature: float
    use_beam_search: bool
    beam_width: Optional[int] = None
    length_penalty: Optional[float] = None
    repetition_penalty: Optional[float] = None
    beam_temperature: Optional[float] = None
    add_punctuation: bool = False
    validate_grammar: bool = False

class ModelInfo(BaseModel):
    """Model information."""
    vocab_size: int
    sequence_length: int
    embedding_dim: int
    lstm_units: int
    num_layers: int
    total_neurons: int

class HealthResponse(BaseModel):
    """Health check response."""
    model_config = {"protected_namespaces": ()}

    status: str
    model_loaded: bool

class TestMetrics(BaseModel):
    """Model testing metrics."""
    test_loss: float
    test_accuracy: float
    perplexity: float
    samples_tested: int
    r_squared: Optional[float] = None

class AvailableModel(BaseModel):
    """Information about an available model."""
    name: str
    display_name: str
    model_path: str
    tokenizer_path: str
    config_path: str

class AvailableModelsResponse(BaseModel):
    """Response containing list of available models."""
    models: list[AvailableModel]
    current_model: Optional[str] = None

class SwitchModelRequest(BaseModel):
    """Request to switch to a different model."""
    model_name: str = Field(..., min_length=1, description="Name of the model to switch to")

class SwitchModelResponse(BaseModel):
    """Response after switching models."""
    success: bool
    message: str
    model_name: str
