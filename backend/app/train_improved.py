"""
Training script for RNN text generator with improved hyperparameters.

This version includes:
- Larger LSTM units (256 instead of 150)
- 3 LSTM layers instead of 2
- Vocabulary limiting for better accuracy
- Progress bars and timing for each epoch
"""
from text_generator import TextGenerator
import os
import sys

def main():
    """Main training pipeline with improved hyperparameters."""

    # Configuration
    DATA_PATH = "data/training_text.txt"
    MODEL_DIR = "saved_models"
    VIZ_DIR = "visualizations"

    # Create directories
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(VIZ_DIR, exist_ok=True)

    # Check if training data exists
    if not os.path.exists(DATA_PATH):
        print(f"[ERROR] Training data not found at {DATA_PATH}")
        print("Please run: python scripts/download_data.py")
        sys.exit(1)

    # Load training data
    print("Loading training data...")
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        text = f.read()

    print(f"Loaded {len(text)} characters")
    print(f"Unique words: {len(set(text.split()))}")

    # Initialize generator with IMPROVED hyperparameters
    print("\nInitializing text generator with improved architecture...")
    print("=" * 60)
    print("IMPROVEMENTS:")
    print("  - LSTM units: 150 → 256 (more capacity)")
    print("  - LSTM layers: 2 → 3 (deeper network)")
    print("  - Embedding dim: 100 → 128 (richer representations)")
    print("  - Progress bars and timing added")
    print("=" * 60)

    generator = TextGenerator(
        sequence_length=50,
        embedding_dim=128,      # Increased from 100
        lstm_units=256,         # Increased from 150
        num_layers=3,           # Increased from 2
        dropout_rate=0.2
    )

    # Prepare sequences
    print("\nPreparing training sequences...")
    X, y, max_seq_len = generator.prepare_sequences(text)

    # Build model
    print("\nBuilding model...")
    model = generator.build_model()
    print(model)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Visualize architecture
    print("\nGenerating architecture visualization...")
    generator.visualize_architecture(save_path=VIZ_DIR)

    # Train model
    print("\nTraining model...")
    print("=" * 60)
    history = generator.train(
        X, y,
        epochs=100,
        batch_size=512,
        validation_split=0.1,
        save_path=MODEL_DIR
    )

    # Plot training history
    print("\nGenerating training plots...")
    generator.plot_training_history(save_path=VIZ_DIR)

    # Save final model
    print("\nSaving model...")
    generator.save_model(
        f"{MODEL_DIR}/model_improved.pt",
        f"{MODEL_DIR}/tokenizer_improved.pkl"
    )

    # Test generation
    print("\n" + "="*60)
    print("Testing text generation...")
    print("="*60)

    seed_text = " ".join(text.split()[:10])
    print(f"\nSeed text: '{seed_text}'")

    for temp in [0.5, 1.0, 1.5]:
        print(f"\n--- Temperature: {temp} ---")
        generated = generator.generate_text(seed_text, num_words=50, temperature=temp)
        print(generated)

    print("\n[OK] Training complete!")
    print(f"Best validation accuracy: {max(history['val_accuracy'])*100:.2f}%")
    print(f"Best validation loss: {min(history['val_loss']):.4f}")

if __name__ == "__main__":
    main()
