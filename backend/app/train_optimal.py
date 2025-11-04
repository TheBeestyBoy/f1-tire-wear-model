"""
OPTIMAL Training script for RNN text generator.

This version combines all improvements:
- Limited vocabulary (20,000 words) for higher accuracy
- Larger LSTM architecture (256 units, 3 layers)
- Progress bars and timing for each epoch
- Better hyperparameters

EXPECTED RESULTS:
- Validation Accuracy: 40-50% (vs current 28%)
- Training Time: Faster per epoch due to smaller vocabulary
- Text Quality: Better and more coherent
"""
from text_generator_limited_vocab import LimitedVocabTextGenerator
import os
import sys
import time
from datetime import datetime

def main():
    """Main training pipeline with optimal hyperparameters."""

    # Configuration
    DATA_PATH = "data/training_text.txt"
    VOCAB_PATH = "data/vocab_10000.pkl"  # Pre-extracted vocabulary (10k words)
    MODEL_DIR = "saved_models"
    VIZ_DIR = "visualizations"

    # Create directories
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(VIZ_DIR, exist_ok=True)

    # Check if vocabulary file exists
    if not os.path.exists(VOCAB_PATH):
        print(f"[WARNING] Vocabulary file not found at {VOCAB_PATH}")
        print("Run: python scripts/extract_vocabulary.py")
        VOCAB_PATH = None

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

    # Initialize generator with OPTIMAL hyperparameters
    print("\n" + "=" * 70)
    print("CONFIGURATION V8 - SMALL + OneCycleLR (MODERN APPROACH)")
    print("=" * 70)
    print("VOCABULARY:")
    print(f"  ✓ Using: {VOCAB_PATH if VOCAB_PATH else 'Dynamic extraction'}")
    print("  ✓ Size: 10,000 words (94.4% token coverage!)")
    print("  ✓ Unknown tokens: Only 5.6% (vs 9.8% with 5k vocab)")
    print("  ✓ Preprocessing: MATCHES vocab extraction (contractions intact)")
    print()
    print("ARCHITECTURE (SMALL CONFIG - SAFE):")
    print("  ✓ Sequence length: 50 tokens")
    print("  ✓ Embedding dim: 128")
    print("  ✓ LSTM units: 256")
    print("  ✓ LSTM layers: 2 with Layer Normalization")
    print("  ✓ Dropout: 0.25")
    print("  ✓ Parameters: ~4.8M (4.7x data-to-param ratio)")
    print("  ✓ Strategy: Start safe, then try medium config")
    print()
    print("TRAINING:")
    print("  ✓ Learning rate: OneCycleLR (Modern approach)")
    print("  ✓ Max LR: 0.003 (peak learning rate)")
    print("  ✓ Schedule: Warmup for 30% of training, then decay")
    print("  ✓ Strategy: Gradual increase → peak → smooth decrease")
    print("  ✓ AdamW optimizer (weight decay 0.005)")
    print("  ✓ Batch size: 512")
    print("  ✓ Epochs: 45")
    print("=" * 70)

    generator = LimitedVocabTextGenerator(
        max_vocab_size=10000,   # 10k vocab for 94% coverage
        sequence_length=50,
        embedding_dim=128,      # *** SMALL CONFIG - 4.8M params, 4.7x ratio ***
        lstm_units=256,         # *** SMALL CONFIG - safe for 22.7M sequences ***
        num_layers=2,
        dropout_rate=0.25,
        vocab_file=VOCAB_PATH
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

    # Compare with baseline
    print("\n" + "=" * 70)
    print("PARAMETER COMPARISON:")
    print(f"  Baseline (51k vocab, 150 units):  ~8.4M parameters")
    print(f"  Optimal (5k vocab, 256 units):    ~{total_params/1e6:.1f}M parameters")
    print(f"  Difference: {((total_params/8.4e6 - 1) * 100):+.1f}%")
    print(f"  Output layer reduction: 51,823 → 5,000 nodes (90.4% reduction)")
    print("=" * 70)

    # Visualize architecture
    print("\nGenerating architecture visualization...")
    generator.visualize_architecture(save_path=VIZ_DIR)

    # Train model
    print("\nTraining model...")
    print("=" * 70)
    history = generator.train(
        X, y,
        epochs=45,          # *** EXTENDED - more LR drops for accuracy jumps ***
        batch_size=512,     # *** SMALLER for better gradient estimates ***
        validation_split=0.1,
        save_path=MODEL_DIR
    )

    # Plot training history
    print("\nGenerating training plots...")
    generator.plot_training_history(save_path=VIZ_DIR)

    # Save final model with validation accuracy and timestamp in filename
    print("\nSaving model...")
    best_val_accuracy = max(history['val_accuracy'])
    val_acc_percent = int(best_val_accuracy * 100)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"model_val_acc_{val_acc_percent}_{timestamp}"

    generator.save_model(
        f"{MODEL_DIR}/{model_name}.pt",
        f"{MODEL_DIR}/tokenizer_{model_name.replace('model_', '')}.pkl"
    )

    print(f"Model saved as: {model_name}.pt (Val Accuracy: {val_acc_percent}%)")

    # Test generation
    print("\n" + "="*70)
    print("Testing text generation...")
    print("="*70)

    seed_text = " ".join(text.split()[:10])
    print(f"\nSeed text: '{seed_text}'")

    for temp in [0.5, 1.0, 1.5]:
        print(f"\n--- Temperature: {temp} ---")
        generated = generator.generate_text(seed_text, num_words=50, temperature=temp)
        print(generated)

    # Final summary
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"Best validation accuracy: {max(history['val_accuracy'])*100:.2f}%")
    print(f"Best validation loss: {min(history['val_loss']):.4f}")
    print(f"Total epochs trained: {len(history['loss'])}")
    print()
    print("COMPARISON WITH YOUR BASELINE (30 epochs):")
    print(f"  Baseline Val Acc:  28.52%")
    print(f"  Optimal Val Acc:   {max(history['val_accuracy'])*100:.2f}%")
    print(f"  Improvement:       {(max(history['val_accuracy']) - 0.2852)*100:+.2f} percentage points")
    print("="*70)

if __name__ == "__main__":
    main()
