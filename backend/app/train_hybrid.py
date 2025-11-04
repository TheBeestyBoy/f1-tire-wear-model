"""
HYBRID Training script for RNN text generator.

This version implements the hybrid approach for 60-70% accuracy:
- More training data (15 classic books + 8 Bibles)
- Optimized architecture (200 units, 3 layers)
- Better regularization (0.3 dropout, gradient clipping)
- Improved training parameters

EXPECTED RESULTS:
- Validation Accuracy: 60-70% (vs current 37%)
- Training Time: Moderate per epoch
- Text Quality: Much better and more coherent
- Better generalization with more data
"""
from text_generator_limited_vocab import LimitedVocabTextGenerator
import os
import sys
import pickle

def main():
    """Main training pipeline with hybrid hyperparameters."""

    # Configuration
    DATA_PATH = "data/training_text.txt"
    VOCAB_PATH = "data/vocab_15000.pkl"
    MODEL_DIR = "saved_models"
    VIZ_DIR = "visualizations"

    # Create directories
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(VIZ_DIR, exist_ok=True)

    # Check if training data exists
    if not os.path.exists(DATA_PATH):
        print(f"[ERROR] Training data not found at {DATA_PATH}")
        print("Please run:")
        print("  1. python scripts/download_training_books.py")
        print("  2. python scripts/combine_bibles.py")
        sys.exit(1)

    # Check if vocabulary file exists
    if not os.path.exists(VOCAB_PATH):
        print(f"[ERROR] Vocabulary file not found at {VOCAB_PATH}")
        print("Please run:")
        print("  python scripts/extract_vocabulary.py")
        sys.exit(1)

    # Load training data
    print("Loading training data...")
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        text = f.read()

    print(f"Loaded {len(text)} characters")
    print(f"Unique words: {len(set(text.split()))}")

    # Load pre-extracted vocabulary
    print(f"\nLoading pre-extracted vocabulary from {VOCAB_PATH}...")
    with open(VOCAB_PATH, 'rb') as f:
        vocab_data = pickle.load(f)

    print(f"Vocabulary size: {vocab_data['vocab_size']}")
    print(f"Token coverage: {vocab_data['coverage']:.2f}%")

    # Initialize generator with HYBRID hyperparameters
    print("\n" + "=" * 70)
    print("HYBRID CONFIGURATION (Target: 60-70% Accuracy)")
    print("=" * 70)
    print("IMPROVEMENTS OVER OPTIMAL:")
    print("  ✓ Training data: ~31M → ~100M+ characters (3x MORE DATA)")
    print("  ✓ LSTM units: 256 → 200 (better balance)")
    print("  ✓ LSTM layers: 3 (kept)")
    print("  ✓ Embedding dim: 128 (kept)")
    print("  ✓ Dropout: 0.2 → 0.3 (better regularization)")
    print("  ✓ Batch size: 512 → 256 (better generalization)")
    print("  ✓ Gradient clipping added (prevents exploding gradients)")
    print("  ✓ More aggressive early stopping")
    print()
    print("DATA SOURCES:")
    print("  • 8 Bible translations")
    print("  • 5 Shakespeare plays")
    print("  • Paradise Lost & Paradise Regained (Milton)")
    print("  • Divine Comedy (Dante)")
    print("  • Pilgrim's Progress (Bunyan)")
    print("  • Classic philosophy texts")
    print("  • Religious works (Augustine, Thomas à Kempis)")
    print()
    print("EXPECTED RESULTS:")
    print("  • Validation Accuracy: 60-70%")
    print("  • Training Time: ~15-16 min/epoch")
    print("  • Much better text generation quality")
    print("  • Better understanding of archaic/formal language")
    print("=" * 70)

    generator = LimitedVocabTextGenerator(
        max_vocab_size=15000,   # *** Using pre-extracted top 15000 words ***
        sequence_length=50,
        embedding_dim=128,      # Rich word representations
        lstm_units=200,         # *** Reduced for better data fit ***
        num_layers=3,           # Deep enough to learn patterns
        dropout_rate=0.3        # *** Increased for regularization ***
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

    # Calculate data-to-parameter ratio
    num_sequences = X.shape[0]
    ratio = num_sequences / total_params
    print(f"\nData sequences: {num_sequences:,}")
    print(f"Parameters: {total_params:,}")
    print(f"Sequences per parameter: {ratio:.2f}x")

    if ratio < 10:
        print("[WARNING] Data-to-parameter ratio is low!")
        print("          Consider adding more training data for best results.")
    elif ratio >= 10 and ratio < 20:
        print("[OK] Data-to-parameter ratio is acceptable.")
    else:
        print("[EXCELLENT] Data-to-parameter ratio is optimal!")

    # Compare with previous configurations
    print("\n" + "=" * 70)
    print("PARAMETER COMPARISON:")
    print(f"  Baseline (51k vocab, 150 units):  ~8.4M parameters")
    print(f"  Optimal (20k vocab, 256 units):   ~9.1M parameters")
    print(f"  Hybrid (15k vocab, 200 units):    ~{total_params/1e6:.1f}M parameters")
    print(f"  Reduction from Optimal: {((total_params/9.1e6 - 1) * 100):.1f}%")
    print("=" * 70)

    # Visualize architecture
    print("\nGenerating architecture visualization...")
    generator.visualize_architecture(save_path=VIZ_DIR)

    # Train model with HYBRID parameters
    print("\nTraining model...")
    print("=" * 70)
    history = generator.train(
        X, y,
        epochs=100,
        batch_size=256,         # *** Reduced for better generalization ***
        validation_split=0.1,
        save_path=MODEL_DIR
    )

    # Plot training history
    print("\nGenerating training plots...")
    generator.plot_training_history(save_path=VIZ_DIR)

    # Save final model
    print("\nSaving model...")
    generator.save_model(
        f"{MODEL_DIR}/model_hybrid.pt",
        f"{MODEL_DIR}/tokenizer_hybrid.pkl"
    )

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
    best_val_acc = max(history['val_accuracy'])*100
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Best validation loss: {min(history['val_loss']):.4f}")
    print(f"Total epochs trained: {len(history['loss'])}")
    print()
    print("COMPARISON WITH PREVIOUS RUNS:")
    print(f"  Baseline Val Acc:  28.52%")
    print(f"  Optimal Val Acc:   ~37-40% (estimated at elbow)")
    print(f"  Hybrid Val Acc:    {best_val_acc:.2f}%")
    if best_val_acc >= 60:
        print(f"  [SUCCESS] Target achieved! ({best_val_acc:.2f}% ≥ 60%)")
    else:
        print(f"  Improvement:       {(best_val_acc - 28.52):+.2f} percentage points")
    print()
    print("MODEL FILES SAVED:")
    print(f"  • {MODEL_DIR}/model_hybrid.pt")
    print(f"  • {MODEL_DIR}/tokenizer_hybrid.pkl")
    print(f"  • {MODEL_DIR}/best_model.pt (best checkpoint)")
    print("="*70)

if __name__ == "__main__":
    main()
