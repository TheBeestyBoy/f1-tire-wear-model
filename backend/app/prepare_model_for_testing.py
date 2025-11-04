"""
Helper script to prepare model files for testing.

This recreates the tokenizer and config from your training data
so you can test your existing model_best.pt checkpoint.
"""
from text_generator import TextGenerator
import os
import shutil
import sys

def main():
    """Prepare model files for testing."""

    DATA_PATH = "data/training_text.txt"
    MODEL_DIR = "saved_models"

    print("=" * 70)
    print("MODEL PREPARATION FOR TESTING")
    print("=" * 70)

    # Check if training data exists
    if not os.path.exists(DATA_PATH):
        print(f"[ERROR] Training data not found at {DATA_PATH}")
        print("Cannot recreate tokenizer without training data.")
        print("\nRECOMMENDATION: Run full training instead:")
        print("  python app/train.py")
        sys.exit(1)

    # Check if model_best.pt exists
    if not os.path.exists(f"{MODEL_DIR}/model_best.pt"):
        print(f"[ERROR] model_best.pt not found in {MODEL_DIR}")
        print("You need to train the model first:")
        print("  python app/train.py")
        sys.exit(1)

    print("\nFound:")
    print(f"  ✓ Training data: {DATA_PATH}")
    print(f"  ✓ Model checkpoint: {MODEL_DIR}/model_best.pt")

    # Load training data
    print("\nLoading training data...")
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        text = f.read()

    print(f"Loaded {len(text)} characters")

    # Initialize generator with same parameters as training
    print("\nInitializing text generator...")
    generator = TextGenerator(
        sequence_length=50,
        embedding_dim=100,
        lstm_units=150,
        num_layers=2,
        dropout_rate=0.2
    )

    # Prepare sequences (this creates the tokenizer)
    print("\nPreparing sequences to build tokenizer...")
    X, y, max_seq_len = generator.prepare_sequences(text)

    # Build model
    print("\nBuilding model architecture...")
    generator.build_model()

    # Copy model_best.pt to model.pt (this is what the API expects)
    print("\nCopying model_best.pt to model.pt...")
    shutil.copy(f"{MODEL_DIR}/model_best.pt", f"{MODEL_DIR}/model.pt")

    # Save tokenizer and config
    print("\nSaving tokenizer and configuration...")
    generator.save_model(
        f"{MODEL_DIR}/model.pt",
        f"{MODEL_DIR}/tokenizer.pkl"
    )

    # Verify files exist
    print("\n" + "=" * 70)
    print("VERIFICATION")
    print("=" * 70)

    files_to_check = [
        "model.pt",
        "tokenizer.pkl",
        "model_config.json"
    ]

    all_exist = True
    for filename in files_to_check:
        filepath = f"{MODEL_DIR}/{filename}"
        if os.path.exists(filepath):
            size = os.path.getsize(filepath)
            print(f"  ✓ {filename}: {size:,} bytes")
        else:
            print(f"  ✗ {filename}: MISSING")
            all_exist = False

    if all_exist:
        print("\n" + "=" * 70)
        print("SUCCESS! Model is ready for testing.")
        print("=" * 70)
        print("\nNext steps:")
        print("  1. Start the backend:")
        print("     python app/main.py")
        print("\n  2. In a new terminal, start the frontend:")
        print("     cd ../frontend")
        print("     npm start")
        print("\n  3. Open http://localhost:3000 in your browser")
        print("=" * 70)
    else:
        print("\n" + "=" * 70)
        print("ERROR: Some files are missing")
        print("=" * 70)
        print("\nRECOMMENDATION: Run full training instead:")
        print("  python app/train.py")
        sys.exit(1)

if __name__ == "__main__":
    main()
