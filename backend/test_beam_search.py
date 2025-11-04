"""
Test script to compare sampling vs beam search generation.
"""
import sys
import os

# Add the backend directory to path so we can import app modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.text_generator import TextGenerator

def main():
    MODEL_DIR = "saved_models"

    # Try to find an available model
    model_path = None
    tokenizer_path = None

    # Priority list of model names to try
    model_names = ["model", "model_best", "model_val_acc_21", "model_val_acc_20"]

    for model_name in model_names:
        test_model = f"{MODEL_DIR}/{model_name}.pt"
        test_tokenizer = f"{MODEL_DIR}/tokenizer.pkl"

        if os.path.exists(test_model) and os.path.exists(test_tokenizer):
            model_path = test_model
            tokenizer_path = test_tokenizer
            print(f"[INFO] Using model: {model_name}.pt")
            break

    # Check if model exists
    if not model_path:
        print("[ERROR] No trained model found. Please train a model first.")
        print("Available models:")
        if os.path.exists(MODEL_DIR):
            for file in os.listdir(MODEL_DIR):
                if file.endswith('.pt'):
                    print(f"  - {file}")
        print("\nMake sure tokenizer.pkl exists in saved_models/")
        return

    print("Loading model...")
    generator = TextGenerator()

    try:
        generator.load_model(model_path, tokenizer_path)
        print("[OK] Model loaded successfully\n")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return

    # Test seed text
    seed_text = "In the beginning"
    num_words = 30

    print("="*70)
    print("COMPARISON: Sampling vs Beam Search")
    print("="*70)
    print(f"Seed text: '{seed_text}'")
    print(f"Words to generate: {num_words}\n")

    # Test 1: Temperature Sampling (original method)
    print("--- METHOD 1: Temperature Sampling (temperature=0.8) ---")
    sampling_result = generator.generate_text(
        seed_text=seed_text,
        num_words=num_words,
        temperature=0.8,
        use_beam_search=False
    )
    print(sampling_result)
    print()

    # Test 2: Beam Search with beam_width=3
    print("--- METHOD 2: Beam Search (beam_width=3) ---")
    beam_result_3 = generator.generate_text(
        seed_text=seed_text,
        num_words=num_words,
        use_beam_search=True,
        beam_width=3,
        length_penalty=1.0
    )
    print(beam_result_3)
    print()

    # Test 3: Beam Search with beam_width=5
    print("--- METHOD 3: Beam Search (beam_width=5) ---")
    beam_result_5 = generator.generate_text(
        seed_text=seed_text,
        num_words=num_words,
        use_beam_search=True,
        beam_width=5,
        length_penalty=1.0
    )
    print(beam_result_5)
    print()

    # Test 4: Beam Search with length penalty
    print("--- METHOD 4: Beam Search (beam_width=5, length_penalty=1.5) ---")
    beam_result_penalty = generator.generate_text(
        seed_text=seed_text,
        num_words=num_words,
        use_beam_search=True,
        beam_width=5,
        length_penalty=1.5
    )
    print(beam_result_penalty)
    print()

    print("="*70)
    print("KEY OBSERVATIONS:")
    print("- Sampling: Random, varies each run")
    print("- Beam Search: Deterministic, finds most probable sequence")
    print("- Higher beam_width: More exploration, potentially better quality")
    print("- Length penalty: Adjusts preference for longer/shorter sequences")
    print("="*70)

if __name__ == "__main__":
    main()
