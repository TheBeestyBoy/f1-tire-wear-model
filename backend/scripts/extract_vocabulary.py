"""
Extract vocabulary from training_text.txt and keep top 15000 most frequent words.

This script:
1. Reads training_text.txt
2. Counts word frequencies
3. Selects top 15000 most frequent words
4. Saves the vocabulary to a file
"""
import os
from collections import Counter
import pickle
import re

def main():
    # Paths
    DATA_PATH = "data/training_text.txt"
    OUTPUT_PATH = "data/vocab_10000.pkl"

    # Check if training data exists
    if not os.path.exists(DATA_PATH):
        print(f"[ERROR] Training data not found at {DATA_PATH}")
        print("Please ensure training_text.txt exists in the data directory.")
        return

    print("=" * 70)
    print("VOCABULARY EXTRACTION - TOP 10000 WORDS")
    print("=" * 70)

    # Read training data
    print(f"\nReading training data from {DATA_PATH}...")
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        text = f.read()

    print(f"Loaded {len(text):,} characters")

    # Tokenize: extract words while handling punctuation properly
    # This regex:
    # - Extracts alphabetic words
    # - Keeps contractions like "don't" or "it's" intact
    # - Converts to lowercase for consistency
    # - Ignores standalone punctuation
    print("\nTokenizing text (removing punctuation, converting to lowercase)...")
    words = re.findall(r'\b[a-zA-Z]+(?:\'[a-z]+)?\b', text.lower())
    print(f"Total words (tokens): {len(words):,}")

    # Count word frequencies
    print("\nCounting word frequencies...")
    word_counts = Counter(words)
    unique_words = len(word_counts)
    print(f"Unique words found: {unique_words:,}")

    # Get top 10000 most frequent words
    top_n = 10000
    if unique_words < top_n:
        print(f"\n[WARNING] Only {unique_words:,} unique words found (less than {top_n:,})")
        print(f"Will use all {unique_words:,} words as vocabulary.")
        top_n = unique_words

    most_common = word_counts.most_common(top_n)
    vocab_words = [word for word, count in most_common]

    print(f"\nExtracted top {len(vocab_words):,} most frequent words")

    # Display statistics
    print("\n" + "=" * 70)
    print("VOCABULARY STATISTICS")
    print("=" * 70)
    print(f"Total words in text:        {len(words):,}")
    print(f"Unique words in text:       {unique_words:,}")
    print(f"Vocabulary size:            {len(vocab_words):,}")
    print(f"Coverage:                   {len(vocab_words)/unique_words*100:.2f}% of unique words")

    # Calculate how many tokens are covered
    total_occurrences = sum(count for word, count in most_common)
    coverage_ratio = total_occurrences / len(words) * 100
    print(f"Token coverage:             {coverage_ratio:.2f}% of all tokens")

    # Show top 20 most frequent words
    print("\nTop 20 most frequent words:")
    for i, (word, count) in enumerate(most_common[:20], 1):
        print(f"  {i:2d}. '{word}' - {count:,} occurrences")

    # Show frequency distribution
    print("\nFrequency distribution:")
    ranges = [(1, 10), (11, 100), (101, 1000), (1001, 10000)]
    for start, end in ranges:
        count = sum(1 for _, freq in word_counts.items() if start <= freq <= end)
        print(f"  Words appearing {start:5d}-{end:5d} times: {count:,}")

    count_once = sum(1 for _, freq in word_counts.items() if freq == 1)
    print(f"  Words appearing only once:           {count_once:,}")

    # Save vocabulary
    print(f"\nSaving vocabulary to {OUTPUT_PATH}...")
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    vocab_data = {
        'vocabulary': vocab_words,
        'word_counts': {word: count for word, count in most_common},
        'total_words': len(words),
        'unique_words': unique_words,
        'vocab_size': len(vocab_words),
        'coverage': coverage_ratio
    }

    with open(OUTPUT_PATH, 'wb') as f:
        pickle.dump(vocab_data, f)

    print(f"[SUCCESS] Vocabulary saved to {OUTPUT_PATH}")
    print("\n" + "=" * 70)
    print("COMPLETE!")
    print("=" * 70)
    print(f"\nYou can now use this vocabulary in your training script.")
    print(f"The vocabulary covers {coverage_ratio:.2f}% of all word tokens in the training data.")

if __name__ == "__main__":
    main()
