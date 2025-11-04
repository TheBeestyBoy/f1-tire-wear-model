"""
TextGenerator with limited vocabulary for better accuracy.
This version can use pre-extracted vocabulary from pickle files.
"""
from text_generator import TextGenerator, SimpleTokenizer
from collections import Counter
import pickle
import os

class LimitedVocabTokenizer(SimpleTokenizer):
    """Tokenizer with vocabulary size limiting or pre-loaded vocabulary."""

    def __init__(self, max_vocab_size=20000, vocab_file=None):
        super().__init__()
        self.max_vocab_size = max_vocab_size
        self.vocab_file = vocab_file
        self.preloaded_vocab = None

        # Load pre-extracted vocabulary if file is provided
        if vocab_file and os.path.exists(vocab_file):
            print(f"Loading pre-extracted vocabulary from {vocab_file}...")
            with open(vocab_file, 'rb') as f:
                vocab_data = pickle.load(f)
                self.preloaded_vocab = vocab_data['vocabulary']
                print(f"Loaded {len(self.preloaded_vocab)} words from vocabulary file")
                print(f"Token coverage: {vocab_data.get('coverage', 'N/A'):.2f}%")

    def fit_on_texts(self, texts):
        """Build vocabulary from texts with size limit or use preloaded vocab."""
        if isinstance(texts, str):
            texts = [texts]

        words = []
        for text in texts:
            words.extend(text.split())

        # Count word frequencies
        word_counts = Counter(words)

        # Build vocabulary (most common first)
        # Add special tokens
        self.word_to_idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx_to_word = {0: '<PAD>', 1: '<UNK>'}

        # Use preloaded vocabulary if available, otherwise extract from text
        if self.preloaded_vocab:
            print(f"Using pre-extracted vocabulary ({len(self.preloaded_vocab)} words)")
            vocab_words = self.preloaded_vocab
        else:
            vocab_words = [word for word, _ in word_counts.most_common(self.max_vocab_size - 2)]

        # Add vocabulary words
        for idx, word in enumerate(vocab_words[:self.max_vocab_size - 2], start=2):
            self.word_to_idx[word] = idx
            self.idx_to_word[idx] = word

        self.vocab_size = len(self.word_to_idx)

        # Calculate coverage
        covered_count = sum(word_counts[word] for word in self.word_to_idx.keys() if word in word_counts)
        total_count = sum(word_counts.values())
        coverage = (covered_count / total_count * 100) if total_count > 0 else 0

        print(f"Vocabulary size: {self.vocab_size} words (from {len(word_counts)} unique words)")
        print(f"Coverage: {coverage:.2f}% of all word tokens")


class LimitedVocabTextGenerator(TextGenerator):
    """Text generator with limited vocabulary for better accuracy."""

    def __init__(
        self,
        max_vocab_size=20000,
        sequence_length=50,
        embedding_dim=128,
        lstm_units=256,
        num_layers=3,
        dropout_rate=0.2,
        vocab_file=None
    ):
        super().__init__(
            sequence_length=sequence_length,
            embedding_dim=embedding_dim,
            lstm_units=lstm_units,
            num_layers=num_layers,
            dropout_rate=dropout_rate
        )
        # Replace tokenizer with limited vocab version
        self.tokenizer = LimitedVocabTokenizer(max_vocab_size=max_vocab_size, vocab_file=vocab_file)
        self.max_vocab_size = max_vocab_size
        self.vocab_file = vocab_file
