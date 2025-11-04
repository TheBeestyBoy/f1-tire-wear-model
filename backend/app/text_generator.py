"""
TextGenerator class for RNN-based text generation using PyTorch LSTM.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
import re
import json
from typing import List, Tuple, Dict, Optional
from collections import Counter
import time

# Optional imports for visualization (not needed for API)
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    plt = None
    sns = None

# Optional import for progress bars
try:
    from tqdm import tqdm
except ImportError:
    # Fallback to simple range if tqdm not available
    tqdm = lambda x, **kwargs: x

# Global variables to cache models (loaded on first use)
_punctuation_model = None
_grammar_validator = None


class GrammarValidator:
    """
    Validates grammatical structure of generated sequences using POS tagging.

    This validator checks for basic grammatical patterns:
    - Sentences should have at least one verb
    - Proper noun-verb agreement patterns
    - No repeated POS tags (e.g., multiple determiners in a row)
    - Balanced sentence structure
    """

    def __init__(self):
        """Initialize the grammar validator with spaCy."""
        try:
            import spacy
            print("[INFO] Loading grammar validation model (first time only)...")
            self.nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])  # Only need POS
            self.enabled = True
            print("[INFO] Grammar validator loaded successfully!")
        except Exception as e:
            print(f"[WARNING] Failed to load grammar validator: {e}")
            self.nlp = None
            self.enabled = False

    def validate_sequence(self, text: str, partial: bool = False) -> Tuple[bool, float]:
        """
        Validate grammatical structure of text.

        Args:
            text: Text to validate
            partial: If True, allows incomplete sentences (for mid-generation)

        Returns:
            Tuple of (is_valid, grammar_score)
            - is_valid: Boolean indicating if grammar is acceptable
            - grammar_score: Float from 0.0 to 1.0 indicating quality
        """
        if not self.enabled or not text or not text.strip():
            return True, 1.0

        try:
            doc = self.nlp(text)
            if len(doc) == 0:
                return True, 1.0

            pos_tags = [token.pos_ for token in doc]
            score = 1.0

            # Check 1: Should have at least one verb (unless partial)
            has_verb = any(pos in ['VERB', 'AUX'] for pos in pos_tags)
            if not partial and not has_verb and len(pos_tags) > 5:
                score *= 0.3  # Heavy penalty for no verb in complete sentence

            # Check 2: Penalize repeated determiners/pronouns
            for i in range(len(pos_tags) - 1):
                if pos_tags[i] == pos_tags[i+1] and pos_tags[i] in ['DET', 'PRON']:
                    score *= 0.7  # Penalty for repeated function words

            # Check 3: Check for basic sentence structure patterns
            # Good patterns: DET-NOUN-VERB, PRON-VERB, NOUN-VERB, etc.
            if len(pos_tags) >= 3:
                # Look for noun-verb combinations
                has_noun_verb_combo = False
                for i in range(len(pos_tags) - 1):
                    if pos_tags[i] in ['NOUN', 'PROPN', 'PRON'] and pos_tags[i+1] in ['VERB', 'AUX']:
                        has_noun_verb_combo = True
                        break

                if not partial and not has_noun_verb_combo and has_verb:
                    score *= 0.8  # Minor penalty for unusual structure

            # Check 4: Penalize too many adjectives/adverbs in a row
            max_consecutive_modifiers = 0
            current_consecutive = 0
            for pos in pos_tags:
                if pos in ['ADJ', 'ADV']:
                    current_consecutive += 1
                    max_consecutive_modifiers = max(max_consecutive_modifiers, current_consecutive)
                else:
                    current_consecutive = 0

            if max_consecutive_modifiers > 3:
                score *= 0.6  # Penalty for too many modifiers

            # Check 5: Balance of content vs function words
            content_words = sum(1 for pos in pos_tags if pos in ['NOUN', 'VERB', 'ADJ', 'ADV', 'PROPN'])
            function_words = sum(1 for pos in pos_tags if pos in ['DET', 'ADP', 'CONJ', 'PRON'])

            if len(pos_tags) > 5:
                content_ratio = content_words / len(pos_tags)
                if content_ratio < 0.2:  # Too few content words
                    score *= 0.7
                elif content_ratio > 0.9:  # Too many content words, not enough function words
                    score *= 0.8

            # Final validation: score above threshold
            is_valid = score >= 0.9  # Threshold for acceptability (very strict grammar)

            return is_valid, score

        except Exception as e:
            print(f"[WARNING] Grammar validation error: {e}")
            return True, 1.0  # Default to accepting on error


def get_grammar_validator():
    """Get or initialize the grammar validator. Uses lazy loading and caching."""
    global _grammar_validator
    if _grammar_validator is None:
        _grammar_validator = GrammarValidator()
    return _grammar_validator if _grammar_validator.enabled else None


def get_punctuation_model():
    """
    Get or initialize the punctuation restoration model.
    Uses lazy loading and caching for performance.
    """
    global _punctuation_model
    if _punctuation_model is None:
        try:
            from deepmultilingualpunctuation import PunctuationModel
            print("[INFO] Loading punctuation restoration model (first time only)...")
            _punctuation_model = PunctuationModel()
            print("[INFO] Punctuation model loaded successfully!")
        except ImportError:
            print("[WARNING] deepmultilingualpunctuation not installed. Run: pip install deepmultilingualpunctuation")
            _punctuation_model = False  # Mark as unavailable
        except Exception as e:
            print(f"[WARNING] Failed to load punctuation model: {e}")
            _punctuation_model = False
    return _punctuation_model if _punctuation_model is not False else None


def add_punctuation_postprocess(text: str, use_external_model: bool = True) -> str:
    """
    Add punctuation and capitalization to generated text.

    Uses a pre-trained transformer model (deepmultilingualpunctuation) for
    high-quality punctuation restoration. Falls back to rule-based approach
    if the external model is unavailable.

    The external model predicts: . , ? - :

    Args:
        text: Generated text without punctuation
        use_external_model: Whether to use external model (default: True)

    Returns:
        Text with added punctuation and capitalization
    """
    if not text or not text.strip():
        return text

    # Try to use external model first
    if use_external_model:
        model = get_punctuation_model()
        if model is not None:
            try:
                # The model handles both punctuation and capitalization
                result = model.restore_punctuation(text)

                # Ensure first letter is capitalized (model sometimes misses this)
                if result and len(result) > 0:
                    result = result[0].upper() + result[1:] if len(result) > 1 else result.upper()

                # Capitalize after sentence-ending punctuation
                import re
                result = re.sub(r'([.!?]\s+)([a-z])', lambda m: m.group(1) + m.group(2).upper(), result)

                return result
            except Exception as e:
                print(f"[WARNING] Punctuation model failed, using fallback: {e}")
                # Fall through to rule-based approach

    # Fallback: Simple rule-based approach
    # This is much simpler than before since it's just a backup
    words = text.split()
    if not words:
        return text

    result = []
    sentence_length = 0
    should_capitalize_next = True  # Track when to capitalize

    for i, word in enumerate(words):
        # Capitalize first word and after periods
        if should_capitalize_next:
            word = word.capitalize()
            should_capitalize_next = False

        result.append(word)
        sentence_length += 1

        # Add period every 10-15 words
        if sentence_length >= 12 and i < len(words) - 1:
            result[-1] += '.'
            sentence_length = 0
            should_capitalize_next = True  # Capitalize after period

    # Ensure text ends with punctuation
    if result and not result[-1][-1] in '.!?':
        result[-1] += '.'

    final_result = ' '.join(result)

    # Double-check: Capitalize first letter
    if final_result:
        final_result = final_result[0].upper() + final_result[1:] if len(final_result) > 1 else final_result.upper()

    return final_result


class TextDataset(Dataset):
    """Dataset for text sequences."""

    def __init__(self, X, y):
        self.X = torch.LongTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LSTMModel(nn.Module):
    """Improved LSTM model with Layer Normalization."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        lstm_units: int,
        num_layers: int,
        dropout_rate: float
    ):
        super(LSTMModel, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding_dropout = nn.Dropout(dropout_rate * 0.5)  # Light dropout on embeddings (0.15 when dropout=0.3)

        self.lstm = nn.LSTM(
            embedding_dim,
            lstm_units,
            num_layers,
            dropout=dropout_rate if num_layers > 1 else 0,
            batch_first=True
        )

        # Add Layer Normalization for better training stability
        self.layer_norm = nn.LayerNorm(lstm_units)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(lstm_units, vocab_size)

        # Better weight initialization
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier/Glorot initialization."""
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        embedded = self.embedding(x)
        embedded = self.embedding_dropout(embedded)

        lstm_out, _ = self.lstm(embedded)
        # Take the last output
        lstm_out = lstm_out[:, -1, :]

        # Apply layer normalization before dropout
        normalized = self.layer_norm(lstm_out)
        dropped = self.dropout(normalized)
        output = self.fc(dropped)
        return output


class SimpleTokenizer:
    """Simple word-level tokenizer."""

    def __init__(self):
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.vocab_size = 0

    def fit_on_texts(self, texts):
        """Build vocabulary from texts."""
        if isinstance(texts, str):
            texts = [texts]

        words = []
        for text in texts:
            words.extend(text.split())

        # Count word frequencies
        word_counts = Counter(words)

        # Build vocabulary (most common first)
        # Add padding token at index 0
        self.word_to_idx = {'<PAD>': 0}
        self.idx_to_word = {0: '<PAD>'}

        for idx, (word, _) in enumerate(word_counts.most_common(), start=1):
            self.word_to_idx[word] = idx
            self.idx_to_word[idx] = word

        self.vocab_size = len(self.word_to_idx)

    def texts_to_sequences(self, texts):
        """Convert texts to sequences of integers."""
        if isinstance(texts, str):
            texts = [texts]

        sequences = []
        for text in texts:
            if isinstance(text, list):
                # Already split into words
                seq = [self.word_to_idx.get(word, 0) for word in text]
            else:
                # Split string into words
                words = text.split()
                seq = [self.word_to_idx.get(word, 0) for word in words]
            sequences.append(seq)

        return sequences


class TextGenerator:
    """
    Advanced RNN-based text generator with LSTM architecture using PyTorch.

    This class handles:
    - Text preprocessing and tokenization
    - Sequence generation for training
    - LSTM model construction
    - Training with visualization
    - Text generation with temperature sampling
    """

    def __init__(
        self,
        sequence_length: int = 50,
        embedding_dim: int = 100,
        lstm_units: int = 150,
        num_layers: int = 2,
        dropout_rate: float = 0.2
    ):
        """
        Initialize the text generator.

        Args:
            sequence_length: Number of words to consider for context
            embedding_dim: Dimension of word embeddings
            lstm_units: Number of units in each LSTM layer
            num_layers: Number of LSTM layers
            dropout_rate: Dropout rate for regularization
        """
        self.sequence_length = sequence_length
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate

        self.tokenizer = SimpleTokenizer()
        self.model = None
        self.vocab_size = 0
        self.max_sequence_len = 0
        self.history = None

        # Set device (GPU if available)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

    def preprocess_text(self, text: str) -> str:
        """
        Clean and normalize input text to match vocabulary extraction.

        CRITICAL: Must use EXACT same tokenization as extract_vocabulary.py!

        Args:
            text: Raw input text

        Returns:
            Cleaned text string with space-separated tokens
        """
        # Use EXACT same regex as extract_vocabulary.py
        # This extracts:
        # - Alphabetic words
        # - Contractions like "don't", "it's" (kept intact!)
        # - Already lowercased
        words = re.findall(r'\b[a-zA-Z]+(?:\'[a-z]+)?\b', text.lower())

        # Join with spaces
        text = ' '.join(words)

        return text

    def prepare_sequences(self, text: str) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Convert text to training sequences.

        Process:
        1. Tokenize text into words
        2. Build vocabulary
        3. Create sliding window sequences
        4. Encode as integer sequences
        5. Separate into X (input) and y (target)

        Args:
            text: Preprocessed text

        Returns:
            Tuple of (X, y, max_sequence_len)
            - X: Input sequences (context words)
            - y: Target words (integer indices)
            - max_sequence_len: Length of longest sequence
        """
        # Preprocess
        text = self.preprocess_text(text)

        # Tokenize text
        self.tokenizer.fit_on_texts([text])
        self.vocab_size = self.tokenizer.vocab_size

        print(f"Vocabulary size: {self.vocab_size}")

        # Create input sequences using sliding window
        input_sequences = []
        words = text.split()

        for i in range(self.sequence_length, len(words)):
            # Take sequence_length + 1 words
            # First sequence_length words = input
            # Last word = target
            seq = words[i - self.sequence_length : i + 1]
            input_sequences.append(seq)

        print(f"Total sequences: {len(input_sequences)}")

        # Convert words to integer sequences
        token_sequences = self.tokenizer.texts_to_sequences(input_sequences)

        # Pad sequences to same length
        self.max_sequence_len = max([len(seq) for seq in token_sequences])

        # Manual padding
        padded_sequences = []
        for seq in token_sequences:
            if len(seq) < self.max_sequence_len:
                padded = [0] * (self.max_sequence_len - len(seq)) + seq
            else:
                padded = seq
            padded_sequences.append(padded)

        padded_sequences = np.array(padded_sequences)

        # Split into inputs and labels
        X = padded_sequences[:, :-1]  # All but last word
        y = padded_sequences[:, -1]   # Last word (as indices, not one-hot)

        print(f"X shape: {X.shape}")
        print(f"y shape: {y.shape}")

        return X, y, self.max_sequence_len

    def build_model(self) -> nn.Module:
        """
        Build LSTM model architecture.

        Architecture:
        1. Embedding layer (word → dense vector)
        2. Multiple LSTM layers with dropout
        3. Dense output layer with softmax

        Returns:
            PyTorch model
        """
        model = LSTMModel(
            vocab_size=self.vocab_size,
            embedding_dim=self.embedding_dim,
            lstm_units=self.lstm_units,
            num_layers=self.num_layers,
            dropout_rate=self.dropout_rate
        )

        model = model.to(self.device)
        self.model = model
        return model

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 100,
        batch_size: int = 128,
        validation_split: float = 0.1,
        save_path: str = "saved_models"
    ) -> Dict:
        """
        Train the LSTM model with visualization.

        Args:
            X: Input sequences
            y: Target words (integer indices)
            epochs: Number of training epochs
            batch_size: Batch size for training
            validation_split: Fraction of data for validation
            save_path: Directory to save checkpoints

        Returns:
            Dictionary with training history
        """
        # Split into train and validation
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        # Create datasets
        train_dataset = TextDataset(X_train, y_train)
        val_dataset = TextDataset(X_val, y_val)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,  # Disable multiprocessing on Windows (fixes shared memory errors)
            pin_memory=True  # Faster GPU transfer
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            num_workers=0,  # Disable multiprocessing on Windows
            pin_memory=True
        )

        # Loss and optimizer - SIMPLIFIED
        criterion = nn.CrossEntropyLoss(label_smoothing=0.05)  # Reduced label smoothing

        # Higher initial learning rate for faster convergence
        optimizer = optim.AdamW(self.model.parameters(), lr=0.002, weight_decay=0.005, betas=(0.9, 0.999))

        # OneCycleLR: Modern approach with warmup and decay (must step after each batch)
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=0.003,                    # Peak learning rate
            steps_per_epoch=len(train_loader),
            epochs=epochs,
            pct_start=0.3                    # 30% warmup, 70% decay
        )

        # Training history
        history = {
            'loss': [],
            'accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }

        best_val_loss = float('inf')
        patience_counter = 0
        patience = 3  # Early stopping after 3 epochs without improvement

        print(f"\nTraining on {self.device}")
        print(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}\n")

        for epoch in range(epochs):
            epoch_start_time = time.time()

            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            # Progress bar for training batches
            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]",
                            unit="batch", leave=False, ncols=100)

            for batch_X, batch_y in train_pbar:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                # Forward pass
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)

                # Backward pass with gradient clipping
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # Prevent exploding gradients
                optimizer.step()
                scheduler.step()  # OneCycleLR requires stepping after each batch

                # Statistics
                train_loss += loss.item() * batch_X.size(0)
                _, predicted = torch.max(outputs, 1)
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()

                # Update progress bar with current loss and accuracy
                current_loss = train_loss / train_total
                current_acc = train_correct / train_total
                train_pbar.set_postfix({'loss': f'{current_loss:.4f}', 'acc': f'{current_acc:.4f}'})

            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            # Progress bar for validation batches
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]",
                          unit="batch", leave=False, ncols=100)

            with torch.no_grad():
                for batch_X, batch_y in val_pbar:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)

                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)

                    val_loss += loss.item() * batch_X.size(0)
                    predicted = self._beam_search_predict(outputs, beam_width=5)
                    val_total += batch_y.size(0)
                    val_correct += (predicted == batch_y).sum().item()

                    # Update progress bar with current loss and accuracy
                    current_val_loss = val_loss / val_total
                    current_val_acc = val_correct / val_total
                    val_pbar.set_postfix({'loss': f'{current_val_loss:.4f}', 'acc': f'{current_val_acc:.4f}'})

            # Calculate epoch time
            epoch_time = time.time() - epoch_start_time

            # Calculate metrics
            train_loss = train_loss / train_total
            train_acc = train_correct / train_total
            val_loss = val_loss / val_total
            val_acc = val_correct / val_total

            # Store history
            history['loss'].append(train_loss)
            history['accuracy'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_acc)

            # Print progress with timing and current LR
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}/{epochs} - Time: {epoch_time:.2f}s - LR: {current_lr:.6f}")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} ({train_acc*100:.2f}%)")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} ({val_acc*100:.2f}%)")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), f"{save_path}/model_best.pt")
                print(f"  [OK] Model saved (val_loss improved)")
            else:
                patience_counter += 1
                print(f"  [INFO] No improvement ({patience_counter}/{patience})")

            # Early stopping
            if patience_counter >= patience:
                print(f"\n[INFO] Early stopping triggered after {epoch+1} epochs")
                break

            print()

        # Load best weights
        self.model.load_state_dict(torch.load(f"{save_path}/model_best.pt", weights_only=True))
        self.history = history

        return history

    def generate_text(
        self,
        seed_text: str,
        num_words: int = 50,
        temperature: float = 1.0,
        use_beam_search: bool = False,
        beam_width: int = 5,
        length_penalty: float = 1.0,
        repetition_penalty: float = 1.2,
        beam_temperature: float = 0.0,
        add_punctuation: bool = False,
        validate_grammar: bool = False
    ) -> str:
        """
        Generate text using trained model with optional beam search.

        Temperature controls randomness (for sampling):
        - Low (0.5): More predictable, coherent
        - Medium (1.0): Balanced
        - High (1.5-2.0): More creative, random

        Beam search parameters:
        - use_beam_search: If True, uses beam search instead of sampling
        - beam_width: Number of beams to maintain (default: 5)
        - length_penalty: Penalty for longer sequences (default: 1.0)
        - repetition_penalty: Penalty for repeated tokens (default: 1.2, higher = less repetition)
        - beam_temperature: Adds randomness to beam search (0.0=deterministic, 0.5-1.0=varied)

        Post-processing:
        - add_punctuation: If True, adds punctuation and capitalization (default: False)
        - validate_grammar: If True, validates grammar during generation (default: False)

        Args:
            seed_text: Starting text
            num_words: Number of words to generate
            temperature: Sampling temperature (used for regular sampling)
            use_beam_search: Whether to use beam search decoding
            beam_width: Number of beams for beam search
            length_penalty: Length penalty for beam search scoring
            repetition_penalty: Repetition penalty for beam search (reduces loops)
            beam_temperature: Temperature for beam search randomization (0.0=fully deterministic)
            add_punctuation: Whether to add punctuation post-processing
            validate_grammar: Whether to validate grammar during generation

        Returns:
            Generated text string
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")

        if use_beam_search:
            generated = self._generate_beam_search(
                seed_text, num_words, beam_width, length_penalty, repetition_penalty, beam_temperature, validate_grammar
            )
        else:
            generated = self._generate_sampling(
                seed_text, num_words, temperature
            )

        # Apply post-processing if requested
        if add_punctuation:
            generated = add_punctuation_postprocess(generated)

        return generated

    def _generate_sampling(
        self,
        seed_text: str,
        num_words: int,
        temperature: float
    ) -> str:
        """Original sampling-based generation (extracted for clarity)."""
        self.model.eval()
        generated_text = seed_text.lower()

        with torch.no_grad():
            for _ in range(num_words):
                # Tokenize current text
                token_list = self.tokenizer.texts_to_sequences([generated_text])[0]

                # Take last sequence_length tokens
                token_list = token_list[-(self.sequence_length):]

                # Pad to model input size
                if len(token_list) < self.max_sequence_len - 1:
                    token_list = [0] * (self.max_sequence_len - 1 - len(token_list)) + token_list
                else:
                    token_list = token_list[-(self.max_sequence_len - 1):]

                # Convert to tensor
                token_tensor = torch.LongTensor([token_list]).to(self.device)

                # Predict next word probabilities
                output = self.model(token_tensor)
                predicted_probs = torch.softmax(output / temperature, dim=-1)
                predicted_probs = predicted_probs.cpu().numpy()[0]

                # Sample from distribution
                predicted_index = np.random.choice(
                    len(predicted_probs),
                    p=predicted_probs
                )

                # Convert index to word
                if predicted_index in self.tokenizer.idx_to_word:
                    word = self.tokenizer.idx_to_word[predicted_index]
                    if word != '<PAD>':
                        # Smart punctuation: don't add space before contraction suffixes
                        if word in ('s', 't', 'd', 'll', 're', 've', 'm') and generated_text and not generated_text.endswith("'"):
                            generated_text += "'" + word
                        else:
                            generated_text += " " + word

        return generated_text

    def _generate_beam_search(
        self,
        seed_text: str,
        num_words: int,
        beam_width: int,
        length_penalty: float,
        repetition_penalty: float = 1.2,
        beam_temperature: float = 0.0,
        validate_grammar: bool = False
    ) -> str:
        """
        Generate text using beam search decoding with optional grammar validation.

        Beam search maintains multiple candidate sequences and selects
        the most probable ones based on cumulative log probability.

        When beam_temperature > 0, adds randomness by sampling from top candidates
        instead of always picking the most probable ones (stochastic beam search).

        When validate_grammar is True, filters candidates based on grammatical
        structure, rejecting sequences with poor grammar scores.

        Args:
            seed_text: Starting text
            num_words: Number of words to generate
            beam_width: Number of beams to maintain
            length_penalty: Penalty for sequence length (higher = prefer longer)
            repetition_penalty: Penalty for repeated tokens (higher = less repetition)
            beam_temperature: Temperature for randomization (0.0=deterministic, >0=stochastic)
            validate_grammar: Whether to validate grammar during generation

        Returns:
            Generated text string
        """
        self.model.eval()
        seed_text_lower = seed_text.lower()

        # Initialize grammar validator if requested
        grammar_validator = None
        if validate_grammar:
            grammar_validator = get_grammar_validator()
            if grammar_validator:
                print("[INFO] Grammar validation enabled for generation")

        # Initialize token counts from seed text
        seed_tokens = self.tokenizer.texts_to_sequences([seed_text_lower])[0]
        initial_token_counts = {}
        for token_id in seed_tokens:
            initial_token_counts[token_id] = initial_token_counts.get(token_id, 0) + 1

        # Initialize beams: (sequence_text, log_probability, token_counts, recent_tokens)
        # recent_tokens is a list of the last N tokens for n-gram blocking
        beams = [(seed_text_lower, 0.0, initial_token_counts, seed_tokens[-10:] if len(seed_tokens) > 10 else seed_tokens)]

        with torch.no_grad():
            for _ in range(num_words):
                candidates = []

                for seq_text, seq_log_prob, token_counts, recent_tokens in beams:
                    # Tokenize current sequence
                    token_list = self.tokenizer.texts_to_sequences([seq_text])[0]

                    # Take last sequence_length tokens
                    token_list = token_list[-(self.sequence_length):]

                    # Pad to model input size
                    if len(token_list) < self.max_sequence_len - 1:
                        token_list = [0] * (self.max_sequence_len - 1 - len(token_list)) + token_list
                    else:
                        token_list = token_list[-(self.max_sequence_len - 1):]

                    # Convert to tensor
                    token_tensor = torch.LongTensor([token_list]).to(self.device)

                    # Predict next word probabilities
                    output = self.model(token_tensor)
                    log_probs = torch.log_softmax(output, dim=-1)
                    log_probs = log_probs.cpu().numpy()[0]

                    # Apply repetition penalty
                    if repetition_penalty != 1.0:
                        for token_id, count in token_counts.items():
                            if token_id < len(log_probs):
                                # Penalize tokens that have appeared before
                                # Use exponential penalty: penalty^(count^1.5) for much stronger effect
                                # This makes repeated tokens increasingly unlikely
                                effective_count = count ** 1.5  # Make penalty grow faster
                                log_probs[token_id] -= np.log(repetition_penalty) * effective_count

                        # N-gram blocking: heavily penalize if this would create a repeated 3-gram or 4-gram
                        if len(recent_tokens) >= 3:
                            for idx in range(len(log_probs)):
                                # Check if adding this token would create a repeated trigram
                                test_sequence = recent_tokens + [idx]
                                for i in range(len(test_sequence) - 6):
                                    # Check for repeated 3-grams
                                    if (test_sequence[i:i+3] == test_sequence[-3:] and
                                        i < len(test_sequence) - 3):
                                        log_probs[idx] -= 10.0  # Heavy penalty for repeated phrases
                                        break

                    # Get top beam_width predictions for this sequence
                    top_indices = np.argsort(log_probs)[-beam_width * 2:]  # Get more candidates to filter

                    for idx in top_indices:
                        if idx in self.tokenizer.idx_to_word:
                            word = self.tokenizer.idx_to_word[idx]
                            if word != '<PAD>':
                                # Smart punctuation: don't add space before contraction suffixes
                                if word in ('s', 't', 'd', 'll', 're', 've', 'm') and seq_text and not seq_text.endswith("'"):
                                    new_seq = seq_text + "'" + word
                                else:
                                    new_seq = seq_text + " " + word
                                # Cumulative log probability
                                new_log_prob = seq_log_prob + log_probs[idx]

                                # Update token counts
                                new_token_counts = token_counts.copy()
                                new_token_counts[idx] = new_token_counts.get(idx, 0) + 1

                                # Update recent tokens for n-gram tracking (keep last 10)
                                new_recent_tokens = (recent_tokens + [idx])[-10:]

                                # Apply length penalty: divide by (length^penalty)
                                # This prevents bias towards shorter sequences
                                seq_length = len(new_seq.split())
                                normalized_score = new_log_prob / (seq_length ** length_penalty)

                                # Grammar validation: check if sequence is grammatically valid
                                if grammar_validator is not None:
                                    is_valid, grammar_score = grammar_validator.validate_sequence(new_seq, partial=True)
                                    # Multiply normalized score by grammar score to prefer grammatical sequences
                                    # This allows grammatically better sequences to rank higher
                                    if is_valid:
                                        normalized_score *= grammar_score
                                    else:
                                        # Heavily penalize invalid grammar but don't completely reject
                                        normalized_score *= 0.1

                                candidates.append((new_seq, new_log_prob, normalized_score, new_token_counts, new_recent_tokens))

                # Select top beam_width candidates based on normalized score
                if candidates:
                    # Sort by normalized score (descending)
                    candidates.sort(key=lambda x: x[2], reverse=True)

                    # Stochastic beam search: add randomness if beam_temperature > 0
                    if beam_temperature > 0:
                        # Take top candidates (more than beam_width to have options)
                        top_k = min(len(candidates), beam_width * 3)
                        top_candidates = candidates[:top_k]

                        # Convert normalized scores to probabilities with temperature
                        scores = np.array([score for _, _, score, _, _ in top_candidates])
                        # Apply temperature and softmax
                        probs = np.exp(scores / beam_temperature)
                        probs = probs / np.sum(probs)

                        # Sample beam_width candidates based on probabilities
                        selected_indices = np.random.choice(
                            len(top_candidates),
                            size=min(beam_width, len(top_candidates)),
                            replace=False,
                            p=probs
                        )
                        beams = [(top_candidates[i][0], top_candidates[i][1], top_candidates[i][3], top_candidates[i][4])
                                 for i in selected_indices]
                    else:
                        # Deterministic: Keep only top beam_width
                        beams = [(seq, log_prob, token_counts, recent_tokens)
                                 for seq, log_prob, _, token_counts, recent_tokens in candidates[:beam_width]]
                else:
                    # No valid candidates, stop generation
                    break

        # Return the best beam (highest cumulative log probability)
        if beams:
            # Re-normalize all beams for final selection
            final_scores = []
            for seq, log_prob, _, _ in beams:
                seq_length = len(seq.split())
                normalized_score = log_prob / (seq_length ** length_penalty)
                final_scores.append((seq, normalized_score))

            final_scores.sort(key=lambda x: x[1], reverse=True)

            # With temperature, sample from top beams; without, take the best
            if beam_temperature > 0 and len(final_scores) > 1:
                # Sample from final beams with temperature
                scores = np.array([score for _, score in final_scores])
                probs = np.exp(scores / beam_temperature)
                probs = probs / np.sum(probs)
                selected_idx = np.random.choice(len(final_scores), p=probs)
                return final_scores[selected_idx][0]
            else:
                return final_scores[0][0]
        else:
            return seed_text_lower

    def visualize_architecture(self, save_path: str = "visualizations"):
        """Generate model architecture visualization."""
        print(f"[INFO] Model architecture saved (use torchviz for detailed visualization)")
        print(f"Model structure:\n{self.model}")

    def plot_training_history(self, save_path: str = "visualizations"):
        """
        Plot training and validation metrics.

        Creates two subplots:
        1. Loss over epochs
        2. Accuracy over epochs
        """
        if not PLOTTING_AVAILABLE:
            print("[WARNING] matplotlib not available, skipping training history plot")
            return

        if self.history is None:
            raise ValueError("No training history available!")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Loss plot
        ax1.plot(self.history['loss'], label='Training Loss', linewidth=2)
        ax1.plot(self.history['val_loss'], label='Validation Loss', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Model Loss Over Time', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Accuracy plot
        ax2.plot(self.history['accuracy'], label='Training Accuracy', linewidth=2)
        ax2.plot(self.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy', fontsize=12)
        ax2.set_title('Model Accuracy Over Time', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{save_path}/training_history.png", dpi=300, bbox_inches='tight')
        plt.close()

    def save_model(self, model_path: str, tokenizer_path: str):
        """Save model and tokenizer."""
        # Change extension from .h5 to .pt
        model_path = model_path.replace('.h5', '.pt')

        torch.save(self.model.state_dict(), model_path)

        with open(tokenizer_path, 'wb') as f:
            pickle.dump(self.tokenizer, f)

        # Save configuration
        config = {
            'sequence_length': self.sequence_length,
            'embedding_dim': self.embedding_dim,
            'lstm_units': self.lstm_units,
            'num_layers': self.num_layers,
            'dropout_rate': self.dropout_rate,
            'vocab_size': self.vocab_size,
            'max_sequence_len': self.max_sequence_len
        }

        config_path = model_path.replace('.pt', '_config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

    def load_model(self, model_path: str, tokenizer_path: str):
        """Load saved model and tokenizer."""
        # Change extension from .h5 to .pt
        model_path = model_path.replace('.h5', '.pt')

        with open(tokenizer_path, 'rb') as f:
            self.tokenizer = pickle.load(f)

        # Load configuration
        config_path = model_path.replace('.pt', '_config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)

        self.sequence_length = config['sequence_length']
        self.embedding_dim = config['embedding_dim']
        self.lstm_units = config['lstm_units']
        self.num_layers = config['num_layers']
        self.dropout_rate = config['dropout_rate']
        self.vocab_size = config['vocab_size']
        self.max_sequence_len = config['max_sequence_len']

        # Build and load model
        self.build_model()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.model.eval()

    def evaluate_model(self, X: np.ndarray, y: np.ndarray, batch_size: int = 128, use_beam_search: bool = True, beam_width: int = 5) -> Dict:
        """
        Evaluate model on test data using beam search or greedy decoding.

        Args:
            X: Input sequences
            y: Target words (integer indices)
            batch_size: Batch size for evaluation
            use_beam_search: If True, uses beam search for prediction (default: True)
            beam_width: Number of beams for beam search (default: 5)

        Returns:
            Dictionary with metrics: loss, accuracy, perplexity, r_squared
        """
        if self.model is None:
            raise ValueError("Model not loaded!")

        self.model.eval()
        test_dataset = TextDataset(X, y)
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False
        )

        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)

                total_loss += loss.item() * batch_X.size(0)

                if use_beam_search:
                    # Use beam search to find best prediction
                    predicted = self._beam_search_predict(outputs, beam_width)
                else:
                    # Use greedy decoding (original method)
                    _, predicted = torch.max(outputs, 1)

                total_samples += batch_y.size(0)
                total_correct += (predicted == batch_y).sum().item()

                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(batch_y.cpu().numpy())

        # Calculate metrics
        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples
        perplexity = np.exp(avg_loss)

        # Calculate R-squared
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)

        # R-squared for classification (pseudo R-squared)
        ss_res = np.sum((all_targets - all_predictions) ** 2)
        ss_tot = np.sum((all_targets - np.mean(all_targets)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        return {
            'test_loss': avg_loss,
            'test_accuracy': accuracy,
            'perplexity': perplexity,
            'samples_tested': total_samples,
            'r_squared': r_squared
        }

    def _beam_search_predict(self, logits: torch.Tensor, beam_width: int = 5) -> torch.Tensor:
        """
        Use beam search to predict the most likely token for a batch.

        This is a simplified beam search for single-token prediction that considers
        the top-k most likely tokens and their probabilities.

        Args:
            logits: Model output logits [batch_size, vocab_size]
            beam_width: Number of top candidates to consider

        Returns:
            Predicted token indices [batch_size]
        """
        # Get log probabilities
        log_probs = torch.log_softmax(logits, dim=-1)

        # Get top beam_width candidates for each sample in the batch
        top_log_probs, top_indices = torch.topk(log_probs, beam_width, dim=-1)

        # For single-token prediction, simply return the highest probability token
        # (the first of the top-k). This is equivalent to greedy decoding but
        # uses the beam search scoring methodology
        predictions = top_indices[:, 0]

        return predictions
