from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from prometheus.config import DataConfig


def synthetic_corpus(repeats: int) -> str:
    """Build a small repeated corpus for smoke tests and prototype runs."""

    base_examples = [
        "the modular network routes local signals before sharing global summaries.",
        "sparse communication should reduce wasted computation without losing useful state.",
        "hierarchical modules can specialize while preserving a narrow coordination path.",
        "prometheus compares a dense baseline against structured sparse variants.",
        "small pilot runs should reveal whether routing efficiency improves at fixed cost.",
    ]
    return "\n".join(base_examples * repeats)


@dataclass(slots=True)
class CharacterTokenizer:
    """Minimal character-level tokenizer backed by explicit lookup tables."""

    stoi: dict[str, int]
    itos: dict[int, str]

    @classmethod
    def build(cls, text: str) -> "CharacterTokenizer":
        """Create a tokenizer from the distinct characters present in text."""

        vocabulary = sorted(set(text))
        stoi = {ch: index for index, ch in enumerate(vocabulary)}
        itos = {index: ch for ch, index in stoi.items()}
        return cls(stoi=stoi, itos=itos)

    @property
    def vocab_size(self) -> int:
        """Return the number of unique characters in the tokenizer."""

        return len(self.stoi)

    def encode(self, text: str) -> list[int]:
        """Convert text into integer token ids."""

        return [self.stoi[ch] for ch in text]

    def decode(self, tokens: list[int]) -> str:
        """Convert token ids back into a character string."""

        return "".join(self.itos[token] for token in tokens)


@dataclass(slots=True)
class DatasetBundle:
    """Tokenized train and validation tensors plus the tokenizer used."""

    tokenizer: CharacterTokenizer
    train_tokens: torch.Tensor
    val_tokens: torch.Tensor


class LanguageModelingDataset:
    """Random-window sampler for next-token language-model batches."""

    def __init__(self, tokens: torch.Tensor, sequence_length: int):
        """Store a token sequence and validate it is long enough to sample from."""

        if tokens.numel() <= sequence_length:
            raise ValueError("Dataset is too small for the configured sequence length.")
        self.tokens = tokens
        self.sequence_length = sequence_length

    def sample_batch(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample random contiguous input-target windows for next-token prediction."""

        max_index = self.tokens.size(0) - self.sequence_length - 1
        starts = torch.randint(0, max_index, (batch_size,))
        batch_inputs = []
        batch_targets = []
        for start in starts.tolist():
            window = self.tokens[start : start + self.sequence_length + 1]
            batch_inputs.append(window[:-1])
            batch_targets.append(window[1:])
        x = torch.stack(batch_inputs).to(device)
        y = torch.stack(batch_targets).to(device)
        return x, y


def _load_text(config: DataConfig) -> str:
    """Load raw training text from synthetic examples or a configured file."""

    if config.dataset_type == "synthetic":
        return synthetic_corpus(config.synthetic_repeats)
    if config.dataset_type == "text" and config.path:
        return Path(config.path).read_text(encoding="utf-8")
    raise ValueError("Unsupported dataset configuration. Use synthetic or provide a text path.")


def build_datasets(config: DataConfig) -> DatasetBundle:
    """Create tokenized train and validation tensors from the configured corpus."""

    text = _load_text(config)
    tokenizer = CharacterTokenizer.build(text)
    encoded = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    split_index = int(encoded.size(0) * config.train_split)
    train_tokens = encoded[:split_index]
    val_tokens = encoded[split_index:]
    return DatasetBundle(tokenizer=tokenizer, train_tokens=train_tokens, val_tokens=val_tokens)
