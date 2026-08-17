from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from pathlib import Path

import torch

from prometheus.config import DataConfig


def synthetic_corpus(repeats: int) -> str:
    """Build a small repeated corpus for smoke tests and prototype runs.

    Args:
        repeats: Number of times to repeat the base sentence set.

    Returns:
        str: Concatenated synthetic corpus text.
    """

    base_examples = [
        "the modular network routes local signals before sharing global summaries.",
        "sparse communication should reduce wasted computation without losing useful state.",
        "hierarchical modules can specialize while preserving a narrow coordination path.",
        "prometheus compares a dense baseline against structured sparse variants.",
        "small pilot runs should reveal whether routing efficiency improves at fixed cost.",
    ]
    return "\n".join(base_examples * repeats)


@dataclass(slots=True)
class ReasoningProblem:
    """One synthetic arithmetic-chain problem with programmatic chain of thought."""

    expression: str
    intermediates: list[int]
    answer: int
    chain_length: int

    def direct_text(self) -> str:
        """Render the problem in direct answer format without intermediate steps."""

        return f"Q{self.expression}=A{self.answer};"

    def cot_text(self) -> str:
        """Render the problem in chain-of-thought format with intermediate values."""

        thoughts = ",".join(str(value) for value in self.intermediates)
        return f"Q{self.expression}=T{thoughts}:A{self.answer};"


def _problem_split(expression: str) -> str:
    """Assign a problem to train or val deterministically by expression hash."""

    digest = hashlib.md5(expression.encode("utf-8")).digest()
    return "val" if digest[0] % 10 == 0 else "train"


_REWRITE_OPS = "rci"


def _apply_rewrite(value: int, operation: str) -> int:
    """Apply one digit-rewrite op to a 0-99 state treated as a 2-digit string."""

    tens, ones = divmod(value, 10)
    if operation == "r":  # reverse digits
        tens, ones = ones, tens
    elif operation == "c":  # nines-complement each digit
        tens, ones = 9 - tens, 9 - ones
    elif operation == "i":  # increment each digit mod 10
        tens, ones = (tens + 1) % 10, (ones + 1) % 10
    else:
        raise ValueError(f"unknown rewrite op {operation!r}")
    return tens * 10 + ones


def _generate_family(config: DataConfig, split: str, family: str, count: int) -> list[ReasoningProblem]:
    """Generate `count` problems of one task family for one split."""

    seed_offset = 0 if family == "arithmetic" else 7919
    rng = random.Random(config.reasoning_seed + seed_offset)
    problems: list[ReasoningProblem] = []
    seen: set[str] = set()
    attempts = 0
    max_attempts = count * 20
    while len(problems) < count and attempts < max_attempts:
        attempts += 1
        chain_length = rng.randint(config.chain_length_min, config.chain_length_max)
        value = rng.randint(1, 99)
        expression = str(value)
        intermediates: list[int] = []
        if family == "arithmetic":
            for _ in range(chain_length):
                operation = rng.choice("+-*")
                operand = rng.randint(2, 9) if operation == "*" else rng.randint(1, 99)
                if operation == "+":
                    value = (value + operand) % 100
                elif operation == "-":
                    value = (value - operand) % 100
                else:
                    value = (value * operand) % 100
                expression += f"{operation}{operand}"
                intermediates.append(value)
        else:  # rewrite: digit-string ops, non-arithmetic by construction
            for _ in range(chain_length):
                operation = rng.choice(_REWRITE_OPS)
                value = _apply_rewrite(value, operation)
                expression += operation
                intermediates.append(value)
        if expression in seen or _problem_split(expression) != split:
            continue
        seen.add(expression)
        problems.append(
            ReasoningProblem(
                expression=expression,
                intermediates=intermediates,
                answer=value,
                chain_length=chain_length,
            )
        )
    return problems


def generate_reasoning_problems(config: DataConfig, split: str) -> list[ReasoningProblem]:
    """Generate deterministic reasoning problems for one data split.

    config.task_family selects the family: "arithmetic" (mod-100 chained
    arithmetic), "rewrite" (chained digit-string rewrites: reverse /
    nines-complement / increment-digits), or "both" (an even interleaved
    mix, for pretraining a trunk competent at both — the substrate for the
    task-transfer probe).

    Args:
        config: Data settings including chain-length bounds and problem count.
        split: Either ``train`` or ``val``; assignment is by expression hash.

    Returns:
        list[ReasoningProblem]: Unique problems belonging to the requested split.
    """

    if split not in {"train", "val"}:
        raise ValueError("split must be 'train' or 'val'")
    family = getattr(config, "task_family", "arithmetic")
    if family == "both":
        half = config.num_problems // 2
        mixed = _generate_family(config, split, "arithmetic", half) + _generate_family(
            config, split, "rewrite", config.num_problems - half
        )
        random.Random(config.reasoning_seed + 13).shuffle(mixed)
        return mixed
    if family not in {"arithmetic", "rewrite"}:
        raise ValueError("task_family must be arithmetic, rewrite, or both")
    return _generate_family(config, split, family, config.num_problems)


def reasoning_chain_corpus(config: DataConfig) -> str:
    """Build a newline-separated corpus of reasoning problems for LM training.

    Args:
        config: Data settings including the reasoning format selector.

    Returns:
        str: Concatenated corpus in direct, cot, or mixed (both) formats.
    """

    problems = generate_reasoning_problems(config, split="train")
    rng = random.Random(config.reasoning_seed + 1)
    lines: list[str] = []
    for problem in problems:
        if config.reasoning_format == "direct":
            lines.append(problem.direct_text())
        elif config.reasoning_format == "cot":
            lines.append(problem.cot_text())
        elif config.reasoning_format == "mixed":
            lines.append(problem.direct_text())
            lines.append(problem.cot_text())
        else:
            raise ValueError("reasoning_format must be direct, cot, or mixed")
    rng.shuffle(lines)
    return "\n".join(lines)


@dataclass(slots=True)
class CharacterTokenizer:
    """Minimal character-level tokenizer backed by explicit lookup tables."""

    stoi: dict[str, int]
    itos: dict[int, str]

    @classmethod
    def build(cls, text: str) -> "CharacterTokenizer":
        """Create a tokenizer from the distinct characters present in text.

        Args:
            text: Source text used to derive the vocabulary.

        Returns:
            CharacterTokenizer: Tokenizer covering every character in the text.
        """

        vocabulary = sorted(set(text))
        stoi = {ch: index for index, ch in enumerate(vocabulary)}
        itos = {index: ch for ch, index in stoi.items()}
        return cls(stoi=stoi, itos=itos)

    @property
    def vocab_size(self) -> int:
        """Return the number of unique characters in the tokenizer.

        Returns:
            int: Size of the tokenizer vocabulary.
        """

        return len(self.stoi)

    def encode(self, text: str) -> list[int]:
        """Convert text into integer token ids.

        Args:
            text: Character string to tokenize.

        Returns:
            list[int]: Token ids corresponding to the input text.
        """

        return [self.stoi[ch] for ch in text]

    def decode(self, tokens: list[int]) -> str:
        """Convert token ids back into a character string.

        Args:
            tokens: Token ids to decode.

        Returns:
            str: Decoded character string.
        """

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
        """Store a token sequence and validate it is long enough to sample from.

        Args:
            tokens: One-dimensional tensor of token ids.
            sequence_length: Context window length to sample.
        """

        if tokens.numel() <= sequence_length:
            raise ValueError("Dataset is too small for the configured sequence length.")
        self.tokens = tokens
        self.sequence_length = sequence_length

    def sample_batch(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample random contiguous input-target windows for next-token prediction.

        Args:
            batch_size: Number of windows to sample.
            device: Target device for the returned tensors.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Input and target token batches.
        """

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
    """Load raw training text from synthetic examples or a configured file.

    Args:
        config: Data settings describing the text source.

    Returns:
        str: Raw training corpus text.
    """

    if config.dataset_type == "synthetic":
        return synthetic_corpus(config.synthetic_repeats)
    if config.dataset_type == "reasoning_chain":
        return reasoning_chain_corpus(config)
    if config.dataset_type == "text" and config.path:
        return Path(config.path).read_text(encoding="utf-8")
    raise ValueError("Unsupported dataset configuration. Use synthetic or provide a text path.")


def build_datasets(config: DataConfig) -> DatasetBundle:
    """Create tokenized train and validation tensors from the configured corpus.

    Args:
        config: Data settings describing corpus loading and split behavior.

    Returns:
        DatasetBundle: Tokenizer and split token tensors ready for training.
    """

    text = _load_text(config)
    tokenizer = CharacterTokenizer.build(text)
    encoded = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    split_index = int(encoded.size(0) * config.train_split)
    train_tokens = encoded[:split_index]
    val_tokens = encoded[split_index:]
    return DatasetBundle(tokenizer=tokenizer, train_tokens=train_tokens, val_tokens=val_tokens)
