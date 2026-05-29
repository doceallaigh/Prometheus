from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class ExperimentConfig:
    """Top-level metadata controlling a single experiment run."""

    run_name: str
    seed: int
    device: str
    output_dir: str


@dataclass(slots=True)
class DataConfig:
    """Dataset construction settings for training and evaluation."""

    dataset_type: str
    sequence_length: int
    batch_size: int
    train_split: float = 0.9
    path: str | None = None
    synthetic_repeats: int = 1000


@dataclass(slots=True)
class ModelConfig:
    """Architecture parameters for dense and modular model variants."""

    vocab_size: int | str
    embedding_dim: int
    num_heads: int
    num_layers: int
    dropout: float
    architecture: str = "dense"
    mlp_ratio: int = 4
    stage_groups: list[int] | None = None
    stage_depths: list[int] | None = None
    routing_topology: str = "dense"
    routing_top_k: int | None = None


@dataclass(slots=True)
class TrainingConfig:
    """Optimizer, logging, and loop control settings for training."""

    max_steps: int
    eval_interval: int
    log_interval: int
    learning_rate: float
    weight_decay: float
    grad_clip: float
    warmup_steps: int = 0


@dataclass(slots=True)
class EvaluationConfig:
    """Evaluation loop limits applied during validation passes."""

    max_batches: int


@dataclass(slots=True)
class PrometheusConfig:
    """Container bundling the full configuration for one experiment."""

    experiment: ExperimentConfig
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    evaluation: EvaluationConfig

    def to_dict(self) -> dict[str, Any]:
        """Return the nested configuration as a plain dictionary."""

        return asdict(self)


def _read_yaml(path: Path) -> dict[str, Any]:
    """Load and validate a YAML document as a mapping."""

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Config at {path} did not parse to a mapping.")
    return data


def load_config(path: str | Path) -> PrometheusConfig:
    """Parse a YAML config file into typed configuration dataclasses."""

    config_path = Path(path)
    raw = _read_yaml(config_path)
    model_values = raw["model"]
    return PrometheusConfig(
        experiment=ExperimentConfig(**raw["experiment"]),
        data=DataConfig(**raw["data"]),
        model=ModelConfig(**model_values),
        training=TrainingConfig(**raw["training"]),
        evaluation=EvaluationConfig(**raw["evaluation"]),
    )
