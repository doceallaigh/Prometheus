from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class ExperimentConfig:
    run_name: str
    seed: int
    device: str
    output_dir: str


@dataclass(slots=True)
class DataConfig:
    dataset_type: str
    sequence_length: int
    batch_size: int
    train_split: float = 0.9
    path: str | None = None
    synthetic_repeats: int = 1000


@dataclass(slots=True)
class ModelConfig:
    vocab_size: int | str
    embedding_dim: int
    num_heads: int
    num_layers: int
    dropout: float
    mlp_ratio: int = 4


@dataclass(slots=True)
class TrainingConfig:
    max_steps: int
    eval_interval: int
    log_interval: int
    learning_rate: float
    weight_decay: float
    grad_clip: float
    warmup_steps: int = 0


@dataclass(slots=True)
class EvaluationConfig:
    max_batches: int


@dataclass(slots=True)
class PrometheusConfig:
    experiment: ExperimentConfig
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    evaluation: EvaluationConfig

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Config at {path} did not parse to a mapping.")
    return data


def load_config(path: str | Path) -> PrometheusConfig:
    config_path = Path(path)
    raw = _read_yaml(config_path)
    return PrometheusConfig(
        experiment=ExperimentConfig(**raw["experiment"]),
        data=DataConfig(**raw["data"]),
        model=ModelConfig(**raw["model"]),
        training=TrainingConfig(**raw["training"]),
        evaluation=EvaluationConfig(**raw["evaluation"]),
    )
