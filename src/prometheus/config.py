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
    chain_length_min: int = 2
    chain_length_max: int = 8
    num_problems: int = 50000
    reasoning_seed: int = 1234
    reasoning_format: str = "mixed"
    task_family: str = "arithmetic"


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
    recurrent_steps: int | None = None
    recurrent_state_blend: float | None = None
    memory_fusion_blend: float | None = None
    memory_update_interval: int | None = None
    stage_groups: list[int] | None = None
    fixed_group_size: int | None = None
    stage_depths: list[int] | None = None
    column_counts: list[int] | None = None
    column_input_count: int | None = None
    column_branching_factor: int | None = None
    target_parameter_count: int | None = None
    max_column_stages: int | None = None
    fixed_column_size: int | None = None
    column_depths: list[int] | None = None
    column_recombination: str | None = None
    column_routing_topology: str | None = None
    column_routing_top_k: int | None = None
    cluster_copies: int | None = None
    cluster_bridge_percent: float | None = None
    cluster_wrap_neighbors: bool = False
    cluster_base_embedding_dim: int | None = None
    cluster_levels: int | None = None
    cluster_top_count: int | None = None
    cluster_target_parameter_count: int | None = None
    cluster_max_levels: int | None = None
    routing_topology: str = "dense"
    routing_top_k: int | None = None
    inflection_pruning_keep_ratio: float | None = None
    inflection_pruning_top_k: int | None = None
    base_checkpoint: str | None = None
    jspace_layer_index: int | None = None
    cfc_dim: int | None = None
    cfc_max_steps: int | None = None
    cfc_cell_type: str = "cfc"
    ponder_cost: float | None = None
    repr_loss_weight: float | None = None


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
    pruning_schedule: str | None = None
    pruning_min_steps: int = 0
    pruning_patience: int = 0
    pruning_min_improvement: float = 0.0


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
