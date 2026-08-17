from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import torch

from prometheus.config import ModelConfig, PrometheusConfig
from prometheus.data import DatasetBundle, LanguageModelingDataset, build_datasets
from prometheus.model import LanguageModelBase, build_model, structural_connectivity_summary


def set_seed(seed: int) -> None:
    """Seed Python and Torch RNG state for reproducible prototype runs.

    Args:
        seed: Integer random seed.

    Returns:
        None: Random number generators are seeded in place.
    """

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(raw_device: str) -> torch.device:
    """Resolve an explicit device string or choose a default available device.

    Args:
        raw_device: Requested device identifier or ``auto``.

    Returns:
        torch.device: Device chosen for model execution.
    """

    if raw_device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(raw_device)


def resolve_model_config(config: PrometheusConfig, data_bundle: DatasetBundle) -> ModelConfig:
    """Fill in runtime-derived model settings such as an automatic vocab size.

    Args:
        config: Experiment configuration template.
        data_bundle: Tokenized data used to resolve runtime values.

    Returns:
        ModelConfig: Model configuration with runtime values filled in.
    """

    vocab_size = data_bundle.tokenizer.vocab_size if config.model.vocab_size == "auto" else config.model.vocab_size
    return replace(config.model, vocab_size=int(vocab_size))


def _make_run_directory(config: PrometheusConfig) -> Path:
    """Create a timestamped output directory for a single run.

    Args:
        config: Experiment configuration containing run metadata.

    Returns:
        Path: Newly created output directory for the run.
    """

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_dir = Path(config.experiment.output_dir) / f"{config.experiment.run_name}-{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _write_json(path: Path, payload: dict) -> None:
    """Write a dictionary to disk as indented JSON.

    Args:
        path: Destination file path.
        payload: Dictionary to serialize.

    Returns:
        None: The JSON file is written to disk.
    """

    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def create_optimizer(model: LanguageModelBase, config: PrometheusConfig) -> torch.optim.Optimizer:
    """Construct the AdamW optimizer used by prototype experiments.

    Args:
        model: Model whose parameters will be optimized.
        config: Experiment training settings.

    Returns:
        torch.optim.Optimizer: Configured AdamW optimizer.
    """

    return torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        betas=(0.9, 0.95),
    )


def parameter_count(model: LanguageModelBase) -> int:
    """Count the total number of trainable and non-trainable parameters.

    Args:
        model: Model to count parameters for.

    Returns:
        int: Total parameter count.
    """

    return sum(parameter.numel() for parameter in model.parameters())


def learning_rate_for_step(config: PrometheusConfig, step: int) -> float:
    """Compute the warmup-plus-cosine learning rate for a given step.

    Args:
        config: Experiment training settings.
        step: Zero-based training step.

    Returns:
        float: Learning rate to use for the given step.
    """

    base_lr = config.training.learning_rate
    warmup_steps = max(config.training.warmup_steps, 1)
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(config.training.max_steps - warmup_steps, 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * cosine


@torch.no_grad()
def evaluate_model(
    model: LanguageModelBase,
    dataset: LanguageModelingDataset,
    batch_size: int,
    device: torch.device,
    max_batches: int,
) -> dict[str, float]:
    """Estimate loss and perplexity over a bounded number of validation batches.

    Args:
        model: Model to evaluate.
        dataset: Validation dataset sampler.
        batch_size: Number of sequences per evaluation batch.
        device: Device on which to run evaluation.
        max_batches: Maximum number of batches to evaluate.

    Returns:
        dict[str, float]: Validation loss and perplexity summary.
    """

    model.eval()
    started_at = time.perf_counter()
    losses = []
    for _ in range(max_batches):
        x, y = dataset.sample_batch(batch_size=batch_size, device=device)
        output = model(x, y)
        if output.loss is None:
            raise RuntimeError("Expected loss during evaluation.")
        losses.append(output.loss.item())
    mean_loss = sum(losses) / len(losses)
    perplexity = math.exp(mean_loss) if mean_loss < 20 else float("inf")
    elapsed_seconds = max(time.perf_counter() - started_at, 1e-9)
    processed_tokens = batch_size * dataset.sequence_length * max_batches
    model.train()
    return {
        "loss": mean_loss,
        "perplexity": perplexity,
        "elapsed_seconds": elapsed_seconds,
        "tokens_per_second": processed_tokens / elapsed_seconds,
    }


def run_training(config: PrometheusConfig) -> Path:
    """Execute one full training run and write its artifacts to disk.

    Args:
        config: Fully specified experiment configuration.

    Returns:
        Path: Output directory containing run artifacts.
    """

    set_seed(config.experiment.seed)
    data_bundle = build_datasets(config.data)
    resolved_model_config = resolve_model_config(config, data_bundle)
    device = resolve_device(config.experiment.device)
    train_dataset = LanguageModelingDataset(data_bundle.train_tokens, config.data.sequence_length)
    val_dataset = LanguageModelingDataset(data_bundle.val_tokens, config.data.sequence_length)
    model = build_model(resolved_model_config, sequence_length=config.data.sequence_length).to(device)
    optimizer = create_optimizer(model, config)
    run_dir = _make_run_directory(config)
    config_snapshot = config.to_dict()
    config_snapshot["experiment"]["requested_device"] = config.experiment.device
    config_snapshot["experiment"]["device"] = str(device)
    config_snapshot["model"] = asdict(resolved_model_config)
    _write_json(run_dir / "config.snapshot.json", config_snapshot)
    _write_json(
        run_dir / "model.summary.json",
        {
            "architecture": resolved_model_config.architecture,
            "parameter_count": parameter_count(model),
            "vocab_size": resolved_model_config.vocab_size,
            "sequence_length": config.data.sequence_length,
            **structural_connectivity_summary(model),
        },
    )
    metrics_path = run_dir / "metrics.jsonl"
    pruning_applied = False
    pruning_summary: dict[str, int | float | str] = {}
    plateau_count = 0
    previous_val_perplexity: float | None = None

    training_started_at = time.perf_counter()
    for step in range(config.training.max_steps):
        step_started_at = time.perf_counter()
        lr = learning_rate_for_step(config, step)
        for group in optimizer.param_groups:
            group["lr"] = lr
        x, y = train_dataset.sample_batch(batch_size=config.data.batch_size, device=device)
        output = model(x, y)
        if output.loss is None:
            raise RuntimeError("Expected training loss.")
        optimizer.zero_grad(set_to_none=True)
        output.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip)
        optimizer.step()
        step_elapsed_seconds = max(time.perf_counter() - step_started_at, 1e-9)
        train_tokens = config.data.batch_size * config.data.sequence_length

        if step % config.training.log_interval == 0 or step == config.training.max_steps - 1:
            train_loss = output.loss.item()
            train_perplexity = math.exp(train_loss) if train_loss < 20 else float("inf")
            record = {
                "step": step,
                "split": "train",
                "loss": train_loss,
                "perplexity": train_perplexity,
                "lr": lr,
                "step_seconds": step_elapsed_seconds,
                "tokens_per_second": train_tokens / step_elapsed_seconds,
            }
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            print(json.dumps(record))

        needs_eval = step % config.training.eval_interval == 0 or step == config.training.max_steps - 1
        if needs_eval:
            evaluation = evaluate_model(
                model=model,
                dataset=val_dataset,
                batch_size=config.data.batch_size,
                device=device,
                max_batches=config.evaluation.max_batches,
            )
            evaluation_record = {"step": step, "split": "val", **evaluation}
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(evaluation_record) + "\n")
            print(json.dumps(evaluation_record))

            if config.training.pruning_schedule == "inflection" and not pruning_applied and step >= config.training.pruning_min_steps:
                current_val_perplexity = evaluation["perplexity"]
                if previous_val_perplexity is not None:
                    improvement = previous_val_perplexity - current_val_perplexity
                    if improvement <= config.training.pruning_min_improvement:
                        plateau_count += 1
                    else:
                        plateau_count = 0
                previous_val_perplexity = current_val_perplexity

                if plateau_count >= config.training.pruning_patience:
                    apply_pruning = getattr(model, "apply_inflection_pruning", None)
                    if callable(apply_pruning):
                        applied = apply_pruning()
                        if applied is not None:
                            pruning_applied = True
                            pruning_summary = {"strategy": "inflection", "step": step, **applied}
                            prune_record = {"step": step, "split": "prune", **pruning_summary}
                            with metrics_path.open("a", encoding="utf-8") as handle:
                                handle.write(json.dumps(prune_record) + "\n")
                            print(json.dumps(prune_record))

    torch.save(
        {
            "model_state": model.state_dict(),
            "model_config": asdict(resolved_model_config),
            "tokenizer": data_bundle.tokenizer.stoi,
        },
        run_dir / "checkpoint.pt",
    )
    total_training_seconds = time.perf_counter() - training_started_at
    _write_json(
        run_dir / "run.summary.json",
        {
            "total_training_seconds": total_training_seconds,
            "training_tokens": config.training.max_steps * config.data.batch_size * config.data.sequence_length,
            "average_training_tokens_per_second": (config.training.max_steps * config.data.batch_size * config.data.sequence_length) / max(total_training_seconds, 1e-9),
            "pruning_applied": pruning_applied,
            **pruning_summary,
        },
    )
    return run_dir
