from __future__ import annotations

import json
import math
import random
from dataclasses import asdict
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import torch

from prometheus.config import ModelConfig, PrometheusConfig
from prometheus.data import DatasetBundle, LanguageModelingDataset, build_datasets
from prometheus.model import LanguageModelBase, build_model


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

    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
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
    losses = []
    for _ in range(max_batches):
        x, y = dataset.sample_batch(batch_size=batch_size, device=device)
        output = model(x, y)
        if output.loss is None:
            raise RuntimeError("Expected loss during evaluation.")
        losses.append(output.loss.item())
    mean_loss = sum(losses) / len(losses)
    perplexity = math.exp(mean_loss) if mean_loss < 20 else float("inf")
    model.train()
    return {"loss": mean_loss, "perplexity": perplexity}


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
        },
    )
    metrics_path = run_dir / "metrics.jsonl"

    for step in range(config.training.max_steps):
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

        if step % config.training.log_interval == 0 or step == config.training.max_steps - 1:
            train_loss = output.loss.item()
            train_perplexity = math.exp(train_loss) if train_loss < 20 else float("inf")
            record = {
                "step": step,
                "split": "train",
                "loss": train_loss,
                "perplexity": train_perplexity,
                "lr": lr,
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

    torch.save(
        {
            "model_state": model.state_dict(),
            "model_config": asdict(resolved_model_config),
            "tokenizer": data_bundle.tokenizer.stoi,
        },
        run_dir / "checkpoint.pt",
    )
    return run_dir
