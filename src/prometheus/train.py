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
from prometheus.model import DenseTransformerLM


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(raw_device: str) -> torch.device:
    if raw_device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(raw_device)


def resolve_model_config(config: PrometheusConfig, data_bundle: DatasetBundle) -> ModelConfig:
    vocab_size = data_bundle.tokenizer.vocab_size if config.model.vocab_size == "auto" else config.model.vocab_size
    return replace(config.model, vocab_size=int(vocab_size))


def _make_run_directory(config: PrometheusConfig) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    run_dir = Path(config.experiment.output_dir) / f"{config.experiment.run_name}-{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def create_optimizer(model: DenseTransformerLM, config: PrometheusConfig) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        betas=(0.9, 0.95),
    )


def learning_rate_for_step(config: PrometheusConfig, step: int) -> float:
    base_lr = config.training.learning_rate
    warmup_steps = max(config.training.warmup_steps, 1)
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(config.training.max_steps - warmup_steps, 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * cosine


@torch.no_grad()
def evaluate_model(
    model: DenseTransformerLM,
    dataset: LanguageModelingDataset,
    batch_size: int,
    device: torch.device,
    max_batches: int,
) -> dict[str, float]:
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
    set_seed(config.experiment.seed)
    data_bundle = build_datasets(config.data)
    resolved_model_config = resolve_model_config(config, data_bundle)
    device = resolve_device(config.experiment.device)
    train_dataset = LanguageModelingDataset(data_bundle.train_tokens, config.data.sequence_length)
    val_dataset = LanguageModelingDataset(data_bundle.val_tokens, config.data.sequence_length)
    model = DenseTransformerLM(resolved_model_config, sequence_length=config.data.sequence_length).to(device)
    optimizer = create_optimizer(model, config)
    run_dir = _make_run_directory(config)
    _write_json(run_dir / "config.snapshot.json", config.to_dict())
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
