from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    """Read a JSON file into a dictionary."""

    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read newline-delimited JSON records from disk."""

    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def summarize_run(run_dir: str | Path) -> dict[str, Any]:
    """Summarize the key metrics and metadata stored for one run directory."""

    path = Path(run_dir)
    model_summary_path = path / "model.summary.json"
    model_summary = _read_json(model_summary_path) if model_summary_path.exists() else {}
    metrics = _read_jsonl(path / "metrics.jsonl")
    config_snapshot_path = path / "config.snapshot.json"
    config_snapshot = _read_json(config_snapshot_path) if config_snapshot_path.exists() else {}
    train_records = [record for record in metrics if record.get("split") == "train"]
    val_records = [record for record in metrics if record.get("split") == "val"]
    latest_train = train_records[-1] if train_records else {}
    latest_val = val_records[-1] if val_records else {}
    best_val = min(val_records, key=lambda record: record["loss"]) if val_records else {}
    return {
        "run_dir": str(path),
        "architecture": model_summary.get("architecture", config_snapshot.get("model", {}).get("architecture", "unknown")),
        "parameter_count": model_summary.get("parameter_count"),
        "vocab_size": model_summary.get("vocab_size", config_snapshot.get("model", {}).get("vocab_size")),
        "sequence_length": model_summary.get("sequence_length", config_snapshot.get("data", {}).get("sequence_length")),
        "latest_train_loss": latest_train.get("loss"),
        "latest_val_loss": latest_val.get("loss"),
        "latest_val_perplexity": latest_val.get("perplexity"),
        "best_val_loss": best_val.get("loss"),
        "best_val_perplexity": best_val.get("perplexity"),
        "steps_logged": len(train_records),
    }


def comparison_markdown(run_summaries: list[dict[str, Any]]) -> str:
    """Render multiple run summaries as a Markdown comparison table."""

    headers = [
        "run",
        "architecture",
        "params",
        "seq_len",
        "latest_val_loss",
        "best_val_loss",
        "latest_val_ppl",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for summary in run_summaries:
        row = [
            Path(summary["run_dir"]).name,
            str(summary.get("architecture", "")),
            str(summary.get("parameter_count", "")),
            str(summary.get("sequence_length", "")),
            _format_number(summary.get("latest_val_loss")),
            _format_number(summary.get("best_val_loss")),
            _format_number(summary.get("latest_val_perplexity")),
        ]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _format_number(value: Any) -> str:
    """Format scalar values for Markdown table output."""

    if isinstance(value, float):
        return f"{value:.4f}"
    if value is None:
        return ""
    return str(value)
