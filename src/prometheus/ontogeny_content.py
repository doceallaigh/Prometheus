from __future__ import annotations

import json
import re
import time
from pathlib import Path

import torch

from prometheus.retrofit import _pearson, _spearman, _upper_half_logits, load_trunk


COUNT_KEYS = (
    "tokens",
    "gold_digits",
    "full_correct",
    "complement_correct",
    "noise_correct",
    "full_wrong",
    "complement_recovers",
    "noise_recovers",
    "complement_disagrees",
    "noise_disagrees",
    "complement_digit_at_gold_digit",
    "noise_digit_at_gold_digit",
    "complement_contending_digit",
    "noise_contending_digit",
    "complement_contextual_digit",
    "noise_contextual_digit",
)


def _summarize_counts(counts: dict[str, float]) -> dict[str, float]:
    tokens = max(counts["tokens"], 1.0)
    digits = max(counts["gold_digits"], 1.0)
    full_wrong = max(counts["full_wrong"], 1.0)
    complement_disagrees = max(counts["complement_disagrees"], 1.0)
    noise_disagrees = max(counts["noise_disagrees"], 1.0)
    return {
        "full_token_accuracy": counts["full_correct"] / tokens,
        "complement_token_accuracy": counts["complement_correct"] / tokens,
        "noise_token_accuracy": counts["noise_correct"] / tokens,
        "complement_recovery_rate": counts["complement_recovers"] / full_wrong,
        "noise_recovery_rate": counts["noise_recovers"] / full_wrong,
        "complement_recovery_precision": counts["complement_recovers"] / complement_disagrees,
        "noise_recovery_precision": counts["noise_recovers"] / noise_disagrees,
        "complement_disagreement_rate": counts["complement_disagrees"] / tokens,
        "noise_disagreement_rate": counts["noise_disagrees"] / tokens,
        "complement_digit_structure": counts["complement_digit_at_gold_digit"] / digits,
        "noise_digit_structure": counts["noise_digit_at_gold_digit"] / digits,
        "complement_contending_digit_rate": counts["complement_contending_digit"] / digits,
        "noise_contending_digit_rate": counts["noise_contending_digit"] / digits,
        "complement_contextual_digit_rate": counts["complement_contextual_digit"] / digits,
        "noise_contextual_digit_rate": counts["noise_contextual_digit"] / digits,
    }


def _aggregate_trace_rows(trace_rows: list[dict[str, float]]) -> dict[str, float]:
    counts = {key: sum(row[key] for row in trace_rows) for key in COUNT_KEYS}
    activation_count = max(sum(row["activation_count"] for row in trace_rows), 1.0)
    return {
        "activation_complement_fraction": sum(row["activation_sum"] for row in trace_rows) / activation_count,
        "random_complement_fraction": sum(row["random_activation_sum"] for row in trace_rows) / activation_count,
        **_summarize_counts(counts),
    }


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    index = min(int(probability * len(ordered)), len(ordered) - 1)
    return ordered[index]


def _bootstrap_intervals(
    trace_rows: list[dict[str, float]], metrics: list[str], generator: torch.Generator, samples: int = 1000
) -> dict[str, list[float]]:
    estimates = {metric: [] for metric in metrics}
    for _ in range(samples):
        indices = torch.randint(len(trace_rows), (len(trace_rows),), generator=generator).tolist()
        summary = _aggregate_trace_rows([trace_rows[index] for index in indices])
        for metric in metrics:
            estimates[metric].append(summary[metric])
    return {metric: [_percentile(values, 0.025), _percentile(values, 0.975)] for metric, values in estimates.items()}


def _paired_bootstrap_change(
    early: list[dict[str, float]], late: list[dict[str, float]], metrics: list[str], seed: int, samples: int = 2000
) -> dict[str, dict[str, float | list[float]]]:
    if len(early) != len(late):
        raise ValueError("Paired phases must contain the same traces")
    generator = torch.Generator().manual_seed(seed)
    changes = {metric: [] for metric in metrics}
    for _ in range(samples):
        indices = torch.randint(len(early), (len(early),), generator=generator).tolist()
        early_summary = _aggregate_trace_rows([early[index] for index in indices])
        late_summary = _aggregate_trace_rows([late[index] for index in indices])
        for metric in metrics:
            changes[metric].append(late_summary[metric] - early_summary[metric])
    early_summary = _aggregate_trace_rows(early)
    late_summary = _aggregate_trace_rows(late)
    return {
        metric: {
            "change": late_summary[metric] - early_summary[metric],
            "ci95": [_percentile(changes[metric], 0.025), _percentile(changes[metric], 0.975)],
        }
        for metric in metrics
    }


def _load_phase(model_spec: str, device: torch.device, dtype: torch.dtype):
    if "::" not in model_spec:
        return load_trunk(model_spec, device, dtype)
    base_model, adapter_path = model_spec.split("::", 1)
    model, tokenizer = load_trunk(base_model, device, dtype)
    from peft import PeftModel

    model = PeftModel.from_pretrained(model, adapter_path).merge_and_unload()
    model.eval()
    model.requires_grad_(False)
    return model, tokenizer


def _plot_html(rows: list[dict]) -> str:
    steps = [row["phase_step"] for row in rows]
    traces = [
        ("structured complement", "complement_digit_structure"),
        ("matched noise", "noise_digit_structure"),
        ("complement recovery", "complement_recovery_rate"),
        ("noise recovery", "noise_recovery_rate"),
        ("contention frequency", "complement_contending_digit_rate"),
        ("activation complement", "activation_complement_fraction"),
    ]
    payload = [
        {"x": steps, "y": [row[key] for row in rows], "name": label, "mode": "lines+markers"}
        for label, key in traces
    ]
    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<title>Longitudinal Complement Content</title><script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>body{{margin:0;padding:20px;background:#101820;color:#f2f4f3;font-family:Segoe UI,sans-serif}}#plot{{height:760px}}</style>
</head><body><h1>Longitudinal Complement Content</h1><div id="plot"></div><script>
Plotly.newPlot('plot',{json.dumps(payload)},{{template:'plotly_dark',xaxis:{{title:'training step'}},
yaxis:{{title:'rate / fraction',range:[0,1]}},legend:{{orientation:'h'}}}},{{responsive:true,displaylogo:false}});
</script></body></html>"""


def ontogeny_content_sweep(
    phase_models: list[tuple[int, str]],
    traces_path: str | Path,
    basis_path: str | Path,
    output_dir: str | Path,
    tap_layer: int = 12,
    num_traces: int = 128,
    max_seq_len: int = 640,
    device_str: str = "auto",
    seed: int = 1337,
    phase_metrics_path: str | Path | None = None,
) -> dict:
    """Track complement prevalence, structure, frequency, and gold utility across phases."""

    if not phase_models:
        raise ValueError("phase_models cannot be empty")
    device = torch.device(device_str if device_str != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    basis = torch.load(basis_path, map_location=device, weights_only=True)["basis_full"].to(device, torch.float32)
    traces = [json.loads(line) for line in Path(traces_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    traces = traces[:num_traces]
    if not traces:
        raise ValueError(f"No traces found in {traces_path}")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    accuracy_by_step = {}
    if phase_metrics_path is not None:
        phase_metrics = [json.loads(line) for line in Path(phase_metrics_path).read_text().splitlines() if line.strip()]
        accuracy_by_step = {int(row["phase_step"]): float(row["trunk_strict_accuracy"]) for row in phase_metrics}

    rows = []
    traces_by_step: dict[int, list[dict[str, float]]] = {}
    start = time.time()
    for phase_step, model_spec in sorted(phase_models):
        model, tokenizer = _load_phase(model_spec, device, dtype)
        d_model = model.config.hidden_size
        generator = torch.Generator().manual_seed(seed)
        random_basis = torch.linalg.qr(torch.randn(d_model, basis.size(0), generator=generator)).Q.T.to(device)
        counts = {key: 0.0 for key in COUNT_KEYS}
        trace_rows = []

        for trace_index, trace in enumerate(traces):
            trace_counts = {key: 0.0 for key in COUNT_KEYS}

            def increment(key: str, value: float | bool = 1.0) -> None:
                counts[key] += value
                trace_counts[key] += value

            prompt_ids = tokenizer(trace["prompt"], add_special_tokens=False)["input_ids"]
            completion_ids = tokenizer(trace["completion"], add_special_tokens=False)["input_ids"]
            ids = (prompt_ids + completion_ids)[:max_seq_len]
            prompt_len = len(prompt_ids)
            if len(ids) <= prompt_len + 4:
                continue
            batch = torch.tensor([ids], device=device)
            with torch.no_grad():
                outputs = model(batch, output_hidden_states=True, use_cache=False)
                h_tap = outputs.hidden_states[tap_layer].float()
            positions = slice(prompt_len - 1, len(ids) - 1)
            region = h_tap[:, positions, :]
            dominant = (region @ basis.T) @ basis
            complement = region - dominant
            random_dominant = (region @ random_basis.T) @ random_basis
            random_complement = region - random_dominant
            energy = region.pow(2).sum(-1).clamp_min(1e-9)
            trace_activation = (complement.pow(2).sum(-1) / energy)[0].tolist()
            trace_random_activation = (random_complement.pow(2).sum(-1) / energy)[0].tolist()

            noise = torch.randn(complement.shape, generator=generator).to(device)
            noise *= complement.norm(dim=-1, keepdim=True) / noise.norm(dim=-1, keepdim=True).clamp_min(1e-9)
            variants = {"full": region, "complement": complement, "noise": noise}
            predictions = {}
            with torch.no_grad():
                for name, replacement in variants.items():
                    stream = h_tap.clone()
                    stream[:, positions, :] = replacement
                    logits = _upper_half_logits(model, tap_layer, stream.to(dtype))
                    predictions[name] = logits[0, positions, :].argmax(-1).tolist()

            targets = ids[prompt_len:]
            question_numbers = set(re.findall(r"\d+", trace.get("question", trace["prompt"])))
            for target, full, complement_token, noise_token in zip(
                targets, predictions["full"], predictions["complement"], predictions["noise"]
            ):
                increment("tokens")
                full_correct = full == target
                increment("full_correct", full_correct)
                increment("complement_correct", complement_token == target)
                increment("noise_correct", noise_token == target)
                increment("full_wrong", not full_correct)
                increment("complement_recovers", (not full_correct) and complement_token == target)
                increment("noise_recovers", (not full_correct) and noise_token == target)
                increment("complement_disagrees", complement_token != full)
                increment("noise_disagrees", noise_token != full)
                target_text = tokenizer.decode([target])
                complement_text = tokenizer.decode([complement_token])
                noise_text = tokenizer.decode([noise_token])
                if any(char.isdigit() for char in target_text):
                    increment("gold_digits")
                    complement_digit = any(char.isdigit() for char in complement_text)
                    noise_digit = any(char.isdigit() for char in noise_text)
                    increment("complement_digit_at_gold_digit", complement_digit)
                    increment("noise_digit_at_gold_digit", noise_digit)
                    increment("complement_contending_digit", complement_digit and complement_text.strip() != tokenizer.decode([full]).strip())
                    increment("noise_contending_digit", noise_digit and noise_text.strip() != tokenizer.decode([full]).strip())
                    increment("complement_contextual_digit", any(run in question_numbers for run in re.findall(r"\d+", complement_text)))
                    increment("noise_contextual_digit", any(run in question_numbers for run in re.findall(r"\d+", noise_text)))

            trace_rows.append({
                **trace_counts,
                "activation_sum": sum(trace_activation),
                "random_activation_sum": sum(trace_random_activation),
                "activation_count": len(trace_activation),
            })

            if (trace_index + 1) % 32 == 0:
                print(json.dumps({"phase": phase_step, "traces": trace_index + 1}), flush=True)

        summary = _aggregate_trace_rows(trace_rows)
        interval_metrics = [
            "activation_complement_fraction", "complement_digit_structure",
            "complement_contending_digit_rate", "complement_recovery_rate",
            "complement_recovery_precision",
        ]
        intervals = _bootstrap_intervals(
            trace_rows, interval_metrics, torch.Generator().manual_seed(seed + phase_step)
        )
        row = {
            "phase_step": phase_step,
            "model": model_spec,
            "traces": len(traces),
            "tokens": int(counts["tokens"]),
            "trunk_strict_accuracy": accuracy_by_step.get(phase_step),
            **summary,
            "bootstrap_ci95": intervals,
            "elapsed_seconds": time.time() - start,
        }
        rows.append(row)
        traces_by_step[phase_step] = trace_rows
        print(json.dumps(row), flush=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    metric_keys = [
        "activation_complement_fraction",
        "complement_digit_structure",
        "complement_contending_digit_rate",
        "complement_recovery_rate",
        "complement_recovery_precision",
    ]
    correlations = []
    for x_name, xs in (
        ("training_step", [float(row["phase_step"]) for row in rows]),
        ("strict_accuracy", [row["trunk_strict_accuracy"] for row in rows]),
    ):
        if any(value is None for value in xs):
            continue
        for metric in metric_keys:
            ys = [float(row[metric]) for row in rows]
            correlations.append({"x": x_name, "metric": metric, "pearson_r": _pearson(xs, ys), "spearman_rho": _spearman(xs, ys)})

    endpoint_change = _paired_bootstrap_change(
        traces_by_step[rows[0]["phase_step"]], traces_by_step[rows[-1]["phase_step"]], metric_keys, seed
    )

    (output / "phase_content_metrics.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    (output / "correlations.json").write_text(json.dumps(correlations, indent=2), encoding="utf-8")
    (output / "endpoint_change.json").write_text(json.dumps(endpoint_change, indent=2), encoding="utf-8")
    (output / "content_plot.html").write_text(_plot_html(rows), encoding="utf-8")

    lines = [
        "# Longitudinal Complement Content", "",
        "Fixed teacher-forced traces and a fixed mature Jacobian basis isolate representational change from changing generated text.", "",
        "| step | strict acc | activation | random activation | digit structure | noise structure | contention | gold recovery | noise recovery | recovery precision |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        strict = "n/a" if row["trunk_strict_accuracy"] is None else f"{row['trunk_strict_accuracy']:.3f}"
        lines.append(
            f"| {row['phase_step']} | {strict} | {row['activation_complement_fraction']:.3f} | {row['random_complement_fraction']:.3f} "
            f"| {row['complement_digit_structure']:.3f} | {row['noise_digit_structure']:.3f} | {row['complement_contending_digit_rate']:.3f} "
            f"| {row['complement_recovery_rate']:.3f} | {row['noise_recovery_rate']:.3f} | {row['complement_recovery_precision']:.3f} |"
        )
    lines += ["", "## Correlations", "", "| x | metric | Pearson r | Spearman rho |", "| --- | --- | ---: | ---: |"]
    for row in correlations:
        lines.append(f"| {row['x']} | {row['metric']} | {row['pearson_r']:.3f} | {row['spearman_rho']:.3f} |")
    lines += ["", "## Paired endpoint change", "", "Positive means the mature checkpoint is higher.", "", "| metric | change | problem-bootstrap 95% CI |", "| --- | ---: | ---: |"]
    for metric, result in endpoint_change.items():
        lines.append(f"| {metric} | {result['change']:.4f} | [{result['ci95'][0]:.4f}, {result['ci95'][1]:.4f}] |")
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"rows": rows, "correlations": correlations, "endpoint_change": endpoint_change, "output_dir": str(output)}