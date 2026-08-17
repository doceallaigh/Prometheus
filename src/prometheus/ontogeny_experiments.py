from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Callable

import torch

from prometheus.ontogeny_content import _load_phase, _percentile
from prometheus.retrofit import COT_PROMPT, _chat_prompt, extract_answer_lenient

ARITHMETIC_OPERATORS = frozenset("+-*/=×÷^%")


def _token_training_gate_mask(
    input_ids: list[int], prompt_len: int, tokenizer, gate: str
) -> torch.Tensor:
    """Select residual row t by the semantic class of its next-token target t+1."""

    if gate not in {"digit", "operator", "digit-or-operator"}:
        raise ValueError(f"Unknown token training gate: {gate}")
    mask = torch.zeros(len(input_ids), dtype=torch.bool)
    for position in range(max(prompt_len - 1, 0), len(input_ids) - 1):
        text = tokenizer.decode([input_ids[position + 1]])
        is_digit = any(char.isdigit() for char in text)
        is_operator = any(char in ARITHMETIC_OPERATORS for char in text)
        mask[position] = is_digit if gate == "digit" else is_operator if gate == "operator" else is_digit or is_operator
    return mask


def _batch_token_training_gate_mask(input_ids: torch.Tensor, tokenizer, gate: str) -> torch.Tensor:
    return torch.stack([
        _token_training_gate_mask(row.tolist(), 1, tokenizer, gate) for row in input_ids.detach().cpu()
    ]).to(input_ids.device)


def _transform_hidden(hidden: torch.Tensor, basis: torch.Tensor, mode: str, noise: torch.Tensor | None = None) -> torch.Tensor:
    basis = basis.to(device=hidden.device, dtype=hidden.dtype)
    dominant = (hidden @ basis.T) @ basis
    complement = hidden - dominant
    if mode == "dominant":
        return dominant
    if mode == "complement":
        return complement
    if mode == "random-complement":
        if noise is None:
            noise = torch.randn_like(hidden)
        noise = noise.to(device=hidden.device, dtype=hidden.dtype)
        noise = noise - (noise @ basis.T) @ basis
        noise *= complement.norm(dim=-1, keepdim=True) / noise.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        return dominant + noise
    raise ValueError(f"Unknown transform mode: {mode}")


def _intervene_training_hidden(
    hidden: torch.Tensor,
    basis: torch.Tensor,
    mode: str,
    generator: torch.Generator | None = None,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if mode == "full":
        return hidden
    basis = basis.to(device=hidden.device, dtype=hidden.dtype)
    dominant = (hidden @ basis.T) @ basis
    if mode == "complement-zero":
        transformed = dominant
    elif mode == "dominant-zero":
        transformed = hidden - dominant
    elif mode == "complement-randomized":
        complement_norm = (hidden - dominant).norm(dim=-1, keepdim=True).detach()
        noise = torch.randn(hidden.shape, generator=generator, device="cpu", dtype=torch.float32).to(hidden)
        noise = noise - (noise @ basis.T) @ basis
        noise = noise * (complement_norm / noise.norm(dim=-1, keepdim=True).clamp_min(1e-9))
        transformed = dominant + noise.detach()
    else:
        raise ValueError(f"Unknown training intervention: {mode}")
    if mask is None:
        return transformed
    if mask.shape != hidden.shape[:-1]:
        raise ValueError(f"Intervention mask shape {tuple(mask.shape)} does not match hidden rows {tuple(hidden.shape[:-1])}")
    return torch.where(mask.to(device=hidden.device, dtype=torch.bool).unsqueeze(-1), transformed, hidden)


def _training_arm_basis(basis: torch.Tensor, mode: str, seed: int) -> tuple[torch.Tensor, str]:
    if not mode.startswith("random-"):
        return basis, mode
    generator = torch.Generator().manual_seed(seed)
    random_basis = torch.linalg.qr(torch.randn(basis.size(1), basis.size(0), generator=generator)).Q.T
    mapped_mode = {
        "random-zero": "complement-zero",
        "random-dominant-zero": "dominant-zero",
        "random-randomized": "complement-randomized",
    }.get(mode)
    if mapped_mode is None:
        raise ValueError(f"Unknown random training control: {mode}")
    return random_basis.to(basis), mapped_mode


def _model_layers(model) -> torch.nn.ModuleList:
    candidate = model
    for _ in range(6):
        if hasattr(candidate, "gpt_neox"):
            return candidate.gpt_neox.layers
        if hasattr(candidate, "layers"):
            return candidate.layers
        if not hasattr(candidate, "model"):
            break
        candidate = candidate.model
    raise TypeError(f"Unsupported causal-LM layer layout: {type(model).__name__}")


def register_training_intervention(
    model,
    basis: torch.Tensor,
    layer_index: int,
    mode: str,
    seed: int,
    mask_provider: Callable[[], torch.Tensor | None] | None = None,
):
    generator = torch.Generator().manual_seed(seed)
    basis, transform_mode = _training_arm_basis(basis, mode, seed)

    def pre_hook(_module, args):
        mask = mask_provider() if mask_provider is not None else None
        return (_intervene_training_hidden(args[0], basis, transform_mode, generator, mask), *args[1:])

    return _model_layers(model)[layer_index].register_forward_pre_hook(pre_hook)


def _load_revision(model_name: str, revision: str, device: torch.device, dtype: torch.dtype):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, revision=revision, dtype=dtype).to(device)
    model.eval()
    model.requires_grad_(False)
    return model, tokenizer


def _collect_c4_windows(tokenizer, num_windows: int, seq_len: int, seed: int, split: str = "validation") -> list[list[int]]:
    from datasets import load_dataset

    dataset = load_dataset("allenai/c4", "en", split=split, streaming=True)
    dataset = dataset.shuffle(seed=seed, buffer_size=2000)
    generic_windows = []
    numeric_windows = []
    numeric_target = num_windows // 2
    generic_target = num_windows - numeric_target
    for row in dataset:
        ids = tokenizer(row["text"], add_special_tokens=False)["input_ids"]
        if len(ids) >= seq_len + 1:
            window = ids[: seq_len + 1]
            digit_tokens = sum(any(char.isdigit() for char in tokenizer.decode([token])) for token in window[1:])
            if digit_tokens >= 2 and len(numeric_windows) < numeric_target:
                numeric_windows.append(window)
            elif len(generic_windows) < generic_target:
                generic_windows.append(window)
        if len(numeric_windows) >= numeric_target and len(generic_windows) >= generic_target:
            break
    windows = generic_windows + numeric_windows
    if len(windows) < num_windows:
        raise RuntimeError(f"Only collected {len(windows)} C4 windows; requested {num_windows}")
    return windows


def _causal_metrics(logits: torch.Tensor, input_ids: torch.Tensor) -> tuple[torch.Tensor, float]:
    logits = logits[:, :-1].float()
    targets = input_ids[:, 1:]
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
    accuracy = float((logits.argmax(-1) == targets).float().mean())
    return loss, accuracy


def _forward_training_intervention(
    model,
    input_ids: torch.Tensor,
    basis: torch.Tensor,
    layer_index: int,
    mode: str,
    seed: int,
    mask: torch.Tensor | None = None,
):
    handle = register_training_intervention(
        model, basis, layer_index, mode, seed,
        mask_provider=(lambda: mask) if mask is not None else None,
    )
    try:
        return model(input_ids, use_cache=False)
    finally:
        handle.remove()


def _evaluate_foundation_arm(
    model,
    tokenizer,
    windows: list[list[int]],
    basis: torch.Tensor,
    layer_index: int,
    mode: str,
    batch_size: int,
    seed: int,
    intervention_gate: str,
) -> dict[str, float]:
    model.eval()
    clean_losses, conditioned_losses = [], []
    clean_correct = conditioned_correct = tokens = 0
    with torch.no_grad():
        for begin in range(0, len(windows), batch_size):
            batch = torch.tensor(windows[begin:begin + batch_size], device=next(model.parameters()).device)
            clean = model(batch, use_cache=False).logits
            mask = (
                _batch_token_training_gate_mask(batch, tokenizer, intervention_gate)
                if intervention_gate != "all" else None
            )
            conditioned = _forward_training_intervention(
                model, batch, basis, layer_index, mode, seed + begin, mask
            ).logits
            clean_loss, clean_accuracy = _causal_metrics(clean, batch)
            conditioned_loss, conditioned_accuracy = _causal_metrics(conditioned, batch)
            batch_tokens = batch.size(0) * (batch.size(1) - 1)
            clean_losses.append(float(clean_loss) * batch_tokens)
            conditioned_losses.append(float(conditioned_loss) * batch_tokens)
            clean_correct += round(clean_accuracy * batch_tokens)
            conditioned_correct += round(conditioned_accuracy * batch_tokens)
            tokens += batch_tokens
    model.train()
    return {
        "clean_val_loss": sum(clean_losses) / tokens,
        "clean_val_accuracy": clean_correct / tokens,
        "conditioned_val_loss": sum(conditioned_losses) / tokens,
        "conditioned_val_accuracy": conditioned_correct / tokens,
    }


def foundational_training_ablation_sweep(
    model_name: str,
    basis_path: str | Path,
    output_dir: str | Path,
    revision: str = "step0",
    modes: tuple[str, ...] = ("full", "complement-zero", "complement-randomized", "random-zero", "random-randomized"),
    max_steps: int = 4000,
    learning_rate: float = 3e-4,
    batch_size: int = 8,
    seq_len: int = 128,
    train_windows: int = 32768,
    eval_windows: int = 64,
    eval_interval: int = 100,
    layer_index: int | None = None,
    device_str: str = "auto",
    seed: int = 1337,
    intervention_gate: str = "all",
) -> dict:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device(device_str if device_str != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if intervention_gate not in {"all", "digit", "operator", "digit-or-operator"}:
        raise ValueError(f"Unsupported foundational training gate: {intervention_gate}")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    train_rows = _collect_c4_windows(tokenizer, train_windows, seq_len, seed, split="train")
    eval_rows = _collect_c4_windows(tokenizer, eval_windows, seq_len, seed + 1, split="validation")
    (output / "train_windows.jsonl").write_text("\n".join(json.dumps({"input_ids": row}) for row in train_rows) + "\n", encoding="utf-8")
    (output / "eval_windows.jsonl").write_text("\n".join(json.dumps({"input_ids": row}) for row in eval_rows) + "\n", encoding="utf-8")
    basis = torch.load(basis_path, map_location=device, weights_only=True)["basis_full"].to(device, torch.float32)
    required_windows = max_steps * batch_size
    if train_windows < required_windows:
        raise ValueError(f"train_windows must be at least max_steps * batch_size ({required_windows}) for without-replacement training")
    shared_order = torch.randperm(train_windows, generator=torch.Generator().manual_seed(seed + 11)).tolist()
    summaries = []
    for mode in modes:
        torch.manual_seed(seed)
        model = AutoModelForCausalLM.from_pretrained(model_name, revision=revision, dtype=torch.float32).to(device)
        model.train()
        model.requires_grad_(True)
        tap = len(_model_layers(model)) // 2 if layer_index is None else layer_index
        arm_basis, transform_mode = _training_arm_basis(basis, mode, seed + 17)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
        arm_output = output / mode
        arm_output.mkdir(parents=True, exist_ok=True)
        metrics_path = arm_output / "metrics.jsonl"
        train_seconds = 0.0
        trained_tokens = 0
        with metrics_path.open("w", encoding="utf-8") as metrics:
            initial = _evaluate_foundation_arm(
                model, tokenizer, eval_rows, arm_basis, tap, transform_mode, batch_size,
                seed + 100_000, intervention_gate,
            )
            initial_record = {"step": 0, "train_loss": None, "train_seconds": 0.0, "trained_tokens": 0, "tokens_per_second": None, **initial}
            metrics.write(json.dumps(initial_record) + "\n")
            print(json.dumps({"mode": mode, **initial_record}), flush=True)
            for step in range(1, max_steps + 1):
                begin = (step - 1) * batch_size
                indices = shared_order[begin:begin + batch_size]
                batch = torch.tensor([train_rows[index] for index in indices], device=device)
                tick = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                mask = (
                    _batch_token_training_gate_mask(batch, tokenizer, intervention_gate)
                    if intervention_gate != "all" else None
                )
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    outputs = _forward_training_intervention(
                        model, batch, arm_basis, tap, transform_mode, seed + step, mask
                    )
                    loss, _ = _causal_metrics(outputs.logits, batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_seconds += time.perf_counter() - tick
                trained_tokens += batch.size(0) * (batch.size(1) - 1)
                if step % eval_interval == 0 or step == max_steps:
                    validation = _evaluate_foundation_arm(
                        model, tokenizer, eval_rows, arm_basis, tap, transform_mode, batch_size,
                        seed + 100_000 + step, intervention_gate,
                    )
                    record = {
                        "step": step,
                        "train_loss": float(loss),
                        "train_seconds": train_seconds,
                        "trained_tokens": trained_tokens,
                        "tokens_per_second": trained_tokens / train_seconds,
                        "intervention_gate": intervention_gate,
                        "active_rows": int(mask.sum()) if mask is not None else batch.numel(),
                        **validation,
                    }
                    metrics.write(json.dumps(record) + "\n")
                    metrics.flush()
                    print(json.dumps({"mode": mode, **record}), flush=True)
        model.save_pretrained(arm_output / "model")
        final = json.loads(metrics_path.read_text(encoding="utf-8").splitlines()[-1])
        summary = {
            "mode": mode, "tap_layer": tap, "model": model_name, "revision": revision,
            "intervention_gate": intervention_gate, **final,
        }
        (arm_output / "run.summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        summaries.append(summary)
        del optimizer, model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    (output / "sweep.summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    return {"rows": summaries, "output_dir": str(output)}


def _capture_tap(model, layer_index: int, input_ids: torch.Tensor, requires_grad: bool = False):
    captured: dict[str, torch.Tensor] = {}

    def pre_hook(_module, args):
        hidden = args[0].detach()
        if requires_grad:
            hidden.requires_grad_(True)
        captured["hidden"] = hidden
        return (hidden, *args[1:])

    handle = _model_layers(model)[layer_index].register_forward_pre_hook(pre_hook)
    try:
        outputs = model(input_ids, output_hidden_states=True, use_cache=False)
    finally:
        handle.remove()
    return outputs, captured["hidden"]


def _fit_influence_basis(
    model,
    windows: list[list[int]],
    layer_index: int,
    rank: int,
    positions_per_window: int,
    directions: int,
    seed: int,
    position_ranges: list[tuple[int, int]] | None = None,
    include_cross_position: bool = False,
) -> tuple[torch.Tensor, dict[str, float]]:
    device = next(model.parameters()).device
    generator = torch.Generator().manual_seed(seed)
    rows = []
    cross_mass = []
    offdiag_rows = []
    for window_index, ids in enumerate(windows):
        batch = torch.tensor([ids], device=device)
        outputs, tap = _capture_tap(model, layer_index, batch, requires_grad=True)
        final = outputs.hidden_states[-1].float()
        low, high = position_ranges[window_index] if position_ranges is not None else (0, len(ids) - 1)
        positions = (torch.randperm(high - low, generator=generator)[:positions_per_window] + low).tolist()
        for position in positions:
            for _ in range(directions):
                direction = torch.randn(final.size(-1), generator=generator).to(device)
                direction /= direction.norm()
                scalar = (final[0, position] * direction).sum()
                (gradient,) = torch.autograd.grad(scalar, tap, retain_graph=True)
                local = gradient[0, position].float()
                total_energy = gradient.float().pow(2).sum().clamp_min(1e-30)
                cross_mass.append(float(1.0 - local.pow(2).sum() / total_energy))
                rows.append(local.cpu())
                if include_cross_position and position > 0:
                    earlier_norms = gradient[0, :position].float().norm(dim=-1)
                    for earlier in earlier_norms.topk(min(3, earlier_norms.numel())).indices.tolist():
                        offdiag_rows.append(gradient[0, earlier].float().cpu())
    fitted_rows = rows + offdiag_rows
    stacked = torch.stack(fitted_rows)
    stacked /= stacked.norm(dim=-1, keepdim=True).clamp_min(1e-30)
    _, spectrum, v_rows = torch.linalg.svd(stacked, full_matrices=False)
    energy = spectrum.square()
    effective_rank = float(energy.sum().square() / energy.square().sum())
    fitted_rank = min(rank, v_rows.size(0))
    return v_rows[:fitted_rank], {
        "effective_rank": effective_rank,
        "energy_at_rank": float(energy[:fitted_rank].sum() / energy.sum()),
        "cross_position_mass": sum(cross_mass) / len(cross_mass),
        "basis_rows": len(fitted_rows),
        "local_basis_rows": len(rows),
        "offdiag_basis_rows": len(offdiag_rows),
    }


def _forward_with_transform(model, input_ids: torch.Tensor, layer_index: int, transform: Callable[[torch.Tensor], torch.Tensor]):
    def pre_hook(_module, args):
        return (transform(args[0]), *args[1:])

    handle = _model_layers(model)[layer_index].register_forward_pre_hook(pre_hook)
    try:
        return model(input_ids, use_cache=False).logits
    finally:
        handle.remove()


def _probe_checkpoint_content(model, tokenizer, windows, basis, layer_index: int, seed: int) -> dict[str, dict[str, float]]:
    device = next(model.parameters()).device
    generator = torch.Generator().manual_seed(seed)
    names = ("full", "dominant", "complement", "noise", "shuffled-complement")
    counts = {name: {"tokens": 0, "correct": 0, "digit_targets": 0, "digit_correct": 0, "digit_decode": 0} for name in names}
    activation_sum = 0.0
    activation_count = 0
    for ids in windows:
        batch = torch.tensor([ids], device=device)
        with torch.no_grad():
            reference, tap = _capture_tap(model, layer_index, batch)
            hidden = tap.float()
            dominant = (hidden @ basis.to(device).T) @ basis.to(device)
            complement = hidden - dominant
            activation_sum += float(complement.square().sum())
            activation_count += int(hidden.numel() / hidden.size(-1))
            noise = torch.randn(hidden.shape, generator=generator).to(device)
            noise *= complement.norm(dim=-1, keepdim=True) / noise.norm(dim=-1, keepdim=True).clamp_min(1e-9)
            permutation = torch.randperm(hidden.size(1), generator=generator).to(device)
            logits = {
                "full": reference.logits,
                "dominant": _forward_with_transform(model, batch, layer_index, lambda _h: dominant.to(_h.dtype)),
                "complement": _forward_with_transform(model, batch, layer_index, lambda _h: complement.to(_h.dtype)),
                "noise": _forward_with_transform(model, batch, layer_index, lambda _h: noise.to(_h.dtype)),
                "shuffled-complement": _forward_with_transform(model, batch, layer_index, lambda _h: complement[:, permutation].to(_h.dtype)),
            }
        targets = ids[1:]
        digit_mask = [any(char.isdigit() for char in tokenizer.decode([token])) for token in targets]
        for name in names:
            predicted = logits[name][0, :-1].argmax(-1).tolist()
            for prediction, target, is_digit in zip(predicted, targets, digit_mask):
                counts[name]["tokens"] += 1
                counts[name]["correct"] += prediction == target
                if is_digit:
                    counts[name]["digit_targets"] += 1
                    counts[name]["digit_correct"] += prediction == target
                    counts[name]["digit_decode"] += any(char.isdigit() for char in tokenizer.decode([prediction]))
    results = {}
    for name, row in counts.items():
        results[name] = {
            "token_accuracy": row["correct"] / max(row["tokens"], 1),
            "digit_token_accuracy": row["digit_correct"] / max(row["digit_targets"], 1),
            "digit_decode_rate": row["digit_decode"] / max(row["digit_targets"], 1),
            "tokens": row["tokens"],
            "digit_targets": row["digit_targets"],
        }
    results["complement"]["activation_energy_per_position"] = activation_sum / max(activation_count, 1)
    return results


def foundational_pretraining_sweep(
    model_name: str,
    checkpoints: list[tuple[int, str]],
    output_dir: str | Path,
    num_windows: int = 64,
    seq_len: int = 128,
    basis_windows: int = 12,
    positions_per_window: int = 4,
    directions: int = 2,
    rank: int = 64,
    layer_index: int | None = None,
    device_str: str = "auto",
    seed: int = 1337,
) -> dict:
    device = torch.device(device_str if device_str != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    windows = None
    start = time.time()
    for step, revision in sorted(checkpoints):
        model, tokenizer = _load_revision(model_name, revision, device, dtype)
        if windows is None:
            windows = _collect_c4_windows(tokenizer, num_windows, seq_len, seed)
            (output / "c4_windows.jsonl").write_text("\n".join(json.dumps({"input_ids": row}) for row in windows) + "\n", encoding="utf-8")
        tap = len(_model_layers(model)) // 2 if layer_index is None else layer_index
        basis, geometry = _fit_influence_basis(model, windows[:basis_windows], tap, rank, positions_per_window, directions, seed)
        torch.save({"basis_full": basis, "rank": basis.size(0), "tap_layer": tap, "model": model_name, "revision": revision}, output / f"basis-step-{step}.pt")
        content = _probe_checkpoint_content(model, tokenizer, windows, basis, tap, seed)
        row = {"step": step, "revision": revision, "tap_layer": tap, **geometry, "streams": content, "elapsed_seconds": time.time() - start}
        rows.append(row)
        print(json.dumps(row), flush=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    (output / "checkpoint_metrics.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    _write_foundation_report(model_name, rows, output)
    return {"rows": rows, "output_dir": str(output)}


def _write_foundation_report(model_name: str, rows: list[dict], output: Path) -> None:
    lines = [
        "# Foundational pretraining ontogeny", "", f"Model: `{model_name}`", "",
        "Each checkpoint uses a checkpoint-local rank-64 influence basis and identical held-out C4 windows.", "",
        "| step | eff. rank | energy@64 | cross-pos mass | full digit acc | complement digit acc | shuffled digit acc | noise digit acc | complement digit rate |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        streams = row["streams"]
        lines.append(f"| {row['step']} | {row['effective_rank']:.2f} | {row['energy_at_rank']:.3f} | {row['cross_position_mass']:.3f} | {streams['full']['digit_token_accuracy']:.3f} | {streams['complement']['digit_token_accuracy']:.3f} | {streams['shuffled-complement']['digit_token_accuracy']:.3f} | {streams['noise']['digit_token_accuracy']:.3f} | {streams['complement']['digit_decode_rate']:.3f} |")
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    payload = json.dumps([
        {"x": [math.log10(row["step"] + 1) for row in rows], "y": [row["streams"][stream][metric] for row in rows], "name": label, "mode": "lines+markers", "text": [str(row["step"]) for row in rows]}
        for stream, metric, label in (("full", "digit_token_accuracy", "full digit accuracy"), ("complement", "digit_token_accuracy", "complement digit accuracy"), ("shuffled-complement", "digit_token_accuracy", "shuffled complement"), ("noise", "digit_token_accuracy", "noise digit accuracy"), ("complement", "digit_decode_rate", "complement digit rate"))
    ])
    html = f"""<!doctype html><meta charset="utf-8"><script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script><div id="plot" style="height:760px"></div><script>Plotly.newPlot('plot',{payload},{{template:'plotly_white',xaxis:{{title:'log10(pretraining step + 1)'}},yaxis:{{title:'rate',range:[0,1]}},hovermode:'x unified'}});</script>"""
    (output / "plot.html").write_text(html, encoding="utf-8")


def _transform_generation_row(hidden: torch.Tensor, basis: torch.Tensor, mode: str, generator: torch.Generator, random_basis: torch.Tensor | None):
    if mode == "full":
        return hidden
    if mode == "complement-zero":
        return _transform_hidden(hidden, basis, "dominant")
    if mode == "dominant-zero":
        return _transform_hidden(hidden, basis, "complement")
    if mode == "complement-randomized":
        noise = torch.randn(hidden.shape, generator=generator).to(hidden.device)
        return _transform_hidden(hidden, basis, "random-complement", noise)
    if mode.startswith("complement-scale-"):
        scale = float(mode.removeprefix("complement-scale-"))
        dominant = _transform_hidden(hidden, basis, "dominant")
        return dominant + scale * (hidden - dominant)
    if mode.startswith("complement-mix-"):
        amount = float(mode.removeprefix("complement-mix-"))
        noise = torch.randn(hidden.shape, generator=generator).to(hidden.device)
        randomized = _transform_hidden(hidden, basis, "random-complement", noise)
        return (1.0 - amount) * hidden + amount * randomized
    if mode.startswith("random-scale-"):
        scale = float(mode.removeprefix("random-scale-"))
        random_basis_device = random_basis.to(hidden.device)
        random_dominant = _transform_hidden(hidden, random_basis_device, "dominant")
        return random_dominant + scale * (hidden - random_dominant)
    if mode == "random-keep":
        return _transform_hidden(hidden, random_basis.to(hidden.device), "dominant")
    raise ValueError(mode)


def _batched_generation_transform_hook(layer, basis: torch.Tensor, modes: tuple[str, ...], generators: list[torch.Generator]):
    random_bases = []
    for mode, generator in zip(modes, generators):
        if mode.startswith("random-scale-") or mode == "random-keep":
            random_bases.append(torch.linalg.qr(torch.randn(basis.size(1), basis.size(0), generator=generator)).Q.T)
        else:
            random_bases.append(None)

    def hook(_module, args):
        hidden = args[0]
        transformed = torch.cat([
            _transform_generation_row(hidden[index:index + 1], basis, mode, generator, random_basis)
            for index, (mode, generator, random_basis) in enumerate(zip(modes, generators, random_bases))
        ])
        return (transformed, *args[1:])

    return layer.register_forward_pre_hook(hook)


def causal_ablation_sweep(
    phase_models: list[tuple[int, str]],
    basis_path: str | Path,
    output_dir: str | Path,
    num_problems: int = 200,
    max_new_tokens: int = 512,
    tap_layer: int = 12,
    device_str: str = "auto",
    seed: int = 1337,
) -> dict:
    from datasets import load_dataset

    device = torch.device(device_str if device_str != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    basis = torch.load(basis_path, map_location=device, weights_only=True)["basis_full"].to(device, torch.float32)
    raw = load_dataset("openai/gsm8k", "main", split="test").select(range(num_problems))
    problems = [{"question": row["question"], "gold": extract_answer_lenient(row["answer"])} for row in raw]
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    modes = (
        "full", "complement-scale-0.9", "complement-scale-0.75", "complement-mix-0.1",
        "random-scale-0.9", "random-scale-0.75", "complement-zero", "dominant-zero",
        "complement-randomized", "random-keep",
    )
    rows = []
    problem_rows = []
    for step, model_spec in sorted(phase_models):
        model, tokenizer = _load_phase(model_spec, device, dtype)
        layer = _model_layers(model)[tap_layer]
        mode_hits = {mode: [] for mode in modes}
        for index, problem in enumerate(problems):
            encoded = tokenizer(_chat_prompt(tokenizer, COT_PROMPT + problem["question"]), return_tensors="pt").to(device)
            record = {"step": step, "index": index, "gold": problem["gold"]}
            generators = [torch.Generator().manual_seed(seed + index * 101 + mode_index) for mode_index in range(len(modes))]
            handle = _batched_generation_transform_hook(layer, basis, modes, generators)
            batched = {key: value.repeat(len(modes), 1) for key, value in encoded.items()}
            try:
                with torch.no_grad():
                    generated = model.generate(**batched, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
            finally:
                handle.remove()
            for mode, output_ids in zip(modes, generated):
                completion = tokenizer.decode(output_ids[encoded.input_ids.size(1):], skip_special_tokens=True)
                answer = extract_answer_lenient(completion)
                hit = answer == problem["gold"]
                mode_hits[mode].append(hit)
                record[mode] = {"answer": answer, "correct": hit}
            problem_rows.append(record)
            if (index + 1) % 10 == 0:
                print(json.dumps({"step": step, "problems": index + 1, **{mode: sum(hits) / len(hits) for mode, hits in mode_hits.items()}}), flush=True)
        full_accuracy = sum(mode_hits["full"]) / len(problems)
        row = {"step": step, "model": model_spec, "problems": len(problems), "full_accuracy": full_accuracy}
        for mode in modes[1:]:
            accuracy = sum(mode_hits[mode]) / len(problems)
            differences = [float(test) - float(base) for test, base in zip(mode_hits[mode], mode_hits["full"])]
            generator = torch.Generator().manual_seed(seed + step + len(mode))
            bootstrap = []
            for _ in range(5000):
                indices = torch.randint(len(differences), (len(differences),), generator=generator).tolist()
                bootstrap.append(sum(differences[index] for index in indices) / len(indices))
            row[f"{mode}_accuracy"] = accuracy
            row[f"{mode}_delta"] = accuracy - full_accuracy
            row[f"{mode}_retention"] = accuracy / full_accuracy if full_accuracy else None
            row[f"{mode}_delta_ci95"] = [_percentile(bootstrap, 0.025), _percentile(bootstrap, 0.975)]
        rows.append(row)
        print(json.dumps(row), flush=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    (output / "phase_metrics.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    (output / "problem_outcomes.jsonl").write_text("\n".join(json.dumps(row) for row in problem_rows) + "\n", encoding="utf-8")
    _write_ablation_report(rows, output)
    return {"rows": rows, "output_dir": str(output)}


def _write_ablation_report(rows: list[dict], output: Path) -> None:
    def retention(row: dict, mode: str) -> str:
        value = row[f"{mode}_retention"]
        return "n/a" if value is None else f"{value:.3f}"

    lines = [
        "# Causal complement ablation across task adaptation", "",
        "Graded interventions avoid the saturation of deleting roughly 82% of residual-stream energy.", "",
        "| step | full | comp x0.9 | random x0.9 | comp x0.75 | random x0.75 | 10% comp noise |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(f"| {row['step']} | {row['full_accuracy']:.3f} | {row['complement-scale-0.9_accuracy']:.3f} | {row['random-scale-0.9_accuracy']:.3f} | {row['complement-scale-0.75_accuracy']:.3f} | {row['random-scale-0.75_accuracy']:.3f} | {row['complement-mix-0.1_accuracy']:.3f} |")
    lines += ["", "Full-strength saturation controls:", "", "| step | complement zero | dominant zero | randomized complement | random keep-64 |", "| ---: | ---: | ---: | ---: | ---: |"]
    for row in rows:
        lines.append(f"| {row['step']} | {row['complement-zero_accuracy']:.3f} | {row['dominant-zero_accuracy']:.3f} | {row['complement-randomized_accuracy']:.3f} | {row['random-keep_accuracy']:.3f} |")
    lines += ["", "Graded retention relative to the unmodified checkpoint:", "", "| step | comp x0.9 | random x0.9 | comp x0.75 | random x0.75 | 10% comp noise |", "| ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in rows:
        lines.append(f"| {row['step']} | {retention(row, 'complement-scale-0.9')} | {retention(row, 'random-scale-0.9')} | {retention(row, 'complement-scale-0.75')} | {retention(row, 'random-scale-0.75')} | {retention(row, 'complement-mix-0.1')} |")
    lines += [
        "", "Paired accuracy deltas and 95% bootstrap confidence intervals:", "",
        "| step | comp x0.75 | random x0.75 | 10% comp noise |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        cells = []
        for mode in ("complement-scale-0.75", "random-scale-0.75", "complement-mix-0.1"):
            low, high = row[f"{mode}_delta_ci95"]
            cells.append(f"{row[f'{mode}_delta']:+.3f} [{low:+.3f}, {high:+.3f}]")
        lines.append(f"| {row['step']} | {' | '.join(cells)} |")
    complement_penalties = [-row["complement-scale-0.75_delta"] for row in rows]
    monotonic = all(right >= left for left, right in zip(complement_penalties, complement_penalties[1:]))
    lines += [
        "", "## Interpretation", "",
        f"The 25% complement-attenuation penalty is {'monotonic' if monotonic else 'not monotonic'} across task adaptation "
        f"({', '.join(f'{100 * value:.1f}' for value in complement_penalties)} accuracy points).",
        "Matched random-subspace attenuation is at least as damaging at every checkpoint, so the graded deletion result does not establish complement-specific causal dependence.",
        "Full complement deletion, dominant deletion, complement randomization, and a random keep-64 control all collapse accuracy; these are saturation controls, not evidence of specificity.",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")