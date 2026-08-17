"""Baselines for the pretrained-LM retrofit: few-shot CoT, self-consistency@k,
and a parameter-matched LoRA fine-tune of the trunk.

These close the "is the baseline overly weak?" question before scaling:
  retrofit-baseline-eval   Few-shot / sampled / majority-vote trunk baselines,
                           optionally with a LoRA adapter loaded.
  retrofit-lora-train      LoRA fine-tune on the same harvested traces with the
                           same answer-weighted CE as the corrector.
All scoring uses the same strict and lenient extraction as retrofit-eval.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

import torch
from torch.nn import functional as F

from prometheus.retrofit import (
    COT_PROMPT,
    _answer_start_index,
    _chat_prompt,
    extract_answer,
    extract_answer_lenient,
    load_trunk,
)

FEWSHOT_EXEMPLARS = [
    (
        "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. "
        "How many clips did Natalia sell altogether in April and May?",
        "In April, Natalia sold 48 clips.\nIn May, she sold half as many, so she sold 48 / 2 = 24 clips.\n"
        "Altogether she sold 48 + 24 = 72 clips.\n#### 72",
    ),
    (
        "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. "
        "How much did she earn?",
        "Weng earns 12 / 60 = $0.2 per minute.\nFor 50 minutes, she earned 0.2 x 50 = $10.\n#### 10",
    ),
    (
        "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. "
        "Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. "
        "How much more money does Betty need to buy the wallet?",
        "Betty has 100 / 2 = $50.\nHer grandparents gave her 15 * 2 = $30.\n"
        "In total she has 50 + 15 + 30 = $95.\nShe still needs 100 - 95 = $5.\n#### 5",
    ),
    (
        "Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she read twice "
        "as many pages as yesterday. If she wants to read half of the remaining pages tomorrow, how many pages "
        "should she read?",
        "Today, Julie read 12 * 2 = 24 pages.\nSo far she has read 12 + 24 = 36 pages.\n"
        "There are 120 - 36 = 84 pages left.\nHalf of the remaining pages is 84 / 2 = 42 pages.\n#### 42",
    ),
]


def _fewshot_prompt(question: str, shots: int) -> str:
    """Build the instruction text with `shots` worked exemplars prepended."""

    parts = [
        "Solve the math problem step by step. End your response with the final "
        "numeric answer on its own line in the form '#### <answer>'.",
    ]
    for exemplar_question, exemplar_solution in FEWSHOT_EXEMPLARS[:shots]:
        parts.append(f"\nProblem: {exemplar_question}\n{exemplar_solution}")
    parts.append(f"\nProblem: {question}")
    return "\n".join(parts)


def _load_model(model_name: str, device: torch.device, lora_dir: str | None):
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model, tokenizer = load_trunk(model_name, device, dtype)
    if lora_dir is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, lora_dir)
        model = model.merge_and_unload()
        model.eval()
    return model, tokenizer


def _sidecar_high_training_gate_mask(
    hidden: torch.Tensor,
    corrector,
    prompt_len: int,
    threshold_z: float,
) -> torch.Tensor:
    """Select completion rows whose frozen-sidecar delta norm is a within-trace excursion."""

    with torch.no_grad():
        scores = corrector(hidden.float()).norm(dim=-1)
    valid = torch.zeros_like(scores, dtype=torch.bool)
    valid[:, max(prompt_len - 1, 0) : hidden.size(1) - 1] = True
    selected = torch.zeros_like(valid)
    for batch_index in range(hidden.size(0)):
        values = scores[batch_index][valid[batch_index]]
        if values.numel() == 0:
            continue
        cutoff = values.mean() + threshold_z * values.std(unbiased=False)
        selected[batch_index] = valid[batch_index] & (scores[batch_index] >= cutoff)
    return selected


def _precompute_training_gate_masks(
    model,
    tokenizer,
    traces: list[dict],
    max_seq_len: int,
    tap_layer: int,
    gate: str,
    gate_corrector_path: str | Path | None,
    gate_threshold_z: float,
    device: torch.device,
) -> tuple[list[torch.Tensor], dict[str, float | int | str]]:
    corrector = None
    if gate == "sidecar-high":
        if gate_corrector_path is None:
            raise ValueError("gate_corrector_path is required for the sidecar-high training gate")
        from prometheus.retrofit import load_corrector

        checkpoint = torch.load(gate_corrector_path, map_location=device, weights_only=True)
        corrector, corrector_tap = load_corrector(checkpoint, device)
        corrector.requires_grad_(False)
        if corrector_tap != tap_layer:
            raise ValueError(f"Gate corrector tap layer {corrector_tap} does not match intervention layer {tap_layer}")
    elif gate not in {"digit", "operator", "digit-or-operator"}:
        raise ValueError(f"Unknown training intervention gate: {gate}")

    from prometheus.ontogeny_experiments import _token_training_gate_mask

    masks = []
    active_rows = eligible_rows = 0
    was_training = model.training
    model.eval()
    for trace in traces:
        prompt_ids = tokenizer(trace["prompt"], add_special_tokens=False)["input_ids"]
        completion_ids = tokenizer(trace["completion"], add_special_tokens=False)["input_ids"]
        input_ids = (prompt_ids + completion_ids)[:max_seq_len]
        if gate == "sidecar-high":
            batch = torch.tensor([input_ids], device=device)
            with torch.no_grad():
                outputs = model(batch, output_hidden_states=True, use_cache=False, logits_to_keep=1)
            mask = _sidecar_high_training_gate_mask(
                outputs.hidden_states[tap_layer], corrector, len(prompt_ids), gate_threshold_z
            )[0].cpu()
        else:
            mask = _token_training_gate_mask(input_ids, len(prompt_ids), tokenizer, gate)
        masks.append(mask)
        active_rows += int(mask.sum())
        eligible_rows += max(len(input_ids) - len(prompt_ids), 0)
    model.train(was_training)
    return masks, {
        "gate": gate,
        "active_rows": active_rows,
        "eligible_rows": eligible_rows,
        "active_fraction": active_rows / max(eligible_rows, 1),
    }


def evaluate_baseline(
    model_name: str,
    num_problems: int,
    max_new_tokens: int,
    device_str: str,
    output_path: str | Path | None,
    shots: int = 0,
    samples: int = 1,
    temperature: float = 0.7,
    lora_dir: str | None = None,
) -> dict:
    """Evaluate a trunk baseline on GSM8K test.

    shots=0, samples=1  -> the original zero-shot greedy CoT baseline.
    shots=4, samples=1  -> few-shot greedy CoT.
    samples=k>1         -> self-consistency@k (temperature sampling, majority
                           vote over lenient answers); emitted tokens count all
                           k sampled chains (the matched-budget comparator).
    lora_dir            -> evaluate the LoRA fine-tuned trunk instead.
    """

    from datasets import load_dataset

    device = torch.device(device_str if device_str != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    model, tokenizer = _load_model(model_name, device, lora_dir)
    dataset = load_dataset("openai/gsm8k", "main", split="test").select(range(num_problems))

    strict_correct = 0
    lenient_correct = 0
    emitted_total = 0
    start = time.time()

    def score(completions: list[str], gold) -> None:
        nonlocal strict_correct, lenient_correct, emitted_total
        for completion in completions:
            emitted_total += len(tokenizer(completion, add_special_tokens=False)["input_ids"])
        if samples > 1:
            votes = Counter(
                answer for answer in (extract_answer_lenient(text) for text in completions) if answer is not None
            )
            majority = votes.most_common(1)[0][0] if votes else None
            strict_hit = majority == gold
            lenient_hit = majority == gold
        else:
            strict_hit = extract_answer(completions[0]) == gold
            lenient_hit = extract_answer_lenient(completions[0]) == gold
        strict_correct += int(strict_hit)
        lenient_correct += int(lenient_hit)

    # Resume: replay completed problems from a previous crashed run's dump.
    done_indices: set[int] = set()
    dump_sink = None
    if output_path is not None:
        dump_path = Path(output_path).with_suffix(".completions.jsonl")
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        if dump_path.exists():
            for line in dump_path.read_text(encoding="utf-8").splitlines():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row["index"] in done_indices:
                    continue
                score(row["completions"], row["gold"])
                done_indices.add(row["index"])
            if done_indices:
                print(json.dumps({"resumed_problems": len(done_indices)}), flush=True)
        dump_sink = dump_path.open("a", encoding="utf-8")

    for index, row in enumerate(dataset):
        if index in done_indices:
            continue
        gold = extract_answer(row["answer"])
        prompt = _chat_prompt(tokenizer, _fewshot_prompt(row["question"], shots))
        encoded = tokenizer(prompt, return_tensors="pt").to(device)
        generate_kwargs = dict(
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
        if samples > 1:
            generate_kwargs.update(do_sample=True, temperature=temperature, num_return_sequences=samples)
        else:
            generate_kwargs.update(do_sample=False)
        with torch.no_grad():
            generated = model.generate(**encoded, **generate_kwargs)

        completions = [
            tokenizer.decode(output_ids[encoded["input_ids"].size(1):], skip_special_tokens=True)
            for output_ids in generated
        ]
        score(completions, gold)

        if dump_sink is not None:
            dump_sink.write(
                json.dumps({"index": index, "question": row["question"], "gold": gold, "completions": completions})
                + "\n"
            )
            dump_sink.flush()
        if (index + 1) % 10 == 0:
            print(
                f"baseline {index + 1}/{num_problems} strict={strict_correct / (index + 1):.4f} "
                f"lenient={lenient_correct / (index + 1):.4f} elapsed={time.time() - start:.0f}s",
                flush=True,
            )

    if dump_sink is not None:
        dump_sink.close()
    count = len(dataset)
    label = f"shots={shots} samples={samples}" + (f" lora={lora_dir}" if lora_dir else "")
    results = {
        "label": label,
        "strict_accuracy": strict_correct / count,
        "lenient_accuracy": lenient_correct / count,
        "mean_emitted_tokens": emitted_total / count,
    }
    lines = [
        "# GSM8K trunk baseline", "",
        f"Model: `{model_name}`, problems: {count}, {label}", "",
        "| strict accuracy | lenient accuracy | mean emitted tokens |",
        "| --- | --- | --- |",
        f"| {results['strict_accuracy']:.4f} | {results['lenient_accuracy']:.4f} | {results['mean_emitted_tokens']:.1f} |",
    ]
    report = "\n".join(lines) + "\n"
    print(report, flush=True)
    if output_path is not None:
        Path(output_path).write_text(report, encoding="utf-8")
    return results


def train_lora(
    model_name: str,
    traces_path: str | Path,
    output_dir: str | Path,
    lora_r: int,
    max_steps: int,
    learning_rate: float,
    answer_weight: float,
    device_str: str,
    max_seq_len: int = 640,
    log_interval: int = 25,
    checkpoint_steps: tuple[int, ...] = (),
    seed: int = 1337,
    intervention: str = "full",
    basis_path: str | Path | None = None,
    tap_layer: int = 12,
    intervention_gate: str = "all",
    gate_corrector_path: str | Path | None = None,
    gate_threshold_z: float = 2.0,
    gate_masks_path: str | Path | None = None,
    basis_refresh_interval: int = 0,
    basis_refresh_traces: int = 8,
    basis_refresh_positions: int = 8,
    basis_refresh_directions: int = 4,
    basis_refresh_at_start: bool = False,
) -> dict:
    """LoRA fine-tune on traces, optionally saving a deterministic phase series."""

    from peft import LoraConfig, get_peft_model

    device = torch.device(device_str if device_str != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model, tokenizer = load_trunk(model_name, device, dtype)
    torch.manual_seed(seed)
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=2 * lora_r,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.0,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.float()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"LoRA trainable parameters: {trainable}", flush=True)
    model.train()

    traces = [json.loads(line) for line in Path(traces_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not traces:
        raise ValueError(f"No traces found in {traces_path}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if intervention != "full" and basis_path is None:
        raise ValueError("basis_path is required for a training intervention")
    if basis_refresh_interval < 0:
        raise ValueError("basis_refresh_interval must be nonnegative")
    if basis_refresh_interval and intervention == "full":
        raise ValueError("Dynamic basis refresh requires a measured-basis intervention")
    if basis_refresh_interval and intervention.startswith("random-"):
        raise ValueError("Dynamic basis refresh is only defined for measured-basis interventions")
    intervention_handle = None
    gate_masks = None
    gate_summary = {"gate": "all", "active_rows": 0, "eligible_rows": 0, "active_fraction": 1.0}
    current_gate: dict[str, torch.Tensor | None] = {"mask": None}
    if intervention_gate != "all":
        if gate_masks_path is not None:
            gate_payload = torch.load(gate_masks_path, map_location="cpu", weights_only=True)
            gate_masks = gate_payload["masks"]
            gate_summary = gate_payload["summary"]
            if len(gate_masks) != len(traces):
                raise ValueError(f"Gate mask count {len(gate_masks)} does not match trace count {len(traces)}")
            if gate_summary["gate"] != intervention_gate:
                raise ValueError(f"Saved gate {gate_summary['gate']} does not match requested gate {intervention_gate}")
        else:
            gate_masks, gate_summary = _precompute_training_gate_masks(
                model, tokenizer, traces, max_seq_len, tap_layer, intervention_gate,
                gate_corrector_path, gate_threshold_z, device,
            )
        torch.save({"masks": gate_masks, "summary": gate_summary}, output_dir / "gate_masks.pt")
        print(json.dumps(gate_summary), flush=True)
    if basis_path is not None:
        from prometheus.ontogeny_experiments import register_training_intervention

        basis = torch.load(basis_path, map_location=device, weights_only=True)["basis_full"].to(device, torch.float32)
        mature_basis = basis.detach().cpu()
        intervention_handle = register_training_intervention(
            model, basis, tap_layer, intervention, seed + 17,
            mask_provider=(lambda: current_gate["mask"]) if gate_masks is not None else None,
        )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=learning_rate, weight_decay=0.01
    )
    checkpoint_steps = tuple(sorted(set(checkpoint_steps)))
    invalid_steps = [step for step in checkpoint_steps if step < 0 or step > max_steps]
    if invalid_steps:
        raise ValueError(f"checkpoint steps must be between 0 and {max_steps}: {invalid_steps}")
    checkpoints_dir = output_dir / "checkpoints"
    if checkpoint_steps:
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
    if 0 in checkpoint_steps:
        model.save_pretrained(str(checkpoints_dir / "step-0"))
    generator = torch.Generator().manual_seed(seed)
    start = time.time()
    metrics_path = output_dir / "metrics.jsonl"
    refresh_records = []
    try:
        with metrics_path.open("w", encoding="utf-8") as metrics:
            for step in range(max_steps):
                refresh_due = basis_refresh_interval and (
                    (step == 0 and basis_refresh_at_start) or (step > 0 and step % basis_refresh_interval == 0)
                )
                if refresh_due:
                    from prometheus.ontogeny_experiments import _fit_influence_basis

                    intervention_handle.remove()
                    intervention_handle = None
                    refresh_windows = []
                    refresh_ranges = []
                    for trace in traces[:basis_refresh_traces]:
                        prompt_ids = tokenizer(trace["prompt"], add_special_tokens=False)["input_ids"]
                        completion_ids = tokenizer(trace["completion"], add_special_tokens=False)["input_ids"]
                        input_ids = (prompt_ids + completion_ids)[:max_seq_len]
                        if len(input_ids) <= len(prompt_ids) + 4:
                            continue
                        refresh_windows.append(input_ids)
                        refresh_ranges.append((len(prompt_ids), len(input_ids) - 1))
                    was_training = model.training
                    model.eval()
                    previous_basis = basis.detach().cpu()
                    basis, geometry = _fit_influence_basis(
                        model, refresh_windows, tap_layer, mature_basis.size(0),
                        basis_refresh_positions, basis_refresh_directions, seed + 29,
                        position_ranges=refresh_ranges, include_cross_position=True,
                    )
                    basis = basis.to(device, torch.float32)
                    if was_training:
                        model.train()
                    rank = basis.size(0)
                    record = {
                        "step": step,
                        **geometry,
                        "overlap_with_mature": float((basis.cpu() @ mature_basis.T).square().sum() / rank),
                        "overlap_with_previous": float((basis.cpu() @ previous_basis.T).square().sum() / rank),
                    }
                    refresh_records.append(record)
                    torch.save({"basis_full": basis.cpu(), "geometry": record}, output_dir / f"basis-refresh-step-{step}.pt")
                    print(json.dumps({"basis_refresh": record}), flush=True)
                    intervention_handle = register_training_intervention(
                        model, basis, tap_layer, intervention, seed + 17,
                        mask_provider=(lambda: current_gate["mask"]) if gate_masks is not None else None,
                    )
                trace_index = int(torch.randint(len(traces), (1,), generator=generator))
                trace = traces[trace_index]
                prompt_ids = tokenizer(trace["prompt"], add_special_tokens=False)["input_ids"]
                completion_ids = tokenizer(trace["completion"], add_special_tokens=False)["input_ids"]
                input_ids = (prompt_ids + completion_ids)[:max_seq_len]
                completion_len = len(input_ids) - len(prompt_ids)
                if completion_len < 4:
                    continue
                batch = torch.tensor([input_ids], device=device)
                current_gate["mask"] = gate_masks[trace_index].unsqueeze(0) if gate_masks is not None else None

                outputs = model(batch)
                logits = outputs.logits[:, len(prompt_ids) - 1 : -1, :].float()
                targets = batch[:, len(prompt_ids):]
                weights = torch.ones(targets.size(1), device=device)
                answer_start = _answer_start_index(input_ids[len(prompt_ids):], tokenizer)
                weights[answer_start:] = answer_weight
                loss_per_token = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction="none"
                )
                loss = (loss_per_token * weights).sum() / weights.sum()

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                optimizer.step()

                completed_steps = step + 1
                if completed_steps in checkpoint_steps:
                    model.save_pretrained(str(checkpoints_dir / f"step-{completed_steps}"))

                if step % log_interval == 0 or step == max_steps - 1:
                    record = {
                        "step": step,
                        "loss": float(loss),
                        "elapsed_seconds": time.time() - start,
                        "intervention": intervention,
                        "intervention_gate": intervention_gate,
                        "active_rows": int(current_gate["mask"].sum()) if current_gate["mask"] is not None else input_ids.__len__(),
                    }
                    metrics.write(json.dumps(record) + "\n")
                    metrics.flush()
                    print(json.dumps(record), flush=True)
    finally:
        if intervention_handle is not None:
            intervention_handle.remove()

    model.save_pretrained(str(output_dir))
    summary = {
        "steps": max_steps,
        "traces": len(traces),
        "trainable_parameters": trainable,
        "seconds": time.time() - start,
        "checkpoint_steps": list(checkpoint_steps),
        "seed": seed,
        "intervention": intervention,
        "basis_path": str(basis_path) if basis_path is not None else None,
        "tap_layer": tap_layer,
        "intervention_gate": intervention_gate,
        "gate_corrector_path": str(gate_corrector_path) if gate_corrector_path is not None else None,
        "gate_masks_path": str(gate_masks_path) if gate_masks_path is not None else None,
        "gate_threshold_z": gate_threshold_z,
        "gate_summary": gate_summary,
        "basis_refresh_interval": basis_refresh_interval,
        "basis_refresh_traces": basis_refresh_traces,
        "basis_refresh_positions": basis_refresh_positions,
        "basis_refresh_directions": basis_refresh_directions,
        "basis_refresh_at_start": basis_refresh_at_start,
        "basis_refreshes": refresh_records,
    }
    (output_dir / "run.summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)
    return summary
