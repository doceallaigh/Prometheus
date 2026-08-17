"""RRS-J-CfC: latent reasoning via a closed-form continuous-time loop at a J-space layer.

Implements the proof of concept from reports/20260709-jspace-cfc-latent-reasoning-poc-design.md,
revised twice after diagnostics (v3): the model performs a *silent scratchpad rollout* — the
frozen base rolls its chain-of-thought internally (THINK tokens are never emitted to the user),
while a CfC recurrent cell rides along the rollout, reading the J-space hidden state of each
internal step and contributing a zero-initialized logit correction. At initialization the
internal rollout is bit-identical to the base model's CoT behavior (inheriting its accuracy as
a floor); training teacher-forces the chain and lets the CfC cell repair the base model's
next-token errors. Only the answer digits are emitted.

Earlier variants failed for diagnosed reasons: v1 injected a single-position residual edit
(oracle ceiling ~5-27%: the answer is computed by upper-layer attention over visible THINK
tokens, not stored at one position); v2 regressed continuous J-space THINK states against
teacher activations (cosine plateaued at ~0.3 — "close" states decode to wrong digits because
the target function is discrete arithmetic).
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from prometheus.config import DataConfig, ModelConfig, PrometheusConfig
from prometheus.data import CharacterTokenizer, ReasoningProblem, generate_reasoning_problems
from prometheus.model import DenseTransformerLM


class CfCCell(nn.Module):
    """Closed-form continuous-time (CfC) recurrent cell (Hasani et al., 2022 style).

    The liquid-network ODE is replaced by its analytic solution, so one step is a
    plain differentiable forward pass: a sigmoidal time gate interpolates between
    two bounded candidate states computed from a shared backbone.
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(input_dim + hidden_dim, hidden_dim), nn.GELU())
        self.time_gate = nn.Linear(hidden_dim, hidden_dim)
        self.candidate_a = nn.Linear(hidden_dim, hidden_dim)
        self.candidate_b = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, context: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """Advance the cell state by one closed-form step.

        Args:
            context: Constant per-problem anchor input, shape ``(batch, input_dim)``.
            state: Previous cell state, shape ``(batch, hidden_dim)``.

        Returns:
            torch.Tensor: Updated cell state, shape ``(batch, hidden_dim)``.
        """

        features = self.backbone(torch.cat([context, state], dim=-1))
        gate = torch.sigmoid(-self.time_gate(features))
        return gate * torch.tanh(self.candidate_a(features)) + (1.0 - gate) * torch.tanh(self.candidate_b(features))


class JSpaceCfCLoop(nn.Module):
    """Silent scratchpad corrector: a CfC cell riding along the internal rollout.

    At each internal step the cell reads the J-space hidden state of the current
    position, advances its liquid state, and produces a vocabulary logit correction.
    The correction head is zero-initialized, so at initialization the internal
    rollout is exactly the frozen base model's chain-of-thought rollout.
    """

    def __init__(self, d_model: int, d_cfc: int, vocab_size: int, max_steps: int, cell_type: str = "cfc"):
        super().__init__()
        if cell_type not in {"cfc", "gru"}:
            raise ValueError("cell_type must be 'cfc' or 'gru'")
        self.max_steps = max_steps
        self.d_cfc = d_cfc
        self.cell_type = cell_type
        self.phi_in = nn.Sequential(nn.Linear(d_model, d_cfc), nn.GELU(), nn.Linear(d_cfc, d_cfc))
        self.cfc = CfCCell(input_dim=d_cfc, hidden_dim=d_cfc) if cell_type == "cfc" else nn.GRUCell(d_cfc, d_cfc)
        self.logit_head = nn.Linear(d_cfc, vocab_size)
        nn.init.zeros_(self.logit_head.weight)
        nn.init.zeros_(self.logit_head.bias)

    def initial_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Return the zero liquid state for a fresh rollout."""

        return torch.zeros(batch_size, self.d_cfc, device=device)

    def step(self, h_j: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Advance the cell one internal step and emit a logit correction.

        Args:
            h_j: J-space hidden state at the current position, ``(batch, d_model)``.
            state: Previous liquid state, ``(batch, d_cfc)``.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Logit correction ``(batch, vocab)``
            and the updated liquid state.
        """

        context = self.phi_in(h_j)
        state = self.cfc(context, state)
        return self.logit_head(state), state

    def forward(self, h_j_sequence: torch.Tensor) -> torch.Tensor:
        """Run the cell across a teacher-forced sequence of J-space states.

        Args:
            h_j_sequence: J-space states, shape ``(batch, sequence, d_model)``.

        Returns:
            torch.Tensor: Per-position logit corrections ``(batch, sequence, vocab)``.
        """

        state = self.initial_state(h_j_sequence.size(0), h_j_sequence.device)
        corrections: list[torch.Tensor] = []
        for position in range(h_j_sequence.size(1)):
            bias, state = self.step(h_j_sequence[:, position], state)
            corrections.append(bias)
        return torch.stack(corrections, dim=1)


class RRSJCfCModel(nn.Module):
    """Frozen dense transformer with a JSpaceCfCLoop riding its internal rollout."""

    def __init__(self, base: DenseTransformerLM, jspace_layer_index: int, loop: JSpaceCfCLoop):
        super().__init__()
        if not 0 <= jspace_layer_index < len(base.blocks):
            raise ValueError("jspace_layer_index must be in [0, num_layers); 0 taps the raw embeddings (ablation)")
        self.base = base
        self.jspace_layer_index = jspace_layer_index
        self.loop = loop
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.base.eval()

    def lower_states(self, tokens: torch.Tensor) -> torch.Tensor:
        """Run embedding and lower blocks without gradients.

        Args:
            tokens: Token ids shaped ``(batch, sequence)``.

        Returns:
            torch.Tensor: Hidden states at the J-space layer.
        """

        with torch.no_grad():
            x = self.base._embed(tokens)
            for block in self.base.blocks[: self.jspace_layer_index]:
                x = block(x)
        return x

    @torch.no_grad()
    def states_and_logits(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the frozen base once, capturing J-space states and final logits.

        Args:
            tokens: Token ids shaped ``(batch, sequence)``.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: J-space states ``(batch, sequence, d_model)``
            and base next-token logits ``(batch, sequence, vocab)``.
        """

        x = self.base._embed(tokens)
        for block in self.base.blocks[: self.jspace_layer_index]:
            x = block(x)
        h_j = x
        for block in self.base.blocks[self.jspace_layer_index :]:
            x = block(x)
        logits = self.base.lm_head(self.base.norm(x))
        return h_j, logits

    def corrected_logits(self, tokens: torch.Tensor) -> torch.Tensor:
        """Teacher-forced forward: frozen base logits plus CfC corrections.

        Args:
            tokens: Token ids shaped ``(batch, sequence)``.

        Returns:
            torch.Tensor: Corrected next-token logits ``(batch, sequence, vocab)``.
        """

        h_j, base_logits = self.states_and_logits(tokens)
        corrections = self.loop(h_j)
        return base_logits + corrections


def load_base_checkpoint(checkpoint_path: str | Path, device: torch.device) -> tuple[DenseTransformerLM, CharacterTokenizer]:
    """Load a frozen dense base model and its tokenizer from a training checkpoint.

    Args:
        checkpoint_path: Path to a ``checkpoint.pt`` written by ``run_training``.
        device: Device onto which the model is loaded.

    Returns:
        tuple[DenseTransformerLM, CharacterTokenizer]: Eval-mode base model and tokenizer.
    """

    payload = torch.load(Path(checkpoint_path), map_location=device, weights_only=False)
    model_config = ModelConfig(**payload["model_config"])
    if model_config.architecture != "dense":
        raise ValueError("rrs_j_cfc requires a dense base checkpoint")
    sequence_length = payload["model_state"]["position_embeddings.weight"].shape[0]
    base = DenseTransformerLM(model_config, sequence_length)
    base.load_state_dict(payload["model_state"])
    base.to(device)
    base.eval()
    stoi = payload["tokenizer"]
    tokenizer = CharacterTokenizer(stoi=stoi, itos={index: ch for ch, index in stoi.items()})
    return base, tokenizer


def _parse_answer(generated: str) -> int | None:
    """Extract the integer answer following the final ``A`` marker, if well formed."""

    marker = generated.rfind("A")
    if marker < 0:
        return None
    digits = ""
    for ch in generated[marker + 1 :]:
        if ch.isdigit():
            digits += ch
        else:
            break
    if not digits:
        return None
    return int(digits)


@torch.no_grad()
def _greedy_generate(
    model: DenseTransformerLM,
    tokenizer: CharacterTokenizer,
    prompt: str,
    max_new_tokens: int,
    device: torch.device,
) -> str:
    """Greedy-decode continuation characters until ``;`` or the token budget is hit."""

    stop_id = tokenizer.stoi[";"]
    tokens = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    generated: list[int] = []
    for _ in range(max_new_tokens):
        window = tokens[:, -model.sequence_length :]
        logits = model(window).logits
        next_id = int(logits[0, -1].argmax().item())
        generated.append(next_id)
        if next_id == stop_id:
            break
        tokens = torch.cat([tokens, torch.tensor([[next_id]], device=device)], dim=1)
    return tokenizer.decode(generated)


@torch.no_grad()
def _greedy_generate_latent(
    model: RRSJCfCModel,
    tokenizer: CharacterTokenizer,
    problem: ReasoningProblem,
    max_internal_tokens: int,
    device: torch.device,
) -> tuple[str, float, int]:
    """Silently roll out the corrected chain and return only the answer segment.

    The rollout starts from ``Q<expr>=T`` and greedy-decodes internal tokens with the
    CfC logit correction applied at every step, stopping at ``;`` or the budget.
    THINK tokens stay internal; only the characters from the final ``A`` onward count
    as emitted output.

    Returns:
        tuple[str, float, int]: Emitted answer text (``A..;``), the number of internal
        rollout steps, and the count of emitted characters.
    """

    stop_id = tokenizer.stoi[";"]
    prompt = f"Q{problem.expression}=T"
    tokens = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    state = model.loop.initial_state(1, device)
    internal: list[int] = []
    for _ in range(max_internal_tokens):
        h_j, base_logits = model.states_and_logits(tokens)
        bias, state = model.loop.step(h_j[:, -1], state)
        logits = base_logits[:, -1] + bias
        next_id = int(logits[0].argmax().item())
        internal.append(next_id)
        if next_id == stop_id:
            break
        tokens = torch.cat([tokens, torch.tensor([[next_id]], device=device)], dim=1)
    rollout = tokenizer.decode(internal)
    marker = rollout.rfind("A")
    emitted = rollout[marker:] if marker >= 0 else ""
    return emitted, float(len(internal)), len(emitted)


def _accumulate_result(
    results: dict[str, dict[str, float]], problem: ReasoningProblem, correct: bool, emitted: int
) -> None:
    """Update overall and per-chain-length accuracy accumulators in place."""

    for key in ("overall", f"chain_{problem.chain_length}"):
        bucket = results.setdefault(key, {"correct": 0.0, "total": 0.0, "emitted": 0.0})
        bucket["correct"] += 1.0 if correct else 0.0
        bucket["total"] += 1.0
        bucket["emitted"] += float(emitted)


def _finalize_results(results: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    """Convert raw accumulators into accuracy and mean-emitted-token statistics."""

    finalized = {}
    for key, bucket in sorted(results.items()):
        finalized[key] = {
            "accuracy": bucket["correct"] / bucket["total"],
            "count": int(bucket["total"]),
            "mean_emitted_tokens": bucket["emitted"] / bucket["total"],
        }
    return finalized


def evaluate_base_reasoning(
    model: DenseTransformerLM,
    tokenizer: CharacterTokenizer,
    problems: list[ReasoningProblem],
    mode: str,
    device: torch.device,
) -> dict[str, dict[str, float]]:
    """Measure exact-match accuracy of the frozen base in direct or cot mode.

    Args:
        model: Frozen dense base model.
        tokenizer: Character tokenizer matching the base checkpoint.
        problems: Held-out problems to evaluate.
        mode: ``direct`` (answer immediately) or ``cot`` (generate THINK tokens first).
        device: Evaluation device.

    Returns:
        dict[str, dict[str, float]]: Accuracy and emitted-token stats, overall and per chain length.
    """

    if mode not in {"direct", "cot"}:
        raise ValueError("mode must be 'direct' or 'cot'")
    marker = "A" if mode == "direct" else "T"
    max_new_tokens = 8 if mode == "direct" else 64
    results: dict[str, dict[str, float]] = {}
    for problem in problems:
        prompt = f"Q{problem.expression}={marker}"
        generated = _greedy_generate(model, tokenizer, prompt, max_new_tokens, device)
        parsed = _parse_answer(marker + generated if mode == "direct" else generated)
        _accumulate_result(results, problem, parsed == problem.answer, len(generated))
    return _finalize_results(results)


def evaluate_latent_reasoning(
    model: RRSJCfCModel,
    tokenizer: CharacterTokenizer,
    problems: list[ReasoningProblem],
    device: torch.device,
) -> dict[str, dict[str, float]]:
    """Measure exact-match accuracy of the latent-loop model on direct-format prompts.

    Args:
        model: Frozen base plus trained latent loop.
        tokenizer: Character tokenizer matching the base checkpoint.
        problems: Held-out problems to evaluate.
        device: Evaluation device.

    Returns:
        dict[str, dict[str, float]]: Accuracy, emitted-token, and loop-step statistics.
    """

    results: dict[str, dict[str, float]] = {}
    steps_by_key: dict[str, list[float]] = {}
    for problem in problems:
        emitted, internal_steps, emitted_count = _greedy_generate_latent(
            model, tokenizer, problem, model.loop.max_steps, device
        )
        parsed = _parse_answer(emitted)
        _accumulate_result(results, problem, parsed == problem.answer, emitted_count)
        for key in ("overall", f"chain_{problem.chain_length}"):
            steps_by_key.setdefault(key, []).append(internal_steps)
    finalized = _finalize_results(results)
    for key, steps in steps_by_key.items():
        finalized[key]["mean_loop_steps"] = sum(steps) / len(steps)
    return finalized


def _build_problem_batch(
    problems: list[ReasoningProblem],
    tokenizer: CharacterTokenizer,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Tokenize a batch of CoT texts with chain and answer supervision masks.

    Args:
        problems: Problems to encode.
        tokenizer: Character tokenizer matching the base checkpoint.
        device: Target device for all returned tensors.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ``(tokens, chain_mask,
        answer_mask)`` where ``chain_mask`` marks positions whose next-token targets
        lie inside the THINK-to-terminator span and ``answer_mask`` marks the subset
        whose targets are answer characters or the terminator.
    """

    pad_id = tokenizer.stoi["\n"]
    cot_texts = [problem.cot_text() for problem in problems]
    max_len = max(len(text) for text in cot_texts)
    tokens = torch.full((len(problems), max_len), pad_id, dtype=torch.long)
    chain_mask = torch.zeros((len(problems), max_len), dtype=torch.bool)
    answer_mask = torch.zeros((len(problems), max_len), dtype=torch.bool)
    for row, cot_text in enumerate(cot_texts):
        tokens[row, : len(cot_text)] = torch.tensor(tokenizer.encode(cot_text))
        think_start = cot_text.index("T")
        a_position = cot_text.rindex("A")
        chain_mask[row, think_start : len(cot_text) - 1] = True
        answer_mask[row, a_position : len(cot_text) - 1] = True
    return tokens.to(device), chain_mask.to(device), answer_mask.to(device)


def run_latent_distillation(config: PrometheusConfig) -> Path:
    """Train the rrs-j-cfc latent loop against a frozen base checkpoint (Phase 2).

    Args:
        config: Full experiment configuration with ``model.architecture == "rrs_j_cfc"``.

    Returns:
        Path: Run directory containing metrics, checkpoint, and summary artifacts.
    """

    from prometheus.train import _make_run_directory, _write_json, resolve_device, set_seed

    model_config = config.model
    if model_config.base_checkpoint is None:
        raise ValueError("rrs_j_cfc requires model.base_checkpoint")
    if model_config.jspace_layer_index is None or model_config.cfc_dim is None or model_config.cfc_max_steps is None:
        raise ValueError("rrs_j_cfc requires jspace_layer_index, cfc_dim, and cfc_max_steps")

    set_seed(config.experiment.seed)
    device = resolve_device(config.experiment.device)
    base, tokenizer = load_base_checkpoint(model_config.base_checkpoint, device)
    loop = JSpaceCfCLoop(
        d_model=base.config.embedding_dim,
        d_cfc=model_config.cfc_dim,
        vocab_size=tokenizer.vocab_size,
        max_steps=model_config.cfc_max_steps,
        cell_type=model_config.cfc_cell_type,
    ).to(device)
    model = RRSJCfCModel(base=base, jspace_layer_index=model_config.jspace_layer_index, loop=loop).to(device)

    train_problems = generate_reasoning_problems(config.data, split="train")
    val_problems = generate_reasoning_problems(config.data, split="val")
    if not train_problems or not val_problems:
        raise ValueError("reasoning_chain problem generation produced an empty split")

    optimizer = torch.optim.AdamW(
        loop.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        betas=(0.9, 0.95),
    )

    run_dir = _make_run_directory(config)
    config_snapshot = config.to_dict()
    config_snapshot["experiment"]["requested_device"] = config.experiment.device
    config_snapshot["experiment"]["device"] = str(device)
    _write_json(run_dir / "config.snapshot.json", config_snapshot)
    _write_json(
        run_dir / "model.summary.json",
        {
            "architecture": "rrs_j_cfc",
            "base_parameter_count": sum(p.numel() for p in base.parameters()),
            "loop_parameter_count": sum(p.numel() for p in loop.parameters()),
            "jspace_layer_index": model_config.jspace_layer_index,
            "cfc_dim": model_config.cfc_dim,
            "cfc_max_steps": model_config.cfc_max_steps,
            "cfc_cell_type": model_config.cfc_cell_type,
            "vocab_size": tokenizer.vocab_size,
        },
    )
    metrics_path = run_dir / "metrics.jsonl"

    generator = torch.Generator().manual_seed(config.experiment.seed)
    best_val_accuracy = 0.0
    training_started_at = time.perf_counter()
    answer_weight = 2.0
    for step in range(config.training.max_steps):
        indices = torch.randint(0, len(train_problems), (config.data.batch_size,), generator=generator)
        batch_problems = [train_problems[index] for index in indices.tolist()]
        tokens, chain_mask, answer_mask = _build_problem_batch(batch_problems, tokenizer, device)

        logits = model.corrected_logits(tokens)
        shifted_logits = logits[:, :-1]
        shifted_targets = tokens[:, 1:]
        shifted_chain = chain_mask[:, :-1]
        shifted_answer = answer_mask[:, :-1]
        chain_loss = F.cross_entropy(shifted_logits[shifted_chain], shifted_targets[shifted_chain])
        answer_loss = F.cross_entropy(shifted_logits[shifted_answer], shifted_targets[shifted_answer])
        loss = chain_loss + answer_weight * answer_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(loop.parameters(), config.training.grad_clip)
        optimizer.step()

        if step % config.training.log_interval == 0 or step == config.training.max_steps - 1:
            record = {
                "step": step,
                "split": "train",
                "loss": loss.item(),
                "chain_loss": chain_loss.item(),
                "answer_loss": answer_loss.item(),
            }
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            print(json.dumps(record))

        if step % config.training.eval_interval == 0 or step == config.training.max_steps - 1:
            eval_problems = val_problems[: config.evaluation.max_batches * config.data.batch_size]
            loop.eval()
            evaluation = evaluate_latent_reasoning(model, tokenizer, eval_problems, device)
            loop.train()
            record = {"step": step, "split": "val", **evaluation["overall"]}
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            print(json.dumps(record))
            if evaluation["overall"]["accuracy"] >= best_val_accuracy:
                best_val_accuracy = evaluation["overall"]["accuracy"]
                torch.save(
                    {
                        "loop_state": loop.state_dict(),
                        "model_config": asdict(model_config),
                        "base_checkpoint": str(model_config.base_checkpoint),
                        "tokenizer": tokenizer.stoi,
                    },
                    run_dir / "checkpoint.pt",
                )

    total_training_seconds = time.perf_counter() - training_started_at
    _write_json(
        run_dir / "run.summary.json",
        {
            "total_training_seconds": total_training_seconds,
            "best_val_accuracy": best_val_accuracy,
        },
    )
    return run_dir


def load_latent_checkpoint(run_dir: str | Path, device: torch.device) -> tuple[RRSJCfCModel, CharacterTokenizer]:
    """Rebuild a trained rrs-j-cfc model from a latent-distillation run directory.

    Args:
        run_dir: Run directory containing a latent ``checkpoint.pt``.
        device: Device onto which the model is loaded.

    Returns:
        tuple[RRSJCfCModel, CharacterTokenizer]: Assembled model and tokenizer.
    """

    payload = torch.load(Path(run_dir) / "checkpoint.pt", map_location=device, weights_only=False)
    model_config = ModelConfig(**payload["model_config"])
    base, tokenizer = load_base_checkpoint(payload["base_checkpoint"], device)
    loop = JSpaceCfCLoop(
        d_model=base.config.embedding_dim,
        d_cfc=model_config.cfc_dim,
        vocab_size=tokenizer.vocab_size,
        max_steps=model_config.cfc_max_steps,
        cell_type=model_config.cfc_cell_type,
    ).to(device)
    loop.load_state_dict(payload["loop_state"])
    loop.eval()
    model = RRSJCfCModel(base=base, jspace_layer_index=model_config.jspace_layer_index, loop=loop).to(device)
    return model, tokenizer


def compare_reasoning_systems(
    base_run_dir: str | Path,
    latent_run_dir: str | Path | None,
    num_problems: int,
    device: torch.device,
    task_family: str | None = None,
) -> dict[str, dict[str, dict[str, float]]]:
    """Evaluate direct, cot, and optional latent systems on shared held-out problems.

    Args:
        base_run_dir: Run directory of the pretrained dense base.
        latent_run_dir: Optional run directory of a trained latent loop.
        num_problems: Number of held-out problems to evaluate.
        device: Evaluation device.
        task_family: Optional override of the base config's task family — set
            this to evaluate a loop zero-shot on the *other* family it was
            not distilled on (the task-transfer probe).

    Returns:
        dict: Mapping of system name to stratified evaluation results.
    """

    base_run_path = Path(base_run_dir)
    snapshot = json.loads((base_run_path / "config.snapshot.json").read_text(encoding="utf-8"))
    data_config = DataConfig(**snapshot["data"])
    if task_family is not None:
        data_config.task_family = task_family
    base, tokenizer = load_base_checkpoint(base_run_path / "checkpoint.pt", device)
    problems = generate_reasoning_problems(data_config, split="val")[:num_problems]
    if not problems:
        raise ValueError("no validation problems available for comparison")
    results = {
        "direct": evaluate_base_reasoning(base, tokenizer, problems, "direct", device),
        "cot": evaluate_base_reasoning(base, tokenizer, problems, "cot", device),
    }
    if latent_run_dir is not None:
        latent_model, latent_tokenizer = load_latent_checkpoint(latent_run_dir, device)
        results["latent_rrs_j_cfc"] = evaluate_latent_reasoning(latent_model, latent_tokenizer, problems, device)
    return results


def comparison_report_markdown(results: dict[str, dict[str, dict[str, float]]]) -> str:
    """Render the three-way comparison as a markdown report.

    Args:
        results: Output of :func:`compare_reasoning_systems`.

    Returns:
        str: Markdown document with overall and per-chain-length tables.
    """

    lines = ["# Reasoning system comparison", ""]
    lines.append("| system | overall accuracy | mean emitted tokens | mean loop steps |")
    lines.append("| --- | --- | --- | --- |")
    for system, stats in results.items():
        overall = stats["overall"]
        loop_steps = f"{overall['mean_loop_steps']:.2f}" if "mean_loop_steps" in overall else "-"
        lines.append(
            f"| {system} | {overall['accuracy']:.4f} | {overall['mean_emitted_tokens']:.1f} | {loop_steps} |"
        )
    lines.append("")
    chain_keys = sorted(
        {key for stats in results.values() for key in stats if key.startswith("chain_")},
        key=lambda key: int(key.split("_")[1]),
    )
    lines.append("## Accuracy by chain length")
    lines.append("")
    header = "| chain length | " + " | ".join(results.keys()) + " |"
    lines.append(header)
    lines.append("| --- |" + " --- |" * len(results))
    for chain_key in chain_keys:
        row = [chain_key.split("_")[1]]
        for stats in results.values():
            row.append(f"{stats[chain_key]['accuracy']:.4f}" if chain_key in stats else "-")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    return "\n".join(lines)
