"""Retrofit the J-space CfC corrector onto a frozen pretrained HuggingFace LM.

Implements the "Planned: pretrained-LM retrofit" experiment from
reports/20260710-rrs-j-cfc-results.md. The recipe mirrors the toy rrs_j_cfc
v3 design, with one change forced by realistic vocabulary sizes: instead of a
zero-init *logit* correction (vocab x d_cfc parameters), the corrector emits a
zero-init *hidden-state delta* added to the final hidden state before the
trunk's own lm_head. The floor guarantee is identical: at initialization the
delta is exactly zero and the rollout is bit-identical to the frozen trunk.

Three phases, each a CLI subcommand:
  retrofit-harvest  Greedy-generate CoT traces on GSM8K train questions.
  retrofit-train    Distill the corrector on harvested traces (trunk frozen).
  retrofit-eval     Compare direct / cot / latent on GSM8K test questions.
"""

from __future__ import annotations

import json
import math
import re
import time
import copy
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from prometheus.latent_reasoning import CfCCell

COT_PROMPT = (
    "Solve the math problem step by step. End your response with the final "
    "numeric answer on its own line in the form '#### <answer>'.\n\nProblem: "
)
DIRECT_PROMPT = (
    "Answer the math problem with ONLY the final numeric answer in the form "
    "'#### <answer>'. Do not show any working.\n\nProblem: "
)
ANSWER_RE = re.compile(r"####\s*\$?(-?[\d,]+(?:\.\d+)?)")
NUMBER_RE = re.compile(r"-?\$?[\d,]+(?:\.\d+)?")
BOXED_RE = re.compile(r"\\boxed\{\s*\$?(-?[\d,]+(?:\.\d+)?)\s*\}")


def _normalize_number(raw: str) -> str:
    value = raw.replace(",", "").replace("$", "").rstrip(".")
    if value.endswith(".0"):
        value = value[:-2]
    return value


def extract_answer(text: str) -> str | None:
    """Extract the final answer: '#### <number>' or, failing that, the last
    numeric \\boxed{...} (QwQ / MATH-style completions)."""

    match = ANSWER_RE.search(text)
    if match is not None:
        return _normalize_number(match.group(1))
    boxed = BOXED_RE.findall(text)
    if boxed:
        return _normalize_number(boxed[-1])
    return None


def extract_answer_lenient(text: str) -> str | None:
    """Strict '####' extraction, falling back to the last number in the text."""

    strict = extract_answer(text)
    if strict is not None:
        return strict
    numbers = NUMBER_RE.findall(text)
    if not numbers:
        return None
    return _normalize_number(numbers[-1])


class LinearRNNCell(nn.Module):
    """Diagonal linear recurrent unit (LRU-style, real-valued; Orvieto et al., 2023).

    s_t = lambda * s_{t-1} + gamma * (W x_t) with lambda = exp(-exp(nu))
    in (0, 1) per channel and gamma = sqrt(1 - lambda^2) (variance-preserving
    normalization). Strictly linear dynamics and linear readout — tests
    whether the corrector needs nonlinear recurrence at all.
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.state_size = hidden_dim
        u = torch.rand(hidden_dim) * 0.39 + 0.6  # lambda init uniform in [0.6, 0.99]
        self.nu_log = nn.Parameter(torch.log(-torch.log(u)))
        self.w_in = nn.Linear(input_dim, hidden_dim, bias=False)

    def forward(self, context: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        lam = torch.exp(-torch.exp(self.nu_log))
        gamma = torch.sqrt(1.0 - lam * lam)
        state = lam * state + gamma * self.w_in(context)
        return state, state


class DiagonalSSMCell(nn.Module):
    """S4D-real style diagonal state-space cell, ZOH-discretized, non-selective.

    Per channel c a bank of N states with poles A_cn = -exp(a_log) (stable,
    S4D-real init -(1..N)) and a shared timestep dt_c = softplus(dt_log):
    s_t = exp(dt*A) * s_{t-1} + (1 - exp(dt*A)) * B * u_t, read out
    y_c = sum_n C_cn s_cn + D_c u_c. Fixed (input-independent) dynamics —
    the non-selective SSM control for the Mamba cells.
    """

    def __init__(self, input_dim: int, hidden_dim: int, state_dim: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        self.state_size = hidden_dim * state_dim
        self.w_in = nn.Linear(input_dim, hidden_dim)
        self.a_log = nn.Parameter(torch.log(torch.arange(1, state_dim + 1).float()).repeat(hidden_dim, 1))
        dt = torch.exp(torch.rand(hidden_dim) * (math.log(0.1) - math.log(0.001)) + math.log(0.001))
        self.dt_log = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))  # inverse softplus
        self.b = nn.Parameter(torch.ones(hidden_dim, state_dim))
        self.c = nn.Parameter(torch.randn(hidden_dim, state_dim) / math.sqrt(state_dim))
        self.d = nn.Parameter(torch.ones(hidden_dim))

    def forward(self, context: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch = context.size(0)
        s = state.view(batch, self.hidden_dim, self.state_dim)
        u = self.w_in(context)  # (batch, H)
        decay = torch.exp(F.softplus(self.dt_log).unsqueeze(-1) * -torch.exp(self.a_log))  # (H, N)
        s = decay * s + (1.0 - decay) * self.b * u.unsqueeze(-1)
        y = (s * self.c).sum(-1) + self.d * u
        return y, s.reshape(batch, -1)


class MambaCell(nn.Module):
    """Minimal selective SSM (Mamba's S6 core) as a stepwise recurrent cell.

    dt, B, C are input-dependent (the selection mechanism), A is diagonal
    per channel with N-state expansion, and the SSM output is gated by
    silu(W_z x) as in the Mamba block. The depthwise conv and fused scan of
    the reference block are omitted: the sidecar consumes one tap state per
    decode step, so a pure-PyTorch recurrence is the natural form (the
    official kernels are Linux/CUDA-only besides).
    """

    def __init__(self, input_dim: int, hidden_dim: int, state_dim: int = 16):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        self.state_size = hidden_dim * state_dim
        self.in_proj = nn.Linear(input_dim, hidden_dim)
        self.gate_proj = nn.Linear(input_dim, hidden_dim)
        self.dt_proj = nn.Linear(hidden_dim, hidden_dim)
        dt = torch.exp(torch.rand(hidden_dim) * (math.log(0.1) - math.log(0.001)) + math.log(0.001))
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))
        self.bc_proj = nn.Linear(hidden_dim, 2 * state_dim, bias=False)
        self.a_log = nn.Parameter(torch.log(torch.arange(1, state_dim + 1).float()).repeat(hidden_dim, 1))
        self.d = nn.Parameter(torch.ones(hidden_dim))

    def forward(self, context: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch = context.size(0)
        s = state.view(batch, self.hidden_dim, self.state_dim)
        u = F.silu(self.in_proj(context))  # (batch, H)
        dt = F.softplus(self.dt_proj(u))  # (batch, H)
        b_sel, c_sel = self.bc_proj(u).split(self.state_dim, dim=-1)  # (batch, N) each
        decay = torch.exp(dt.unsqueeze(-1) * -torch.exp(self.a_log))  # (batch, H, N)
        s = decay * s + dt.unsqueeze(-1) * b_sel.unsqueeze(1) * u.unsqueeze(-1)
        y = (s * c_sel.unsqueeze(1)).sum(-1) + self.d * u
        y = y * F.silu(self.gate_proj(context))
        return y, s.reshape(batch, -1)


class Mamba2Cell(nn.Module):
    """Minimal SSD (Mamba-2) core: multi-head, scalar per-head decay.

    The Mamba-2 restriction of S6: A collapses to one scalar per head
    (shared across the head's channels and states) and the selective B, C
    are shared across a head's channels — the structure that makes the scan
    attention-like. dt per head, silu-gated output, N-state expansion per
    channel. Same stepwise pure-PyTorch form as MambaCell.
    """

    def __init__(self, input_dim: int, hidden_dim: int, state_dim: int = 32, n_heads: int = 8):
        super().__init__()
        if hidden_dim % n_heads:
            raise ValueError("hidden_dim must be divisible by n_heads")
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.state_size = hidden_dim * state_dim
        self.in_proj = nn.Linear(input_dim, hidden_dim)
        self.gate_proj = nn.Linear(input_dim, hidden_dim)
        self.dt_proj = nn.Linear(input_dim, n_heads)
        dt = torch.exp(torch.rand(n_heads) * (math.log(0.1) - math.log(0.001)) + math.log(0.001))
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))
        self.bc_proj = nn.Linear(input_dim, 2 * n_heads * state_dim, bias=False)
        self.a_log = nn.Parameter(torch.zeros(n_heads))
        self.d = nn.Parameter(torch.ones(hidden_dim))

    def forward(self, context: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch = context.size(0)
        s = state.view(batch, self.n_heads, self.head_dim, self.state_dim)
        u = F.silu(self.in_proj(context)).view(batch, self.n_heads, self.head_dim)
        dt = F.softplus(self.dt_proj(context))  # (batch, heads)
        bc = self.bc_proj(context).view(batch, self.n_heads, 2 * self.state_dim)
        b_sel, c_sel = bc.split(self.state_dim, dim=-1)  # (batch, heads, N)
        decay = torch.exp(dt * -torch.exp(self.a_log))  # (batch, heads)
        s = decay[:, :, None, None] * s + dt[:, :, None, None] * u.unsqueeze(-1) * b_sel.unsqueeze(2)
        y = (s * c_sel.unsqueeze(2)).sum(-1).reshape(batch, self.hidden_dim) + self.d * u.reshape(batch, self.hidden_dim)
        y = y * F.silu(self.gate_proj(context))
        return y, s.reshape(batch, -1)


CORRECTOR_CELLS = {
    "cfc": lambda d: CfCCell(input_dim=d, hidden_dim=d),
    "gru": lambda d: nn.GRUCell(d, d),
    "linear": lambda d: LinearRNNCell(input_dim=d, hidden_dim=d),
    "ssm": lambda d: DiagonalSSMCell(input_dim=d, hidden_dim=d),
    "mamba": lambda d: MambaCell(input_dim=d, hidden_dim=d),
    "mamba2": lambda d: Mamba2Cell(input_dim=d, hidden_dim=d),
}


class HiddenDeltaCorrector(nn.Module):
    """Recurrent corrector emitting a zero-init hidden-state delta for a frozen LM.

    The recurrent core is pluggable (`cell`): CfC (default), GRU, diagonal
    linear RNN, non-selective diagonal SSM, or minimal Mamba/Mamba-2
    selective SSMs — all sharing phi_in and the zero-init delta_head so the
    ablation isolates the recurrence family.

    `input_proj`, if given, is a fixed (non-trainable) d_model x d_model
    projection applied to the tap state before phi_in — used to train
    correctors that read only a subspace of the residual stream (e.g. the
    orthogonal complement of the trunk's dominant Jacobian subspace). It is
    saved in the checkpoint and applied transparently at inference.
    """

    def __init__(self, d_model: int, d_cfc: int, cell: str = "cfc", input_proj: torch.Tensor | None = None):
        super().__init__()
        if cell not in CORRECTOR_CELLS:
            raise ValueError(f"cell must be one of {sorted(CORRECTOR_CELLS)}")
        self.d_cfc = d_cfc
        self.cell = cell
        if input_proj is not None:
            self.register_buffer("input_proj", input_proj.to(torch.float32))
        else:
            self.input_proj = None
        self.phi_in = nn.Sequential(nn.Linear(d_model, d_cfc), nn.GELU(), nn.Linear(d_cfc, d_cfc))
        # Attribute kept named `cfc` for checkpoint compatibility across cells.
        self.cfc = CORRECTOR_CELLS[cell](d_cfc)
        self.state_size = getattr(self.cfc, "state_size", d_cfc)
        self.delta_head = nn.Linear(d_cfc, d_model)
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)

    def initial_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.state_size, device=device)

    def step(self, h_j: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.input_proj is not None:
            h_j = h_j @ self.input_proj.T
        context = self.phi_in(h_j)
        result = self.cfc(context, state)
        if isinstance(result, tuple):  # cells whose readout differs from their state
            out, state = result
        else:  # CfCCell / GRUCell return the new state directly
            out = state = result
        return self.delta_head(out), state

    def forward(self, h_j_sequence: torch.Tensor) -> torch.Tensor:
        state = self.initial_state(h_j_sequence.size(0), h_j_sequence.device)
        deltas = []
        for t in range(h_j_sequence.size(1)):
            delta, state = self.step(h_j_sequence[:, t, :], state)
            deltas.append(delta)
        return torch.stack(deltas, dim=1)


class LinearCorrector(nn.Module):
    """Closed-form (ridge/Procrustes) linear corrector baseline.

    delta = W·h_tap + b, fit analytically from harvested traces with a
    geometric loss — no gradient training, no recurrence, no zero-anchor
    guarantee. Diagnostic for how much of the trained corrector is a fixed
    linear re-basis versus learned recurrent computation (roadmap item 9's
    empirical on-ramp).
    """

    def __init__(self, d_model: int, scale: float = 1.0):
        super().__init__()
        self.linear = nn.Linear(d_model, d_model)
        self.scale = scale

    def initial_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, 0, device=device)

    def step(self, h_j: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.scale * self.linear(h_j), state


class QuorumCorrector(nn.Module):
    """k-member corrector quorum: members vote on the delta (SC at the corrector).

    Self-consistency over full rollouts pays k times the trunk's decode
    cost; the corrector is ~1e-4 of the trunk, so an ensemble vote *inside*
    a single rollout is nearly free. Members are either independently
    trained checkpoints (true ensemble) or replicas of one corrector whose
    tap inputs are perturbed with Gaussian noise scaled to the tap state's
    own per-sequence std (member 0 always reads the clean tap, so the
    reference trajectory stays in the quorum). Each member carries its own
    recurrent state, so perturbation histories diverge over the rollout.

    Aggregation over the k deltas at every step:
    - "mean": ensemble average (variance reduction),
    - "median": coordinatewise robust consensus,
    - "sign": strict vote — the mean delta masked to coordinates where at
      least `agree` fraction of members share its sign; disputed
      coordinates emit zero, deferring to the trunk's own computation
      (the zero-anchor made per-coordinate and dynamic).

    With identical members, zero noise, mean/median aggregation this
    reduces exactly to the single corrector (floor guarantee).
    """

    def __init__(self, members: list[nn.Module], noise: float = 0.0, agg: str = "mean", agree: float = 1.0):
        super().__init__()
        if agg not in {"mean", "median", "sign"}:
            raise ValueError("agg must be one of mean/median/sign")
        self.members = nn.ModuleList(members)
        self.noise = noise
        self.agg = agg
        self.agree = agree

    def initial_state(self, batch_size: int, device: torch.device) -> list:
        return [member.initial_state(batch_size, device) for member in self.members]

    def step(self, h_j: torch.Tensor, state: list) -> tuple[torch.Tensor, list]:
        deltas, new_states = [], []
        for index, (member, member_state) in enumerate(zip(self.members, state)):
            member_input = h_j
            if self.noise > 0 and index > 0:
                scale = self.noise * h_j.std(dim=-1, keepdim=True)
                member_input = h_j + scale * torch.randn_like(h_j)
            delta, member_state = member.step(member_input, member_state)
            deltas.append(delta)
            new_states.append(member_state)
        stacked = torch.stack(deltas)
        if self.agg == "median":
            out = stacked.median(dim=0).values
        elif self.agg == "sign":
            consensus = torch.sign(stacked).mean(dim=0).abs()  # 1.0 = unanimous sign
            out = stacked.mean(dim=0) * (consensus >= self.agree)
        else:
            out = stacked.mean(dim=0)
        return out, new_states


def load_corrector(checkpoint: dict, device: torch.device):
    """Build the right corrector variant from a checkpoint dict."""

    cfg = checkpoint["config"]
    if "linear_state" in checkpoint:
        corrector = LinearCorrector(d_model=cfg["d_model"], scale=cfg.get("scale", 1.0)).to(device=device, dtype=torch.float32)
        corrector.load_state_dict(checkpoint["linear_state"])
    else:
        state = checkpoint["corrector_state"]
        corrector = HiddenDeltaCorrector(
            d_model=cfg["d_model"], d_cfc=cfg["d_cfc"], cell=cfg.get("cell", "cfc"),
            input_proj=state.get("input_proj"),
        ).to(device=device, dtype=torch.float32)
        corrector.load_state_dict(state)
    corrector.eval()
    return corrector, cfg["tap_layer"]


class SnapProjector(nn.Module):
    """Residual MLP that snaps continuous feedback vectors back onto the token manifold.

    The sidecar-to-the-sidecar for autoregressive drift: continuous latent
    steps feed E_p[e] (or a norm-matched hidden state) back as the next
    input embedding, and that vector blurs off the training distribution as
    steps compound. g(x) = x + mlp(x) with a zero-init output layer is the
    identity at initialization (the usual floor guarantee: enabling an
    untrained snap changes nothing) and is trained to be a contraction
    toward the embedding manifold: given the trunk's own feedback vector at
    position t of a correct trace, recover the embedding of the token that
    actually continues the trace.
    """

    def __init__(self, d_model: int, d_hidden: int = 512):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(d_model, d_hidden), nn.GELU(), nn.Linear(d_hidden, d_model))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mlp(x)


def load_snap(snap_path: str | Path, device: torch.device) -> tuple[SnapProjector, str]:
    """Load a snap projector checkpoint; returns (module, snap_input)."""

    checkpoint = torch.load(snap_path, map_location=device, weights_only=True)
    cfg = checkpoint["config"]
    snap = SnapProjector(d_model=cfg["d_model"], d_hidden=cfg["d_hidden"]).to(device=device, dtype=torch.float32)
    snap.load_state_dict(checkpoint["snap_state"])
    snap.eval()
    # Freeze: a grad-requiring module inside a generation hook makes every
    # downstream hidden state and KV-cache entry carry an autograd graph,
    # which leaks unboundedly across a rollout (observed as OOM mid-eval).
    snap.requires_grad_(False)
    return snap, cfg.get("snap_input", "expected")


def train_tap_snap(
    model_name: str,
    traces_path: str | Path,
    basis_path: str | Path,
    output_path: str | Path,
    tap_layer: int = 12,
    d_hidden: int = 512,
    max_steps: int = 3000,
    learning_rate: float = 1e-3,
    max_seq_len: int = 640,
    device_str: str = "auto",
    log_interval: int = 100,
) -> dict:
    """Train a snap projector on the *tap manifold*: a learned projection for fork injections.

    The fork's norm-shell constraint fixes the perturbed state's magnitude
    but not its direction; a true convex hull has no vertex set at the tap.
    The learned middle ground is the same denoising recipe as the
    embedding-space snap, retargeted: sample clean tap states h from
    teacher-forced correct traces, perturb them with the same family of
    edits the fork applies — dominant-subspace suppression h − γBBᵀh
    (γ ~ U(0.5, 1.25)), excursion-direction tilts h + ε·ĉ_j·||h|| using
    complement vectors from other positions of the same trace
    (ε ~ U(0.1, 0.5)), and random tilts — each norm-matched to ||h|| as the
    deployment hook does, and train g(x) = x + mlp(x) (zero-init: identity
    floor) to recover h. Loss is normalized MSE. At deployment the snap is
    applied to injected rows only, after injection and norm-matching.
    """

    device = torch.device(device_str if device_str != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model, tokenizer = load_trunk(model_name, device, dtype)
    basis = torch.load(basis_path, map_location=device, weights_only=True)["basis_full"].to(torch.float32)

    traces = []
    with Path(traces_path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                traces.append(json.loads(line))

    snap = SnapProjector(d_model=model.config.hidden_size, d_hidden=d_hidden).to(device=device, dtype=torch.float32)
    optimizer = torch.optim.Adam(snap.parameters(), lr=learning_rate)
    generator = torch.Generator(device="cpu").manual_seed(1337)

    start = time.time()
    step = 0
    while step < max_steps:
        trace = traces[int(torch.randint(len(traces), (1,), generator=generator))]
        prompt_ids = tokenizer(trace["prompt"], return_tensors="pt")["input_ids"]
        completion_ids = tokenizer(trace["completion"], add_special_tokens=False, return_tensors="pt")["input_ids"]
        ids = torch.cat([prompt_ids, completion_ids], dim=1)[:, :max_seq_len].to(device)
        P = min(prompt_ids.size(1), ids.size(1) - 1)
        with torch.no_grad():
            outputs = model(ids, output_hidden_states=True, use_cache=False)
            h = outputs.hidden_states[tap_layer][0, P:, :].float()  # (T, d)
        if h.size(0) < 8:
            continue

        dominant = (h @ basis.T) @ basis
        complement = h - dominant
        norms = h.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        T = h.size(0)
        mode = torch.randint(3, (T,), generator=generator).to(device)
        gamma = (0.5 + 0.75 * torch.rand(T, 1, generator=generator)).to(device)
        eps = (0.1 + 0.4 * torch.rand(T, 1, generator=generator)).to(device)
        perm = torch.randperm(T, generator=generator).to(device)
        tilt_dir = complement[perm] / complement[perm].norm(dim=-1, keepdim=True).clamp_min(1e-6)
        rand_dir = torch.randn(h.shape, generator=generator).to(device)
        rand_dir = rand_dir / rand_dir.norm(dim=-1, keepdim=True).clamp_min(1e-6)

        perturbed = torch.where(
            (mode == 0).unsqueeze(-1), h - gamma * dominant,
            torch.where((mode == 1).unsqueeze(-1), h + eps * tilt_dir * norms, h + eps * rand_dir * norms),
        )
        perturbed = perturbed * (norms / perturbed.norm(dim=-1, keepdim=True).clamp_min(1e-6))

        recovered = snap(perturbed)
        loss = ((recovered - h).pow(2).sum(dim=-1) / norms.squeeze(-1).pow(2)).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % log_interval == 0 or step == max_steps - 1:
            with torch.no_grad():
                base = ((perturbed - h).pow(2).sum(dim=-1) / norms.squeeze(-1).pow(2)).mean()
            print(json.dumps({"step": step, "loss": round(float(loss), 4),
                              "identity_loss": round(float(base), 4),
                              "elapsed": round(time.time() - start)}), flush=True)
        step += 1

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"snap_state": snap.state_dict(),
                "config": {"d_model": model.config.hidden_size, "d_hidden": d_hidden,
                           "kind": "tap", "tap_layer": tap_layer}}, out)
    return {"steps": max_steps, "path": str(out)}


def train_snap(
    model_name: str,
    traces_path: str | Path,
    output_path: str | Path,
    corrector_path: str | Path | None = None,
    snap_input: str = "expected",
    d_hidden: int = 512,
    max_steps: int = 2000,
    learning_rate: float = 1e-3,
    max_seq_len: int = 640,
    device_str: str = "auto",
    log_interval: int = 25,
) -> dict:
    """Train the snap projector on the trunk's own feedback distribution.

    One teacher-forced pass per trace supplies, at every completion position
    t, exactly the vector the continuous rollout would feed back —
    snap_input="expected": E_p[e] from the (corrector-adjusted, if given)
    logits, with temperature augmentation (tau ~ U(1,2) half the time) to
    cover the deeper blur of compounded drift; snap_input="hidden": the
    norm-matched final hidden state (the collapse arm — tests whether a
    learned projection can rescue raw-hidden feedback). The target is the
    embedding of the trace's actual next token, so the loss is a one-step
    denoising distillation matched to the deployment input distribution.
    Multi-step scheduled-sampling unrolls are the follow-up if one-step
    training moves the dose-response curve.
    """

    if snap_input not in {"expected", "hidden"}:
        raise ValueError("snap_input must be 'expected' or 'hidden'")
    device = torch.device(device_str if device_str != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model, tokenizer = load_trunk(model_name, device, dtype)
    d_model = model.config.hidden_size
    embed = model.get_input_embeddings()
    lm_head = model.get_output_embeddings()
    head_dtype = lm_head.weight.dtype
    mean_embed_norm = embed.weight.float().norm(dim=1).mean()

    corrector = None
    tap_layer = 0
    if corrector_path is not None:
        checkpoint = torch.load(corrector_path, map_location=device, weights_only=True)
        corrector, tap_layer = load_corrector(checkpoint, device)

    snap = SnapProjector(d_model=d_model, d_hidden=d_hidden).to(device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(snap.parameters(), lr=learning_rate, weight_decay=0.01)

    traces = [json.loads(line) for line in Path(traces_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not traces:
        raise ValueError(f"No traces found in {traces_path}")

    generator = torch.Generator().manual_seed(1337)
    start = time.time()
    for step in range(max_steps):
        trace = traces[int(torch.randint(len(traces), (1,), generator=generator))]
        prompt_ids = tokenizer(trace["prompt"], add_special_tokens=False)["input_ids"]
        completion_ids = tokenizer(trace["completion"], add_special_tokens=False)["input_ids"]
        input_ids = (prompt_ids + completion_ids)[:max_seq_len]
        if len(input_ids) - len(prompt_ids) < 4:
            continue
        batch = torch.tensor([input_ids], device=device)

        with torch.no_grad():
            outputs = model(batch, output_hidden_states=True, use_cache=False)
            h_final = outputs.hidden_states[-1][:, len(prompt_ids) - 1 : -1, :].float()
            if corrector is not None:
                h_tap = outputs.hidden_states[tap_layer][:, len(prompt_ids) - 1 : -1, :].float()
                h_final = h_final + corrector(h_tap)
            del outputs
            logits = lm_head(h_final.to(head_dtype)).float()
            if snap_input == "hidden":
                x = h_final * (mean_embed_norm / (h_final.norm(dim=-1, keepdim=True) + 1e-6))
            else:
                tau = 1.0
                if float(torch.rand((), generator=generator)) < 0.5:
                    tau = 1.0 + float(torch.rand((), generator=generator))  # blur augmentation, U(1,2)
                x = torch.softmax(logits / tau, dim=-1) @ embed.weight.float()
            targets = embed(batch[:, len(prompt_ids):]).float()

        predicted = snap(x)
        loss = F.mse_loss(predicted, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(snap.parameters(), 1.0)
        optimizer.step()
        if step % log_interval == 0 or step == max_steps - 1:
            print(json.dumps({"step": step, "loss": float(loss), "elapsed_seconds": time.time() - start}), flush=True)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "snap_state": snap.state_dict(),
            "config": {
                "model_name": model_name,
                "d_model": d_model,
                "d_hidden": d_hidden,
                "snap_input": snap_input,
                "corrector": str(corrector_path) if corrector_path else None,
                "max_steps": max_steps,
                "learning_rate": learning_rate,
            },
        },
        output_path,
    )
    summary = {"steps": max_steps, "traces": len(traces), "seconds": time.time() - start, "final_loss": float(loss)}
    print(json.dumps(summary), flush=True)
    return summary


def load_trunk(model_name: str, device: torch.device, dtype: torch.dtype, quantize: bool = False):
    """Load a frozen HF causal LM and its tokenizer.

    quantize=True loads NF4 4-bit via bitsandbytes (double-quantized,
    compute in `dtype`) — the local path for 32B-class trunks on a 24GB
    card. hidden_states come out of the quantized model in compute dtype,
    so the corrector/tap machinery is unchanged.
    """

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if quantize:
        from transformers import BitsAndBytesConfig

        config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=config, device_map={"": device.index or 0}
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
        model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model, tokenizer


def _chat_prompt(tokenizer, instruction: str) -> str:
    """Render an instruction through the tokenizer's chat template if present."""

    if tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": instruction}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return instruction


def harvest(
    model_name: str,
    output_path: str | Path,
    num_problems: int,
    max_new_tokens: int,
    device_str: str,
    batch_size: int = 8,
    dataset_name: str = "gsm8k",
    quantize: bool = False,
    resume: bool = False,
) -> dict:
    """Greedy-generate CoT traces; keep correct-answer traces.

    dataset_name="gsm8k" (default) uses the GSM8K train split with gold
    '#### ' answers; "math" uses the MATH train split (numeric-\\boxed{}
    problems only — the error-rich harvest ground for strong trunks, per the
    data-scaling law). quantize=True enables the 4-bit local path for
    32B-class trunks. resume=True continues from a `<output>.progress`
    checkpoint (appending to the trace file) instead of restarting —
    essential for multi-day 32B harvests on crash-prone drivers.
    """

    from datasets import load_dataset

    device = torch.device(device_str if device_str != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model, tokenizer = load_trunk(model_name, device, dtype, quantize=quantize)
    if dataset_name == "math":
        raw = load_dataset("DigitalLearningGmbH/MATH-lighteval", "default", split="train")
        rows_all = [
            {"question": row["problem"], "answer": f"#### {gold}"}
            for row in raw
            if (gold := extract_answer(row["solution"])) is not None
        ]
        from datasets import Dataset

        dataset = Dataset.from_list(rows_all)
        print(json.dumps({"math_numeric_problems": len(dataset)}), flush=True)
    else:
        dataset = load_dataset("openai/gsm8k", "main", split="train")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path = output_path.with_name(output_path.name + ".progress")
    kept = 0
    total = 0
    processed = 0
    if resume and progress_path.exists():
        state = json.loads(progress_path.read_text(encoding="utf-8"))
        processed, kept, total = state["processed"], state["kept"], state["total"]
        print(json.dumps({"resumed_at": processed, "kept": kept}), flush=True)
    start = time.time()
    mode = "a" if (resume and processed) else "w"
    with output_path.open(mode, encoding="utf-8") as sink:
        for begin in range(processed, min(num_problems, len(dataset)), batch_size):
            rows = dataset.select(range(begin, min(begin + batch_size, num_problems)))
            prompts = [_chat_prompt(tokenizer, COT_PROMPT + row["question"]) for row in rows]
            encoded = tokenizer(prompts, return_tensors="pt", padding=True, padding_side="left").to(device)
            with torch.no_grad():
                generated = model.generate(
                    **encoded,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )
            for row, prompt, output_ids in zip(rows, prompts, generated):
                completion = tokenizer.decode(output_ids[encoded["input_ids"].size(1):], skip_special_tokens=True)
                total += 1
                gold = extract_answer(row["answer"])
                predicted = extract_answer(completion)
                if gold is not None and predicted == gold:
                    kept += 1
                    sink.write(json.dumps({"prompt": prompt, "completion": completion, "answer": gold}) + "\n")
            sink.flush()
            progress_path.write_text(
                json.dumps({"processed": min(begin + batch_size, num_problems), "kept": kept, "total": total}),
                encoding="utf-8",
            )
            print(
                f"harvest {min(begin + batch_size, num_problems)}/{num_problems} "
                f"kept={kept} ({kept / max(total, 1):.2%}) elapsed={time.time() - start:.0f}s",
                flush=True,
            )
    summary = {"model": model_name, "total": total, "kept": kept, "keep_rate": kept / max(total, 1)}
    print(json.dumps(summary), flush=True)
    return summary


def _answer_start_index(completion_ids: list[int], tokenizer) -> int:
    """Index in completion_ids where the answer region begins ('####' or, for
    MATH/QwQ-style traces, the final \\boxed{...})."""

    text = tokenizer.decode(completion_ids, skip_special_tokens=True)
    marker = text.rfind("####")
    if marker < 0:
        marker = text.rfind("\\boxed")
    if marker < 0:
        return max(len(completion_ids) - 8, 0)
    prefix_ids = tokenizer(text[:marker], add_special_tokens=False)["input_ids"]
    return min(len(prefix_ids), len(completion_ids) - 1)


def train_corrector(
    model_name: str,
    traces_path: str | Path,
    output_dir: str | Path,
    tap_layer: int,
    d_cfc: int,
    max_steps: int,
    learning_rate: float,
    answer_weight: float,
    device_str: str,
    max_seq_len: int = 640,
    log_interval: int = 25,
    quantize: bool = False,
    bptt_chunk: int = 0,
    cell: str = "cfc",
    tap_project: str | Path | None = None,
    tap_project_mode: str = "remove",
    tap_project_basis: str = "full",
    seed: int = 0,
    monitor_basis: str | Path | None = None,
) -> dict:
    """Distill the corrector on harvested traces with the trunk frozen.

    seed != 0 makes the run a reproducible ensemble member: it seeds the
    corrector's weight init and offsets the trace-sampling order (seed 0
    keeps the historical behaviour: unseeded init, canonical data order).
    Used to train k independent members for a QuorumCorrector.

    monitor_basis (a basis .pt from retrofit-jspace-verify) enables the
    intrusive-thoughts monitor: at every log step, the complement-energy
    fraction and within-trace excursion rate (positions with complement
    z-score >= 2) are logged for the trunk's tap states and for the
    corrector's deltas. Tap-state statistics are trunk-fixed (a stationary
    baseline over training); any training-time boom must show up in the
    delta statistics. Pure instrumentation — training is unchanged.

    quantize=True loads the trunk in 4-bit NF4 (local path for 32B-class
    trunks; lm_head stays in compute dtype so distillation logits are
    unchanged). bptt_chunk > 0 enables truncated BPTT: the corrector is
    unrolled and backpropagated in chunks of that many tokens, carrying the
    CfC state detached across chunk boundaries and accumulating gradients
    into a single optimizer step per trace. This bounds both the recurrent
    graph depth and the vocab-sized logits buffer, which is what makes
    2-6k-token QwQ traces trainable on a 24GB card (set max_seq_len high
    enough to cover the traces).
    """

    device = torch.device(device_str if device_str != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model, tokenizer = load_trunk(model_name, device, dtype, quantize=quantize)
    d_model = model.config.hidden_size

    input_proj = None
    if tap_project is not None:
        bases = torch.load(tap_project, map_location="cpu", weights_only=True)
        basis = bases["basis_full" if tap_project_basis == "full" else "basis_local"].to(torch.float32)
        keep = basis.T @ basis
        input_proj = keep if tap_project_mode == "keep" else torch.eye(d_model) - keep

    monitor = None
    if monitor_basis is not None:
        monitor = torch.load(monitor_basis, map_location=device, weights_only=True)["basis_full"].to(
            device=device, dtype=torch.float32
        )  # (rank, d)

    def _intrusion_stats(rows: torch.Tensor, prefix: str) -> dict:
        """Complement-energy fraction + within-trace excursion rate for (T, d) rows."""

        comp = rows - (rows @ monitor.T) @ monitor
        energy = comp.norm(dim=-1)
        frac = comp.pow(2).sum(-1) / rows.pow(2).sum(-1).clamp_min(1e-9)
        z = (energy - energy.mean()) / (energy.std() + 1e-6)
        return {
            f"{prefix}_comp_frac": round(float(frac.mean()), 4),
            f"{prefix}_intrusion_rate": round(float((z >= 2.0).float().mean()), 4),
        }

    if seed:
        torch.manual_seed(seed)
    corrector = HiddenDeltaCorrector(d_model=d_model, d_cfc=d_cfc, cell=cell, input_proj=input_proj).to(
        device=device, dtype=torch.float32
    )
    optimizer = torch.optim.AdamW(corrector.parameters(), lr=learning_rate, weight_decay=0.01)

    traces = [json.loads(line) for line in Path(traces_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not traces:
        raise ValueError(f"No traces found in {traces_path}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    generator = torch.Generator().manual_seed(1337 + seed)
    start = time.time()

    # Resume from a mid-training snapshot if a previous run crashed partway.
    snapshot_path = output_dir / "snapshot.pt"
    start_step = 0
    if snapshot_path.exists():
        snapshot = torch.load(snapshot_path, map_location=device, weights_only=True)
        corrector.load_state_dict(snapshot["corrector_state"])
        optimizer.load_state_dict(snapshot["optimizer_state"])
        generator.set_state(snapshot["generator_state"])
        start_step = snapshot["step"] + 1
        print(json.dumps({"resumed_from_step": start_step}), flush=True)

    with metrics_path.open("a" if start_step else "w", encoding="utf-8") as metrics:
        for step in range(start_step, max_steps):
            trace = traces[int(torch.randint(len(traces), (1,), generator=generator))]
            prompt_ids = tokenizer(trace["prompt"], add_special_tokens=False)["input_ids"]
            completion_ids = tokenizer(trace["completion"], add_special_tokens=False)["input_ids"]
            input_ids = (prompt_ids + completion_ids)[:max_seq_len]
            completion_len = len(input_ids) - len(prompt_ids)
            if completion_len < 4:
                continue
            batch = torch.tensor([input_ids], device=device)

            with torch.no_grad():
                # Skip the full-vocab logits (recomputed per-chunk below) and
                # the KV cache — at 32B/4k-token scale each costs GBs.
                outputs = model(batch, output_hidden_states=True, use_cache=False, logits_to_keep=1)
            h_tap = outputs.hidden_states[tap_layer][:, len(prompt_ids) - 1 : -1, :].float()
            h_final = outputs.hidden_states[-1][:, len(prompt_ids) - 1 : -1, :].float()
            del outputs
            targets = batch[:, len(prompt_ids):]
            lm_head = model.get_output_embeddings()
            head_dtype = lm_head.weight.dtype  # NF4 trunks keep the head in the checkpoint dtype (bf16)

            weights = torch.ones(targets.size(1), device=device)
            answer_start = _answer_start_index(input_ids[len(prompt_ids):], tokenizer)
            weights[answer_start:] = answer_weight
            weight_total = weights.sum()

            optimizer.zero_grad(set_to_none=True)
            seq_len = h_tap.size(1)
            monitored_deltas: list[torch.Tensor] = []
            if bptt_chunk > 0 and seq_len > bptt_chunk:
                # Truncated BPTT: backward per chunk, CfC state detached across
                # boundaries, gradients accumulate into one optimizer step.
                state = corrector.initial_state(1, device)
                loss_value = 0.0
                for begin in range(0, seq_len, bptt_chunk):
                    end = min(begin + bptt_chunk, seq_len)
                    state = state.detach()
                    chunk_deltas = []
                    for t in range(begin, end):
                        delta, state = corrector.step(h_tap[:, t, :], state)
                        chunk_deltas.append(delta)
                    deltas = torch.stack(chunk_deltas, dim=1)
                    if monitor is not None:
                        monitored_deltas.append(deltas.detach())
                    logits = lm_head((h_final[:, begin:end, :] + deltas).to(head_dtype)).float()
                    loss_per_token = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)), targets[:, begin:end].reshape(-1), reduction="none"
                    )
                    chunk_loss = (loss_per_token * weights[begin:end]).sum() / weight_total
                    chunk_loss.backward()
                    loss_value += float(chunk_loss)
                loss = torch.tensor(loss_value)
            else:
                deltas = corrector(h_tap)
                if monitor is not None:
                    monitored_deltas.append(deltas.detach())
                logits = lm_head((h_final + deltas).to(head_dtype)).float()
                loss_per_token = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction="none"
                )
                loss = (loss_per_token * weights).sum() / weight_total
                loss.backward()
            torch.nn.utils.clip_grad_norm_(corrector.parameters(), 1.0)
            optimizer.step()

            if step % log_interval == 0 or step == max_steps - 1:
                record = {"step": step, "loss": float(loss), "elapsed_seconds": time.time() - start}
                if monitor is not None:
                    with torch.no_grad():
                        record.update(_intrusion_stats(h_tap[0], "tap"))
                        all_deltas = torch.cat(monitored_deltas, dim=1)[0]
                        record.update(_intrusion_stats(all_deltas, "delta"))
                        record["delta_norm"] = round(float(all_deltas.norm(dim=-1).mean()), 4)
                metrics.write(json.dumps(record) + "\n")
                metrics.flush()
                print(json.dumps(record), flush=True)
            if step % 100 == 0:
                torch.save(
                    {
                        "corrector_state": corrector.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "generator_state": generator.get_state(),
                        "step": step,
                    },
                    snapshot_path,
                )

    snapshot_path.unlink(missing_ok=True)

    checkpoint = {
        "corrector_state": corrector.state_dict(),
        "config": {
            "model_name": model_name,
            "tap_layer": tap_layer,
            "d_cfc": d_cfc,
            "d_model": d_model,
            "cell": cell,
            "max_steps": max_steps,
            "learning_rate": learning_rate,
            "answer_weight": answer_weight,
            "tap_project": str(tap_project) if tap_project is not None else None,
            "tap_project_mode": tap_project_mode if tap_project is not None else None,
            "tap_project_basis": tap_project_basis if tap_project is not None else None,
            "seed": seed,
        },
    }
    torch.save(checkpoint, output_dir / "corrector.pt")
    summary = {"steps": max_steps, "traces": len(traces), "seconds": time.time() - start}
    (output_dir / "run.summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)
    return summary


def _rank_values(values: list[float]) -> list[float]:
    """Average-tie ranks (1-based) for Spearman correlation."""

    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        avg_rank = 0.5 * (start + 1 + end)
        for idx in order[start:end]:
            ranks[idx] = avg_rank
        start = end
    return ranks


def _pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2 or len(xs) != len(ys):
        return float("nan")
    x = torch.tensor(xs, dtype=torch.float64)
    y = torch.tensor(ys, dtype=torch.float64)
    x = x - x.mean()
    y = y - y.mean()
    denom = (x.pow(2).sum().sqrt() * y.pow(2).sum().sqrt()).item()
    if denom <= 0:
        return float("nan")
    return float((x * y).sum().item() / denom)


def _spearman(xs: list[float], ys: list[float]) -> float:
    return _pearson(_rank_values(xs), _rank_values(ys))


@torch.no_grad()
def _trunk_accuracy(
    model,
    tokenizer,
    dataset_name: str,
    num_problems: int,
    max_new_tokens: int,
    device: torch.device,
    batch_size: int,
) -> tuple[float, float]:
    """Evaluate strict/lenient trunk accuracy under greedy CoT decoding."""

    from datasets import load_dataset

    if dataset_name == "math":
        raw = load_dataset("DigitalLearningGmbH/MATH-lighteval", "default", split="test")
        rows = []
        for row in raw:
            gold = extract_answer(row["solution"])
            if gold is not None:
                rows.append({"question": row["problem"], "gold": gold})
            if len(rows) >= num_problems:
                break
    else:
        raw = load_dataset("openai/gsm8k", "main", split="test").select(range(num_problems))
        rows = [{"question": row["question"], "gold": extract_answer(row["answer"])} for row in raw]

    strict = 0
    lenient = 0
    valid = 0
    for begin in range(0, len(rows), batch_size):
        chunk = rows[begin: begin + batch_size]
        prompts = [_chat_prompt(tokenizer, COT_PROMPT + row["question"]) for row in chunk]
        encoded = tokenizer(prompts, return_tensors="pt", padding=True, padding_side="left").to(device)
        generated = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
        prompt_len = encoded["input_ids"].size(1)
        for row, output_ids in zip(chunk, generated):
            gold = row["gold"]
            if gold is None:
                continue
            completion = tokenizer.decode(output_ids[prompt_len:], skip_special_tokens=True)
            strict += int(extract_answer(completion) == gold)
            lenient += int(extract_answer_lenient(completion) == gold)
            valid += 1
    if valid == 0:
        return 0.0, 0.0
    return strict / valid, lenient / valid


def _ontogeny_plot_html(phase_rows: list[dict], corr_rows: list[dict], dataset_name: str) -> str:
    x_steps = [int(row["phase_step"]) for row in phase_rows]
    labels = [str(row["phase_label"]) for row in phase_rows]
    acc = [float(row["trunk_strict_accuracy"]) for row in phase_rows]
    comp = [float(row["grad_comp_frac_mean"]) for row in phase_rows]
    dom = [float(row["grad_dom_frac_mean"]) for row in phase_rows]
    gnorm = [float(row["grad_delta_norm_mean"]) for row in phase_rows]
    corr_table_rows = "\n".join(
        "<tr><td>{metric}</td><td>{pearson:.4f}</td><td>{spearman:.4f}</td></tr>".format(
            metric=row["metric"],
            pearson=row["pearson_r"],
            spearman=row["spearman_rho"],
        )
        for row in corr_rows
    )
    return f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Ontogeny Sweep Report</title>
  <script src=\"https://cdn.plot.ly/plotly-2.35.2.min.js\"></script>
  <style>
    body {{
      font-family: Segoe UI, Helvetica, Arial, sans-serif;
      margin: 0;
      padding: 16px;
      background: #0f172a;
      color: #e2e8f0;
    }}
    .card {{
      background: #111827;
      border: 1px solid #334155;
      border-radius: 12px;
      padding: 16px;
      margin-bottom: 16px;
    }}
    table {{
      border-collapse: collapse;
      width: 100%;
    }}
    th, td {{
      border: 1px solid #334155;
      padding: 8px;
      text-align: left;
    }}
    th {{
      background: #1f2937;
    }}
  </style>
</head>
<body>
  <div class=\"card\">
    <h2>Ontogenetic Sweep ({dataset_name})</h2>
    <p>Gradient geometry of the sidecar vs trunk training phase.</p>
  </div>
  <div class=\"card\"><div id=\"phase_chart\" style=\"height: 460px;\"></div></div>
  <div class=\"card\"><div id=\"acc_corr_chart\" style=\"height: 420px;\"></div></div>
  <div class=\"card\">
    <h3>Correlation summary</h3>
    <table>
      <thead><tr><th>Metric</th><th>Pearson r</th><th>Spearman rho</th></tr></thead>
      <tbody>
        {corr_table_rows}
      </tbody>
    </table>
  </div>
  <script>
    const steps = {json.dumps(x_steps)};
    const labels = {json.dumps(labels)};
    const acc = {json.dumps(acc)};
    const comp = {json.dumps(comp)};
    const dom = {json.dumps(dom)};
    const gnorm = {json.dumps(gnorm)};

    Plotly.newPlot('phase_chart', [
      {{x: steps, y: acc, name: 'trunk strict accuracy', mode: 'lines+markers', yaxis: 'y'}},
      {{x: steps, y: comp, name: 'grad complement fraction', mode: 'lines+markers', yaxis: 'y2'}},
      {{x: steps, y: dom, name: 'grad dominant fraction', mode: 'lines+markers', yaxis: 'y2'}},
      {{x: steps, y: gnorm, name: 'grad delta norm', mode: 'lines+markers', yaxis: 'y3'}},
    ], {{
      template: 'plotly_dark',
      title: 'Phase vs accuracy and gradient geometry',
      xaxis: {{title: 'trunk training step', tickmode: 'array', tickvals: steps, ticktext: labels}},
      yaxis: {{title: 'strict accuracy'}},
      yaxis2: {{title: 'fraction', overlaying: 'y', side: 'right', range: [0, 1]}},
      yaxis3: {{title: 'grad norm', anchor: 'free', overlaying: 'y', side: 'right', position: 0.95}},
      legend: {{orientation: 'h'}},
    }}, {{responsive: true, displaylogo: false}});

    Plotly.newPlot('acc_corr_chart', [
      {{x: acc, y: comp, mode: 'markers+text', text: labels, textposition: 'top center', name: 'comp frac vs acc'}},
      {{x: acc, y: gnorm, mode: 'markers+text', text: labels, textposition: 'bottom center', name: 'grad norm vs acc'}},
    ], {{
      template: 'plotly_dark',
      title: 'Gradient metrics vs trunk accuracy',
      xaxis: {{title: 'trunk strict accuracy'}},
      yaxis: {{title: 'metric value'}},
    }}, {{responsive: true, displaylogo: false}});
  </script>
</body>
</html>
"""


def ontogeny_sweep(
    phase_models: list[tuple[int, str]],
    traces_path: str | Path,
    basis_path: str | Path,
    output_dir: str | Path,
    tap_layer: int,
    d_cfc: int = 512,
    cell: str = "cfc",
    gradient_steps: int = 128,
    learning_rate: float = 1e-3,
    answer_weight: float = 2.0,
    max_seq_len: int = 640,
    eval_problems: int = 200,
    eval_max_new_tokens: int = 512,
    eval_batch_size: int = 8,
    dataset_name: str = "gsm8k",
    device_str: str = "auto",
    quantize: bool = False,
    seed: int = 0,
) -> dict:
    """Sweep trunk phases and measure sidecar-gradient geometry vs trunk accuracy.

    For each (phase_step, model) pair:
    1) evaluate trunk strict/lenient accuracy,
    2) run a fixed number of sidecar gradient updates on shared traces,
    3) decompose delta gradients into dominant vs orthogonal-complement energy,
    4) correlate phase metrics against trunk accuracy.
    """

    if not phase_models:
        raise ValueError("phase_models cannot be empty")

    device = torch.device(device_str if device_str != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    basis = torch.load(basis_path, map_location=device, weights_only=True)["basis_full"].to(device=device, dtype=torch.float32)
    traces = [json.loads(line) for line in Path(traces_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not traces:
        raise ValueError(f"No traces found in {traces_path}")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    gradient_metrics_path = output / "gradient_metrics.jsonl"
    phase_metrics_path = output / "phase_metrics.jsonl"
    corr_path = output / "correlations.json"
    report_path = output / "report.md"
    plot_path = output / "ontogeny_plot.html"

    phase_rows: list[dict] = []
    gradient_rows: list[dict] = []

    phase_models = sorted(phase_models, key=lambda item: item[0])
    start = time.time()

    for phase_step, model_name in phase_models:
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        if "::" in model_name:
            if quantize:
                raise ValueError("LoRA phase adapters are not supported with --quantize")
            base_model_name, adapter_path = model_name.split("::", 1)
            model, tokenizer = load_trunk(base_model_name, device, dtype)
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter_path).merge_and_unload()
            model.eval()
            model.requires_grad_(False)
        else:
            model, tokenizer = load_trunk(model_name, device, dtype, quantize=quantize)
        strict_acc, lenient_acc = _trunk_accuracy(
            model=model,
            tokenizer=tokenizer,
            dataset_name=dataset_name,
            num_problems=eval_problems,
            max_new_tokens=eval_max_new_tokens,
            device=device,
            batch_size=eval_batch_size,
        )

        d_model = model.config.hidden_size
        torch.manual_seed(seed)
        corrector = HiddenDeltaCorrector(d_model=d_model, d_cfc=d_cfc, cell=cell).to(device=device, dtype=torch.float32)
        optimizer = torch.optim.AdamW(corrector.parameters(), lr=learning_rate, weight_decay=0.01)
        generator = torch.Generator().manual_seed(1337 + seed)
        lm_head = model.get_output_embeddings()
        head_dtype = lm_head.weight.dtype

        local_rows: list[dict] = []
        for gstep in range(gradient_steps):
            trace = traces[int(torch.randint(len(traces), (1,), generator=generator))]
            prompt_ids = tokenizer(trace["prompt"], add_special_tokens=False)["input_ids"]
            completion_ids = tokenizer(trace["completion"], add_special_tokens=False)["input_ids"]
            input_ids = (prompt_ids + completion_ids)[:max_seq_len]
            completion_len = len(input_ids) - len(prompt_ids)
            if completion_len < 4:
                continue
            batch = torch.tensor([input_ids], device=device)

            with torch.no_grad():
                outputs = model(batch, output_hidden_states=True, use_cache=False, logits_to_keep=1)
            h_tap = outputs.hidden_states[tap_layer][:, len(prompt_ids) - 1: -1, :].float()
            h_final = outputs.hidden_states[-1][:, len(prompt_ids) - 1: -1, :].float()
            del outputs

            targets = batch[:, len(prompt_ids):]
            weights = torch.ones(targets.size(1), device=device)
            answer_start = _answer_start_index(input_ids[len(prompt_ids):], tokenizer)
            weights[answer_start:] = answer_weight
            weight_total = weights.sum()

            optimizer.zero_grad(set_to_none=True)
            deltas = corrector(h_tap)
            deltas.retain_grad()
            logits = lm_head((h_final + deltas).to(head_dtype)).float()
            loss_per_token = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction="none")
            loss = (loss_per_token * weights).sum() / weight_total
            loss.backward()

            with torch.no_grad():
                grad_delta = deltas.grad[0].float()
                grad_dom = (grad_delta @ basis.T) @ basis
                grad_comp = grad_delta - grad_dom
                grad_energy = grad_delta.pow(2).sum(-1).clamp_min(1e-9)
                comp_frac = (grad_comp.pow(2).sum(-1) / grad_energy).mean().item()
                dom_frac = (grad_dom.pow(2).sum(-1) / grad_energy).mean().item()
                grad_delta_norm = grad_delta.norm(dim=-1).mean().item()
                delta_norm = deltas[0].norm(dim=-1).mean().item()
                param_grad_sq = torch.tensor(0.0, device=device)
                for param in corrector.parameters():
                    if param.grad is not None:
                        param_grad_sq += param.grad.detach().float().pow(2).sum()
                param_grad_norm = float(param_grad_sq.sqrt().item())

            torch.nn.utils.clip_grad_norm_(corrector.parameters(), 1.0)
            optimizer.step()

            row = {
                "phase_step": phase_step,
                "phase_label": f"{phase_step}",
                "model": model_name,
                "grad_step": gstep,
                "loss": float(loss.item()),
                "grad_comp_frac": comp_frac,
                "grad_dom_frac": dom_frac,
                "grad_delta_norm": grad_delta_norm,
                "delta_norm": delta_norm,
                "param_grad_norm": param_grad_norm,
            }
            local_rows.append(row)
            gradient_rows.append(row)

        if not local_rows:
            raise RuntimeError("No valid gradient rows were produced; check max_seq_len and traces")

        phase_row = {
            "phase_step": phase_step,
            "phase_label": f"{phase_step}",
            "model": model_name,
            "trunk_strict_accuracy": strict_acc,
            "trunk_lenient_accuracy": lenient_acc,
            "grad_steps": len(local_rows),
            "loss_mean": float(sum(row["loss"] for row in local_rows) / len(local_rows)),
            "grad_comp_frac_mean": float(sum(row["grad_comp_frac"] for row in local_rows) / len(local_rows)),
            "grad_dom_frac_mean": float(sum(row["grad_dom_frac"] for row in local_rows) / len(local_rows)),
            "grad_delta_norm_mean": float(sum(row["grad_delta_norm"] for row in local_rows) / len(local_rows)),
            "delta_norm_mean": float(sum(row["delta_norm"] for row in local_rows) / len(local_rows)),
            "param_grad_norm_mean": float(sum(row["param_grad_norm"] for row in local_rows) / len(local_rows)),
            "elapsed_seconds": time.time() - start,
        }
        phase_rows.append(phase_row)
        print(json.dumps(phase_row), flush=True)

        del corrector
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    metric_keys = [
        "grad_comp_frac_mean",
        "grad_dom_frac_mean",
        "grad_delta_norm_mean",
        "delta_norm_mean",
        "param_grad_norm_mean",
        "loss_mean",
    ]
    acc_values = [float(row["trunk_strict_accuracy"]) for row in phase_rows]
    corr_rows = []
    for key in metric_keys:
        ys = [float(row[key]) for row in phase_rows]
        corr_rows.append({
            "metric": key,
            "pearson_r": _pearson(acc_values, ys),
            "spearman_rho": _spearman(acc_values, ys),
        })

    gradient_metrics_path.write_text(
        "\n".join(json.dumps(row) for row in gradient_rows) + "\n",
        encoding="utf-8",
    )
    phase_metrics_path.write_text(
        "\n".join(json.dumps(row) for row in phase_rows) + "\n",
        encoding="utf-8",
    )
    corr_path.write_text(json.dumps(corr_rows, indent=2), encoding="utf-8")

    plot_path.write_text(_ontogeny_plot_html(phase_rows, corr_rows, dataset_name=dataset_name), encoding="utf-8")

    lines = [
        "# Ontogeny Sweep",
        "",
        f"Dataset: `{dataset_name}`",
        f"Gradient steps per phase: {gradient_steps}",
        f"Trace source: `{traces_path}`",
        "",
        "## Phase metrics",
        "",
        "| phase_step | strict_acc | lenient_acc | grad_comp_frac | grad_dom_frac | grad_delta_norm | param_grad_norm |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in phase_rows:
        lines.append(
            "| {phase_step} | {trunk_strict_accuracy:.4f} | {trunk_lenient_accuracy:.4f} | "
            "{grad_comp_frac_mean:.4f} | {grad_dom_frac_mean:.4f} | {grad_delta_norm_mean:.4f} | {param_grad_norm_mean:.4f} |".format(
                **row
            )
        )
    lines.extend([
        "",
        "## Correlations vs strict accuracy",
        "",
        "| metric | pearson_r | spearman_rho |",
        "| --- | --- | --- |",
    ])
    for row in corr_rows:
        lines.append(
            f"| {row['metric']} | {row['pearson_r']:.4f} | {row['spearman_rho']:.4f} |"
        )
    lines.extend([
        "",
        f"Interactive plot: `{plot_path}`",
        f"Per-step gradient log: `{gradient_metrics_path}`",
        f"Per-phase summary: `{phase_metrics_path}`",
    ])
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    summary = {
        "phases": len(phase_rows),
        "gradient_rows": len(gradient_rows),
        "phase_metrics": str(phase_metrics_path),
        "gradient_metrics": str(gradient_metrics_path),
        "correlations": str(corr_path),
        "plot": str(plot_path),
        "report": str(report_path),
    }
    print(json.dumps(summary), flush=True)
    return summary


@torch.no_grad()
def fit_linear_corrector(
    model_name: str,
    traces_path: str | Path,
    output_path: str | Path,
    tap_layer: int,
    device_str: str = "cuda",
    mode: str = "ridge",
    ridge_lambda: float = 1.0,
    max_traces: int = 0,
    target: str = "geometric",
    scale: float = 1.0,
) -> dict:
    """Fit a linear corrector in closed form — no gradient training.

    target="geometric": push the final hidden state toward the correct next
    token's unembedding direction at the trunk's typical hidden norm,
    D_t = tau * E[y_{t+1}]/||E[y_{t+1}]|| - h_final_t — exactly the loss
    family v2 failed with; kept as the aggressive-target diagnostic.
    target="grad": the CE-gradient direction
    D_t = e_{y_{t+1}} - sum_v p_v e_v (minus the gradient of cross-entropy
    w.r.t. h_final) — identically zero where the trunk already concentrates
    its mass on the gold token, so the fitted map has a data-level
    zero-anchor: it is trained to emit ~0 on the correct manifold and
    corrections only at error positions. `scale` multiplies the fitted map
    at inference (stored in the checkpoint config) for strength sweeps.
    Solve min ||H_tap W - D||^2 (+ridge) analytically; mode="procrustes"
    constrains W to a scaled orthogonal map (SVD of the cross-covariance).
    """

    device = torch.device(device_str if device_str != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model, tokenizer = load_trunk(model_name, device, dtype)
    embed = model.get_output_embeddings().weight.detach().float()  # [V, d]
    d_model = embed.size(1)

    xtx = torch.zeros(d_model, d_model, device=device)
    xtd = torch.zeros(d_model, d_model, device=device)
    x_sum = torch.zeros(d_model, device=device)
    d_sum = torch.zeros(d_model, device=device)
    count = 0
    tau_sum, tau_count = 0.0, 0

    rows = [json.loads(line) for line in Path(traces_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if max_traces:
        rows = rows[:max_traces]
    start = time.time()
    with torch.no_grad():
        for i, row in enumerate(rows):
            prompt_ids = tokenizer(row["prompt"], add_special_tokens=False)["input_ids"]
            completion_ids = tokenizer(row["completion"], add_special_tokens=False)["input_ids"]
            batch = torch.tensor([prompt_ids + completion_ids], device=device)
            outputs = model(batch, output_hidden_states=True)
            span = slice(len(prompt_ids) - 1, len(prompt_ids) + len(completion_ids) - 1)
            h_tap = outputs.hidden_states[tap_layer][0, span, :].float()  # predicts completion tokens
            h_final = outputs.hidden_states[-1][0, span, :].float()
            gold_embed = embed[torch.tensor(completion_ids, device=device)]  # [T, d]
            tau = h_final.norm(dim=-1).mean().item()
            tau_sum += tau
            tau_count += 1
            if target == "grad":
                logits = h_final @ embed.T  # [T, V]
                probabilities = torch.softmax(logits, dim=-1)
                delta_star = gold_embed - probabilities @ embed
            else:
                targets = gold_embed / (gold_embed.norm(dim=-1, keepdim=True) + 1e-6)
                delta_star = targets * tau - h_final
            xtx += h_tap.T @ h_tap
            xtd += h_tap.T @ delta_star
            x_sum += h_tap.sum(0)
            d_sum += delta_star.sum(0)
            count += h_tap.size(0)
            if (i + 1) % 100 == 0:
                print(json.dumps({"traces": i + 1, "positions": count, "elapsed": round(time.time() - start)}), flush=True)

    x_mean = x_sum / count
    d_mean = d_sum / count
    # Center: solve on centered data, fold means into the bias.
    xtx_c = xtx - count * torch.outer(x_mean, x_mean)
    xtd_c = xtd - count * torch.outer(x_mean, d_mean)
    if mode == "procrustes":
        u, s, vt = torch.linalg.svd(xtd_c, full_matrices=False)
        w = u @ vt  # orthogonal map
        scale = s.sum() / torch.diagonal(xtx_c).sum()
        w = (w * scale).T
    else:
        w = torch.linalg.solve(xtx_c + ridge_lambda * count * torch.eye(d_model, device=device), xtd_c)
    bias = d_mean - x_mean @ w

    corrector = LinearCorrector(d_model=d_model, scale=scale)
    with torch.no_grad():
        corrector.linear.weight.copy_(w.T.cpu())
        corrector.linear.bias.copy_(bias.cpu())

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "linear_state": corrector.state_dict(),
            "config": {"d_model": d_model, "tap_layer": tap_layer, "mode": mode, "ridge_lambda": ridge_lambda, "target": target, "scale": scale},
        },
        output_path,
    )
    summary = {
        "mode": mode,
        "target": target,
        "scale": scale,
        "traces": len(rows),
        "positions": count,
        "mean_hidden_norm": tau_sum / max(tau_count, 1),
        "seconds": time.time() - start,
        "output": str(output_path),
    }
    print(json.dumps(summary), flush=True)
    return summary


def _generate_with_corrector(
    model,
    tokenizer,
    corrector,
    tap_layer: int,
    prompt: str,
    max_new_tokens: int,
    device: torch.device,
    temperature: float = 0.0,
    prefix_text: str = "",
) -> str:
    """Rollout applying the corrector's hidden delta at every step.

    temperature=0 gives greedy decoding; temperature>0 samples, enabling
    latent self-consistency (majority vote over k internal rollouts).

    prefix_text continues generation from an existing partial rollout
    (rollback resume): the prefix is teacher-forced through the corrector to
    warm its state, then decoding proceeds normally. Returns only the newly
    generated text (caller re-attaches the prefix).
    """

    encoded = tokenizer(prompt + prefix_text, return_tensors="pt").to(device)
    input_ids = encoded["input_ids"]
    state = corrector.initial_state(1, device) if corrector is not None else None
    warm_start = None
    if corrector is not None and prefix_text:
        prompt_len = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        warm_start = prompt_len - 1  # step the corrector over prefix positions before decoding
    lm_head = model.get_output_embeddings()
    past = None
    tokens = input_ids
    generated: list[int] = []
    eos_id = tokenizer.eos_token_id
    first = True
    for _ in range(max_new_tokens):
        outputs = model(tokens, past_key_values=past, output_hidden_states=True, use_cache=True)
        past = outputs.past_key_values
        if first and warm_start is not None:
            taps = outputs.hidden_states[tap_layer][0, warm_start:-1, :].float()
            for t in range(taps.size(0)):
                _, state = corrector.step(taps[t : t + 1], state)
        first = False
        h_final = outputs.hidden_states[-1][:, -1, :].float()
        if corrector is not None:
            h_tap = outputs.hidden_states[tap_layer][:, -1, :].float()
            delta, state = corrector.step(h_tap, state)
            h_final = h_final + delta
        logits = lm_head(h_final.to(outputs.hidden_states[-1].dtype)).float()
        if temperature > 0:
            probabilities = torch.softmax(logits / temperature, dim=-1)
            next_id = int(torch.multinomial(probabilities, num_samples=1))
        else:
            next_id = int(logits.argmax(dim=-1))
        generated.append(next_id)
        if eos_id is not None and next_id == eos_id:
            break
        tokens = torch.tensor([[next_id]], device=device)
    return tokenizer.decode(generated, skip_special_tokens=True)


def _multiturn_instruction(question: str) -> str:
    return (
        "Solve the next math problem independently. Work step by step and end "
        "with '#### <answer>'.\n\nProblem: " + question
    )


def _chat_turn_suffix(tokenizer, instruction: str) -> str:
    """Render the exact template suffix from one assistant turn to the next."""

    if tokenizer.chat_template:
        sentinel = "PROMETHEUS_ASSISTANT_CONTENT"
        rendered = tokenizer.apply_chat_template(
            [
                {"role": "assistant", "content": sentinel},
                {"role": "user", "content": instruction},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        _, found, suffix = rendered.partition(sentinel)
        if not found:
            raise ValueError("Chat template did not preserve assistant content")
        return suffix
    return "\n\nUser: " + instruction + "\nAssistant:"


def _multiturn_suffix(tokenizer, question: str) -> str:
    return _chat_turn_suffix(tokenizer, _multiturn_instruction(question))


def _chat_messages_prompt(tokenizer, messages: list[dict[str, str]]) -> str:
    if tokenizer.chat_template:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    parts = [f"{message['role'].title()}: {message['content']}" for message in messages]
    return "\n\n".join(parts) + "\nAssistant:"


def _surface_answer(text: str) -> str:
    """Canonical answer-only response exposed by a latent turn."""

    answer = extract_answer(text)
    return f"#### {answer}" if answer is not None else "#### INVALID"


@torch.no_grad()
def _generate_stateful_turn(
    model,
    tokenizer,
    corrector,
    tap_layer: int,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    device: torch.device,
    past=None,
    state=None,
    temperature: float = 0.0,
) -> tuple[str, list[int], object, object]:
    """Decode one turn while optionally retaining trunk and corrector state.

    The returned KV cache contains every input token and every generated token
    except the final one. Callers that persist the cache prepend that final
    token to the next turn's suffix, making the continued sequence exact.
    """

    if corrector is not None and state is None:
        state = corrector.initial_state(1, device)
    lm_head = model.get_output_embeddings()
    tokens = input_ids.to(device)
    generated: list[int] = []
    eos_id = tokenizer.eos_token_id
    for _ in range(max_new_tokens):
        outputs = model(tokens, past_key_values=past, output_hidden_states=True, use_cache=True)
        past = outputs.past_key_values
        h_final = outputs.hidden_states[-1][:, -1, :].float()
        if corrector is not None:
            h_tap = outputs.hidden_states[tap_layer][:, -1, :].float()
            delta, state = corrector.step(h_tap, state)
            h_final = h_final + delta
        logits = lm_head(h_final.to(outputs.hidden_states[-1].dtype)).float()
        if temperature > 0:
            next_id = int(torch.multinomial(torch.softmax(logits / temperature, dim=-1), num_samples=1))
        else:
            next_id = int(logits.argmax(dim=-1))
        generated.append(next_id)
        if eos_id is not None and next_id == eos_id:
            break
        tokens = torch.tensor([[next_id]], device=device)
    return tokenizer.decode(generated, skip_special_tokens=True), generated, past, state


def _copy_corrector_state(state):
    if state is None:
        return None
    if isinstance(state, list):
        return [_copy_corrector_state(member) for member in state]
    return state.clone()


def _sample_stateful_vote(
    model,
    tokenizer,
    corrector,
    tap_layer: int,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    device: torch.device,
    past,
    state,
    samples: int,
    temperature: float,
    stop_agreement: int = 0,
) -> tuple[str, str | None, list[int], object, object, list[str], int]:
    """Sample from one conversation snapshot and continue from a winning branch."""

    from collections import Counter

    candidates = []
    votes: Counter = Counter()
    for _ in range(samples):
        text, generated_ids, candidate_past, candidate_state = _generate_stateful_turn(
            model,
            tokenizer,
            corrector,
            tap_layer,
            input_ids,
            max_new_tokens,
            device,
            past=copy.deepcopy(past),
            state=_copy_corrector_state(state),
            temperature=temperature,
        )
        answer = extract_answer_lenient(text)
        candidates.append((text, answer, generated_ids, candidate_past, candidate_state))
        if answer is not None:
            votes[answer] += 1
            if stop_agreement > 0 and votes[answer] >= stop_agreement:
                break
    majority = votes.most_common(1)[0][0] if votes else None
    chosen = next((candidate for candidate in candidates if candidate[1] == majority), candidates[0])
    text, _, generated_ids, chosen_past, chosen_state = chosen
    return (
        text,
        majority,
        generated_ids,
        chosen_past,
        chosen_state,
        [candidate[0] for candidate in candidates],
        sum(len(candidate[2]) for candidate in candidates),
    )


def _synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _context_key(question: str, episode_index: int) -> int:
    """Choose a deterministic two-digit key absent from the user question."""

    present = {_normalize_number(value) for value in NUMBER_RE.findall(question)}
    candidates = [value for value in range(10, 100) if str(value) not in present]
    return candidates[(episode_index * 37 + 17) % len(candidates)]


def _dependent_turn(question: str, key: int, turn_index: int) -> tuple[str, str | None, str]:
    """Return instruction, gold answer, and assistant-only prefix for one turn."""

    if turn_index == 0:
        return COT_PROMPT + question, None, f"CONTEXT KEY: {key}\n\n"
    if turn_index == 1:
        return (
            "What two-digit CONTEXT KEY was stated at the start of your previous reasoning? "
            "Answer with ONLY '#### <answer>' and no other text.",
            str(key),
            "",
        )
    return (
        "Add 7 to the CONTEXT KEY from your first response. Answer with ONLY '#### <answer>' "
        "and no other text.",
        str(key + 7),
        "",
    )


def evaluate_multiturn_retrofit(
    model_name: str,
    corrector_path: str | Path,
    num_episodes: int,
    turns: int,
    max_new_tokens: int,
    device_str: str,
    output_path: str | Path | None,
    problem_offset: int = 0,
    context_mode: str = "independent",
    sc_samples: int = 1,
    stop_agreement: int = 4,
    temperature: float = 0.6,
    seed: int = 20260804,
) -> dict:
    """Evaluate sequential GSM8K problem solving under three context policies.

    Visible CoT retains its rendered chains in the conversation KV cache.
    Latent-persistent surfaces only answers but retains hidden-chain KV entries.
    Latent-reset rebuilds the trunk context from surfaced answers each turn,
    discarding hidden-chain KV entries. Both latent arms preserve the corrector
    state across turns so the manipulated factor is trunk-cache persistence.
    Independent mode measures context accumulation and interference. Dependent
    mode places a controlled key in first-turn assistant scratch work, then asks
    later turns to recall and transform it, directly probing context carryover.
    """

    from datasets import load_dataset

    if num_episodes < 1 or turns < 2 or max_new_tokens < 1:
        raise ValueError("num_episodes and max_new_tokens must be positive; turns must be at least 2")
    if context_mode not in {"independent", "dependent"}:
        raise ValueError("context_mode must be 'independent' or 'dependent'")
    if context_mode == "dependent" and turns != 3:
        raise ValueError("dependent context mode requires exactly 3 turns")
    if sc_samples < 1 or stop_agreement < 1 or (sc_samples > 1 and stop_agreement > sc_samples):
        raise ValueError("SC samples must be positive; when enabled, stop_agreement must not exceed samples")
    device = torch.device(device_str if device_str != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model, tokenizer = load_trunk(model_name, device, dtype)
    checkpoint = torch.load(corrector_path, map_location=device, weights_only=True)
    corrector, tap_layer = load_corrector(checkpoint, device)
    torch.manual_seed(seed)
    start = problem_offset
    stop = start + (num_episodes * turns if context_mode == "independent" else num_episodes)
    rows = list(load_dataset("openai/gsm8k", "main", split="test").select(range(start, stop)))

    arm_names = ["visible_cot"]
    if sc_samples > 1:
        arm_names.append(f"visible_sc{sc_samples}")
    arm_names.append("latent_kv_persist")
    if sc_samples > 1:
        arm_names.append(f"latent_stop{stop_agreement}of{sc_samples}")
    arm_names.append("latent_kv_reset")
    stats = {
        name: {
            "correct": 0,
            "episodes_correct": 0,
            "missing_answer": 0,
            "budget_exhausted": 0,
            "emitted": 0,
            "internal": 0,
            "rollouts": 0,
            "oracle_correct": 0,
            "generation_seconds": 0.0,
            "by_turn": [0] * turns,
        }
        for name in arm_names
    }
    records: list[dict] = []
    started = time.time()

    for episode_index in range(num_episodes):
        if context_mode == "independent":
            episode = rows[episode_index * turns : (episode_index + 1) * turns]
            key = None
        else:
            episode = [rows[episode_index]] * turns
            key = _context_key(episode[0]["question"], episode_index)
        for arm in arm_names:
            sampled = arm.startswith("visible_sc") or arm.startswith("latent_stop")
            latent = arm.startswith("latent")
            reset_cache = arm == "latent_kv_reset"
            past = None
            state = None
            pending: list[int] = []
            messages: list[dict[str, str]] = []
            episode_hits = 0

            for turn_index, row in enumerate(episode):
                assistant_prefix = ""
                if context_mode == "dependent":
                    instruction, turn_gold, assistant_prefix = _dependent_turn(
                        row["question"], key, turn_index
                    )
                    if turn_index == 0:
                        turn_gold = extract_answer(row["answer"])
                else:
                    instruction = COT_PROMPT + row["question"] if turn_index == 0 else _multiturn_instruction(row["question"])
                    turn_gold = extract_answer(row["answer"])
                if turn_index == 0:
                    messages.append({"role": "user", "content": instruction})
                    prompt_text = _chat_messages_prompt(tokenizer, messages)
                    input_ids = tokenizer(prompt_text + assistant_prefix, return_tensors="pt")["input_ids"]
                else:
                    messages.append({"role": "user", "content": instruction})
                    if reset_cache:
                        prompt_text = _chat_messages_prompt(tokenizer, messages)
                        input_ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"]
                        past = None
                    else:
                        suffix = _chat_turn_suffix(tokenizer, instruction)
                        suffix_ids = tokenizer(suffix, add_special_tokens=False, return_tensors="pt")["input_ids"]
                        if pending and suffix_ids.size(1) and pending[-1] == int(suffix_ids[0, 0]):
                            suffix_ids = suffix_ids[:, 1:]
                        input_ids = torch.cat(
                            [torch.tensor([pending], dtype=suffix_ids.dtype), suffix_ids], dim=1
                        )

                _synchronize_device(device)
                generation_started = time.perf_counter()
                if sampled:
                    text, voted_answer, generated_ids, past, state, rollout_texts, internal_tokens = _sample_stateful_vote(
                        model,
                        tokenizer,
                        corrector if latent else None,
                        tap_layer,
                        input_ids,
                        max_new_tokens,
                        device,
                        past,
                        state,
                        sc_samples,
                        temperature,
                        stop_agreement if latent else 0,
                    )
                else:
                    text, generated_ids, past, state = _generate_stateful_turn(
                        model,
                        tokenizer,
                        corrector if latent else None,
                        tap_layer,
                        input_ids,
                        max_new_tokens,
                        device,
                        past=past,
                        state=state,
                    )
                    voted_answer = None
                    rollout_texts = [text]
                    internal_tokens = len(generated_ids)
                _synchronize_device(device)
                generation_seconds = time.perf_counter() - generation_started
                pending = generated_ids[-1:]
                prediction = voted_answer if sampled else extract_answer(text)
                gold = turn_gold
                hit = prediction == gold
                budget_exhausted = len(generated_ids) == max_new_tokens and (
                    tokenizer.eos_token_id is None or generated_ids[-1] != tokenizer.eos_token_id
                )
                episode_hits += int(hit)
                stats[arm]["correct"] += int(hit)
                stats[arm]["missing_answer"] += int(prediction is None)
                stats[arm]["budget_exhausted"] += int(budget_exhausted)
                stats[arm]["by_turn"][turn_index] += int(hit)
                stats[arm]["internal"] += internal_tokens
                stats[arm]["rollouts"] += len(rollout_texts)
                stats[arm]["oracle_correct"] += int(
                    any(extract_answer_lenient(rollout) == gold for rollout in rollout_texts)
                )
                stats[arm]["generation_seconds"] += generation_seconds
                full_text = assistant_prefix + text
                surfaced = full_text if not latent else (
                    f"#### {prediction}" if prediction is not None else "#### INVALID"
                )
                stats[arm]["emitted"] += len(tokenizer(surfaced, add_special_tokens=False)["input_ids"])
                if reset_cache:
                    messages.append({"role": "assistant", "content": surfaced})
                records.append(
                    {
                        "episode": episode_index,
                        "turn": turn_index + 1,
                        "arm": arm,
                        "question": instruction,
                        "context_key": key,
                        "gold": gold,
                        "prediction": prediction,
                        "correct": hit,
                        "generated_tokens": len(generated_ids),
                        "internal_tokens": internal_tokens,
                        "rollouts_used": len(rollout_texts),
                        "rollouts": rollout_texts,
                        "generation_seconds": generation_seconds,
                        "budget_exhausted": budget_exhausted,
                        "completion": full_text,
                        "surfaced": surfaced,
                    }
                )
            stats[arm]["episodes_correct"] += int(episode_hits == turns)
        print(
            json.dumps({"episodes": episode_index + 1, "elapsed": round(time.time() - started)}),
            flush=True,
        )

    total_turns = num_episodes * turns
    end_to_end_seconds = time.time() - started
    results = {}
    lines = [
        "# Multi-turn frozen-trunk retrofit evaluation",
        "",
        f"Model: `{model_name}`; corrector: `{corrector_path}`; episodes: {num_episodes}; turns: {turns}; "
        f"GSM8K offset: {problem_offset}; context mode: `{context_mode}`.",
        "",
        (
            "Each episode contains independent GSM8K problems and measures accumulated-context interference."
            if context_mode == "independent"
            else "Turn 1 solves GSM8K with an assistant-only context key in its scratch work; turn 2 recalls "
            "the key and turn 3 adds 7. The key is absent from the user prompt and latent surfaced answer."
        ),
        "Visible CoT retains prior visible chains; latent-kv-persist retains hidden-chain KV entries; "
        "latent-kv-reset rebuilds from surfaced answers. SC arms sample from an identical conversation "
        "snapshot and persist one majority-vote branch. All latent arms retain the CfC state across turns.",
        "",
        "Prompt-to-answer time is synchronized device wall time around each turn's generation call; it includes "
        "prefill and autoregressive decoding but excludes model/dataset loading, prompt tokenization, scoring, "
        "and report serialization.",
        "",
        "| system | turn accuracy | all-turn episode accuracy | missing / exhausted | emitted tok/turn | internal tok/turn | rollouts/turn | sec/turn | internal tok/sec |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm in arm_names:
        entry = stats[arm]
        results[arm] = {
            "accuracy": entry["correct"] / total_turns,
            "episode_accuracy": entry["episodes_correct"] / num_episodes,
            "missing_answer_rate": entry["missing_answer"] / total_turns,
            "budget_exhaustion_rate": entry["budget_exhausted"] / total_turns,
            "mean_emitted_tokens": entry["emitted"] / total_turns,
            "mean_internal_tokens": entry["internal"] / total_turns,
            "mean_rollouts": entry["rollouts"] / total_turns,
            "oracle_accuracy": entry["oracle_correct"] / total_turns,
            "generation_seconds": entry["generation_seconds"],
            "mean_generation_seconds": entry["generation_seconds"] / total_turns,
            "internal_tokens_per_second": entry["internal"] / entry["generation_seconds"],
            "accuracy_by_turn": [hits / num_episodes for hits in entry["by_turn"]],
            "followup_accuracy": sum(entry["by_turn"][1:]) / (num_episodes * (turns - 1)),
        }
        result = results[arm]
        lines.append(
            f"| {arm} | {result['accuracy']:.4f} | {result['episode_accuracy']:.4f} | "
            f"{result['missing_answer_rate']:.4f} / {result['budget_exhaustion_rate']:.4f} | "
            f"{result['mean_emitted_tokens']:.1f} | "
            f"{result['mean_internal_tokens']:.1f} | {result['mean_rollouts']:.2f} | "
            f"{result['mean_generation_seconds']:.3f} | "
            f"{result['internal_tokens_per_second']:.1f} |"
        )
    lines += ["", f"End-to-end evaluation time after model and dataset loading: {end_to_end_seconds:.1f} seconds."]
    lines += ["", "Accuracy by turn:", ""]
    for arm in arm_names:
        values = ", ".join(f"t{i + 1}={value:.4f}" for i, value in enumerate(results[arm]["accuracy_by_turn"]))
        lines.append(f"- `{arm}`: {values}")
    if context_mode == "dependent":
        lines += ["", "Context-dependent follow-up accuracy (turns 2-3):", ""]
        for arm in arm_names:
            lines.append(f"- `{arm}`: {results[arm]['followup_accuracy']:.4f}")
    paired = {}
    lines += ["", "Paired turn outcomes:", ""]
    comparisons = [
        ("latent_kv_persist", "visible_cot"),
        ("latent_kv_persist", "latent_kv_reset"),
        ("latent_kv_reset", "visible_cot"),
    ]
    if sc_samples > 1:
        comparisons += [
            (f"latent_stop{stop_agreement}of{sc_samples}", f"visible_sc{sc_samples}"),
            (f"latent_stop{stop_agreement}of{sc_samples}", "latent_kv_persist"),
            (f"latent_stop{stop_agreement}of{sc_samples}", "latent_kv_reset"),
        ]
    for left, right in comparisons:
        left_rows = [row for row in records if row["arm"] == left]
        right_rows = [row for row in records if row["arm"] == right]
        wins = sum(left_row["correct"] and not right_row["correct"] for left_row, right_row in zip(left_rows, right_rows))
        losses = sum(right_row["correct"] and not left_row["correct"] for left_row, right_row in zip(left_rows, right_rows))
        ties = total_turns - wins - losses
        comparison = {"wins": wins, "losses": losses, "ties": ties, "accuracy_delta": (wins - losses) / total_turns}
        paired[f"{left}_vs_{right}"] = comparison
        lines.append(
            f"- `{left}` vs. `{right}`: {wins} wins, {losses} losses, {ties} ties; "
            f"accuracy delta {comparison['accuracy_delta']:+.4f}."
        )
    report = "\n".join(lines) + "\n"
    print(report, flush=True)
    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report, encoding="utf-8")
        output.with_suffix(".json").write_text(
            json.dumps(
                {
                    "timing": {
                        "definition": "Synchronized prompt-to-answer wall time per turn",
                        "includes": ["prefill", "autoregressive decoding"],
                        "excludes": [
                            "model and dataset loading",
                            "prompt tokenization",
                            "scoring",
                            "report serialization",
                        ],
                        "per_system_warmup": False,
                        "device": str(device),
                        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                        "end_to_end_seconds_after_loading": end_to_end_seconds,
                    },
                    "context_mode": context_mode,
                    "self_consistency": {
                        "samples": sc_samples,
                        "stop_agreement": stop_agreement,
                        "temperature": temperature,
                        "seed": seed,
                    },
                    "results": results,
                    "paired": paired,
                    "records": records,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    return results


def _instrumented_rollout(
    model, tokenizer, corrector, tap_layer: int, basis: torch.Tensor,
    prompt: str, max_new_tokens: int, device: torch.device, temperature: float = 0.0,
) -> dict:
    """Latent rollout that records per-position corrector statistics and the
    answer-span window (positions after the `####` marker appears in the
    decoded text). Returns text plus chain/answer-span summary statistics."""

    encoded = tokenizer(prompt, return_tensors="pt").to(device)
    state = corrector.initial_state(1, device)
    lm_head = model.get_output_embeddings()
    past = None
    tokens = encoded["input_ids"]
    generated: list[int] = []
    eos_id = tokenizer.eos_token_id
    delta_norms: list[float] = []
    comp_fracs: list[float] = []
    marker_pos: int | None = None
    for _ in range(max_new_tokens):
        outputs = model(tokens, past_key_values=past, output_hidden_states=True, use_cache=True)
        past = outputs.past_key_values
        h_final = outputs.hidden_states[-1][:, -1, :].float()
        h_tap = outputs.hidden_states[tap_layer][:, -1, :].float()
        delta, state = corrector.step(h_tap, state)
        logits = lm_head((h_final + delta).to(outputs.hidden_states[-1].dtype)).float()
        if temperature > 0:
            next_id = int(torch.multinomial(torch.softmax(logits / temperature, dim=-1), num_samples=1))
        else:
            next_id = int(logits.argmax(dim=-1))
        generated.append(next_id)
        delta_norms.append(float(delta.norm()))
        comp = h_tap - (h_tap @ basis.T) @ basis
        comp_fracs.append(float(comp.norm() ** 2 / h_tap.norm().clamp_min(1e-6) ** 2))
        if marker_pos is None and len(generated) >= 2 and "####" in tokenizer.decode(generated[-6:]):
            marker_pos = len(generated)
        if eos_id is not None and next_id == eos_id:
            break
        tokens = torch.tensor([[next_id]], device=device)

    text = tokenizer.decode(generated, skip_special_tokens=True)
    T = len(generated)
    a0 = marker_pos if marker_pos is not None else T  # answer span start (empty if no marker)
    chain_deltas, ans_deltas = delta_norms[:a0], delta_norms[a0:]
    ans_fracs = comp_fracs[a0:]
    mean = sum(delta_norms) / max(T, 1)
    std = (sum((v - mean) ** 2 for v in delta_norms) / max(T - 1, 1)) ** 0.5
    return {
        "text": text,
        "has_marker": marker_pos is not None,
        "chain_mean_delta": sum(chain_deltas) / max(len(chain_deltas), 1),
        "ans_mean_delta": sum(ans_deltas) / max(len(ans_deltas), 1),
        "ans_max_delta": max(ans_deltas, default=0.0),
        "ans_mean_frac": sum(ans_fracs) / max(len(ans_fracs), 1),
        "ans_z": ((max(ans_deltas, default=mean) - mean) / std) if std > 1e-9 else 0.0,
        "ans_len": len(ans_deltas),
    }


def _rank_auc(scores: list[float], labels: list[int]) -> float:
    """Rank AUC of scores predicting labels==1 (ties get half credit)."""

    pos = [s for s, l in zip(scores, labels) if l == 1]
    neg = [s for s, l in zip(scores, labels) if l == 0]
    if not pos or not neg:
        return float("nan")
    wins = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def answer_monitor(
    model_name: str,
    corrector_path: str | Path,
    basis_path: str | Path,
    num_problems: int,
    samples: int,
    temperature: float,
    max_new_tokens: int,
    device_str: str,
    output_path: str | Path | None,
) -> dict:
    """Monitor the corrector's error vector during the answer phase.

    The `#### <answer>` token structure supplies a free monitoring window:
    unlike chain-phase triggers, no localization is needed. Two questions,
    one decode. (A) Signal: over greedy rollouts, does the answer-span
    corrector statistic (mean/max delta norm, complement fraction,
    within-rollout z) discriminate correct from wrong final answers (rank
    AUC), and how does it compare to the whole-chain statistic? (B)
    Iteration: over k sampled rollouts per problem, do answer-span-gated
    policies beat the plain majority vote at equal or fewer samples \u2014
    minimum answer-span delta pick, signal-weighted vote (weight
    1/(1+ans_mean_delta)), and a sequential accept-first-quiet policy
    (threshold = median answer-span delta of correct greedy rollouts,
    deployable calibration; falls back to the vote if no sample passes,
    reporting mean rollouts consumed).
    """

    from collections import Counter

    from datasets import load_dataset

    device = torch.device(device_str if device_str != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model, tokenizer = load_trunk(model_name, device, dtype)
    checkpoint = torch.load(corrector_path, map_location=device, weights_only=True)
    corrector, tap_layer = load_corrector(checkpoint, device)
    basis = torch.load(basis_path, map_location=device, weights_only=True)["basis_full"].to(torch.float32)

    dataset = load_dataset("openai/gsm8k", "main", split="test").select(range(num_problems))

    greedy_rows: list[dict] = []
    sampled_rows: list[list[dict]] = []
    start = time.time()
    for index, row in enumerate(dataset):
        gold = extract_answer(row["answer"])
        cot_prompt = _chat_prompt(tokenizer, COT_PROMPT + row["question"])
        greedy = _instrumented_rollout(model, tokenizer, corrector, tap_layer, basis, cot_prompt, max_new_tokens, device)
        greedy["correct"] = int(extract_answer_lenient(greedy["text"]) == gold)
        greedy["gold"] = gold
        greedy_rows.append(greedy)
        arms = []
        for _ in range(samples):
            r = _instrumented_rollout(model, tokenizer, corrector, tap_layer, basis, cot_prompt, max_new_tokens, device, temperature)
            r["answer"] = extract_answer_lenient(r["text"])
            arms.append(r)
        sampled_rows.append(arms)
        if (index + 1) % 10 == 0:
            print(json.dumps({"answer_monitor": index + 1, "elapsed": round(time.time() - start)}), flush=True)

    # (A) Signal: AUC of each statistic for predicting a WRONG greedy answer.
    marked = [g for g in greedy_rows if g["has_marker"]]
    labels = [1 - g["correct"] for g in marked]
    stats = ["ans_mean_delta", "ans_max_delta", "ans_mean_frac", "ans_z", "chain_mean_delta"]
    aucs = {s: _rank_auc([g[s] for g in marked], labels) for s in stats}

    # (B) Policies over the k sampled rollouts.
    threshold = sorted(g["ans_mean_delta"] for g in marked if g["correct"])
    threshold = threshold[len(threshold) // 2] if threshold else 0.0
    gate_names = ["vote", "min-ans-delta", "weighted-vote", "seq-accept", "greedy"]
    correct = {name: 0 for name in gate_names}
    seq_rollouts = 0
    for g, arms in zip(greedy_rows, sampled_rows):
        gold = g["gold"]
        answered = [a for a in arms if a["answer"] is not None]
        votes = Counter(a["answer"] for a in answered)
        vote = votes.most_common(1)[0][0] if votes else None
        min_pick = min(answered, key=lambda a: a["ans_mean_delta"])["answer"] if answered else None
        weights: dict = {}
        for a in answered:
            weights[a["answer"]] = weights.get(a["answer"], 0.0) + 1.0 / (1.0 + a["ans_mean_delta"])
        weighted = max(weights, key=weights.get) if weights else None
        seq = None
        used = len(arms)
        for i, a in enumerate(arms):
            if a["answer"] is not None and a["has_marker"] and a["ans_mean_delta"] <= threshold:
                seq = a["answer"]
                used = i + 1
                break
        if seq is None:
            seq = vote
        seq_rollouts += used
        picks = {"vote": vote, "min-ans-delta": min_pick, "weighted-vote": weighted, "seq-accept": seq,
                 "greedy": extract_answer_lenient(g["text"])}
        for name in gate_names:
            correct[name] += int(picks[name] == gold)

    count = len(greedy_rows)
    lines = [
        "# Answer-phase error monitoring", "",
        f"Model: `{model_name}`, corrector: `{corrector_path}`, problems: {count}, "
        f"samples: {samples} @ T={temperature}",
        f"Greedy rollouts with `####` marker: {len(marked)}/{count}; "
        f"mean answer-span length {sum(g['ans_len'] for g in marked) / max(len(marked), 1):.1f} tokens; "
        f"sequential-accept threshold {threshold:.2f} (median of correct greedy)", "",
        "## (A) Does the answer-span signal predict a wrong answer? (rank AUC, greedy arm)", "",
        "| statistic | AUC |", "| --- | --- |",
    ]
    for s in stats:
        lines.append(f"| {s} | {aucs[s]:.4f} |")
    lines += ["", "## (B) Answer-span-gated policies over the sampled rollouts", "",
              "| policy | accuracy | mean rollouts |", "| --- | --- | --- |"]
    for name in gate_names:
        rolls = f"{seq_rollouts / count:.2f}" if name == "seq-accept" else (f"{samples}" if name != "greedy" else "1")
        lines.append(f"| {name} | {correct[name] / count:.4f} | {rolls} |")
    report = "\n".join(lines) + "\n"
    print(report, flush=True)
    if output_path is not None:
        Path(output_path).write_text(report, encoding="utf-8")
    return {"aucs": aucs, "policies": {k: v / count for k, v in correct.items()}}


def _generate_batch_with_corrector(
    model,
    tokenizer,
    corrector,
    tap_layer: int,
    prompt: str,
    max_new_tokens: int,
    device: torch.device,
    temperature: float,
    batch_size: int,
) -> list[str]:
    """Batched sampled rollouts sharing one prompt (roadmap 4b).

    The latent chain has no streaming contract, so a k-way vote's rollouts
    can run as one batch: identical prompt prefill, per-sequence sampling,
    batched corrector state. Finished sequences keep feeding EOS to keep the
    KV cache aligned; their outputs are discarded.
    """

    encoded = tokenizer(prompt, return_tensors="pt").to(device)
    tokens = encoded["input_ids"].expand(batch_size, -1).contiguous()
    state = corrector.initial_state(batch_size, device) if corrector is not None else None
    lm_head = model.get_output_embeddings()
    past = None
    generated: list[list[int]] = [[] for _ in range(batch_size)]
    eos_id = tokenizer.eos_token_id
    done = torch.zeros(batch_size, dtype=torch.bool, device=device)
    for _ in range(max_new_tokens):
        outputs = model(tokens, past_key_values=past, output_hidden_states=True, use_cache=True)
        past = outputs.past_key_values
        h_final = outputs.hidden_states[-1][:, -1, :].float()
        if corrector is not None:
            h_tap = outputs.hidden_states[tap_layer][:, -1, :].float()
            delta, state = corrector.step(h_tap, state)
            h_final = h_final + delta
        logits = lm_head(h_final.to(outputs.hidden_states[-1].dtype)).float()
        if temperature > 0:
            probabilities = torch.softmax(logits / temperature, dim=-1)
            next_ids = torch.multinomial(probabilities, num_samples=1).squeeze(1)
        else:
            next_ids = logits.argmax(dim=-1)
        if eos_id is not None:
            next_ids = torch.where(done, torch.full_like(next_ids, eos_id), next_ids)
        for b in range(batch_size):
            if not done[b]:
                generated[b].append(int(next_ids[b]))
        if eos_id is not None:
            done = done | (next_ids == eos_id)
            if bool(done.all()):
                break
        tokens = next_ids.unsqueeze(1)
    return [tokenizer.decode(seq, skip_special_tokens=True) for seq in generated]


def benchmark_retrofit_latency(
    model_name: str,
    corrector_path: str | Path,
    num_problems: int,
    max_new_tokens: int,
    device_str: str,
    output_path: str | Path | None,
    temperature: float = 0.6,
    seed: int = 20260725,
) -> dict:
    """Measure synchronized prompt-to-answer latency after per-system warm-up."""

    import statistics
    from collections import Counter
    from datasets import load_dataset

    device = torch.device(device_str if device_str != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model, tokenizer = load_trunk(model_name, device, dtype)
    checkpoint = torch.load(corrector_path, map_location=device, weights_only=True)
    corrector, tap_layer = load_corrector(checkpoint, device)
    dataset = load_dataset("openai/gsm8k", "main", split="test").select(range(num_problems))
    torch.manual_seed(seed)

    def synchronize() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def generate(system: str, prompt: str) -> list[str]:
        if system == "visible_cot":
            return [_generate_with_corrector(model, tokenizer, None, 0, prompt, max_new_tokens, device)]
        if system == "visible_sc8":
            return _generate_batch_with_corrector(
                model, tokenizer, None, 0, prompt, max_new_tokens, device, temperature, 8
            )
        if system == "latent_greedy":
            return [_generate_with_corrector(
                model, tokenizer, corrector, tap_layer, prompt, max_new_tokens, device
            )]
        rollouts: list[str] = []
        votes: Counter = Counter()
        while len(rollouts) < 8 and (not votes or votes.most_common(1)[0][1] < 4):
            wave = 4 if not rollouts else min(2, 8 - len(rollouts))
            sampled = _generate_batch_with_corrector(
                model, tokenizer, corrector, tap_layer, prompt, max_new_tokens, device, temperature, wave
            )
            rollouts.extend(sampled)
            votes.update(answer for text in sampled if (answer := extract_answer_lenient(text)) is not None)
        return rollouts

    systems = ("visible_cot", "visible_sc8", "latent_greedy", "latent_stop4of8")
    results = {}
    for system in systems:
        warmup_prompt = _chat_prompt(tokenizer, COT_PROMPT + dataset[0]["question"])
        generate(system, warmup_prompt)
        synchronize()
        seconds = []
        rollout_counts = []
        for row in dataset:
            prompt = _chat_prompt(tokenizer, COT_PROMPT + row["question"])
            synchronize()
            started = time.perf_counter()
            rollouts = generate(system, prompt)
            synchronize()
            seconds.append(time.perf_counter() - started)
            rollout_counts.append(len(rollouts))
        ordered = sorted(seconds)
        p90_index = min(len(ordered) - 1, math.ceil(0.9 * len(ordered)) - 1)
        results[system] = {
            "mean_seconds": statistics.mean(seconds),
            "median_seconds": statistics.median(seconds),
            "p90_seconds": ordered[p90_index],
            "mean_rollouts": statistics.mean(rollout_counts),
            "samples": seconds,
        }
        print(json.dumps({"system": system, **results[system]}, default=float), flush=True)

    hardware = torch.cuda.get_device_name(device) if device.type == "cuda" else str(device)
    payload = {
        "model": model_name,
        "hardware": hardware,
        "problems": num_problems,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "seed": seed,
        "results": results,
    }
    lines = [
        "# Prompt-to-answer latency", "",
        f"Model: `{model_name}`; hardware: {hardware}; GSM8K problems: {num_problems}; "
        "one warm-up per system; CUDA synchronized before and after every problem.", "",
        "| system | mean s/problem | median | p90 | mean rollouts |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for system, entry in results.items():
        lines.append(
            f"| {system} | {entry['mean_seconds']:.3f} | {entry['median_seconds']:.3f} "
            f"| {entry['p90_seconds']:.3f} | {entry['mean_rollouts']:.2f} |"
        )
    report = "\n".join(lines) + "\n"
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(report, encoding="utf-8")
        path.with_suffix(".json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(report, flush=True)
    return payload


def _clone_state_row(state, row: int):
    """Append a copy of `row` to a corrector state (tensor or nested list)."""

    if state is None:
        return None
    if isinstance(state, list):
        return [_clone_state_row(member, row) for member in state]
    return torch.cat([state, state[row : row + 1]], dim=0)


def _clone_kv_row(past, row: int):
    """Append a copy of batch `row` to every layer of a KV cache (in place)."""

    if hasattr(past, "layers"):  # transformers >= 4.54 DynamicCache
        for layer in past.layers:
            layer.keys = torch.cat([layer.keys, layer.keys[row : row + 1]], dim=0)
            layer.values = torch.cat([layer.values, layer.values[row : row + 1]], dim=0)
        return past
    if hasattr(past, "key_cache"):  # older DynamicCache
        for i in range(len(past.key_cache)):
            past.key_cache[i] = torch.cat([past.key_cache[i], past.key_cache[i][row : row + 1]], dim=0)
            past.value_cache[i] = torch.cat([past.value_cache[i], past.value_cache[i][row : row + 1]], dim=0)
        return past
    return tuple(tuple(torch.cat([t, t[row : row + 1]], dim=0) for t in layer) for layer in past)


def _generate_dynamic_sc(
    model,
    tokenizer,
    corrector,
    tap_layer: int,
    prompt: str,
    max_new_tokens: int,
    device: torch.device,
    temperature: float,
    max_rollouts: int,
    branch_z: float = 2.5,
    branch_cooldown: int = 16,
    warmup: int = 16,
) -> list[str]:
    """Dynamic self-consistency: branch rollouts only where the corrector signals.

    Fixed SC@k pays k rollouts on every problem; stop-on-agreement still
    starts sampling from step 0. Here a single greedy rollout runs until the
    corrector's own correction magnitude flags trouble — the per-beam
    z-score of ||delta|| against that beam's running statistics (the
    self-normalized trigger of the trigger-lab bake-off) crossing
    `branch_z` — and only then does the decode *branch*: the beam's KV
    cache and corrector state are duplicated, and the child is forced onto
    a different token at the flagged site (the sibling-token move from the
    rollback experiments: probabilities renormalized with the parent's
    choice masked out), then continues sampling at `temperature`. Beams cap
    at `max_rollouts`; `branch_cooldown` decode steps must pass on a beam
    between branches; the first `warmup` steps only collect statistics.
    Beam 0 stays greedy throughout, so with no trigger firing the output is
    exactly the greedy latent rollout (the floor, again). The final answer
    is the majority lenient vote over all beams, exactly as in fixed SC.

    In effect SC@k where k is chosen per problem, and per *site*, by the
    corrector: consensus is bought only where the error-monitor asks for it.
    """

    encoded = tokenizer(prompt, return_tensors="pt").to(device)
    tokens = encoded["input_ids"]
    state = corrector.initial_state(1, device)
    lm_head = model.get_output_embeddings()
    past = None
    generated: list[list[int]] = [[]]
    eos_id = tokenizer.eos_token_id
    done = [False]
    sampled = [False]  # beam 0 greedy; children sampled
    # Per-beam Welford statistics over ||delta|| and per-beam branch cooldown.
    stat_count, stat_mean, stat_m2, cooldown = [0], [0.0], [0.0], [0]

    for _ in range(max_new_tokens):
        outputs = model(tokens, past_key_values=past, output_hidden_states=True, use_cache=True)
        past = outputs.past_key_values
        h_final = outputs.hidden_states[-1][:, -1, :].float()
        h_tap = outputs.hidden_states[tap_layer][:, -1, :].float()
        delta, state = corrector.step(h_tap, state)
        logits = lm_head((h_final + delta).to(outputs.hidden_states[-1].dtype)).float()
        norms = delta.norm(dim=-1)
        probabilities = torch.softmax(logits / max(temperature, 1e-6), dim=-1)
        greedy_ids = logits.argmax(dim=-1)
        sampled_ids = torch.multinomial(probabilities, num_samples=1).squeeze(1)

        n_beams = len(generated)
        next_ids: list[int] = [0] * n_beams
        branch_from: list[int] = []
        for b in range(n_beams):
            if done[b]:
                next_ids[b] = eos_id if eos_id is not None else 0
                continue
            value = float(norms[b])
            # Trigger check against the beam's own history, then update stats.
            triggered = False
            if stat_count[b] >= warmup and cooldown[b] <= 0 and len(generated) + len(branch_from) < max_rollouts:
                std = (stat_m2[b] / max(stat_count[b] - 1, 1)) ** 0.5
                if std > 1e-9 and (value - stat_mean[b]) / std >= branch_z:
                    triggered = True
            stat_count[b] += 1
            diff = value - stat_mean[b]
            stat_mean[b] += diff / stat_count[b]
            stat_m2[b] += diff * (value - stat_mean[b])
            cooldown[b] = max(cooldown[b] - 1, 0)
            next_ids[b] = int(sampled_ids[b]) if sampled[b] else int(greedy_ids[b])
            if triggered:
                branch_from.append(b)
                cooldown[b] = branch_cooldown

        for b in branch_from:
            past = _clone_kv_row(past, b)
            state = _clone_state_row(state, b)
            # Sibling token: sample the child away from the parent's choice.
            child_probs = probabilities[b].clone()
            child_probs[next_ids[b]] = 0.0
            total = float(child_probs.sum())
            child_id = int(torch.multinomial(child_probs, 1)) if total > 1e-9 else next_ids[b]
            generated.append(list(generated[b]))
            done.append(False)
            sampled.append(True)
            stat_count.append(stat_count[b])
            stat_mean.append(stat_mean[b])
            stat_m2.append(stat_m2[b])
            cooldown.append(branch_cooldown)
            next_ids.append(child_id)

        for b in range(len(generated)):
            if not done[b]:
                generated[b].append(next_ids[b])
                if eos_id is not None and next_ids[b] == eos_id:
                    done[b] = True
        if all(done):
            break
        tokens = torch.tensor([[i] for i in next_ids], device=device)
    return [tokenizer.decode(seq, skip_special_tokens=True) for seq in generated]


def _generate_grounded_continuous(
    model,
    tokenizer,
    corrector,
    tap_layer: int,
    prompt: str,
    reasoning_steps: int,
    device: torch.device,
    ground_every: int,
    feedback: str = "expected",
    answer_tokens: int = 12,
    snap: SnapProjector | None = None,
) -> tuple[str, int, str]:
    """Continuous latent rollout with periodic token grounding.

    Coconut-style continuous reasoning: instead of decoding a token at every
    step, feed a continuous vector back as the next input embedding —
    feedback="expected" uses the probability-weighted embedding
    E_p[e] = softmax(logits) @ embed (stays on the embedding manifold by
    construction); feedback="hidden" feeds the final hidden state itself,
    norm-matched to the embedding scale (the raw Coconut move).

    ground_every=G decodes a *real* greedy token every G-th step, snapping
    the trajectory back onto the token manifold; G=1 is the ordinary latent
    rollout (control), G=0 never grounds (pure continuous — the v2-collapse
    arm).

    Termination is natural, not scheduled: every step's argmax is a "shadow
    token" (the output the chain would have emitted — computed anyway,
    normally discarded during continuous steps). The shadow stream is
    monitored for the answer marker '####'; when it appears the answer phase
    starts — on a grounded step the marker is already in context, on a
    continuous step the anchor tokens are injected ("we think we see it, so
    we force the start of the answer phase"). `reasoning_steps` is only a
    budget cap: if the marker never surfaces, the anchor is injected anyway.

    Returns (decoded_text, total_internal_steps, termination) where
    termination is 'natural' (marker emitted on a grounded step),
    'detected' (marker seen in the shadow stream mid-continuous, anchor
    injected), or 'budget' (cap reached, anchor injected). The corrector
    (optional) watches h_tap and adjusts logits at every step, continuous or
    grounded, exactly as in the token-space rollout.
    """

    encoded = tokenizer(prompt, return_tensors="pt").to(device)
    state = corrector.initial_state(1, device) if corrector is not None else None
    lm_head = model.get_output_embeddings()
    embed = model.get_input_embeddings()
    model_dtype = embed.weight.dtype
    mean_embed_norm = embed.weight.float().norm(dim=1).mean()
    past = None
    generated: list[int] = []
    eos_id = tokenizer.eos_token_id
    step_inputs = {"input_ids": encoded["input_ids"]}
    steps_taken = 0
    shadow_window: list[int] = []
    termination = "budget"
    need_anchor = True

    def marker_seen() -> bool:
        return "####" in tokenizer.decode(shadow_window, skip_special_tokens=True)

    for step in range(reasoning_steps):
        outputs = model(**step_inputs, past_key_values=past, output_hidden_states=True, use_cache=True)
        past = outputs.past_key_values
        steps_taken += 1
        h_final = outputs.hidden_states[-1][:, -1, :].float()
        if corrector is not None:
            h_tap = outputs.hidden_states[tap_layer][:, -1, :].float()
            delta, state = corrector.step(h_tap, state)
            h_final = h_final + delta
        logits = lm_head(h_final.to(model_dtype)).float()
        shadow_id = int(logits.argmax(dim=-1))
        shadow_window = (shadow_window + [shadow_id])[-8:]
        grounded = ground_every > 0 and (step + 1) % ground_every == 0
        if grounded:
            if eos_id is not None and shadow_id == eos_id:
                termination = "natural"
                need_anchor = "####" not in tokenizer.decode(generated, skip_special_tokens=True)
                break
            generated.append(shadow_id)
            step_inputs = {"input_ids": torch.tensor([[shadow_id]], device=device)}
            if marker_seen():
                termination = "natural"
                need_anchor = False
                break
        else:
            if marker_seen():
                termination = "detected"
                break
            if feedback == "hidden":
                vector = h_final * (mean_embed_norm / (h_final.norm() + 1e-6))
            else:
                probabilities = torch.softmax(logits, dim=-1)
                vector = (probabilities.to(model_dtype) @ embed.weight).float()
            if snap is not None:
                vector = snap(vector.float())
            step_inputs = {"inputs_embeds": vector.to(model_dtype).unsqueeze(1)}

    # Answer phase: real-token decode; inject the anchor unless the marker is
    # already sitting in context from a grounded step.
    if need_anchor:
        anchor_ids = tokenizer("\n#### ", add_special_tokens=False)["input_ids"]
        step_inputs = {"input_ids": torch.tensor([anchor_ids], device=device)}
        generated.extend(anchor_ids)
    for _ in range(answer_tokens):
        outputs = model(**step_inputs, past_key_values=past, output_hidden_states=True, use_cache=True)
        past = outputs.past_key_values
        steps_taken += 1
        h_final = outputs.hidden_states[-1][:, -1, :].float()
        if corrector is not None:
            h_tap = outputs.hidden_states[tap_layer][:, -1, :].float()
            delta, state = corrector.step(h_tap, state)
            h_final = h_final + delta
        logits = lm_head(h_final.to(model_dtype)).float()
        next_id = int(logits.argmax(dim=-1))
        if eos_id is not None and next_id == eos_id:
            break
        generated.append(next_id)
        step_inputs = {"input_ids": torch.tensor([[next_id]], device=device)}
    return tokenizer.decode(generated, skip_special_tokens=True), steps_taken, termination


def evaluate_grounded_continuous(
    model_name: str,
    corrector_path: str | Path | None,
    num_problems: int,
    reasoning_steps: int,
    device_str: str,
    output_path: str | Path | None,
    ground_every_values: tuple[int, ...] = (1, 4, 8, 16, 0),
    feedback: str = "expected",
    snap_path: str | Path | None = None,
) -> dict:
    """Grounding-frequency dose-response for continuous latent reasoning.

    Hypothesis (periodic-grounding): pure continuous feedback drifts off the
    token manifold and collapses (the v2/H10 failure), but grounding with a
    real token every ~G steps re-anchors the trajectory — accuracy should
    fall from the G=1 control toward the G=0 collapse arm as G grows, and
    the shape of that curve says how much reasoning survives between
    groundings.
    """

    from datasets import load_dataset

    device = torch.device(device_str if device_str != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model, tokenizer = load_trunk(model_name, device, dtype)

    corrector = None
    tap_layer = 0
    if corrector_path is not None:
        checkpoint = torch.load(corrector_path, map_location=device, weights_only=True)
        corrector, tap_layer = load_corrector(checkpoint, device)

    snap = None
    if snap_path is not None:
        snap, snap_trained_input = load_snap(snap_path, device)
        if snap_trained_input != feedback:
            raise ValueError(f"snap module was trained for feedback='{snap_trained_input}', eval uses '{feedback}'")

    dataset = load_dataset("openai/gsm8k", "main", split="test").select(range(num_problems))
    arms = {g: {"correct": 0, "steps": 0, "natural": 0, "detected": 0, "budget": 0} for g in ground_every_values}
    start = time.time()
    for i, row in enumerate(dataset):
        gold = extract_answer(row["answer"])
        prompt = _chat_prompt(tokenizer, COT_PROMPT + row["question"])
        for g in ground_every_values:
            with torch.no_grad():
                text, steps, termination = _generate_grounded_continuous(
                    model, tokenizer, corrector, tap_layer, prompt,
                    reasoning_steps, device, ground_every=g, feedback=feedback, snap=snap,
                )
            arms[g]["steps"] += steps
            arms[g][termination] += 1
            arms[g]["correct"] += extract_answer_lenient(text) == gold
        if (i + 1) % 20 == 0:
            print(
                json.dumps(
                    {
                        "grounded_eval": i + 1,
                        "accuracies": {str(g): round(arms[g]["correct"] / (i + 1), 4) for g in ground_every_values},
                        "elapsed": round(time.time() - start),
                    }
                ),
                flush=True,
            )

    total = len(dataset)
    results = [
        {
            "ground_every": g,
            "accuracy": arms[g]["correct"] / max(total, 1),
            "mean_internal_steps": arms[g]["steps"] / max(total, 1),
            "natural": arms[g]["natural"],
            "detected": arms[g]["detected"],
            "budget": arms[g]["budget"],
        }
        for g in ground_every_values
    ]
    summary = {
        "model": model_name,
        "corrector": str(corrector_path) if corrector_path else None,
        "snap": str(snap_path) if snap_path else None,
        "feedback": feedback,
        "problems": total,
        "reasoning_steps": reasoning_steps,
        "results": results,
    }
    print(json.dumps(summary), flush=True)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# Grounded continuous latent reasoning (periodic re-anchoring)",
            "",
            f"Model: `{model_name}`, corrector: `{corrector_path}`, feedback: {feedback}, "
            f"snap: `{snap_path}`, "
            f"problems: {total}, reasoning steps: {reasoning_steps}",
            "",
            "Continuous (Coconut-style) latent steps feed a vector back as the",
            "next input embedding; every G-th step decodes a real greedy token",
            "to re-anchor the trajectory on the token manifold. G=1 is the",
            "ordinary latent rollout (control); G=0 never grounds (pure",
            "continuous). Termination is natural: every step's argmax is a",
            "shadow token (discarded during continuous steps) monitored for the",
            "'####' answer marker — natural = marker emitted on a grounded step,",
            "detected = marker seen in the shadow stream mid-continuous (anchor",
            "injected), budget = step cap hit (anchor injected).",
            "",
            "| ground every | accuracy (lenient) | mean internal steps | natural | detected | budget |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for entry in results:
            label = str(entry["ground_every"]) if entry["ground_every"] > 0 else "never"
            lines.append(
                f"| {label} | **{entry['accuracy']:.4f}** | {entry['mean_internal_steps']:.1f} | "
                f"{entry['natural']} | {entry['detected']} | {entry['budget']} |"
            )
        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def train_dynamics(
    model_name: str,
    traces_path: str | Path,
    output_dir: str | Path,
    tap_layer: int,
    d_cfc: int,
    max_steps: int,
    learning_rate: float,
    device_str: str,
    max_seq_len: int = 640,
    log_interval: int = 25,
    bptt_chunk: int = 0,
    cell: str = "cfc",
) -> dict:
    """Train a tap-space dynamics model: predict the *next* tap state.

    The sandwich architecture's middle: where the corrector reads h_tap and
    nudges the trunk's output, the dynamics model must *continue* the tap
    trajectory on its own — pred_t = h_tap[t] + delta_t targeting
    h_tap[t+1], teacher-forced over harvested traces (state warmed on the
    prompt region, loss on the completion region only, since prompt
    dynamics are driven by external tokens the cell cannot foresee). Same
    chassis as the corrector (zero-init delta head: the initial prediction
    is "the state stays put"), so load_corrector loads it unchanged.

    Loss is per-position normalized MSE ||pred - target||^2 / ||target||^2;
    every log_interval steps an *open-loop* diagnostic rolls the cell
    closed-loop from the completion start and reports cosine similarity to
    the true tap trajectory at horizons 1/8/32 — the number that predicts
    whether the sandwich can survive without grounding.
    """

    device = torch.device(device_str if device_str != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model, tokenizer = load_trunk(model_name, device, dtype)
    d_model = model.config.hidden_size

    dynamics = HiddenDeltaCorrector(d_model=d_model, d_cfc=d_cfc, cell=cell).to(device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(dynamics.parameters(), lr=learning_rate, weight_decay=0.01)

    traces = [json.loads(line) for line in Path(traces_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not traces:
        raise ValueError(f"No traces found in {traces_path}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    generator = torch.Generator().manual_seed(1337)
    start = time.time()

    def _openloop_cosines(h_tap: torch.Tensor, prompt_len: int, horizons=(1, 8, 32)) -> dict:
        """Closed-loop rollout from the completion start vs the true trajectory."""

        with torch.no_grad():
            state = dynamics.initial_state(1, device)
            for t in range(prompt_len):
                _, state = dynamics.step(h_tap[:, t, :], state)
            h = h_tap[:, prompt_len - 1, :]
            out = {}
            steps = min(max(horizons), h_tap.size(1) - prompt_len)
            for k in range(1, steps + 1):
                delta, state = dynamics.step(h, state)
                h = h + delta
                if k in horizons:
                    out[f"openloop_cos@{k}"] = float(F.cosine_similarity(h, h_tap[:, prompt_len - 1 + k, :], dim=-1))
        return out

    with metrics_path.open("w", encoding="utf-8") as metrics:
        for step in range(max_steps):
            trace = traces[int(torch.randint(len(traces), (1,), generator=generator))]
            prompt_ids = tokenizer(trace["prompt"], add_special_tokens=False)["input_ids"]
            completion_ids = tokenizer(trace["completion"], add_special_tokens=False)["input_ids"]
            input_ids = (prompt_ids + completion_ids)[:max_seq_len]
            prompt_len = len(prompt_ids)
            if len(input_ids) - prompt_len < 8:
                continue
            batch = torch.tensor([input_ids], device=device)

            with torch.no_grad():
                outputs = model(batch, output_hidden_states=True, use_cache=False, logits_to_keep=1)
            h_tap = outputs.hidden_states[tap_layer].float()  # (1, L, d)
            del outputs
            seq_len = h_tap.size(1)
            target_norm = h_tap[:, prompt_len:, :].pow(2).sum(-1).clamp_min(1e-6)  # (1, L-P)

            optimizer.zero_grad(set_to_none=True)

            def _chunk_loss(begin: int, end: int, state):
                """Teacher-forced predictions for input positions [begin, end)."""

                preds = []
                for t in range(begin, end):
                    delta, state = dynamics.step(h_tap[:, t, :], state)
                    preds.append(h_tap[:, t, :] + delta)
                pred = torch.stack(preds, dim=1)  # predicts positions begin+1..end
                lo, hi = max(begin + 1, prompt_len), end + 1
                if lo >= hi:
                    return None, state
                err = (pred[:, lo - begin - 1 :, :] - h_tap[:, lo:hi, :]).pow(2).sum(-1)
                loss = (err / target_norm[:, lo - prompt_len : hi - prompt_len]).sum()
                return loss, state

            n_targets = seq_len - prompt_len
            chunk = bptt_chunk if bptt_chunk > 0 else seq_len
            state = dynamics.initial_state(1, device)
            loss_value = 0.0
            for begin in range(0, seq_len - 1, chunk):
                end = min(begin + chunk, seq_len - 1)
                state = state.detach() if isinstance(state, torch.Tensor) else state
                loss, state = _chunk_loss(begin, end, state)
                if loss is not None:
                    (loss / n_targets).backward()
                    loss_value += float(loss) / n_targets
            torch.nn.utils.clip_grad_norm_(dynamics.parameters(), 1.0)
            optimizer.step()

            if step % log_interval == 0 or step == max_steps - 1:
                record = {"step": step, "loss": loss_value, "elapsed_seconds": time.time() - start}
                record.update(_openloop_cosines(h_tap, prompt_len))
                metrics.write(json.dumps(record) + "\n")
                metrics.flush()
                print(json.dumps(record), flush=True)

    checkpoint = {
        "corrector_state": dynamics.state_dict(),
        "config": {
            "model_name": model_name,
            "tap_layer": tap_layer,
            "d_cfc": d_cfc,
            "d_model": d_model,
            "cell": cell,
            "max_steps": max_steps,
            "learning_rate": learning_rate,
            "objective": "dynamics",
        },
    }
    torch.save(checkpoint, output_dir / "dynamics.pt")
    summary = {"steps": max_steps, "traces": len(traces), "seconds": time.time() - start}
    (output_dir / "run.summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)
    return summary


def _upper_half_logits(model, tap_layer: int, hidden: torch.Tensor) -> torch.Tensor:
    """Run the trunk's upper half (layers tap_layer..end, norm, lm_head).

    `hidden` is a (1, T, d) sequence interpreted as hidden_states[tap_layer]
    — the input to decoder layer tap_layer. Causal attention over the full
    sequence with positions 0..T-1, no cache. This is the sandwich's
    discrete decoder: it re-embeds a latent tap trajectory into token
    logits using nothing but the frozen trunk's own upper layers.
    """

    base = model.model
    seq = hidden.size(1)
    position_ids = torch.arange(seq, device=hidden.device).unsqueeze(0)
    position_embeddings = base.rotary_emb(hidden, position_ids)
    mask = torch.full((seq, seq), torch.finfo(hidden.dtype).min, device=hidden.device, dtype=hidden.dtype)
    mask = torch.triu(mask, diagonal=1)[None, None]
    for layer in base.layers[tap_layer:]:
        out = layer(hidden, attention_mask=mask, position_ids=position_ids, position_embeddings=position_embeddings)
        hidden = out[0] if isinstance(out, tuple) else out
    return model.get_output_embeddings()(base.norm(hidden))


def _generate_sandwich(
    model,
    tokenizer,
    dynamics,
    tap_layer: int,
    prompt: str,
    latent_steps: int,
    device: torch.device,
    answer_tokens: int = 24,
) -> tuple[str, str]:
    """Sandwich rollout: discrete encoder | latent CfC recurrence | discrete decoder.

    The v2/H10 collapse (Section 8/9 of the paper) came from continuous
    vectors re-entering the trunk's *input embedding* — the one interface
    that is strictly token-manifold-competent. The sandwich never touches
    it: (1) the trunk's lower half runs once on the discrete prompt and
    frontloads the tap-layer state; (2) the dynamics cell recurs purely in
    tap space for `latent_steps` steps at ~1e-4 the cost of a trunk forward
    each — zero trunk forwards during reasoning; (3) the trunk's upper half
    runs once over [prompt taps ; latent trajectory] to unembed the latent
    CoT into discrete tokens, and a normal token-space answer phase runs on
    [prompt + decoded CoT + '#### ' anchor].

    Returns (final_text, decoded_cot) — final_text carries the '####'
    answer span, decoded_cot is the rendered latent chain for inspection.
    """

    encoded = tokenizer(prompt, return_tensors="pt").to(device)
    model_dtype = model.get_input_embeddings().weight.dtype
    with torch.no_grad():
        outputs = model(**encoded, output_hidden_states=True, use_cache=False, logits_to_keep=1)
        prompt_tap = outputs.hidden_states[tap_layer].float()  # (1, P, d)
        del outputs
        prompt_len = prompt_tap.size(1)

        # Warm the cell on the prompt trajectory (teacher-forced), then roll open loop.
        state = dynamics.initial_state(1, device)
        for t in range(prompt_len):
            delta, state = dynamics.step(prompt_tap[:, t, :], state)
        h = prompt_tap[:, -1, :] + delta
        latents = [h]
        for _ in range(latent_steps - 1):
            delta, state = dynamics.step(h, state)
            h = h + delta
            latents.append(h)
        latent_seq = torch.stack(latents, dim=1)

        full = torch.cat([prompt_tap, latent_seq], dim=1).to(model_dtype)
        logits = _upper_half_logits(model, tap_layer, full)
        decoded_ids = logits[0, prompt_len - 1 : prompt_len + latent_steps - 1].argmax(-1).tolist()

    eos_id = tokenizer.eos_token_id
    if eos_id is not None and eos_id in decoded_ids:
        decoded_ids = decoded_ids[: decoded_ids.index(eos_id)]
    decoded_cot = tokenizer.decode(decoded_ids, skip_special_tokens=True)
    cot_text = decoded_cot.split("####")[0]  # keep the anchor we inject as the first marker

    answer = _generate_with_corrector(
        model, tokenizer, None, 0, prompt + cot_text + "\n#### ", answer_tokens, device
    )
    return cot_text + "\n#### " + answer, decoded_cot


def evaluate_sandwich(
    model_name: str,
    dynamics_path: str | Path,
    num_problems: int,
    latent_steps: int,
    device_str: str,
    output_path: str | Path | None,
    answer_tokens: int = 24,
    max_new_tokens: int = 512,
) -> dict:
    """Compare cot / sandwich on GSM8K test; verify the decoder plumbing first.

    The plumbing check runs the upper half on *real* tap states from the
    first problem's CoT prompt and asserts the logits match the full
    forward (the decoder is exact on-manifold, so any sandwich failure is
    the dynamics model's, not the decoder's).
    """

    from datasets import load_dataset

    device = torch.device(device_str if device_str != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model, tokenizer = load_trunk(model_name, device, dtype)

    checkpoint = torch.load(dynamics_path, map_location=device, weights_only=True)
    if checkpoint["config"].get("objective") != "dynamics":
        raise ValueError(f"{dynamics_path} is not a dynamics checkpoint (train with retrofit-train-dynamics)")
    dynamics, tap_layer = load_corrector(checkpoint, device)

    dataset = load_dataset("openai/gsm8k", "main", split="test").select(range(num_problems))

    # Plumbing check: upper half on real taps must reproduce the model's own logits.
    probe = tokenizer(_chat_prompt(tokenizer, COT_PROMPT + dataset[0]["question"]), return_tensors="pt").to(device)
    with torch.no_grad():
        ref = model(**probe, output_hidden_states=True, use_cache=False)
        rebuilt = _upper_half_logits(model, tap_layer, ref.hidden_states[tap_layer])
        agreement = float((rebuilt.argmax(-1) == ref.logits.argmax(-1)).float().mean())
        del ref, rebuilt
    print(json.dumps({"decoder_plumbing_argmax_agreement": agreement}), flush=True)
    if agreement < 0.99:
        raise RuntimeError(f"upper-half decoder disagrees with full forward (argmax agreement {agreement:.4f})")

    names = ["cot", "sandwich"]
    strict = {name: 0 for name in names}
    lenient = {name: 0 for name in names}
    internal = {name: 0 for name in names}

    dump_sink = None
    if output_path is not None:
        dump_path = Path(output_path).with_suffix(".completions.jsonl")
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        dump_sink = dump_path.open("w", encoding="utf-8")

    start = time.time()
    for index, row in enumerate(dataset):
        gold = extract_answer(row["answer"])
        cot_prompt = _chat_prompt(tokenizer, COT_PROMPT + row["question"])
        cot_text = _generate_with_corrector(model, tokenizer, None, 0, cot_prompt, max_new_tokens, device)
        sandwich_text, decoded_cot = _generate_sandwich(
            model, tokenizer, dynamics, tap_layer, cot_prompt, latent_steps, device, answer_tokens
        )
        texts = {"cot": cot_text, "sandwich": sandwich_text}
        for name in names:
            internal[name] += len(tokenizer(texts[name], add_special_tokens=False)["input_ids"])
            if extract_answer(texts[name]) == gold:
                strict[name] += 1
            if extract_answer_lenient(texts[name]) == gold:
                lenient[name] += 1
        if dump_sink is not None:
            dump_sink.write(json.dumps({
                "index": index, "question": row["question"], "gold": gold,
                "cot": cot_text, "sandwich": sandwich_text, "sandwich_decoded_cot": decoded_cot,
            }) + "\n")
            dump_sink.flush()
        if (index + 1) % 10 == 0:
            progress = {name: round(count / (index + 1), 4) for name, count in lenient.items()}
            print(f"eval {index + 1}/{num_problems} lenient={progress} elapsed={time.time() - start:.0f}s", flush=True)

    if dump_sink is not None:
        dump_sink.close()

    count = len(dataset)
    lines = [
        "# Sandwich rollout: discrete encoder | latent recurrence | discrete decoder", "",
        f"Model: `{model_name}`, dynamics: `{dynamics_path}`, problems: {count}, "
        f"latent steps: {latent_steps}", "",
        "| system | strict accuracy | lenient accuracy | mean internal tokens |",
        "| --- | --- | --- | --- |",
    ]
    results = {}
    for name in names:
        results[name] = {
            "strict_accuracy": strict[name] / count,
            "lenient_accuracy": lenient[name] / count,
            "mean_internal_tokens": internal[name] / count,
        }
        lines.append(
            f"| {name} | {strict[name] / count:.4f} | {lenient[name] / count:.4f} | {internal[name] / count:.1f} |"
        )
    report = "\n".join(lines) + "\n"
    print(report, flush=True)
    if output_path is not None:
        Path(output_path).write_text(report, encoding="utf-8")
    return results


@torch.no_grad()
def _generate_complement_fork(
    model,
    tokenizer,
    corrector,
    tap_layer: int,
    basis: torch.Tensor,
    prompt: str,
    max_new_tokens: int,
    device: torch.device,
    max_branches: int = 4,
    fork_z: float = 2.5,
    fork_cooldown: int = 16,
    warmup: int = 16,
    gamma: float = 1.0,
    persist: int = 4,
    child_mode: str = "suppress-dominant",
    steer_mode: str = "closed-loop",
    hull: bool = True,
    hull_clip: float = 0.25,
    record_taps: int = 0,
    tap_snap=None,
) -> list[dict]:
    """Complement-fork rollout: intrusive thoughts play out in offshoots.

    A single greedy latent rollout monitors the tap state's complement
    energy ||h - BB^T h|| against its own running statistics (Welford, as
    in dynamic SC but on the intrusion signal, not ||delta||). When the
    z-score crosses `fork_z`, the branch forks: the *root* has the
    excursion (the above-baseline complement component along its current
    direction) *suppressed* — injected negatively at the tap layer for the
    next `persist` forwards — while the *offshoot* plays the intrusive
    hypothesis out. `child_mode` selects how: "suppress-dominant"
    (default) injects -gamma x the tap state's dominant-subspace component
    BB^T h, so the intrusion drives the offshoot unencumbered by the
    mainline computation; "reinforce" injects +gamma x the excursion
    itself. Both branches stay greedy: decorrelation comes from the latent
    split, not sampling. With no trigger the output is exactly the greedy
    latent rollout (the floor).

    Every branch — offshoots included — runs to its own completion with a
    full `max_new_tokens` post-birth budget before any gate sees it: the
    feedback gate must arbitrate finished testimony, never a truncated
    offshoot. Testimony collected during decode — mean/max corrector delta
    norm (correction pressure), post-birth intrusion excursion count, mean
    chosen-token logprob (trunk confidence), and length — supports many
    offline gate rules from a single decode. Returns a list of branch
    dicts.

    `steer_mode` selects the injection controller. "open-loop" (legacy)
    freezes the fork-time vector and injects it for `persist` forwards.
    "closed-loop" (default) is a proportional controller: every step after
    a branch acquires a role, the injection is recomputed from the
    *currently measured* state — the root suppresses its current
    above-baseline complement excursion, the offshoot suppresses its
    current dominant component (or reinforces its current excursion) — so
    the push decays as the error decays and tracks it while it persists.
    `hull=True` adds two on-manifold constraints: the injected vector is
    clipped to `hull_clip` x the branch's current tap norm, and the hook
    rescales the perturbed hidden state back to its pre-injection norm
    (direction changes, energy does not), keeping offshoots in the shell
    of states the trunk and corrector were trained on. `record_taps=k>0`
    records every k-th tap state per branch (offshoots inherit the parent
    prefix records) under the "taps" key, for training sequence-level
    arbiters.
    """

    encoded = tokenizer(prompt, return_tensors="pt").to(device)
    tokens = encoded["input_ids"]
    state = corrector.initial_state(1, device)
    lm_head = model.get_output_embeddings()
    basis = basis.to(device=device, dtype=torch.float32)
    past = None
    eos_id = tokenizer.eos_token_id

    generated: list[list[int]] = [[]]
    done = [False]
    born_at = [0]
    # Per-branch Welford stats over complement energy, cooldowns, injections.
    stat_count, stat_mean, stat_m2, cooldown = [0], [0.0], [0.0], [0]
    inject_vec: list[torch.Tensor | None] = [None]
    inject_left = [0]
    role = ["none"]
    tap_records: list[list[torch.Tensor]] = [[]]
    # Report-back testimony.
    delta_sum, delta_max, logprob_sum, intrusions, steps_alive = [0.0], [0.0], [0.0], [0], [0]
    fork_count = 0

    holder: dict = {"matrix": None}
    layer_module = model.model.layers[tap_layer - 1]

    def _fork_hook(_module, _inputs, output):
        if holder["matrix"] is None:
            return output
        hidden = output[0] if isinstance(output, tuple) else output
        hidden = hidden.clone()
        matrix = holder["matrix"].to(hidden.dtype)
        if hull:
            last = hidden[:, -1, :]
            norms_before = last.norm(dim=-1, keepdim=True)
            last = last + matrix
            norms_after = last.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            modified = (matrix.norm(dim=-1, keepdim=True) > 0).to(last.dtype)
            scale = 1.0 + modified * (norms_before / norms_after - 1.0)
            hidden[:, -1, :] = last * scale
        else:
            hidden[:, -1, :] += matrix
        if tap_snap is not None:
            mod_mask = matrix.norm(dim=-1) > 0
            if bool(mod_mask.any()):
                last = hidden[:, -1, :]
                snapped = tap_snap(last[mod_mask].float()).to(last.dtype)
                last = last.clone()
                last[mod_mask] = snapped
                hidden[:, -1, :] = last
        if isinstance(output, tuple):
            return (hidden,) + tuple(output[1:])
        return hidden

    hook_handle = layer_module.register_forward_hook(_fork_hook)
    try:
        # Safety cap only: each branch is force-finished at max_new_tokens
        # of its own post-birth steps, so the loop drains naturally.
        for _ in range(max_new_tokens * (max_branches + 1)):
            n_beams = len(generated)
            if any(inject_vec[b] is not None and (steer_mode == "closed-loop" or inject_left[b] > 0) for b in range(n_beams)):
                matrix = torch.zeros(n_beams, basis.size(1), device=device, dtype=torch.float32)
                for b in range(n_beams):
                    if inject_vec[b] is None:
                        continue
                    if steer_mode == "closed-loop":
                        matrix[b] = inject_vec[b]
                        inject_vec[b] = None
                    elif inject_left[b] > 0:
                        matrix[b] = inject_vec[b]
                        inject_left[b] -= 1
                holder["matrix"] = matrix
            else:
                holder["matrix"] = None

            outputs = model(tokens, past_key_values=past, output_hidden_states=True, use_cache=True)
            past = outputs.past_key_values
            h_final = outputs.hidden_states[-1][:, -1, :].float()
            h_tap = outputs.hidden_states[tap_layer][:, -1, :].float()
            delta, state = corrector.step(h_tap, state)
            logits = lm_head((h_final + delta).to(outputs.hidden_states[-1].dtype)).float()
            log_probs = torch.log_softmax(logits, dim=-1)
            greedy_ids = logits.argmax(dim=-1)

            complement = h_tap - (h_tap @ basis.T) @ basis
            energies = complement.norm(dim=-1)
            delta_norms = delta.norm(dim=-1)

            next_ids: list[int] = [0] * n_beams
            forks: list[tuple[int, torch.Tensor]] = []
            for b in range(n_beams):
                if done[b]:
                    next_ids[b] = eos_id if eos_id is not None else 0
                    continue
                value = float(energies[b])
                pre_mean = stat_mean[b]
                std = (stat_m2[b] / max(stat_count[b] - 1, 1)) ** 0.5 if stat_count[b] > 1 else 0.0
                z = (value - stat_mean[b]) / std if std > 1e-9 else 0.0
                triggered = (
                    stat_count[b] >= warmup
                    and cooldown[b] <= 0
                    and len(generated) + len(forks) < max_branches
                    and z >= fork_z
                )
                excursion = None
                if triggered:
                    # Above-baseline complement component, measured against the
                    # pre-update running mean.
                    excursion = complement[b] * ((value - stat_mean[b]) / max(value, 1e-6))
                if stat_count[b] >= warmup and z >= 2.0:
                    intrusions[b] += 1
                stat_count[b] += 1
                diff = value - stat_mean[b]
                stat_mean[b] += diff / stat_count[b]
                stat_m2[b] += diff * (value - stat_mean[b])
                cooldown[b] = max(cooldown[b] - 1, 0)

                next_ids[b] = int(greedy_ids[b])
                delta_sum[b] += float(delta_norms[b])
                delta_max[b] = max(delta_max[b], float(delta_norms[b]))
                logprob_sum[b] += float(log_probs[b, next_ids[b]])
                if record_taps > 0 and steps_alive[b] % record_taps == 0:
                    tap_records[b].append(h_tap[b].detach().to(torch.float16).cpu())
                steps_alive[b] += 1

                if triggered:
                    if child_mode == "suppress-dominant":
                        child_vec = -gamma * (h_tap[b].float() - complement[b])
                    else:
                        child_vec = gamma * excursion
                    root_vec = -excursion
                    if hull:
                        cap = hull_clip * float(h_tap[b].norm())
                        for _name, _v in (("child", child_vec), ("root", root_vec)):
                            norm = float(_v.norm())
                            if norm > cap and norm > 1e-9:
                                _v.mul_(cap / norm)
                    forks.append((b, child_vec))
                    role[b] = "root"
                    inject_vec[b] = root_vec
                    inject_left[b] = persist
                    cooldown[b] = fork_cooldown

                if steer_mode == "closed-loop" and role[b] != "none" and not triggered:
                    # Proportional controller: recompute the push from the
                    # currently measured state; decays as the error decays.
                    vec = None
                    if role[b] == "root":
                        if value > pre_mean and value > 1e-6:
                            vec = -gamma * complement[b] * ((value - pre_mean) / value)
                    elif role[b] == "child-suppdom":
                        vec = -gamma * (h_tap[b].float() - complement[b])
                    elif role[b] == "child-reinforce":
                        if value > pre_mean and value > 1e-6:
                            vec = gamma * complement[b] * ((value - pre_mean) / value)
                    if vec is not None and hull:
                        cap = hull_clip * float(h_tap[b].norm())
                        norm = float(vec.norm())
                        if norm > cap and norm > 1e-9:
                            vec = vec * (cap / norm)
                    inject_vec[b] = vec

            for b, child_vec in forks:
                fork_count += 1
                past = _clone_kv_row(past, b)
                state = _clone_state_row(state, b)
                generated.append(list(generated[b]))
                done.append(False)
                born_at.append(steps_alive[b])
                stat_count.append(stat_count[b])
                stat_mean.append(stat_mean[b])
                stat_m2.append(stat_m2[b])
                cooldown.append(fork_cooldown)
                role.append("child-suppdom" if child_mode == "suppress-dominant" else "child-reinforce")
                tap_records.append(list(tap_records[b]))
                inject_vec.append(child_vec)
                inject_left.append(persist)
                delta_sum.append(0.0)
                delta_max.append(0.0)
                logprob_sum.append(0.0)
                intrusions.append(0)
                steps_alive.append(0)
                next_ids.append(next_ids[b])

            for b in range(len(generated)):
                if not done[b]:
                    generated[b].append(next_ids[b])
                    if (eos_id is not None and next_ids[b] == eos_id) or steps_alive[b] >= max_new_tokens:
                        done[b] = True
            if all(done):
                break
            tokens = torch.tensor([[i] for i in next_ids], device=device)
    finally:
        hook_handle.remove()

    return [
        {
            "text": tokenizer.decode(seq, skip_special_tokens=True),
            "root": b == 0,
            "role": role[b] if role[b] != "none" else ("root" if b == 0 else "child"),
            "born_at": born_at[b],
            "steps": steps_alive[b],
            "mean_delta": delta_sum[b] / max(steps_alive[b], 1),
            "max_delta": delta_max[b],
            "mean_logprob": logprob_sum[b] / max(steps_alive[b], 1),
            "intrusions": intrusions[b],
            "intrusion_rate": intrusions[b] / max(steps_alive[b], 1),
            **({"taps": torch.stack(tap_records[b]) if tap_records[b] else torch.zeros(0, basis.size(1), dtype=torch.float16)} if record_taps > 0 else {}),
        }
        for b, seq in enumerate(generated)
    ]


class BranchArbiter(nn.Module):
    """Bidirectional cross-attention arbiter over forked tap-state sequences.

    All branches' (strided) tap states are projected to a small width,
    tagged with branch and position embeddings, concatenated into one
    sequence, and run through a bidirectional transformer encoder — every
    position attends within its own branch and across sibling branches.
    Mean-pooling per branch feeds a scalar score head: higher = more
    likely correct. ~0.5M parameters at the defaults.
    """

    def __init__(self, d_model: int, d_arb: int = 128, heads: int = 4, layers: int = 2, max_branches: int = 8):
        super().__init__()
        self.proj = nn.Linear(d_model, d_arb)
        self.branch_embed = nn.Embedding(max_branches, d_arb)
        layer = nn.TransformerEncoderLayer(
            d_arb, heads, dim_feedforward=4 * d_arb, batch_first=True, norm_first=True, dropout=0.1,
        )
        self.encoder = nn.TransformerEncoder(layer, layers)
        self.score = nn.Linear(d_arb, 1)
        self.d_arb = d_arb
        self.max_branches = max_branches

    def _positional(self, length: int, device) -> torch.Tensor:
        pos = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(-1)
        freq = torch.exp(
            torch.arange(0, self.d_arb, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / self.d_arb)
        )
        pe = torch.zeros(length, self.d_arb, device=device)
        pe[:, 0::2] = torch.sin(pos * freq)
        pe[:, 1::2] = torch.cos(pos * freq)
        return pe

    def forward(self, branch_taps: list[torch.Tensor]) -> torch.Tensor:
        """branch_taps: per-branch (T_b, d_model) tensors for ONE problem.

        Returns a (n_branches,) tensor of scores (logits).
        """
        device = next(self.parameters()).device
        tokens, owner = [], []
        for i, taps in enumerate(branch_taps):
            if taps.size(0) == 0:
                taps = torch.zeros(1, self.proj.in_features, dtype=taps.dtype)
            x = self.proj(taps.to(device=device, dtype=torch.float32))
            x = x + self._positional(x.size(0), device)
            x = x + self.branch_embed(torch.tensor(min(i, self.max_branches - 1), device=device))
            tokens.append(x)
            owner.extend([i] * x.size(0))
        seq = torch.cat(tokens, dim=0).unsqueeze(0)
        encoded = self.encoder(seq).squeeze(0)
        owner_ids = torch.tensor(owner, device=device)
        scores = []
        for i in range(len(branch_taps)):
            pooled = encoded[owner_ids == i].mean(dim=0)
            scores.append(self.score(pooled).squeeze(-1))
        return torch.stack(scores)


def harvest_fork_traces(
    model_name: str,
    corrector_path: str | Path,
    basis_path: str | Path,
    num_problems: int,
    max_new_tokens: int,
    device_str: str,
    out_path: str | Path,
    stride: int = 8,
    max_branches: int = 4,
    fork_z: float = 2.5,
    fork_cooldown: int = 16,
    gamma: float = 1.0,
    persist: int = 4,
    child_mode: str = "suppress-dominant",
    steer_mode: str = "closed-loop",
    hull: bool = True,
) -> dict:
    """Run complement-fork rollouts over GSM8K *train* problems and save
    arbiter training data: per-branch strided tap sequences with gold
    correctness labels. Keeps only informative problems (>= 2 branches,
    at least one correct AND one wrong branch): at harvest time the
    oracle is free, which is exactly the wall-free supervision the
    inference-time gates lack.
    """

    from datasets import load_dataset

    device = torch.device(device_str if device_str != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model, tokenizer = load_trunk(model_name, device, dtype)
    checkpoint = torch.load(corrector_path, map_location=device, weights_only=True)
    corrector, tap_layer = load_corrector(checkpoint, device)
    basis = torch.load(basis_path, map_location="cpu", weights_only=True)["basis_full"].to(torch.float32)

    dataset = load_dataset("openai/gsm8k", "main", split="train").select(range(num_problems))

    rows: list[dict] = []
    start = time.time()
    for index, row in enumerate(dataset):
        gold = extract_answer(row["answer"])
        cot_prompt = _chat_prompt(tokenizer, COT_PROMPT + row["question"])
        branches = _generate_complement_fork(
            model, tokenizer, corrector, tap_layer, basis, cot_prompt, max_new_tokens, device,
            max_branches=max_branches, fork_z=fork_z, fork_cooldown=fork_cooldown,
            gamma=gamma, persist=persist, child_mode=child_mode,
            steer_mode=steer_mode, hull=hull, record_taps=stride,
        )
        labels = [1 if extract_answer_lenient(branch["text"]) == gold else 0 for branch in branches]
        if len(branches) >= 2 and any(labels) and not all(labels):
            rows.append({
                "index": index,
                "taps": [branch["taps"] for branch in branches],
                "labels": labels,
            })
        if (index + 1) % 25 == 0:
            print(json.dumps({"scanned": index + 1, "kept": len(rows), "elapsed": round(time.time() - start)}), flush=True)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"rows": rows, "config": {
        "model": model_name, "stride": stride, "d_model": int(basis.size(1)),
        "fork_z": fork_z, "child_mode": child_mode, "steer_mode": steer_mode, "hull": hull,
    }}, out)
    summary = {"scanned": len(dataset), "kept": len(rows), "path": str(out)}
    print(json.dumps(summary), flush=True)
    return summary


def train_arbiter(
    traces_path: str | Path,
    out_path: str | Path,
    d_arb: int = 128,
    heads: int = 4,
    layers: int = 2,
    steps: int = 3000,
    lr: float = 1e-4,
    device_str: str = "auto",
    holdout: int = 50,
) -> dict:
    """Train the BranchArbiter on harvested fork traces (BCE per branch,
    gold labels). The last `holdout` problems are kept for validation
    accuracy: fraction of holdout problems where the arbiter's argmax
    branch is correct."""

    device = torch.device(device_str if device_str != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    payload = torch.load(traces_path, map_location="cpu", weights_only=False)
    rows, config = payload["rows"], payload["config"]
    if len(rows) <= holdout + 10:
        holdout = max(len(rows) // 5, 1)
    train_rows, val_rows = rows[:-holdout], rows[-holdout:]

    arbiter = BranchArbiter(config["d_model"], d_arb=d_arb, heads=heads, layers=layers).to(device)
    optimizer = torch.optim.Adam(arbiter.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    generator = torch.Generator().manual_seed(1337)
    start = time.time()
    for step in range(steps):
        row = train_rows[int(torch.randint(len(train_rows), (1,), generator=generator))]
        scores = arbiter([taps for taps in row["taps"]])
        loss = criterion(scores, torch.tensor(row["labels"], device=device, dtype=torch.float32))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 200 == 0 or step == steps - 1:
            with torch.no_grad():
                hits = sum(
                    1 for v in val_rows
                    if v["labels"][int(arbiter([t for t in v["taps"]]).argmax())] == 1
                )
            print(json.dumps({
                "step": step, "loss": round(float(loss), 4),
                "val_argmax_acc": round(hits / max(len(val_rows), 1), 4),
                "elapsed": round(time.time() - start),
            }), flush=True)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": arbiter.state_dict(),
        "config": {"d_model": config["d_model"], "d_arb": d_arb, "heads": heads,
                   "layers": layers, "stride": config["stride"]},
    }, out)
    return {"steps": steps, "train_rows": len(train_rows), "val_rows": len(val_rows), "path": str(out)}


def complement_lens(
    model_name: str,
    basis_path: str | Path,
    dump_path: str | Path,
    num_problems: int,
    device_str: str,
    output_path: str | Path | None,
    tap_layer: int = 12,
    max_examples: int = 40,
) -> dict:
    """Decode the tap stream's Jacobian-split components with the trunk's own upper half.

    Direct test of the intrusive-thoughts claim: if the complement of the
    rank-64 influence subspace carries the trunk's *contending hypotheses*,
    then decoding a complement-only stream should produce structured,
    problem-relevant content — numbers, arithmetic — rather than noise, and
    at numeric positions it should produce *different* numbers than the
    mainline (contending values, not echoes).

    Method: teacher-force each dumped latent completion through the trunk
    once to recover the generation-time tap states (the corrector never
    touches the stream, so replay is exact). Split each completion
    position's tap state h into dominant BB^T h and complement h - BB^T h.
    Build five streams — full (reference and plumbing sanity), dominant,
    complement, norm-matched Gaussian noise (per-position ||complement||,
    the gibberish baseline), and position-shuffled complement (structure
    without position binding) — with prompt positions kept full in all
    variants so the upper half attends to the real problem. Decode each
    with `_upper_half_logits` (verified bit-exact on real taps) and score
    per-position argmax tokens: agreement with the full-stream decode,
    digit rate, digit rate at mainline-digit positions, contending-digit
    rate (digit AND different from mainline's), word rate, and contextual
    number rate (decoded digit strings that appear in the problem text).
    """

    device = torch.device(device_str if device_str != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model, tokenizer = load_trunk(model_name, device, dtype)
    basis = torch.load(basis_path, map_location=device, weights_only=True)["basis_full"].to(torch.float32)

    rows = []
    with Path(dump_path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    rows = rows[:num_problems]

    variant_names = ["full", "dominant", "complement", "complement-parallel", "complement-perp", "random", "shuffled-complement"]
    categories = ["digit", "operator", "word", "space", "other"]
    stats = {name: {"n": 0, "agree": 0, "digit": 0, "digit_at_digit": 0, "contend": 0,
                    "word": 0, "var_digit": 0, "ctx_number": 0, "runnerup": 0, "disagree": 0,
                    "cat": {c: 0 for c in categories}} for name in variant_names}
    digit_positions = 0
    sanity_agree, sanity_total = 0, 0
    examples: list[dict] = []
    generator = torch.Generator(device="cpu").manual_seed(1337)

    def _classify(tok: str) -> str:
        stripped = tok.strip()
        if any(c.isdigit() for c in tok):
            return "digit"
        if not stripped:
            return "space"
        if all(c in "+-*/=%$.,:;()<>^_\\{}[]|!?'\"#&×÷−" for c in stripped):
            return "operator"
        if stripped.isalpha():
            return "word"
        return "other"

    start = time.time()
    for index, row in enumerate(rows):
        completion = row.get("latent", "")
        if not completion.strip():
            continue
        prompt = _chat_prompt(tokenizer, COT_PROMPT + row["question"])
        prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
        completion_ids = tokenizer(completion, add_special_tokens=False, return_tensors="pt")["input_ids"]
        ids = torch.cat([prompt_ids, completion_ids], dim=1).to(device)
        P, T = prompt_ids.size(1), ids.size(1)
        if T - P < 8:
            continue

        with torch.no_grad():
            outputs = model(ids, output_hidden_states=True, use_cache=False)
            h_tap = outputs.hidden_states[tap_layer].float()  # (1, T, d)

        comp_region = h_tap[:, P:, :]
        dominant = (comp_region @ basis.T) @ basis
        complement = comp_region - dominant
        unit_full = comp_region / comp_region.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        complement_parallel = (complement * unit_full).sum(dim=-1, keepdim=True) * unit_full
        complement_perp = complement - complement_parallel
        noise = torch.randn(complement.shape, generator=generator).to(device)
        noise = noise * (complement.norm(dim=-1, keepdim=True) / noise.norm(dim=-1, keepdim=True).clamp_min(1e-6))
        perm = torch.randperm(complement.size(1), generator=generator).to(device)
        variants = {
            "full": comp_region,
            "dominant": dominant,
            "complement": complement,
            "complement-parallel": complement_parallel,
            "complement-perp": complement_perp,
            "random": noise,
            "shuffled-complement": complement[:, perm, :],
        }

        decoded: dict[str, list[int]] = {}
        full_top2: list[int] = []
        with torch.no_grad():
            for name, region in variants.items():
                stream = h_tap.clone()
                stream[:, P:, :] = region
                logits = _upper_half_logits(model, tap_layer, stream.to(dtype))
                decoded[name] = logits[0, P:-1].argmax(dim=-1).tolist()
                if name == "full":
                    full_top2 = logits[0, P:-1].topk(2, dim=-1).indices[:, 1].tolist()

        # Sanity: full-stream decode vs the teacher-forced actual next tokens.
        actual = ids[0, P + 1:].tolist()
        sanity_agree += sum(1 for a, b in zip(decoded["full"], actual) if a == b)
        sanity_total += len(actual)

        full_tokens = [tokenizer.decode([t]) for t in decoded["full"]]
        question = row["question"]
        for name in variant_names:
            tokens = [tokenizer.decode([t]) for t in decoded[name]]
            for pos, (tok, full_tok) in enumerate(zip(tokens, full_tokens)):
                s = stats[name]
                s["n"] += 1
                s["agree"] += tok == full_tok
                if decoded[name][pos] != decoded["full"][pos]:
                    s["disagree"] += 1
                    s["runnerup"] += decoded[name][pos] == full_top2[pos]
                has_digit = any(c.isdigit() for c in tok)
                full_digit = any(c.isdigit() for c in full_tok)
                s["digit"] += has_digit
                if full_digit:
                    if name == "full":
                        digit_positions += 1
                    s["digit_at_digit"] += has_digit
                    s["contend"] += has_digit and tok.strip() != full_tok.strip()
                    s["cat"][_classify(tok)] += 1
                    if (name == "complement" and has_digit and tok.strip() != full_tok.strip()
                            and len(examples) < max_examples):
                        lo = max(pos - 6, 0)
                        examples.append({
                            "problem": index,
                            "context": "".join(full_tokens[lo:pos + 1]),
                            "mainline": full_tok,
                            "complement": tok,
                        })
                stripped = tok.strip()
                s["word"] += stripped.isalpha() and len(stripped) >= 2
                if has_digit:
                    s["var_digit"] += 1
                    runs = re.findall(r"\d+", tok)
                    s["ctx_number"] += any(run in question for run in runs)

        if (index + 1) % 25 == 0:
            print(json.dumps({"lens_problems": index + 1, "elapsed": round(time.time() - start)}), flush=True)

    lines = [
        "# Complement lens: decoding the Jacobian split with the trunk's upper half", "",
        f"Model: `{model_name}`, basis: `{basis_path}`, tap layer: {tap_layer}, dump: `{dump_path}`, "
        f"problems: {len(rows)}",
        f"Plumbing sanity (full-stream decode vs teacher-forced tokens): "
        f"{sanity_agree / max(sanity_total, 1):.4f} argmax agreement", "",
        "| stream | agree w/ full | digit rate | digit @ digit pos | contending digit | word rate | contextual number | runner-up align |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    results = {}
    for name in variant_names:
        s = stats[name]
        n = max(s["n"], 1)
        dd = max(digit_positions, 1)
        vd = max(s["var_digit"], 1)
        dis = max(s["disagree"], 1)
        results[name] = {
            "agree": s["agree"] / n, "digit": s["digit"] / n,
            "digit_at_digit": s["digit_at_digit"] / dd, "contend": s["contend"] / dd,
            "word": s["word"] / n, "ctx_number": s["ctx_number"] / vd,
            "runnerup": s["runnerup"] / dis,
        }
        r = results[name]
        lines.append(
            f"| {name} | {r['agree']:.4f} | {r['digit']:.4f} | {r['digit_at_digit']:.4f} "
            f"| {r['contend']:.4f} | {r['word']:.4f} | {r['ctx_number']:.4f} | {r['runnerup']:.4f} |"
        )
    lines += ["", f"Digit positions (mainline): {digit_positions} of {stats['full']['n']}", "",
              "## Decoded-token categories at mainline-digit positions", "",
              "| stream | digit | operator | word | space | other |",
              "| --- | --- | --- | --- | --- | --- |"]
    for name in variant_names:
        cat = stats[name]["cat"]
        dd = max(digit_positions, 1)
        lines.append("| " + name + " | " + " | ".join(f"{cat[c] / dd:.4f}" for c in categories) + " |")
    lines += ["",
              "## Contending-number examples (complement decode at mainline-digit positions)", ""]
    for ex in examples:
        lines.append(f"- p{ex['problem']} `...{ex['context']}` mainline=`{ex['mainline']}` complement=`{ex['complement']}`")
    report = "\n".join(lines) + "\n"
    print(report, flush=True)
    if output_path is not None:
        Path(output_path).write_text(report, encoding="utf-8")
    return results


def evaluate_complement_fork(
    model_name: str,
    corrector_path: str | Path,
    basis_path: str | Path,
    num_problems: int,
    max_new_tokens: int,
    device_str: str,
    output_path: str | Path | None,
    max_branches: int = 4,
    fork_z: float = 2.5,
    fork_cooldown: int = 16,
    gamma: float = 1.0,
    persist: int = 4,
    child_mode: str = "suppress-dominant",
    steer_mode: str = "closed-loop",
    hull: bool = True,
    arbiter_path: str | Path | None = None,
    tap_snap_path: str | Path | None = None,
) -> dict:
    """Complement-fork eval: decode once, arbitrate many ways.

    Gate rules evaluated offline from each branch's report-back testimony:
    - root: the cleansed primary branch alone (tests suppression by itself);
    - vote: majority lenient answer across branches;
    - agree-else-mindelta: unanimous answer if branches agree, otherwise
      the branch with the lowest mean correction pressure (the report-back
      gate: the offshoot's testimony arbitrated by the corrector);
    - min-delta / min-intrusion / max-logprob: single-signal gates;
    - oracle: correct if any branch is correct (the fork ceiling — whether
      suppress/reinforce splits produce complementary correctness at all).
    A plain greedy latent arm is decoded per problem as the same-run
    reference.
    """

    from collections import Counter

    from datasets import load_dataset

    device = torch.device(device_str if device_str != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model, tokenizer = load_trunk(model_name, device, dtype)
    checkpoint = torch.load(corrector_path, map_location=device, weights_only=True)
    corrector, tap_layer = load_corrector(checkpoint, device)
    basis = torch.load(basis_path, map_location="cpu", weights_only=True)["basis_full"].to(torch.float32)

    arbiter = None
    record_taps = 0
    if arbiter_path is not None:
        arb_ckpt = torch.load(arbiter_path, map_location=device, weights_only=True)
        arb_cfg = arb_ckpt["config"]
        arbiter = BranchArbiter(arb_cfg["d_model"], d_arb=arb_cfg["d_arb"], heads=arb_cfg["heads"], layers=arb_cfg["layers"]).to(device)
        arbiter.load_state_dict(arb_ckpt["state_dict"])
        arbiter.eval()
        record_taps = arb_cfg["stride"]

    tap_snap = None
    if tap_snap_path is not None:
        tap_snap, _ = load_snap(tap_snap_path, device)

    dataset = load_dataset("openai/gsm8k", "main", split="test").select(range(num_problems))

    gate_names = ["latent", "root", "vote", "agree-else-mindelta", "min-delta", "min-intrusion", "max-logprob", "oracle"]
    if arbiter is not None:
        gate_names.insert(7, "arbiter")
    correct = {name: 0 for name in gate_names}
    total_branches = 0
    total_forked = 0
    diverged = 0
    internal_tokens = 0

    dump_sink = None
    if output_path is not None:
        dump_path = Path(output_path).with_suffix(".completions.jsonl")
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        dump_sink = dump_path.open("w", encoding="utf-8")

    start = time.time()
    for index, row in enumerate(dataset):
        gold = extract_answer(row["answer"])
        cot_prompt = _chat_prompt(tokenizer, COT_PROMPT + row["question"])

        latent_text = _generate_with_corrector(model, tokenizer, corrector, tap_layer, cot_prompt, max_new_tokens, device)
        branches = _generate_complement_fork(
            model, tokenizer, corrector, tap_layer, basis, cot_prompt, max_new_tokens, device,
            max_branches=max_branches, fork_z=fork_z, fork_cooldown=fork_cooldown,
            gamma=gamma, persist=persist, child_mode=child_mode,
            steer_mode=steer_mode, hull=hull, record_taps=record_taps, tap_snap=tap_snap,
        )
        branch_taps = [branch.pop("taps", None) for branch in branches]
        answers = [extract_answer_lenient(branch["text"]) for branch in branches]
        answered = [(branch, answer) for branch, answer in zip(branches, answers) if answer is not None]

        total_branches += len(branches)
        internal_tokens += sum(
            len(tokenizer(branch["text"], add_special_tokens=False)["input_ids"]) for branch in branches
        )
        if len(branches) > 1:
            total_forked += 1
            if any(answer != answers[0] for answer in answers[1:]):
                diverged += 1

        def _pick(key, largest=False):
            if not answered:
                return None
            chosen = max(answered, key=lambda pair: pair[0][key]) if largest else min(answered, key=lambda pair: pair[0][key])
            return chosen[1]

        votes = Counter(answer for _, answer in answered)
        unanimous = len(votes) == 1 and answered
        gates = {
            "latent": extract_answer_lenient(latent_text),
            "root": answers[0],
            "vote": votes.most_common(1)[0][0] if votes else None,
            "agree-else-mindelta": (answered[0][1] if unanimous else _pick("mean_delta")),
            "min-delta": _pick("mean_delta"),
            "min-intrusion": _pick("intrusion_rate"),
            "max-logprob": _pick("mean_logprob", largest=True),
            "oracle": gold if any(answer == gold for answer in answers) else None,
        }
        if arbiter is not None:
            with torch.no_grad():
                scores = arbiter([taps for taps in branch_taps])
            ranked = sorted(range(len(branches)), key=lambda i: float(scores[i]), reverse=True)
            gates["arbiter"] = next((answers[i] for i in ranked if answers[i] is not None), None)
        for name in gate_names:
            if gates[name] == gold:
                correct[name] += 1

        if dump_sink is not None:
            dump_sink.write(json.dumps({
                "index": index, "question": row["question"], "gold": gold,
                "latent": latent_text, "branches": branches,
            }) + "\n")
            dump_sink.flush()

        if (index + 1) % 10 == 0:
            progress = {name: round(count / (index + 1), 4) for name, count in correct.items()}
            print(f"cfork {index + 1}/{num_problems} acc={progress} elapsed={time.time() - start:.0f}s", flush=True)

    if dump_sink is not None:
        dump_sink.close()

    count = len(dataset)
    lines = [
        f"# Complement-fork rollout: suppress in root, {child_mode} in offshoot", "",
        f"Model: `{model_name}`, corrector: `{corrector_path}`, basis: `{basis_path}`, problems: {count}",
        f"Fork z: {fork_z}, cap: {max_branches}, cooldown: {fork_cooldown}, gamma: {gamma}, persist: {persist}, "
        f"child mode: {child_mode}, steer mode: {steer_mode}, hull: {hull}"
        + (f", arbiter: `{arbiter_path}`" if arbiter_path else "")
        + (f", tap snap: `{tap_snap_path}`" if tap_snap_path else ""), "",
        f"Mean branches/problem: {total_branches / count:.2f}; problems forked: {total_forked}/{count}; "
        f"forked problems where an offshoot answered differently: {diverged}/{max(total_forked, 1)}; "
        f"mean internal tokens: {internal_tokens / count:.1f}", "",
        "| gate | accuracy (lenient) |",
        "| --- | --- |",
    ]
    results = {}
    for name in gate_names:
        results[name] = correct[name] / count
        lines.append(f"| {name} | {correct[name] / count:.4f} |")
    report = "\n".join(lines) + "\n"
    print(report, flush=True)
    if output_path is not None:
        Path(output_path).write_text(report, encoding="utf-8")
    return results


def evaluate_retrofit(
    model_name: str,
    corrector_path: str | Path | None,
    num_problems: int,
    max_new_tokens: int,
    device_str: str,
    output_path: str | Path | None,
    latent_samples: int = 1,
    temperature: float = 0.7,
    stop_agreement: int = 0,
    problem_offset: int = 0,
    sequential_rollouts: bool = False,
    dataset_name: str = "gsm8k",
    quantize: bool = False,
    quorum: int = 1,
    quorum_noise: float = 0.0,
    quorum_agg: str = "mean",
    quorum_agree: float = 1.0,
    quorum_correctors: list[str] | None = None,
    dynamic_sc: int = 0,
    branch_z: float = 2.5,
    branch_cooldown: int = 16,
    seed: int = 0,
) -> dict:
    """Compare direct / cot / latent-corrected systems on a held-out test set.

    dataset_name="gsm8k" (default) uses the GSM8K test split; "math" uses
    the MATH-lighteval test split restricted to numeric-\\boxed{} problems
    (the same filter as the harvest side), scored through the boxed
    fallback in extract_answer. quantize=True loads the trunk in 4-bit NF4
    (local path for 32B-class trunks).

    latent_samples > 1 adds a latent self-consistency system: k sampled
    internal rollouts with majority vote over lenient answers, still
    surfacing only the final answer span.

    stop_agreement > 0 makes the vote adaptive: rollouts are sampled one at
    a time and sampling stops as soon as any answer has stop_agreement
    votes (latent_samples remains the hard cap). This is the stop-on-
    agreement policy validated by rollback-simulate, run as a real decode
    mode instead of a post-hoc simulation.

    problem_offset evaluates the slice [offset, offset + num_problems) of
    the test split (disjoint-slice validation). Rollouts are batched by
    default (waves of stop_agreement then 2 when adaptive); pass
    sequential_rollouts=True to restore one-at-a-time sampling.

    quorum > 1 (or extra quorum_correctors) wraps the corrector in a
    QuorumCorrector: k members vote on the delta at every decode step —
    replicas reading noise-perturbed taps (quorum_noise, relative to the
    tap's per-sequence std) or independently trained checkpoints
    (quorum_correctors), aggregated by quorum_agg (mean/median/sign with
    sign-agreement threshold quorum_agree). SC at the corrector instead of
    the rollout: ~free at decode time versus k times rollout cost.

    dynamic_sc > 1 adds a dynamic self-consistency system: a single greedy
    rollout that branches into sampled siblings (up to dynamic_sc beams)
    only when the corrector's correction magnitude z-scores past branch_z
    against the beam's own history — SC@k with k chosen per problem, and
    per site, by the error monitor (see _generate_dynamic_sc).
    """

    from datasets import load_dataset

    device = torch.device(device_str if device_str != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model, tokenizer = load_trunk(model_name, device, dtype, quantize=quantize)
    if seed:
        torch.manual_seed(seed)

    corrector = None
    tap_layer = 0
    if corrector_path is not None:
        checkpoint = torch.load(corrector_path, map_location=device, weights_only=True)
        corrector, tap_layer = load_corrector(checkpoint, device)
        extra_members = []
        for member_path in quorum_correctors or []:
            member, member_tap = load_corrector(
                torch.load(member_path, map_location=device, weights_only=True), device
            )
            if member_tap != tap_layer:
                raise ValueError(f"quorum member {member_path} taps layer {member_tap}, expected {tap_layer}")
            extra_members.append(member)
        if extra_members or quorum > 1:
            members = [corrector] + extra_members
            if len(members) == 1:
                if quorum_noise <= 0:
                    raise ValueError("replica quorum (single checkpoint) requires quorum_noise > 0")
                members = members * quorum  # shared weights; noise + per-member state give diversity
            corrector = QuorumCorrector(members, noise=quorum_noise, agg=quorum_agg, agree=quorum_agree)
            print(json.dumps({"quorum_members": len(members), "quorum_noise": quorum_noise,
                              "quorum_agg": quorum_agg, "quorum_agree": quorum_agree}), flush=True)

    if dataset_name == "math":
        from datasets import Dataset

        raw = load_dataset("DigitalLearningGmbH/MATH-lighteval", "default", split="test")
        rows_all = [
            {"question": row["problem"], "answer": f"#### {gold}"}
            for row in raw
            if (gold := extract_answer(row["solution"])) is not None
        ]
        dataset = Dataset.from_list(rows_all).select(range(problem_offset, problem_offset + num_problems))
        print(json.dumps({"math_numeric_test_problems": len(rows_all)}), flush=True)
    else:
        dataset = load_dataset("openai/gsm8k", "main", split="test").select(
            range(problem_offset, problem_offset + num_problems)
        )
    names = ["direct", "cot"] + (["latent"] if corrector is not None else [])
    if corrector is not None and latent_samples > 1:
        names.append(
            f"latent_asc{stop_agreement}of{latent_samples}" if stop_agreement > 0 else f"latent_sc{latent_samples}"
        )
    dsc_name = f"latent_dsc{dynamic_sc}" if corrector is not None and dynamic_sc > 1 else None
    latent_sc_name = names[-1] if corrector is not None and latent_samples > 1 else f"latent_sc{latent_samples}"
    if dsc_name is not None:
        names.append(dsc_name)
    vote_names = {latent_sc_name} | ({dsc_name} if dsc_name is not None else set())
    strict = {name: 0 for name in names}
    lenient = {name: 0 for name in names}
    emitted = {name: 0 for name in names}
    internal = {name: 0 for name in names}
    rollouts_used = 0
    dsc_rollouts_used = 0

    from collections import Counter


    def score(texts: dict, gold) -> None:
        nonlocal rollouts_used, dsc_rollouts_used
        for name in names:
            if name not in texts:
                continue
            text = texts[name]
            if name in vote_names:
                rollouts = json.loads(text)
                if name == dsc_name:
                    dsc_rollouts_used += len(rollouts)
                else:
                    rollouts_used += len(rollouts)
                for rollout in rollouts:
                    internal[name] += len(tokenizer(rollout, add_special_tokens=False)["input_ids"])
                votes = Counter(
                    answer for answer in (extract_answer_lenient(t) for t in rollouts) if answer is not None
                )
                majority = votes.most_common(1)[0][0] if votes else None
                emitted[name] += (
                    len(tokenizer(f"#### {majority}", add_special_tokens=False)["input_ids"]) if majority is not None else 0
                )
                if majority == gold:
                    strict[name] += 1
                    lenient[name] += 1
                continue
            tokens = len(tokenizer(text, add_special_tokens=False)["input_ids"])
            internal[name] += tokens
            if name == "latent":
                # By design only the final '#### <answer>' span is surfaced; the chain stays internal.
                answer = extract_answer(text)
                emitted[name] += (
                    len(tokenizer(f"#### {answer}", add_special_tokens=False)["input_ids"]) if answer is not None else 0
                )
            else:
                emitted[name] += tokens
            if extract_answer(text) == gold:
                strict[name] += 1
            if extract_answer_lenient(text) == gold:
                lenient[name] += 1

    # Resume: replay any completed problems from a previous crashed run's dump.
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
                    continue  # partial line from a crash mid-write; recompute it
                if row["index"] in done_indices:
                    continue
                score(row, row["gold"])
                done_indices.add(row["index"])
            if done_indices:
                print(json.dumps({"resumed_problems": len(done_indices)}), flush=True)
        dump_sink = dump_path.open("a", encoding="utf-8")

    start = time.time()
    for local_index, row in enumerate(dataset):
        index = problem_offset + local_index
        if index in done_indices:
            continue
        gold = extract_answer(row["answer"])
        direct_prompt = _chat_prompt(tokenizer, DIRECT_PROMPT + row["question"])
        cot_prompt = _chat_prompt(tokenizer, COT_PROMPT + row["question"])

        texts = {}
        texts["direct"] = _generate_with_corrector(model, tokenizer, None, 0, direct_prompt, 32, device)
        texts["cot"] = _generate_with_corrector(model, tokenizer, None, 0, cot_prompt, max_new_tokens, device)
        if corrector is not None:
            texts["latent"] = _generate_with_corrector(
                model, tokenizer, corrector, tap_layer, cot_prompt, max_new_tokens, device
            )

        latent_sc_rollouts = None
        if corrector is not None and latent_samples > 1:
            latent_sc_rollouts = []
            running_votes: Counter = Counter()
            if sequential_rollouts:
                for _ in range(latent_samples):
                    rollout = _generate_with_corrector(
                        model, tokenizer, corrector, tap_layer, cot_prompt, max_new_tokens, device,
                        temperature=temperature,
                    )
                    latent_sc_rollouts.append(rollout)
                    if stop_agreement > 0:
                        answer = extract_answer_lenient(rollout)
                        if answer is not None:
                            running_votes[answer] += 1
                            if running_votes[answer] >= stop_agreement:
                                break
            else:
                # Batched (roadmap 4b): full vote in one batch, or waves when adaptive.
                stopped = False
                while len(latent_sc_rollouts) < latent_samples and not stopped:
                    if stop_agreement > 0:
                        wave = max(stop_agreement, 2) if not latent_sc_rollouts else 2
                    else:
                        wave = latent_samples
                    wave = min(wave, latent_samples - len(latent_sc_rollouts))
                    rollouts = _generate_batch_with_corrector(
                        model, tokenizer, corrector, tap_layer, cot_prompt, max_new_tokens, device,
                        temperature=temperature, batch_size=wave,
                    )
                    latent_sc_rollouts.extend(rollouts)
                    if stop_agreement > 0:
                        for rollout in rollouts:
                            answer = extract_answer_lenient(rollout)
                            if answer is not None:
                                running_votes[answer] += 1
                                if running_votes[answer] >= stop_agreement:
                                    stopped = True
                    else:
                        break
            texts[latent_sc_name] = json.dumps(latent_sc_rollouts)

        if dsc_name is not None:
            dsc_beams = _generate_dynamic_sc(
                model, tokenizer, corrector, tap_layer, cot_prompt, max_new_tokens, device,
                temperature=temperature, max_rollouts=dynamic_sc,
                branch_z=branch_z, branch_cooldown=branch_cooldown,
            )
            texts[dsc_name] = json.dumps(dsc_beams)

        score(texts, gold)

        if dump_sink is not None:
            dump_sink.write(
                json.dumps({"index": index, "question": row["question"], "gold": gold, **texts}) + "\n"
            )
            dump_sink.flush()

        if (local_index + 1) % 10 == 0:
            progress = {name: round(count / (local_index + 1), 4) for name, count in lenient.items()}
            print(f"eval {local_index + 1}/{num_problems} lenient={progress} elapsed={time.time() - start:.0f}s", flush=True)

    if dump_sink is not None:
        dump_sink.close()

    count = len(dataset)
    lines = [
        f"# {'MATH' if dataset_name == 'math' else 'GSM8K'} retrofit comparison", "",
        f"Model: `{model_name}`, problems: {count}" + (f", seed: {seed}" if seed else ""), "",
        "| system | strict accuracy | lenient accuracy | mean emitted tokens | mean internal rollout tokens |",
        "| --- | --- | --- | --- | --- |",
    ]
    results = {}
    for name in names:
        results[name] = {
            "strict_accuracy": strict[name] / count,
            "lenient_accuracy": lenient[name] / count,
            "mean_emitted_tokens": emitted[name] / count,
            "mean_internal_tokens": internal[name] / count,
        }
        lines.append(
            f"| {name} | {strict[name] / count:.4f} | {lenient[name] / count:.4f} "
            f"| {emitted[name] / count:.1f} | {internal[name] / count:.1f} |"
        )
    if corrector is not None and latent_samples > 1:
        mean_rollouts = rollouts_used / count
        results[latent_sc_name]["mean_rollouts"] = mean_rollouts
        lines += ["", f"Mean latent rollouts per problem: {mean_rollouts:.2f} "
                  f"(cap {latent_samples}" + (f", stop at {stop_agreement} agreeing)" if stop_agreement > 0 else ")")]
    if dsc_name is not None:
        mean_dsc = dsc_rollouts_used / count
        results[dsc_name]["mean_rollouts"] = mean_dsc
        lines += ["", f"Mean dynamic-SC beams per problem: {mean_dsc:.2f} "
                  f"(cap {dynamic_sc}, branch z {branch_z}, cooldown {branch_cooldown})"]
    if isinstance(corrector, QuorumCorrector):
        lines += ["", f"Corrector quorum: {len(corrector.members)} members, noise {corrector.noise}, "
                  f"agg {corrector.agg}" + (f" (agree >= {corrector.agree})" if corrector.agg == "sign" else "")]
    report = "\n".join(lines) + "\n"
    print(report, flush=True)
    if output_path is not None:
        Path(output_path).write_text(report, encoding="utf-8")
    return results


def _trunk_flops_per_token(config, context_len: int) -> float:
    """Forward FLOPs for one decode token at the given KV-context length.

    Exact multiply-accumulate accounting (2 FLOPs per MAC) over the trunk's
    linear maps — QKV/output projections (GQA-aware), gated MLP, lm_head —
    plus the context-dependent attention score/value term. Layernorms,
    activations, and rotary embeddings are omitted (sub-percent).
    """

    d = config.hidden_size
    layers = config.num_hidden_layers
    heads = config.num_attention_heads
    kv_heads = getattr(config, "num_key_value_heads", heads) or heads
    head_dim = getattr(config, "head_dim", None) or d // heads
    d_ff = config.intermediate_size
    vocab = config.vocab_size

    q_proj = d * heads * head_dim
    kv_proj = 2 * d * kv_heads * head_dim
    o_proj = heads * head_dim * d
    mlp = 3 * d * d_ff  # gate + up + down
    per_layer_weights = 2 * (q_proj + kv_proj + o_proj + mlp)
    attention = 2 * (2 * heads * head_dim * context_len)  # scores + weighted values
    head = 2 * d * vocab
    return layers * (per_layer_weights + attention) + head


def _corrector_flops_per_token(corrector) -> float:
    """Forward FLOPs for one corrector step (2 FLOPs per MAC over its linears)."""

    total = 0
    for module in corrector.modules():
        if isinstance(module, nn.Linear):
            total += 2 * module.in_features * module.out_features
    return total


def flops_report(
    model_name: str,
    dump_path: str | Path,
    output_path: str | Path | None,
    corrector_path: str | Path | None = None,
    d_cfc: int = 512,
) -> dict:
    """Direct FLOPs evaluation of the sidecar against its token savings.

    Reads a *.completions.jsonl eval dump, measures per-system token counts
    (prompt + generated, per rollout), and prices every system in FLOPs:
    trunk decode FLOPs at the measured mean context length, plus the
    corrector's per-token overhead for latent systems. Addresses the
    concern that the sidecar's added per-token cost could negate the
    emitted-token reductions — the comparison is end-to-end compute per
    problem, not token counts.

    Only the model config and corrector shapes are needed (no GPU): FLOPs
    come from exact matmul accounting and the dump supplies the measured
    token statistics.
    """

    from transformers import AutoConfig, AutoTokenizer

    config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if corrector_path is not None:
        checkpoint = torch.load(corrector_path, map_location="cpu", weights_only=True)
        corrector, _ = load_corrector(checkpoint, torch.device("cpu"))
    else:
        corrector = HiddenDeltaCorrector(d_model=config.hidden_size, d_cfc=d_cfc)
    corrector_flops = _corrector_flops_per_token(corrector)

    rows = []
    for line in Path(dump_path).read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    if not rows:
        raise ValueError(f"No rows found in {dump_path}")

    def token_len(text: str) -> int:
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])

    skip_keys = {"index", "question", "gold"}
    system_names = [key for key in rows[0] if key not in skip_keys]
    prompt_tokens = sum(token_len(COT_PROMPT + row["question"]) for row in rows) / len(rows)

    stats: dict[str, dict] = {}
    for name in system_names:
        generated, rollouts = 0.0, 0.0
        for row in rows:
            text = row.get(name)
            if text is None:
                continue
            samples = json.loads(text) if name.startswith("latent_sc") or name.startswith("latent_asc") else [text]
            rollouts += len(samples)
            generated += sum(token_len(sample) for sample in samples)
        mean_rollouts = rollouts / len(rows)
        mean_generated = generated / max(rollouts, 1)  # per rollout
        # Mean KV-context length over a decode of T tokens from prompt P: P + T/2.
        mean_context = prompt_tokens + mean_generated / 2
        trunk = _trunk_flops_per_token(config, int(mean_context))
        uses_corrector = name.startswith("latent")
        per_token = trunk + (corrector_flops if uses_corrector else 0)
        stats[name] = {
            "mean_rollouts": mean_rollouts,
            "mean_generated_tokens_per_rollout": mean_generated,
            "trunk_flops_per_token": trunk,
            "corrector_overhead_per_token": corrector_flops if uses_corrector else 0,
            "overhead_fraction": corrector_flops / trunk if uses_corrector else 0.0,
            "total_flops_per_problem": per_token * mean_generated * mean_rollouts,
        }

    baseline = stats.get("cot")
    lines = [
        "# Direct FLOPs evaluation: sidecar overhead vs token savings", "",
        f"Model: `{model_name}`, dump: `{dump_path}`, problems: {len(rows)}",
        f"Corrector forward: {corrector_flops / 1e6:.2f} MFLOPs/token "
        f"({corrector_flops / stats[system_names[0]]['trunk_flops_per_token']:.2%} of trunk decode)", "",
        "| system | rollouts | gen tokens/rollout | trunk GFLOPs/tok | sidecar overhead | total TFLOPs/problem | vs cot |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, entry in stats.items():
        relative = entry["total_flops_per_problem"] / baseline["total_flops_per_problem"] if baseline else float("nan")
        lines.append(
            f"| {name} | {entry['mean_rollouts']:.2f} | {entry['mean_generated_tokens_per_rollout']:.1f} "
            f"| {entry['trunk_flops_per_token'] / 1e9:.2f} | {entry['overhead_fraction']:.3%} "
            f"| {entry['total_flops_per_problem'] / 1e12:.3f} | {relative:.3f}x |"
        )
    report = "\n".join(lines) + "\n"
    print(report, flush=True)
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(report, encoding="utf-8")
    return stats


def _staged_upper_forward(model, input_ids: torch.Tensor, tap_layer: int):
    """Re-run the trunk with the tap state as a grad-enabled autograd leaf.

    Replays the decoder stack manually (rotary embeddings + causal sdpa via
    attention_mask=None), inserting a `.detach().requires_grad_(True)` leaf
    exactly where `hidden_states[tap_layer]` lives: before block `tap_layer`
    for mid-stack taps, after the final norm for tap_layer == num_layers.

    Returns (h_tap_leaf, h_final_postnorm). Comparing these against the
    standard forward's hidden_states[tap_layer] / hidden_states[-1] is the
    tap-authenticity check: exact agreement proves the tapped tensor is the
    genuine residual stream the upper trunk consumes, and the leaf gives
    autograd access to J = d h_final / d h_tap.
    """

    core = model.model
    hidden = core.embed_tokens(input_ids)
    position_ids = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0)
    position_embeddings = core.rotary_emb(hidden, position_ids)
    h_tap = None
    for index, layer in enumerate(core.layers):
        if index == tap_layer:
            h_tap = hidden.detach().requires_grad_(True)
            hidden = h_tap
        out = layer(
            hidden,
            attention_mask=None,  # sdpa falls back to is_causal=True
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            use_cache=False,
        )
        hidden = out[0] if isinstance(out, tuple) else out
    hidden = core.norm(hidden)
    if h_tap is None:  # tap_layer == num_layers: the post-norm final state
        h_tap = hidden.detach().requires_grad_(True)
        hidden = h_tap
    return h_tap, hidden


def jspace_verify(
    model_name: str,
    traces_path: str | Path,
    output_path: str | Path | None,
    corrector_path: str | Path | None = None,
    tap_layer: int | None = None,
    device_str: str = "auto",
    num_traces: int = 8,
    positions_per_trace: int = 8,
    directions: int = 4,
    rank: int = 64,
    max_seq_len: int = 640,
    quantize: bool = False,
    basis_out: str | Path | None = None,
) -> dict:
    """Verify the residual-stream tap with a Jacobian projection analysis.

    Four questions, in order of increasing strength:

    1. **Tap authenticity.** Does hidden_states[tap_layer] fed through
       layers[tap:] + norm reproduce the trunk's own h_final and logits?
       (If not, the "tap" is not the stream the trunk consumes.)
    2. **Influence subspace.** Sample the row space of the local Jacobian
       block J_t = d h_final_t / d h_tap_t via vector-Jacobian products with
       random unit output directions; SVD the collected rows. Reports the
       spectrum's effective rank (participation ratio), the energy captured
       at `rank`, and how much influence flows through *other* positions
       (attention-mediated, off the local block).
    3. **Corrector alignment.** Sample the corrector's read directions
       d <v, delta_t> / d h_tap_t the same way and measure the fraction of
       each direction's energy inside the top-`rank` influence subspace,
       against the rank/d random-subspace baseline. Low alignment means the
       corrector's read directions live largely in the *complement* of the
       trunk's dominant Jacobian subspace.
    4. **Projection ablation (the functional test).** Feed the corrector
       keep-only / remove-only projections of its input (top-`rank` Jacobian
       basis, local and full, plus a random-basis control) and measure how
       much of its function survives: delta cosine to baseline, delta norm
       ratio, and corrected-token agreement at positions where the
       correction actually changes the argmax token. If function survives
       `remove` but not `keep`, the corrector monitors the orthogonal
       complement of the dominant influence subspace, and a concept-aligned
       (Jacobian-isolated) interface would discard its signal.
    """

    device = torch.device(device_str if device_str != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model, tokenizer = load_trunk(model_name, device, dtype, quantize=quantize)

    corrector = None
    if corrector_path is not None:
        checkpoint = torch.load(corrector_path, map_location=device, weights_only=True)
        corrector, tap_layer = load_corrector(checkpoint, device)
    if tap_layer is None:
        raise ValueError("Provide --corrector or --tap-layer")

    traces = [json.loads(line) for line in Path(traces_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not traces:
        raise ValueError(f"No traces found in {traces_path}")
    generator = torch.Generator().manual_seed(1337)

    recon_final_err: list[float] = []
    recon_tap_err: list[float] = []
    argmax_agree = 0
    argmax_total = 0
    influence_rows: list[torch.Tensor] = []
    offdiag_influence_rows: list[torch.Tensor] = []
    corrector_rows: list[torch.Tensor] = []
    tap_states: list[torch.Tensor] = []
    cross_mass: list[float] = []
    cached_traces: list[tuple[torch.Tensor, torch.Tensor, int, int]] = []
    start = time.time()

    for trace_index in range(min(num_traces, len(traces))):
        trace = traces[trace_index]
        prompt_ids = tokenizer(trace["prompt"], add_special_tokens=False)["input_ids"]
        full_ids = prompt_ids + tokenizer(trace["completion"], add_special_tokens=False)["input_ids"]
        full_ids = full_ids[:max_seq_len]
        if len(full_ids) <= len(prompt_ids) + 4:
            continue
        batch = torch.tensor([full_ids], device=device)

        # Reference forward: what the eval/training code actually taps.
        with torch.no_grad():
            reference = model(batch, output_hidden_states=True, use_cache=False)
        h_tap_ref = reference.hidden_states[tap_layer][0].float()
        h_final_ref = reference.hidden_states[-1][0].float()
        if corrector is not None:
            cached_traces.append((h_tap_ref.cpu(), h_final_ref.cpu(), len(prompt_ids), len(full_ids)))

        # Staged forward with the tap as an autograd leaf.
        h_tap_leaf, h_final_staged = _staged_upper_forward(model, batch, tap_layer)

        # 1. Tap authenticity.
        tap_err = (h_tap_leaf[0].float() - h_tap_ref).norm() / h_tap_ref.norm()
        final_err = (h_final_staged[0].float() - h_final_ref).norm() / h_final_ref.norm()
        recon_tap_err.append(tap_err.item())
        recon_final_err.append(final_err.item())
        with torch.no_grad():
            logits_staged = model.lm_head(h_final_staged[0])
            logits_ref = model.lm_head(reference.hidden_states[-1][0])
        argmax_agree += int((logits_staged.argmax(-1) == logits_ref.argmax(-1)).sum())
        argmax_total += logits_ref.size(0)

        # Sample completion positions (predicting completion tokens).
        low, high = len(prompt_ids), len(full_ids) - 1
        positions = sorted(
            {int(low + torch.randint(high - low, (1,), generator=generator)) for _ in range(positions_per_trace)}
        )
        d_model = h_tap_ref.size(-1)

        # 2. Influence-subspace rows: J^T u for random unit u at each position.
        for t in positions:
            tap_states.append(h_tap_ref[t].cpu())
            for _ in range(directions):
                u = torch.randn(d_model, generator=generator).to(device=device)
                u = u / u.norm()
                scalar = (h_final_staged[0, t, :].float() * u).sum()
                (grad,) = torch.autograd.grad(scalar, h_tap_leaf, retain_graph=True)
                grad = grad[0].float()
                local = grad[t]
                total_sq = float((grad.norm() ** 2))
                cross_mass.append(1.0 - float(local.norm() ** 2) / max(total_sq, 1e-30))
                influence_rows.append(local.cpu())
                # Attention-mediated rows: the earlier tap positions whose
                # states most influence h_final at t (via keys/values).
                other_norms = grad[:t].norm(dim=1) if t > 0 else grad[:0].norm(dim=1)
                for t_other in other_norms.topk(min(3, other_norms.numel())).indices.tolist():
                    offdiag_influence_rows.append(grad[t_other].cpu())

        # 3. Corrector read directions through the recurrent chain.
        if corrector is not None:
            state = corrector.initial_state(1, device)
            h_seq = h_tap_leaf.float()
            deltas = []
            for t in range(len(full_ids)):
                delta, state = corrector.step(h_seq[:, t, :], state)
                deltas.append(delta)
            for t in positions:
                for _ in range(directions):
                    v = torch.randn(d_model, generator=generator).to(device=device)
                    v = v / v.norm()
                    scalar = (deltas[t][0] * v).sum()
                    (grad,) = torch.autograd.grad(scalar, h_tap_leaf, retain_graph=True)
                    corrector_rows.append(grad[0, t].float().cpu())

        del h_tap_leaf, h_final_staged
        print(
            f"jspace-verify trace {trace_index + 1}/{min(num_traces, len(traces))} "
            f"positions={len(positions)} elapsed={time.time() - start:.0f}s",
            flush=True,
        )

    # SVD of the influence rows -> functional J-space bases (local block only,
    # and full = local + attention-mediated rows).
    def fit_basis(rows: list[torch.Tensor], basis_rank: int):
        stacked = torch.stack(rows)
        stacked = stacked / stacked.norm(dim=1, keepdim=True).clamp_min(1e-30)
        _, spectrum, v_rows = torch.linalg.svd(stacked, full_matrices=False)
        energy = spectrum**2
        eff_rank = float(energy.sum() ** 2 / (energy**2).sum())
        basis_rank = min(basis_rank, v_rows.size(0))
        captured = float(energy[:basis_rank].sum() / energy.sum())
        return v_rows[:basis_rank], eff_rank, captured

    basis, effective_rank, energy_at_rank = fit_basis(influence_rows, rank)
    basis_full, effective_rank_full, energy_at_rank_full = fit_basis(
        influence_rows + offdiag_influence_rows, rank
    )
    rank = basis.size(0)
    d_model = basis.size(1)
    baseline = rank / d_model

    def projection_fraction(rows: list[torch.Tensor], target: torch.Tensor) -> float:
        stacked = torch.stack(rows)
        stacked = stacked / stacked.norm(dim=1, keepdim=True).clamp_min(1e-30)
        return float(((stacked @ target.T) ** 2).sum(dim=1).mean())

    tap_fraction = projection_fraction(tap_states, basis)
    tap_fraction_full = projection_fraction(tap_states, basis_full)
    corrector_fraction = projection_fraction(corrector_rows, basis) if corrector_rows else None
    corrector_fraction_full = projection_fraction(corrector_rows, basis_full) if corrector_rows else None

    if basis_out is not None:
        Path(basis_out).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"basis_local": basis, "basis_full": basis_full, "rank": rank,
             "tap_layer": tap_layer, "model": model_name},
            basis_out,
        )

    # 4. Projection ablation: what survives when the corrector's *input* is
    # restricted to the top-rank influence subspace (keep) or its orthogonal
    # complement (remove)? The trunk stream itself is untouched.
    projection_ablation: dict[str, dict[str, float]] | None = None
    if corrector is not None and cached_traces:
        rand_basis = torch.linalg.qr(torch.randn(d_model, rank, generator=generator)).Q.T
        variants: dict[str, tuple[torch.Tensor, str]] = {
            "keep-local": (basis, "keep"), "remove-local": (basis, "remove"),
            "keep-full": (basis_full, "keep"), "remove-full": (basis_full, "remove"),
            "keep-random": (rand_basis, "keep"), "remove-random": (rand_basis, "remove"),
        }

        def run_deltas(h_tap: torch.Tensor) -> torch.Tensor:
            state = corrector.initial_state(1, device)
            out = []
            with torch.no_grad():
                for t in range(h_tap.size(0)):
                    delta, state = corrector.step(h_tap[t : t + 1].to(device), state)
                    out.append(delta[0].float().cpu())
            return torch.stack(out)

        agg = {name: {"energy": [], "cos": [], "norm": [], "agree": 0, "active": 0} for name in variants}
        lm_dtype = model.lm_head.weight.dtype
        for h_tap_c, h_final_c, p_len, s_len in cached_traces:
            span = slice(p_len, s_len)
            base_deltas = run_deltas(h_tap_c)
            with torch.no_grad():
                plain_tok = model.lm_head(h_final_c[span].to(device=device, dtype=lm_dtype)).argmax(-1).cpu()
                base_tok = model.lm_head(
                    (h_final_c[span] + base_deltas[span]).to(device=device, dtype=lm_dtype)
                ).argmax(-1).cpu()
            active = base_tok != plain_tok
            for name, (b, mode) in variants.items():
                proj = h_tap_c @ b.T @ b
                h_variant = proj if mode == "keep" else h_tap_c - proj
                agg[name]["energy"].extend(
                    ((h_variant[span].norm(dim=1) / h_tap_c[span].norm(dim=1).clamp_min(1e-30)) ** 2).tolist()
                )
                deltas_v = run_deltas(h_variant)
                b_span, v_span = base_deltas[span], deltas_v[span]
                denom = (b_span.norm(dim=1) * v_span.norm(dim=1)).clamp_min(1e-12)
                agg[name]["cos"].extend(((b_span * v_span).sum(dim=1) / denom).tolist())
                agg[name]["norm"].extend((v_span.norm(dim=1) / b_span.norm(dim=1).clamp_min(1e-12)).tolist())
                with torch.no_grad():
                    v_tok = model.lm_head(
                        (h_final_c[span] + v_span).to(device=device, dtype=lm_dtype)
                    ).argmax(-1).cpu()
                agg[name]["agree"] += int((v_tok[active] == base_tok[active]).sum())
                agg[name]["active"] += int(active.sum())
        projection_ablation = {
            name: {
                "tap_energy_kept": sum(s["energy"]) / len(s["energy"]),
                "delta_cos": sum(s["cos"]) / len(s["cos"]),
                "delta_norm_ratio": sum(s["norm"]) / len(s["norm"]),
                "active_token_agreement": s["agree"] / max(s["active"], 1),
            }
            for name, s in agg.items()
        }

    results = {
        "tap_layer": tap_layer,
        "reconstruction": {
            "tap_rel_error_max": max(recon_tap_err),
            "final_rel_error_max": max(recon_final_err),
            "logit_argmax_agreement": argmax_agree / max(argmax_total, 1),
        },
        "influence_subspace": {
            "samples": len(influence_rows),
            "offdiag_samples": len(offdiag_influence_rows),
            "effective_rank": effective_rank,
            "effective_rank_full": effective_rank_full,
            "rank": rank,
            "energy_at_rank": energy_at_rank,
            "energy_at_rank_full": energy_at_rank_full,
            "cross_position_influence_mass": sum(cross_mass) / len(cross_mass),
        },
        "alignment": {
            "baseline_random": baseline,
            "tap_state_fraction_in_jspace": tap_fraction,
            "tap_state_fraction_in_jspace_full": tap_fraction_full,
            "corrector_read_fraction_in_jspace": corrector_fraction,
            "corrector_read_fraction_in_jspace_full": corrector_fraction_full,
        },
        "projection_ablation": projection_ablation,
    }
    lines = [
        "# Residual-stream tap verification (Jacobian projection)", "",
        f"Model: `{model_name}`, tap layer: {tap_layer}, traces: {min(num_traces, len(traces))}, "
        f"positions/trace: {positions_per_trace}, directions: {directions}", "",
        "## 1. Tap authenticity (staged re-run of layers[tap:])", "",
        f"- max relative error, tap state vs reference: {max(recon_tap_err):.2e}",
        f"- max relative error, reconstructed h_final vs reference: {max(recon_final_err):.2e}",
        f"- logit argmax agreement: {argmax_agree}/{argmax_total} ({argmax_agree / max(argmax_total, 1):.2%})", "",
        "## 2. Influence subspace (row space of J_t = dh_final_t/dh_tap_t)", "",
        f"- sampled local rows: {len(influence_rows)}, attention-mediated rows: {len(offdiag_influence_rows)} (d = {d_model})",
        f"- effective rank (participation ratio): local {effective_rank:.1f}, full {effective_rank_full:.1f}",
        f"- energy captured at rank {rank}: local {energy_at_rank:.2%}, full {energy_at_rank_full:.2%}",
        f"- influence mass via other positions (attention-mediated): {sum(cross_mass) / len(cross_mass):.2%}", "",
        "## 3. Alignment with the top-{0} influence basis (random baseline {1:.3f})".format(rank, baseline), "",
        f"- tap-state energy inside the influence subspace: local {tap_fraction:.3f}, full {tap_fraction_full:.3f}",
    ]
    if corrector_fraction is not None:
        lines.append(
            f"- corrector read-direction energy inside the influence subspace: local {corrector_fraction:.3f}, "
            f"full {corrector_fraction_full:.3f}"
        )
    if projection_ablation is not None:
        lines += [
            "", "## 4. Projection ablation of the corrector input (trunk stream untouched)", "",
            "| variant | tap energy kept | delta cosine | delta norm ratio | active-token agreement |",
            "| --- | --- | --- | --- | --- |",
        ]
        for name, s in projection_ablation.items():
            lines.append(
                f"| {name} | {s['tap_energy_kept']:.3f} | {s['delta_cos']:.3f} | "
                f"{s['delta_norm_ratio']:.3f} | {s['active_token_agreement']:.3f} |"
            )
        lines.append(
            f"\nActive positions (correction changes argmax token): "
            f"{agg['keep-local']['active']} of {len(agg['keep-local']['cos'])} completion positions."
        )
    lines += ["", "Interpretation: (1) near-zero reconstruction error proves the tapped",
              "tensor is the exact residual stream the upper trunk consumes; (2) the",
              "spectrum measures how low-dimensional the functionally live subspace is",
              "(local = same-position block; full adds the attention-mediated rows,",
              "i.e. how earlier tap states influence later outputs via keys/values);",
              "(3) low corrector alignment means the corrector's read directions live",
              "largely in the complement of the trunk's dominant Jacobian subspace;",
              "(4) if corrector function survives `remove` but not `keep`, restricting",
              "the interface to a concept-aligned (Jacobian-isolated) subspace would",
              "discard exactly the signal the corrector monitors."]
    report = "\n".join(lines) + "\n"
    print(report, flush=True)
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(report, encoding="utf-8")
    return results
