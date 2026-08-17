# Design Doc: J-Space CfC Latent Reasoning Loop — Proof of Concept

Date: 2026-07-09
Status: Proposed (supersedes `reports/20260709-superposition-phase-zones-poc-design.md`)
Owner: Prometheus autoresearch

## 1. Summary

Implement and train a proof of concept of a hybrid **Transformer + Closed-form Continuous-time
(CfC)** architecture that replaces explicit text Chain-of-Thought with a silent recurrent loop in
latent space. A frozen transformer supplies semantics; a small CfC "working memory" core reads the
hidden state at a mid-to-late layer (**J-space layer** `L_j`), iterates internally for a variable
number of steps, and writes the converged state back into the residual stream before the upper
layers emit the answer — no intermediate reasoning tokens are generated.

The reference blueprint targets a frontier-class frozen LLM and MATH/GSM8K distillation. Prometheus
has neither frontier checkpoints nor natural-language reasoning data, so this PoC scales the idea
down to a fully self-contained setting: a small dense transformer pretrained on a synthetic
multi-step reasoning task with programmatic CoT, then distilled into the latent loop. Everything
in the blueprint's causal chain (J-space tap, nonlinear bridges, CfC loop, gated readout, two-phase
distillation) is preserved; only the scale changes.

## 2. Goals and non-goals

### Goals

- G1: A `jspace_cfc` architecture in `src/prometheus/model.py`: frozen dense trunk + inbound/
  outbound MLP bridges + CfC cell + termination head, hooked at a configurable layer `L_j`.
- G2: A synthetic reasoning dataset with programmatic CoT, where answer accuracy is separable from
  language modeling (exact-match evaluable).
- G3: The two-phase training pipeline: (1) supervised CoT pretraining of the base model + target
  activation harvesting, (2) trajectory distillation of the CfC loop against those targets.
- G4: A three-way evaluation: direct answering (no CoT), explicit text CoT, and the latent CfC
  loop — on accuracy, output token count, wall-clock, and loop steps used.
- G5: Adaptive computation: demonstrate that the learned termination head allocates more loop
  steps to harder problems.

### Non-goals

- Frontier-model integration, MATH/GSM8K, or natural-language CoT.
- True continuous-time / irregularly-sampled inputs (we use the CfC cell in fixed-Δt recurrent
  mode; the closed-form solution is what makes it cheap, not the ODE view).
- Beating the text-CoT ceiling. The PoC succeeds if the latent loop **recovers a substantial
  fraction of the CoT accuracy gain at a fraction of the emitted tokens** (see Section 8).

## 3. Background: why this fits Prometheus

- The repo already has a shared-weight `recurrent_loop` architecture (`RecurrentLoopTransformerLM`,
  added 2026-06-13) with `recurrent_steps` and `recurrent_state_blend`. The J-space CfC design is
  the natural next step: instead of recurring the whole trunk over the token embedding, it recurs a
  small dedicated cell over a **single layer's hidden state**, with a frozen trunk and a learned
  stopping rule.
- The 2026-06-05 logic-protocol design doc (`reports/20260605-logic-protocol-emergence-experiment-design.md`)
  already proposed a synthetic logic dataset with `reasoning_mode: direct | formal_dsl |
  learned_protocol`. This PoC reuses that dataset direction; the CfC loop is effectively a
  continuous-valued `learned_protocol` that never surfaces as tokens.
- The `dense_ring_memory` sidecar pattern (trunk + fused auxiliary module) is a precedent for
  bolting a stateful module onto the dense trunk without rewriting it.

## 4. Architecture specification

### 4.1 Component overview

```
[ tokens ] → embed → blocks[0 .. L_j-1] ─┬─→ h_Lj ── Φ_in ──→ h_cfc(0)
                                         │                       │
                                         │              CfC cell loop (t = 1..T)
                                         │              p_stop = σ(w_stop · h_cfc(t))
                                         │                       │  stop when p_stop ≥ θ or t = T_max
                                         │                       ▼
                                         └─→ h'_Lj = h_Lj + g · Φ_out(h_cfc(T))
                                                       │
                                    blocks[L_j .. N-1] → norm → lm_head → [ answer tokens ]
```

### 4.2 Semantic anchor (frozen base transformer)

- `DenseTransformerLM`, pretrained in Phase 1 (Section 6), then frozen (`requires_grad=False`,
  eval-mode dropout off).
- PoC scale: `embedding_dim 256, num_heads 8, num_layers 8, sequence_length 256`. Roughly the
  `baseline_dense_scaled_*` band — big enough to have a meaningful mid-late abstraction zone,
  small enough for fast RTX 3090 iteration.
- **J-space layer selection.** In the blueprint, `L_j` is "where abstract concept representation
  peaks before token translation begins." At PoC scale we make this empirical rather than assumed:
  during Phase 1 we train a linear probe per layer to predict the final answer from the hidden
  state at the last-CoT-token position, and set `L_j` to the probe-accuracy peak (expected around
  2/3 depth, i.e. layer 5–6 of 8). `jspace_layer_index` remains a config override.

### 4.3 Nonlinear alignment bridges

- `Φ_in`: `Linear(d_model → d_cfc) → GELU → Linear(d_cfc → d_cfc)`.
- `Φ_out`: `Linear(d_cfc → d_cfc) → GELU → Linear(d_cfc → d_model)`, zero-initialized final layer
  so the loop starts as an exact no-op on the frozen trunk.
- Write-back is **residual and gated**: `h'_Lj = h_Lj + g · Φ_out(h_cfc(T))` with a learned scalar
  (or per-channel) gate `g` initialized at 0. This keeps Phase 2 training stable: the base model's
  behavior is exactly preserved at initialization and the loop's influence grows only as it helps.
- Default `d_cfc = 128` (half of `d_model`); swept in Section 8.

### 4.4 CfC working-memory core

- A single **CfC cell** (Hasani et al. 2022, "Closed-form Continuous-time Neural Networks"):
  liquid time-constant dynamics with the ODE solved analytically, so each step is a standard
  differentiable forward pass — no ODE solver.
- Implemented self-contained in `src/prometheus/latent_reasoning.py` (~40 lines: the closed-form
  gate `h(t+1) = σ(-f(x,h)) ⊙ g(x,h) + (1 − σ(-f(x,h))) ⊙ k(x,h)` with small backbone MLPs), to
  avoid adding the `ncps` dependency. If the hand-rolled cell underperforms, `ncps.torch.CfC` is
  the fallback (add as optional dependency).
- Loop input at every step is the cell state itself plus a constant context injection of
  `Φ_in(h_Lj)` (the "unmoving anchor"): `h_cfc(t) = CfC(input=Φ_in(h_Lj), state=h_cfc(t−1))`.
  This implements the blueprint's claim that the frozen prompt encoding anchors the trajectory.
- The loop runs at the **final prompt position only** (the position about to emit the answer),
  not across all sequence positions — one trajectory per problem.

### 4.5 Gated readout / termination

- `p_stop(t) = σ(w_stop · h_cfc(t) + b_stop)`; loop halts when `p_stop ≥ θ` (default 0.9) or
  `t = T_max` (default 16).
- Training uses an **ACT-style soft halting** (Graves 2016): the effective output is the
  halting-probability-weighted mixture of per-step states, plus a small ponder cost `τ · E[steps]`
  so the model learns to stop early when it can. Hard thresholding is inference-only. This fixes a
  gap in the reference sketch, where the termination head appears in the loop but receives no
  training signal.

## 5. Dataset

Synthetic **multi-step arithmetic-chain** task (new `dataset_type: reasoning_chain` in
`src/prometheus/data.py`):

- Problem: `x = 7; x = x + 12; x = x * 3; x = x - 5; x = ?` with 2–8 chained operations over a
  bounded modular integer domain (e.g. mod 100, so the answer is always 1–2 tokens and exact-match
  evaluable with a character-level vocab).
- CoT form: `... x = ? ; THINK 7 19 57 52 ; ANSWER 52` — the intermediate values are the
  programmatic chain of thought.
- Direct form: `... x = ? ; ANSWER 52`.
- Difficulty is controlled by chain length, giving a clean axis for the adaptive-computation claim
  (G5). Splits are generated by problem hash so no chain appears in both train and eval.
- Rationale vs. the blueprint's MATH/GSM8K: at Prometheus scale a natural-language benchmark is
  untrainable, but the essential property — **problems where explicit intermediate steps
  demonstrably improve accuracy** — is reproducible synthetically, and prior repo work
  (logic-protocol design) already committed to this route.

## 6. Training plan

### Phase 0 — Base pretraining

Train the dense base model on a mixture of direct and CoT-format problems (so both formats are
in-distribution), standard next-token CE, until validation exact-match accuracy plateaus. Verify
the premise: CoT format must beat direct format at equal parameters (expected gap grows with chain
length). If direct answering already saturates, lengthen chains until it does not — a task where
CoT does not help cannot validate latent reasoning.

### Phase 1 — Target activation harvesting

- Run the frozen base with CoT prompts over the training set; record `H_target` = hidden state at
  `L_j` at the position immediately before the answer token is emitted (i.e., after the full THINK
  span has been consumed).
- Also record `h_Lj_start` = hidden state at `L_j` for the **direct-format** prompt at the same
  logical position (before any THINK tokens). The pair `(h_Lj_start → H_target)` defines what the
  loop must accomplish: transform the "just read the problem" state into the "finished reasoning"
  state.
- Run the per-layer linear probe sweep here to finalize `L_j` (Section 4.2).
- Artifacts: `outputs/<run>/jspace_targets.pt` (memory-mapped tensor store + index), plus a
  `probe.summary.json` documenting the layer choice.

### Phase 2 — Latent loop distillation

Trainable: `Φ_in`, `Φ_out`, CfC cell, termination head, write-back gate. Frozen: everything else.

Loss (fixing two defects in the reference sketch):

1. **Terminal trajectory loss**, not per-step-averaged: the sketch averages cosine loss over all
   unrolled steps, which pressures the trajectory to jump to the target at step 1 and makes the
   loop pointless. We apply the representation-matching loss to the **ACT-weighted final state**:
   `L_repr = 1 − cos(h_Lj_start + Φ_out(h̄_cfc), H_target) + λ‖(h_Lj_start + Φ_out(h̄_cfc)) − H_target‖² / d_model`.
   The cosine term matches direction; the small MSE term (λ ≈ 0.1) pins magnitude, which cosine
   alone ignores — residual-stream magnitude matters to downstream LayerNorm.
2. **Answer loss through the frozen upper layers**: `L_ans = CE(lm_head(upper_layers(h'_Lj)), answer)`.
   Representation matching alone optimizes a proxy; gradients flowing through the frozen upper
   layers train the loop for what actually matters. Weighting: `L = L_ans + β·L_repr + τ·E[steps]`
   with β annealed 1 → 0.1 over training (representation matching as a scaffold, answer accuracy
   as the endpoint).
3. **Ponder cost** `τ·E[steps]` trains the termination head (Section 4.5).

Curriculum: start with short chains (2–3 ops) and `T_max = 4`, grow to 8 ops / `T_max = 16`.

### Phase 3 — Evaluation

Three matched systems on held-out problems, stratified by chain length:

| System | Description |
| --- | --- |
| Direct | Frozen base, direct prompt, no loop |
| Text CoT | Frozen base, generates THINK tokens autoregressively, then answer |
| Latent CfC | Frozen base + trained loop, direct prompt, no THINK tokens |

Metrics per system: exact-match accuracy, emitted tokens per problem, wall-clock per problem
(existing `tokens_per_second` machinery), and for the latent system: loop steps used vs. chain
length (G5), plus trajectory diagnostics (cosine-to-target vs. t, drift after convergence).

## 7. Implementation plan (repo integration)

- `src/prometheus/latent_reasoning.py` — CfC cell, bridges, ACT halting, `JSpaceCfCLoop` module.
- `src/prometheus/model.py` — `JSpaceCfCTransformerLM(LanguageModelBase)`: loads a frozen
  `DenseTransformerLM` checkpoint (`base_checkpoint` config field), splits blocks at
  `jspace_layer_index`, inserts the loop. Registered as `architecture: jspace_cfc`.
- `src/prometheus/config.py` — new `ModelConfig` fields: `base_checkpoint`, `jspace_layer_index`,
  `cfc_dim`, `cfc_max_steps`, `cfc_stop_threshold`, `ponder_cost`, `repr_loss_weight`.
- `src/prometheus/data.py` — `dataset_type: reasoning_chain` with `chain_length_min/max`,
  `reasoning_format: direct | cot | mixed`.
- `src/prometheus/train.py` — Phase-2 trainer path: harvested-target loading, composite loss,
  frozen-trunk handling; reuses existing metrics/checkpoint/reporting plumbing.
- `src/prometheus/cli.py` — `harvest-targets` subcommand (Phase 1) alongside existing `train`.
- Configs: `configs/jspace_base_pretrain.yaml`, `configs/jspace_harvest.yaml`,
  `configs/jspace_cfc_distill.yaml`, `configs/smoke_jspace_cfc.yaml` (CPU, tiny dims, `T_max 4`).
- Tests: CfC cell shape/gradient tests; no-op-at-init test (zero-gated loop reproduces frozen base
  logits bit-exactly); ACT halting monotonicity test; dataset determinism test.

Verified command shape (repo conventions, explicit venv python):

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m prometheus.cli train --config configs/jspace_base_pretrain.yaml
.\.venv\Scripts\python.exe -m prometheus.cli harvest-targets --config configs/jspace_harvest.yaml --run-dir outputs/<base-run>
.\.venv\Scripts\python.exe -m prometheus.cli train --config configs/jspace_cfc_distill.yaml
```

## 8. Success criteria and sweeps

1. **Premise check (Phase 0):** text CoT beats direct answering by ≥ 15 accuracy points on chains
   of length ≥ 5. If not, the task is retuned before any loop work proceeds.
2. **Primary:** latent CfC recovers ≥ 50% of the (CoT − direct) accuracy gap while emitting the
   same number of tokens as the direct system (PoC bar; higher recovery is upside).
3. **Adaptive compute:** mean loop steps increases monotonically with chain length, and ponder
   cost yields ≥ 30% fewer average steps than always running `T_max`, at ≤ 2 points accuracy cost.
4. **Stability:** no-op-at-init test passes; trajectory cosine-to-target is non-decreasing in t on
   ≥ 80% of eval problems (bounded dynamics, no drift-to-nonsense).
5. Sweeps (after the primary result): `d_cfc ∈ {64, 128, 256}`, `L_j ∈ {4, 5, 6, 7}` vs. the
   probe-selected layer, `T_max ∈ {4, 8, 16}`, and an ablation replacing the CfC cell with a plain
   GRU cell — this isolates whether the *closed-form liquid dynamics* matter or any gated
   recurrence suffices (the honest headline question for a skeptical reviewer).

## 9. Milestones

- M1: `reasoning_chain` dataset + Phase-0 base pretraining + premise check. ~1 session.
- M2: CfC cell + bridges + `jspace_cfc` model + smoke config + no-op/gradient tests. ~1 session.
- M3: Phase-1 harvesting CLI + layer-probe selection. ~0.5 session.
- M4: Phase-2 distillation trainer + first full training run. ~1 session.
- M5: Phase-3 three-way evaluation + report under `reports/`. ~0.5 session.
- M6 (stretch): GRU ablation and sweeps; multi-position loop (run the loop at every position
  rather than only the answer position) if the single-position result is strong.

## 10. Risks and open questions

- **Representation matching may be too easy to shortcut:** `Φ_out` could learn to output
  `H_target − h_Lj_start` in one step for memorized problems, bypassing the loop. Mitigations:
  held-out-chain evaluation (G2 split hygiene), β annealing toward the answer loss, and the
  step-count/accuracy curve as a diagnostic (a genuine iterative reasoner should improve with more
  steps on longer chains).
- **Single-position injection may be too weak** at PoC scale, where "reasoning" is distributed
  across positions. Fallback: inject the loop output as a KV-visible memory token rather than a
  single-position residual edit (small design delta, same components).
- **CfC value-add unproven at this scale:** the GRU ablation (Section 8.5) is deliberately part of
  the success criteria; if GRU matches CfC, the honest conclusion is that the architecture's win is
  the *loop + gating + distillation recipe*, not liquid dynamics.
- **Blueprint deviations (intentional):** per-step-averaged loss replaced by terminal ACT loss;
  cosine-only loss augmented with magnitude term; untrained termination head replaced with
  ACT/ponder training; frontier LLM replaced by repo-scale frozen base. Each is flagged above at
  the point of deviation.
- **BPTT cost through the unrolled loop** is modest at `T_max ≤ 16` and `d_cfc ≤ 256`; if it
  becomes limiting, truncate BPTT to the last 8 steps.
