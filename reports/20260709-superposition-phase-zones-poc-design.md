# Design Doc: Superposition Phase-Zone Proof of Concept (Anthropic Mech Interp Reproduction)

Date: 2026-07-09
Status: Superseded — the requester clarified that "J zone" referred to the J-space layer tap in a
Transformer-CfC latent reasoning blueprint, not Anthropic superposition phase zones. See
`reports/20260709-jspace-cfc-latent-reasoning-poc-design.md` for the active design.
Owner: Prometheus autoresearch

## 1. Background and source verification

The originating request referenced "Anthropic's research on the j zone as it pertains to mech interp"
via a Google AI-mode search link. That link could not be retrieved (JavaScript-gated), and a direct
review of Anthropic's interpretability publication index (transformer-circuits.pub, checked
2026-07-09, covering 2021 through July 2026) found **no research artifact named the "j zone"**. The
term is most likely a voice-transcription or AI-answer artifact.

The closest well-defined body of Anthropic work involving named "zones"/regimes in mech interp is:

1. **Toy Models of Superposition** (Elhage et al., 2022) — small ReLU models exhibit a *phase
   diagram* with distinct zones as feature sparsity and relative importance vary: a zone where a
   feature gets a dedicated dimension, a zone where it is stored in superposition (antipodal pairs
   and higher polytopes), and a zone where it is not represented at all.
2. **Superposition, Memorization, and Double Descent** (Henighan et al., 2023) — extends the toy
   model to show distinct *training-set-size zones* (memorization vs. generalization) with a phase
   transition between them.
3. Alternative candidate: **Verbalizable Representations Form a Global Workspace** (Gurnee et al.,
   July 2026). Rejected for this PoC: it requires frontier-scale models and is not reproducible on
   Prometheus infrastructure.

**Decision:** implement a proof of concept reproducing the superposition **phase zones** of (1),
with (2) as a stretch goal. This is the most defensible interpretation, is fully trainable on CPU in
minutes per run, and directly serves Prometheus's existing research program (sparsity, pruning,
fan-in limits, and capacity allocation are all superposition questions in disguise). If the
requester meant a different "zone," this doc's Section 8 lists the pivot options; the harness built
here is reusable for any of them.

## 2. Goals and non-goals

### Goals

- G1: Implement Anthropic's ReLU-output toy model of superposition inside `src/prometheus`.
- G2: Train sweeps over feature sparsity and relative importance; reproduce the characteristic
  phase diagram (dedicated-dimension zone / superposition zone / dropped-feature zone).
- G3: Emit quantitative zone metrics per run (dimensions-per-feature, feature norms, interference)
  and an aggregate markdown + figure report, consistent with existing Prometheus reporting.
- G4: Connect findings back to the main Prometheus program: use the measured superposition capacity
  of hidden layers to explain the observed dense fan-in inflection (best-val-loss turnover in the
  mid-2k fan-in band) and keep-ratio pruning cliffs (degradation below keep_ratio 0.75).

### Non-goals

- Reproducing SAE/dictionary-learning results at LM scale (Towards Monosemanticity, Scaling
  Monosemanticity). A small SAE on the toy model is a stretch goal only.
- Any frontier-model work (attribution graphs, global workspace, introspection).
- New LM architectures. This PoC trains toy models, not language models.

## 3. Technical design

### 3.1 Toy model (from Elhage et al., 2022)

- Data: synthetic feature vectors `x ∈ R^n`. Each feature `x_i` is zero with probability `S`
  (sparsity) and uniform on `[0, 1]` otherwise. Default `n = 20` features.
- Model: linear map down, tied linear map up, ReLU, bias:
  `h = W x` with `W ∈ R^{m×n}` (hidden size `m << n`, default `m = 5`), and
  `x' = ReLU(Wᵀ h + b)`.
- Loss: importance-weighted MSE, `L = Σ_i I_i (x_i − x'_i)²`, with importance schedule
  `I_i = r^i` for decay `r` (default 0.9), or a two-group schedule for the phase-diagram sweep
  (one probe feature with variable relative importance against a uniform background).
- Optimizer: AdamW, full-batch synthetic sampling, ~10k steps. Each run is seconds-to-minutes
  on CPU.

### 3.2 Phase-diagram sweep

Grid over:

- Sparsity axis: `1 − S ∈ {1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001}` (log-spaced density).
- Relative-importance axis: probe-feature importance in `{0.1 … 10}` (log-spaced, ~9 points).
- ≥ 3 seeds per cell; small `n`/`m` (e.g. `n = 2, m = 1` for the exact Anthropic phase diagram,
  plus an `n = 20, m = 5` uniform-importance sweep for the polytope/geometry zone plot).

### 3.3 Zone metrics (per trained model)

- **Dimensions per feature**: `D_i = ‖W_i‖² / Σ_j (Ŵ_i · W_j)²` — the core Anthropic statistic;
  `D_i ≈ 1` means dedicated dimension, fractional values indicate superposition polytopes.
- **Feature norm** `‖W_i‖`: ≈ 1 represented, ≈ 0 dropped.
- **Interference**: `Σ_{j≠i} (Ŵ_i · W_j)²`.
- **Zone classification** per (sparsity, importance) cell: `dedicated | superposed | dropped`,
  decided from thresholded `D_i` and `‖W_i‖`, majority over seeds.
- Aggregates written to `run.summary.json`-style artifacts; phase-diagram heatmap rendered via the
  existing `visualization.py` matplotlib path.

### 3.4 Code layout

- `src/prometheus/interp/__init__.py`
- `src/prometheus/interp/toy_superposition.py` — model, data sampler, trainer, metrics.
- `src/prometheus/interp/phase_sweep.py` — grid runner + report writer.
- CLI: `python -m prometheus.cli interp toy-superposition --config configs/interp_toy_superposition.yaml`
  (new `interp` subcommand group; reuses config-loading and output-dir conventions from `train`).
- Configs: `configs/interp_toy_superposition.yaml` (single run smoke),
  `configs/interp_phase_sweep_2f1d.yaml` (exact 2-feature/1-dim phase diagram),
  `configs/interp_phase_sweep_20f5d.yaml` (geometry sweep).
- Outputs: `outputs/interp-<name>-<timestamp>/` with `metrics.jsonl`, `run.summary.json`,
  `phase_diagram.png`, and a generated report under `reports/`.

### 3.5 Stretch goals

- S1 (double-descent zones, Henighan et al. 2023): rerun the toy model with a *fixed finite
  dataset* of size `T`, sweep `T`, and plot test loss + dimensions-per-datapoint to reproduce the
  memorization → generalization zone transition.
- S2 (feature recovery): train a small sparse autoencoder on the toy model's hidden layer and
  measure recovery of ground-truth features vs. zone (features in the superposition zone should be
  recoverable; dropped features should not).
- S3 (bridge to Prometheus LMs): estimate effective features-per-neuron in the hidden layers of the
  existing `fanin_dense_*` checkpoints and test whether the fan-in 2560 inflection coincides with a
  superposition capacity limit.

## 4. Training and compute plan

- All runs CPU-feasible; RTX 3090 optional for the large sweep. Full 2-axis grid
  (7 × 9 × 3 seeds ≈ 189 runs × ~30 s) ≈ 1.5–2 CPU-hours, parallelizable.
- Use repo-local `.venv\Scripts\python.exe` explicitly (known Conda-base pitfall).
- Verified command shape:
  `$env:PYTHONPATH = "src"; .\.venv\Scripts\python.exe -m prometheus.cli interp toy-superposition --config configs/interp_toy_superposition.yaml`

## 5. Success criteria

1. Smoke run trains and reproduces the qualitative Anthropic result: with dense features the model
   keeps only the top-`m` important features; with sparse features it stores more than `m` features
   in superposition (antipodal pairs first).
2. Phase-diagram heatmap shows the three zones with clean boundaries and "sticky" plateaus at
   simple fractional `D_i` values (1, 1/2, ...), matching the published figure qualitatively.
3. Zone classification is stable across seeds (≥ 2/3 agreement per cell).
4. Report generated under `reports/` with heatmap, metric definitions, and a comparison paragraph
   against the published figures.

## 6. Milestones

- M1: `interp` module + single-run smoke config + metrics (G1, G3 partial). ~1 session.
- M2: phase sweep runner + heatmap + zone classifier (G2, G3). ~1 session.
- M3: report + comparison to published figures; success-criteria check (G3). ~0.5 session.
- M4 (optional): stretch goals S1–S3, prioritized S3 → S1 → S2 by relevance to Prometheus.

## 7. Risks

- **Interpretation risk (highest):** "j zone" may refer to something else entirely. Mitigation:
  Section 8 pivots; the sweep/report harness generalizes.
- Threshold sensitivity in zone classification → report raw `D_i` heatmaps alongside the
  discretized zones.
- Toy-model conclusions may not transfer to the LM benchmark → S3 is framed as a correlation
  check, not a causal claim.

## 8. Pivot options if the intended topic differs

| If "j zone" meant… | Pivot |
| --- | --- |
| Memorization/generalization regimes | Promote stretch goal S1 to primary |
| Global workspace (Jul 2026) | Not reproducible here; nearest analog is a probe for "privileged" verbalizable dims in Prometheus LMs — new design doc needed |
| Induction-head phase change (2022) | Train 2-layer attention-only LM on existing synthetic data; track in-context-learning score through training |
| SAE features / monosemanticity | Promote stretch goal S2; train SAE on `fanin_dense_2560` activations |
