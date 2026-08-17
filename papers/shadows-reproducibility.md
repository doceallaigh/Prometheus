# Latent Chain-of-Thought Reproducibility Protocol

This document is the command-level companion to `papers/shadows.tex`. The code repository's intended release URL is:

<https://github.com/doceallaigh/Prometheus>

Unauthenticated access currently returns 404. Before external reproduction, the authors must make this repository public or provide an equivalent supplementary archive. After access is enabled, clone it and run commands from its root:

```powershell
git clone https://github.com/doceallaigh/Prometheus.git
Set-Location Prometheus
```

The protocol separates retained-checkpoint evaluations from fresh training because they have very different compute and artifact requirements. Paths under `outputs/` are ignored by Git and are not included in a fresh clone. The timestamped paths below identify the authors' local provenance artifacts; use them only when those artifacts have been obtained separately. Otherwise, follow the full-training instructions and replace every example run path with the path printed by your run.

## Environment

The verified reproduction environment was:

- Windows with PowerShell
- Python 3.12.10
- PyTorch 2.6.0+cu124
- NVIDIA RTX 3090 with 24 GB VRAM for the feasible local runs

Create or activate the repository environment, install the project dependencies, and expose the source tree:

```powershell
$env:PYTHONPATH = "src"
```

Use the repository-local interpreter explicitly if shell activation resolves to another Python:

```powershell
.\.venv\Scripts\python.exe -m prometheus.cli --help
```

Commands below use `python`; replace it with `.\.venv\Scripts\python.exe` when needed. Hugging Face evaluations require access to the named model and dataset, plus sufficient local cache or network access.

## 1. Retained Synthetic Corrector Evaluation

The headline synthetic evaluation uses the authors' retained base and CfC sidecar checkpoints and the base snapshot's deterministic validation split. These timestamped directories are provenance identifiers, not files distributed through GitHub:

```powershell
python -m prometheus.cli evaluate-reasoning `
  --base-run outputs/rrs-base-cot-20260710-035418 `
  --latent-run outputs/rrs-j-cfc-20260714-012846 `
  --num-problems 300 `
  --device cuda
```

Expected headline values are approximately 0.503 visible-CoT accuracy and 0.863 latent accuracy, with about 3.9 emitted tokens and 18.9 internal steps for the latent system.

Evaluate the retained ablations by replacing `--latent-run` with one of:

- `outputs/rrs-j-cfc-ablate-embed-20260714-024933`
- `outputs/rrs-j-cfc-ablate-gru-20260714-024515`
- `outputs/rrs-j-cfc-ood26-20260714-024934`

The OOD run trains on chain depths 2-6. Its unseen depth-7/depth-8 accuracies should be approximately 0.854/0.652, versus visible CoT at 0.396/0.326.

Detailed rerun reports are in:

- `reports/20260723-reproduction-rrs-j-cfc-headline.md`
- `reports/20260723-reproduction-rrs-j-cfc-embed.md`
- `reports/20260723-reproduction-rrs-j-cfc-gru.md`
- `reports/20260723-reproduction-rrs-j-cfc-ood26.md`

### Full synthetic training

Every `train` command creates `outputs/<run-name>-<UTC timestamp>` and prints `Run artifacts written to <path>`. Train the trunk first:

```powershell
python -m prometheus.cli train --config configs/rrs_base_pretrain.yaml
```

Record the printed directory as `<new-base-run>`. Before training the sidecar, change `model.base_checkpoint` in `configs/rrs_j_cfc_distill.yaml` from the provenance path to `<new-base-run>/checkpoint.pt`, then run:

```powershell
python -m prometheus.cli train --config configs/rrs_j_cfc_distill.yaml
```

Record that printed directory as `<new-latent-run>` and evaluate the fresh pair with:

```powershell
python -m prometheus.cli evaluate-reasoning `
  --base-run <new-base-run> `
  --latent-run <new-latent-run> `
  --num-problems 300 `
  --device cuda
```

Do not reuse the example timestamps after rebuilding. The base and sidecar configurations fix the data seed, model dimensions, tap layer, optimizer, and schedules. Ablation YAML files alter the stated factor while retaining the shared setup; likewise update any base-checkpoint field and evaluation argument to the newly printed paths.

## 2. Relational Latent Composition From Scratch

Run one independent training/evaluation per seed:

```powershell
$seeds = 20260720, 20260721, 20260722
foreach ($seed in $seeds) {
  python -m prometheus.cli latent-composition `
    --output-dir "outputs/latent-composition-$seed" `
    --train-proofs 1000 `
    --test-per-depth 200 `
    --train-min-depth 2 `
    --train-max-depth 4 `
    --test-max-depth 10 `
    --entities 24 `
    --distractors 4 `
    --d-model 128 `
    --trunk-steps 1200 `
    --sidecar-steps 1800 `
    --batch-size 64 `
    --learning-rate 0.0003 `
    --seed $seed `
    --device cuda
}
```

Each output directory contains:

- `formal_proofs.jsonl`: generated training and held-out proofs
- `trunk.pt`: trained no-CoT trunk
- `sidecar.pt`: trained relational sidecar
- `metrics.jsonl`: accuracy by proof depth and latent step count
- `accuracy_vs_latent_steps.html`: interactive result plot
- `report.md`: tabular summary

Aggregate expectations across the three seeds are:

- mean OOD direct accuracy: `0.0861 +/- 0.0048`
- mean matched-step latent accuracy: `0.9797 +/- 0.0086`
- mean gain: `+0.8936 +/- 0.0125`
- unique best step count equals proof depth in all 27 seed-by-depth conditions

The independently rerun artifact directories are `outputs/reproduction-latent-composition-{20260720,20260721,20260722}-20260723/`. See `reports/20260723-latent-cot-reproduction.md` for the aggregate audit.

## 3. Qwen2.5-0.5B Evaluation

The paper's 200-problem evaluation uses the retained full-harvest corrector:

```powershell
python -m prometheus.cli retrofit-eval `
  --model Qwen/Qwen2.5-0.5B-Instruct `
  --corrector outputs/retrofit-qwen05b/corrector-7k/corrector.pt `
  --num-problems 200 `
  --max-new-tokens 512 `
  --latent-samples 8 `
  --temperature 0.6 `
  --device cuda
```

Use `--latent-samples 1` for greedy latent decoding. Add `--output <path>` to retain a Markdown report. The command writes completion-level data alongside its report when the evaluation path supports a dump.

A fresh 50-problem offline audit reproduced visible-CoT/latent lenient parity at 0.440 and approximately 72-fold emitted-token compression. It is a precision-limited audit, not a replacement for the paper's 200- or 1,319-problem estimates. See `reports/20260723-reproduction-qwen05b-n50.md`.

### Controlled context-carryover evaluation

The earlier `reports/20260724-multiturn-qwen05b-n51.md` run groups independent GSM8K problems. It is an accumulated-context interference baseline and does **not** test whether later turns require information from earlier reasoning.

The controlled carryover audit places a deterministic two-digit key only in the assistant's first-turn scratch work. Turn 2 asks for that key and turn 3 asks for the key plus seven. The key never appears in a user prompt or latent surfaced answer. This command reproduces the retained 17-episode comparison:

```powershell
python -m prometheus.cli retrofit-eval-multiturn `
  --model Qwen/Qwen2.5-0.5B-Instruct `
  --corrector outputs/retrofit-qwen05b/corrector-7k/corrector.pt `
  --episodes 17 `
  --turns 3 `
  --context-mode dependent `
  --max-new-tokens 512 `
  --problem-offset 0 `
  --device cuda `
  --output reports/20260804-context-carryover-qwen05b-n17.md
```

Visible CoT retains the key in visible reasoning. `latent_kv_persist` retains the hidden key in its KV entries, whereas `latent_kv_reset` rebuilds the trunk context from answer-only surfaced responses that omit it. Both latent arms retain the recurrent corrector state, so their comparison isolates trunk KV-cache persistence rather than resetting two state mechanisms at once. Strict context-dependent follow-up accuracy is 0.618/0.529/0.000 for visible CoT, latent KV-persist, and latent KV-reset. Persistent latent wins 18 paired follow-ups against reset and loses none.

As in the dedicated wall-clock benchmark, the evaluator uses `perf_counter` and synchronizes CUDA immediately before and after each generation call. Reported prompt-to-answer wall time per turn includes model prefill and autoregressive decoding, but excludes model and dataset loading, prompt tokenization, scoring, and report serialization. Unlike that benchmark, the retained carryover run did not perform a separate per-system warm-up. On the retained NVIDIA RTX 3090 run, visible CoT, latent KV-persist, and latent KV-reset required 1.206, 1.342, and 1.439 seconds per turn, respectively. Post-loading end-to-end time was 203.5 seconds. Because generated chain lengths differ across arms, these gross times do not isolate a causal cache-speed effect; token throughput is the better hardware-normalized comparison.

The Markdown report contains aggregate, timing, and paired outcomes; the adjacent JSON contains per-arm totals and every completion's per-turn wall time. This is a preliminary fixed-budget audit, not a powered significance test.

To compare visible SC@8 with adaptive latent SC from the same persistent conversation snapshots, add:

```powershell
  --sc-samples 8 --stop-agreement 4 --temperature 0.6 --seed 20260804
```

The retained run is `reports/20260804-context-carryover-sc-qwen05b-n17.md`. On the 34 dependent follow-ups, latent stop-4-of-8 scored 0.529 versus visible SC@8 at 0.500 (five paired wins, four losses, 25 ties) while using 5.29/4.88 rollouts on turns 2/3 rather than eight. This small difference is unresolved. Adaptive SC did not improve over greedy persistent latent on aggregate follow-up accuracy. Post-hoc oracle coverage from the retained rollouts was 0.676 latent and 0.706 visible, showing some headroom for arbitration but also substantial correlated failure.

### Scoring and token accounting

- Strict GSM8K scoring requires a valid `####` answer marker.
- Lenient scoring adds a last-number fallback uniformly across compared systems.
- Emitted tokens are surfaced to the user.
- Internal tokens include the hidden scratchpad rollout.
- Internalization does not by itself imply lower single-rollout trunk FLOPs.
- Self-consistency rollouts should use the same sample count, temperature, problem slice, and seed when comparing systems.

The `outputs/` completion dumps and dated `reports/` files are authoritative for the exact cohorts behind individual table rows.

## 4. QwQ-32B on MATH

The 32B run requires local QwQ-32B weights, 4-bit NF4 loading, and an A100-class GPU. Use the MATH dataset selector and quantized loading supported by `retrofit-eval`:

```powershell
python -m prometheus.cli retrofit-eval `
  --model <local-or-hugging-face-QwQ-32B-path> `
  --corrector <QwQ-32B-corrector.pt> `
  --dataset math `
  --quantize `
  --num-problems 100 `
  --max-new-tokens 4096 `
  --latent-samples 1 `
  --device cuda
```

The retained result and hardware context are documented in `reports/20260720-qwq32b-math.md`. Model and corrector paths depend on the local artifact layout and are therefore placeholders above.

## 5. Geometry, Detection, and Intervention Pipelines

The CLI exposes the full analysis pipeline used by the geometry and arbitration sections. Start from a retained `retrofit-eval` completion dump and the matching model/corrector. Relevant commands include:

- `retrofit-jspace-verify`: staged tap reconstruction, sampled Jacobian basis extraction, and projection ablations
- `divergence-label`: fork- or arithmetic-error-aligned labels
- `divergence-probe`: recognition curves around those anchors
- `trigger-lab`: matched-rollout trigger comparison
- `rollback-oracle` and `rollback-probe`: oracle and deployable rollback
- `steer-inject`: oracle/probe/CUSUM steering interventions
- `retrofit-answer-monitor`: answer-phase error-vector monitoring
- `retrofit-complement-fork` and `retrofit-basin-analyze`: complement suppression, branching, and basin analysis

Inspect current arguments before running a stage:

```powershell
python -m prometheus.cli <command> --help
```

Keep the model, corrector, tap layer, problem slice, basis, and completion dump matched across stages. Trigger comparisons in the paper use a common 5% rollout-level FPR. Bootstrap and matched-noise analyses use the seeds stored in their reports and CLI defaults unless a report states otherwise.

These analyses are substantially more expensive than retained synthetic evaluation. Their dated reports and artifacts should be treated as provenance unless the complete pipeline is rerun.

## 6. Provenance and Scope

The focused independent reproduction is `reports/20260723-latent-cot-reproduction.md`. It freshly reran:

- three relational-composition trainings from scratch
- four retained synthetic checkpoint evaluations
- one limited offline Qwen2.5-0.5B evaluation

It did not independently rerun the full multi-scale SC@8 ladder, full GSM8K test, 32B transfer, complement geometry, detection/intervention suite, Pythia ontogeny, or train-through LoRA ablations. Those claims retain their original support in dated reports and output artifacts.

When a paper value and a later audit differ because of cohort size, seed, or retained versus fresh execution, use the exact report named by the paper row rather than averaging across cohorts.
