# RRS-J-CfC proof-of-concept results

Companion to the design doc:
[20260709-jspace-cfc-latent-reasoning-poc-design.md](20260709-jspace-cfc-latent-reasoning-poc-design.md).
Raw comparison table: [20260710-rrs-j-cfc-comparison.md](20260710-rrs-j-cfc-comparison.md).

## Summary

- **Premise verified:** the base model answers far better with chain-of-thought
  than with direct answering (0.50 vs 0.03 exact match).
- **Final latent model (v3) beats CoT:** 0.863 overall exact match vs 0.503 for
  visible CoT and 0.030 for direct answering, while emitting only ~3.9 answer
  characters to the user (CoT emits ~19.1). The reasoning chain stays internal.
- Two earlier injection designs failed for diagnosable reasons; the oracle-first
  debugging loop is documented below because the negative results shaped the
  final architecture.

## Setup

- Base model: `outputs/rrs-base-cot-20260710-035418` — dense 8-layer, 256-dim
  char-level transformer trained 8k steps on mixed direct/CoT arithmetic-chain
  text (`configs/rrs_base_pretrain.yaml`). Final val loss 1.309 (ppl 3.70).
- Latent model (v3): `outputs/rrs-j-cfc-20260714-012846` — frozen base plus a
  CfC cell (dim 256) that rides the model's *silent scratchpad rollout*: the
  base greedily rolls out its chain-of-thought internally (never shown to the
  user), while the CfC cell reads the layer-6 J-space state at each internal
  step and adds a zero-initialized logit correction. Trained 4k steps with
  teacher-forced cross-entropy over the chain plus double-weighted answer
  positions (`configs/rrs_j_cfc_distill.yaml`).
- Evaluation: 300 shared held-out problems, exact-match answer accuracy
  (`prometheus.cli evaluate-reasoning`). "Emitted tokens" counts only what the
  user sees (`A<digits>;`); internal rollout steps are reported separately.

## Headline comparison

| system | overall accuracy | mean emitted tokens | mean internal steps |
| --- | --- | --- | --- |
| direct | 0.0300 | 2.9 | - |
| cot | 0.5033 | 19.1 | - |
| latent_rrs_j_cfc | 0.8633 | 3.9 | 18.9 |

Accuracy by chain length:

| chain length | direct | cot | latent_rrs_j_cfc |
| --- | --- | --- | --- |
| 2 | 0.044 | 0.689 | 0.889 |
| 3 | 0.036 | 0.679 | 0.929 |
| 4 | 0.050 | 0.650 | 0.900 |
| 5 | 0.020 | 0.551 | 0.878 |
| 6 | 0.023 | 0.318 | 0.864 |
| 7 | 0.021 | 0.396 | 0.771 |
| 8 | 0.022 | 0.326 | 0.848 |

The latent model degrades far more slowly with chain length than visible CoT
(0.89 → 0.85 vs 0.69 → 0.33): the CfC corrector fixes exactly the intermediate
arithmetic errors that compound over long chains.

## Why v3 works: design rationale

The v3 architecture has a **structural accuracy floor at CoT level**. The CfC
logit head is zero-initialized, so at initialization the internal rollout is
bit-identical to the base model's greedy CoT rollout (~0.50 accuracy). Training
teacher-forces the true chain and asks the CfC cell — which carries a liquid
state across the whole rollout — to repair the frozen base's next-token errors.
Validation accuracy climbed monotonically from 0.64 (step 500) to 0.875
(step 3500). From the user's perspective the reasoning is latent: only the
final `A<digits>;` segment is emitted.

## Failed variants and oracle diagnostics

**v1 — single-position residual edit (design doc's original plan).** Inject the
loop's output as a residual delta at the answer position of a mid-late layer.
Oracle test (patching the *true* CoT-derived target activation,
`scripts/oracle_jspace_patch.py`): 5% at layer 5, 25-27.5% at layers 6-7 —
the answer is computed by upper-layer attention over visible THINK tokens, not
stored in any single pre-answer position. Trained accuracy: 0.037 (= direct).

**v2 — continuous THINK-span emission.** Emit one virtual THINK *state* per
loop iteration into placeholder positions at layer 1, trained by cosine+MSE
regression against teacher activations. The span oracle showed a 100% ceiling
at layer 1 (patching true span states recovers full CoT accuracy), but
training plateaued: cosine loss stalled at ~0.30 and accuracy stayed at 0.027.
Regressing continuous states is the wrong loss for a discrete target function —
states that are "close" in cosine space decode to wrong digits, and errors
compound across the autoregressive span.

**Span oracle sweep** (true CoT states patched into placeholder prompts,
n=200): layer 1 → 1.000, layer 2 → 0.445, layer 4 → 0.025, layer 5 → 0.010,
layer 7 → 0.020. Deep layers fail because THINK-token attention happens in the
lower blocks; once past them, injected states are never read again.

The v3 pivot keeps the computation in *token space* (where the base model is
already competent) and uses the CfC loop as a continuous-time error corrector
rather than a from-scratch reasoning engine.

## Pre-scaling validity checks (2026-07-14)

Three ablations were run to verify the result is architecturally meaningful
before scaling (all identical to the baseline recipe except one factor):

| experiment | run | final val accuracy |
| --- | --- | --- |
| v3 baseline (layer-6 tap, CfC) | rrs-j-cfc-20260714-012846 | 0.875 |
| embedding-only tap (layer 0) | rrs-j-cfc-ablate-embed-20260714 | 0.543 |
| GRU cell instead of CfC | rrs-j-cfc-ablate-gru-20260714-024515 | 0.828 |
| OOD: train chains 2-6 only | rrs-j-cfc-ood26-20260714-024934 | 0.914 (in-dist) |

- **The J-space tap is doing real work.** With the corrector reading only raw
  token embeddings, accuracy stays pinned near the CoT floor (0.50 → 0.54 over
  4k steps). The +33-point gap to the layer-6 tap proves the cell exploits the
  trunk's partial computation rather than re-deriving mod-100 arithmetic
  itself.
- **The corrector generalizes out of distribution.** Trained on chains 2-6
  only, it scores 0.854 on chain 7 and 0.652 on chain 8 — lengths it never saw
  — vs the base CoT's 0.396 and 0.326 (overall 0.8533 on the full 2-8 eval,
  `20260714-rrs-j-cfc-ood-eval.md`). The learned repair is a genuine
  error-correction function, not length-specific memorization.
- **CfC beats GRU but not by much** (0.875 vs 0.828). The recurrent-corrector
  *architecture* is the main effect; the continuous-time cell is a modest
  refinement.
- **No teacher-forcing gap.** Validation already uses free-running greedy
  rollout, so the training-eval discrepancy (0.875 vs 0.863) is sampling noise.

## Stage 1 cloud replication (2026-07-14, GCP)

The main recipe and both ablations were re-run end to end on GCP
(e2-standard-4 CPU VMs, stage tags `stage1-20260714-002257` and
`stage1-20260714-023640`; artifacts in
`gs://prometheus-rrs-stage-artifacts/`). Full 300-problem reasoning eval:

| run | latent overall | cot | direct | best val acc | gate | verdict |
| --- | --- | --- | --- | --- | --- | --- |
| main (CfC, layer-6) | **0.8833** | 0.5033 | 0.0300 | 0.8828 | >= 0.84 | pass |
| embed-only ablation | 0.5567 | 0.5033 | 0.0300 | 0.5586 | <= 0.60 | pass |
| GRU cell | 0.7833 | 0.5033 | 0.0300 | 0.8086 | < main | pass |

Cloud results replicate the local runs within noise (0.8833 vs 0.875,
0.5567 vs 0.543, 0.7833 vs 0.828), confirming the finding is robust across
hardware, OS, and seed/environment differences. Stage 1 gate: **passed**.

## Stage 2 results (2026-07-14/15)

Three scale/robustness probes, each a full pretrain + distill + 300-problem
eval (local RTX 3090 + GCP CPU replication; stage tag
`stage2-20260714-165138` in `gs://prometheus-rrs-stage-artifacts/`):

| run | base CoT | latent | emitted tokens (CoT → latent) | verdict |
| --- | --- | --- | --- | --- |
| seed2 (seed 2024, GCP) | 0.5033 | 0.8867 | 19.1 → 3.9 | replicates main (0.8833) |
| width512 (2x width) | 0.9833 | **0.9967** | 18.8 → 3.9 | exceeds a near-ceiling base |
| chains12 (chains 2-12, 24k-step base) | 1.0000 | **1.0000** | 23.6 → 3.9 | perfect internalization, 6x token cut |

Two additional findings:

- **Failure boundary (negative result, informative).** The first chains-2-12
  base was undertrained (8k steps, CoT accuracy 0.045); the corrector
  collapsed with it (~0.05, structural floor). The corrector amplifies base
  competence — it repairs recoverable errors and cannot substitute for absent
  competence. Together with the embed ablation this pins down the mechanism.
- **Dose-response across base competence.** Base CoT 0.045 → latent 0.05;
  0.50 → 0.88; 0.98 → 1.00; 1.00 → 1.00. The corrector's value peaks in the
  band where the base is competent-but-erring, and converges to lossless
  rollout internalization (~5-6x fewer emitted tokens) as base errors vanish.

Stage 2 conclusion: the recipe is seed-robust, survives 2x width, and scales
to a harder task given an adequately trained base. The efficiency claim
(CoT-level accuracy at ~5x fewer emitted tokens) held in every passing run.

## Conclusions

- Latent reasoning with a frozen trunk is achievable and can *exceed* visible
  CoT when the latent module corrects the rollout instead of replacing it:
  +36 points over CoT, +83 points over direct, with 5x fewer emitted tokens.
- Oracle-first debugging (bounding what a trained module could possibly achieve
  by patching ground-truth signals) identified both failure modes cheaply,
  before any retraining.
- The pre-scaling checks pass: the J-space signal is necessary, the corrector
  generalizes beyond training chain lengths, and the recipe is robust to the
  recurrent cell choice. The architecture is a reasonable candidate for
  scaling.
- Open follow-ups: distill the internal rollout into fewer steps (the current
  internal budget matches CoT length), and probe how far OOD generalization
  extends (chains 9+ require a longer base context).

## Planned: mechanistic interpretability CLI (`interpret`)

Add a `python -m prometheus.cli interpret` command that runs the standard
mech-interp battery against any trained rrs_j_cfc run, exploiting the fact
that the synthetic dataset provides ground-truth intermediates for free:

1. **Probe sweep** — linear probes for the running intermediate value (mod
   100) at every trunk layer x rollout step, with shuffled-label control
   probes; report accuracy and selectivity per layer. Expected to explain
   *why* the layer-6 tap wins (decodability should peak there) and to turn
   the tap-depth choice from empirical into principled.
2. **Activation patching** — mid-rollout, swap the tapped J-space state
   between two problems and measure whether the corrector's output tracks
   the patched value (interchange intervention accuracy). This upgrades the
   embed-ablation evidence to a direct causal claim: the corrector reads
   exactly the variable we say it reads.
3. **CfC state decoding** — linear probes on the corrector's hidden state
   for (a) current running value, (b) base-rollout-error indicator, and
   (c) correction delta, to distinguish "error detector/repairer" from
   "shadow re-computer" — the central mechanistic question about this
   architecture.

Outputs: a markdown report (probe accuracy heatmap table, patching recovery
rates, state-decoding table) written to `reports/`. All three analyses run in
minutes at current model scale. Scheduled after Stage 2 results land.

## Planned: pretrained-LM retrofit (priority 1 after interp) — DONE, see results above

The decisive test of frontier relevance: retrofit the corrector onto a real
pretrained language model instead of our toy trunk.

- **Trunk**: a frozen open-weights causal LM, starting at the smallest scale
  with usable GSM8K-style CoT (Qwen2.5-0.5B-Instruct or Pythia-1.4B class;
  fits the granted T4, LoRA-scale budget).
- **Recipe** (unchanged in spirit from the toy): the frozen trunk rolls its
  own CoT internally (never emitted); a CfC cell taps a mid-late layer's
  hidden state each step and adds a zero-init logit correction; train by
  teacher-forcing on the model's own harvested CoT traces with answer-weighted
  CE. Structural floor = the trunk's own CoT accuracy.
- **Data**: GSM8K train split for harvest/distill, GSM8K test for eval;
  report accuracy AND emitted-token counts vs (a) direct answering,
  (b) emitted CoT, (c) self-consistency@k at matched token budget.
- **What it tests that the toy cannot**: heterogeneous error distributions,
  natural-language J-space at realistic width/depth, and whether the
  token-efficiency result transfers off synthetic arithmetic.
- **Success bar**: latent accuracy within a few points of emitted CoT at a
  large emitted-token reduction. Beating CoT is a stretch goal; matching it
  cheaply is the efficiency claim the paper needs.

## Pretrained-LM retrofit results (2026-07-15)

The priority-1 experiment ran to completion (branch
`feature/pretrained-retrofit`, module `src/prometheus/retrofit.py`, CLI
`retrofit-harvest` / `retrofit-train` / `retrofit-eval`). Trunk:
Qwen2.5-0.5B-Instruct, frozen. One architectural change from the toy forced
by the 150k vocabulary: the corrector emits a zero-init *hidden-state delta*
added before the trunk's own lm_head instead of a logit delta — the same
structural floor at far fewer parameters.

- Harvest: 2,000 GSM8K train problems, greedy CoT → 290 correct traces kept
  (14.5% under strict format scoring).
- Train: tap layer 12/24, d_cfc 512 (~2.9M trainable params, <0.6% of trunk),
  3,000 teacher-forced steps, ~7 minutes on the RTX 3090.

**Baseline audit (2026-07-15).** The first eval scored by strict `####`
extraction and showed CoT 0.105 / latent 0.405 — an apparent +30-point gain.
Lenient scoring (fall back to the last number, applied identically to all
systems; completions dumped for inspection) revealed the trunk's true CoT
accuracy is 0.455: the strict number was a format artifact and most of the
apparent gain was learned format compliance. Full audited comparison
(n=200, `reports/20260715-retrofit-qwen05b-eval-audited.md` and
`reports/20260715-retrofit-qwen05b-baseline-*.md`):

| system | strict | lenient | mean emitted tokens | trainable params |
| --- | --- | --- | --- | --- |
| direct | 0.035 | 0.035 | 7.4 | 0 |
| cot (zero-shot) | 0.105 | 0.455 | 276.4 | 0 |
| cot (4-shot) | 0.130 | 0.415 | 186.1 | 0 |
| self-consistency@8 | 0.515 | 0.515 | 2,344.3 | 0 |
| LoRA r=16 (same 290 traces) | 0.350 | 0.380 | 314.7 | 1.08M |
| latent corrector | 0.405 | **0.425** | **4.1** | 2.9M |

Honest verdict: at 0.5B with 290 traces, the corrector **matches** the
trunk's CoT accuracy (0.425 vs 0.455, within noise) at 67× fewer emitted
tokens (internal rollout 302.2, ~9% above CoT — emitted-channel savings, not
compute). It beats the same-data LoRA on both axes (LoRA *degrades* the trunk
to 0.380 while still emitting the full chain); SC@8 gains +6 points over
single-chain CoT at 570× the latent system's emitted tokens. The accuracy
*gain* from the toy setting did not transfer at this scale; the efficiency
claim transferred exactly. Paper §5 corrected accordingly.

Next levers before/while scaling: full 7.4k-problem harvest (≈3.4k traces
expected at 0.455 keep-rate with lenient harvest scoring), then the trunk
ladder (Qwen2.5-1.5B → 3B) to test the dose-response prediction.

## Trunk ladder: Qwen2.5-1.5B-Instruct (2026-07-15)

Same recipe one scale up: tap layer 14/28, d_cfc 512 (3.15M params, ~0.2% of
trunk), 423 traces harvested from 2,000 train problems (21.2% strict keep vs
14.5% at 0.5B), 3,000 steps (~9.4 min). Two additions: **latent
self-consistency@k** (temperature-sampled internal rollouts, majority vote on
lenient answers, only the winning answer emitted; `retrofit-eval
--latent-samples 8 --temperature 0.6`, commit dd16308) and a visible SC@8
comparator at 1.5B.

n=200 results (`reports/20260715-retrofit-qwen15b-eval.md`,
`reports/20260715-retrofit-qwen15b-baseline-sc8.md`; 0.5B latent-SC@8 in
`reports/20260715-retrofit-qwen05b-latent-sc8.md`):

| system | strict | lenient | mean emitted tokens |
| --- | --- | --- | --- |
| direct | 0.000 | 0.065 | 2.7 |
| cot (zero-shot) | 0.215 | 0.650 | 254.0 |
| visible self-consistency@8 | 0.755 | 0.755 | 2,053.0 |
| latent corrector | 0.715 | **0.715** | **4.4** |
| latent self-consistency@8 | 0.775 | **0.775** | **4.4** |

Verdict: the accuracy gain reappears at scale. Latent beats visible CoT by
+6.5 points (0.715 vs 0.650) at 58× fewer emitted tokens, and latent SC@8
**beats visible SC@8** (0.775 vs 0.755) at 467× fewer emitted tokens — the
user-facing goal ("match SC@8 accuracy") is met and exceeded at 1.5B. At
0.5B latent SC@8 reached 0.450 (vs visible 0.515): it narrows but does not
close the gap at that scale, consistent with dose-response. Strict = lenient
for all latent rows (formatting fully internalized). Next rung: 3B.

## Trunk ladder: Qwen2.5-3B-Instruct (2026-07-15)

Third rung: tap layer 18/36, d_cfc 512 (3.68M params, ~0.12% of trunk),
1,605 traces, 3,000 steps (loss 0.120; training survived three 0xC0000005
driver crashes via the step-snapshot resume added in 6b59fe4, and eval
survived three more via the completions-dump resume in 76e50a0).

n=200 results (`reports/20260715-retrofit-qwen3b-eval.md`,
`reports/20260715-retrofit-qwen3b-baseline-sc8.md`):

| system | strict | lenient | mean emitted tokens |
| --- | --- | --- | --- |
| direct | 0.075 | 0.090 | 7.1 |
| cot (zero-shot) | 0.785 | 0.815 | 299.8 |
| visible self-consistency@8 | 0.860 | 0.860 | 2,473.1 |
| latent corrector | 0.800 | **0.805** | **4.2** |
| latent self-consistency@8 | 0.885 | **0.885** | **4.4** |

Verdict: latent SC@8 **beats visible SC@8 again** (0.885 vs 0.860) at 562×
fewer emitted tokens — the inversion holds at two consecutive rungs.
Single-rollout latent compresses to CoT parity (0.805 vs 0.815) as the trunk
nears task ceiling; the latent vote is more productive per rollout than the
visible vote (+8.0 vs +4.5 points over the respective single-chain systems).
Ladder summary (latent-SC@8 vs visible-SC@8, lenient): 0.5B 0.450 vs 0.515;
1.5B 0.775 vs 0.755; 3B 0.885 vs 0.860.

Paper draft: `papers/20260715-latent-cot-frozen-trunk-correction.md`.

## Internal-channel compute: adaptive voting, divergence probe, rollback (2026-07-15/16)

Three results from the divergence-gating branch, all over the n=200 dumps
(paper §5.4):

**Stop-on-agreement voting** (`rollback-simulate`, and as a real decode mode
`retrofit-eval --stop-agree N`): sample latent rollouts sequentially, stop
when N agree. Accuracy holds at 30–46% fewer rollouts at every scale — 0.5B
stop-at-2 0.450 @ 4.33 (= full@8); 1.5B stop-at-4 0.785 @ 5.57 (*above* full
0.775); 3B stop-at-4 0.885 @ 4.76 (= full), stop-at-2 0.870 @ 2.38. Rollouts
needed shrink with scale.

**Divergence recognition probe** (`divergence-probe`): linear probe on
(h_tap, s_t), wrong rollout vs correct sibling. At the *sampling fork*: null
(AUC ≤ 0.56, 1.5B, all offsets to +64). At the *arithmetic error site*
(first computed value not sanctioned by gold `<<>>` annotations, median +458
chars past the fork): AUC **0.985** at +1 token, 0.967 at +4, 0.840 at +8,
chance by +32. Signal is strong, local, and h_tap ≥ s_t — the trunk knows
the moment it errs; a trigger must fire within ~8 tokens.

**Oracle rollback** (`rollback-oracle`, 1.5B n=200 budget 4): rewind wrong
greedy rollouts to 8 tokens before the error site, warm-start corrector over
the shared prefix, resample. Greedy 0.715 → ceiling **0.750** / deployable
detector-accept rule 0.725, at 0.96 re-rolls and **+161 internal tokens per
problem** vs latent-SC@8's +1,830 for 0.775 — 58% of the vote's gain at 9%
of its overhead. Detector conservatism (2 accepts) is the gap to the
ceiling; the 0.985-AUC probe is the obvious replacement trigger.

Reports: `reports/20260715-adaptive-sc-qwen*.md`,
`reports/20260715-divergence-probe-qwen15b*.md`,
`reports/20260715-rollback-oracle-qwen15b.md`.

## Task-transfer probe results (2026-07-16) — was priority 2

Design as planned: one toy trunk pretrained on a 50/50 mix of two task
families (`transfer-base-both-20260716-163553`, dim 256 / 8 layers, 10k
steps, val ppl 2.78), then two correctors distilled over the SAME frozen
trunk — one on arithmetic chains only, one on a new digit-rewrite family only
(ops: reverse digits, nines-complement, increment-mod-10 on a 0–99 state,
same vocab and rendering). Each corrector evaluated zero-shot on the *other*
family (256 held-out problems per cell):

| corrector trained on | eval family | direct | CoT | latent | emitted (CoT → latent) |
| --- | --- | --- | --- | --- | --- |
| arithmetic | arithmetic (in-family) | 0.027 | 0.777 | **0.977** | 18.8 → 3.9 |
| arithmetic | rewrite (**transfer**) | 0.238 | 1.000 | **0.949** | 18.7 → 3.9 |
| rewrite | rewrite (in-family) | 0.238 | 1.000 | **1.000** | 18.7 → 3.9 |
| rewrite | arithmetic (**transfer**) | 0.027 | 0.777 | 0.316 | 18.8 → 3.9 |

Transfer is **asymmetric, and the asymmetry is informative**. The
arithmetic-trained corrector transfers zero-shot to rewrite at 0.949 — within
5 points of the CoT ceiling on a family it never saw, at the same 5× token
savings. The rewrite-trained corrector reaches only 0.316 on arithmetic
(well above the 0.027 direct floor, far below the 0.777 CoT reference), and
degrades with chain length (0.605 at length 2 → 0.195 at length 5).

The asymmetry lines up with the dose-response law: on rewrite the trunk's CoT
is *perfect* (1.000), so the rewrite corrector's distillation data contained
essentially no trunk errors — it learned rollout internalization but was
never shown repair. The arithmetic corrector trained where the trunk errs
22% of the time, i.e. on abundant repair demonstrations, and that repair
skill carried across families intact. Conclusion: the repair function is
general once learned, but it is only learnable from an error-rich training
family. For scaling this is good news with a caveat — harvest corrector
training data from tasks the trunk finds hard, not easy.

Configs: `configs/transfer_base_pretrain.yaml`,
`configs/transfer_j_cfc_{arith,rewrite}.yaml`. Reports:
`reports/20260716-task-transfer-*.md`.

## Full-harvest retrain at 0.5B (2026-07-16): the H4 failure was data starvation

Scaled the 0.5B GSM8K harvest from 2,000 to all 7,473 training problems →
1,078 kept traces (14.4% keep, 3.7× the original 290). Retrained the
corrector with the identical recipe (tap 12, d_cfc 512, 3,000 steps — data is
the only variable; final loss 0.124 vs 0.131). Eval n=200, lenient scoring:

| system | 290 traces | 1,078 traces | visible baseline |
| --- | --- | --- | --- |
| latent (greedy) | 0.425 | **0.460** | CoT 0.455 |
| latent SC@8 (temp 0.6) | 0.450 | **0.580** | SC@8 0.515 |

Latent SC@8 +13 points from 3.7× data, now beating visible SC@8 at 0.5B too —
the one scale where H4 had failed. All three ladder rungs now show latent
SC@8 > visible SC@8 (0.580/0.515, 0.775/0.755, 0.885/0.860). Emitted 4.6 vs
2,344 tokens (~510×); internal rollout cost ≈ compute parity with visible
SC@8 (2,360 vs 2,344 tokens). Consistent with the task-transfer finding:
repair is learned from error-rich data, and a 0.5B trunk supplies plenty.
Artifacts: `outputs/retrofit-qwen05b/{traces-7k.jsonl,corrector-7k/}`,
report `reports/20260716-retrofit-qwen05b-corrector7k.md`.

## Batched latent rollouts (2026-07-16)

Self-consistency rollouts share one prompt and one frozen trunk, so all k
samples (or adaptive stop-on-agreement waves) now decode as a single batch
with per-sequence corrector state and EOS masking
(`_generate_batch_with_corrector`, default; `--sequential-rollouts` to
disable). Validated accuracy-identical to sequential on a matched slice
(every system equal, including SC@8), with wall-clock 387s → 132s at n=10
SC@8 on the 3090 — **2.9× end-to-end, ~3.4× on the voting portion**. Makes
the 1.5B/3B full-harvest evals and future SC@8 sweeps proportionally
cheaper. Also added `--problem-offset` for disjoint eval slices (dump rows
carry absolute indices).

## Probe-triggered rollback, deployable rule (2026-07-16) — negative

`rollback-probe` CLI: train the h_tap error-site probe with *deployable*
supervision only (sibling pairs from problems whose greedy rollout is
correct — problem-disjoint from all rollback targets, no gold labels at
inference), calibrate a rollout-level max-score threshold at 5% FPR, scan
every greedy rollout, rewind (margin 8) + re-roll (budget 4) on first
trigger, accept a candidate only if its own rescan is clean. 1.5B, n=200:

| metric | value |
| --- | --- |
| baseline greedy | 0.715 |
| **final** | **0.705** |
| recall on wrong rollouts | 8/57 |
| false alarms on correct | 6/143 |
| flips up / down | 0 / 2 |
| re-roll tokens per problem | 33.9 |

The 0.985 error-site AUC does not survive the max-over-positions scan: a
threshold at 5% rollout-level FPR sits above most true error scores, so
recall collapses while residual false alarms flip correct answers. Gap to
the oracle (0.750) and annotation rule (0.725) is a calibration/multiplicity
problem, not signal absence. Next candidates: sequential-testing rules over
scan length, or probes trained against in-context negatives instead of
sibling contrasts. Report: `reports/20260716-rollback-probe-qwen15b.md`.

## Insurance replication: corrector-7k on a disjoint slice (2026-07-16)

Same 0.5B corrector-7k, problems 200–399 (untouched by any tuning): latent
greedy 0.435, latent SC@8 **0.555** vs that slice's visible CoT 0.465 —
within noise of the primary slice's 0.580 at n=200. The data-scaling claim
holds out-of-slice. Report:
`reports/20260716-retrofit-qwen05b-corrector7k-slice2.md`.

## Steering injection: repair without rewinding (2026-07-16)

`steer-inject` CLI (user-proposed alternative to rollback, "wait-"-style):
contrastive vector v = mean(h_tap[correct sibling] − h_tap[wrong]) at
error-site offsets +1..+4 (deployable supervision), added as α·v to the tap
layer's residual stream for 24 decode steps after the trigger; erroneous
tokens stay in context, greedy regeneration makes α=0 an exact-reproduction
control (42/48 exact, baseline bit-equal). Oracle trigger, 1.5B, n=200:

| α | accuracy | flips up/down | regen tokens/problem |
| --- | --- | --- | --- |
| 0 | 0.715 | 0/0 | 36.6 |
| 1 | 0.720 | 1/0 | 36.2 |
| 2 | 0.730 | 3/0 | 43.2 |
| 4 | **0.735** | 4/0 | 51.2 |
| 8 | 0.715 | 0/0 | 40.8 |

Dose-response peaks at α=4: 57% of the oracle-rollback gap (0.750) at a
third of its token overhead (+51 vs +161), beating the annotation-detector
rollback rule (0.725); α=8 overshoots (0 flips either way). Deployable
probe trigger at relaxed 20% FPR: recall 11/57 (up from 8/57 at 5%) but
false-alarm damage scales with α — 0.705 at α=1 (3 of 22 false alarms
flip correct→wrong), 0.680 at α=2 (9 flips down). The trigger, not the
repair mechanism, is the binding constraint for both rollback and
steering.
Reports: `reports/20260716-steer-oracle-qwen15b.md`,
`reports/20260716-steer-oracle-a8-qwen15b.md`,
`reports/20260716-steer-probe-qwen15b.md`.

## Trigger lab + repair-decay sweep (2026-07-16)

`trigger-lab` offline bake-off on the 1.5B greedy dump: 3 landmark
directions (sibling probe, in-context-negative probe, steering vector v as
detector) × 5 decision rules (raw threshold, per-rollout selfnorm, 8-token
one-sided CUSUM, periodic audit16/audit32), thresholds calibrated to
rollout-level FPR 5/10/20% on correct rollouts (max-over-positions
statistic). Winner: **probe-sibling + CUSUM** — recall 13/19/**30** of 57
at 5/10/20% FPR (raw point rule: 8/11/16), median trigger delay 6 tokens.
Deviation evidence is weak per-token but persistent; integration defeats
the scan-length multiplicity that sank the raw threshold. In-context
negatives help the raw rule (15, 20/57 at 10/20%) but don't compose with
CUSUM; v is a mediocre detector. Periodic audits underperform dense scans
at matched FPR (best 18/57 at 20%) with 10–15-token delays.
Repair-decay (oracle α=4, inject offset 4/16/32/64 past error site):
0.735 / 0.720 / 0.725 / 0.720 — repair window ≈ 4–16 tokens; late
injection never harms (0 down-flips) but keeps only ~¼ of the effect.
Composite verdict: pure periodic auditing sacrifices most repair value to
its delay; CUSUM's 6-token delay sits inside the repair window —
sequential evidence accumulation is the deployable trigger design.
Reports: `reports/20260716-trigger-lab-qwen15b.md`,
`reports/20260716-steer-delay-{16,32,64}-qwen15b.md`.

## End-to-end CUSUM steering + 1.5B full-harvest retrain (2026-07-17)

CUSUM trigger wired into `steer-inject` (`--trigger cusum`) and run
end-to-end at FPR 10/20%, alphas 0/2/4. Recall arrives as promised (20–21
of 57 wrong rollouts triggered vs 8–11 for point rules; alpha=0 control
bit-reproduces baseline) but the system stays net-negative: 0.705/0.700
(fpr10 a2/a4), 0.700/0.700 (fpr20) vs 0.715 baseline. Arithmetic: ~28%
base error rate + 10–20% rollout FPR ≈ one false alarm per hit (19–24 vs
20–21), and steered-correct rollouts flip down at ~25% vs steered-wrong
flipping up at ~15%. The binding constraint moved from recall to
*precision*; candidate routes: confidence-gated alpha, second-opinion
check before injection, or offline audit-and-repair where localization is
given. Reports: `reports/20260717-steer-cusum-fpr{10,20}-qwen15b.md`.

1.5B data-scaling replication (full 7,473-problem harvest → 1,552 traces,
20.8% keep, 3.7× data; identical recipe): greedy 0.715 → **0.725**, latent
SC@8 0.775 → **0.790** (visible SC@8 0.755; strict = lenient). Smaller
gains than 0.5B's +13 (diminishing exponent in trunk competence), but
full-split harvesting is cheap (56 min batched) and worth default. Paper
abstract/§5.2/§5.4/scorecard updated. Checkpoint:
`outputs/retrofit-qwen15b/corrector-7k/`. Report:
`reports/20260717-retrofit-qwen15b-corrector7k.md`.

3B data-scaling null completes the law (2026-07-17): full harvest gives
5,931 traces (79.4% keep, 3.7× data) but SC@8 lands 0.870 vs v1's 0.885
(−1.5 ≈ 3 problems, noise; greedy 0.815 = CoT parity unchanged). The data
exponent decays to zero as keep-rate rises (14.4% → 20.8% → 79.4%): extra
traces teach repair, and a 79%-correct trunk supplies almost no errors to
learn from — §4.5's error-rich prescription measured as a scaling law. At
the strong end the constraint is error diversity, not trace count.
Paper §5.3 + H4 updated. Report:
`reports/20260717-retrofit-qwen3b-corrector7k.md`.

Stop-on-agreement replication on the full-harvest dumps: 1.5B corrector-7k
stop-at-3 = **0.800 at 4.4 mean rollouts** (best 1.5B number yet, +1.0 over
its own full vote 0.790@8, −45% rollouts); 3B corrector-7k stop-at-2 =
0.870 at 2.4 (= full vote, −70%). Reports:
`reports/20260717-adaptive-qwen{15b,3b}-corrector7k.md`.

## State-space consensus probe (2026-07-17)

`consensus-probe` CLI: dispersion = 1 − mean pairwise cosine of h_tap
mean-pooled over each vote rollout's first t tokens, across the 8 SC
rollouts per problem. Question: is the vote outcome legible in h-space
before answers commit? Answer: only on the vote's own schedule. AUC for
final answer-disagreement: chance at t≤16, 0.58–0.61 at t=32, **0.75
(1.5B) / 0.73 (3B) at t=128** (~half the rollout); vote-wrong AUC tracks
lower (0.67/0.62). Split problems run ~25–30% higher dispersion at t≥64,
but there is no early window — divergence accumulates with the computation
(matches the fork-probe: diverging ≠ erring). Answer-level agreement stays
the right adaptive gate; dispersion is at best a mid-rollout prior for
allocating extra samples. Reports:
`reports/20260717-consensus-qwen{15b,3b}.md`.

## Margin-gated steering (2026-07-17)

`steer-inject --gate margin`: alpha scaled per rollout by
min(1, (CUSUM max − threshold)/gate_scale), gate_scale = calibration q95 −
threshold — fully deployment-legal. The margin separates populations (mean
gate 0.82–0.83 hits vs 0.60–0.68 false alarms) and flips the sign of the
end-to-end result: fpr10 a2 = **0.720** (ungated 0.705; baseline 0.715),
fpr20 a2 = 0.715 (ungated 0.700); a4 remains slightly negative. +0.5 pt ≈
1 problem at n=200 — honest read: gating recovers essentially all
false-alarm damage but leaves little net repair; the gate raises the
floor, landmark separation must raise the ceiling. Paper §5.4 + H12
updated. Reports: `reports/20260717-steer-gated-fpr{10,20}-qwen15b.md`.

## Closed-form linear corrector (2026-07-16) — geometric target collapses

`retrofit-fit-linear` (roadmap item 9 on-ramp): delta = W·h_tap + b fit
analytically on the 0.5B 7.4k-harvest traces with the v2 geometric target
D_t = τ·ê(y_t) − h_final. Ridge and Procrustes both collapse generation
totally (latent 0.000, rollout never terminates). Diagnosis (after user
pushback, correct): the target demands ~‖h‖-scale deltas even where the
trunk is already right, injected ungated at every step — a target artifact,
not a fair test of closed-form linear capacity. Sharpened v2 lesson: the
loss family, not linearity, is the failure. Follow-up complete: the zero-
anchored CE-gradient target (D_t = e_y − E_p[e], identically zero on the
correct manifold) holds the CoT floor exactly at every inference scale
(lenient 0.455/0.455/0.450 at scale 0.25/1/4 vs CoT 0.455; internal rollout
length 276 ≈ visible chain) and adds zero repair. Verdict for roadmap item
9: a training-free linear re-basis captures none of the corrector's gain —
safe targets are inert, aggressive ones destructive; repair lives in the
learned recurrent computation. SC@8 control confirms: sampling the inert
map 8× gives 0.520 ≈ visible SC@8 (0.515), far below the trained
corrector's 0.580 — the SC gain is learned repair, not the latent scaffold.
Reports:
`reports/20260716-linear-{ridge,procrustes}-qwen05b.md`,
`reports/20260716-linear-grad-{0p25,1p0,4p0}-qwen05b.md`,
`reports/20260716-linear-grad-1p0-sc8-qwen05b.md`.

## Grounded continuous latent reasoning (2026-07-17)

User idea: "periodic (every 10 steps or so) full rollouts could keep it
grounded" + redesign: natural termination via shadow-token monitoring, no
scheduled anchor. `retrofit-grounded`: continuous steps feed back the
expected embedding E_p[e_v] (convex hull of vocab embeddings); every G-th
step decodes a real greedy token; shadow argmax stream watched for `####`
(natural = on grounded step / detected = mid-continuous, anchor injected /
budget = 400-step cap). 200 problems, corrector-7k, no retraining.

1.5B expected: G=1 **0.725** (193 nat) / G=4 0.665 / G=8 0.630 / G=16
0.625 / never **0.590** (176 detected, 24 budget). 0.5B expected: 0.450 /
0.390 / 0.305 / 0.350 / 0.320. 0.5B hidden-state feedback (norm-matched
h_final): **collapse** — 0.005 at G=8, 0.000 pure-continuous, 70–91%
budget truncation.

Readings: (1) monotone dose-response at 1.5B; pure continuous keeps 81% of
control zero-shot, G=4 recovers ~half the gap — real tokens repair
off-manifold drift; (2) hidden-feedback collapse = v2 lesson re-derived at
inference: the convex hull is the load-bearing constraint; (3) answer
onset legible in the shadow stream in 79–88% of pure-continuous rollouts →
iteration 2 = J-space answer-onset landmark probe as injection trigger
(supervision free from existing dumps). Not a compute win (one forward per
continuous step) and continuous steps aren't token-meterable; G is the
knob trading channel richness vs manifold fidelity. Paper §5.5 + H13.
Reports: `reports/20260717-grounded-expected-qwen{05,15}b.md`,
`reports/20260717-grounded-hidden-qwen05b.md`.

## Optical context compression, iteration 1 (2026-07-17) — plumbing pass, benchmark confounded

Roadmap item 8 on-ramp (`optical-resume`, Qwen2.5-VL-3B-Instruct, 100
GSM8K problems): model generates its own CoT; first 60% handed back as
text (upper bound) / rendered image at scale 1.0/0.75/0.5/0.35 (pillow
monospace render, downscaled) / omitted (lower bound); model resumes to a
final answer. Failure first: initial launch died on missing torchvision
(Qwen2VL video sub-processor requires it); installed 0.21.0+cu124,
clean rerun.

| arm | acc | image tokens | compression |
| --- | --- | --- | --- |
| text_resume | 0.770 | — | — |
| no_prefix | 0.760 | — | — |
| optical_1.0 | 0.760 | 210 | 0.63× |
| optical_0.75 | 0.760 | 123 | 1.07× |
| optical_0.5 | 0.720 | 55 | 2.39× |
| optical_0.35 | 0.740 | 27 | 4.87× |

Two readings. (1) Plumbing validated: 4.87× token compression with no
accuracy cliff — 27 image tokens carry a 131-token prefix at 0.740 vs
0.770 text. (2) BUT the benchmark has no dynamic range: no_prefix scores
0.760 — GSM8K problems are self-contained, the model just re-solves from
scratch, so "read the fuzzy image" and "ignored it and re-derived" are
indistinguishable. All arms sit within ±2.5 pts of each other. Iteration
2 must render *non-recoverable* content — the problem statement itself
(no_prefix floor collapses toward 0), or injected intermediate values not
derivable from the visible question — so the compression sweep measures
actual optical reading. Only then is the corrector-reads-compressed-form
experiment (roadmap 8 proper) worth wiring. Report:
`reports/20260717-optical-resume-qwen3bvl.md` (smoke:
`reports/smoke-optical-qwen3bvl.md`).

## Cloud MATH harvest at 32B (2026-07-18) — the scalability rung begins

QwQ-32B trace harvest moved to a GCP spot A100-40GB
(`a2-highgpu-1g`, ~$1.2/h) after local 3090 estimates came in at 30+
hours: vLLM 0.25.1 serving QwQ-32B-AWQ (4-bit), MATH-lighteval train
split filtered to numeric-boxed problems, temperature 0.6, 4096-token
budget, keep-if-correct. Three bootstrap failures, each environmental:
(1) the DLVM image's system torchaudio (CUDA 12.9) clashed with pip
vLLM's torch — uninstall it; (2,3) flashinfer JIT needs `ninja` on the
nohup PATH — `pip install --user` lands in `~/.local/bin` which nohup
never sees; `apt-get install ninja-build` fixed it. Clean run: **4,850
problems → 3,247 kept traces (67.0%)** in ~4.5 h wall (~$6). Keep-rate
drifted 87% → 44–58% across the dataset's difficulty ordering. Trace
lengths (QwQ tokenizer, n=701 sample): p50 1,870, p90 3,522, max 4,347 —
96.9% fit the 4,096 training window. Quantization tiers are deliberately
mismatched-but-close: AWQ 4-bit generated the traces, NF4 4-bit is the
training/eval trunk; both sit in the same 4-bit tier, flagged in the
paper. The 3B lesson (79% keep → null gains) says 67% is workable but
error diversity, not volume, is the risk to watch.

Enablement shipped for the 32B rung: MATH-test eval branch
(3,198 numeric-boxed problems), NF4 trunk loading behind `--quantize`
for train and eval, truncated BPTT (`--bptt-chunk`, CfC state detached
across chunks; step-0 loss bit-identical to full-sequence unroll), and a
`\boxed` answer-region fallback in `_answer_start_index`. Corrector
training launched on the same A100 (tap 32 of 64, d_cfc 512, 3,000
steps, 4,096-token window, chunk 512): 23.5 GB of 40 GB used, no OOM.
One portability fix en route: `datetime.UTC` is 3.11+; the VM's Python
3.10 needed `timezone.utc`.

## Direct FLOPs accounting (2026-07-18) — sidecar cost is noise

User concern: does the sidecar's per-token cost eat the latent channel's
token savings? `retrofit-flops` answers with exact 2×MAC counts
(GQA-aware attention, gated MLP, lm_head, plus context-dependent
attention at measured mean context) priced against per-system token
counts measured from existing eval dumps. Verdict: the corrector is a
fixed ~6.3 MFLOPs/token, which is **0.49% / 0.20% / 0.12%** of trunk
decode cost at 0.5B/1.5B/3B — and <0.03% projected at 32B. Latent greedy
lands at 0.97–1.04× CoT end-to-end FLOPs; multi-rollout modes cost what
their rollout counts say. The token-reduction story survives contact
with the FLOPs ledger. Paper §5.4 rewritten around the direct
accounting. Reports: `reports/20260718-flops-qwen{05,15,3}b.md`.

## Tap-position ablation at 0.5B (2026-07-18) — the tap differentiates the vote, not the greedy rollout

User request: ablate the tap at the extremes (layer 0 = embeddings,
layer 24 = final post-norm; L−1 = layer 23 queued). Same recipe as the
7k-harvest corrector (1,078 traces, 3,000 steps, ~7.5 min train each);
eval on the standard 200-problem slice with SC@8:

| tap | latent greedy (lenient) | latent SC@8 |
| --- | --- | --- |
| 0 | 0.450 | 0.545 |
| 12 (default) | 0.460 | 0.580 |
| 24 | 0.465 | 0.535 |

Two findings. (1) The toy's embedding-tap collapse does **not** reproduce
at retrofit scale on the greedy rollout: all three taps sit within noise
of the 0.455 CoT floor — the pretrained trunk's own depth supplies what
the shallow toy trunk could not, and the zero-anchored floor dominates
the greedy number. (2) The tap position *does* matter for the sampled
channel: mid-stack SC@8 beats both extremes by 3.5–4.5 points. The
repair signal that shapes the sampling distribution is a mid-stack
property — too early and the error hasn't been computed, too late and
the unembedding basis has committed — consistent with the error-site
probe geometry (h_tap strongest mid-stack). Paper §5.1 updated with the
ablation table. Reports: `reports/20260718-tap{0,24}-ablation-qwen05b.md`.
tap23 (L−1) and the recurrent-cell zoo sweep (GRU, linear RNN, S4D-real,
Mamba/S6, Mamba-2/SSD at tap 12) run next in queue14.

## J-space verification via Jacobian projection (2026-07-18) — the tap is exact; the corrector reads complementary directions

User request: "I want to be certain that we're actually tapping J-space.
Can we feed our hidden states through a Jacobian projection matrix or
something similar?" New tool: `prometheus.cli retrofit-jspace-verify`
(staged re-run of the upper trunk with a leaf tensor at the tap +
vector-Jacobian products of h_final w.r.t. h_tap). Run at 0.5B, tap 12,
8 traces, 252 sampled positions, ~2 min on CPU. Report:
`reports/20260718-jspace-verify-qwen05b.md`. Three results:

1. **Tap authenticity: bit-exact.** Replaying layers 12–23 + final norm
   from the tapped tensor reproduces the reference h_final with 0.00e+00
   max relative error and 100% logit-argmax agreement (2,933 positions).
   The corrector's delta provably enters the trunk's own computation.
2. **Influence geometry: low-rank and heavily attention-mediated.** The
   local Jacobian block's row space has effective rank 142 of d=896
   (rank-64 basis captures 56% of energy), and **61% of the tap's total
   influence mass flows through other positions** via keys/values.
3. **Corrector alignment: barely above chance against both bases.**
   Corrector read-direction energy in the top-64 influence basis is
   0.087 (local) / 0.084 (attention-inclusive) vs a 0.071 random
   baseline, while the tap states themselves sit at 0.155/0.178. The
   corrector uses mid-stack information (tap ablation above) but reads
   directions largely complementary to the trunk's dominant Jacobian
   subspace — geometric evidence for "error detector" over "shadow
   re-computer" (a shadow re-computer would mimic the downstream
   readout and align with J). Paper: new §6.4.

Follow-up (same day, user terminology challenge): in formal mech-interp
usage "J-space" means concept-aligned *directions*, not the raw residual
stream — the paper must not claim to "tap J-space." Response: (a) paper
terminology corrected everywhere (now "layer tap" / "residual-stream
tap", with an explicit terminology note in §3.2); (b) new **projection
ablation** demonstrates the raw-stream tap is *preferable* because the
corrector monitors the orthogonal error manifold. Feeding the corrector
keep/remove projections of its input (trunk stream untouched), scored on
the 89/1,923 positions where the correction changes the argmax token:

| corrector input | tap energy kept | delta cos | active-token agreement |
| --- | --- | --- | --- |
| keep top-64 (local / attn-incl / random-64) | 0.157 / 0.180 / 0.067 | 0.71 / 0.73 / 0.73 | 0.506 / 0.483 / 0.483 |
| remove top-64 (local / attn-incl / random-64) | 0.843 / 0.820 / 0.933 | 0.96 / 0.95 / 0.99 | 0.888 / 0.876 / 0.899 |

Removing the ENTIRE dominant influence subspace barely dents the
corrector (87.6% ≈ random-removal control 89.9%); keeping ONLY it is no
better than keeping a random subspace (0.483 = 0.483) despite 2.7×
the energy. Top influence directions are neither necessary nor
privileged; the signal is broadly distributed in the complement. A
concept-aligned (Jacobian-isolated) interface would discard exactly what
the corrector monitors. Paper §6.4 rewritten around this argument.
Tool: --basis-out saves the bases (outputs/retrofit-qwen05b/
jspace-basis-rank64.pt) for future behavioral projection evals.



## Recurrent-core zoo at 0.5B (2026-07-18) � the CfC is load-bearing

Queue14 completed the tap sweep (layer 23 added: greedy 0.470 / SC@8
0.550, confirming a gradual falloff toward the late-stack taps rather
than a post-norm artifact) and the recurrent-core ablation: identical
recipe (tap 12, d=512, 1,078 traces, 3,000 steps), CfC swapped for
gru / diagonal linear RNN / diagonal SSM / minimal Mamba / minimal
Mamba-2, sharing phi_in and the zero-init delta head.

| core | final loss | greedy | SC@8 |
| --- | --- | --- | --- |
| CfC | 0.124 | 0.460 | 0.580 |
| GRU | 0.119 | 0.400 | 0.495 |
| linear RNN | 0.107 | 0.435 | 0.515 |
| diagonal SSM | 0.115 | 0.445 | 0.515 |
| Mamba | NaN @ step 175 | � | � |
| Mamba-2 | NaN @ step 2,575 | � | � |

Findings: (1) no alternative recovers the CfC's sampled-channel gain �
linear/SSM SC@8 land exactly on the visible SC@8 baseline (0.515),
i.e. zero added vote diversity; (2) the GRU trains *below* the CoT
floor (0.400 vs 0.455) � the zero-init floor holds at init, not
against destructive learned dynamics; (3) distillation loss
anti-correlates with downstream repair (linear RNN best loss, worse
everywhere), and both selective SSMs diverge to NaN without the loss
signalling trouble first (mamba2 was at a healthy 0.165 at step 2,550).
Operational note: a NaN checkpoint crashes sampled eval with a CUDA
device-side assert in torch.multinomial � softmax of NaN logits � which
is how the divergence was first noticed. Paper: �5.1 tap table extended
+ new recurrent-core subsection.

## Snap projector results (2026-07-19) - drift is partially reversible, geometry is not

Queue15: snap = residual MLP (zero-init output) on the continuous
feedback path, trained by one-step denoising distillation (feedback
vector at position t of a correct trace -> embedding of the actual
next token; temperature augmentation). Dose-response re-run:

- 0.5B expected: G1 0.450 (=control), G8 0.305->0.360, G16
  0.350->0.370, G0 0.320->0.345.
- 1.5B expected: G1 0.725 (=control), G8 0.630->0.655, G16
  0.625->0.630, G0 0.590->0.670 - snap recovers 59% of the
  pure-continuous gap; G0+snap now BEATS sparse grounding (snap
  corrects every step, grounding only every G-th).
- 0.5B hidden+snap: 0.020/0.015 - still collapsed; a local learned
  projection repairs blur near the manifold but cannot move states
  from a different geometry onto it.
Paper: section 8 snap paragraph replaced with results table.

## Trained-projection 2x2 (2026-07-19) - complement > dominant, full stream still best

Queue16 (user question: is tapping exclusively the complement
worthwhile?): correctors trained from scratch with input_proj baked in
(rank-64 influence basis from jspace-verify):

| input | greedy | SC@8 |
| --- | --- | --- |
| full stream | 0.460 | 0.580 |
| complement only (remove) | 0.440 | 0.525 |
| dominant only (keep) | 0.420 | 0.495 |

Ordering matches the frozen-corrector projection ablation (complement
carries most of the function, dominant-only is worst and below the CoT
floor on SC), but full stream is strictly best: train-time restriction
forfeits real signal (-5.5 pts SC@8) even though inference-time removal
barely dented the full-trained corrector. Answer to the user question:
complement-only training is NOT worthwhile as a deployment choice; the
finding is diagnostic. Design rule in strongest form: read everything -
the repair signal is spread across subspaces and the full stream costs
nothing. Paper: section 6.4 causal follow-up paragraph.

## Corrector quorum (2026-07-19) - replicas help greedy, seeds do not average

Queue17 (user idea: SC@k at the corrector). k members vote on the
delta each step; replica mode (noise 5% of tap std, member 0 clean,
per-member recurrent state) vs 3 seed-trained members; agg mean vs
sign vote (>=75% coordinate sign agreement, else 0):

| ensemble | greedy | SC@8 |
| --- | --- | --- |
| single (baseline) | 0.460 | 0.580 |
| 4 replicas mean | 0.480 | 0.560 |
| 4 replicas sign | 0.480 | 0.555 |
| 3 seeds mean | 0.430 | 0.525 |
| 3 seeds sign | 0.420 | 0.515 |

Findings: (1) replica quorum +2 pts greedy at ~zero FLOPs (corrector
is ~1e-3 of trunk); (2) same smoothing DAMPENS the rollout vote
(0.580->0.560) - the two ensembling levels are substitutes, not
complements; (3) seed ensembles are destructive: each seed learns its
own repair basis, deltas do not average across bases (echoes 6.4:
repair function distributed, basis-dependent, nonidentifiable).
Paper: section 5.4 "Self-consistency at the corrector".

Follow-up in flight (queue18): DYNAMIC SC - user synthesis: use the
corrector signal as a branch trigger. Single greedy beam; at z-scored
||delta|| trigger sites the KV cache + corrector state fork and the
child takes the sibling token (parent's choice masked) then samples
on; cap 8 beams, majority vote; cooldown 16; floor = greedy rollout
verified bit-identical when no trigger fires. Sweep branch-z
1.5/2.5/3.5 running -> reports/20260719-dsc8-z{1p5,2p5,3p5}-qwen05b.md.
Baselines: greedy 0.460, fixed SC@8 0.580 @ 8.0 rollouts, stop-at-2
0.450 @ 4.33.

## Dynamic self-consistency (2026-07-19) - clean negative, same trigger ceiling

Queue18: user synthesis (branch rollouts at corrector-flagged sites,
collapse via majority vote) implemented as _generate_dynamic_sc
(greedy beam 0, per-beam Welford z on ||delta||, KV+state fork, child
takes sibling token then samples, cap 8, cooldown 16). Sweep at 0.5B:

| z | acc | mean beams | internal tok |
| --- | --- | --- | --- |
| greedy | 0.460 | 1.0 | 285 |
| 3.5 | 0.460 | 1.85 | 531 |
| 2.5 | 0.455 | 2.94 | 896 |
| 1.5 | 0.470 | 7.43 | 2280 |
| SC@8 | 0.580 | 8.0 | 2319 |

At matched cost (z=1.5) dynamic recovers 1 of the fixed vote's 12
points. Diagnosis: (a) trigger recall ceiling (23-35% at deployable
FPR, section 7.5) - most error sites never branch; (b) fundamental:
SC's value is decorrelated whole paths; branches share the greedy
prefix so the vote is over correlated rollouts. Third instance of the
same wall (steering, rollback, now branching): the correction signal
says THAT something went wrong, not WHERE the alternative path lives.
Paper: section 5.4 dynamic-SC subsection after the quorum one.

## Quorum scale check at 1.5B (2026-07-19) - both replica effects vanish

Queue19: replica-quorum(4, noise 0.05, mean) at 1.5B, 200 problems
(reports/20260719-quorum-replica-mean-qwen15b.md):

| system | 1.5B single | 1.5B quorum |
| --- | --- | --- |
| latent greedy | 0.715 | 0.725 |
| latent SC@8 | 0.790 | 0.795 |

Both within noise. The 0.5B +2pt greedy gain and -2pt sampled dampening
do not replicate: quorum smoothing only matters where the corrector
delta is noisy. Paper: closing paragraph of the 5.4 quorum subsection.

## Sandwich architecture (2026-07-19) - interface fixed, capacity wall exposed

Queue20: user idea - sandwich the CfC between the trunk's two discrete
halves. Lower half (embed->tap12) frontloads the tap state from the
discrete prompt; corrector chassis retrained as a tap-space dynamics
model (retrofit-train-dynamics: pred = h + delta targeting next tap
state, normalized MSE, teacher-forced); upper half (_upper_half_logits)
unembeds the latent trajectory into a discrete CoT once at the end;
token-space answer phase. Continuous states never touch the input
embedding (the section-8 v2-collapse interface). Decoder plumbing exact
(argmax agreement 1.0 on real taps).

Results (0.5B, 200 problems, 300 latent steps,
reports/20260719-sandwich-qwen05b.md):

| system | lenient acc | notes |
| --- | --- | --- |
| cot (same run) | 0.455 | |
| sandwich | 0.035 | = direct-answer floor |
| sec-8 pure continuous | 0.320 | trunk still reasons per step |

Training diagnostics (dynamics-7k/metrics.jsonl): teacher-forced loss
0.84 -> 0.26, openloop_cos@1 = 0.80, @8 = 0.37, @32 = 0.16. Decoded
chains legible for the first ~10 tokens ("To solve this problem, we...")
then dissolve into repetition - the interface failure is GONE, the
failure moved to the reasoner: a 13M cell cannot BE the transition
function of a 0.5B trunk, only nudge it. Natural follow-up licensed by
cos@1 = 0.80: interleave single latent steps between trunk forwards
(amortize every 2nd forward) instead of replacing the trunk wholesale.
Paper: closing subsection of section 8.

## Action items: intrusive-thoughts program (2026-07-19, user)

Framing: the Jacobian complement as "intrusive thoughts" - contending
wrong answers, functional during training (pruning pressure), problematic
at inference. Supported by the 6.4 result (complement-only corrector
beats dominant-only: the error signal rides in the complement).

- [x] AI-1: monitor intrusion rate during corrector training (complement
      excursion z>=2, tap + delta streams), test the boom-before-the-
      mini-explosion hypothesis against the loss curve. IMPLEMENTED:
      retrofit-train --monitor-basis. Note tap stats are trunk-fixed
      (stationary control); any boom must appear in the deltas.
      RESULT (queue21): boom refuted; corrector born complement-dwelling.
- [x] AI-3a (gate): complement-energy/frac as trigger-lab rules -
      does the intrusion signal beat the ~23-35% recall wall of
      probe/meandiff rules? IMPLEMENTED: trigger-lab --basis. Runs on
      existing 0.5B corrector-7k dump.
      RESULT (queue21): GATE PASSES - 0.444@20% FPR, fires 60-110 tokens
      BEFORE the error (first leading indicator). See section below.
- [x] AI-3b (gated on 3a): complement-fork adaptive SC@k - fork on
      intrusion excursion; SUPPRESS the complement spike in the root
      (project out at tap layer, steer-inject hook machinery) and let the
      intrusion play out in the offshoot; both greedy (decorrelation from
      the latent split, not sampling); vote or report-back gate at the
      end. Addresses both dsc failure coordinates (trigger signal +
      branch correlation).
      AMENDMENTS (2026-07-19, user): (1) offshoots must run to full
      completion before the feedback gate - the gate arbitrates finished
      testimony only, so the mainline captures the offshoot's full value
      with no uninformed computation (implemented: per-branch
      max_new_tokens post-birth budget, late-born children no longer
      truncated by the root's leftover budget); (2) the offshoot should
      SUPPRESS THE DOMINANT JACOBIAN component (-gamma x BB^T h at the
      tap) rather than reinforce the complement excursion - let the
      intrusive thought drive unencumbered by the mainline computation
      (implemented: --child-mode suppress-dominant, new default;
      reinforce kept as the comparison arm).
      RESULT (queue22/23): ceiling real in both arms (+6.5/+7.5 pts),
      no gate harvests it - see section below. Distillation route (AI-5)
      is the way to capture the trapped value.
- [ ] AI-2 (design): "let the intrusion play out and inform the primary"
      beyond voting - child rolls ahead N tokens and its trajectory
      summary feeds the root corrector as an extra input channel;
      requires training-time simulation pairs. Design only, post-deadline.
- [ ] AI-4 (design, 2026-07-19, user): pair complement-fork with a
      LIVE-LEARNING model - the fork's report-back testimony is exactly
      the supervision a live learner lacks. Test: unadulterated live
      learning vs live learning + this dynamic-SC (complement-fork)
      model. Post-deadline.
- [ ] AI-5 (design, 2026-07-19, user; gated on cfork benefit): distill
      the fork into POST-TRAINING of the discrete model, preserving
      parallelization - synthesize training data from offshoots at
      detected divergences, loss compares the offshoot's result against
      the measured good result. Agent refinement (user invited rework):
      frame as FORK-ANCHORED PREFERENCE PAIRS. At harvest time, run the
      cfork machinery over training problems; each detected divergence
      whose branches disagree yields a minimal pair - shared prefix up to
      the fork token, two greedy continuations, gold answer arbitrates
      chosen vs rejected. Train DPO/IPO-style on these pairs (or, cheaper,
      SFT on the winning branch with loss up-weighted from the fork site,
      like the existing answer-weighting). Three properties make this
      better than generic rejection sampling: (1) the arbitration wall
      DOES NOT APPLY - queue22 shows the +6.5-point oracle ceiling is
      inaccessible at inference because gates are weak, but at harvest
      time the oracle is free (gold labels), so the fork's trapped value
      converts to supervision; (2) credit assignment is sharp - branches
      are greedy and share everything up to the fork, so each pair
      isolates exactly the contended decision, and the trigger fires
      60-110 tokens BEFORE the error commits (part-1 result), splitting
      while the good continuation is still reachable; (3) all supervision
      is teacher-forced - full training parallelism, no inference-time
      forking in the deployed model. Online variant (pairs generated
      on-policy during deployment) is the concrete mechanism for AI-4's
      live-learning arm. Post-deadline build; design recorded.
- [x] AI-6 (2026-07-19, user; IMPLEMENTED 344706c, queue24 running):
      three-component attack on the branch-level arbitration wall.
      (1) CLOSED-LOOP PROPORTIONAL STEERING - replace the open-loop
      fixed-persist injection with a P-controller: each step the push is
      recomputed from the currently measured state (root suppresses its
      current above-baseline excursion, offshoot suppresses its current
      dominant component), so it decays as the error decays and tracks it
      while it persists (--steer-mode closed-loop, default).
      (2) CONVEX HULL CONSTRAINTS on offshoots - injections clipped to
      0.25x the branch's current tap norm, and the hook rescales the
      perturbed hidden back to its pre-injection norm (direction changes,
      energy does not): offshoots stay in the shell of states the trunk
      and corrector were trained on, targeting the min-delta testimony
      inversion (queue23's off-manifold failure). Design note: this is a
      norm-shell constraint, chosen deliberately - the tap has no hull
      vertex set (unlike the vocabulary at the embedding interface), a
      QP against the empirical point cloud is near-vacuous in d=896, the
      RMSNorm-downstream architecture makes the shell the functionally
      binding boundary, and the injection algebra at gamma<=1 is already
      a convex interpolation (rationale in paper 2 section 6.2). The
      learned upgrade is the tap-manifold snap projector (below).
      (3) BRANCH ARBITER - lightweight (0.51M param) bidirectional
      cross-attention encoder reading the full strided tap-state
      sequences of ALL branches jointly (branch + position embeddings,
      2 layers, mean-pool per branch, scalar score head); trained on
      harvest-time fork traces with free gold arbitration (the AI-5
      insight applied to the arbiter rather than the trunk); deployed as
      a learned gate (retrofit-fork-harvest / retrofit-train-arbiter /
      retrofit-cfork --arbiter). Pipeline smoke-tested end to end; floor
      preserved (z50: no forks, gates == latent). Queue24: closed-loop+
      hull eval (200), harvest 1k train problems (~33% informative keep),
      train 3000 steps, arbiter-gated eval (200).

## Intrusive thoughts, part 1: monitor + trigger bake-off (2026-07-19, queue21)

Setup: 0.5B, corrector retrained with --monitor-basis (identical recipe,
outputs/retrofit-qwen05b/corrector-7k-monitored/, metrics.jsonl has the
full log), then trigger-lab --basis on the existing corrector-7k greedy
dump (200 problems, 108 wrong rollouts, 168 pairs / 672 positive / 594
sibling-negative / 14,408 in-context-negative positions).

**AI-1 (training monitor): boom hypothesis refuted; corrector is born
complement-dwelling.** delta_comp_frac = 0.936 at the FIRST logged step
(25) and stays 0.91-0.93 for all 3000 steps; delta_intrusion_rate
fluctuates 0.014-0.067 with no trend; delta_norm grows 7 -> ~28 over the
first ~1000 steps then plateaus (25-31). Tap stats are stationary as
designed (~0.82 frac, 0.02-0.06 rate). No boom before any loss drop: the
complement orientation is the immediate gradient direction from step
one, not a late phase transition. Amplitude, not orientation, is what
training learns. Caveat: this monitors corrector distillation, not trunk
pretraining, where the original boom hunch may still apply.

**AI-3a (bake-off): the complement is the first LEADING indicator.**
Rollout-level recall at calibrated FPR, with median delay in tokens
relative to the oracle error site (negative = fires before):

| rule | variant | recall@5% | recall@10% | recall@20% | median delay @20% |
| --- | --- | --- | --- | --- | --- |
| probe-sibling | audit32 | 0.176 | 0.194 | 0.287 | +20 |
| probe-incontext | audit32 | 0.102 | 0.148 | 0.306 | +9 |
| meandiff-v | audit32 | 0.120 | 0.185 | 0.343 | +148 |
| complement-energy | raw | 0.120 | 0.185 | 0.306 | -69 |
| complement-frac | cusum8 | 0.111 | 0.157 | 0.398 | -92 |
| **complement-frac** | **audit32** | 0.083 | 0.222 | **0.444** | **-94** |

Two findings. (1) complement-frac audit32 is the best rule in the lab at
20% FPR (48/108 wrong rollouts), the first to clear the ~23-35% recall
wall. (2) The qualitative departure: every baseline rule fires AFTER the
error site (delays +1 to +219); every complement variant fires 28-113
tokens BEFORE it. The intrusion signal is anticipatory - contending
hypotheses surface in the complement before the error commits to tokens.
That is the wrong shape for rollback (which needs to know where to
rewind) and exactly the right shape for forking, which is the AI-3b
design. Gate passes; AI-3b proceeds (queue22).

Paper: sections 6.4 (monitor + framing) and 7.4 (leading-indicator
paragraph).

## Intrusive thoughts, part 2: complement-fork rollout (2026-07-19, queue22/23)

Setup: 0.5B, corrector-7k, rank-64 full basis; fork on complement-energy
Welford z >= 2.5 (warmup 16, cooldown 16), cap 4 branches, gamma 1.0,
persist 4; both branches greedy; decode once, gate many. Floor check
passed (fork-z 50: 0 forks, all gates == latent, 0.450 on 20 problems).

**Reinforce arm (original design, offshoot gets +gamma x excursion;
200 problems, reports/20260719-cfork-z2p5-qwen05b.md):**

| gate | accuracy |
| --- | --- |
| latent (same-run ref) | 0.460 |
| root (suppression alone) | 0.455 |
| vote | 0.470 |
| agree-else-mindelta (report-back) | 0.470 |
| min-delta | 0.470 |
| min-intrusion | 0.460 |
| max-logprob | 0.470 |
| oracle (any-correct ceiling) | **0.525** |

Mean branches 3.50, 191/200 forked, 54/191 forked problems had an
offshoot answer differently; 1017.8 internal tokens/problem.

Readings: (1) THE CEILING IS REAL - suppress/reinforce splits produce
complementary correctness worth +6.5 points over greedy, at 2.4x greedy
cost (vs SC@8's +12 at ~8x). The latent split does decorrelate branches
where temperature-sampled dynamic SC (queue18) could not. (2) The gates
cannot harvest it: report-back = vote = min-delta = 0.470, +1 point,
within noise. The arbitration wall recurs at branch level - testimony
separates populations weakly, same as the steering-gate margin story.
(3) Root alone == latent: suppressing the excursion in the mainline
neither helps nor hurts. (4) The raw-z trigger is permissive (95% of
problems fork); the bake-off says the sharp variants are audit32/cusum8,
so trigger reformulation is the obvious next lever if the suppress-
dominant arm shows the same shape.

**Suppress-dominant arm (amended design: offshoot gets -gamma x BB^T h,
per-branch completion budgets; 200 problems,
reports/20260719-cfork-suppdom-z2p5-qwen05b.md):**

| gate | reinforce | suppress-dominant |
| --- | --- | --- |
| latent | 0.460 | 0.460 |
| root | 0.455 | 0.450 |
| vote | 0.470 | 0.445 |
| agree-else-mindelta | 0.470 | 0.385 |
| min-delta | 0.470 | 0.385 |
| min-intrusion | 0.460 | 0.395 |
| max-logprob | 0.470 | 0.440 |
| oracle | 0.525 | **0.535** |

Mean branches 3.52, 191/200 forked, 99/191 diverged (vs 54/191
reinforce), 1051.4 internal tokens/problem. Amended floor clean (z50: 0
forks, gates == latent).

VERDICT (AI-3b closed): the two arms dissociate GENERATION from
ARBITRATION quality. Suppress-dominant is the stronger divergence engine
- nearly double the divergent offshoots, higher oracle ceiling (0.535) -
but its off-manifold offshoots break the testimony: min-delta flips from
best gate (0.470, reinforce) to worst (0.385), because the corrector,
trained on-manifold, responds weakly to a tap shoved off the dominant
subspace - low correction pressure becomes a mark of degeneracy, not
health. Report-back gating fails to harvest the ceiling in both arms;
the arbitration wall recurs at branch level. The constructive route is
AI-5: at harvest time the oracle is free, so the fork's divergent pairs
(more of them, and wilder, under suppress-dominant) convert directly to
post-training supervision. For a data engine, suppress-dominant wins;
for inference-time SC, neither arm is deployable at 0.5B.

Paper: new section 7.6 (both arms + the distillation reading).

## Intrusive thoughts, part 3: closed-loop, hull, arbiter (2026-07-19, queue24)

AI-6 closed. Setup: suppress-dominant arm, closed-loop proportional
steering + hull constraints (new defaults), 200 test problems; then
harvest 1,000 train problems (446 informative branch sets kept), train
the 0.51M BranchArbiter 3,000 steps, redeploy with the arbiter gate.

| gate | open-loop suppdom | closed-loop + hull | + arbiter |
| --- | --- | --- | --- |
| latent | 0.460 | 0.460 | 0.460 |
| root | 0.450 | **0.490** | 0.490 |
| vote | 0.445 | 0.460 | 0.460 |
| min-delta / report-back | 0.385 | 0.300 | 0.300 |
| max-logprob | 0.440 | 0.420 | 0.420 |
| arbiter | - | - | **0.490** |
| oracle | 0.535 | **0.570** | 0.570 |
| divergent offshoots | 99/191 | 133/191 | 133/191 |

Three verdicts. (1) CLOSED-LOOP MAINLINE IS THE FIRST NET-POSITIVE
STEERING RESULT in the program: root 0.490 vs latent 0.460 (+3, ~1.5 SE
at n=200 - replicate before leaning on it). Proportional suppression of
the measured excursion beats every open-loop/triggered/gated variant,
all of which were negative to break-even. (2) THE CEILING WIDENS AGAIN
(0.570, +11 over greedy at 3.0x cost; 54 -> 99 -> 133 divergent
offshoots across the three fork configs) but scalar testimony degrades
further (min-delta 0.300) - hull constraints keep offshoots decodable,
not scalar-legible. (3) THE ARBITER TIES THE ROOT AND ADDS NOTHING:
deployment 0.490 = root gate; in-distribution holdout argmax 0.72 vs
0.68 always-pick-root baseline (12.5% of available headroom; train 0.79
vs holdout 0.72, so capacity/data is not obviously binding). The
arbitration wall survives a learned sequence-level reader: branch
correctness is at best weakly represented in the tap trajectory; it is
legible in OUTCOMES. This is (a) consistent with answer-level agreement
being the only gate that ever worked, and (b) the strongest argument yet
for AI-5: the supervision route consumes outcomes, which is where the
signal actually lives.

Paper 2 (intrusive-thoughts) sections 6.2-6.3 + scorecard I8 updated.

REPLICATION (queue25, n=400): the closed-loop root gain DOES NOT
REPLICATE. First 200 problems reproduce queue24 deterministically (root
0.490, latent 0.460); on the fresh slice 200-399, root 0.415 vs latent
0.435; pooled n=400 root 0.4525 vs latent 0.4475 (within noise, ~1.4 SE
units). Verdict: closed-loop mainline suppression is NEUTRAL. The
ceiling and the divergence engine DO replicate and strengthen: oracle
0.550 pooled (+10.3 over latent), 284/381 forked problems with a
divergent offshoot (151/190 on the fresh slice). Paper 2 abstract,
contribution 3, section 6.2, and scorecard I8 corrected; the initial
0.490 is retained in the text as a worked example of single-slice
gains at ~1.5 SE failing replication.
Report: reports/20260719-cfork-closedloop-hull-n400-qwen05b.md.

## Intrusive thoughts, part 4: the complement lens (2026-07-19, user experiment)

User request: directly test the intrusive-thoughts claim by decoding the
complement and checking for structure (erroneous numbers/arithmetic)
rather than gibberish. Implemented retrofit-complement-lens: teacher-force
each latent completion (replay recovers generation-time taps exactly),
split tap states into dominant BB^T h and complement h - BB^T h, decode
each component stream with the trunk's OWN upper half (_upper_half_logits,
the sandwich decoder, bit-exact on real taps; prompt positions kept full
for context). Controls: per-position norm-matched Gaussian noise +
position-shuffled complement. 200 problems, 8,879 mainline-digit
positions, plumbing sanity 0.955.

| stream | agree w/ full | digit@digit pos | contending digit | ctx number |
| --- | --- | --- | --- | --- |
| dominant | 0.104 | 0.544 | 0.340 | 0.739 |
| complement | 0.653 | 0.908 | 0.258 | 0.770 |
| random (norm-matched) | 0.011 | 0.040 | 0.033 | 0.540 |
| shuffled complement | 0.027 | 0.163 | 0.123 | 0.792 |

CLAIM SUPPORTED on all three predicted properties: STRUCTURED (90.8%
digit rate at numeric positions vs 4% noise), POSITION-BOUND (shuffle
collapses to 16.3% - same vectors, wrong steps, no content), CONTENDING
(25.8% of numeric positions decode a DIFFERENT digit, 8x noise; 77% of
decoded numbers are the problem's own quantities). Examples read as
near-miss arithmetic, not corruption: mainline "16 - 3" complement "1";
mainline "= 23" complement "0"; mainline "9 x 6" complement "3".

FOLLOW-UP (user: "anything we can do about that caveat? what accounts
for the missing 9%?"): added c-parallel (attenuated mainline echo),
c-perp (off-axis tilt), and runner-up-alignment metrics. Results resolve
the caveat into a MECHANISM:
- c-parallel alone: 0.995 digit rate, 1.3% contention, dissents 85.5%
  = the trunk's own runner-up. Attenuation exposes the second choice,
  nothing else.
- c-perp alone: junk (1.4% digits) - algebraically forced (c-perp =
  -d-perp in a two-part split); no on-manifold carrier, no decode (the
  hull lesson again).
- Full complement (carrier + tilt): contention jumps 20x (0.013 ->
  0.258) while staying numeric; only 34.5% of dissents land on the
  trunk's top-2, so TWO-THIRDS of intrusions voice hypotheses OUTSIDE
  the trunk's top two choices, still built from the problem's
  quantities. The intrusion is a directional tilt riding an attenuated
  mainline carrier - not echo, not noise, not the runner-up.
- Missing 9.2% = equation syntax: operators 5.7% + whitespace 2.6%
  (noise control: 39% operators + 45% word-junk).
Paper 2: new decode subsection in section 3, scorecard I10 (Pass), intro
+ abstract updated. Report: reports/20260719-complement-lens-qwen05b.md.

## Paper revision program (2026-07-19, user notes)

Text-level items DONE in the same pass (both papers): metaphorical
language reduced (voicing/dissents/testimony/born/costume/earns-its-keep
etc. replaced with evaluative phrasing), "honest" flags removed, gamma
notation formalized (fork gains are gamma_1 = root suppression, gamma_2 =
offshoot treatment; experiments set both to 1), experimental-framework
sections added to both papers (common config tables: trunks, corrector
recipe, harvest, eval protocol, trigger calibration, fork defaults,
hardware, statistical note at n=200), diagrams added (paper 1: quorum,
snap, sandwich modular diagrams; paper 2: Jacobian decomposition,
token-fork vs complement-fork timing comparison, closed-loop controller
block diagram).

Experiments required (tracked here; queue27 covers phase 1):

- [ ] R1 benchmark breadth: MATH at 0.5B-3B locally (harness exists:
      --dataset math; 32B in flight), a logic suite, and a coding suite
      (pass@1; needs new harness + answer checking). Largest item.
- [x] R1a: 32B MATH (DONE 2026-07-20: latent 0.910 = cot 0.910 strict at
      3.4 vs 2,022 emitted tokens; section below; paper 1 section 5.3).
- [x] R2 param-matched LoRA: r=43 (~2.9M params = corrector) on the same
      full harvest, same loss; n=200 eval. DONE 2026-07-20: 0.385/0.395
      (section below; paper 1 section 5.1).
- [x] R3 statistical confidence: full GSM8K test n=1319 for the headline
      0.5B rows. DONE 2026-07-20: ordering unchanged, latent SC@8 0.548 >
      visible SC@8 0.514 (section below; paper 1 section 5.1).
      1.5B/3B n=1319 follow in phase 2.
- [ ] R4 wall-clock + peak-VRAM profiling per system (paper 1 5.5
      companion): instrument evaluate_retrofit with timing + 
      torch.cuda.max_memory_allocated per arm.
- [ ] R5 layer x offset error-probe heatmap (paper 1 tap-choice figure;
      probe machinery exists in divergence.py; sweep layers 8-16 +
      0/23/24 x offsets 1-32, render matplotlib PNG).
- [ ] R6 accuracy vs relative-decode-compute scatter (paper 1 5.4/5.5
      figure; data exists in FLOPs tables; matplotlib PNG).
- [ ] R7 snap drift curve: cosine-to-true-trajectory vs step horizon,
      raw vs snap feedback (paper 1 section 7 figure; needs a small
      instrumented rollout run).

Format/editorial round 2 (user notes, 2026-07-19 late):

- [x] R8 paper 1 content: abstract de-domained (token counts removed,
      cost claim deferred to body), intro cost justification (serial
      decode economics, output-length multiplier, transcript persistence
      + KV-cache footprint in multi-turn contexts; explicit statement
      that compute is NOT reduced, the emitted channel is), containment
      motivation removed (weak), zero-ablation note moved under
      mechanistic access, paper-2 references recast as "separate work",
      related work finalized with citations, References section added
      (17 entries), "headline" removed both papers.
- [ ] R9 LaTeX two-column conversion, both papers, 8-9 page target
      (implies ~40% condensation; em-dash sweep folded in since every
      sentence is touched; mermaid diagrams become TikZ/PDF figures).
- [ ] R10 paper 2 citations + references section (same treatment as
      paper 1; additional refs: CUSUM/Page 1954, DPO/Rafailov 2023,
      process supervision/Lightman 2023, activation patching lineage).
- [ ] R11 full em-dash sweep of any text not rewritten during R9.

## The 32B rung lands: QwQ-32B on MATH (2026-07-20)

The cloud eval completed after ~20 hours of A100-40 decode (100 MATH test
problems, three arms, NF4 trunk). Full recipe: 3,247 harvested traces
(67% keep), corrector d_cfc=512 at d=5120 (6.8M params, ~0.02% of trunk),
tap layer 32 of 64, 3,000 BPTT-chunked training steps.

| system | strict | lenient | mean emitted tok | mean internal tok |
| --- | --- | --- | --- | --- |
| direct | 0.000 | 0.030 | 32.0 | 32.0 |
| cot | 0.910 | 0.930 | 2,022.4 | 2,022.4 |
| latent | **0.910** | 0.920 | **3.4** | 1,929.4 |

Readings: (1) strict parity 0.910 = 0.910 at **595x fewer emitted
tokens** — the emitted-channel claim transfers to a reasoning-tuned
trunk, a second benchmark, 4-bit quantization, and 10x the previous top
scale, all unturned; (2) internal rollout 5% shorter than the visible
chain, so latent is also marginally compute-cheaper here (~0.95x); (3) no
accuracy gain, as predicted by the keep-rate law (67% keep ≈ 3B's 79%
regime); (4) lenient gap (0.920 vs 0.930) is one problem at n=100, within
noise. Documented: paper 1 abstract, contribution 4, section 5.3 block,
section 5.5 FLOPs paragraph, limitations, H3, conclusion. Artifacts:
reports/20260720-qwq32b-math.md + outputs/retrofit-qwq32b/
qwq32b-math.completions.jsonl (100 rows). VM deleted after artifact pull.

## Param-matched LoRA r=43 (2026-07-20, queue27 R2)

LoRA rank 43 on q_proj+v_proj = 2.90M trainable params (corrector-7k:
2.9M), trained 3,000 steps on the same traces-7k harvest with the same
answer-weighted loss; n=200 GSM8K eval: **0.385 strict / 0.395 lenient**
at 284 emitted tokens. vs r=16 (1.08M, 290 traces): 0.380 lenient.
Tripling parameters AND data moves LoRA +1.5 points — still 6 below the
untuned trunk (0.455) and 6.5 below the corrector (0.460). Closes the
capacity objection: the corrector-vs-LoRA gap is architectural (zero-
anchored additive floor vs unconstrained weight update), not a parameter
budget artifact. Paper 1 section 5.1 subsection + framework table +
limitations updated.

## Answer-phase monitoring: clean negative (2026-07-20, queue27)

User hypothesis tested: the corrector keeps operating during the final
answer decode (post-####), and the marker makes the answer span
self-localizing — the wall's localization problem does not apply. If
magnitude statistics carry arbitration signal anywhere, it is here.
n=200 greedy (181 with marker, mean span 17.2 tok) + k=8 samples:

- Wrongness AUCs (answer span): mean delta 0.406, max delta 0.431 —
  INVERTED (calm answer spans predict wrong answers); comp frac 0.543,
  span-z 0.537 — chance; chain mean delta 0.343 — strongly inverted.
- Policies: vote 0.540 (ref) > weighted-vote 0.515 > greedy 0.460 >
  seq-accept 0.405@2.04 rollouts > min-ans-delta 0.370. Nothing beats
  the plain vote; picking the calmest rollout is 17 points worse.

Reading: reproduces the min-delta inversion (queue23/6.2) in a setting
with zero localization excuse — low corrector output = disengagement,
not health. Sharpens the wall diagnosis: not a localization failure but
an information failure of scalar magnitude statistics. Documented as
paper 2 section 5.5 + scorecard I11. Tool: retrofit-answer-monitor;
report reports/20260719-answer-monitor-qwen05b.md.

## Tap-snap eval OOM: diagnosis and fix (2026-07-20)

Queue26's cfork+tap-snap eval crashed CUDA OOM at ~problem 120 on all 5
attempts (identical failure point; "72 GiB allocated" via WDDM shared-
memory spill = unbounded growth). ROOT CAUSE: _generate_complement_fork
had no torch.no_grad() fence — harmless for two years of baseline runs
because load_trunk freezes trunk params and the corrector steps under
its own no_grad, but load_snap did NOT freeze the SnapProjector, so the
snap call inside the forward hook created grad-requiring outputs, and
every downstream hidden state + KV-cache entry retained an autograd
graph for the whole rollout, accumulating across problems. FIX (two
layers): @torch.no_grad() on _generate_complement_fork +
snap.requires_grad_(False) in load_snap. LESSON: any trained module
added to a generation-time hook must be explicitly frozen; the absence
of a no_grad fence is latent until the first hook module with live
grads.

Partial pre-crash readings at 120/200 (biased low? no — accuracy
readings valid, sample just truncated): latent 0.433, root 0.392,
oracle 0.467 vs queue25 same-slice root ~0.475/oracle ~0.525.
Preliminary signal: snapping injected rows to the learned tap manifold
HURTS the fork arms — consistent with the hull lesson (norm containment
sufficient, geometry surgery harmful) but not conclusive at n=120.
Queue28 armed (waits QUEUE27 COMPLETE): full 200-problem rerun with the
fix → reports/20260720-cfork-tapsnap-qwen05b.md.

## Tap-snap rerun complete: learned manifold projection is net-negative (2026-07-20, queue28)

The fixed rerun completed cleanly (200/200, no OOM — confirming the
autograd-leak diagnosis). Tap-snap arm (closed-loop + hull + snap on
injected rows) vs the matched-slice closed-loop + hull arm:

| gate | cl + hull | + tap snap |
| --- | --- | --- |
| latent | 0.460 | 0.460 |
| root | 0.490 | 0.425 |
| vote | 0.460 | 0.435 |
| min-delta | 0.300 | 0.370 |
| max-logprob | 0.420 | 0.405 |
| oracle | 0.570 | **0.535** |
| divergent offshoots | 133/191 | 116/191 |
| mean internal tokens | 1,239 | 998 |

Readings: every treatment-sensitive number retreats toward the un-forked
baseline (fewer divergent offshoots, shorter rollouts, weaker root,
oracle back at the open-loop 0.535). ROOT CAUSE IS THE OBJECTIVE: the
projector is trained to recover clean tap states from fork-family
perturbations, i.e. it is an approximate inverse of the fork; applied to
injected rows it partially undoes the intervention (a smaller effective
dose). Confirms and sharpens the hull-ladder lesson: containment must
bound how far a state is pushed (norm shell — preserves treatment), not
restore where it was (geometry projection — cancels treatment). The
pre-crash partial at 120 matched this run exactly (latent 0.4333/root
0.3917/oracle 0.4667 vs 0.4333/0.3917/0.4667 here), so the OOM never
corrupted accuracy. Documented: paper 2 section 6.2 closing block + I8
scorecard row. Report: reports/20260720-cfork-tapsnap-qwen05b.md.

## Full GSM8K test set n=1,319: headline rows confirmed (2026-07-20, queue27 R3)

All 0.5B headline arms rerun on the complete test set (corrector-7k;
single-arm SE ~0.014 vs ~0.035 at n=200):

| system | strict | lenient | emitted tok | internal tok | n=200 ref |
| --- | --- | --- | --- | --- | --- |
| direct | 0.056 | 0.056 | 7.5 | 7.5 | 0.035 |
| visible CoT | 0.268 | 0.464 | 284.1 | 284.1 | 0.455 |
| visible SC@8 | 0.514 | 0.514 | 2,335.4 | 2,335.4 | 0.515 |
| latent greedy | 0.447 | 0.450 | 4.4 | 292.9 | 0.460 |
| latent SC@8 | **0.548** | **0.548** | 4.5 | 2,373.4 | 0.580 |

Readings: (1) every n=200 estimate within ~3 points of full-test value,
ordering fully preserved — the n=200 protocol was not lucky; (2) latent
greedy = CoT parity (0.450 vs 0.464, within noise); (3) latent SC@8 beats
visible SC@8 by 3.4 points at ~520x fewer emitted tokens — the vote margin
survives at scale, though narrower than the n=200 slice suggested (6.5);
(4) strict-vs-lenient at full test: visible CoT loses 20 points to format
noncompliance (0.268 strict), latent concedes at most 0.3 — the emitted-
answer-only design also buys format robustness. Documented: paper 1
framework statistical note, contribution 4, section 5.1 full-test block,
H4 scorecard. Reports: reports/20260719-retrofit-qwen05b-n1319.md +
reports/20260719-baseline-sc8-qwen05b-n1319.md (+ dumps). Runtimes: latent
arm ~5.0h, visible SC@8 ~2.1h on the 3090. 1.5B/3B n=1319 remain phase 2.
