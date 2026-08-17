# Latent-CoT Combined Paper Reproduction

- Date: 2026-07-23
- Source: `papers/latex/latent-cot-combined.pdf` at commit `a64ee5c`
- Hardware: NVIDIA RTX 3090 (24 GB)
- Environment: Windows, Python 3.12.10, PyTorch 2.6.0+cu124

## Scope

This pass selected claims that could be independently rerun with repository-local code and artifacts in under six hours. It includes one from-scratch, three-seed training experiment; four deterministic reevaluations of retained synthetic checkpoints; and one fresh offline evaluation of the retained Qwen2.5-0.5B corrector. It does not treat existing reports as reproduction evidence unless the underlying command was rerun.

Skipped as prohibitively expensive or insufficiently independent for this pass: fresh Qwen 0.5B/1.5B/3B harvesting and corrector training, SC@8 at all scales, full GSM8K n=1,319, QwQ-32B/MATH, Pythia checkpoint downloads and ontogeny training, LoRA train-through ablations, Jacobian/complement analyses, and n=400 forking. Their retained artifacts remain in `outputs/` and `reports/`, but were not freshly reproduced here.

## Fresh Results

### Multi-hop continuous composition

All three models were trained from scratch using 1,000 proofs at depths 2-4 and evaluated on 200 held-out proofs at every depth 2-10. The sidecar and trunk settings match the paper protocol.

| seed | mean OOD direct | mean OOD matched-step latent | gain | best steps = depth |
| ---: | ---: | ---: | ---: | ---: |
| 20260720 | 0.0833 | 0.9892 | +0.9058 | 9/9 |
| 20260721 | 0.0833 | 0.9775 | +0.8942 | 9/9 |
| 20260722 | 0.0917 | 0.9725 | +0.8808 | 9/9 |
| **mean** | **0.0861** | **0.9797** | **+0.8936** | **27/27** |

This exactly reproduces the paper's aggregate values (`0.0861 +/- 0.0048` direct, `0.9797 +/- 0.0086` latent, `+89.36 +/- 1.25` points using sample SD) and its strongest structural claim: the unique best latent step count equals proof depth in every seed-by-depth condition.

Artifacts: `outputs/reproduction-latent-composition-{20260720,20260721,20260722}-20260723/`.

### Synthetic frozen-trunk corrector

The retained checkpoints were reevaluated on the same deterministic 300-problem validation split.

| system | latent accuracy | CoT accuracy | emitted tokens | internal steps |
| --- | ---: | ---: | ---: | ---: |
| CfC, layer-6 tap | 0.8633 | 0.5033 | 3.9 | 18.88 |
| embedding-only tap | 0.5567 | 0.5033 | 3.9 | 19.10 |
| GRU, layer-6 tap | 0.8200 | 0.5033 | 3.9 | 18.94 |
| train depths 2-6 | 0.8533 | 0.5033 | 3.9 | 18.86 |

For the OOD model, unseen depth-7/depth-8 accuracy was `0.8542/0.6522`, versus visible CoT `0.3958/0.3261`. The headline and embedding/OOD rows reproduce the cited retained-checkpoint values exactly. The GRU value is consistent with the paper's local/cloud spread (`0.828` local, `0.783` cloud) and remains below CfC.

Reports: `reports/20260723-reproduction-rrs-j-cfc-{headline,embed,gru,ood26}.md`.

### Qwen2.5-0.5B greedy audit

A fresh offline run used the cached model, cached GSM8K test split, and full-harvest corrector on the first 50 test problems. This is a precision-limited audit, not a replacement for the paper's n=200 or n=1,319 estimates.

| system | strict | lenient | emitted tokens | internal tokens |
| --- | ---: | ---: | ---: | ---: |
| direct | 0.000 | 0.000 | 7.8 | 7.8 |
| visible CoT | 0.220 | 0.440 | 287.8 | 287.8 |
| latent greedy | 0.440 | 0.440 | 4.0 | 295.5 |

The audit reproduces CoT parity and emitted-channel compression: latent and visible CoT both score `0.440` lenient, while latent emits 72 times fewer tokens. Internal rollout cost remains near parity (1.027 times visible CoT), so this does not support a compute-savings claim and is consistent with the paper's qualification.

Report: `reports/20260723-reproduction-qwen05b-n50.md`.

## Verdict

The feasible fresh checks support the paper's central architecture and composition findings. The strongest result reproduced independently from scratch and exactly. Retained-checkpoint evaluations reproduced the synthetic accuracy, ablation ordering, OOD transfer, and token counts. A limited pretrained-model audit reproduced 0.5B CoT parity and approximately 70-fold emitted-token compression.

This pass does not independently verify the multi-scale SC@8 ladder, 32B transfer, complement geometry, detection/intervention, ontogeny, or training-time causal claims. Those remain supported only by their original retained runs and reports until their expensive pipelines are rerun.