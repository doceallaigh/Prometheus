# Training-time complement ablation

## Question

What happens when the orthogonal complement of a fixed rank-64 Jacobian influence basis is deleted, or replaced by norm-matched Gaussian noise, throughout foundational pretraining and downstream task adaptation? Does the effect exceed a dimensionality-matched random-subspace control, and does a model trained under the intervention remain useful at test time?

## Interventions

At the input to the midpoint transformer block, every token state is split into a retained rank-64 subspace and its orthogonal complement.

- `full`: unmodified residual stream.
- `complement-zero`: retain only the measured Jacobian-dominant rank-64 projection.
- `complement-randomized`: retain that projection and replace the complement with orthogonal Gaussian noise whose per-token norm matches the removed complement.
- `random-zero`: retain only a seeded random rank-64 projection.
- `random-randomized`: retain the random projection and replace its complement with matched Gaussian noise.

The Gaussian direction and removed-complement norm are detached, so gradients cannot transmit complement information through the replacement. Focused autograd tests verify that zeroing and randomization block complement-directed gradients and that replacement norms match.

## Foundational pretraining

### Design

- Model: `EleutherAI/pythia-70m-deduped`, restarted from the identical `step0` weights for every arm.
- Geometry: fixed mature rank-64 basis from Pythia step 143,000; intervention at the input to GPT-NeoX layer 3 of 6.
- Data: identical seeded without-replacement ordering of 32,000 C4 training windows, batch size 8, sequence length 128, for 4,000 optimizer steps (4.096M tokens). Sixty-four disjoint validation windows are fixed across arms.
- Optimization: AdamW, learning rate 3e-4, weight decay 0.01, BF16 autocast, one seed.
- Evaluation: clean validation and validation under the arm's matched intervention every 100 steps.

### Equal-token results

| arm | clean CE | clean next-token acc. | matched CE | matched acc. | train tok/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| full | 5.701 | 0.1783 | 5.701 | 0.1783 | 78,139 |
| complement-zero | 5.934 | 0.1659 | 5.788 | 0.1719 | 77,953 |
| random-zero | 5.931 | 0.1611 | 5.782 | 0.1730 | 78,344 |
| complement-randomized | 5.949 | 0.1588 | 5.951 | 0.1606 | 66,537 |
| random-randomized | 5.950 | 0.1591 | 5.952 | 0.1589 | 68,612 |

Clean CE learning thresholds show slower sample efficiency:

| arm | steps to CE <= 6.5 | steps to CE <= 6.0 | mean logged clean CE |
| --- | ---: | ---: | ---: |
| full | 800 | 2,300 | 6.158 |
| complement-zero | 1,000 | 3,400 | 6.294 |
| random-zero | 1,000 | 3,200 | 6.296 |
| complement-randomized | 1,100 | 3,600 | 6.331 |
| random-randomized | 1,100 | 3,600 | 6.335 |

Deleting the complement slows clean learning and costs 1.25 accuracy points at equal tokens. However, retaining a random 64D subspace costs 1.72 points and has an almost identical learning curve. The measured dominant carrier is therefore modestly better by endpoint accuracy, but most harm is explained by the 64D bottleneck. Gaussian replacement is even clearer: measured and random splits are effectively identical in clean CE and accuracy. Gaussian projection also costs 12-15% systems throughput, while zeroing has negligible overhead. Matched-condition evaluation partly recovers zero-trained models but does not recover the full-stream control.

## Downstream task adaptation

### Design

- Model: `Qwen/Qwen2.5-0.5B-Instruct` with LoRA rank 43 on `q_proj` and `v_proj`.
- Geometry: fixed mature Qwen rank-64 basis; intervention at the input to layer 12.
- Data and optimization: the same 1,078 teacher-forced GSM8K traces, seed, trace order, AdamW optimizer, learning rate 1e-4, answer weight 2, and 3,000-step schedule as the existing full-stream control.
- Resultant-model evaluation: greedy GSM8K, first 200 test problems, 512-token budget. The already documented inference-time causal sweep supplies the test-time intervention evidence.

### Optimization results

| arm | loss step 0 | step 100 | step 1,000 | endpoint / last finite | wall time | finite endpoint |
| --- | ---: | ---: | ---: | ---: | ---: | :---: |
| full | 0.291 | 0.115 | 0.073 | 0.047 | 170.3 s | yes |
| complement-zero | 10.657 | 2.567 | 0.557 | 0.557 | 166.3 s | yes |
| random-zero | 15.462 | 4.945 | 1.380 | 0.821 | 165.8 s | yes |
| complement-randomized | 9.433 | 4.550 | 7.743 | 6.674 at 2,700; then NaN | 171.5 s | no |
| random-randomized | 12.000 | 6.113 | 7.121 | 6.848 at 2,625; then NaN | 178.4 s | no |

Zeroing leaves optimization numerically stable and has no meaningful wall-time penalty, but it makes adaptation much less sample-efficient. The measured Jacobian-dominant carrier supports lower teacher-forced loss than the random carrier, establishing geometry-specific privilege within the severe 64D bottleneck. Both norm-matched Gaussian arms destabilize at nearly the same point and produce nonfinite adapters; this is generic high-dimensional corruption, not complement-specific failure.

### Resultant-model performance

| training condition | clean strict | clean lenient | matched-condition accuracy |
| --- | ---: | ---: | ---: |
| full | about 0.40 | about 0.40 | same as clean |
| complement-zero | 0.035 | 0.060 | 0.015 under complement-zero |
| random-zero | 0.000 | 0.000 | not needed after clean collapse |
| complement-randomized, step 1,000 | 0.000 | 0.000 | 0.000 under matched noise |

The random-zero model emits 510.7 tokens on average, nearly exhausting the generation budget, despite its declining teacher-forced loss. Complement-zero retains only a small clean foothold and also degrades under its matched training condition. Thus neither zero-trained model learned a useful alternate inference route; the low training loss is a teacher-forced condition shortcut that does not survive free generation. Gaussian endpoints are not evaluable because their adapters contain nonfinite weights; their finite step-1,000 checkpoint already scores zero.

## Relation to existing test-time ablations

The test-time experiment was already complete and was not repeated. Across full-stream task-adaptation saves, full complement deletion and replacement collapse accuracy, but graded 25% complement attenuation is nonmonotonic over adaptation and matched random-subspace attenuation is more damaging at every save. Test-time sensitivity therefore does not establish increasing or complement-specific dependence.

## Interpretation

The experiments support three bounded conclusions.

1. An intact high-dimensional residual stream is important during both foundational learning and task adaptation. Complement deletion slows foundational learning and nearly eliminates useful task adaptation; Gaussian replacement is worse.
2. The measured Jacobian-dominant rank-64 carrier is privileged relative to a random rank-64 carrier during task adaptation and modestly at the foundational endpoint. This is evidence that the measured geometry matters.
3. The strong harms are not specific to removing the measured complement. Random keep-64 controls reproduce most foundational degradation, and random Gaussian replacement reproduces both foundational harm and task-adaptation divergence. Because the removed complement holds roughly 82% of residual energy, the experiments cannot identify it as a uniquely necessary learning channel.

The defensible mechanistic statement is therefore narrower than the intrusive-thought analogy: structured, position-bound complement content emerges during pretraining and is consequential, but learning broadly requires intact residual capacity. The complement's unique necessity beyond dimensional capacity and generic corruption remains unestablished.

## Limitations and artifacts

These are one-seed interventions. Foundational training covers only 4.096M tokens in a 70M model and uses a fixed mature basis from the first update rather than a co-evolving basis. Task adaptation uses one 0.5B model, one task, and an extreme keep-64 bottleneck. The Gaussian arms alter both information and optimization noise, and their numerical divergence prevents endpoint comparison.

Foundational artifacts are under `outputs/foundation-training-ablation-pythia70m/<arm>/`. Task-adaptation artifacts are under `outputs/training-ablation-qwen05b/<arm>/`. Existing inference-time paired outcomes are under `outputs/causal-ablation-qwen05b-n200-graded/` and summarized in `reports/20260721-causal-ablation-qwen05b.md`.
