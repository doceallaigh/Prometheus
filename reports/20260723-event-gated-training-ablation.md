# Event-gated training-time complement ablation

## Question

Can targeted intervention at likely counterfactual or arithmetic-relevant positions distinguish the measured Jacobian-dominant carrier from an arbitrary rank-64 carrier, without the destructive saturation of intervening at every token?

## Design

- Model: `Qwen/Qwen2.5-0.5B-Instruct` with parameter-matched LoRA rank 43 on `q_proj` and `v_proj`.
- Training: the same 1,078 teacher-forced GSM8K traces, initialization, trace order, AdamW optimizer, learning rate 1e-4, answer weight 2, and 3,000-step schedule in every arm.
- Geometry: the fixed mature rank-64 Jacobian influence basis at the input to layer 12. `complement-zero` retains this measured carrier; `random-zero` retains a seeded random rank-64 carrier.
- Pairing: event masks are computed once from the frozen pre-adaptation trunk, saved, and reused exactly in treatment and random control. Adaptation therefore cannot move either arm onto a different event set.
- Sidecar-high gate: completion positions whose frozen corrector delta norm is at least two within-trace standard deviations above its completion mean. This is a counterfactual-presence proxy, not a verified label of a correct counterfactual.
- Semantic gates: residual row t is selected when the next-token target at t+1 contains a digit or an arithmetic operator (`+-*/=x/^%`, including multiplication and division glyphs). Prompt positions and the final non-predictive row are excluded.
- Evaluation: clean greedy generation on the same first 200 GSM8K test problems with a 512-token budget. Accuracy differences are paired by problem; intervals use 10,000 bootstrap resamples.

## Event prevalence

| gate | selected completion positions | fraction |
| --- | ---: | ---: |
| sidecar-high | 9,354 / 270,908 | 3.45% |
| digit | 38,763 / 270,908 | 14.31% |
| operator | 19,638 / 270,908 | 7.25% |

## Optimization

| gate | arm | loss step 0 | step 100 | step 1,000 | endpoint | wall time |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| sidecar-high | complement-zero | 0.458 | 0.203 | 0.068 | 0.071 | 169.2 s |
| sidecar-high | random-zero | 0.622 | 0.237 | 0.075 | 0.060 | 168.3 s |
| digit | complement-zero | 0.620 | 0.226 | 0.044 | 0.043 | 162.6 s |
| digit | random-zero | 2.916 | 0.514 | 0.122 | 0.106 | 162.2 s |
| operator | complement-zero | 0.377 | 0.139 | 0.063 | 0.031 | 162.4 s |
| operator | random-zero | 0.625 | 0.179 | 0.091 | 0.060 | 162.3 s |

Digit- and operator-gated training fit the teacher-forced traces substantially better through the measured carrier than through the random carrier. Sidecar-high training does not preserve that ordering at the endpoint. All six runs remain finite, complete in essentially equal wall time, and avoid the global keep-64 collapse (endpoint losses 0.557 measured and 0.821 random).

## Clean endpoint performance

| gate | complement strict / lenient | random strict / lenient | strict treatment-control delta (paired 95% CI) |
| --- | ---: | ---: | ---: |
| sidecar-high | 0.360 / 0.360 | 0.400 / 0.400 | -0.040 [-0.110, +0.030] |
| digit | 0.340 / 0.350 | 0.315 / 0.320 | +0.025 [-0.040, +0.090] |
| operator | 0.350 / 0.350 | 0.380 / 0.380 | -0.030 [-0.095, +0.035] |

The digit comparison is directionally consistent with its optimization advantage: measured-carrier training gains 2.5 strict and 3.0 lenient points over the exact-event random control. The interval includes zero. Operator gating reverses the teacher-forced ordering during free generation, with the random control ahead by three points. Sidecar-high gating also favors the random control by four points. Its interval likewise includes zero.

## Interpretation

Event timing solves the saturation problem but not the specificity problem. Restricting deletion to 3.5-14.3% of completion positions leaves every endpoint useful at 31.5-40.0% strict accuracy, whereas intervention at every token had reduced the measured keep-64 endpoint to 3.5% and the random keep-64 endpoint to zero. The earlier global result was therefore dominated by imposing a severe bottleneck throughout the sequence.

The arithmetic gates do reveal geometry-specific optimization privilege: at digit- and operator-predicting positions, the measured carrier supports endpoint teacher-forced losses 59% and 49% below their random controls. This does not become a stable clean-accuracy advantage. Only the digit arm has the same ordering in free generation, and its small paired difference is unresolved at n=200; the operator and sidecar-high controls score higher.

The defensible conclusion is that the measured dominant carrier is better aligned with fitting arithmetic targets under targeted intervention, while these experiments do not establish unique complement necessity for resultant task performance. The sidecar delta norm should also be described as an excursion detector or counterfactual-presence proxy: high correction pressure need not mean that a useful counterfactual is present.

## Limitations and artifacts

These are one-seed interventions on one 0.5B model and one arithmetic task. Event definitions are fixed from the pre-adaptation model for causal pairing, so they do not track counterfactual excursions that emerge during adaptation. Token gates depend on tokenizer pieces and select target classes, not semantic reasoning steps. At n=200, all treatment-control accuracy intervals include zero. A larger paired evaluation or multi-seed training is required to resolve effects of a few accuracy points.

Training artifacts are under `outputs/event-gated-training-qwen05b/`. Per-arm evaluation summaries and completion records are under `reports/20260723-*-eval.*`.