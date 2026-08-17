# Causal complement ablation across task adaptation

## Question

Does task adaptation make generation increasingly dependent on the complement of a fixed mature Jacobian influence basis?

## Design

- Model: Qwen2.5-0.5B-Instruct with parameter-matched LoRA r=43 task-adaptation saves at steps 0, 100, 1,000, and 3,000.
- Data: the same first 200 GSM8K test problems at every save, with greedy 512-token generation.
- Intervention site: input to layer 12, using the fixed mature rank-64 basis.
- Primary graded arms: retain 90% or 75% of the complement; replace 10% of the complement with norm-matched Gaussian content.
- Specificity controls: apply the same attenuation to a seeded random rank-64 split.
- Saturation controls: complement zero, dominant zero, norm-matched randomized complement, and random keep-64.
- Statistics: paired problem-level accuracy differences with 5,000-resample bootstrap 95% confidence intervals.

All ten arms were decoded as a single batch per problem. Prompts are therefore identical across arms, and seeded random bases and noise are paired across adaptation saves.

## Results

| step | full | comp x0.9 | random x0.9 | comp x0.75 | random x0.75 | 10% comp noise |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.435 | 0.430 | 0.465 | 0.330 | 0.290 | 0.400 |
| 100 | 0.420 | 0.380 | 0.405 | 0.355 | 0.290 | 0.380 |
| 1,000 | 0.395 | 0.390 | 0.395 | 0.325 | 0.300 | 0.395 |
| 3,000 | 0.420 | 0.420 | 0.375 | 0.335 | 0.245 | 0.350 |

Paired deltas from the unmodified checkpoint:

| step | comp x0.75 | random x0.75 | 10% comp noise |
| ---: | ---: | ---: | ---: |
| 0 | -0.105 [-0.180, -0.030] | -0.145 [-0.220, -0.065] | -0.035 [-0.100, +0.030] |
| 100 | -0.065 [-0.140, +0.015] | -0.130 [-0.205, -0.060] | -0.040 [-0.115, +0.030] |
| 1,000 | -0.070 [-0.140, +0.005] | -0.095 [-0.175, -0.020] | +0.000 [-0.065, +0.065] |
| 3,000 | -0.085 [-0.145, -0.020] | -0.175 [-0.245, -0.110] | -0.070 [-0.135, -0.005] |

Every full-strength arm saturates: complement-zero accuracy is 0.000 at all four saves; randomized-complement accuracy is 0.000-0.005; dominant-zero and random keep-64 accuracy are 0.000-0.015. Because the complement contains about 82% of residual-stream energy, those controls diagnose destructive intervention strength rather than subspace specificity.

## Interpretation

The requested increasing-penalty hypothesis is not supported. The 25% complement-attenuation penalties are 10.5, 6.5, 7.0, and 8.5 accuracy points: nonmonotonic and not larger at the final save than at initialization. Ten-percent complement randomization produces a significant seven-point penalty at step 3,000, but its full trajectory is also nonmonotonic.

The stronger causal claim is also not supported. Matched random-subspace attenuation is more damaging than complement attenuation at every checkpoint. The complement is causally consequential as part of the residual state, but this sweep does not isolate increasing or complement-specific dependence during task adaptation.

Artifacts: `outputs/causal-ablation-qwen05b-n200-graded/phase_metrics.jsonl` and `problem_outcomes.jsonl` contain aggregate and paired outcomes.