# Longitudinal complement content across task adaptation

## Question

Does the orthogonal complement become less frequent, more structured, or more accurate as cross-entropy task adaptation improves the trunk?

The earlier ontogeny test measured the orientation of gradients emitted by a newly initialized sidecar. That is a learning-pressure measurement, not a measurement of the content already represented in the trunk. This experiment directly measures complement prevalence, decoded structure, and gold-token utility at six checkpoints of the same deterministic LoRA r=43 trajectory.

## Design

- Model: `Qwen/Qwen2.5-0.5B-Instruct`.
- Checkpoints: steps 0, 25, 100, 300, 1000, and 3000.
- Evaluation: the same 200 correct teacher-forced GSM8K traces at every checkpoint, totaling 51,011 scored next-token positions per phase.
- Coordinates: one fixed mature rank-64 Jacobian influence basis at tap layer 12. This supplies a shared coordinate system and prevents phase-local basis rotation from masquerading as content change.
- Controls: a seeded random rank-64 basis for residual-energy scale and a norm-matched Gaussian stream for decoded structure and recovery.
- Accuracy: exact gold-next-token recovery by complement-only decoding, conditioned on positions where the full stream is wrong. Recovery precision is also reported over all complement/full disagreements.
- Uncertainty: paired bootstrap over problems with 2,000 resamples for endpoint changes.

## Results

| step | strict acc | complement activation | random-basis complement | digit structure | noise structure | contention | gold recovery | noise recovery | recovery precision |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.175 | 0.821 | 0.930 | 0.895 | 0.043 | 0.251 | 0.164 | 0.004 | 0.024 |
| 25 | 0.345 | 0.822 | 0.931 | 0.866 | 0.044 | 0.258 | 0.207 | 0.004 | 0.031 |
| 100 | 0.325 | 0.822 | 0.930 | 0.863 | 0.038 | 0.257 | 0.224 | 0.005 | 0.030 |
| 300 | 0.320 | 0.821 | 0.931 | 0.861 | 0.037 | 0.254 | 0.225 | 0.004 | 0.027 |
| 1000 | 0.360 | 0.822 | 0.930 | 0.864 | 0.039 | 0.255 | 0.219 | 0.003 | 0.019 |
| 3000 | 0.395 | 0.822 | 0.930 | 0.814 | 0.030 | 0.243 | 0.205 | 0.003 | 0.009 |

Paired mature-minus-initial endpoint changes:

| metric | change | problem-bootstrap 95% CI |
| --- | ---: | ---: |
| complement activation fraction | +0.0007 | [+0.0007, +0.0008] |
| digit structure | -0.0811 | [-0.0896, -0.0725] |
| contending-digit rate | -0.0080 | [-0.0165, -0.0001] |
| gold recovery rate | +0.0416 | [+0.0147, +0.0684] |
| recovery precision | -0.0149 | [-0.0176, -0.0124] |

Overall complement token accuracy declines from 0.627 to 0.593 while full-stream teacher-forced token accuracy rises from 0.948 to 0.983. Digit structure remains far above the 0.030-0.044 norm-matched-noise range at every phase.

## Interpretation

The results reject a random-energy pruning account: complement prevalence does not diminish. They also reject broad honing: decoded digit structure, overall token accuracy, contention, and recovery precision all decline. The positive result is narrower. On the shrinking subset of positions that the mature full stream still misses, complement-only decoding is 4.2 points more likely to recover the gold token.

The supported picture is a stable, structured complement reservoir with modest specialization to residual errors while competence consolidates in the dominant/full computation. This is evidence from task adaptation of an already pretrained model, not evidence about emergence during pretraining from random initialization.