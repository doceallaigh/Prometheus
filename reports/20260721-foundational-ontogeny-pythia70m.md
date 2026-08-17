# Foundational pretraining ontogeny in Pythia-70M

## Question

Does structured content in the orthogonal complement arise during foundational language-model pretraining, rather than only during downstream task adaptation? Here "foundational" means learned during pretraining, not present innately at random initialization.

## Design

- Model: `EleutherAI/pythia-70m-deduped`.
- Revisions: steps 0, 64, 512, 4,000, 16,000, 64,000, and 143,000.
- Data: the same 64 held-out C4 validation windows at every checkpoint, each 128 tokens. Half are unfiltered generic-language windows and half are C4 windows containing at least two digit-bearing tokens. The scored set contains 8,192 next-token positions and 147 digit-token targets per checkpoint.
- Tap: input to GPT-NeoX layer 3 of 6, the architectural midpoint.
- Geometry: a checkpoint-local rank-64 influence basis fitted from 96 local Jacobian rows (12 windows, 4 positions, 2 random output directions). A local basis asks whether a complement exists relative to what each checkpoint's upper trunk currently uses, avoiding a mature-coordinate drift confound.
- Decode arms: unmodified, dominant-only, complement-only, position-shuffled complement, and per-position norm-matched Gaussian noise.
- Primary metric: exact next-token accuracy at positions whose gold token contains a digit. Digit-form decode rate is secondary because it measures structure without requiring the exact gold digit.

## Results

| step | effective rank | energy at rank 64 | cross-position influence | full digit accuracy | complement digit accuracy | shuffled complement | matched noise | complement digit-form rate |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 72.47 | 0.857 | 0.016 | 0.000 | 0.000 | 0.000 | 0.000 | 0.068 |
| 64 | 75.30 | 0.844 | 0.007 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 512 | 71.80 | 0.855 | 0.104 | 0.014 | 0.027 | 0.0068 | 0.000 | 0.224 |
| 4,000 | 74.77 | 0.841 | 0.220 | 0.136 | 0.088 | 0.0136 | 0.000 | 0.367 |
| 16,000 | 75.30 | 0.842 | 0.269 | 0.150 | 0.109 | 0.0068 | 0.000 | 0.429 |
| 64,000 | 71.25 | 0.854 | 0.275 | 0.190 | 0.109 | 0.000 | 0.000 | 0.429 |
| 143,000 | 48.59 | 0.913 | 0.231 | 0.170 | 0.095 | 0.0068 | 0.000 | 0.252 |

The complement's general next-token accuracy follows the same onset but peaks earlier: 0.000, 0.082, 0.160, 0.232, 0.209, 0.162, and 0.121 across the seven checkpoints. This decline is not a loss of language competence in the full model, whose general next-token accuracy rises to about 0.31 by steps 16,000-64,000.

## Interpretation

The result supports foundational emergence but not a monotonic S-curve.

1. Exact complement digit accuracy is absent at initialization and step 64, appears by step 512 (0.027, Wilson 95% CI [0.011, 0.068]), reaches 0.109 at steps 16,000-64,000 (CI [0.068, 0.169]), and remains 0.095 at the final checkpoint (CI [0.058, 0.154]). Matched Gaussian noise never predicts an exact digit token correctly. The overlapping mature intervals support a plateau, not a significant late exact-accuracy decline.
2. Position shuffling reduces exact digit accuracy to 0.000-0.014. The complement signal is therefore temporally aligned content, not merely high-dimensional residual energy.
3. Emergence is coincident with the formation of attention-mediated Jacobian structure. Cross-position influence mass rises from 0.007 at step 64 to 0.104 at step 512 and 0.220 at step 4,000, the same interval in which complement digit content first becomes measurable.
4. The broader digit-form trajectory is an inverted-U rather than an S-curve: it peaks at 0.429 at steps 16,000-64,000 (CI [0.351, 0.509]) and falls to 0.252 at step 143,000 (CI [0.188, 0.328]). Exact digit accuracy instead plateaus within uncertainty. The form-rate decline is consistent with late consolidation into the dominant/full computation, but that mechanism remains an interpretation rather than a direct causal result.

The defensible claim is that structured, position-bound complement content arises during foundational next-token pretraining before downstream task adaptation. It is not innate at initialization. This single 70M model does not establish universality across architectures or scales, and the checkpoint-local basis tracks each checkpoint's own functional geometry rather than a fixed mature coordinate system.
