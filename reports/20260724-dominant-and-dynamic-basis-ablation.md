# Gated dominant-carrier deletion and dynamic basis tracking

## Questions

1. Is the measured rank-64 Jacobian-dominant carrier more necessary than an arbitrary rank-64 carrier at sidecar-high, digit-target, or operator-target positions?
2. During task adaptation, does the dominant carrier remain aligned with a fixed mature basis, and does tracking its drift change the resultant model?

## Shared design

- Model: `Qwen/Qwen2.5-0.5B-Instruct` with LoRA rank 43 on `q_proj` and `v_proj`.
- Training: the same 1,078 teacher-forced GSM8K traces, initialization, trace order, AdamW optimizer, learning rate 1e-4, answer weight 2, and 3,000-step schedule as the preceding event-gated study.
- Site: input to layer 12. `dominant-zero` removes the selected rank-64 projection and retains its 832-dimensional complement. `random-dominant-zero` removes a seeded random rank-64 projection.
- Gates: exact frozen masks from the preceding study, reused across all arms. Sidecar-high, digit, and operator gates select 3.45%, 14.31%, and 7.25% of completion positions.
- Evaluation: clean greedy generation on the complete 1,319-problem GSM8K test split, paired by problem with 10,000-resample bootstrap intervals.

## Gated dominant deletion

### Optimization

| gate | measured endpoint loss | random endpoint loss |
| --- | ---: | ---: |
| sidecar-high | 0.0472 | 0.0425 |
| digit | 0.0462 | 0.0484 |
| operator | 0.0448 | 0.0405 |

All arms remain near the unmodified LoRA endpoint loss of 0.0473. Removing 64 dimensions at only 3.5-14.3% of completion rows therefore imposes little optimization burden, whether the removed dimensions are measured or random.

### Clean endpoint performance

| gate | measured strict / lenient | random strict / lenient | strict measured-random delta (paired 95% CI) |
| --- | ---: | ---: | ---: |
| sidecar-high | 0.4056 / 0.4071 | 0.3829 / 0.3874 | +0.0227 [-0.0015, +0.0462] |
| digit | 0.3624 / 0.3692 | 0.3768 / 0.3829 | -0.0144 [-0.0394, +0.0106] |
| operator | 0.3920 / 0.3973 | 0.3912 / 0.3980 | +0.0008 [-0.0250, +0.0265] |

No paired interval excludes zero. Sidecar-high directionally favors deleting the measured carrier by 2.27 strict points (144 measured-only correct versus 114 random-only), but its interval narrowly includes zero. Digit directionally favors random deletion by 1.44 points (133 versus 152), and operator is effectively tied (150 versus 149). These gate-dependent directions do not identify the fixed measured dominant carrier as uniquely necessary at the selected events.

## Dynamic tracking versus a static mature basis

The static arm is the digit-gated `dominant-zero` run above. The dynamic arm starts from the same stored mature basis, then recomputes an attention-inclusive rank-64 influence basis after every 512 completed updates (steps 512, 1,024, 1,536, 2,048, and 2,560). Every refresh uses the same first eight traces, eight sampled completion positions per trace, four random output directions per position, and fixed seed. The intervention hook is removed during VJP estimation, so each basis describes the unablated adapting model.

Each refresh contains 256 local and 768 attention-mediated rows. A separate step-zero fit with identical samples and directions supplies an estimator-matched reference; this is necessary because independent finite-sample estimates overlap the older stored mature basis by only 0.336.

| refresh step | overlap with step-zero matched basis | overlap with previous dynamic basis | effective rank | energy at rank 64 |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 0.714 | n/a | 287.1 | 0.363 |
| 1,024 | 0.654 | 0.800 | 292.0 | 0.355 |
| 1,536 | 0.646 | 0.825 | 298.0 | 0.355 |
| 2,048 | 0.642 | 0.848 | 291.5 | 0.360 |
| 2,560 | 0.597 | 0.819 | 300.9 | 0.355 |

The carrier geometry drifts progressively from its estimator-matched initial orientation while changing smoothly between adjacent refreshes. The drift is not a sudden basis failure: adjacent rank-64 subspaces retain 80-85% projection overlap.

| arm | loss step 0 | step 1,000 | endpoint loss | wall time | strict / lenient |
| --- | ---: | ---: | ---: | ---: | ---: |
| static mature basis | 0.3894 | 0.0494 | 0.0462 | 168.4 s | 0.3624 / 0.3692 |
| dynamic basis | 0.3894 | 0.0520 | 0.0427 | 179.3 s | 0.3632 / 0.3723 |

Dynamic tracking lowers the endpoint teacher-forced loss by 0.0035 while adding 10.9 seconds (6.5%) of wall time. Clean task accuracy is effectively tied: dynamic minus static is +0.0008 strict (paired 95% CI [-0.0220, +0.0235]; 120 dynamic-only versus 119 static-only) and +0.0030 lenient ([-0.0197, +0.0258]; 122 versus 118).

## Interpretation

The dominant Jacobian carrier is not stationary during LoRA adaptation, but the observed drift is not behaviorally decisive under this intervention. Updating the deleted rank-64 projection follows a smoothly rotating functional geometry and modestly improves teacher-forced fit, yet produces no resolved clean-accuracy difference from deleting the fixed mature basis.

Together with the complement-retention arms, the result sharpens the capacity account. Keeping only 64 dimensions is destructive because it discards most of the residual stream; deleting only 64 dimensions at selected positions is mild and not specifically worse for the measured carrier. The measured basis is useful as a functional coordinate system, but these training interventions do not establish it as a uniquely necessary, fixed conduit.

## Limitations and artifacts

This is one seed, one 0.5B model, one arithmetic task, and one dynamic schedule. Dynamic refresh uses a finite VJP sample; fixed samples make temporal comparisons paired but do not remove estimator bias. Even on the complete test split, the static-vs-dynamic interval permits differences of about two points in either direction. Refreshing every 512 updates may miss faster rotation, while more frequent refresh would increase intervention-estimation feedback and compute cost.

Training artifacts are under `outputs/event-gated-dominant-qwen05b/` and `outputs/dynamic-vs-static-basis-qwen05b/`. Evaluation summaries and auditable completion records are under `reports/20260724-*-dominant-zero-eval.*`.