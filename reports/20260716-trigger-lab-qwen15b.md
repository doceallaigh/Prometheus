# Trigger lab: offline detection-rule bake-off

Model: `Qwen/Qwen2.5-1.5B-Instruct`, dump: `reports/20260715-retrofit-qwen15b-eval.completions.jsonl`, problems: 200, wrong rollouts: 57, training pairs: 184

Rollout-level recall on wrong greedy rollouts at thresholds calibrated to
fixed false-alarm rates over correct greedy rollouts (max-over-positions
statistic). Delay = tokens from oracle error site to first trigger
(negative = fired before the site).

| rule | variant | FPR | recall | median delay (tok) |
| --- | --- | --- | --- | --- |
| probe-sibling | raw | 0.05 | **0.140** (8/57) | 3 |
| probe-sibling | raw | 0.10 | **0.193** (11/57) | 2 |
| probe-sibling | raw | 0.20 | **0.281** (16/57) | 3 |
| probe-sibling | selfnorm | 0.05 | **0.088** (5/57) | 3 |
| probe-sibling | selfnorm | 0.10 | **0.158** (9/57) | 3 |
| probe-sibling | selfnorm | 0.20 | **0.351** (20/57) | 2 |
| probe-sibling | cusum8 | 0.05 | **0.228** (13/57) | 6 |
| probe-sibling | cusum8 | 0.10 | **0.333** (19/57) | 6 |
| probe-sibling | cusum8 | 0.20 | **0.526** (30/57) | 6 |
| probe-sibling | audit16 | 0.05 | **0.088** (5/57) | 10 |
| probe-sibling | audit16 | 0.10 | **0.123** (7/57) | 10 |
| probe-sibling | audit16 | 0.20 | **0.228** (13/57) | 10 |
| probe-sibling | audit32 | 0.05 | **0.088** (5/57) | 10 |
| probe-sibling | audit32 | 0.10 | **0.193** (11/57) | 10 |
| probe-sibling | audit32 | 0.20 | **0.281** (16/57) | 10 |
| probe-incontext | raw | 0.05 | **0.140** (8/57) | 2 |
| probe-incontext | raw | 0.10 | **0.263** (15/57) | 3 |
| probe-incontext | raw | 0.20 | **0.351** (20/57) | 2 |
| probe-incontext | selfnorm | 0.05 | **0.035** (2/57) | 73 |
| probe-incontext | selfnorm | 0.10 | **0.088** (5/57) | 73 |
| probe-incontext | selfnorm | 0.20 | **0.281** (16/57) | 3 |
| probe-incontext | cusum8 | 0.05 | **0.158** (9/57) | 5 |
| probe-incontext | cusum8 | 0.10 | **0.246** (14/57) | 5 |
| probe-incontext | cusum8 | 0.20 | **0.421** (24/57) | 4 |
| probe-incontext | audit16 | 0.05 | **0.070** (4/57) | 15 |
| probe-incontext | audit16 | 0.10 | **0.175** (10/57) | 13 |
| probe-incontext | audit16 | 0.20 | **0.316** (18/57) | 10 |
| probe-incontext | audit32 | 0.05 | **0.088** (5/57) | 15 |
| probe-incontext | audit32 | 0.10 | **0.140** (8/57) | 15 |
| probe-incontext | audit32 | 0.20 | **0.246** (14/57) | 10 |
| meandiff-v | raw | 0.05 | **0.035** (2/57) | 2 |
| meandiff-v | raw | 0.10 | **0.105** (6/57) | 60 |
| meandiff-v | raw | 0.20 | **0.193** (11/57) | 60 |
| meandiff-v | selfnorm | 0.05 | **0.158** (9/57) | 2 |
| meandiff-v | selfnorm | 0.10 | **0.193** (11/57) | 57 |
| meandiff-v | selfnorm | 0.20 | **0.316** (18/57) | 2 |
| meandiff-v | cusum8 | 0.05 | **0.140** (8/57) | 7 |
| meandiff-v | cusum8 | 0.10 | **0.158** (9/57) | 7 |
| meandiff-v | cusum8 | 0.20 | **0.298** (17/57) | 4 |
| meandiff-v | audit16 | 0.05 | **0.140** (8/57) | 15 |
| meandiff-v | audit16 | 0.10 | **0.246** (14/57) | 15 |
| meandiff-v | audit16 | 0.20 | **0.316** (18/57) | 48 |
| meandiff-v | audit32 | 0.05 | **0.088** (5/57) | 138 |
| meandiff-v | audit32 | 0.10 | **0.210** (12/57) | 19 |
| meandiff-v | audit32 | 0.20 | **0.333** (19/57) | 19 |
