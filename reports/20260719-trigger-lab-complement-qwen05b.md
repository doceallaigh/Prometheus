# Trigger lab: offline detection-rule bake-off

Model: `Qwen/Qwen2.5-0.5B-Instruct`, dump: `reports/20260716-retrofit-qwen05b-corrector7k.completions.jsonl`, problems: 200, wrong rollouts: 108, training pairs: 168

Rollout-level recall on wrong greedy rollouts at thresholds calibrated to
fixed false-alarm rates over correct greedy rollouts (max-over-positions
statistic). Delay = tokens from oracle error site to first trigger
(negative = fired before the site).

| rule | variant | FPR | recall | median delay (tok) |
| --- | --- | --- | --- | --- |
| probe-sibling | raw | 0.05 | **0.083** (9/108) | 147 |
| probe-sibling | raw | 0.10 | **0.102** (11/108) | 94 |
| probe-sibling | raw | 0.20 | **0.250** (27/108) | 23 |
| probe-sibling | selfnorm | 0.05 | **0.046** (5/108) | 2 |
| probe-sibling | selfnorm | 0.10 | **0.139** (15/108) | 2 |
| probe-sibling | selfnorm | 0.20 | **0.250** (27/108) | 72 |
| probe-sibling | cusum8 | 0.05 | **0.074** (8/108) | 219 |
| probe-sibling | cusum8 | 0.10 | **0.148** (16/108) | 139 |
| probe-sibling | cusum8 | 0.20 | **0.250** (27/108) | 58 |
| probe-sibling | audit16 | 0.05 | **0.074** (8/108) | 167 |
| probe-sibling | audit16 | 0.10 | **0.111** (12/108) | 79 |
| probe-sibling | audit16 | 0.20 | **0.204** (22/108) | 15 |
| probe-sibling | audit32 | 0.05 | **0.176** (19/108) | 30 |
| probe-sibling | audit32 | 0.10 | **0.194** (21/108) | 30 |
| probe-sibling | audit32 | 0.20 | **0.287** (31/108) | 20 |
| probe-incontext | raw | 0.05 | **0.093** (10/108) | 1 |
| probe-incontext | raw | 0.10 | **0.130** (14/108) | 0 |
| probe-incontext | raw | 0.20 | **0.278** (30/108) | 1 |
| probe-incontext | selfnorm | 0.05 | **0.074** (8/108) | 2 |
| probe-incontext | selfnorm | 0.10 | **0.130** (14/108) | 1 |
| probe-incontext | selfnorm | 0.20 | **0.241** (26/108) | 1 |
| probe-incontext | cusum8 | 0.05 | **0.102** (11/108) | 6 |
| probe-incontext | cusum8 | 0.10 | **0.185** (20/108) | 6 |
| probe-incontext | cusum8 | 0.20 | **0.259** (28/108) | 5 |
| probe-incontext | audit16 | 0.05 | **0.018** (2/108) | 7 |
| probe-incontext | audit16 | 0.10 | **0.074** (8/108) | 7 |
| probe-incontext | audit16 | 0.20 | **0.241** (26/108) | 5 |
| probe-incontext | audit32 | 0.05 | **0.102** (11/108) | 7 |
| probe-incontext | audit32 | 0.10 | **0.148** (16/108) | 9 |
| probe-incontext | audit32 | 0.20 | **0.306** (33/108) | 9 |
| meandiff-v | raw | 0.05 | **0.056** (6/108) | 2 |
| meandiff-v | raw | 0.10 | **0.065** (7/108) | 2 |
| meandiff-v | raw | 0.20 | **0.148** (16/108) | 2 |
| meandiff-v | selfnorm | 0.05 | **0.102** (11/108) | 1 |
| meandiff-v | selfnorm | 0.10 | **0.120** (13/108) | 1 |
| meandiff-v | selfnorm | 0.20 | **0.222** (24/108) | 1 |
| meandiff-v | cusum8 | 0.05 | **0.093** (10/108) | 30 |
| meandiff-v | cusum8 | 0.10 | **0.176** (19/108) | 41 |
| meandiff-v | cusum8 | 0.20 | **0.278** (30/108) | 20 |
| meandiff-v | audit16 | 0.05 | **0.037** (4/108) | 215 |
| meandiff-v | audit16 | 0.10 | **0.074** (8/108) | 138 |
| meandiff-v | audit16 | 0.20 | **0.157** (17/108) | 7 |
| meandiff-v | audit32 | 0.05 | **0.120** (13/108) | 30 |
| meandiff-v | audit32 | 0.10 | **0.185** (20/108) | 42 |
| meandiff-v | audit32 | 0.20 | **0.343** (37/108) | 148 |
| complement-energy | raw | 0.05 | **0.120** (13/108) | -38 |
| complement-energy | raw | 0.10 | **0.185** (20/108) | -59 |
| complement-energy | raw | 0.20 | **0.306** (33/108) | -69 |
| complement-energy | selfnorm | 0.05 | **0.028** (3/108) | 121 |
| complement-energy | selfnorm | 0.10 | **0.102** (11/108) | -87 |
| complement-energy | selfnorm | 0.20 | **0.333** (36/108) | -81 |
| complement-energy | cusum8 | 0.05 | **0.111** (12/108) | -85 |
| complement-energy | cusum8 | 0.10 | **0.120** (13/108) | -38 |
| complement-energy | cusum8 | 0.20 | **0.250** (27/108) | -76 |
| complement-energy | audit16 | 0.05 | **0.130** (14/108) | -63 |
| complement-energy | audit16 | 0.10 | **0.148** (16/108) | -28 |
| complement-energy | audit16 | 0.20 | **0.213** (23/108) | -76 |
| complement-energy | audit32 | 0.05 | **0.065** (7/108) | -99 |
| complement-energy | audit32 | 0.10 | **0.130** (14/108) | -56 |
| complement-energy | audit32 | 0.20 | **0.287** (31/108) | -68 |
| complement-frac | raw | 0.05 | **0.056** (6/108) | -50 |
| complement-frac | raw | 0.10 | **0.074** (8/108) | -113 |
| complement-frac | raw | 0.20 | **0.232** (25/108) | -57 |
| complement-frac | selfnorm | 0.05 | **0.083** (9/108) | -56 |
| complement-frac | selfnorm | 0.10 | **0.194** (21/108) | -61 |
| complement-frac | selfnorm | 0.20 | **0.287** (31/108) | -66 |
| complement-frac | cusum8 | 0.05 | **0.111** (12/108) | -79 |
| complement-frac | cusum8 | 0.10 | **0.157** (17/108) | -86 |
| complement-frac | cusum8 | 0.20 | **0.398** (43/108) | -92 |
| complement-frac | audit16 | 0.05 | **0.083** (9/108) | -79 |
| complement-frac | audit16 | 0.10 | **0.139** (15/108) | -79 |
| complement-frac | audit16 | 0.20 | **0.315** (34/108) | -104 |
| complement-frac | audit32 | 0.05 | **0.083** (9/108) | -79 |
| complement-frac | audit32 | 0.10 | **0.222** (24/108) | -106 |
| complement-frac | audit32 | 0.20 | **0.444** (48/108) | -94 |
