# Oracle rollback experiment

Model: `Qwen/Qwen2.5-0.5B-Instruct`, dump: `reports/20260715-adaptive-decode-validation-qwen05b.completions.jsonl`, budget: 2, temperature: 0.6, rewind margin: 8 tokens

Wrong greedy latent rollouts are rewound to just before the arithmetic
error site (oracle localization) and resampled with a warm-started
corrector state over the shared prefix.

| metric | value |
| --- | --- |
| problems | 30 |
| greedy latent accuracy | 0.2667 |
| wrong rollouts | 22 |
| wrong without detectable error site | 0 |
| rollback attempted | 22 |
| ceiling: recovered (any re-roll correct) | 4 |
| **ceiling accuracy** | **0.4000** |
| detector-accepted re-rolls | 1 |
| detector-accepted correct | 1 |
| **detector-rule accuracy** | **0.3000** |
| mean re-rolls per problem (all problems) | 1.47 |
| mean re-roll tokens per problem | 243.0 |
