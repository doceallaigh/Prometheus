# Oracle rollback experiment

Model: `Qwen/Qwen2.5-1.5B-Instruct`, dump: `reports/20260715-retrofit-qwen15b-eval.completions.jsonl`, budget: 4, temperature: 0.6, rewind margin: 8 tokens

Wrong greedy latent rollouts are rewound to just before the arithmetic
error site (oracle localization) and resampled with a warm-started
corrector state over the shared prefix.

| metric | value |
| --- | --- |
| problems | 200 |
| greedy latent accuracy | 0.7150 |
| wrong rollouts | 57 |
| wrong without detectable error site | 9 |
| rollback attempted | 48 |
| ceiling: recovered (any re-roll correct) | 7 |
| **ceiling accuracy** | **0.7500** |
| detector-accepted re-rolls | 2 |
| detector-accepted correct | 2 |
| **detector-rule accuracy** | **0.7250** |
| mean re-rolls per problem (all problems) | 0.96 |
| mean re-roll tokens per problem | 161.2 |
