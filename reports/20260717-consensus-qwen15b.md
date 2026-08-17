# State-space consensus probe

Model: `Qwen/Qwen2.5-1.5B-Instruct`, dump: `reports/20260717-retrofit-qwen15b-corrector7k.completions.jsonl`, problems: 200, unanimous: 69, vote correct: 158

Dispersion = 1 − mean pairwise cosine of h_tap mean-pooled over the
first t tokens of each of the 8 vote rollouts. AUCs: does early
dispersion predict final answer disagreement (split) or a wrong
majority vote?

| t | mean disp (unanimous) | mean disp (split) | AUC split | AUC vote-wrong |
| --- | --- | --- | --- | --- |
| 8 | 0.0939 | 0.0915 | **0.484** | **0.538** |
| 16 | 0.0767 | 0.0734 | **0.467** | **0.556** |
| 32 | 0.0396 | 0.0430 | **0.579** | **0.571** |
| 64 | 0.0238 | 0.0290 | **0.701** | **0.620** |
| 128 | 0.0152 | 0.0193 | **0.750** | **0.670** |
| full | 0.0115 | 0.0151 | **0.703** | **0.711** |
