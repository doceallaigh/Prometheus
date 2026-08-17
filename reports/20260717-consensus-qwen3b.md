# State-space consensus probe

Model: `Qwen/Qwen2.5-3B-Instruct`, dump: `reports/20260717-retrofit-qwen3b-corrector7k.completions.jsonl`, problems: 200, unanimous: 114, vote correct: 174

Dispersion = 1 − mean pairwise cosine of h_tap mean-pooled over the
first t tokens of each of the 8 vote rollouts. AUCs: does early
dispersion predict final answer disagreement (split) or a wrong
majority vote?

| t | mean disp (unanimous) | mean disp (split) | AUC split | AUC vote-wrong |
| --- | --- | --- | --- | --- |
| 8 | 0.0225 | 0.0231 | **0.514** | **0.531** |
| 16 | 0.0453 | 0.0494 | **0.552** | **0.458** |
| 32 | 0.0290 | 0.0341 | **0.606** | **0.490** |
| 64 | 0.0183 | 0.0230 | **0.663** | **0.567** |
| 128 | 0.0127 | 0.0165 | **0.727** | **0.618** |
| full | 0.0092 | 0.0120 | **0.743** | **0.632** |
