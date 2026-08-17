# Direct FLOPs evaluation: sidecar overhead vs token savings

Model: `Qwen/Qwen2.5-3B-Instruct`, dump: `reports/20260717-retrofit-qwen3b-corrector7k.completions.jsonl`, problems: 200
Corrector forward: 7.34 MFLOPs/token (0.12% of trunk decode)

| system | rollouts | gen tokens/rollout | trunk GFLOPs/tok | sidecar overhead | total TFLOPs/problem | vs cot |
| --- | --- | --- | --- | --- | --- | --- |
| direct | 1.00 | 7.1 | 6.20 | 0.000% | 0.044 | 0.024x |
| cot | 1.00 | 299.8 | 6.24 | 0.000% | 1.871 | 1.000x |
| latent | 1.00 | 290.8 | 6.24 | 0.118% | 1.817 | 0.971x |
| latent_sc8 | 8.00 | 294.9 | 6.24 | 0.118% | 14.744 | 7.879x |

## Prompt-to-answer wall clock

RTX 3090, PyTorch eager decoding, 20 GSM8K prompts, one warm-up per system,
and CUDA synchronization around each complete prompt-to-answer call. Model load
and dataset access are excluded. SC rollouts are batched within each prompt.

| system | mean s/problem | median | p90 | mean rollouts |
| --- | ---: | ---: | ---: | ---: |
| visible CoT | 7.026 | 6.633 | 10.643 | 1.00 |
| visible SC@8 | 11.418 | 11.586 | 14.726 | 8.00 |
| latent greedy | 7.404 | 7.119 | 11.027 | 1.00 |
| latent stop-at-4 (cap 8) | 13.332 | 10.320 | 23.622 | 5.00 |
