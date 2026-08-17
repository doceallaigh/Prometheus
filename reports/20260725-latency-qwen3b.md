# Prompt-to-answer latency

Model: `Qwen/Qwen2.5-3B-Instruct`; hardware: NVIDIA GeForce RTX 3090; GSM8K problems: 20; one warm-up per system; CUDA synchronized before and after every problem.

| system | mean s/problem | median | p90 | mean rollouts |
| --- | ---: | ---: | ---: | ---: |
| visible_cot | 7.026 | 6.633 | 10.643 | 1.00 |
| visible_sc8 | 11.418 | 11.586 | 14.726 | 8.00 |
| latent_greedy | 7.404 | 7.119 | 11.027 | 1.00 |
| latent_stop4of8 | 13.332 | 10.320 | 23.622 | 5.00 |
