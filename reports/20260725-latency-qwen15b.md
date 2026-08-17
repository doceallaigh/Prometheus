# Prompt-to-answer latency

Model: `Qwen/Qwen2.5-1.5B-Instruct`; hardware: NVIDIA GeForce RTX 3090; GSM8K problems: 20; one warm-up per system; CUDA synchronized before and after every problem.

| system | mean s/problem | median | p90 | mean rollouts |
| --- | ---: | ---: | ---: | ---: |
| visible_cot | 5.006 | 4.499 | 8.534 | 1.00 |
| visible_sc8 | 7.047 | 6.530 | 9.894 | 8.00 |
| latent_greedy | 4.783 | 4.490 | 6.471 | 1.00 |
| latent_stop4of8 | 10.697 | 8.259 | 19.301 | 5.80 |
