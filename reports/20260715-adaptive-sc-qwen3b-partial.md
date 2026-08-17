# Adaptive latent self-consistency simulation

Dump: `reports\20260715-retrofit-qwen3b-eval.completions.jsonl`, problems: 68, samples available: 8

Sequential draw over existing rollouts; stop when k answers agree.

| policy | accuracy | mean internal rollouts |
| --- | --- | --- |
| stop_at_1_agreeing | 0.7794 | 1.00 |
| stop_at_2_agreeing | 0.8529 | 2.38 |
| stop_at_3_agreeing | 0.8676 | 3.74 |
| stop_at_4_agreeing | 0.8824 | 4.84 |
| full_majority@8 | 0.8824 | 8.00 |
