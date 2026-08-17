# Adaptive latent self-consistency simulation

Dump: `reports\20260715-retrofit-qwen15b-eval.completions.jsonl`, problems: 200, samples available: 8

Sequential draw over existing rollouts; stop when k answers agree.

| policy | accuracy | mean internal rollouts |
| --- | --- | --- |
| stop_at_1_agreeing | 0.6650 | 1.00 |
| stop_at_2_agreeing | 0.7550 | 3.00 |
| stop_at_3_agreeing | 0.7700 | 4.58 |
| stop_at_4_agreeing | 0.7850 | 5.57 |
| full_majority@8 | 0.7750 | 8.00 |
