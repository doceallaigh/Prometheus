# Adaptive latent self-consistency simulation

Dump: `reports\20260715-retrofit-qwen05b-latent-sc8.completions.jsonl`, problems: 200, samples available: 8

Sequential draw over existing rollouts; stop when k answers agree.

| policy | accuracy | mean internal rollouts |
| --- | --- | --- |
| stop_at_1_agreeing | 0.3350 | 1.00 |
| stop_at_2_agreeing | 0.4500 | 4.33 |
| stop_at_3_agreeing | 0.4500 | 6.05 |
| stop_at_4_agreeing | 0.4500 | 6.81 |
| full_majority@8 | 0.4500 | 8.00 |
