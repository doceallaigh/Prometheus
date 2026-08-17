# Adaptive latent self-consistency simulation

Dump: `reports\20260717-retrofit-qwen15b-corrector7k.completions.jsonl`, problems: 200, samples available: 8

Sequential draw over existing rollouts; stop when k answers agree.

| policy | accuracy | mean internal rollouts |
| --- | --- | --- |
| stop_at_1_agreeing | 0.6900 | 1.00 |
| stop_at_2_agreeing | 0.7850 | 2.84 |
| stop_at_3_agreeing | 0.8000 | 4.41 |
| stop_at_4_agreeing | 0.8000 | 5.53 |
| full_majority@8 | 0.7900 | 8.00 |
