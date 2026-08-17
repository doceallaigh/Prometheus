# Adaptive latent self-consistency simulation

Dump: `reports\20260717-retrofit-qwen3b-corrector7k.completions.jsonl`, problems: 200, samples available: 8

Sequential draw over existing rollouts; stop when k answers agree.

| policy | accuracy | mean internal rollouts |
| --- | --- | --- |
| stop_at_1_agreeing | 0.8250 | 1.00 |
| stop_at_2_agreeing | 0.8700 | 2.37 |
| stop_at_3_agreeing | 0.8650 | 3.66 |
| stop_at_4_agreeing | 0.8700 | 4.75 |
| full_majority@8 | 0.8700 | 8.00 |
