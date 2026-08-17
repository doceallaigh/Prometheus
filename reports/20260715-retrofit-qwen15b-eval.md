# GSM8K retrofit comparison

Model: `Qwen/Qwen2.5-1.5B-Instruct`, problems: 200

| system | strict accuracy | lenient accuracy | mean emitted tokens | mean internal rollout tokens |
| --- | --- | --- | --- | --- |
| direct | 0.0000 | 0.0650 | 2.7 | 2.7 |
| cot | 0.2150 | 0.6500 | 254.0 | 254.0 |
| latent | 0.7150 | 0.7150 | 4.4 | 260.9 |
| latent_sc8 | 0.7750 | 0.7750 | 4.4 | 2090.8 |
