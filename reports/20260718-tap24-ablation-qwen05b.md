# GSM8K retrofit comparison

Model: `Qwen/Qwen2.5-0.5B-Instruct`, problems: 200

| system | strict accuracy | lenient accuracy | mean emitted tokens | mean internal rollout tokens |
| --- | --- | --- | --- | --- |
| direct | 0.0350 | 0.0350 | 7.4 | 7.4 |
| cot | 0.2750 | 0.4550 | 276.4 | 276.4 |
| latent | 0.4600 | 0.4650 | 4.4 | 298.0 |
| latent_sc8 | 0.5350 | 0.5350 | 4.5 | 2511.9 |

Mean latent rollouts per problem: 8.00 (cap 8)
