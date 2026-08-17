# GSM8K retrofit comparison

Model: `Qwen/Qwen2.5-0.5B-Instruct`, problems: 200

| system | strict accuracy | lenient accuracy | mean emitted tokens | mean internal rollout tokens |
| --- | --- | --- | --- | --- |
| direct | 0.0350 | 0.0350 | 7.4 | 7.4 |
| cot | 0.2750 | 0.4550 | 276.4 | 276.4 |
| latent | 0.3850 | 0.4200 | 4.2 | 274.7 |
| latent_sc8 | 0.4950 | 0.4950 | 4.6 | 2375.1 |

Mean latent rollouts per problem: 8.00 (cap 8)
