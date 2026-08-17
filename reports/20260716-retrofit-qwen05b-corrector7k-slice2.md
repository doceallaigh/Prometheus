# GSM8K retrofit comparison

Model: `Qwen/Qwen2.5-0.5B-Instruct`, problems: 200

| system | strict accuracy | lenient accuracy | mean emitted tokens | mean internal rollout tokens |
| --- | --- | --- | --- | --- |
| direct | 0.0550 | 0.0550 | 7.3 | 7.3 |
| cot | 0.1300 | 0.4650 | 285.8 | 285.8 |
| latent | 0.3850 | 0.4350 | 3.7 | 302.0 |
| latent_sc8 | 0.5550 | 0.5550 | 4.4 | 2366.0 |

Mean latent rollouts per problem: 8.00 (cap 8)
