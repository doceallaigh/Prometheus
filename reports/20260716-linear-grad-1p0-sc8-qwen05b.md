# GSM8K retrofit comparison

Model: `Qwen/Qwen2.5-0.5B-Instruct`, problems: 200

| system | strict accuracy | lenient accuracy | mean emitted tokens | mean internal rollout tokens |
| --- | --- | --- | --- | --- |
| direct | 0.0350 | 0.0350 | 7.4 | 7.4 |
| cot | 0.1050 | 0.4550 | 276.4 | 276.4 |
| latent | 0.1050 | 0.4550 | 1.1 | 276.2 |
| latent_sc8 | 0.5200 | 0.5200 | 4.5 | 2200.5 |

Mean latent rollouts per problem: 8.00 (cap 8)
