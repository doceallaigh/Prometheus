# GSM8K retrofit comparison

Model: `Qwen/Qwen2.5-0.5B-Instruct`, problems: 200

| system | strict accuracy | lenient accuracy | mean emitted tokens | mean internal rollout tokens |
| --- | --- | --- | --- | --- |
| direct | 0.0350 | 0.0350 | 7.4 | 7.4 |
| cot | 0.2750 | 0.4550 | 276.4 | 276.4 |
| latent | 0.4250 | 0.4500 | 3.9 | 279.3 |
| latent_sc8 | 0.5450 | 0.5450 | 4.5 | 2328.3 |

Mean latent rollouts per problem: 8.00 (cap 8)
