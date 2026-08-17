# GSM8K retrofit comparison

Model: `Qwen/Qwen2.5-0.5B-Instruct`, problems: 200, seed: 20260725

| system | strict accuracy | lenient accuracy | mean emitted tokens | mean internal rollout tokens |
| --- | --- | --- | --- | --- |
| direct | 0.0350 | 0.0350 | 7.4 | 7.4 |
| cot | 0.2750 | 0.4550 | 276.4 | 276.4 |
| latent | 0.4000 | 0.4050 | 4.3 | 298.8 |
| latent_sc8 | 0.5600 | 0.5600 | 4.5 | 2413.9 |

Mean latent rollouts per problem: 8.00 (cap 8)
