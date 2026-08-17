# GSM8K retrofit comparison

Model: `Qwen/Qwen2.5-0.5B-Instruct`, problems: 200

| system | strict accuracy | lenient accuracy | mean emitted tokens | mean internal rollout tokens |
| --- | --- | --- | --- | --- |
| direct | 0.0350 | 0.0350 | 7.4 | 7.4 |
| cot | 0.2750 | 0.4550 | 276.4 | 276.4 |
| latent | 0.4300 | 0.4300 | 4.2 | 283.3 |
| latent_sc8 | 0.5250 | 0.5250 | 4.5 | 2343.6 |

Mean latent rollouts per problem: 8.00 (cap 8)

Corrector quorum: 3 members, noise 0.0, agg mean
