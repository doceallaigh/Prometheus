# GSM8K retrofit comparison

Model: `Qwen/Qwen2.5-0.5B-Instruct`, problems: 200

| system | strict accuracy | lenient accuracy | mean emitted tokens | mean internal rollout tokens |
| --- | --- | --- | --- | --- |
| direct | 0.0350 | 0.0350 | 7.4 | 7.4 |
| cot | 0.2750 | 0.4550 | 276.4 | 276.4 |
| latent | 0.4200 | 0.4200 | 4.2 | 282.3 |
| latent_sc8 | 0.5150 | 0.5150 | 4.6 | 2364.6 |

Mean latent rollouts per problem: 8.00 (cap 8)

Corrector quorum: 3 members, noise 0.0, agg sign (agree >= 1.0)
