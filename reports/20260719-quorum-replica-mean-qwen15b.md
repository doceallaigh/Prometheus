# GSM8K retrofit comparison

Model: `Qwen/Qwen2.5-1.5B-Instruct`, problems: 200

| system | strict accuracy | lenient accuracy | mean emitted tokens | mean internal rollout tokens |
| --- | --- | --- | --- | --- |
| direct | 0.0000 | 0.0650 | 2.7 | 2.7 |
| cot | 0.2300 | 0.6500 | 254.0 | 254.0 |
| latent | 0.6900 | 0.7250 | 4.1 | 247.3 |
| latent_sc8 | 0.7950 | 0.7950 | 4.4 | 1956.0 |

Mean latent rollouts per problem: 8.00 (cap 8)

Corrector quorum: 4 members, noise 0.05, agg mean
