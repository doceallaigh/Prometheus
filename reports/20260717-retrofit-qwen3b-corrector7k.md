# GSM8K retrofit comparison

Model: `Qwen/Qwen2.5-3B-Instruct`, problems: 200

| system | strict accuracy | lenient accuracy | mean emitted tokens | mean internal rollout tokens |
| --- | --- | --- | --- | --- |
| direct | 0.0750 | 0.0900 | 7.1 | 7.1 |
| cot | 0.7850 | 0.8150 | 299.8 | 299.8 |
| latent | 0.7900 | 0.8150 | 4.2 | 290.8 |
| latent_sc8 | 0.8700 | 0.8700 | 4.4 | 2359.2 |

Mean latent rollouts per problem: 8.00 (cap 8)
