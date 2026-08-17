# GSM8K retrofit comparison

Model: `Qwen/Qwen2.5-0.5B-Instruct`, problems: 30

| system | strict accuracy | lenient accuracy | mean emitted tokens | mean internal rollout tokens |
| --- | --- | --- | --- | --- |
| direct | 0.0000 | 0.0000 | 6.3 | 6.3 |
| cot | 0.0667 | 0.3667 | 289.1 | 289.1 |
| latent | 0.2667 | 0.2667 | 4.0 | 309.8 |
| latent_asc2of8 | 0.3667 | 0.3667 | 4.5 | 1541.6 |

Mean latent rollouts per problem: 4.80 (cap 8, stop at 2 agreeing)
