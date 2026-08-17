# Reasoning system comparison

| system | overall accuracy | mean emitted tokens | mean loop steps |
| --- | --- | --- | --- |
| direct | 0.0273 | 2.9 | - |
| cot | 0.7773 | 18.8 | - |
| latent_rrs_j_cfc | 0.3164 | 3.9 | 18.66 |

## Accuracy by chain length

| chain length | direct | cot | latent_rrs_j_cfc |
| --- | --- | --- | --- |
| 2 | 0.0263 | 0.9474 | 0.6053 |
| 3 | 0.0800 | 0.9600 | 0.4000 |
| 4 | 0.0270 | 0.8919 | 0.3243 |
| 5 | 0.0000 | 0.7561 | 0.1951 |
| 6 | 0.0000 | 0.6316 | 0.3421 |
| 7 | 0.0465 | 0.7442 | 0.1628 |
| 8 | 0.0294 | 0.5588 | 0.2353 |
