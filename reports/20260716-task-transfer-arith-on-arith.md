# Reasoning system comparison

| system | overall accuracy | mean emitted tokens | mean loop steps |
| --- | --- | --- | --- |
| direct | 0.0273 | 2.9 | - |
| cot | 0.7773 | 18.8 | - |
| latent_rrs_j_cfc | 0.9766 | 3.9 | 18.68 |

## Accuracy by chain length

| chain length | direct | cot | latent_rrs_j_cfc |
| --- | --- | --- | --- |
| 2 | 0.0263 | 0.9474 | 0.9737 |
| 3 | 0.0800 | 0.9600 | 1.0000 |
| 4 | 0.0270 | 0.8919 | 0.9189 |
| 5 | 0.0000 | 0.7561 | 1.0000 |
| 6 | 0.0000 | 0.6316 | 0.9737 |
| 7 | 0.0465 | 0.7442 | 0.9767 |
| 8 | 0.0294 | 0.5588 | 1.0000 |
