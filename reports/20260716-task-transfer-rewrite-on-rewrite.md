# Reasoning system comparison

| system | overall accuracy | mean emitted tokens | mean loop steps |
| --- | --- | --- | --- |
| direct | 0.2383 | 2.9 | - |
| cot | 1.0000 | 18.7 | - |
| latent_rrs_j_cfc | 1.0000 | 3.9 | 18.67 |

## Accuracy by chain length

| chain length | direct | cot | latent_rrs_j_cfc |
| --- | --- | --- | --- |
| 2 | 0.4848 | 1.0000 | 1.0000 |
| 3 | 0.1429 | 1.0000 | 1.0000 |
| 4 | 0.2326 | 1.0000 | 1.0000 |
| 5 | 0.3256 | 1.0000 | 1.0000 |
| 6 | 0.1429 | 1.0000 | 1.0000 |
| 7 | 0.2778 | 1.0000 | 1.0000 |
| 8 | 0.0526 | 1.0000 | 1.0000 |
