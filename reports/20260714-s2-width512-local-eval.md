# Reasoning system comparison

| system | overall accuracy | mean emitted tokens | mean loop steps |
| --- | --- | --- | --- |
| direct | 0.0233 | 2.9 | - |
| cot | 0.9833 | 18.8 | - |
| latent_rrs_j_cfc | 0.9967 | 3.9 | 18.82 |

## Accuracy by chain length

| chain length | direct | cot | latent_rrs_j_cfc |
| --- | --- | --- | --- |
| 2 | 0.0222 | 0.9778 | 1.0000 |
| 3 | 0.0000 | 1.0000 | 1.0000 |
| 4 | 0.0250 | 0.9750 | 1.0000 |
| 5 | 0.0204 | 1.0000 | 1.0000 |
| 6 | 0.0000 | 0.9773 | 0.9773 |
| 7 | 0.0417 | 0.9792 | 1.0000 |
| 8 | 0.0435 | 0.9783 | 1.0000 |
