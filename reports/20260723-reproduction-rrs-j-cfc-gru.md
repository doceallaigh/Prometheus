# Reasoning system comparison

| system | overall accuracy | mean emitted tokens | mean loop steps |
| --- | --- | --- | --- |
| direct | 0.0300 | 2.9 | - |
| cot | 0.5033 | 19.1 | - |
| latent_rrs_j_cfc | 0.8200 | 3.9 | 18.94 |

## Accuracy by chain length

| chain length | direct | cot | latent_rrs_j_cfc |
| --- | --- | --- | --- |
| 2 | 0.0444 | 0.6889 | 0.8889 |
| 3 | 0.0357 | 0.6786 | 0.9286 |
| 4 | 0.0500 | 0.6500 | 0.8250 |
| 5 | 0.0204 | 0.5510 | 0.8571 |
| 6 | 0.0227 | 0.3182 | 0.7955 |
| 7 | 0.0208 | 0.3958 | 0.6875 |
| 8 | 0.0217 | 0.3261 | 0.8043 |
