# Reasoning system comparison

| system | overall accuracy | mean emitted tokens | mean loop steps |
| --- | --- | --- | --- |
| direct | 0.0300 | 2.9 | - |
| cot | 0.5033 | 19.1 | - |
| latent_rrs_j_cfc | 0.5567 | 3.9 | 19.10 |

## Accuracy by chain length

| chain length | direct | cot | latent_rrs_j_cfc |
| --- | --- | --- | --- |
| 2 | 0.0444 | 0.6889 | 0.7333 |
| 3 | 0.0357 | 0.6786 | 0.7500 |
| 4 | 0.0500 | 0.6500 | 0.6750 |
| 5 | 0.0204 | 0.5510 | 0.5918 |
| 6 | 0.0227 | 0.3182 | 0.4091 |
| 7 | 0.0208 | 0.3958 | 0.4792 |
| 8 | 0.0217 | 0.3261 | 0.3478 |
