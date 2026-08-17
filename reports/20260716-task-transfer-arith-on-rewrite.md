# Reasoning system comparison

| system | overall accuracy | mean emitted tokens | mean loop steps |
| --- | --- | --- | --- |
| direct | 0.2383 | 2.9 | - |
| cot | 1.0000 | 18.7 | - |
| latent_rrs_j_cfc | 0.9492 | 3.9 | 18.69 |

## Accuracy by chain length

| chain length | direct | cot | latent_rrs_j_cfc |
| --- | --- | --- | --- |
| 2 | 0.4848 | 1.0000 | 0.9697 |
| 3 | 0.1429 | 1.0000 | 0.9286 |
| 4 | 0.2326 | 1.0000 | 1.0000 |
| 5 | 0.3256 | 1.0000 | 0.9767 |
| 6 | 0.1429 | 1.0000 | 0.8857 |
| 7 | 0.2778 | 1.0000 | 0.9722 |
| 8 | 0.0526 | 1.0000 | 0.8947 |
