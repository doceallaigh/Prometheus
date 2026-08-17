# Divergence recognition curve

Model: `Qwen/Qwen2.5-0.5B-Instruct`, corrector: `outputs/retrofit-qwen05b/corrector-v1/corrector.pt`, pairs: 520

Probe: 5-fold CV logistic regression, wrong rollout vs correct sibling,
features taken k tokens past the token-level fork point.

| tokens past fork | n | AUC h_tap | AUC s_t | AUC both |
| --- | --- | --- | --- | --- |
| 1 | 1040 | 0.503 | 0.503 | 0.503 |
| 2 | 1040 | 0.513 | 0.508 | 0.500 |
| 4 | 1040 | 0.519 | 0.496 | 0.512 |
| 8 | 1040 | 0.452 | 0.541 | 0.458 |
| 16 | 1040 | 0.568 | 0.532 | 0.563 |
| 32 | 1040 | 0.564 | 0.505 | 0.553 |
| 64 | 1040 | 0.540 | 0.469 | 0.539 |
