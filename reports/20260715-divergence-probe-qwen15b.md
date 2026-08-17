# Divergence recognition curve

Model: `Qwen/Qwen2.5-1.5B-Instruct`, corrector: `outputs/retrofit-qwen15b/corrector-v1/corrector.pt`, pairs: 451

Probe: 5-fold CV logistic regression, wrong rollout vs correct sibling,
features taken k tokens past the token-level fork point.

| tokens past fork | n | AUC h_tap | AUC s_t | AUC both |
| --- | --- | --- | --- | --- |
| 1 | 902 | 0.504 | 0.504 | 0.504 |
| 2 | 902 | 0.499 | 0.438 | 0.486 |
| 4 | 902 | 0.543 | 0.529 | 0.555 |
| 8 | 902 | 0.501 | 0.461 | 0.494 |
| 16 | 902 | 0.478 | 0.510 | 0.476 |
| 32 | 902 | 0.542 | 0.509 | 0.545 |
| 64 | 902 | 0.529 | 0.524 | 0.531 |
