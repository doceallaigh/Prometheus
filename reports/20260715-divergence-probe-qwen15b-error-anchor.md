# Divergence recognition curve

Model: `Qwen/Qwen2.5-1.5B-Instruct`, corrector: `outputs/retrofit-qwen15b/corrector-v1/corrector.pt`, pairs: 371, anchor: error (skipped 80 unlabeled)

Probe: 5-fold CV logistic regression, wrong rollout vs correct sibling,
features taken k tokens past the arithmetic error site.

| tokens past anchor | n | AUC h_tap | AUC s_t | AUC both |
| --- | --- | --- | --- | --- |
| 1 | 707 | 0.985 | 0.958 | 0.985 |
| 2 | 707 | 0.984 | 0.930 | 0.983 |
| 4 | 707 | 0.967 | 0.794 | 0.965 |
| 8 | 702 | 0.840 | 0.703 | 0.834 |
| 16 | 692 | 0.698 | 0.514 | 0.689 |
| 32 | 640 | 0.474 | 0.472 | 0.466 |
| 64 | 575 | 0.562 | 0.523 | 0.563 |
