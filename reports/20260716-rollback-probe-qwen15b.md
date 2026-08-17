# Probe-triggered rollback (deployable rule)

Model: `Qwen/Qwen2.5-1.5B-Instruct`, dump: `reports/20260715-retrofit-qwen15b-eval.completions.jsonl`, budget: 4, temperature: 0.6, rewind margin: 8 tokens, calibrated rollout FPR: 0.05

Trigger: logistic probe on h_tap, trained on error-site sibling pairs from
problems whose greedy rollout is correct (problem-disjoint from all rollback
targets). No gold labels are used at inference time.

| metric | value |
| --- | --- |
| problems | 200 |
| probe training pairs | 184 |
| baseline greedy accuracy | 0.7150 |
| **final accuracy** | **0.7050** |
| wrong rollouts | 57 |
| triggered on wrong (recall) | 8/57 |
| triggered on correct (false alarms) | 6/143 |
| re-rolls accepted | 2 |
| flips wrong→correct | 0 |
| flips correct→wrong | 2 |
| mean re-rolls per problem | 0.26 |
| mean re-roll tokens per problem | 33.9 |
