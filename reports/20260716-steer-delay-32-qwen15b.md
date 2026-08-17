# Steering-vector repair (inject, don't rewind)

Model: `Qwen/Qwen2.5-1.5B-Instruct`, dump: `reports/20260715-retrofit-qwen15b-eval.completions.jsonl`, trigger: oracle, steer window: 24 tokens, inject offset: 32 tokens

v = mean(h_tap[correct sibling] − h_tap[wrong]) at error-site offsets +1..+4,
from 184 greedy-correct-problem pairs (636 positions).
‖v‖ = 15.32, mean ‖h_tap‖ = 49.86 (ratio 0.3072).
The erroneous tokens stay in context; alpha·v is added to the tap layer's
residual stream for the steer window, then greedy decoding continues.
alpha=0 is the determinism control (must reproduce the original rollout).

Problems: 200, baseline greedy accuracy: 0.7150, wrong: 57, steer targets: 43 (no site: 9, no room: 5)

| alpha | final accuracy | flips up | flips down | reproduced exactly | regen tokens/problem |
| --- | --- | --- | --- | --- | --- |
| 4.0 | **0.7250** | 2 | 0 | 1/43 | 50.0 |
