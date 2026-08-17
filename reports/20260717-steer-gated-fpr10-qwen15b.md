# Steering-vector repair (inject, don't rewind)

Model: `Qwen/Qwen2.5-1.5B-Instruct`, dump: `reports/20260715-retrofit-qwen15b-eval.completions.jsonl`, trigger: cusum, steer window: 24 tokens, inject offset: 4 tokens, rollout FPR: 0.1, gate: margin

v = mean(h_tap[correct sibling] − h_tap[wrong]) at error-site offsets +1..+4,
from 184 greedy-correct-problem pairs (636 positions).
‖v‖ = 15.32, mean ‖h_tap‖ = 49.86 (ratio 0.3072).
The erroneous tokens stay in context; alpha·v is added to the tap layer's
residual stream for the steer window, then greedy decoding continues.
alpha=0 is the determinism control (must reproduce the original rollout).

Problems: 200, baseline greedy accuracy: 0.7150, wrong: 57, steer targets: 39 (false alarms: 19, on wrong: 20, no room: 0, mean gate false-alarm/hit: 0.680/0.825)

| alpha | final accuracy | flips up | flips down | reproduced exactly | regen tokens/problem |
| --- | --- | --- | --- | --- | --- |
| 0.0 | **0.7150** | 0 | 0 | 35/39 | 24.2 |
| 2.0 | **0.7200** | 2 | 1 | 6/39 | 21.9 |
| 4.0 | **0.7000** | 1 | 4 | 2/39 | 30.7 |
