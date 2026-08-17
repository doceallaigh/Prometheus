# Answer-phase error monitoring

Model: `Qwen/Qwen2.5-0.5B-Instruct`, corrector: `outputs/retrofit-qwen05b/corrector-7k/corrector.pt`, problems: 200, samples: 8 @ T=0.6
Greedy rollouts with `####` marker: 181/200; mean answer-span length 17.2 tokens; sequential-accept threshold 39.46 (median of correct greedy)

## (A) Does the answer-span signal predict a wrong answer? (rank AUC, greedy arm)

| statistic | AUC |
| --- | --- |
| ans_mean_delta | 0.4057 |
| ans_max_delta | 0.4308 |
| ans_mean_frac | 0.5427 |
| ans_z | 0.5365 |
| chain_mean_delta | 0.3435 |

## (B) Answer-span-gated policies over the sampled rollouts

| policy | accuracy | mean rollouts |
| --- | --- | --- |
| vote | 0.5400 | 8 |
| min-ans-delta | 0.3700 | 8 |
| weighted-vote | 0.5150 | 8 |
| seq-accept | 0.4050 | 2.04 |
| greedy | 0.4600 | 1 |
