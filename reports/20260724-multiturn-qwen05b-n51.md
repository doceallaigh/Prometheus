# Independent-turn context-interference baseline

Model: `Qwen/Qwen2.5-0.5B-Instruct`; corrector: `outputs\retrofit-qwen05b\corrector-7k\corrector.pt`; episodes: 17; turns: 3; GSM8K offset: 0.

Each episode contains independent GSM8K problems. This baseline measures accumulated-context interference and does not test context-dependent follow-ups. Visible CoT retains prior visible chains; latent-kv-persist retains hidden-chain KV entries; latent-kv-reset rebuilds from surfaced answers. Both latent arms retain the CfC state across turns.

Generation time is synchronized device wall time around model decoding only; it excludes model/dataset loading, prompt tokenization, scoring, and report serialization.

| system | turn accuracy | all-turn episode accuracy | missing / exhausted | emitted tok/turn | internal tok/turn | sec/turn | internal tok/sec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| visible_cot | 0.3725 | 0.1765 | 0.3529 / 0.0588 | 296.9 | 297.9 | 4.259 | 69.9 |
| latent_kv_persist | 0.4706 | 0.1176 | 0.0784 / 0.0784 | 4.1 | 312.7 | 4.513 | 69.3 |
| latent_kv_reset | 0.3922 | 0.0588 | 0.0588 / 0.0392 | 4.2 | 286.5 | 4.136 | 69.3 |

End-to-end evaluation time after model and dataset loading: 658.4 seconds.

Accuracy by turn:

- `visible_cot`: t1=0.4118, t2=0.4706, t3=0.2353
- `latent_kv_persist`: t1=0.5294, t2=0.5294, t3=0.3529
- `latent_kv_reset`: t1=0.5294, t2=0.4706, t3=0.1765

Paired turn outcomes:

- `latent_kv_persist` vs. `visible_cot`: 8 wins, 3 losses, 40 ties; accuracy delta +0.0980.
- `latent_kv_persist` vs. `latent_kv_reset`: 4 wins, 0 losses, 47 ties; accuracy delta +0.0784.
- `latent_kv_reset` vs. `visible_cot`: 6 wins, 5 losses, 40 ties; accuracy delta +0.0196.
