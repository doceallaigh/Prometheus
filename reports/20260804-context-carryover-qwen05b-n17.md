# Multi-turn frozen-trunk retrofit evaluation

Model: `Qwen/Qwen2.5-0.5B-Instruct`; corrector: `outputs\retrofit-qwen05b\corrector-7k\corrector.pt`; episodes: 17; turns: 3; GSM8K offset: 0; context mode: `dependent`.

Turn 1 solves GSM8K with an assistant-only context key in its scratch work; turn 2 recalls the key and turn 3 adds 7. The key is absent from the user prompt and latent surfaced answer.
Visible CoT retains prior visible chains; latent-kv-persist retains hidden-chain KV entries; latent-kv-reset rebuilds from surfaced answers. Both latent arms retain the CfC state across turns.

Generation time is synchronized device wall time around model decoding only; it excludes model/dataset loading, prompt tokenization, scoring, and report serialization.

| system | turn accuracy | all-turn episode accuracy | missing / exhausted | emitted tok/turn | internal tok/turn | sec/turn | internal tok/sec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| visible_cot | 0.5098 | 0.1176 | 0.1569 / 0.0196 | 85.1 | 83.8 | 1.206 | 69.4 |
| latent_kv_persist | 0.4706 | 0.0588 | 0.1176 / 0.0000 | 4.1 | 92.4 | 1.342 | 68.9 |
| latent_kv_reset | 0.1176 | 0.0000 | 0.1765 / 0.0196 | 4.2 | 99.2 | 1.439 | 68.9 |

End-to-end evaluation time after model and dataset loading: 203.5 seconds.

Accuracy by turn:

- `visible_cot`: t1=0.2941, t2=0.7059, t3=0.5294
- `latent_kv_persist`: t1=0.3529, t2=0.6471, t3=0.4118
- `latent_kv_reset`: t1=0.3529, t2=0.0000, t3=0.0000

Context-dependent follow-up accuracy (turns 2-3):

- `visible_cot`: 0.6176
- `latent_kv_persist`: 0.5294
- `latent_kv_reset`: 0.0000

Paired turn outcomes:

- `latent_kv_persist` vs. `visible_cot`: 7 wins, 9 losses, 35 ties; accuracy delta -0.0392.
- `latent_kv_persist` vs. `latent_kv_reset`: 18 wins, 0 losses, 33 ties; accuracy delta +0.3529.
- `latent_kv_reset` vs. `visible_cot`: 4 wins, 24 losses, 23 ties; accuracy delta -0.3922.
