# Multi-turn frozen-trunk retrofit evaluation

Model: `Qwen/Qwen2.5-0.5B-Instruct`; corrector: `outputs\retrofit-qwen05b\corrector-7k\corrector.pt`; episodes: 17; turns: 3; GSM8K offset: 0; context mode: `dependent`.

Turn 1 solves GSM8K with an assistant-only context key in its scratch work; turn 2 recalls the key and turn 3 adds 7. The key is absent from the user prompt and latent surfaced answer.
Visible CoT retains prior visible chains; latent-kv-persist retains hidden-chain KV entries; latent-kv-reset rebuilds from surfaced answers. SC arms sample from an identical conversation snapshot and persist one majority-vote branch. All latent arms retain the CfC state across turns.

Generation time is synchronized device wall time around model decoding only; it excludes model/dataset loading, prompt tokenization, scoring, and report serialization.

| system | turn accuracy | all-turn episode accuracy | missing / exhausted | emitted tok/turn | internal tok/turn | rollouts/turn | sec/turn | internal tok/sec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| visible_cot | 0.5098 | 0.1176 | 0.1569 / 0.0196 | 85.1 | 83.8 | 1.00 | 1.204 | 69.6 |
| visible_sc8 | 0.5098 | 0.2353 | 0.0000 / 0.0000 | 90.3 | 660.5 | 8.00 | 9.702 | 68.1 |
| latent_kv_persist | 0.4706 | 0.0588 | 0.1176 / 0.0000 | 4.1 | 92.4 | 1.00 | 1.338 | 69.0 |
| latent_stop4of8 | 0.5098 | 0.1176 | 0.0000 / 0.0196 | 4.4 | 710.2 | 5.82 | 10.416 | 68.2 |
| latent_kv_reset | 0.1176 | 0.0000 | 0.1765 / 0.0196 | 4.2 | 99.2 | 1.00 | 1.437 | 69.0 |

End-to-end evaluation time after model and dataset loading: 1229.1 seconds.

Accuracy by turn:

- `visible_cot`: t1=0.2941, t2=0.7059, t3=0.5294
- `visible_sc8`: t1=0.5294, t2=0.5882, t3=0.4118
- `latent_kv_persist`: t1=0.3529, t2=0.6471, t3=0.4118
- `latent_stop4of8`: t1=0.4706, t2=0.7059, t3=0.3529
- `latent_kv_reset`: t1=0.3529, t2=0.0000, t3=0.0000

Context-dependent follow-up accuracy (turns 2-3):

- `visible_cot`: 0.6176
- `visible_sc8`: 0.5000
- `latent_kv_persist`: 0.5294
- `latent_stop4of8`: 0.5294
- `latent_kv_reset`: 0.0000

Paired turn outcomes:

- `latent_kv_persist` vs. `visible_cot`: 7 wins, 9 losses, 35 ties; accuracy delta -0.0392.
- `latent_kv_persist` vs. `latent_kv_reset`: 18 wins, 0 losses, 33 ties; accuracy delta +0.3529.
- `latent_kv_reset` vs. `visible_cot`: 4 wins, 24 losses, 23 ties; accuracy delta -0.3922.
- `latent_stop4of8` vs. `visible_sc8`: 6 wins, 6 losses, 39 ties; accuracy delta +0.0000.
- `latent_stop4of8` vs. `latent_kv_persist`: 10 wins, 8 losses, 33 ties; accuracy delta +0.0392.
- `latent_stop4of8` vs. `latent_kv_reset`: 22 wins, 2 losses, 27 ties; accuracy delta +0.3922.

## Follow-up SC analysis

- `latent_stop4of8` vs. `visible_sc8` on turns 2-3: 5 wins, 4 losses, 25 ties; accuracy delta +0.0294.
- Mean latent rollouts were 5.29 on recall turns and 4.88 on transform turns, versus 8 fixed visible rollouts.
- Post-hoc oracle coverage (any sampled answer correct) was 0.6765 latent and 0.7059 visible.
- Correct-sample rates were 0.4335 latent and 0.4779 visible; latent samples were unanimous on 0.5294 of follow-ups.

The latent-vs-visible SC difference is unresolved at this sample size. Adaptive SC matches greedy latent follow-up accuracy rather than improving it; its main observed benefit is fewer rollouts than fixed visible SC@8. The oracle gap indicates reranking headroom, while the high unanimity rate indicates correlated retrieval failures that larger sample counts alone will not fix.
