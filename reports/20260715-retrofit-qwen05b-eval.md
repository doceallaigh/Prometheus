# GSM8K retrofit comparison

Model: `Qwen/Qwen2.5-0.5B-Instruct`, problems: 200

| system | accuracy | mean emitted tokens | mean internal rollout tokens |
| --- | --- | --- | --- |
| direct | 0.0350 | 7.4 | 7.4 |
| cot | 0.1050 | 276.4 | 276.4 |
| latent | 0.4050 | 4.1 | 302.2 |

Notes:

- Corrector: `outputs/retrofit-qwen05b/corrector-v1` (tap layer 12, d_cfc 512,
  3000 steps on 290 harvested traces; ~2.9M trainable params, trunk frozen).
- "Emitted tokens" counts what a user sees. For the latent system, by design
  only the final `#### <answer>` span is surfaced; the corrected chain rolls
  out internally and is reported under "internal rollout tokens".
- Both greedy (deterministic) eval passes over the same 200 test problems
  measured identical accuracies; the two token columns come from the
  answer-span and full-rollout counting passes respectively.
- The latent internal rollout (302.2) is slightly longer than the trunk's own
  visible CoT (276.4): compute is not saved, only the emitted channel.
