# Grounded continuous latent reasoning (periodic re-anchoring)

Model: `Qwen/Qwen2.5-1.5B-Instruct`, corrector: `outputs/retrofit-qwen15b/corrector-7k/corrector.pt`, feedback: expected, problems: 200, reasoning steps: 400

Continuous (Coconut-style) latent steps feed a vector back as the
next input embedding; every G-th step decodes a real greedy token
to re-anchor the trajectory on the token manifold. G=1 is the
ordinary latent rollout (control); G=0 never grounds (pure
continuous). Termination is natural: every step's argmax is a
shadow token (discarded during continuous steps) monitored for the
'####' answer marker — natural = marker emitted on a grounded step,
detected = marker seen in the shadow stream mid-continuous (anchor
injected), budget = step cap hit (anchor injected).

| ground every | accuracy (lenient) | mean internal steps | natural | detected | budget |
| --- | --- | --- | --- | --- | --- |
| 1 | **0.7250** | 247.5 | 193 | 0 | 7 |
| 4 | **0.6650** | 232.0 | 50 | 135 | 15 |
| 8 | **0.6300** | 234.5 | 23 | 156 | 21 |
| 16 | **0.6250** | 229.9 | 14 | 164 | 22 |
| never | **0.5900** | 230.6 | 0 | 176 | 24 |
