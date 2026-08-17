# Grounded continuous latent reasoning (periodic re-anchoring)

Model: `Qwen/Qwen2.5-0.5B-Instruct`, corrector: `outputs/retrofit-qwen05b/corrector-7k/corrector.pt`, feedback: expected, problems: 200, reasoning steps: 400

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
| 1 | **0.4500** | 276.3 | 183 | 0 | 17 |
| 4 | **0.3900** | 251.3 | 51 | 133 | 16 |
| 8 | **0.3050** | 242.0 | 24 | 160 | 16 |
| 16 | **0.3500** | 247.9 | 9 | 162 | 29 |
| never | **0.3200** | 249.8 | 0 | 158 | 42 |
