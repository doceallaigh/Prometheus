# Grounded continuous latent reasoning (periodic re-anchoring)

Model: `Qwen/Qwen2.5-0.5B-Instruct`, corrector: `outputs/retrofit-qwen05b/corrector-7k/corrector.pt`, feedback: expected, snap: `outputs/retrofit-qwen05b/snap-expected.pt`, problems: 200, reasoning steps: 400

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
| 8 | **0.3600** | 255.7 | 23 | 161 | 16 |
| 16 | **0.3700** | 251.7 | 13 | 165 | 22 |
| never | **0.3450** | 257.0 | 0 | 168 | 32 |
