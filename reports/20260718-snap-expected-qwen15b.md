# Grounded continuous latent reasoning (periodic re-anchoring)

Model: `Qwen/Qwen2.5-1.5B-Instruct`, corrector: `outputs/retrofit-qwen15b/corrector-7k/corrector.pt`, feedback: expected, snap: `outputs/retrofit-qwen15b/snap-expected.pt`, problems: 200, reasoning steps: 400

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
| 8 | **0.6550** | 236.9 | 23 | 161 | 16 |
| 16 | **0.6300** | 236.9 | 11 | 174 | 15 |
| never | **0.6700** | 239.2 | 0 | 179 | 21 |
