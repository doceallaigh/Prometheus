# Complement-fork rollout: suppress in root, suppress-dominant in offshoot

Model: `Qwen/Qwen2.5-0.5B-Instruct`, corrector: `outputs/retrofit-qwen05b/corrector-7k/corrector.pt`, basis: `outputs/retrofit-qwen05b/jspace-basis-rank64.pt`, problems: 200
Fork z: 2.5, cap: 4, cooldown: 16, gamma: 1.0, persist: 4, child mode: suppress-dominant, steer mode: closed-loop, hull: True, tap snap: `outputs/retrofit-qwen05b/tap-snap/tap-snap.pt`

Mean branches/problem: 3.49; problems forked: 191/200; forked problems where an offshoot answered differently: 116/191; mean internal tokens: 997.7

| gate | accuracy (lenient) |
| --- | --- |
| latent | 0.4600 |
| root | 0.4250 |
| vote | 0.4350 |
| agree-else-mindelta | 0.3700 |
| min-delta | 0.3700 |
| min-intrusion | 0.3750 |
| max-logprob | 0.4050 |
| oracle | 0.5350 |
