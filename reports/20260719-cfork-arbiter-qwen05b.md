# Complement-fork rollout: suppress in root, suppress-dominant in offshoot

Model: `Qwen/Qwen2.5-0.5B-Instruct`, corrector: `outputs/retrofit-qwen05b/corrector-7k/corrector.pt`, basis: `outputs/retrofit-qwen05b/jspace-basis-rank64.pt`, problems: 200
Fork z: 2.5, cap: 4, cooldown: 16, gamma: 1.0, persist: 4, child mode: suppress-dominant, steer mode: closed-loop, hull: True, arbiter: `outputs/retrofit-qwen05b/arbiter-v1/arbiter.pt`

Mean branches/problem: 3.60; problems forked: 191/200; forked problems where an offshoot answered differently: 133/191; mean internal tokens: 1239.1

| gate | accuracy (lenient) |
| --- | --- |
| latent | 0.4600 |
| root | 0.4900 |
| vote | 0.4600 |
| agree-else-mindelta | 0.3000 |
| min-delta | 0.3000 |
| min-intrusion | 0.3300 |
| max-logprob | 0.4200 |
| arbiter | 0.4900 |
| oracle | 0.5700 |
