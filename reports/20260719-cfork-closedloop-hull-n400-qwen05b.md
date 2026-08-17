# Complement-fork rollout: suppress in root, suppress-dominant in offshoot

Model: `Qwen/Qwen2.5-0.5B-Instruct`, corrector: `outputs/retrofit-qwen05b/corrector-7k/corrector.pt`, basis: `outputs/retrofit-qwen05b/jspace-basis-rank64.pt`, problems: 400
Fork z: 2.5, cap: 4, cooldown: 16, gamma: 1.0, persist: 4, child mode: suppress-dominant, steer mode: closed-loop, hull: True

Mean branches/problem: 3.63; problems forked: 381/400; forked problems where an offshoot answered differently: 284/381; mean internal tokens: 1251.8

| gate | accuracy (lenient) |
| --- | --- |
| latent | 0.4475 |
| root | 0.4525 |
| vote | 0.4300 |
| agree-else-mindelta | 0.2775 |
| min-delta | 0.2775 |
| min-intrusion | 0.3025 |
| max-logprob | 0.4000 |
| oracle | 0.5500 |
