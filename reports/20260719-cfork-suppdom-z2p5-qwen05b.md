# Complement-fork rollout: suppress in root, suppress-dominant in offshoot

Model: `Qwen/Qwen2.5-0.5B-Instruct`, corrector: `outputs/retrofit-qwen05b/corrector-7k/corrector.pt`, basis: `outputs/retrofit-qwen05b/jspace-basis-rank64.pt`, problems: 200
Fork z: 2.5, cap: 4, cooldown: 16, gamma: 1.0, persist: 4, child mode: suppress-dominant

Mean branches/problem: 3.52; problems forked: 191/200; forked problems where an offshoot answered differently: 99/191; mean internal tokens: 1051.4

| gate | accuracy (lenient) |
| --- | --- |
| latent | 0.4600 |
| root | 0.4500 |
| vote | 0.4450 |
| agree-else-mindelta | 0.3850 |
| min-delta | 0.3850 |
| min-intrusion | 0.3950 |
| max-logprob | 0.4400 |
| oracle | 0.5350 |
