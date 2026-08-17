# Complement-fork rollout: suppress in root, suppress-dominant in offshoot

Model: `Qwen/Qwen2.5-0.5B-Instruct`, corrector: `outputs/retrofit-qwen05b/corrector-7k/corrector.pt`, basis: `outputs/retrofit-qwen05b/jspace-basis-rank64.pt`, problems: 20
Fork z: 50.0, cap: 4, cooldown: 16, gamma: 1.0, persist: 4, child mode: suppress-dominant

Mean branches/problem: 1.00; problems forked: 0/20; forked problems where an offshoot answered differently: 0/1; mean internal tokens: 295.8

| gate | accuracy (lenient) |
| --- | --- |
| latent | 0.4500 |
| root | 0.4500 |
| vote | 0.4500 |
| agree-else-mindelta | 0.4500 |
| min-delta | 0.4500 |
| min-intrusion | 0.4500 |
| max-logprob | 0.4500 |
| oracle | 0.4500 |
