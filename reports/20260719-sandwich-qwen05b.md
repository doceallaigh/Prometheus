# Sandwich rollout: discrete encoder | latent recurrence | discrete decoder

Model: `Qwen/Qwen2.5-0.5B-Instruct`, dynamics: `outputs/retrofit-qwen05b/dynamics-7k/dynamics.pt`, problems: 200, latent steps: 300

| system | strict accuracy | lenient accuracy | mean internal tokens |
| --- | --- | --- | --- |
| cot | 0.2750 | 0.4550 | 276.4 |
| sandwich | 0.0350 | 0.0350 | 273.7 |
