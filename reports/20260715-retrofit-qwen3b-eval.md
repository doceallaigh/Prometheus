# GSM8K retrofit comparison

Model: `Qwen/Qwen2.5-3B-Instruct`, problems: 200

| system | strict accuracy | lenient accuracy | mean emitted tokens | mean internal rollout tokens |
| --- | --- | --- | --- | --- |
| direct | 0.0750 | 0.0900 | 7.1 | 7.1 |
| cot | 0.7850 | 0.8150 | 299.8 | 299.8 |
| latent | 0.8000 | 0.8050 | 4.2 | 288.3 |
| latent_sc8 | 0.8850 | 0.8850 | 4.4 | 2377.9 |
