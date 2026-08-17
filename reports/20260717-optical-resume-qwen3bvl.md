# Optical context compression: resume fidelity vs compression

Model: `Qwen/Qwen2.5-VL-3B-Instruct`, problems: 100, prefix fraction: 0.6, font: 14px, mean prefix text tokens: 131.4

The model generates its own CoT; the first fraction of that chain is
handed back as text (upper bound), as a rendered image at decreasing
resolution (optical compression sweep), or omitted (lower bound), and
the model must resume to a final answer. Compression = mean prefix
text tokens / mean image tokens.

| arm | accuracy | prompt tokens | image tokens | compression |
| --- | --- | --- | --- | --- |
| full_cot | **0.7200** | 0 | 0 | - |
| text_resume | **0.7700** | 246 | 0 | - |
| no_prefix | **0.7600** | 122 | 0 | - |
| optical_1.0 | **0.7600** | 334 | 210 | 0.63x |
| optical_0.75 | **0.7600** | 247 | 123 | 1.07x |
| optical_0.5 | **0.7200** | 179 | 55 | 2.39x |
| optical_0.35 | **0.7400** | 151 | 27 | 4.87x |
