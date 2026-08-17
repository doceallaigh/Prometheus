# Residual-stream tap verification (Jacobian projection)

Model: `Qwen/Qwen2.5-0.5B-Instruct`, tap layer: 12, traces: 8, positions/trace: 8, directions: 4

## 1. Tap authenticity (staged re-run of layers[tap:])

- max relative error, tap state vs reference: 0.00e+00
- max relative error, reconstructed h_final vs reference: 0.00e+00
- logit argmax agreement: 2933/2933 (100.00%)

## 2. Influence subspace (row space of J_t = dh_final_t/dh_tap_t)

- sampled local rows: 252, attention-mediated rows: 756 (d = 896)
- effective rank (participation ratio): local 142.3, full 278.5
- energy captured at rank 64: local 56.29%, full 35.95%
- influence mass via other positions (attention-mediated): 61.45%

## 3. Alignment with the top-64 influence basis (random baseline 0.071)

- tap-state energy inside the influence subspace: local 0.155, full 0.178
- corrector read-direction energy inside the influence subspace: local 0.087, full 0.084

## 4. Projection ablation of the corrector input (trunk stream untouched)

| variant | tap energy kept | delta cosine | delta norm ratio | active-token agreement |
| --- | --- | --- | --- | --- |
| keep-local | 0.157 | 0.714 | 1.151 | 0.506 |
| remove-local | 0.843 | 0.964 | 0.990 | 0.888 |
| keep-full | 0.180 | 0.731 | 1.090 | 0.483 |
| remove-full | 0.820 | 0.950 | 1.008 | 0.876 |
| keep-random | 0.067 | 0.732 | 1.233 | 0.483 |
| remove-random | 0.933 | 0.986 | 1.011 | 0.899 |

Active positions (correction changes argmax token): 89 of 1923 completion positions.

Interpretation: (1) near-zero reconstruction error proves the tapped
tensor is the exact residual stream the upper trunk consumes; (2) the
spectrum measures how low-dimensional the functionally live subspace is
(local = same-position block; full adds the attention-mediated rows,
i.e. how earlier tap states influence later outputs via keys/values);
(3) low corrector alignment means the corrector's read directions live
largely in the complement of the trunk's dominant Jacobian subspace;
(4) if corrector function survives `remove` but not `keep`, restricting
the interface to a concept-aligned (Jacobian-isolated) subspace would
discard exactly the signal the corrector monitors.
