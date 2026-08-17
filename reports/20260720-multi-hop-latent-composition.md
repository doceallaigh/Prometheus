# Multi-Hop Latent Composition

## Question

Can useful computation be offloaded to a continuous sidecar that emits no intermediate tokens and returns only one final hidden state to a frozen trunk?

## Controlled benchmark

We generated 1,000 formal directed-relation proofs for training and 200 held-out proofs at each test depth from 2 through 10. Each problem specifies relation edges and asks for the entity reached from a query entity. Solving a depth-$k$ problem requires composing $k$ relations.

The no-CoT trunk receives only premises and the query. It is trained on depths 2-4, then frozen. The sidecar propagates a continuous entity distribution through the encoded relation graph for $N$ recurrent steps and maps the final state through a learned interface into the frozen upper trunk. It emits no intermediate tokens. Setting $N=0$ exactly recovers the direct frozen trunk, which is enforced by a unit test.

The sidecar interface is trained only on depths 2-4. Evaluation at depths 5-10 therefore tests whether the recurrent computation extends beyond the training horizon. We repeated the full experiment for seeds 20260720, 20260721, and 20260722, independently regenerating proofs and initializing models.

## Results

| proof depth | direct accuracy | matched-step accuracy | matched-step SD | best latent steps in three runs |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 0.182 | 0.985 | 0.000 | 2, 2, 2 |
| 3 | 0.177 | 0.987 | 0.003 | 3, 3, 3 |
| 4 | 0.130 | 0.988 | 0.006 | 4, 4, 4 |
| 5 | 0.105 | 0.983 | 0.006 | 5, 5, 5 |
| 6 | 0.090 | 0.978 | 0.003 | 6, 6, 6 |
| 7 | 0.102 | 0.983 | 0.008 | 7, 7, 7 |
| 8 | 0.082 | 0.977 | 0.032 | 8, 8, 8 |
| 9 | 0.070 | 0.977 | 0.014 | 9, 9, 9 |
| 10 | 0.068 | 0.980 | 0.013 | 10, 10, 10 |

Across OOD depths 5-10, mean direct accuracy was $0.0861 \pm 0.0048$, while matched-step latent accuracy was $0.9797 \pm 0.0086$. The mean absolute gain was $0.8936 \pm 0.0125$ (sample SD across three seeds).

The step sweep is the strongest mechanistic control. In all 27 seed-by-depth conditions, the best latent-step count exactly equaled proof depth. In the representative seed-20260720 run, depths 6-10 scored essentially zero when $N$ was shorter than the required chain, then rose to 0.980, 0.985, 1.000, 0.985, and 0.995 respectively at the matching step. The dependence on $N$ shows that the sidecar is performing sequential composition rather than supplying a static answer feature.

Interactive accuracy-versus-latent-step plot: [20260720-multi-hop-latent-composition.html](20260720-multi-hop-latent-composition.html)

Representative local run artifacts are under `outputs/latent-composition-relational-k4/`, including the generated report, metrics, formal proofs, and checkpoints.

## Negative control and interpretation

An unconstrained attention-GRU sidecar trained under the same 1,000-proof regime did not discover graph traversal: its OOD matched-step accuracy was 0.078 versus 0.083 for the direct trunk. The successful sidecar therefore uses an explicit relational propagation inductive bias; only its continuous interface to the frozen trunk is learned.

This experiment conclusively demonstrates the existence claim in the controlled formal setting: multi-step computation can occur entirely in a continuous side channel, cross a frozen-trunk interface once, and generalize from training depths 2-4 to depths 5-10. It does not establish that generic recurrent sidecars will learn such algorithms from small datasets, nor that the same result automatically transfers to pretrained language models or unconstrained natural-language reasoning.

## Reproduction

```powershell
$env:PYTHONPATH='src'
.\.venv\Scripts\python.exe -m prometheus.cli latent-composition `
  --output-dir outputs\latent-composition-relational-k4 `
  --train-proofs 1000 --test-per-depth 200 `
  --train-min-depth 2 --train-max-depth 4 --test-max-depth 10 `
  --entities 24 --distractors 4 --d-model 128 `
  --trunk-steps 1200 --sidecar-steps 1800 --batch-size 64 `
  --learning-rate 3e-4 --seed 20260720 --device cuda
```
