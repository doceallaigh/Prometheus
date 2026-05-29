# Prometheus Overnight Report

## Scope

This report continues the Phase 1 feasibility pass described in [PLAN.md](../PLAN.md) and [ROADMAP.md](../ROADMAP.md).

Work completed in this session:

1. Revalidated the training and reporting stack in the repo-local virtual environment.
2. Fixed one artifact-tracking defect so saved configs now record resolved runtime values.
3. Reproduced fresh smoke runs on GPU.
4. Ran the original 300-step dense and modular comparison.
5. Identified that the original modular variants were badly under parameter budget relative to the dense baseline.
6. Added and ran a parameter-matched modular benchmark set.
7. Added and ran one focused follow-up ablation on uneven branching plus sparse routing.

## Environment Notes

1. The reliable interpreter for this repository is `.venv\\Scripts\\python.exe`.
2. Shell activation alone can still leave `python` on the Conda base interpreter.
3. CUDA was available in the repo venv through `torch 2.6.0+cu124` on an RTX 3090.
4. Torch emitted a non-blocking startup warning because `numpy` is not installed in the venv.

## Code Change Made

The training harness previously wrote unresolved config values to `config.snapshot.json`, including `model.vocab_size: auto` and the requested device rather than the actual resolved runtime device.

That is now fixed.

Relevant code and test:

1. [src/prometheus/train.py](../src/prometheus/train.py)
2. [tests/test_train_reporting_inference.py](../tests/test_train_reporting_inference.py)

## Roadmap Status

### Completed or mostly completed

1. Build the experimental foundation
2. Implement the dense baseline
3. Implement the first architectural variants
4. Run a first controlled benchmark pass on the available language-modeling task

### Incomplete relative to roadmap

1. The benchmark suite still covers only language-modeling loss on the synthetic corpus.
2. Long-context retrieval and routing-heavy reasoning tasks are not yet implemented.
3. The harness does not yet log throughput, memory use, routing cost, or active edge counts as first-class metrics.
4. The current sparse mechanism is a simple fixed topology plus optional top-k route restriction, not a stronger sufficient-set pruning implementation.

## Fresh Baseline and Early Variant Runs

The original 300-step comparison was directionally useful but unfair on parameter count.

| run | params | best val loss | note |
| --- | ---: | ---: | --- |
| baseline-tiny-20260529-053434 | 2,699,328 | 0.3189 | dense baseline |
| variant-modular-dense-20260529-053504 | 812,469 | 2.5771 | underpowered modular dense |
| variant-modular-sparse-20260529-053446 | 812,469 | 2.8828 | underpowered modular sparse |

Interpretation:

1. These runs should not be used as the main investment comparison because the modular models had only about 30 percent of the dense baseline parameter budget.
2. They were still useful for surfacing the fairness problem and motivating a corrected pass.

## Fair Matched Benchmark Pass

The following matched runs were executed at approximately the same parameter count as the dense baseline.

Reference comparison artifact:

1. [outputs/comparison-matched-20260529-0548.md](./comparison-matched-20260529-0548.md)

### Results

| run | params | topology | groups | best val loss | best val ppl | wall seconds |
| --- | ---: | --- | --- | ---: | ---: | ---: |
| baseline-tiny-20260529-054742 | 2,699,328 | dense baseline | n/a | 0.3189 | 1.3756 | 4.51 |
| variant-modular-dense-matched-20260529-054630 | 2,672,229 | dense routing | [4,2,1] | 2.0042 | 7.4205 | 6.19 |
| variant-modular-sparse-matched-20260529-054655 | 2,672,229 | small-world top-k=2 | [4,2,1] | 2.1817 | 8.8617 | 5.91 |
| variant-modular-uneven-matched-20260529-054701 | 2,719,581 | dense routing | [8,4,1] | 2.0627 | 7.8674 | 7.71 |
| variant-modular-uneven-sparse-matched-20260529-054809 | 2,719,581 | small-world top-k=2 | [8,4,1] | 2.6187 | 13.7175 | 7.85 |

### Key Findings

1. Parameter matching matters a lot. The balanced modular dense run improved from 2.5771 to 2.0042 best validation loss once it was given a fairer parameter budget.
2. Even after parameter matching, the dense baseline remained decisively better on the only benchmark currently implemented.
3. Sparse routing hurt performance relative to dense routing at the same modular parameter budget.
4. Uneven branching did not help in the current implementation. The `[8,4,1]` dense-routing model was slightly worse than the balanced `[4,2,1]` dense-routing model, and the uneven sparse model was substantially worse.
5. The dense baseline was also faster in coarse wall-clock training time than every matched modular variant that was tested.

## Current Ranking by Evidence Strength

Within the currently implemented modular family:

1. Balanced modular dense routing: strongest modular result so far
2. Balanced modular sparse routing: worse than balanced dense routing
3. Uneven modular dense routing: slightly worse than balanced dense routing
4. Uneven modular sparse routing: weakest matched variant tested

Across all tested architectures:

1. Dense baseline remains clearly best on synthetic language modeling at this budget.

## Interpretation Against the Plan

The plan requires a narrow but credible signal before additional complexity earns more funding.

At the moment, the evidence says:

1. The current modular implementations do train when given comparable capacity.
2. The large original gap was exaggerated by an unfair parameter budget.
3. The fairness correction does not reverse the conclusion.
4. On the available task, there is still no win for modularity, sparsity, or uneven branching.

That means the current evidence does **not** support a go-forward claim that hierarchical sparse modularity improves capability per parameter or capability per unit compute on this benchmark.

## Recommendation

Recommended decision category from the roadmap: **simplify and retest**, not scale.

Concretely:

1. Do not spend scale-up budget on the current sparse or uneven-branching variants.
2. If more budget is approved, spend it on focused iteration around the strongest surviving mechanism, which is the balanced modular dense-routing stack.
3. Treat sparse routing as currently unsupported, and uneven branching as currently negative on this benchmark.
4. Expand the benchmark suite before making any stronger architectural claims.

## Highest-Value Next Work Before Bigger Spend

1. Add routing-cost, active-edge, throughput, and memory metrics to the training and reporting harness.
2. Add at least one long-context retrieval benchmark and one routing-heavy composition benchmark.
3. Debug why the modular stack remains so far behind the dense baseline at equal parameters.
4. Only after that, revisit sparse routing or stronger sufficient-set pruning.

## Artifacts Produced This Session

### Reports

1. [outputs/comparison-20260529-0535.md](./comparison-20260529-0535.md)
2. [outputs/comparison-matched-20260529-0548.md](./comparison-matched-20260529-0548.md)
3. [outputs/overnight-report-20260529.md](./overnight-report-20260529.md)

### Fresh run directories

1. [outputs/smoke-tiny-20260529-053407](./smoke-tiny-20260529-053407)
2. [outputs/smoke-modular-20260529-053415](./smoke-modular-20260529-053415)
3. [outputs/baseline-tiny-20260529-053434](./baseline-tiny-20260529-053434)
4. [outputs/variant-modular-dense-20260529-053504](./variant-modular-dense-20260529-053504)
5. [outputs/variant-modular-sparse-20260529-053446](./variant-modular-sparse-20260529-053446)
6. [outputs/variant-modular-dense-matched-20260529-054630](./variant-modular-dense-matched-20260529-054630)
7. [outputs/variant-modular-sparse-matched-20260529-054655](./variant-modular-sparse-matched-20260529-054655)
8. [outputs/variant-modular-uneven-matched-20260529-054701](./variant-modular-uneven-matched-20260529-054701)
9. [outputs/baseline-tiny-20260529-054742](./baseline-tiny-20260529-054742)
10. [outputs/variant-modular-uneven-sparse-matched-20260529-054809](./variant-modular-uneven-sparse-matched-20260529-054809)