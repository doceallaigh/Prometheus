# Prometheus

Prometheus is a research repository for exploring whether hierarchical sparse modularity can improve capability per parameter and capability per unit compute relative to dense transformer baselines.

The repository is not intended to jump directly to a frontier-scale model. Its first purpose is to support a disciplined low-budget research program that can test narrow architectural hypotheses, measure them cleanly, and decide whether further investment is justified.

The guiding research direction is defined in [PLAN.md](PLAN.md), and the staged execution path is defined in [ROADMAP.md](ROADMAP.md).

If you are new to this repository, read this file first, then [PLAN.md](PLAN.md), then [ROADMAP.md](ROADMAP.md).

## Plain-Language Summary

Prometheus is trying to answer a focused question:

Can a neural-network design with structured hierarchy and sparse internal communication do more useful work per unit of size or compute than a more conventional dense design?

In simpler terms, the project is testing whether a model can become more efficient by organizing computation into nested modules rather than a flat stack, limiting unnecessary internal connections, and preserving only the most useful pathways needed for a downstream computation.

The repository does not assume this idea is true. It exists to test the idea carefully.

```mermaid
flowchart LR
	A[Dense baseline] --> B[Hierarchical modules]
	B --> C[Sparse connectivity]
	C --> D[Sufficient-set pruning]
	D --> E[Measured comparison]
```

## Top-Level Purpose

The repository exists to do four things well: define a small number of architectural hypotheses clearly, implement controlled baseline and variant models fairly, evaluate those models on capability and efficiency metrics that match the hypotheses, and produce evidence strong enough to justify either further investment or abandonment.

This is an experimental research codebase. It should optimize for clarity, repeatability, and decision quality before it optimizes for breadth.

## Current Project Stance

Prometheus is currently focused on a first-pass budget-constrained experiment. In practice, that means small models before large models, static sparse topology before dynamic rewiring, shared training objectives before hand-labeled cognitive subsystems, a few controlled variants before many speculative ones, and reproducible evidence before narrative appeal.

## Repository Documents

The main documents are [PLAN.md](PLAN.md), which defines the research plan, hypotheses, and success criteria; [ROADMAP.md](ROADMAP.md), which defines the staged execution plan, user dependencies, and pre-flight setup; and [README.md](README.md), which defines repository purpose and implementation guidance.

If a code change conflicts with those documents, the conflict should be resolved explicitly rather than ignored.

## Core Terms

The documents in this repository use a small set of recurring terms.

1. Dense baseline: a conventional reference model with broadly available communication paths, used as the fairness control for comparisons
2. Variant: any experimental model that changes one or more architectural features relative to the baseline
3. Hierarchical modularity: organizing computation into nested groups or modules rather than treating every layer or block as part of one flat structure
4. Sparse connectivity: allowing only a limited subset of possible internal connections so that communication has lower cost
5. Routing: the path by which information moves through modules or connections inside the model
6. Sufficient-set pruning: removing incoming connections until only a near-minimal set remains that still supports the desired downstream behavior
7. Ablation: turning off or removing one feature at a time to measure whether that feature actually caused an observed result
8. Capability per parameter: how much useful performance is achieved for a given model size
9. Capability per unit compute: how much useful performance is achieved for a given amount of computation, time, or hardware effort
10. Agent-driven project: a project where implementation is performed partly by AI coding agents across multiple sessions, which makes explicit documentation and validation especially important

## Developer Guide

This guide applies to both human contributors and agent-driven iteration.

### 1. Optimize for Testable Claims

Every meaningful implementation step should support a claim that can be tested.

Good examples:

1. A sparse communication layer reduces routing cost at similar quality.
2. A hierarchical modular layout improves performance on long-context retrieval.
3. A sufficient-set pruning mechanism preserves quality while removing fan-in edges.

Bad examples:

1. The architecture feels more brain-like.
2. The design seems elegant.
3. A new component might help later.

If a change does not strengthen a measurable experiment, it should be questioned.

### 2. Keep Baselines Sacred

The dense baseline is not a placeholder. It is the control against which the architecture lives or dies.

Rules:

1. Match baselines and variants as fairly as possible.
2. Track parameter count, compute budget, context length, optimizer family, and training tokens.
3. Do not let variants quietly gain extra advantages that are unrelated to the architectural hypothesis.
4. Save exact configs used for every result worth citing.

Unfair comparisons are worse than no comparisons.

### 3. Add One Novelty at a Time

This repository should resist stacked novelty.

Preferred order:

1. Baseline
2. Hierarchical modular structure
3. Static sparse connectivity
4. Sufficient-set fan-in pruning
5. Uneven branching
6. Optional learned gating

If multiple new mechanisms land at once, the result becomes harder to interpret and harder to trust.

### 4. Design for Ablation

Every architectural feature should be removable by configuration.

Implementation guidance:

1. Put feature switches behind explicit config flags.
2. Keep interfaces stable across baseline and variants when practical.
3. Make it easy to run with and without a given mechanism.
4. Prefer composition over tangled conditional logic.

If a component cannot be ablated cleanly, it is probably too entangled.

### 5. Prefer Readability Over Cleverness

The first version of this repository should be readable by someone trying to verify an experimental claim, not by someone trying to admire a clever abstraction.

Preferred practices:

1. Small modules with clear names
2. Explicit data flow
3. Config-driven experiments
4. Short, focused functions
5. Minimal hidden state

Avoid:

1. Framework-heavy indirection without a clear payoff
2. Excessively abstract training orchestration
3. Dense metaprogramming or magic registration patterns
4. Reformatting unrelated code during experimental work

### 6. Build for Repeatability

A result that cannot be repeated should not drive major project decisions.

Minimum expectations:

1. Fixed seeds where practical
2. Versioned config files
3. Logged environment and dependency information
4. Stable dataset references
5. Saved evaluation outputs for important runs

Whenever possible, a future contributor should be able to answer: what exactly produced this result?

### 7. Treat Efficiency as a First-Class Output

Prometheus is not only about quality metrics. It is also about neural efficiency and communication efficiency.

Track things such as:

1. Throughput
2. Memory use
3. Active edges or routing paths
4. Quality under constrained routing budgets
5. Performance under ablation or edge dropout

A variant that is slightly worse on raw loss but much stronger on efficiency may still matter. A variant that is more complex and not more efficient usually does not.

### 8. Keep the Data Story Simple Early

The first pass should rely on public or easy-to-access data whenever possible.

Guidance:

1. Prefer public benchmarks and public corpora.
2. Avoid custom data pipelines unless they are necessary for the core hypothesis.
3. Do not create a data engineering project by accident.
4. Add data complexity only after the architecture itself shows signal.

This repository should not stall on bespoke dataset generation during the first pass.

### 9. Sustainability in an Agent-Driven Project

Agent-driven work can move quickly, but it can also generate brittle complexity if not constrained.

Sustainable practices:

1. Keep intent in repository files, not only in chat history.
2. Prefer explicit plans and checkpoints over open-ended iteration.
3. Make small, reversible changes when exploring uncertain ground.
4. Validate immediately after meaningful edits.
5. Keep logs, configs, and outputs organized enough that future sessions can resume cleanly.
6. Avoid broad speculative rewrites when a narrow local change can answer the question.
7. Record assumptions that would otherwise be lost between sessions.

Unsustainable practices:

1. Letting major decisions live only in conversation context
2. Mixing unrelated refactors with experimental changes
3. Growing many half-implemented variants at once
4. Repeatedly changing the experimental target mid-implementation
5. Allowing infrastructure choices to remain implicit

The repository should remain understandable even if implementation is spread across many agent sessions.

### 10. How Agents Should Iterate Here

When iterating on this repository, prefer the following workflow:

1. Read the relevant plan or roadmap section first.
2. Identify the smallest concrete artifact to produce next.
3. Implement the minimum change needed to support that artifact.
4. Run the narrowest useful validation immediately.
5. Summarize what changed, what was validated, and what remains uncertain.

In practice, that means:

1. Do not build multiple experimental mechanisms before the baseline works.
2. Do not widen scope between an edit and its first validation.
3. Do not treat git diff as sufficient validation when an executable check exists.
4. Do not rely on memory alone for decisions that belong in files.

### 11. Decision Discipline

This repository should make it easy to stop weak directions.

Every major phase should end with a decision artifact:

1. What was tested
2. What changed relative to baseline
3. What the result means
4. Whether the result justifies further work

A clean negative result is a success if it prevents larger wasted investment.

## Practical Expectations for the First Pass

For the $5k plan, success does not mean proving universal architectural superiority. Success means producing a credible early signal.

Examples of acceptable first-pass outcomes:

1. A sparse modular variant wins on a targeted routing-heavy benchmark.
2. A sufficient-set pruning variant preserves quality while reducing effective connectivity.
3. A hierarchical layout degrades more gracefully under tighter communication budgets.

Examples of outcomes that should not be oversold:

1. A single lucky run
2. An unreplicated benchmark win
3. A more complex model with no clear efficiency gain
4. A better story without better measurements

## Operational Guidance

Keep credentials, account secrets, service-account keys, and banking details out of the repository.

Keep large generated artifacts out of version control unless there is a deliberate reason to store them.

Prefer simple directory structure and explicit naming as the repository grows. Likely top-level folders will eventually include model code, experiment configs, scripts, evaluation outputs, and notes. Add structure only when there is enough implementation to justify it.

## Contribution Standard

Before considering a change complete, confirm:

1. The change supports a specific experimental goal.
2. The baseline comparison remains fair.
3. The feature can be enabled or disabled cleanly if needed.
4. The result is documented well enough to survive a future session.
5. The change did not add avoidable complexity.

That standard is more important here than speed alone.