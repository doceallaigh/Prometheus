# Prometheus Research Plan

This document explains what Prometheus is trying to test, why the project believes the idea is worth testing, and what would count as success or failure.

It is written to be understandable on its own, but it is most useful when read after [README.md](README.md) and before [ROADMAP.md](ROADMAP.md).

## Objective

Prometheus is a low-budget research program aimed at testing whether a hierarchical, fractal-style neural architecture can deliver better capability per parameter than a conventional dense transformer at roughly the same model size and training cost.

In plain language, the project is asking whether a model with nested modules and limited internal communication can solve useful tasks more efficiently than a more standard densely connected model.

The near-term goal is not to build a frontier model immediately. The near-term goal is to identify one architectural advantage that is:

1. Measurable
2. Reproducible
3. Cheap enough to validate on small models
4. Plausibly scalable if it works

The long-term goal is more ambitious: if the architecture demonstrates persistent wins under tight compute constraints, scale it into a model family that can outperform stronger baselines at its size class.

## Key Terms

This plan uses the following terms repeatedly:

1. Dense transformer baseline: a standard reference model used as the control for comparison
2. Hierarchical modular network: a model whose computation is organized into nested groups or modules
3. Sparse connectivity: a design where only a limited subset of possible internal communication paths are active
4. Routing: the way information travels through the model's internal modules or connections
5. Fan-in: the set of upstream inputs that feed into a downstream unit or module
6. Sufficient-set activation: the idea that only a smaller subset of possible upstream inputs may be needed to trigger the right downstream behavior
7. FLOPs: a standard rough measure of how much computation a model run or training run requires
8. Ablation: a comparison where one feature is removed so its causal contribution can be measured

## Working Thesis

The human brain should be treated as a source of architectural hypotheses rather than a literal implementation blueprint. The most useful hypotheses from that analogy are:

1. Intelligence may benefit from localized computation combined with sparse long-range communication.
2. Hierarchical or fractal organization may improve reuse, specialization, and efficiency.
3. Bounded communication cost may matter more than unrestricted global connectivity.
4. Reliable downstream activation may depend on a sufficient subset of upstream units rather than all-to-all fan-in.
5. Biological branching may be uneven and superlinear at some scales rather than following a fixed branching factor.
6. Intelligence may depend partly on how quickly useful internal states can be reached, not just on total parameter count or raw compute.

Prometheus will focus on a machine-learning interpretation of those ideas rather than a strict mapping to anatomical lobes.

## Core Hypothesis

A hierarchical modular network with mostly local computation, sparse long-range shortcuts, and structured upstream-to-downstream sparsity can improve neural efficiency by reducing redundant communication and unnecessary fan-in. If that is true, the architecture should outperform a dense baseline on selected tasks that require multi-step composition, routing, or long-context information flow, while matching or approaching baseline performance on general language modeling at similar parameter count and training FLOPs.

The practical meaning of this hypothesis is simple: if the architecture is working as intended, it should either do better work at the same cost or similar work at lower cost.

## Non-Goals for the First Pass

The first pass will not attempt to:

1. Reproduce the biological brain faithfully
2. Implement dynamic rewiring during training
3. Build hard-coded anatomy-labeled subsystems such as frontal or temporal lobes
4. Solve multimodality on day one
5. Compete directly with frontier labs on general-purpose capability
6. Optimize for ethical reasoning as a separate architectural module

Those ideas may still matter later, but they add too much ambiguity and cost for the initial stage.

## Architectural Direction

### 1. Fractal Modular Hierarchy

The model should be organized as a recursive hierarchy of modules rather than a flat stack of identical blocks.

At a high level:

1. Small units perform local computation.
2. Small units are grouped into modules.
3. Modules are grouped into larger modules.
4. Communication is dense within a local group and sparse across distant groups.

This preserves the intuitive fractal structure while keeping the system implementable in modern deep learning frameworks.

The word fractal here should not be read as a strict mathematical claim. In this plan it means that similar organizational patterns may repeat across scales of the model.

### 2. Nonuniform Branching Hypothesis

The hierarchy should not assume a constant branching factor at every level. A more realistic and potentially more useful formulation is that branching changes with scale.

Examples of patterns worth testing:

1. Constant branching, such as 4, 4, 4
2. Expanding branching, such as 2, 4, 8
3. Superlinear branching, such as 2, 4, 16
4. Hybrid branching, where early levels are narrow and deeper levels fan out aggressively

The practical question is whether uneven branching improves the tradeoff between local specialization and global coverage. A narrow top of the hierarchy may constrain coordination cost, while wider lower levels may increase representational capacity where it is cheapest.

### 3. Parent-Child Coupling Rather Than Single Edges

Parent-child relationships in the hierarchy should not be modeled as one logical edge between aggregate modules. A better abstraction is partial population overlap or partial population coupling.

In practice, that means:

1. A parent module may connect to a subset of units inside each child module.
2. A child module may expose only a subset of units or channels back to the parent.
3. Connectivity should be expressed as a density or routing budget, not as a binary yes or no edge.

This better matches the intuition that communication between scales is distributed and redundant, not singular.

### 4. Sufficient-Set Activation Hypothesis

One promising idea is to prune upstream-to-downstream connectivity so that a downstream unit is driven by a minimal sufficient subset of upstream units rather than by every possible input.

Informally:

1. Let upstream units be A[1] through A[n].
2. Let downstream target be B.
3. Instead of keeping all A to B connections, search for the smallest subset S such that when the relevant units in S activate, B fires with high probability.

This suggests two useful design goals:

1. Remove redundant fan-in that does not materially improve downstream predictability.
2. Encourage sparse causal structure where important upstream combinations remain sufficient to trigger the correct downstream behavior.

This is an architectural hypothesis, not a claim that the exact minimal subset can always be identified cleanly in practice.

For the first pass, this should not be enforced with a hard statistical guarantee. It should be tested through approximations such as:

1. Learned sparse masks over fan-in
2. Top-k routing into downstream modules
3. L0 or L1-style penalties on incoming edges
4. Post-training pruning based on activation statistics and ablation sensitivity

The research question is whether minimal sufficient fan-in improves parameter efficiency, interpretability, or robustness without damaging performance.

This is also a latency hypothesis. If useful downstream states can be reached through smaller sufficient sets, the model may require fewer effective coordination steps to produce the right internal computation.

### 5. Static Sparse Connectivity

Instead of enforcing a literal six-hop maximum, the first implementation should use a fixed communication topology with these properties:

1. High local clustering
2. A limited number of cross-cluster shortcut connections
3. Low average communication distance between modules
4. Predictable computational cost

The useful concept is not exactly six degrees of separation. The useful concept is efficient communication in a sparse, clustered network.

One motivation for this is cognitive speed. A slower thought may reflect wasted routing, redundant activation, or interference from competing internal pathways. Even if that analogy is imperfect, it suggests a practical machine-learning question: can we reduce the number and cost of internal communication steps needed to reach a useful representation?

Candidate topologies for comparison:

1. Dense baseline communication
2. Tree-only hierarchy
3. Tree plus shortcuts
4. Small-world sparse graph over modules
5. Superlinear hierarchy plus shortcut edges

### 6. Shared Pretraining Objective

The first pass should train all modules on a shared autoregressive language-modeling objective. This avoids representational fragmentation caused by pretraining separate subsystems on unrelated data silos.

Specialization, if it emerges, should come from structure, routing, or auxiliary constraints, not from manually assigning one module to ethics and another to memory before the architecture has proved itself.

### 7. Optional Gating

If needed, add lightweight learned gating over module communication after the static topology baseline is working. Gating is a second-step experiment, not a prerequisite for version one.

## Why This Direction Is Worth Testing

This plan is worth funding only if it can answer a practical question:

Can structured sparsity, hierarchical routing, and minimal sufficient connectivity improve capability per parameter under constrained compute while reducing the internal cost of reaching useful states?

That question is materially different from asking whether the brain is a good metaphor. A metaphor can inspire the design, but only measured efficiency gains can justify continued investment.

In other words, the project is not trying to prove that brains and language models are the same thing. It is trying to test whether a few brain-inspired structural ideas produce measurable engineering value.

## First-Pass Experimental Program

### Experiment Goal

Compare a small dense transformer baseline against modular variants at equal or near-equal parameter count and roughly comparable training compute.

### Model Set

Build and compare these models:

1. Baseline dense transformer
2. Modular transformer with hierarchical partitions but standard dense inter-layer communication
3. Modular transformer with hierarchical partitions and static sparse structured communication
4. Modular transformer with hierarchical partitions, sparse structured communication, and sufficient-set fan-in pruning

Optional follow-on variants:

1. Uneven or superlinear branching schedule
2. Sparse structured communication plus learned gating
3. Post-training pruning using activation statistics

### Controlled Variables

As much as possible, keep the following fixed across experiments:

1. Training tokens
2. Parameter count
3. Optimizer family
4. Context length
5. Hardware budget
6. Approximate total FLOPs

The reason for controlling these variables is to make results interpretable. If too many things change at once, it becomes hard to know whether the architecture helped or whether the comparison was simply unfair.

### Evaluation Targets

The first pass should evaluate on a narrow but meaningful suite:

1. Language-modeling loss or perplexity
2. Long-context retrieval or needle-in-a-haystack style tasks
3. Multi-step reasoning or algorithmic composition tasks
4. Training stability
5. Inference latency and throughput
6. Edge sparsity achieved at equal quality
7. Sensitivity of downstream modules to removal of fan-in edges
8. Performance as a function of allowed routing or communication budget

The initial success condition is not universal dominance. The initial success condition is a credible, repeated advantage in at least one target capability without unacceptable regression elsewhere.

This matters because an early-stage research project should first look for one credible signal, not for proof of universal superiority.

## Success Criteria

Prometheus earns a second round of funding only if at least one of the following is true:

1. The sparse modular model clearly beats the dense baseline on long-context or routing-heavy tasks at similar cost.
2. The sparse modular model matches the dense baseline on general language modeling while using materially less compute or memory.
3. The sufficient-set pruning variant preserves quality while removing a meaningful fraction of incoming edges or routing paths.
4. The modular architecture shows a scaling trend that looks stronger than the baseline as model size increases modestly.
5. The modular variants degrade more gracefully than the dense baseline when routing, edge, or compute budgets are tightened.

If none of those happen, the architecture has not yet earned more complexity.

That is an intentional standard. A more complex model should be required to justify its own existence.

## Budget-Gated Roadmap

### Phase 0: Design and Benchmark Definition

Goal: define an experiment that can fail clearly.

Deliverables:

1. Exact model families to compare
2. Exact parameter budgets
3. Exact datasets
4. Exact benchmark suite
5. Exact go or no-go thresholds
6. Exact branching schedules to test
7. Exact sparsity targets for parent-child coupling and fan-in pruning

Budget guidance: minimal

### Phase 1: $5k Feasibility Pass

Goal: determine whether the architecture produces any measurable win on small models.

Deliverables:

1. Baseline dense implementation
2. Static modular implementation
3. At least one uneven or superlinear branching variant
4. One sparse fan-in or sufficient-set pruning mechanism
5. Reproducible training scripts
6. Evaluation report with ablations

Decision rule:

1. Continue only if at least one narrow benchmark shows a meaningful gain with tolerable regression elsewhere.

### Phase 2: Focused Iteration

Goal: improve only the components that appear causal for the gain.

Possible next steps:

1. Better sparse topology
2. Better branching schedule
3. Lightweight learned routing
4. Improved memory handling
5. Longer contexts
6. Better pruning criteria for minimal sufficient fan-in
7. Better training efficiency

### Phase 3: Scaling Test

Goal: determine whether the observed benefit survives moderate scaling.

This stage should begin only after a real positive signal appears in Phase 1 or Phase 2.

## Risks

### 1. No Real Advantage Over Standard Transformers

The architecture may simply add coordination overhead without improving useful computation.

Mitigation:

1. Keep the first comparison tight and controlled.
2. Kill weak variants quickly.

### 2. Architectural Complexity Masks Results

Too many moving parts will make outcomes uninterpretable.

Mitigation:

1. Use static topology first.
2. Add only one major novelty at a time.
3. Treat uneven branching and sufficient-set pruning as separable ablations.

### 3. Cheap Experiments May Not Predict Larger-Scale Behavior

A small-model win may disappear at scale, or a small-model loss may hide a later scaling benefit.

Mitigation:

1. Track trends across at least a small size ladder.
2. Avoid over-interpreting a single run.

### 4. Brain Analogies Become a Distraction

The biological inspiration may encourage elegant stories without measurable value.

Mitigation:

1. Translate every biological idea into a concrete machine-learning claim.
2. Drop any concept that does not produce a useful experimental prediction.

### 5. Sufficient-Set Pruning Breaks Robustness

Pruning to a minimal fan-in set may create brittle dependencies or unstable routing.

Mitigation:

1. Measure performance under activation noise or edge dropout.
2. Prefer near-minimal sufficient sets over exact minimality.
3. Keep redundant backup pathways where the cost is low.

## Open Research Questions

1. What is the best unit of modularity: attention heads, blocks, block groups, experts, or memory modules?
2. What topology gives the best tradeoff between local specialization and global coordination?
3. Does hierarchical structure help mainly with long context, reasoning depth, training efficiency, or something else?
4. Is learned routing necessary, or is fixed sparse topology already enough to matter?
5. Does the fractal hierarchy remain useful as model size increases?
6. What branching schedule gives the best efficiency-quality tradeoff?
7. How much parent-child connectivity is enough before additional edges become redundant?
8. Can minimal sufficient fan-in be learned online, or is post-training pruning the cleaner path?

## Immediate Implementation Plan

### Milestone 1

Implement a clean dense baseline and a modular baseline with no sparse graph constraint.

### Milestone 2

Implement a static sparse communication pattern over the modular variant.

### Milestone 3

Implement one uneven branching schedule and one fan-in pruning mechanism.

### Milestone 4

Run a controlled benchmark suite and compare:

1. Loss
2. Retrieval accuracy
3. Reasoning performance
4. Speed
5. Stability
6. Sparsity achieved

### Milestone 5

Write an internal report answering one question only:

Is there enough evidence that hierarchical sparse modularity, nonuniform branching, or sufficient-set connectivity improves capability per parameter to justify further investment?

## Practical Position

Prometheus should be treated as an architectural efficiency bet, not as an attempt to brute-force a frontier model race.

If the architecture works, the payoff is large because efficiency advantages compound. If it does not show early evidence under controlled conditions, the right move is to simplify or abandon it quickly.

That is the standard the project must meet.