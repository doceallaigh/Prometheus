# Prometheus Roadmap

## Pre-Flight User Setup

This section exists to remove execution friction before any meaningful engineering work begins. For the $5k plan, the fastest path is to frontload account setup, spending controls, access, and public-data readiness.

### Goal

Make it possible to start implementation and pilot runs immediately without blocking on account approvals, missing permissions, or unclear spending authority.

### Actionable Steps

1. Confirm the hard spending cap for the first pass.
2. Confirm which cloud provider will be used first. Google Cloud is the default if already available.
3. Confirm that billing is active on the cloud account.
4. Create or identify a project dedicated to Prometheus experiments.
5. Enable access to GPU-capable compute in that project.
6. Set budget alerts and hard internal spending checkpoints.
7. Confirm whether storage buckets, artifact storage, and logs can be created in that project.
8. Decide how credentials will be provided to the execution environment.
9. Prepare any non-secret identifiers needed for setup, such as project IDs, region preferences, and storage names.
10. Confirm whether public datasets may be downloaded directly from public sources.
11. Confirm whether lightweight paid public datasets are allowed if needed later.
12. Confirm whether training outputs and checkpoints may be stored in cloud storage.

### Recommended Output

Produce a ready state with:

1. One approved cloud project
2. Billing enabled
3. GPU quota or approved compute path
4. Storage path for artifacts and checkpoints
5. Budget alerts configured
6. A safe credential-delivery plan
7. A short list of approved public data sources

### Copilot Usage Note

For the $5k plan, model-usage budgeting should be treated as an operational constraint, but not the primary one. The practical goal is to avoid wasting context and iteration budget on unnecessary churn.

Recommended practice:

1. Keep project requirements in repository files instead of repeatedly pasting them into chat.
2. Work in checkpoints such as planning, scaffolding, baseline, variants, and evaluation.
3. Avoid dumping large logs unless they are needed for debugging.
4. Start a fresh session after major milestones if context has become noisy.
5. Expect total Copilot usage for the first implementation push to be materially higher than planning-only usage.

Rough operating bands:

1. Planning and roadmap refinement: low
2. Initial scaffolding and implementation: moderate
3. Full day of prototype execution with debugging and evaluation: high

The exact token count is less important than reducing avoidable rework, repeated file reads, and oversized command output.

### User-Dependent Progress

Progress depends on the user for:

1. Approving the first-pass budget ceiling
2. Providing or authorizing access to the cloud account
3. Providing non-secret environment details such as project IDs and preferred regions
4. Handling any secret credentials directly and securely rather than storing them in the repository
5. Approving whether public datasets can be downloaded automatically
6. Approving whether cloud costs may be incurred immediately for pilot runs

### Security Note

Credentials, API keys, service-account keys, and banking details should not be written into the repository or pasted into project files. They should be handled through secure environment configuration or direct login flows.

### Suggested Decision Gate

Do not begin implementation until cloud access, billing, and credential handling are confirmed.

## High-Level Step Tree

### 0. Remove execution friction
#### 0.1 Confirm budget and spending authority
#### 0.2 Confirm cloud project, billing, and compute access
#### 0.3 Confirm credential handling and storage paths
#### 0.4 Confirm approved public data sources

### 1. Define the first-pass experiment
#### 1.1 Choose the exact problem scope
#### 1.2 Choose the baseline and comparison variants
#### 1.3 Choose the budget, hardware, and success gates

### 2. Build the experimental foundation
#### 2.1 Set up the codebase and training pipeline
#### 2.2 Implement the dense baseline
#### 2.3 Implement evaluation and reporting

### 3. Add the first architectural hypotheses
#### 3.1 Add static sparse connectivity
#### 3.2 Add hierarchical modular structure
#### 3.3 Add sufficient-set fan-in pruning

### 4. Run the first controlled benchmark pass
#### 4.1 Train the baseline and variants
#### 4.2 Compare quality, efficiency, and routing behavior
#### 4.3 Decide whether the signal is real

### 5. Narrow to the strongest idea
#### 5.1 Kill weak variants
#### 5.2 Refine the strongest topology or pruning mechanism
#### 5.3 Test limited scaling behavior

### 6. Make the go or no-go decision
#### 6.1 Decide whether to invest more money
#### 6.2 Decide whether to simplify, pivot, or stop

## Guiding Principle

The roadmap is designed to answer one question as cheaply and clearly as possible:

Can hierarchical sparse modularity improve capability per parameter or capability per unit compute enough to justify further investment?

This is not a roadmap for building a frontier model immediately. It is a roadmap for producing a disciplined early signal.

For the $5k plan, the first source of failure is usually not model design. It is operational friction. This roadmap therefore starts by eliminating user-side blockers before engineering work begins.

## 1. Define the First-Pass Experiment

### Goal

Turn the current research thesis into a narrow experiment that can fail clearly.

### Actionable Steps

1. Write down the exact first-pass claim to test.
2. Choose the primary evaluation target.
3. Choose one dense baseline architecture.
4. Choose two or three architectural variants maximum.
5. Choose a parameter budget for the first pass.
6. Choose the training-token budget.
7. Define hard success and failure criteria.
8. Decide what evidence is strong enough to justify a second round.

### Recommended Output

Produce a one-page experiment brief with:

1. Baseline model
2. Variant list
3. Datasets
4. Metrics
5. Hardware assumptions
6. Budget cap
7. Go or no-go thresholds

### User-Dependent Progress

Progress depends on the user for:

1. Approving the size of the initial financial risk
2. Approving the definition of success for the first pass
3. Deciding whether the goal is primarily scientific validation, investor-grade evidence, or a path to a usable prototype

### Suggested Decision Gate

Do not move forward until the first-pass claim is short enough to fit in two sentences.

## 2. Build the Experimental Foundation

### Goal

Create a minimal, reproducible training and evaluation environment that makes later comparisons trustworthy.

### Actionable Steps

1. Choose the training framework.
2. Set up repository structure for models, configs, training, and evaluation.
3. Add experiment configuration files.
4. Add logging for loss, throughput, memory use, and evaluation results.
5. Add reproducibility controls such as seeds and fixed config snapshots.
6. Add an evaluation harness that can run baseline and variants consistently.
7. Add a reporting format for side-by-side comparison.

### Recommended Output

Produce a runnable baseline experiment pipeline that can be repeated without manual intervention.

### User-Dependent Progress

Progress depends on the user for:

1. Approving the software stack if there are strong preferences
2. Approving whether cloud compute, rented GPUs, or owned hardware will be used
3. Approving any spending on infrastructure, storage, or hosted training resources

### Suggested Decision Gate

Do not implement novel architecture until the baseline training and evaluation loop works end to end.

## 3. Implement the Dense Baseline

### Goal

Create the reference model that every later claim must beat or match.

### Actionable Steps

1. Implement a clean dense transformer baseline.
2. Match the baseline to the chosen parameter budget.
3. Train a short pilot run to confirm stability.
4. Verify evaluation metrics on held-out tasks.
5. Measure training speed, inference speed, and memory use.
6. Save baseline configs and checkpoints needed for comparison.

### Recommended Output

Produce a stable baseline run with reliable metrics and known cost.

### User-Dependent Progress

Progress depends on the user for:

1. Approving the fairness criteria for baseline comparisons
2. Deciding whether a standard published baseline is sufficient or whether a custom baseline is preferred

### Suggested Decision Gate

If the baseline itself is unstable or too expensive, stop and reduce scope before implementing variants.

## 4. Implement the First Architectural Variants

### Goal

Test the ideas most central to Prometheus without introducing too many moving parts at once.

### Variant Tree

1. Variant A: Hierarchical modular structure without sparse communication
2. Variant B: Hierarchical modular structure with static sparse connectivity
3. Variant C: Variant B plus sufficient-set fan-in pruning
4. Optional Variant D: Uneven or superlinear branching schedule

### Actionable Steps

1. Define the unit of modularity.
2. Implement hierarchical grouping of those units.
3. Add a fixed sparse communication topology.
4. Add metrics for active edges, routing cost, and communication budget.
5. Implement one pruning or sparse-fan-in mechanism.
6. Add ablation toggles so each idea can be disabled independently.
7. Keep parameter count and training conditions as close to baseline as possible.

### Recommended Output

Produce a small set of controlled variants whose differences are easy to explain.

### User-Dependent Progress

Progress depends on the user for:

1. Choosing whether uneven branching belongs in the first pass or in a later pass
2. Approving how much architectural complexity is acceptable in version one
3. Deciding how much interpretability matters versus raw benchmark performance

### Suggested Decision Gate

If a variant cannot be explained simply, it is too complex for the first pass.

## 5. Define the Benchmark Suite

### Goal

Measure whether the variants actually improve the properties the project cares about.

### Actionable Steps

1. Choose a language-modeling benchmark.
2. Choose a long-context retrieval benchmark.
3. Choose a routing-heavy or multi-step reasoning benchmark.
4. Define efficiency metrics such as throughput, memory use, and quality under reduced routing budgets.
5. Define robustness checks such as edge dropout or constrained communication.
6. Define how results will be averaged and compared.

### Recommended Output

Produce a benchmark suite that measures both capability and neural efficiency.

### User-Dependent Progress

Progress depends on the user for:

1. Deciding which capabilities matter most for the first signal
2. Deciding whether long-context, reasoning, or efficiency is the top priority
3. Approving the amount of benchmark diversity versus cost discipline

### Suggested Decision Gate

If the benchmark suite does not directly test the core hypothesis, cut it.

## 6. Run the First Controlled Benchmark Pass

### Goal

Generate the first real evidence for or against the architecture.

### Actionable Steps

1. Train the dense baseline under the agreed budget.
2. Train each approved variant under the same or closely matched conditions.
3. Record cost, time, memory, and training stability.
4. Run the full benchmark suite.
5. Compare quality against compute and communication cost.
6. Inspect whether sparse variants degrade more gracefully under tighter budgets.
7. Summarize the results in a short internal report.

### Recommended Output

Produce a comparison table that clearly shows where each variant wins, loses, or is inconclusive.

### User-Dependent Progress

Progress depends on the user for:

1. Approving the final spend for the benchmark run
2. Deciding whether inconclusive results justify another pass
3. Deciding whether a narrow win is enough to continue

### Suggested Decision Gate

Do not rely on intuition after this point. Continue only if the numbers justify it.

## 7. Narrow to the Strongest Idea

### Goal

Prevent the project from drifting into a bundle of loosely related ideas.

### Actionable Steps

1. Rank variants by signal strength.
2. Kill variants that add complexity without clear gain.
3. Identify whether the best signal comes from topology, pruning, or branching.
4. Run one or two focused follow-up ablations on the strongest component.
5. Check whether the result survives a modest increase in model size or training budget.

### Recommended Output

Produce a second-round recommendation focused on one main mechanism.

### User-Dependent Progress

Progress depends on the user for:

1. Approving the discipline to abandon favored ideas if they do not work
2. Choosing whether to optimize for publishable evidence, product relevance, or long-term architectural promise

### Suggested Decision Gate

If the project cannot name the strongest mechanism, it is not ready to scale.

## 8. Make the Go or No-Go Decision

### Goal

Decide whether Prometheus has earned more money, more time, and more complexity.

### Actionable Steps

1. Compare the observed gains against the predefined thresholds.
2. Estimate whether the strongest result is likely to scale.
3. Estimate the cost of the next round.
4. Decide whether to continue, simplify, pivot, or stop.
5. Capture the reasoning in a short written decision memo.

### Recommended Output

Produce a clear decision with one of four outcomes:

1. Continue with focused iteration
2. Continue with a larger budget
3. Simplify and retest
4. Stop the current direction

### User-Dependent Progress

Progress depends on the user for:

1. Approving further capital allocation
2. Deciding what level of evidence is required before scaling investment
3. Deciding whether strategic patience or rapid iteration is the better posture

### Suggested Decision Gate

No second-round spending should happen without a written decision memo tied to measured results.

## Cross-Cutting User Dependencies

Some decisions affect every phase:

1. Total budget ceiling for the first pass
2. Preferred speed versus rigor tradeoff
3. Tolerance for inconclusive outcomes
4. Willingness to abandon biologically appealing ideas that do not produce measurable value
5. Whether the primary success mode is scientific evidence, architectural insight, or a path toward a usable model
6. Whether cloud credentials and account access can be provided quickly and safely
7. Whether public data can be used directly without legal or policy review
8. Whether Copilot usage should be optimized for speed, cost discipline, or maximal persistence within a session

## Immediate Next Actions

1. Confirm the cloud project, billing status, and GPU access path.
2. Confirm how credentials and non-secret environment details will be provided securely.
3. Confirm the $5k cap and the first internal spending checkpoint.
4. Confirm whether approved public datasets can be downloaded directly.
5. Lock the first-pass claim in two sentences.
6. Decide the exact parameter and token budget.
7. Decide the baseline and maximum number of variants.
8. Decide the benchmark suite.
9. Decide the hard go or no-go threshold for spending beyond the first pass.