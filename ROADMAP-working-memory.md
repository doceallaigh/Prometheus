# Roadmap: Recurrent Working-Memory Model (Latent CoT Corrector)

Date opened: 2026-07-15
Scope: future action items for the frozen-trunk + CfC-corrector line of work
("recurrent working memory"). Companion to
`papers/latex/latent-cot-combined.tex` (results and framing). Items are grouped by
horizon; each carries a motivation and a concrete first step so it can be
picked up cold.

## Near term (queued, mechanical)

1. **Finish the trunk ladder.** 3B eval + visible-SC@8 comparator in flight;
   7B (quantized or GCP) is the next rung. Each rung so far has improved the
   corrector's relative position (0.5B parity → 1.5B beats all visible
   baselines); find where the trend saturates.
2. **Full 7.4k GSM8K harvest + retrain at each scale.** Current correctors
   saw 290-1,605 traces from 2k problems. Lenient harvest scoring would
   roughly triple the keep rate at 0.5B.
3. **Task-transfer probe.** Corrector trained on one task family, evaluated
   zero-shot on another over the same frozen trunk. Tests whether the learned
   repair function is general or task-specific — the biggest open validity
   question for scaling claims.
4. **Interpretability battery (`interpret` CLI).** Probe sweep (layer × step),
   activation patching / IIA on the J-space tap, CfC-state decoding (running
   value vs error-flag vs correction). Paper §7 commits to this; item 7 below
   depends on its error-flag probe.
4b. **Batched latent rollouts.** (2026-07-15) Because the scratch space is Because the scratch space is
   latent, the chain has no streaming/ordering contract — only the answer
   span has a delivery obligation — so rollouts (and problems) parallelize
   freely up to compute. Concretely: `_generate_with_corrector` is batch-1;
   the corrector is batch-friendly, so running latent-SC@k's k rollouts as
   one batch is a near-k× wall-clock win on the dominant eval cost, and
   composes with stop-on-agreement (draw 2 in parallel, spawn more only on
   disagreement). Qualification for the paper: server-side batching also
   exists for visible CoT; the novel part is the absent streaming contract
   and within-problem parallelism under adaptive stopping.
     *Extension — early-consensus truncation (2026-07-15): exact cross-rollout
     tensor reuse requires identical token prefixes (KV entries are position-
     and context-specific; v2's failure showed cosine-close states decode to
     different digits, so approximate KV substitution compounds errors). The
     practical form of cross-rollout similarity is consensus detection in
     state space: if two rollouts' s_t trajectories converge mid-rollout,
     count them as agreeing and truncate one — moving stop-on-agreement from
     answer space to state space and cutting the tail of every agreeing pair;
     dually, kill and resample duplicates for diversity. First step (no new
     inference): from SC@8 dumps + probe infra, measure how early pairwise
     s_t distance at step t predicts same-vs-different final answer.*

5. **True FLOP reduction, not just emitted-token reduction.** Today the
   internal rollout costs ~1 CoT worth of trunk FLOPs (+9%). Two levers,
   separable:
   - **5a. Divergence-gated intervention.** Train a lightweight detector (SAE
     or linear probe on J-space / CfC state) that predicts *imminent
     divergence* (hallucination / derailment) and gates the system: run the
     cheap path by default, engage correction — or extra compute — only when
     divergence is likely. Note an important accounting fact: the corrector
     itself is ~0.2% of trunk FLOPs, so gating the *corrector* saves nothing;
     the win is gating the *rollout* (early exit when the answer is already
     determined, extended rollout only when the detector fires). First step:
     from existing completion dumps, label divergence onsets (first token
     where a wrong rollout departs from a correct one) and test whether a
     linear probe on h_tap / s_t predicts them.
     *Mechanistic hypothesis (2026-07-15): structural commitment precedes
     content availability — models commit to a schema (a citation follows, an
     equation goes here) before the content filling it is resolved, then
     confabulate rather than back out. Predicts (i) divergence onsets cluster
     at plan-commitment points, not at content errors (consistent with the
     measured early onsets, median ~4-5% of rollout), and (ii) a detectable
     signature at slot-opening positions: high structural confidence with low
     content confidence. The probe should test for this gap explicitly.*
   - **5b. Multi-jump latent steps via s_t.** The CfC state summarizes the
     rollout so far; use it as a *multi-jump vector* — one corrector step
     standing in for k trunk decoding steps — to compress reasoning
     iterations and cut rollout length itself. First step: train the
     corrector to predict the trunk's hidden state k steps ahead
     (teacher-forced, k∈{2,4,8}) and measure answer accuracy when the trunk
     consumes the jump instead of decoding through.
     *Deployment note (2026-07-15): token-space latent rollout is billing-
     compatible — internal tokens are discrete and meterable, matching the
     industry's hidden-reasoning-token billing precedent. Multi-jump (and
     item 6's continuous vectors) replaces countable tokens with continuous
     state transitions, forcing FLOP- or time-based billing — a real
     adoption-friction argument against fully continuous designs to weigh
     when choosing how far to push compression.*
   - **5c. Latent rollback: detect-and-rewind instead of predict.** (2026-07-15)
     Because the rollout is never emitted, hallucinations need only be
     *recognized* before emission, not predicted in advance — a strictly
     easier detection problem with post-hoc evidence available. Rollback
     machinery is nearly free: a checkpoint is a KV-cache truncation point
     plus a CfC state snapshot (d_cfc floats) every k internal tokens; on
     resume, perturb (temperature resample or ban the offending continuation)
     to avoid deterministic replay, under a bounded re-roll budget. Compute
     beats latent SC@k: one rollout plus partial re-rolls proportional to the
     error rate with shared prefixes (internal tree search, detector-gated),
     versus k unconditional full rollouts. Composes with 5a — same detector
     infrastructure with a relaxed requirement (post-hoc AUC should beat
     predictive AUC). Caveat: repairs derailments, not incapacity; the
     dose-response law bounds what re-rolling can recover. First step: from
     the 5a labels, measure detector AUC as a function of tokens-past-fork to
     quantify the prediction-vs-recognition gap and pick the rollback trigger
     point.
     *Measured (2026-07-15, 1.5B, 371 pairs): anchor matters entirely. At the
     sampling fork the probe is null (max AUC 0.55) — wrong rollouts are not
     yet wrong there. Re-anchored at the arithmetic error site (first
     computed value unsanctioned by gold `<<>>` annotations; median +458
     chars past the fork), a linear probe on h_tap reads the error at
     AUC 0.985 one token after it is written, 0.97 at +4, decaying to chance
     by +32. Two design consequences: (i) the rollback trigger must be a
     per-token scanner over a short trailing window, not a one-shot
     classifier — the error signature is sharp but fades from the residual
     stream within ~30 tokens; (ii) h_tap ≥ s_t at every offset here, so the
     trigger can read the trunk directly and needs no corrector state. Next:
     wire the h_tap probe as the rollback trigger and measure the oracle
     re-roll ceiling.*
6. **Latent reasoning vectors as training signal.** Train the sidecar (and
   optionally the trunk, forfeiting the frozen-trunk guarantee — keep as a
   separate arm) on continuous reasoning vectors rather than token traces.
   Data is the hard part; the practical production recipe is a Coconut-style
   curriculum: start from harvested token traces (we have these at three
   scales) and progressively replace token spans with their hidden-state
   summaries, so the model manufactures its own latent targets. Composes with
   5b — the jump vector *is* a latent reasoning vector.
7. **Mech-interp in the loop at inference: hallucination sense / "sense of
   self".** Give the system a live read on its own reliability: probes from
   item 4 (error-flag direction in J-space or CfC state) surfaced as a signal
   the corrector — or the emission gate — conditions on. This is the
   principled version of 5a's detector and shares its infrastructure. The
   more speculative "sense of self" framing (a persistent self-model in the
   liquid state) should be treated as an interpretation to *test* (does s_t
   encode calibrated confidence?) rather than a design goal, until the probe
   data says otherwise.

## Far term (research-grade, speculative)

8. **DeepSeek visual primitives integration — optical context compression.**
   Direction resolved 2026-07-15: the OCR-line "context as compressed image
   tokens" primitive, not VL-style grounding. The working memory's internal
   rollout is stored/consumed as a dense optical scratchpad encoding rather
   than token-space text. Natural experiment: can the internal rollout be
   compressed into optical tokens at materially lower FLOP cost than
   token-space rollout, with the corrector reading the compressed form
   (connects to 5b)?
   *Status (2026-07-17, iteration 1 done): `optical-resume` harness built
   and run (Qwen2.5-VL-3B, 100 GSM8K problems). Plumbing works — 4.87×
   token compression (27 image tokens vs 131 text) with no accuracy cliff
   (0.740 vs 0.770 text resume). Confound: no_prefix scores 0.760; GSM8K
   is self-contained so the model re-solves rather than reads. Iteration 2:
   render non-recoverable content (the problem statement itself, or
   injected intermediate values) so the no-prefix floor collapses and the
   sweep measures optical reading, then wire the corrector to consume the
   compressed form.*
9. **Mech-interp-guided training: a mathematical formulation of optimal
   shape.** Gradient descent already optimizes the model's internal shape —
   implicitly, against the task loss, with representational geometry as an
   unconstrained byproduct. The research goal is to make that optimization
   *explicit*: derive a mathematical formulation of what an optimal internal
   shape looks like (consolidation vs redundancy, minimized vs maximized
   polysemanticity, alignment to an interpretable basis via e.g. orthogonal
   Procrustes), validated against performance metrics rather than assumed.
   Empirical on-ramp: measure whether corrector performance correlates with
   state-space polysemy across existing checkpoints (three scales already
   trained — the dimensions-per-feature statistic from the toy-superposition
   design doc applies directly), and use any correlation to constrain the
   candidate formulations before adding a training-time term.
10. **Representational affective state — panic/desperation as a J-space
   vector or circuit.** (2026-07-17) Capture an *emotional-state
   representation* (panic, desperation, and by extension calm/confidence)
   either as a direction in J-space or as a distributed circuit, and test it
   causally by injecting the representation into an otherwise neutral model
   state and observing behavioral change.
   *Notes (agent): the injection harness already exists — `steer-inject` is
   exactly this experiment with a different vector; the margin-gate and
   dose-response protocol (α sweep, α=0 determinism control, flips
   accounting) carry over unchanged. Supervision is the design question,
   not machinery: elicit contrastive rollouts from matched prompts (same
   problem framed neutrally vs under duress — "you will be terminated if
   wrong", imminent-deadline framings, sunk-cost pressure) and take
   v = mean(h_tap[duress] − h_tap[neutral]) exactly as the repair vector
   was built. Programmatic success criteria, strongest first: (i)
   behavioral dose-response under injection into neutral contexts —
   measurable endpoints: accuracy, rollout length, hedging-language
   frequency, repetition/perseveration rate, retry patterns, answer-commit
   latency (tokens to `####`), all already instrumented or trivially
   countable from dumps; (ii) a held-out linear probe trained on elicited
   duress-vs-neutral rollouts fires on *injected* rollouts (the vector
   induces the state, not just its surface correlates); (iii) tonic-vs-
   phasic signature: unlike our phasic error flags (sharp, fades in ~30
   tokens), an affective state should be *tonic* — persistent in h_tap and
   CfC state s_t across the rollout — measurable with the same trigger-lab
   window statistics. The circuit version (beyond a single direction):
   layer-sweep the probe to find where the state is readable, patch
   at multiple taps simultaneously, and test whether a direction found at
   one layer transfers to adjacent layers (single-direction hypothesis) or
   requires per-layer components (distributed circuit). Caution for the
   writeup: behavioral shift under injection demonstrates a *causal
   representational handle*, not subjective experience; frame as
   representation engineering.*

11. **Shadow-trunk auditing of API-only models — cross-model J-space
   transfer.** (2026-07-17) Frontier models served over an API expose only
   tokens: no hidden states, no logits, no injection path back into the
   residual stream. Their J-space cannot be tapped directly — but their
   *transcripts* can be teacher-forced through a local open-weight trunk,
   and the sidecar can read the local trunk's h_tap as a secondhand
   representation of the remote model's reasoning. If the divergence signal
   survives this transfer, the corrector stack becomes a provider-agnostic
   *external auditor*: a local monitor that reads any frontier model's
   token stream and flags derailment in its own J-space, with feedback
   limited to the token channel (steering text / forced anchors, per the
   grounding iteration-2 design).
   *Notes (agent): the harness already does the mechanics — harvest →
   teacher-force → h_tap dump → probe. The open question is empirical:
   does the error-anchored probe (AUC 0.985 on self-traces at 1.5B)
   still fire when the trace was written by a different model? First
   step, cheap: collect a few hundred GSM8K traces from any strong API
   model (mixed correct/incorrect), teacher-force through the 1.5B trunk,
   rerun the existing divergence/error-anchored probes, and compare AUC
   to the self-trace baseline. Prediction from the consensus-probe result
   (dispersion in h_tap tracks answer agreement regardless of source):
   partial transfer, degraded near error sites where the reader's own
   surprise may dominate the writer's error signature — worth separating
   'reader surprisal' from 'writer error' with a perplexity covariate.
   Dual framing for the paper: inside a provider the full loop attaches
   trivially (frozen flagship + hook + sidecar = the retrofit recipe,
   no retraining of the flagship); outside, the shadow-trunk auditor is
   the only deployable form. Also the honest answer to 'hook the sidecar
   to your own J-space': not possible across the API boundary — both the
   tap and the delta injection are severed; tokens are the only interface
   that crosses organizational boundaries.*

## Standing engineering debt

- Internal step count not compressed (superseded by 5b if it works).
- Snapshot/resume exists for training; evals should get the same treatment
  (multi-hour latent-SC runs currently restart from scratch on crash).
- Local driver instability (0xC0000005 / shm.dll) makes long runs retry-heavy;
  consider moving multi-hour evals to GCP when quota allows.
