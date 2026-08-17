from __future__ import annotations

import argparse
import json
from pathlib import Path

from prometheus.config import load_config
from prometheus.reporting import comparison_markdown, summarize_run
from prometheus.train import run_training
from prometheus.visualization import write_structure_html


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser for training and reporting commands."""

    parser = argparse.ArgumentParser(description="Prometheus experiment CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Run a training job from a YAML config")
    train_parser.add_argument("--config", required=True, help="Path to the YAML config")

    summarize_parser = subparsers.add_parser("summarize-run", help="Summarize a completed run directory")
    summarize_parser.add_argument("--run-dir", required=True, help="Path to a run output directory")

    compare_parser = subparsers.add_parser("compare", help="Compare multiple completed run directories")
    compare_parser.add_argument("--run-dir", action="append", required=True, help="Path to a run output directory. Repeat for multiple runs.")
    compare_parser.add_argument("--output", help="Optional path to write the markdown comparison table")

    visualize_parser = subparsers.add_parser("visualize-model", help="Render an interactive HTML view of a model config")
    visualize_parser.add_argument("--config", required=True, help="Path to the YAML config")
    visualize_parser.add_argument("--output", required=True, help="Path to write the HTML visualization")

    evaluate_reasoning_parser = subparsers.add_parser(
        "evaluate-reasoning", help="Compare direct, cot, and optional latent reasoning on held-out problems"
    )
    evaluate_reasoning_parser.add_argument("--base-run", required=True, help="Run directory of the pretrained dense base")
    evaluate_reasoning_parser.add_argument("--latent-run", help="Optional run directory of a trained rrs-j-cfc loop")
    evaluate_reasoning_parser.add_argument("--num-problems", type=int, default=300, help="Held-out problems to evaluate")
    evaluate_reasoning_parser.add_argument("--device", default="auto", help="Evaluation device")
    evaluate_reasoning_parser.add_argument("--output", help="Optional path to write the markdown comparison report")
    evaluate_reasoning_parser.add_argument("--task-family", choices=["arithmetic", "rewrite", "both"], help="Override the base config's task family (task-transfer probe)")

    harvest_parser = subparsers.add_parser("retrofit-harvest", help="Harvest correct CoT traces from a frozen HF model on GSM8K train")
    harvest_parser.add_argument("--model", required=True, help="HuggingFace model name")
    harvest_parser.add_argument("--output", required=True, help="Path to write the JSONL trace file")
    harvest_parser.add_argument("--num-problems", type=int, default=2000, help="Train problems to attempt")
    harvest_parser.add_argument("--max-new-tokens", type=int, default=512, help="Generation budget per problem")
    harvest_parser.add_argument("--batch-size", type=int, default=8, help="Generation batch size")
    harvest_parser.add_argument("--dataset", choices=["gsm8k", "math"], default="gsm8k", help="Harvest corpus: GSM8K train or MATH train (numeric-answer subset, for strong trunks)")
    harvest_parser.add_argument("--quantize", action="store_true", help="Load the trunk in 4-bit NF4 (bitsandbytes) - local path for 32B-class models")
    harvest_parser.add_argument("--resume", action="store_true", help="Continue from the <output>.progress checkpoint, appending to the trace file")
    harvest_parser.add_argument("--device", default="auto", help="Device")

    retrofit_train_parser = subparsers.add_parser("retrofit-train", help="Distill a CfC corrector on harvested traces (trunk frozen)")
    retrofit_train_parser.add_argument("--model", required=True, help="HuggingFace model name")
    retrofit_train_parser.add_argument("--traces", required=True, help="JSONL trace file from retrofit-harvest")
    retrofit_train_parser.add_argument("--output-dir", required=True, help="Directory for corrector checkpoint and metrics")
    retrofit_train_parser.add_argument("--tap-layer", type=int, required=True, help="Hidden-states index to tap (0=embeddings)")
    retrofit_train_parser.add_argument("--d-cfc", type=int, default=512, help="Corrector hidden width")
    retrofit_train_parser.add_argument("--cell", default="cfc", choices=["cfc", "gru", "linear", "ssm", "mamba", "mamba2"], help="Recurrent core for the corrector")
    retrofit_train_parser.add_argument("--max-steps", type=int, default=2000, help="Training steps")
    retrofit_train_parser.add_argument("--learning-rate", type=float, default=1e-3, help="AdamW learning rate")
    retrofit_train_parser.add_argument("--answer-weight", type=float, default=2.0, help="Loss weight on the answer region")
    retrofit_train_parser.add_argument("--max-seq-len", type=int, default=640, help="Token cap per trace (prompt + completion)")
    retrofit_train_parser.add_argument("--quantize", action="store_true", help="Load the trunk in 4-bit NF4 (bitsandbytes) - local path for 32B-class models")
    retrofit_train_parser.add_argument("--bptt-chunk", type=int, default=0, help="Truncated-BPTT chunk size in tokens (0 = full-sequence backprop)")
    retrofit_train_parser.add_argument("--tap-project", help="Optional basis .pt from retrofit-jspace-verify --basis-out: restrict the corrector's input to a fixed subspace")
    retrofit_train_parser.add_argument("--tap-project-mode", default="remove", choices=["keep", "remove"], help="keep = top-J subspace only, remove = orthogonal complement only")
    retrofit_train_parser.add_argument("--tap-project-basis", default="full", choices=["local", "full"], help="Which saved influence basis to project against")
    retrofit_train_parser.add_argument("--seed", type=int, default=0, help="Nonzero: reproducible ensemble member (seeds init + offsets data order)")
    retrofit_train_parser.add_argument("--monitor-basis", help="Optional basis .pt: log complement-energy intrusion statistics (tap + delta) at every log step")
    retrofit_train_parser.add_argument("--device", default="auto", help="Device")

    ontogeny_parser = subparsers.add_parser("retrofit-ontogeny-sweep", help="Phase sweep: sidecar-gradient geometry vs trunk training step and trunk accuracy")
    ontogeny_parser.add_argument("--phase", action="append", required=True, help="Repeatable phase: <step>=<model_or_path>, or <step>=<base>::<lora_adapter>")
    ontogeny_parser.add_argument("--traces", required=True, help="JSONL trace file from retrofit-harvest")
    ontogeny_parser.add_argument("--basis", required=True, help="Basis .pt from retrofit-jspace-verify")
    ontogeny_parser.add_argument("--output-dir", required=True, help="Directory for metrics, plot, and report")
    ontogeny_parser.add_argument("--tap-layer", type=int, required=True, help="Hidden-states index to tap (0=embeddings)")
    ontogeny_parser.add_argument("--d-cfc", type=int, default=512, help="Sidecar hidden width for gradient diagnostics")
    ontogeny_parser.add_argument("--cell", default="cfc", choices=["cfc", "gru", "linear", "ssm", "mamba", "mamba2"], help="Sidecar recurrent core")
    ontogeny_parser.add_argument("--gradient-steps", type=int, default=128, help="Gradient-update steps per trunk phase")
    ontogeny_parser.add_argument("--learning-rate", type=float, default=1e-3, help="AdamW learning rate for gradient-update diagnostics")
    ontogeny_parser.add_argument("--answer-weight", type=float, default=2.0, help="Loss weight on the answer region")
    ontogeny_parser.add_argument("--max-seq-len", type=int, default=640, help="Token cap per trace (prompt + completion)")
    ontogeny_parser.add_argument("--eval-problems", type=int, default=200, help="Problems for trunk accuracy at each phase")
    ontogeny_parser.add_argument("--eval-max-new-tokens", type=int, default=512, help="Generation budget for trunk accuracy eval")
    ontogeny_parser.add_argument("--eval-batch-size", type=int, default=8, help="Batch size for trunk accuracy eval")
    ontogeny_parser.add_argument("--dataset", choices=["gsm8k", "math"], default="gsm8k", help="Evaluation dataset for trunk accuracy")
    ontogeny_parser.add_argument("--quantize", action="store_true", help="Load each trunk in 4-bit NF4 (bitsandbytes)")
    ontogeny_parser.add_argument("--seed", type=int, default=0, help="Seed offset for reproducible phase diagnostics")
    ontogeny_parser.add_argument("--device", default="auto", help="Device")

    ontogeny_content_parser = subparsers.add_parser("retrofit-ontogeny-content", help="Longitudinal complement prevalence, structure, contention, and gold-token recovery")
    ontogeny_content_parser.add_argument("--phase", action="append", required=True, help="Repeatable phase: <step>=<model>, or <step>=<base>::<lora_adapter>")
    ontogeny_content_parser.add_argument("--traces", required=True, help="Fixed correct traces replayed at every phase")
    ontogeny_content_parser.add_argument("--basis", required=True, help="Fixed mature Jacobian basis")
    ontogeny_content_parser.add_argument("--output-dir", required=True, help="Directory for metrics, report, and plot")
    ontogeny_content_parser.add_argument("--phase-metrics", help="Optional phase_metrics.jsonl supplying trunk strict accuracy")
    ontogeny_content_parser.add_argument("--tap-layer", type=int, default=12, help="Tap layer of the basis")
    ontogeny_content_parser.add_argument("--num-traces", type=int, default=128, help="Shared teacher-forced traces per phase")
    ontogeny_content_parser.add_argument("--max-seq-len", type=int, default=640, help="Maximum prompt plus completion length")
    ontogeny_content_parser.add_argument("--seed", type=int, default=1337, help="Random-basis and noise-control seed")
    ontogeny_content_parser.add_argument("--device", default="auto", help="Device")

    foundation_parser = subparsers.add_parser("foundation-ontogeny", help="Probe complement structure across foundational pretraining checkpoints")
    foundation_parser.add_argument("--model", default="EleutherAI/pythia-70m-deduped", help="HuggingFace model with revision checkpoints")
    foundation_parser.add_argument("--checkpoint", action="append", required=True, help="Repeatable <step>=<revision> checkpoint")
    foundation_parser.add_argument("--output-dir", required=True, help="Directory for bases, metrics, report, and plot")
    foundation_parser.add_argument("--num-windows", type=int, default=64, help="Shared C4 validation windows")
    foundation_parser.add_argument("--seq-len", type=int, default=128, help="Tokens per C4 window")
    foundation_parser.add_argument("--basis-windows", type=int, default=12, help="Windows used to fit each local basis")
    foundation_parser.add_argument("--positions", type=int, default=4, help="Jacobian positions per basis window")
    foundation_parser.add_argument("--directions", type=int, default=2, help="Random output directions per position")
    foundation_parser.add_argument("--rank", type=int, default=64, help="Influence basis rank")
    foundation_parser.add_argument("--tap-layer", type=int, help="Layer input to probe; default is midpoint")
    foundation_parser.add_argument("--seed", type=int, default=1337, help="Sampling and control seed")
    foundation_parser.add_argument("--device", default="auto", help="Device")

    foundation_train_parser = subparsers.add_parser("foundation-training-ablation", help="Train Pythia from initialization under fixed complement interventions")
    foundation_train_parser.add_argument("--model", default="EleutherAI/pythia-70m-deduped", help="HuggingFace causal LM")
    foundation_train_parser.add_argument("--revision", default="step0", help="Shared initialization revision")
    foundation_train_parser.add_argument("--basis", required=True, help="Fixed mature Jacobian basis")
    foundation_train_parser.add_argument("--output-dir", required=True, help="Directory for models, metrics, and summaries")
    foundation_train_parser.add_argument("--modes", default="full,complement-zero,complement-randomized,random-zero,random-randomized", help="Comma-separated training arms")
    foundation_train_parser.add_argument("--max-steps", type=int, default=4000, help="Optimizer steps per arm")
    foundation_train_parser.add_argument("--learning-rate", type=float, default=3e-4, help="AdamW learning rate")
    foundation_train_parser.add_argument("--batch-size", type=int, default=8, help="C4 windows per optimizer step")
    foundation_train_parser.add_argument("--seq-len", type=int, default=128, help="Tokens per C4 window")
    foundation_train_parser.add_argument("--train-windows", type=int, default=32768, help="Fixed C4 training windows; must cover every batch without replacement")
    foundation_train_parser.add_argument("--eval-windows", type=int, default=64, help="Fixed held-out C4 windows")
    foundation_train_parser.add_argument("--eval-interval", type=int, default=100, help="Validation interval in optimizer steps")
    foundation_train_parser.add_argument("--tap-layer", type=int, help="Layer input at which to intervene; default midpoint")
    foundation_train_parser.add_argument("--intervention-gate", choices=["all", "digit", "operator", "digit-or-operator"], default="all", help="Restrict intervention to next-token semantic events")
    foundation_train_parser.add_argument("--seed", type=int, default=1337, help="Initialization, data-order, and noise seed")
    foundation_train_parser.add_argument("--device", default="auto", help="Device")

    causal_parser = subparsers.add_parser("retrofit-causal-ablation-sweep", help="Measure task accuracy under complement interventions across adaptation checkpoints")
    causal_parser.add_argument("--phase", action="append", required=True, help="Repeatable <step>=<model> or <step>=<base>::<adapter>")
    causal_parser.add_argument("--basis", required=True, help="Fixed mature Jacobian basis")
    causal_parser.add_argument("--output-dir", required=True, help="Directory for paired outcomes and report")
    causal_parser.add_argument("--num-problems", type=int, default=200, help="GSM8K test problems")
    causal_parser.add_argument("--max-new-tokens", type=int, default=512, help="Generation budget")
    causal_parser.add_argument("--tap-layer", type=int, default=12, help="Residual stream layer input")
    causal_parser.add_argument("--seed", type=int, default=1337, help="Matched-noise and bootstrap seed")
    causal_parser.add_argument("--device", default="auto", help="Device")

    basin_parser = subparsers.add_parser("retrofit-basin-analyze", help="Analyze why complement suppression fails using fork trajectory basin proxies")
    basin_parser.add_argument("--dump", required=True, help="Complement-fork completions JSONL")
    basin_parser.add_argument("--output-dir", required=True, help="Directory for basin metrics, plot, and report")
    basin_parser.add_argument("--bootstrap-samples", type=int, default=10000, help="Bootstrap replicates for mean-difference intervals")
    basin_parser.add_argument("--seed", type=int, default=1337, help="Bootstrap seed")

    composition_parser = subparsers.add_parser("latent-composition", help="Multi-hop symbolic transitive logic with a continuous recurrent sidecar")
    composition_parser.add_argument("--output-dir", required=True, help="Directory for proofs, checkpoints, metrics, plot, and report")
    composition_parser.add_argument("--train-proofs", type=int, default=1000, help="Programmatically generated short-chain training proofs")
    composition_parser.add_argument("--test-per-depth", type=int, default=200, help="Held-out proofs at each test depth")
    composition_parser.add_argument("--train-min-depth", type=int, default=2, help="Minimum training chain length")
    composition_parser.add_argument("--train-max-depth", type=int, default=4, help="Maximum training chain length k")
    composition_parser.add_argument("--test-max-depth", type=int, default=10, help="Maximum OOD test chain length")
    composition_parser.add_argument("--entities", type=int, default=24, help="Entity vocabulary size")
    composition_parser.add_argument("--distractors", type=int, default=4, help="Unrelated edges per proof")
    composition_parser.add_argument("--d-model", type=int, default=128, help="Trunk and sidecar hidden width")
    composition_parser.add_argument("--trunk-steps", type=int, default=1200, help="No-CoT trunk training steps")
    composition_parser.add_argument("--sidecar-steps", type=int, default=1800, help="Continuous sidecar training steps")
    composition_parser.add_argument("--batch-size", type=int, default=64, help="Training and evaluation batch size")
    composition_parser.add_argument("--learning-rate", type=float, default=3e-4, help="AdamW learning rate")
    composition_parser.add_argument("--seed", type=int, default=20260720, help="Dataset and training seed")
    composition_parser.add_argument("--device", default="auto", help="Device")

    retrofit_eval_parser = subparsers.add_parser("retrofit-eval", help="Compare direct/cot/latent on GSM8K test")
    retrofit_eval_parser.add_argument("--model", required=True, help="HuggingFace model name")
    retrofit_eval_parser.add_argument("--corrector", help="Optional corrector.pt from retrofit-train")
    retrofit_eval_parser.add_argument("--num-problems", type=int, default=200, help="Test problems to evaluate")
    retrofit_eval_parser.add_argument("--max-new-tokens", type=int, default=512, help="Generation budget per problem")
    retrofit_eval_parser.add_argument("--latent-samples", type=int, default=1, help="Latent self-consistency sample count (1 = greedy only)")
    retrofit_eval_parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature for latent self-consistency")
    retrofit_eval_parser.add_argument("--stop-agree", type=int, default=0, help="Adaptive vote: stop sampling once N rollouts agree (0 = fixed k)")
    retrofit_eval_parser.add_argument("--problem-offset", type=int, default=0, help="Start of the test-split slice (disjoint-slice validation)")
    retrofit_eval_parser.add_argument("--sequential-rollouts", action="store_true", help="Disable batched rollouts (one-at-a-time sampling)")
    retrofit_eval_parser.add_argument("--dataset", choices=["gsm8k", "math"], default="gsm8k", help="Test corpus: GSM8K test or MATH test (numeric-answer subset)")
    retrofit_eval_parser.add_argument("--quantize", action="store_true", help="Load the trunk in 4-bit NF4 (bitsandbytes) - local path for 32B-class models")
    retrofit_eval_parser.add_argument("--quorum", type=int, default=1, help="Corrector quorum size: k members vote on the delta each step (1 = single corrector)")
    retrofit_eval_parser.add_argument("--quorum-noise", type=float, default=0.0, help="Relative tap-noise std for replica quorum members (member 0 stays clean)")
    retrofit_eval_parser.add_argument("--quorum-agg", default="mean", choices=["mean", "median", "sign"], help="Delta aggregation across quorum members")
    retrofit_eval_parser.add_argument("--quorum-agree", type=float, default=1.0, help="Sign agg: minimum fraction of members agreeing on a coordinate's sign")
    retrofit_eval_parser.add_argument("--quorum-correctors", nargs="+", help="Extra corrector.pt paths for a true ensemble quorum (same tap layer)")
    retrofit_eval_parser.add_argument("--dynamic-sc", type=int, default=0, help="Dynamic self-consistency beam cap: branch sampled siblings only where the corrector's delta-norm z-score fires (0 = off)")
    retrofit_eval_parser.add_argument("--branch-z", type=float, default=2.5, help="Dynamic SC: per-beam z-score threshold on ||delta|| that triggers a branch")
    retrofit_eval_parser.add_argument("--branch-cooldown", type=int, default=16, help="Dynamic SC: minimum decode steps between branches on one beam")
    retrofit_eval_parser.add_argument("--seed", type=int, default=0, help="Nonzero: seed stochastic latent rollouts reproducibly")
    retrofit_eval_parser.add_argument("--device", default="auto", help="Device")
    retrofit_eval_parser.add_argument("--output", help="Optional path to write the markdown report")

    multiturn_parser = subparsers.add_parser(
        "retrofit-eval-multiturn",
        help="Compare visible CoT with latent KV-cache persistence across GSM8K conversation turns",
    )
    multiturn_parser.add_argument("--model", required=True, help="HuggingFace model name")
    multiturn_parser.add_argument("--corrector", required=True, help="corrector.pt from retrofit-train")
    multiturn_parser.add_argument("--episodes", type=int, default=50, help="Number of multi-turn episodes")
    multiturn_parser.add_argument("--turns", type=int, default=3, help="Turns per episode; dependent mode requires 3")
    multiturn_parser.add_argument("--max-new-tokens", type=int, default=512, help="Generation budget per turn")
    multiturn_parser.add_argument("--problem-offset", type=int, default=0, help="Start of the GSM8K test slice")
    multiturn_parser.add_argument(
        "--context-mode",
        choices=["independent", "dependent"],
        default="independent",
        help="Independent problems measure interference; dependent turns probe hidden-context carryover",
    )
    multiturn_parser.add_argument("--sc-samples", type=int, default=1, help="SC rollout cap; values above 1 add visible SC and adaptive latent SC arms")
    multiturn_parser.add_argument("--stop-agreement", type=int, default=4, help="Adaptive latent SC votes required to stop")
    multiturn_parser.add_argument("--temperature", type=float, default=0.6, help="SC sampling temperature")
    multiturn_parser.add_argument("--seed", type=int, default=20260804, help="SC sampling seed")
    multiturn_parser.add_argument("--device", default="auto", help="Device")
    multiturn_parser.add_argument("--output", help="Optional path to write the Markdown report and JSON records")

    grounded_parser = subparsers.add_parser("retrofit-grounded", help="Continuous latent reasoning with periodic token grounding (dose-response over grounding frequency)")
    grounded_parser.add_argument("--model", required=True, help="HuggingFace model name")
    grounded_parser.add_argument("--corrector", help="Optional corrector.pt from retrofit-train")
    grounded_parser.add_argument("--num-problems", type=int, default=100, help="Test problems to evaluate")
    grounded_parser.add_argument("--reasoning-steps", type=int, default=300, help="Internal reasoning steps before the forced answer decode")
    grounded_parser.add_argument("--ground-every", default="1,4,8,16,0", help="Comma-separated grounding periods (1 = control, 0 = never/pure continuous)")
    grounded_parser.add_argument("--feedback", choices=["expected", "hidden"], default="expected", help="Continuous feedback vector: expected embedding E_p[e] or norm-matched final hidden state")
    grounded_parser.add_argument("--snap", help="Optional snap.pt from retrofit-train-snap (learned manifold re-projection of the feedback vector)")
    grounded_parser.add_argument("--output", help="Optional path to write the markdown report")
    grounded_parser.add_argument("--device", default="auto", help="Device")

    dynamics_parser = subparsers.add_parser("retrofit-train-dynamics", help="Train a tap-space dynamics model: predict the next tap state (the sandwich architecture's latent reasoner)")
    dynamics_parser.add_argument("--model", required=True, help="HuggingFace model name")
    dynamics_parser.add_argument("--traces", required=True, help="JSONL trace file from retrofit-harvest")
    dynamics_parser.add_argument("--output-dir", required=True, help="Directory for dynamics checkpoint and metrics")
    dynamics_parser.add_argument("--tap-layer", type=int, required=True, help="Hidden-states index to tap (0=embeddings)")
    dynamics_parser.add_argument("--d-cfc", type=int, default=512, help="Dynamics cell hidden width")
    dynamics_parser.add_argument("--cell", default="cfc", choices=["cfc", "gru", "linear", "ssm", "mamba", "mamba2"], help="Recurrent core")
    dynamics_parser.add_argument("--max-steps", type=int, default=2000, help="Training steps")
    dynamics_parser.add_argument("--learning-rate", type=float, default=1e-3, help="AdamW learning rate")
    dynamics_parser.add_argument("--max-seq-len", type=int, default=640, help="Token cap per trace (prompt + completion)")
    dynamics_parser.add_argument("--bptt-chunk", type=int, default=0, help="Truncated-BPTT chunk size in tokens (0 = full-sequence backprop)")
    dynamics_parser.add_argument("--device", default="auto", help="Device")

    sandwich_parser = subparsers.add_parser("retrofit-eval-sandwich", help="Sandwich rollout eval: discrete encoder | latent CfC recurrence | discrete decoder (no trunk forwards during reasoning)")
    sandwich_parser.add_argument("--model", required=True, help="HuggingFace model name")
    sandwich_parser.add_argument("--dynamics", required=True, help="dynamics.pt from retrofit-train-dynamics")
    sandwich_parser.add_argument("--num-problems", type=int, default=200, help="Test problems to evaluate")
    sandwich_parser.add_argument("--latent-steps", type=int, default=300, help="Latent recurrence budget (tap-space steps before decoding)")
    sandwich_parser.add_argument("--answer-tokens", type=int, default=24, help="Token-space answer-phase budget after the anchor")
    sandwich_parser.add_argument("--max-new-tokens", type=int, default=512, help="Generation budget for the cot reference arm")
    sandwich_parser.add_argument("--device", default="auto", help="Device")
    sandwich_parser.add_argument("--output", help="Optional path to write the markdown report")

    cfork_parser = subparsers.add_parser("retrofit-cfork", help="Complement-fork rollout: fork on intrusion excursions, suppress in root / reinforce in offshoot, report-back gate at the end")
    cfork_parser.add_argument("--model", required=True, help="HuggingFace model name")
    cfork_parser.add_argument("--corrector", required=True, help="Corrector checkpoint from retrofit-train")
    cfork_parser.add_argument("--basis", required=True, help="Basis .pt from retrofit-jspace-verify (defines the complement)")
    cfork_parser.add_argument("--num-problems", type=int, default=200, help="Test problems to evaluate")
    cfork_parser.add_argument("--max-new-tokens", type=int, default=512, help="Generation budget per branch")
    cfork_parser.add_argument("--max-branches", type=int, default=4, help="Branch cap per problem")
    cfork_parser.add_argument("--fork-z", type=float, default=2.5, help="Complement-energy z-score that triggers a fork")
    cfork_parser.add_argument("--fork-cooldown", type=int, default=16, help="Decode steps between forks on one branch")
    cfork_parser.add_argument("--gamma", type=float, default=1.0, help="Gain on the offshoot injection (excursion or dominant component)")
    cfork_parser.add_argument("--persist", type=int, default=4, help="Forwards over which the suppress/reinforce injection persists")
    cfork_parser.add_argument("--child-mode", default="suppress-dominant", choices=["suppress-dominant", "reinforce"], help="Offshoot treatment: suppress the dominant Jacobian component (default) or reinforce the complement excursion")
    cfork_parser.add_argument("--steer-mode", default="closed-loop", choices=["closed-loop", "open-loop"], help="Injection controller: proportional closed-loop (default) or legacy fixed-persist open-loop")
    cfork_parser.add_argument("--no-hull", action="store_true", help="Disable convex-hull constraints (injection clipping + norm-preserving hook)")
    cfork_parser.add_argument("--arbiter", help="Optional BranchArbiter checkpoint from retrofit-train-arbiter; adds the learned arbiter gate")
    cfork_parser.add_argument("--tap-snap", help="Optional tap-manifold snap projector from retrofit-train-tap-snap, applied to injected rows post norm-match")
    cfork_parser.add_argument("--device", default="auto", help="Device")
    cfork_parser.add_argument("--output", help="Optional path to write the markdown report")

    fharvest_parser = subparsers.add_parser("retrofit-fork-harvest", help="Harvest complement-fork branch traces with gold labels on GSM8K train (arbiter training data)")
    fharvest_parser.add_argument("--model", required=True, help="HuggingFace model name")
    fharvest_parser.add_argument("--corrector", required=True, help="Corrector checkpoint from retrofit-train")
    fharvest_parser.add_argument("--basis", required=True, help="Basis .pt from retrofit-jspace-verify")
    fharvest_parser.add_argument("--num-problems", type=int, default=1000, help="Train problems to scan")
    fharvest_parser.add_argument("--max-new-tokens", type=int, default=512, help="Generation budget per branch")
    fharvest_parser.add_argument("--stride", type=int, default=8, help="Record every k-th tap state")
    fharvest_parser.add_argument("--fork-z", type=float, default=2.5, help="Fork trigger z-score")
    fharvest_parser.add_argument("--max-branches", type=int, default=4, help="Branch cap per problem")
    fharvest_parser.add_argument("--gamma", type=float, default=1.0, help="Injection gain")
    fharvest_parser.add_argument("--child-mode", default="suppress-dominant", choices=["suppress-dominant", "reinforce"], help="Offshoot treatment")
    fharvest_parser.add_argument("--steer-mode", default="closed-loop", choices=["closed-loop", "open-loop"], help="Injection controller")
    fharvest_parser.add_argument("--no-hull", action="store_true", help="Disable convex-hull constraints")
    fharvest_parser.add_argument("--device", default="auto", help="Device")
    fharvest_parser.add_argument("--output", required=True, help="Output .pt path for the trace archive")

    tarb_parser = subparsers.add_parser("retrofit-train-arbiter", help="Train the bidirectional cross-attention BranchArbiter on harvested fork traces")
    tarb_parser.add_argument("--traces", required=True, help="Trace archive from retrofit-fork-harvest")
    tarb_parser.add_argument("--d-arb", type=int, default=128, help="Arbiter width")
    tarb_parser.add_argument("--heads", type=int, default=4, help="Attention heads")
    tarb_parser.add_argument("--layers", type=int, default=2, help="Encoder layers")
    tarb_parser.add_argument("--steps", type=int, default=3000, help="Training steps")
    tarb_parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    tarb_parser.add_argument("--holdout", type=int, default=50, help="Problems held out for validation")
    tarb_parser.add_argument("--device", default="auto", help="Device")
    tarb_parser.add_argument("--output", required=True, help="Output .pt path for the arbiter checkpoint")

    clens_parser = subparsers.add_parser("retrofit-complement-lens", help="Decode dominant/complement tap components with the trunk's upper half: do intrusions carry structured contending content?")
    clens_parser.add_argument("--model", required=True, help="HuggingFace model name")
    clens_parser.add_argument("--basis", required=True, help="Basis .pt from retrofit-jspace-verify")
    clens_parser.add_argument("--dump", required=True, help="Completions .jsonl with latent rollouts to replay")
    clens_parser.add_argument("--num-problems", type=int, default=200, help="Problems to analyze")
    clens_parser.add_argument("--tap-layer", type=int, default=12, help="Tap layer of the basis")
    clens_parser.add_argument("--device", default="auto", help="Device")
    clens_parser.add_argument("--output", help="Optional path to write the markdown report")

    tsnap_parser = subparsers.add_parser("retrofit-train-tap-snap", help="Train a tap-manifold snap projector: learned projection for fork injections")
    tsnap_parser.add_argument("--model", required=True, help="HuggingFace model name")
    tsnap_parser.add_argument("--traces", required=True, help="Harvested traces .jsonl (correct CoT traces)")
    tsnap_parser.add_argument("--basis", required=True, help="Basis .pt from retrofit-jspace-verify")
    tsnap_parser.add_argument("--tap-layer", type=int, default=12, help="Tap layer")
    tsnap_parser.add_argument("--steps", type=int, default=3000, help="Training steps")
    tsnap_parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    tsnap_parser.add_argument("--device", default="auto", help="Device")
    tsnap_parser.add_argument("--output", required=True, help="Output .pt path")

    amon_parser = subparsers.add_parser("retrofit-answer-monitor", help="Monitor the corrector error vector during the #### answer phase; AUC + answer-gated sampling policies")
    amon_parser.add_argument("--model", required=True, help="HuggingFace model name")
    amon_parser.add_argument("--corrector", required=True, help="Corrector checkpoint")
    amon_parser.add_argument("--basis", required=True, help="Basis .pt from retrofit-jspace-verify")
    amon_parser.add_argument("--num-problems", type=int, default=200, help="Test problems")
    amon_parser.add_argument("--samples", type=int, default=8, help="Sampled rollouts per problem for the policy arm")
    amon_parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature")
    amon_parser.add_argument("--max-new-tokens", type=int, default=512, help="Generation budget")
    amon_parser.add_argument("--device", default="auto", help="Device")
    amon_parser.add_argument("--output", help="Optional markdown report path")

    snap_parser = subparsers.add_parser("retrofit-train-snap", help="Train a snap projector: residual MLP mapping continuous feedback vectors back onto the token-embedding manifold")
    snap_parser.add_argument("--model", required=True, help="HuggingFace model name")
    snap_parser.add_argument("--traces", required=True, help="JSONL trace file from retrofit-harvest")
    snap_parser.add_argument("--output", required=True, help="Path for the snap.pt checkpoint")
    snap_parser.add_argument("--corrector", help="Optional corrector.pt so training logits match deployment (corrector active)")
    snap_parser.add_argument("--snap-input", choices=["expected", "hidden"], default="expected", help="Feedback distribution to train against")
    snap_parser.add_argument("--d-hidden", type=int, default=512, help="Snap MLP hidden width")
    snap_parser.add_argument("--max-steps", type=int, default=2000, help="Training steps")
    snap_parser.add_argument("--learning-rate", type=float, default=1e-3, help="AdamW learning rate")
    snap_parser.add_argument("--max-seq-len", type=int, default=640, help="Token cap per trace")
    snap_parser.add_argument("--device", default="auto", help="Device")

    baseline_eval_parser = subparsers.add_parser("retrofit-baseline-eval", help="Trunk baselines: few-shot CoT, self-consistency@k, LoRA")
    baseline_eval_parser.add_argument("--model", required=True, help="HuggingFace model name")
    baseline_eval_parser.add_argument("--num-problems", type=int, default=200, help="Test problems to evaluate")
    baseline_eval_parser.add_argument("--max-new-tokens", type=int, default=512, help="Generation budget per sample")
    baseline_eval_parser.add_argument("--shots", type=int, default=0, help="Few-shot exemplars to prepend (0-4)")
    baseline_eval_parser.add_argument("--samples", type=int, default=1, help="Self-consistency sample count (1 = greedy)")
    baseline_eval_parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature when samples > 1")
    baseline_eval_parser.add_argument("--lora-dir", help="Optional LoRA adapter directory from retrofit-lora-train")
    baseline_eval_parser.add_argument("--device", default="auto", help="Device")
    baseline_eval_parser.add_argument("--output", help="Optional path to write the markdown report")

    lora_train_parser = subparsers.add_parser("retrofit-lora-train", help="Param-matched LoRA fine-tune on harvested traces")
    lora_train_parser.add_argument("--model", required=True, help="HuggingFace model name")
    lora_train_parser.add_argument("--traces", required=True, help="JSONL trace file from retrofit-harvest")
    lora_train_parser.add_argument("--output-dir", required=True, help="Directory for the LoRA adapter and metrics")
    lora_train_parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    lora_train_parser.add_argument("--max-steps", type=int, default=3000, help="Training steps")
    lora_train_parser.add_argument("--learning-rate", type=float, default=1e-4, help="AdamW learning rate")
    lora_train_parser.add_argument("--answer-weight", type=float, default=2.0, help="Loss weight on the answer region")
    lora_train_parser.add_argument("--checkpoint-steps", default="", help="Comma-separated completed steps at which to save LoRA phase adapters (0 includes initialization)")
    lora_train_parser.add_argument("--seed", type=int, default=1337, help="Seed for LoRA initialization and trace order")
    lora_train_parser.add_argument("--intervention", choices=["full", "complement-zero", "dominant-zero", "complement-randomized", "random-zero", "random-dominant-zero", "random-randomized"], default="full", help="Residual-stream intervention active throughout training")
    lora_train_parser.add_argument("--basis", help="Fixed Jacobian basis required for complement intervention")
    lora_train_parser.add_argument("--tap-layer", type=int, default=12, help="Layer input at which to intervene")
    lora_train_parser.add_argument("--intervention-gate", choices=["all", "sidecar-high", "digit", "operator", "digit-or-operator"], default="all", help="Restrict intervention to fixed reference events")
    lora_train_parser.add_argument("--gate-corrector", help="Frozen corrector checkpoint required by --intervention-gate sidecar-high")
    lora_train_parser.add_argument("--gate-masks", help="Reuse an exact gate_masks.pt from a matched treatment arm")
    lora_train_parser.add_argument("--gate-threshold-z", type=float, default=2.0, help="Within-trace sidecar delta-norm z threshold")
    lora_train_parser.add_argument("--basis-refresh-interval", type=int, default=0, help="Recompute the measured Jacobian basis every N completed updates; 0 keeps it fixed")
    lora_train_parser.add_argument("--basis-refresh-traces", type=int, default=8, help="Fixed leading traces used for each dynamic basis refresh")
    lora_train_parser.add_argument("--basis-refresh-positions", type=int, default=8, help="Completion positions sampled per basis-refresh trace")
    lora_train_parser.add_argument("--basis-refresh-directions", type=int, default=4, help="VJP output directions per refresh position")
    lora_train_parser.add_argument("--basis-refresh-at-start", action="store_true", help="Also fit a dynamic basis before the first optimizer update")
    lora_train_parser.add_argument("--device", default="auto", help="Device")

    flops_parser = subparsers.add_parser("retrofit-flops", help="Direct FLOPs evaluation: sidecar overhead vs token savings, from an eval dump")
    flops_parser.add_argument("--model", required=True, help="HuggingFace model name (config + tokenizer only, no GPU)")
    flops_parser.add_argument("--dump", required=True, help="Path to a *.completions.jsonl file from retrofit-eval")
    flops_parser.add_argument("--corrector", help="Optional corrector.pt for exact sidecar shapes (default: d-cfc width)")
    flops_parser.add_argument("--d-cfc", type=int, default=512, help="Corrector width when no checkpoint is given")
    flops_parser.add_argument("--output", help="Optional path to write the markdown report")

    latency_parser = subparsers.add_parser("retrofit-latency", help="Benchmark synchronized prompt-to-answer latency")
    latency_parser.add_argument("--model", required=True, help="HuggingFace model name")
    latency_parser.add_argument("--corrector", required=True, help="Corrector checkpoint")
    latency_parser.add_argument("--num-problems", type=int, default=20, help="GSM8K test problems to time")
    latency_parser.add_argument("--max-new-tokens", type=int, default=512, help="Generation budget per rollout")
    latency_parser.add_argument("--temperature", type=float, default=0.6, help="SC sampling temperature")
    latency_parser.add_argument("--seed", type=int, default=20260725, help="Sampling seed")
    latency_parser.add_argument("--device", default="auto", help="Device")
    latency_parser.add_argument("--output", help="Optional path to write the markdown and JSON reports")

    jspace_parser = subparsers.add_parser("retrofit-jspace-verify", help="Verify the residual-stream tap: staged reconstruction + Jacobian influence-subspace alignment + projection ablation")
    jspace_parser.add_argument("--model", required=True, help="HuggingFace model name")
    jspace_parser.add_argument("--traces", required=True, help="JSONL trace file from retrofit-harvest")
    jspace_parser.add_argument("--corrector", help="Corrector checkpoint (supplies tap layer and enables alignment)")
    jspace_parser.add_argument("--tap-layer", type=int, help="Tap layer when no corrector is given")
    jspace_parser.add_argument("--num-traces", type=int, default=8, help="Traces to analyze")
    jspace_parser.add_argument("--positions", type=int, default=8, help="Sampled completion positions per trace")
    jspace_parser.add_argument("--directions", type=int, default=4, help="Random probe directions per position")
    jspace_parser.add_argument("--rank", type=int, default=64, help="Influence-basis rank for alignment scoring")
    jspace_parser.add_argument("--max-seq-len", type=int, default=640, help="Trace truncation length")
    jspace_parser.add_argument("--quantize", action="store_true", help="Load the trunk in 4-bit NF4")
    jspace_parser.add_argument("--device", default="auto", help="Device")
    jspace_parser.add_argument("--output", help="Optional path to write the markdown report")
    jspace_parser.add_argument("--basis-out", help="Optional .pt path to save the influence bases for projection evals")

    divergence_parser = subparsers.add_parser("divergence-label", help="Label divergence onsets from a latent-SC completions dump")
    divergence_parser.add_argument("--dump", required=True, help="Path to a *.completions.jsonl file with latent_sc rollouts")
    divergence_parser.add_argument("--output", required=True, help="Path for the labels JSONL")
    divergence_parser.add_argument("--anchor", choices=["fork", "error"], default="fork", help="fork = sampling divergence point; error = first unsanctioned computed value (gold <<>> annotations)")

    probe_parser = subparsers.add_parser("divergence-probe", help="Fit the divergence recognition curve (AUC vs tokens past fork)")
    probe_parser.add_argument("--model", required=True, help="HuggingFace model name")
    probe_parser.add_argument("--corrector", required=True, help="Corrector checkpoint from retrofit-train")
    probe_parser.add_argument("--dump", required=True, help="Path to a *.completions.jsonl file with latent_sc rollouts")
    probe_parser.add_argument("--output", required=True, help="Path for the markdown report")
    probe_parser.add_argument("--max-pairs", type=int, default=400, help="Max wrong/correct rollout pairs")
    probe_parser.add_argument("--device", default="cuda", help="Device")
    probe_parser.add_argument("--anchor", choices=["fork", "error"], default="fork", help="Probe anchor: sampling fork or arithmetic error site")

    adaptive_parser = subparsers.add_parser("rollback-simulate", help="Simulate agreement-based early stopping over latent-SC rollouts")
    adaptive_parser.add_argument("--dump", required=True, help="Path to a *.completions.jsonl file with latent_sc rollouts")
    adaptive_parser.add_argument("--output", required=True, help="Path for the markdown report")

    rollback_parser = subparsers.add_parser("rollback-oracle", help="Oracle-ceiling rollback: rewind wrong latent rollouts to the error site and resample")
    rollback_parser.add_argument("--model", required=True, help="HuggingFace model name")
    rollback_parser.add_argument("--corrector", required=True, help="Corrector checkpoint from retrofit-train")
    rollback_parser.add_argument("--dump", required=True, help="Path to a *.completions.jsonl file with latent rollouts")
    rollback_parser.add_argument("--output", required=True, help="Path for the markdown report")
    rollback_parser.add_argument("--budget", type=int, default=4, help="Max re-rolls per wrong rollout")
    rollback_parser.add_argument("--temperature", type=float, default=0.6, help="Re-roll sampling temperature")
    rollback_parser.add_argument("--rewind-margin", type=int, default=8, help="Tokens to rewind before the error site")
    rollback_parser.add_argument("--max-new-tokens", type=int, default=512, help="Generation budget per re-roll")
    rollback_parser.add_argument("--device", default="cuda", help="Device")

    probe_rollback_parser = subparsers.add_parser("rollback-probe", help="Deployable rollback: h_tap probe scanner triggers rewind+resample, no gold labels at inference")
    probe_rollback_parser.add_argument("--model", required=True, help="HuggingFace model name")
    probe_rollback_parser.add_argument("--corrector", required=True, help="Corrector checkpoint from retrofit-train")
    probe_rollback_parser.add_argument("--dump", required=True, help="Path to a *.completions.jsonl file with latent + latent_sc rollouts")
    probe_rollback_parser.add_argument("--output", required=True, help="Path for the markdown report")
    probe_rollback_parser.add_argument("--budget", type=int, default=4, help="Max re-rolls per triggered rollout")
    probe_rollback_parser.add_argument("--temperature", type=float, default=0.6, help="Re-roll sampling temperature")
    probe_rollback_parser.add_argument("--rewind-margin", type=int, default=8, help="Tokens to rewind before the trigger position")
    probe_rollback_parser.add_argument("--max-new-tokens", type=int, default=512, help="Generation budget per re-roll")
    probe_rollback_parser.add_argument("--threshold-fpr", type=float, default=0.05, help="Per-rollout false-alarm rate used to calibrate the trigger threshold")
    probe_rollback_parser.add_argument("--device", default="cuda", help="Device")

    steer_parser = subparsers.add_parser("steer-inject", help="Steering-vector repair: inject a contrastive 'wait-' vector at the tap layer instead of rolling back")
    steer_parser.add_argument("--model", required=True, help="HuggingFace model name")
    steer_parser.add_argument("--corrector", required=True, help="Corrector checkpoint from retrofit-train")
    steer_parser.add_argument("--dump", required=True, help="Path to a *.completions.jsonl file with latent + latent_sc rollouts")
    steer_parser.add_argument("--output", required=True, help="Path for the markdown report")
    steer_parser.add_argument("--trigger", choices=["oracle", "probe", "cusum"], default="oracle", help="Trigger localization: annotation error sites (ceiling), deployable h_tap probe, or probe + 8-token CUSUM (trigger-lab winner)")
    steer_parser.add_argument("--alphas", default="0,1,2,4", help="Comma-separated steering scales (0 = determinism control)")
    steer_parser.add_argument("--steer-window", type=int, default=24, help="Decode steps to keep the injection active")
    steer_parser.add_argument("--inject-offset", type=int, default=4, help="Tokens past the trigger to keep in context before steering")
    steer_parser.add_argument("--max-new-tokens", type=int, default=512, help="Generation budget per regeneration")
    steer_parser.add_argument("--threshold-fpr", type=float, default=0.20, help="Rollout false-alarm rate for the probe trigger (relaxed: false alarms are cheap)")
    steer_parser.add_argument("--gate", choices=["none", "margin"], default="none", help="Confidence gating: scale alpha by the trigger margin over threshold (precision lever)")
    steer_parser.add_argument("--device", default="cuda", help="Device")

    trigger_lab_parser = subparsers.add_parser("trigger-lab", help="Offline detection-rule bake-off: score h_tap streams once, compare trigger rules with no regeneration")
    trigger_lab_parser.add_argument("--model", required=True, help="HuggingFace model name")
    trigger_lab_parser.add_argument("--corrector", required=True, help="Corrector checkpoint (for tap layer config)")
    trigger_lab_parser.add_argument("--dump", required=True, help="Path to a *.completions.jsonl file with latent + latent_sc rollouts")
    trigger_lab_parser.add_argument("--output", required=True, help="Path for the markdown report")
    trigger_lab_parser.add_argument("--basis", help="Optional basis .pt from retrofit-jspace-verify: adds complement-energy/frac trigger rules (intrusive-thoughts hypothesis)")
    trigger_lab_parser.add_argument("--device", default="cuda", help="Device")

    consensus_parser = subparsers.add_parser("consensus-probe", help="State-space consensus: does early h_tap trajectory dispersion predict vote disagreement?")
    consensus_parser.add_argument("--model", required=True, help="HuggingFace model name")
    consensus_parser.add_argument("--corrector", required=True, help="Corrector checkpoint (for tap layer config)")
    consensus_parser.add_argument("--dump", required=True, help="Path to a *.completions.jsonl file with latent_sc8 rollouts")
    consensus_parser.add_argument("--output", required=True, help="Path for the markdown report")
    consensus_parser.add_argument("--device", default="cuda", help="Device")

    optical_parser = subparsers.add_parser("optical-resume", help="Optical context compression: resume own CoT from a rendered image, sweep compression")
    optical_parser.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct", help="HuggingFace VLM name")
    optical_parser.add_argument("--num-problems", type=int, default=100, help="Test problems to evaluate")
    optical_parser.add_argument("--prefix-fraction", type=float, default=0.6, help="Fraction of the chain handed back for resumption")
    optical_parser.add_argument("--scales", default="1.0,0.75,0.5,0.35", help="Comma-separated image downscale factors (optical compression sweep)")
    optical_parser.add_argument("--font-size", type=int, default=14, help="Render font size in px")
    optical_parser.add_argument("--max-new-tokens", type=int, default=512, help="Generation budget")
    optical_parser.add_argument("--output", help="Optional path to write the markdown report")
    optical_parser.add_argument("--device", default="cuda", help="Device")

    fit_linear_parser = subparsers.add_parser("retrofit-fit-linear", help="Closed-form (ridge/Procrustes) linear corrector baseline: fit without gradient training")
    fit_linear_parser.add_argument("--model", required=True, help="HuggingFace model name")
    fit_linear_parser.add_argument("--traces", required=True, help="JSONL trace file from retrofit-harvest")
    fit_linear_parser.add_argument("--output", required=True, help="Path for the fitted corrector checkpoint (.pt)")
    fit_linear_parser.add_argument("--tap-layer", type=int, required=True, help="Hidden-states index to tap")
    fit_linear_parser.add_argument("--mode", choices=["ridge", "procrustes"], default="ridge", help="Closed-form solver")
    fit_linear_parser.add_argument("--ridge-lambda", type=float, default=1.0, help="Ridge regularization strength")
    fit_linear_parser.add_argument("--target", choices=["geometric", "grad"], default="geometric", help="Fit target: v2-style geometric or zero-anchored CE-gradient")
    fit_linear_parser.add_argument("--scale", type=float, default=1.0, help="Inference-time multiplier on the fitted map (stored in checkpoint)")
    fit_linear_parser.add_argument("--device", default="cuda", help="Device")
    return parser


def main() -> None:
    """Dispatch the selected CLI command."""

    parser = build_parser()
    args = parser.parse_args()
    if args.command == "train":
        config = load_config(args.config)
        if config.model.architecture == "rrs_j_cfc":
            from prometheus.latent_reasoning import run_latent_distillation

            run_dir = run_latent_distillation(config)
        else:
            run_dir = run_training(config)
        print(f"Run artifacts written to {run_dir}")
        return
    if args.command == "summarize-run":
        print(json.dumps(summarize_run(args.run_dir), indent=2))
        return
    if args.command == "compare":
        summaries = [summarize_run(path) for path in args.run_dir]
        markdown = comparison_markdown(summaries)
        if args.output:
            Path(args.output).write_text(markdown, encoding="utf-8")
        print(markdown)
        return
    if args.command == "visualize-model":
        config = load_config(args.config)
        output_path = write_structure_html(config, args.output)
        print(f"Visualization written to {output_path}")
        return
    if args.command == "evaluate-reasoning":
        from prometheus.latent_reasoning import compare_reasoning_systems, comparison_report_markdown
        from prometheus.train import resolve_device

        results = compare_reasoning_systems(
            base_run_dir=args.base_run,
            latent_run_dir=args.latent_run,
            num_problems=args.num_problems,
            device=resolve_device(args.device),
            task_family=args.task_family,
        )
        markdown = comparison_report_markdown(results)
        if args.output:
            Path(args.output).write_text(markdown, encoding="utf-8")
        print(markdown)
        return
    if args.command == "retrofit-harvest":
        from prometheus.retrofit import harvest

        harvest(
            model_name=args.model,
            output_path=args.output,
            num_problems=args.num_problems,
            max_new_tokens=args.max_new_tokens,
            device_str=args.device,
            batch_size=args.batch_size,
            dataset_name=args.dataset,
            quantize=args.quantize,
            resume=args.resume,
        )
        return
    if args.command == "retrofit-train":
        from prometheus.retrofit import train_corrector

        train_corrector(
            model_name=args.model,
            traces_path=args.traces,
            output_dir=args.output_dir,
            tap_layer=args.tap_layer,
            d_cfc=args.d_cfc,
            max_steps=args.max_steps,
            learning_rate=args.learning_rate,
            answer_weight=args.answer_weight,
            device_str=args.device,
            max_seq_len=args.max_seq_len,
            quantize=args.quantize,
            bptt_chunk=args.bptt_chunk,
            cell=args.cell,
            tap_project=args.tap_project,
            tap_project_mode=args.tap_project_mode,
            tap_project_basis=args.tap_project_basis,
            seed=args.seed,
            monitor_basis=args.monitor_basis,
        )
        return
    if args.command == "retrofit-ontogeny-sweep":
        from prometheus.retrofit import ontogeny_sweep

        phase_models = []
        for spec in args.phase:
            if "=" not in spec:
                raise ValueError(f"Invalid --phase spec '{spec}'. Expected <step>=<model_or_path>")
            step_raw, model_name = spec.split("=", 1)
            phase_models.append((int(step_raw.strip()), model_name.strip()))

        ontogeny_sweep(
            phase_models=phase_models,
            traces_path=args.traces,
            basis_path=args.basis,
            output_dir=args.output_dir,
            tap_layer=args.tap_layer,
            d_cfc=args.d_cfc,
            cell=args.cell,
            gradient_steps=args.gradient_steps,
            learning_rate=args.learning_rate,
            answer_weight=args.answer_weight,
            max_seq_len=args.max_seq_len,
            eval_problems=args.eval_problems,
            eval_max_new_tokens=args.eval_max_new_tokens,
            eval_batch_size=args.eval_batch_size,
            dataset_name=args.dataset,
            device_str=args.device,
            quantize=args.quantize,
            seed=args.seed,
        )
        return
    if args.command == "retrofit-ontogeny-content":
        from prometheus.ontogeny_content import ontogeny_content_sweep

        phase_models = []
        for spec in args.phase:
            if "=" not in spec:
                raise ValueError(f"Invalid --phase spec '{spec}'. Expected <step>=<model_or_path>")
            step_raw, model_name = spec.split("=", 1)
            phase_models.append((int(step_raw.strip()), model_name.strip()))
        ontogeny_content_sweep(
            phase_models=phase_models,
            traces_path=args.traces,
            basis_path=args.basis,
            output_dir=args.output_dir,
            tap_layer=args.tap_layer,
            num_traces=args.num_traces,
            max_seq_len=args.max_seq_len,
            device_str=args.device,
            seed=args.seed,
            phase_metrics_path=args.phase_metrics,
        )
        return
    if args.command == "foundation-ontogeny":
        from prometheus.ontogeny_experiments import foundational_pretraining_sweep

        checkpoints = []
        for spec in args.checkpoint:
            step_raw, revision = spec.split("=", 1)
            checkpoints.append((int(step_raw), revision))
        foundational_pretraining_sweep(
            model_name=args.model,
            checkpoints=checkpoints,
            output_dir=args.output_dir,
            num_windows=args.num_windows,
            seq_len=args.seq_len,
            basis_windows=args.basis_windows,
            positions_per_window=args.positions,
            directions=args.directions,
            rank=args.rank,
            layer_index=args.tap_layer,
            device_str=args.device,
            seed=args.seed,
        )
        return
    if args.command == "foundation-training-ablation":
        from prometheus.ontogeny_experiments import foundational_training_ablation_sweep

        foundational_training_ablation_sweep(
            model_name=args.model,
            revision=args.revision,
            basis_path=args.basis,
            output_dir=args.output_dir,
            modes=tuple(mode.strip() for mode in args.modes.split(",") if mode.strip()),
            max_steps=args.max_steps,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            train_windows=args.train_windows,
            eval_windows=args.eval_windows,
            eval_interval=args.eval_interval,
            layer_index=args.tap_layer,
            device_str=args.device,
            seed=args.seed,
            intervention_gate=args.intervention_gate,
        )
        return
    if args.command == "retrofit-causal-ablation-sweep":
        from prometheus.ontogeny_experiments import causal_ablation_sweep

        phases = []
        for spec in args.phase:
            step_raw, model_spec = spec.split("=", 1)
            phases.append((int(step_raw), model_spec))
        causal_ablation_sweep(
            phase_models=phases,
            basis_path=args.basis,
            output_dir=args.output_dir,
            num_problems=args.num_problems,
            max_new_tokens=args.max_new_tokens,
            tap_layer=args.tap_layer,
            device_str=args.device,
            seed=args.seed,
        )
        return
    if args.command == "retrofit-basin-analyze":
        from prometheus.basin_analysis import analyze_suppression_basins

        analyze_suppression_basins(
            dump_path=args.dump,
            output_dir=args.output_dir,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        )
        return
    if args.command == "latent-composition":
        from prometheus.latent_composition import run_latent_composition

        run_latent_composition(
            output_dir=args.output_dir,
            train_proofs=args.train_proofs,
            test_per_depth=args.test_per_depth,
            train_min_depth=args.train_min_depth,
            train_max_depth=args.train_max_depth,
            test_max_depth=args.test_max_depth,
            entities=args.entities,
            distractors=args.distractors,
            d_model=args.d_model,
            trunk_steps=args.trunk_steps,
            sidecar_steps=args.sidecar_steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            device_str=args.device,
            seed=args.seed,
        )
        return
    if args.command == "retrofit-eval":
        from prometheus.retrofit import evaluate_retrofit

        evaluate_retrofit(
            model_name=args.model,
            corrector_path=args.corrector,
            num_problems=args.num_problems,
            max_new_tokens=args.max_new_tokens,
            device_str=args.device,
            output_path=args.output,
            latent_samples=args.latent_samples,
            temperature=args.temperature,
            stop_agreement=args.stop_agree,
            problem_offset=args.problem_offset,
            sequential_rollouts=args.sequential_rollouts,
            dataset_name=args.dataset,
            quantize=args.quantize,
            quorum=args.quorum,
            quorum_noise=args.quorum_noise,
            quorum_agg=args.quorum_agg,
            quorum_agree=args.quorum_agree,
            quorum_correctors=args.quorum_correctors,
            dynamic_sc=args.dynamic_sc,
            branch_z=args.branch_z,
            branch_cooldown=args.branch_cooldown,
            seed=args.seed,
        )
        return
    if args.command == "retrofit-eval-multiturn":
        from prometheus.retrofit import evaluate_multiturn_retrofit

        evaluate_multiturn_retrofit(
            model_name=args.model,
            corrector_path=args.corrector,
            num_episodes=args.episodes,
            turns=args.turns,
            max_new_tokens=args.max_new_tokens,
            device_str=args.device,
            output_path=args.output,
            problem_offset=args.problem_offset,
            context_mode=args.context_mode,
            sc_samples=args.sc_samples,
            stop_agreement=args.stop_agreement,
            temperature=args.temperature,
            seed=args.seed,
        )
        return
    if args.command == "retrofit-flops":
        from prometheus.retrofit import flops_report

        flops_report(
            model_name=args.model,
            dump_path=args.dump,
            output_path=args.output,
            corrector_path=args.corrector,
            d_cfc=args.d_cfc,
        )
        return
    if args.command == "retrofit-latency":
        from prometheus.retrofit import benchmark_retrofit_latency

        benchmark_retrofit_latency(
            model_name=args.model,
            corrector_path=args.corrector,
            num_problems=args.num_problems,
            max_new_tokens=args.max_new_tokens,
            device_str=args.device,
            output_path=args.output,
            temperature=args.temperature,
            seed=args.seed,
        )
        return
    if args.command == "retrofit-jspace-verify":
        from prometheus.retrofit import jspace_verify

        jspace_verify(
            model_name=args.model,
            traces_path=args.traces,
            output_path=args.output,
            corrector_path=args.corrector,
            tap_layer=args.tap_layer,
            device_str=args.device,
            num_traces=args.num_traces,
            positions_per_trace=args.positions,
            directions=args.directions,
            rank=args.rank,
            max_seq_len=args.max_seq_len,
            quantize=args.quantize,
            basis_out=args.basis_out,
        )
        return
    if args.command == "retrofit-grounded":
        from prometheus.retrofit import evaluate_grounded_continuous

        evaluate_grounded_continuous(
            model_name=args.model,
            corrector_path=args.corrector,
            num_problems=args.num_problems,
            reasoning_steps=args.reasoning_steps,
            device_str=args.device,
            output_path=args.output,
            ground_every_values=tuple(int(g) for g in args.ground_every.split(",")),
            feedback=args.feedback,
            snap_path=args.snap,
        )
        return
    if args.command == "retrofit-train-dynamics":
        from prometheus.retrofit import train_dynamics

        train_dynamics(
            model_name=args.model,
            traces_path=args.traces,
            output_dir=args.output_dir,
            tap_layer=args.tap_layer,
            d_cfc=args.d_cfc,
            max_steps=args.max_steps,
            learning_rate=args.learning_rate,
            device_str=args.device,
            max_seq_len=args.max_seq_len,
            bptt_chunk=args.bptt_chunk,
            cell=args.cell,
        )
        return
    if args.command == "retrofit-eval-sandwich":
        from prometheus.retrofit import evaluate_sandwich

        evaluate_sandwich(
            model_name=args.model,
            dynamics_path=args.dynamics,
            num_problems=args.num_problems,
            latent_steps=args.latent_steps,
            device_str=args.device,
            output_path=args.output,
            answer_tokens=args.answer_tokens,
            max_new_tokens=args.max_new_tokens,
        )
        return
    if args.command == "retrofit-cfork":
        from prometheus.retrofit import evaluate_complement_fork

        evaluate_complement_fork(
            model_name=args.model,
            corrector_path=args.corrector,
            basis_path=args.basis,
            num_problems=args.num_problems,
            max_new_tokens=args.max_new_tokens,
            device_str=args.device,
            output_path=args.output,
            max_branches=args.max_branches,
            fork_z=args.fork_z,
            fork_cooldown=args.fork_cooldown,
            gamma=args.gamma,
            persist=args.persist,
            child_mode=args.child_mode,
            steer_mode=args.steer_mode,
            hull=not args.no_hull,
            arbiter_path=args.arbiter,
            tap_snap_path=args.tap_snap,
        )
        return
    if args.command == "retrofit-fork-harvest":
        from prometheus.retrofit import harvest_fork_traces

        harvest_fork_traces(
            model_name=args.model,
            corrector_path=args.corrector,
            basis_path=args.basis,
            num_problems=args.num_problems,
            max_new_tokens=args.max_new_tokens,
            device_str=args.device,
            out_path=args.output,
            stride=args.stride,
            max_branches=args.max_branches,
            fork_z=args.fork_z,
            gamma=args.gamma,
            child_mode=args.child_mode,
            steer_mode=args.steer_mode,
            hull=not args.no_hull,
        )
        return
    if args.command == "retrofit-train-arbiter":
        from prometheus.retrofit import train_arbiter

        train_arbiter(
            traces_path=args.traces,
            out_path=args.output,
            d_arb=args.d_arb,
            heads=args.heads,
            layers=args.layers,
            steps=args.steps,
            lr=args.lr,
            device_str=args.device,
            holdout=args.holdout,
        )
        return
    if args.command == "retrofit-complement-lens":
        from prometheus.retrofit import complement_lens

        complement_lens(
            model_name=args.model,
            basis_path=args.basis,
            dump_path=args.dump,
            num_problems=args.num_problems,
            device_str=args.device,
            output_path=args.output,
            tap_layer=args.tap_layer,
        )
        return
    if args.command == "retrofit-train-tap-snap":
        from prometheus.retrofit import train_tap_snap

        train_tap_snap(
            model_name=args.model,
            traces_path=args.traces,
            basis_path=args.basis,
            output_path=args.output,
            tap_layer=args.tap_layer,
            max_steps=args.steps,
            learning_rate=args.lr,
            device_str=args.device,
        )
        return
    if args.command == "retrofit-answer-monitor":
        from prometheus.retrofit import answer_monitor

        answer_monitor(
            model_name=args.model,
            corrector_path=args.corrector,
            basis_path=args.basis,
            num_problems=args.num_problems,
            samples=args.samples,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            device_str=args.device,
            output_path=args.output,
        )
        return
    if args.command == "retrofit-train-snap":
        from prometheus.retrofit import train_snap

        train_snap(
            model_name=args.model,
            traces_path=args.traces,
            output_path=args.output,
            corrector_path=args.corrector,
            snap_input=args.snap_input,
            d_hidden=args.d_hidden,
            max_steps=args.max_steps,
            learning_rate=args.learning_rate,
            max_seq_len=args.max_seq_len,
            device_str=args.device,
        )
        return
    if args.command == "retrofit-baseline-eval":
        from prometheus.retrofit_baselines import evaluate_baseline

        evaluate_baseline(
            model_name=args.model,
            num_problems=args.num_problems,
            max_new_tokens=args.max_new_tokens,
            device_str=args.device,
            output_path=args.output,
            shots=args.shots,
            samples=args.samples,
            temperature=args.temperature,
            lora_dir=args.lora_dir,
        )
        return
    if args.command == "retrofit-lora-train":
        from prometheus.retrofit_baselines import train_lora

        train_lora(
            model_name=args.model,
            traces_path=args.traces,
            output_dir=args.output_dir,
            lora_r=args.lora_r,
            max_steps=args.max_steps,
            learning_rate=args.learning_rate,
            answer_weight=args.answer_weight,
            device_str=args.device,
            checkpoint_steps=tuple(int(step) for step in args.checkpoint_steps.split(",") if step.strip()),
            seed=args.seed,
            intervention=args.intervention,
            basis_path=args.basis,
            tap_layer=args.tap_layer,
            intervention_gate=args.intervention_gate,
            gate_corrector_path=args.gate_corrector,
            gate_threshold_z=args.gate_threshold_z,
            gate_masks_path=args.gate_masks,
            basis_refresh_interval=args.basis_refresh_interval,
            basis_refresh_traces=args.basis_refresh_traces,
            basis_refresh_positions=args.basis_refresh_positions,
            basis_refresh_directions=args.basis_refresh_directions,
            basis_refresh_at_start=args.basis_refresh_at_start,
        )
        return
    if args.command == "divergence-label":
        if args.anchor == "error":
            from prometheus.divergence import label_error_sites

            label_error_sites(dump_path=args.dump, output_path=args.output)
        else:
            from prometheus.divergence import label_divergence

            label_divergence(dump_path=args.dump, output_path=args.output)
        return
    if args.command == "divergence-probe":
        from prometheus.divergence import probe_divergence

        probe_divergence(
            model_name=args.model,
            corrector_path=args.corrector,
            dump_path=args.dump,
            output_path=args.output,
            device_str=args.device,
            max_pairs=args.max_pairs,
            anchor=args.anchor,
        )
        return
    if args.command == "rollback-simulate":
        from prometheus.divergence import simulate_adaptive_sc

        simulate_adaptive_sc(dump_path=args.dump, output_path=args.output)
        return
    if args.command == "rollback-oracle":
        from prometheus.divergence import oracle_rollback

        oracle_rollback(
            model_name=args.model,
            corrector_path=args.corrector,
            dump_path=args.dump,
            output_path=args.output,
            device_str=args.device,
            budget=args.budget,
            temperature=args.temperature,
            rewind_margin_tokens=args.rewind_margin,
            max_new_tokens=args.max_new_tokens,
        )
        return
    if args.command == "rollback-probe":
        from prometheus.divergence import probe_rollback

        probe_rollback(
            model_name=args.model,
            corrector_path=args.corrector,
            dump_path=args.dump,
            output_path=args.output,
            device_str=args.device,
            budget=args.budget,
            temperature=args.temperature,
            rewind_margin_tokens=args.rewind_margin,
            max_new_tokens=args.max_new_tokens,
            threshold_fpr=args.threshold_fpr,
        )
        return
    if args.command == "steer-inject":
        from prometheus.divergence import steer_inject

        steer_inject(
            model_name=args.model,
            corrector_path=args.corrector,
            dump_path=args.dump,
            output_path=args.output,
            device_str=args.device,
            trigger=args.trigger,
            alphas=tuple(float(a) for a in args.alphas.split(",")),
            steer_window=args.steer_window,
            inject_offset=args.inject_offset,
            max_new_tokens=args.max_new_tokens,
            threshold_fpr=args.threshold_fpr,
            gate=args.gate,
        )
        return
    if args.command == "trigger-lab":
        from prometheus.divergence import trigger_lab

        trigger_lab(
            model_name=args.model,
            corrector_path=args.corrector,
            dump_path=args.dump,
            output_path=args.output,
            device_str=args.device,
            basis_path=args.basis,
        )
        return
    if args.command == "consensus-probe":
        from prometheus.divergence import consensus_probe

        consensus_probe(
            model_name=args.model,
            corrector_path=args.corrector,
            dump_path=args.dump,
            output_path=args.output,
            device_str=args.device,
        )
        return
    if args.command == "optical-resume":
        from prometheus.optical import evaluate_optical_resume

        evaluate_optical_resume(
            model_name=args.model,
            num_problems=args.num_problems,
            device_str=args.device,
            output_path=args.output,
            prefix_fraction=args.prefix_fraction,
            scales=tuple(float(s) for s in args.scales.split(",")),
            font_size=args.font_size,
            max_new_tokens=args.max_new_tokens,
        )
        return
    if args.command == "retrofit-fit-linear":
        from prometheus.retrofit import fit_linear_corrector

        fit_linear_corrector(
            model_name=args.model,
            traces_path=args.traces,
            output_path=args.output,
            tap_layer=args.tap_layer,
            device_str=args.device,
            mode=args.mode,
            ridge_lambda=args.ridge_lambda,
            target=args.target,
            scale=args.scale,
        )
        return


if __name__ == "__main__":
    main()