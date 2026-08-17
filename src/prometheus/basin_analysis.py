from __future__ import annotations

import json
import math
import random
from collections import Counter
from pathlib import Path

from prometheus.retrofit import extract_answer_lenient


FEATURES = (
    "root_answer_consensus",
    "branch_answer_diversity",
    "root_mean_logprob",
    "root_mean_delta",
    "root_max_delta",
    "root_intrusion_rate",
    "root_steps",
)


def _auc(negative: list[float], positive: list[float]) -> float:
    """Probability that a random positive score exceeds a random negative score."""

    if not negative or not positive:
        return float("nan")
    wins = 0.0
    for positive_value in positive:
        for negative_value in negative:
            wins += float(positive_value > negative_value)
            wins += 0.5 * float(positive_value == negative_value)
    return wins / (len(negative) * len(positive))


def _bootstrap_difference(
    failures: list[float], rescues: list[float], samples: int, seed: int
) -> tuple[float, float, float]:
    if not failures or not rescues:
        return float("nan"), float("nan"), float("nan")
    observed = sum(failures) / len(failures) - sum(rescues) / len(rescues)
    generator = random.Random(seed)
    differences = []
    for _ in range(samples):
        failure_mean = sum(generator.choice(failures) for _ in failures) / len(failures)
        rescue_mean = sum(generator.choice(rescues) for _ in rescues) / len(rescues)
        differences.append(failure_mean - rescue_mean)
    differences.sort()
    low = differences[int(0.025 * (samples - 1))]
    high = differences[int(0.975 * (samples - 1))]
    return observed, low, high


def _trajectory_row(record: dict) -> dict | None:
    branches = record.get("branches", [])
    root = next((branch for branch in branches if branch.get("root") is True), None)
    if root is None:
        return None

    gold = str(record["gold"])
    latent_answer = extract_answer_lenient(record.get("latent", ""))
    root_answer = extract_answer_lenient(root.get("text", ""))
    branch_answers = [extract_answer_lenient(branch.get("text", "")) for branch in branches]
    branch_answers = [answer for answer in branch_answers if answer is not None]
    counts = Counter(branch_answers)
    consensus = counts[root_answer] / len(branch_answers) if root_answer is not None and branch_answers else 0.0
    diversity = len(counts) / len(branch_answers) if branch_answers else 0.0

    latent_correct = latent_answer == gold
    root_correct = root_answer == gold
    if latent_correct and root_correct:
        outcome = "preserved"
    elif latent_correct:
        outcome = "harm"
    elif root_correct:
        outcome = "rescue"
    else:
        outcome = "suppression_failure"

    return {
        "index": record.get("index"),
        "gold": gold,
        "latent_answer": latent_answer,
        "root_answer": root_answer,
        "outcome": outcome,
        "branch_count": len(branches),
        "correct_alternative_present": any(answer == gold for answer in branch_answers),
        "root_answer_consensus": consensus,
        "branch_answer_diversity": diversity,
        "root_mean_logprob": float(root.get("mean_logprob", 0.0)),
        "root_mean_delta": float(root.get("mean_delta", 0.0)),
        "root_max_delta": float(root.get("max_delta", 0.0)),
        "root_intrusion_rate": float(root.get("intrusion_rate", 0.0)),
        "root_steps": float(root.get("steps", 0.0)),
    }


def _plot_html(rows: list[dict], comparisons: list[dict]) -> str:
    failures = [row for row in rows if row["outcome"] == "suppression_failure"]
    rescues = [row for row in rows if row["outcome"] == "rescue"]
    outcomes = ["suppression_failure", "rescue", "preserved", "harm"]
    counts = [sum(row["outcome"] == outcome for row in rows) for outcome in outcomes]
    auc_labels = [row["feature"] for row in comparisons]
    auc_values = [row["auc_failure_vs_rescue"] for row in comparisons]
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Attractor Basin Suppression Analysis</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body {{ margin: 0; padding: 20px; background: #101418; color: #edf2f4; font-family: Segoe UI, sans-serif; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 16px; }}
    .panel {{ background: #182027; border: 1px solid #34434f; border-radius: 8px; padding: 12px; }}
  </style>
</head>
<body>
  <h1>Attractor Basin Suppression Analysis</h1>
  <p>Failure means the unperturbed latent answer and the complement-suppressed root are both wrong.</p>
  <div class="grid">
    <div class="panel"><div id="outcomes" style="height:380px"></div></div>
    <div class="panel"><div id="consensus" style="height:380px"></div></div>
    <div class="panel"><div id="auc" style="height:420px"></div></div>
  </div>
  <script>
    const template = 'plotly_dark';
    Plotly.newPlot('outcomes', [{{type:'bar', x:{json.dumps(outcomes)}, y:{json.dumps(counts)}}}],
      {{template, title:'Intervention outcomes', yaxis:{{title:'problems'}}}}, {{responsive:true, displaylogo:false}});
    Plotly.newPlot('consensus', [
      {{type:'box', name:'failure', y:{json.dumps([row['root_answer_consensus'] for row in failures])}}},
      {{type:'box', name:'rescue', y:{json.dumps([row['root_answer_consensus'] for row in rescues])}}}
    ], {{template, title:'Wrong-basin persistence after suppression', yaxis:{{title:'fraction of branches sharing root answer'}}}},
      {{responsive:true, displaylogo:false}});
    Plotly.newPlot('auc', [{{type:'bar', orientation:'h', y:{json.dumps(auc_labels)}, x:{json.dumps(auc_values)}}}],
      {{template, title:'Univariate discrimination: failure vs rescue', xaxis:{{title:'AUC', range:[0,1]}}}},
      {{responsive:true, displaylogo:false}});
  </script>
</body>
</html>
"""


def analyze_suppression_basins(
    dump_path: str | Path,
    output_dir: str | Path,
    bootstrap_samples: int = 10_000,
    seed: int = 1337,
) -> dict:
    """Explain complement-suppression failures using branch-level basin proxies."""

    records = [json.loads(line) for line in Path(dump_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [row for record in records if (row := _trajectory_row(record)) is not None]
    if not rows:
        raise ValueError(f"No complement-fork trajectories found in {dump_path}")

    failures = [row for row in rows if row["outcome"] == "suppression_failure"]
    rescues = [row for row in rows if row["outcome"] == "rescue"]
    comparisons = []
    for feature_index, feature in enumerate(FEATURES):
        failure_values = [float(row[feature]) for row in failures]
        rescue_values = [float(row[feature]) for row in rescues]
        difference, ci_low, ci_high = _bootstrap_difference(
            failure_values,
            rescue_values,
            samples=bootstrap_samples,
            seed=seed + feature_index,
        )
        comparisons.append({
            "feature": feature,
            "failure_mean": sum(failure_values) / len(failure_values) if failure_values else float("nan"),
            "rescue_mean": sum(rescue_values) / len(rescue_values) if rescue_values else float("nan"),
            "failure_minus_rescue": difference,
            "ci95_low": ci_low,
            "ci95_high": ci_high,
            "auc_failure_vs_rescue": _auc(rescue_values, failure_values),
        })

    outcome_counts = Counter(row["outcome"] for row in rows)
    wrong_rows = failures + rescues
    correct_alternative_rate = (
        sum(bool(row["correct_alternative_present"]) for row in failures) / len(failures) if failures else float("nan")
    )
    rescue_rate = len(rescues) / len(wrong_rows) if wrong_rows else float("nan")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows_path = output / "basin_rows.jsonl"
    summary_path = output / "summary.json"
    report_path = output / "report.md"
    plot_path = output / "basin_plot.html"
    rows_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    summary = {
        "dump": str(dump_path),
        "problems": len(rows),
        "outcomes": dict(outcome_counts),
        "rescue_rate_given_latent_wrong": rescue_rate,
        "failure_correct_alternative_rate": correct_alternative_rate,
        "comparisons": comparisons,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plot_path.write_text(_plot_html(rows, comparisons), encoding="utf-8")

    lines = [
        "# Attractor Basin Suppression Analysis",
        "",
        f"Source: `{dump_path}`; problems: {len(rows)}.",
        "",
        "The analysis treats the complement-suppressed root as a causal intervention, not merely a detector. "
        "A suppression failure is a wrong unperturbed latent answer that remains wrong after suppression; a rescue becomes correct.",
        "",
        "## Outcomes",
        "",
        "| outcome | count |",
        "| --- | ---: |",
    ]
    for outcome in ("suppression_failure", "rescue", "preserved", "harm"):
        lines.append(f"| {outcome} | {outcome_counts.get(outcome, 0)} |")
    lines.extend([
        "",
        f"Rescue rate given an initially wrong latent answer: {rescue_rate:.3f}.",
        f"Suppression failures with a correct answer present in another branch: {correct_alternative_rate:.3f}.",
        "",
        "## Failure versus rescue",
        "",
        "| feature | failure mean | rescue mean | difference | bootstrap 95% CI | AUC |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in comparisons:
        lines.append(
            f"| {row['feature']} | {row['failure_mean']:.4f} | {row['rescue_mean']:.4f} | "
            f"{row['failure_minus_rescue']:.4f} | [{row['ci95_low']:.4f}, {row['ci95_high']:.4f}] | "
            f"{row['auc_failure_vs_rescue']:.4f} |"
        )
    lines.extend([
        "",
        "## Interpretation boundary",
        "",
        "These data identify computational conditions under which suppression fails in this model. They do not directly "
        "measure biological inhibition, neuromodulation, or clinical hallucination, so any comparison with healthy brains "
        "must remain a mechanistic analogy until tested against neural data.",
        "",
        f"Interactive plot: `{plot_path}`.",
    ])
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary), flush=True)
    return summary