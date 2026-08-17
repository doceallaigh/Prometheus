"""Train, evaluate, and aggregate a reproducible corrector tap-layer sweep."""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str]) -> None:
    for attempt in range(3):
        result = subprocess.run(command, cwd=ROOT)
        if result.returncode == 0:
            return
        if attempt < 2:
            print(f"command failed with {result.returncode}; retrying ({attempt + 2}/3)", flush=True)
            time.sleep(1)
    raise subprocess.CalledProcessError(result.returncode, command)


def complete_dump(path: Path, expected: int) -> bool:
    if not path.exists():
        return False
    indices = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            indices.add(json.loads(line)["index"])
        except json.JSONDecodeError:
            return False
    return indices == set(range(expected))


def parse_report(path: Path) -> dict[str, float]:
    rows = {}
    pattern = re.compile(r"^\| ([a-z0-9_]+) \| ([0-9.]+) \| ([0-9.]+) \| ([0-9.]+) \| ([0-9.]+) \|$")
    for line in path.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line)
        if match:
            rows[match.group(1)] = [float(value) for value in match.groups()[1:]]
    return {
        "greedy_strict": rows["latent"][0],
        "greedy_lenient": rows["latent"][1],
        "sc8_accuracy": rows["latent_sc8"][1],
        "mean_internal_tokens": rows["latent_sc8"][3],
    }


def aggregate(layers: list[int], output_root: Path, report_root: Path) -> list[dict]:
    rows = []
    for layer in layers:
        metrics_path = output_root / f"layer-{layer:02d}" / "metrics.jsonl"
        report_path = report_root / f"layer-{layer:02d}.md"
        if not metrics_path.exists() or not report_path.exists():
            continue
        final_metric = json.loads(metrics_path.read_text(encoding="utf-8").splitlines()[-1])
        rows.append({"layer": layer, "final_loss": final_metric["loss"], **parse_report(report_path)})

    csv_path = report_root / "metrics.csv"
    report_root.mkdir(parents=True, exist_ok=True)
    fields = ["layer", "final_loss", "greedy_strict", "greedy_lenient", "sc8_accuracy", "mean_internal_tokens"]
    with csv_path.open("w", newline="", encoding="utf-8") as sink:
        writer = csv.DictWriter(sink, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Corrector tap-layer sweep", "",
        "Qwen2.5-0.5B-Instruct; identical seed-20260725 CfC correctors; 3,000 training steps;",
        "GSM8K first 200 test problems; greedy and latent SC@8 at temperature 0.6.", "",
        "| layer | final loss | greedy strict | greedy lenient | latent SC@8 | SC@8 internal tokens |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['layer']} | {row['final_loss']:.4f} | {row['greedy_strict']:.3f} | "
            f"{row['greedy_lenient']:.3f} | {row['sc8_accuracy']:.3f} | {row['mean_internal_tokens']:.1f} |"
        )
    lines += [
        "", "The controlled sweep does not reproduce a unique midpoint optimum. Greedy",
        "accuracy rises late and peaks at layer 23 (0.490 lenient), while SC@8 is broad",
        "and irregular: layers 17 and 22 tie at 0.565, versus 0.525 at layer 12 and",
        "0.545 at the final state. Across all taps SC@8 spans 0.480--0.565. These data",
        "support tap robustness and a weak late-stack preference, not a sharply localized",
        "repair layer. The original four-tap pilot remains the historical reason layer 12",
        "was selected, but it confounded tap position with an uncontrolled initialization;",
        "the common-seed sweep is the stronger layer-selection result.",
    ]
    (report_root.parent / "20260725-tap-layer-sweep-qwen05b.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("train", "eval", "all", "aggregate"), default="all")
    parser.add_argument("--layers", default="0-24")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--traces", default="outputs/retrofit-qwen05b/traces-7k.jsonl")
    parser.add_argument("--output-root", default="outputs/tap-layer-sweep-qwen05b")
    parser.add_argument("--report-root", default="reports/20260725-tap-layer-sweep-qwen05b")
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--num-problems", type=int, default=200)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    layers = []
    for part in args.layers.split(","):
        if "-" in part:
            start, end = map(int, part.split("-", 1))
            layers.extend(range(start, end + 1))
        else:
            layers.append(int(part))
    layers = sorted(set(layers))
    output_root = ROOT / args.output_root
    report_root = ROOT / args.report_root

    if args.stage in {"train", "all"}:
        for layer in layers:
            output_dir = output_root / f"layer-{layer:02d}"
            checkpoint = output_dir / "corrector.pt"
            if checkpoint.exists():
                continue
            run([
                sys.executable, "-m", "prometheus.cli", "retrofit-train",
                "--model", args.model, "--traces", args.traces,
                "--output-dir", str(output_dir), "--tap-layer", str(layer),
                "--d-cfc", "512", "--max-steps", str(args.max_steps),
                "--learning-rate", "0.001", "--answer-weight", "2",
                "--max-seq-len", "640", "--seed", str(args.seed), "--device", args.device,
            ])

    if args.stage in {"eval", "all"}:
        for layer in layers:
            checkpoint = output_root / f"layer-{layer:02d}" / "corrector.pt"
            if not checkpoint.exists():
                raise FileNotFoundError(checkpoint)
            report = report_root / f"layer-{layer:02d}.md"
            if complete_dump(report.with_suffix(".completions.jsonl"), args.num_problems):
                continue
            run([
                sys.executable, "-m", "prometheus.cli", "retrofit-eval",
                "--model", args.model, "--corrector", str(checkpoint),
                "--num-problems", str(args.num_problems), "--max-new-tokens", str(args.max_new_tokens),
                "--latent-samples", "8", "--temperature", "0.6", "--seed", str(args.seed),
                "--device", args.device, "--output", str(report),
            ])
            aggregate(layers, output_root, report_root)

    rows = aggregate(layers, output_root, report_root)
    print(json.dumps({"completed_layers": len(rows), "layers": [row["layer"] for row in rows]}))


if __name__ == "__main__":
    main()