from __future__ import annotations

import argparse
import json
from pathlib import Path

from prometheus.config import load_config
from prometheus.reporting import comparison_markdown, summarize_run
from prometheus.train import run_training


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
    return parser


def main() -> None:
    """Dispatch the selected CLI command."""

    parser = build_parser()
    args = parser.parse_args()
    if args.command == "train":
        config = load_config(args.config)
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


if __name__ == "__main__":
    main()