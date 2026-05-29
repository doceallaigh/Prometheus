from __future__ import annotations

import argparse

from prometheus.config import load_config
from prometheus.train import run_training


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prometheus experiment CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Run a training job from a YAML config")
    train_parser.add_argument("--config", required=True, help="Path to the YAML config")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "train":
        config = load_config(args.config)
        run_dir = run_training(config)
        print(f"Run artifacts written to {run_dir}")


if __name__ == "__main__":
    main()