from __future__ import annotations

import json
import sys
import tempfile
import unittest
from importlib import util
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock

from prometheus import cli
from prometheus.train import run_training

from tests.test_train_reporting_inference import make_config


HAS_INFERENCE = util.find_spec("prometheus.inference") is not None


class CliTests(unittest.TestCase):
    def test_compare_command_prints_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"
            run_dir.mkdir()
            (run_dir / "config.snapshot.json").write_text(
                json.dumps({"model": {"architecture": "dense", "vocab_size": 12}, "data": {"sequence_length": 8}}),
                encoding="utf-8",
            )
            (run_dir / "metrics.jsonl").write_text(
                json.dumps({"split": "train", "step": 0, "loss": 1.0, "perplexity": 2.7}) + "\n"
                + json.dumps({"split": "val", "step": 0, "loss": 0.8, "perplexity": 2.2})
                + "\n",
                encoding="utf-8",
            )
            buffer = StringIO()
            with mock.patch.object(sys, "argv", ["prometheus", "compare", "--run-dir", str(run_dir)]):
                with redirect_stdout(buffer):
                    cli.main()

        self.assertIn("| run | architecture |", buffer.getvalue())

    def test_summarize_run_command_prints_json(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"
            run_dir.mkdir()
            (run_dir / "config.snapshot.json").write_text(
                json.dumps({"model": {"architecture": "dense", "vocab_size": 12}, "data": {"sequence_length": 8}}),
                encoding="utf-8",
            )
            (run_dir / "metrics.jsonl").write_text(
                json.dumps({"split": "train", "step": 0, "loss": 1.0, "perplexity": 2.7}) + "\n"
                + json.dumps({"split": "val", "step": 0, "loss": 0.8, "perplexity": 2.2})
                + "\n",
                encoding="utf-8",
            )
            buffer = StringIO()
            with mock.patch.object(sys, "argv", ["prometheus", "summarize-run", "--run-dir", str(run_dir)]):
                with redirect_stdout(buffer):
                    cli.main()

        self.assertIn('"architecture": "dense"', buffer.getvalue())

    def test_visualize_model_command_writes_html(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.yaml"
            config_path.write_text(
                """
experiment:
    run_name: visual-test
    seed: 1
    device: cpu
    output_dir: outputs
data:
    dataset_type: synthetic
    sequence_length: 8
    batch_size: 2
    train_split: 0.8
    synthetic_repeats: 20
model:
    vocab_size: auto
    embedding_dim: 16
    num_heads: 4
    num_layers: 2
    dropout: 0.0
training:
    max_steps: 2
    eval_interval: 1
    log_interval: 1
    learning_rate: 0.001
    weight_decay: 0.01
    grad_clip: 1.0
evaluation:
    max_batches: 1
""".strip(),
                encoding="utf-8",
            )
            output_path = Path(temp_dir) / "structure.html"
            buffer = StringIO()
            with mock.patch.object(sys, "argv", ["prometheus", "visualize-model", "--config", str(config_path), "--output", str(output_path)]):
                with redirect_stdout(buffer):
                    cli.main()

            rendered = output_path.read_text(encoding="utf-8")
            self.assertTrue(output_path.exists())

        self.assertIn("Plotly.newPlot", rendered)
        self.assertIn("Visualization written to", buffer.getvalue())

    def test_autoresearch_dry_run_writes_report(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = Path(temp_dir) / "autoresearch.md"
            buffer = StringIO()
            with mock.patch.object(
                sys,
                "argv",
                [
                    "prometheus",
                    "autoresearch",
                    "--dry-run",
                    "--idea",
                    "dense-control",
                    "--report-path",
                    str(report_path),
                ],
            ):
                with redirect_stdout(buffer):
                    cli.main()

            rendered = report_path.read_text(encoding="utf-8")

        self.assertIn("Prometheus Directed Autoresearch Report", rendered)
        self.assertIn("Dense control", rendered)
        self.assertIn("Autoresearch report written to", buffer.getvalue())

    @unittest.skipUnless(HAS_INFERENCE, "Inference module is not present on this branch.")
    def test_generate_command_prints_text(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = make_config(output_dir=temp_dir)
            run_dir = run_training(config)
            buffer = StringIO()
            with mock.patch.object(
                sys,
                "argv",
                [
                    "prometheus",
                    "generate",
                    "--run-dir",
                    str(run_dir),
                    "--prompt",
                    "the ",
                    "--max-new-tokens",
                    "3",
                    "--top-k",
                    "1",
                    "--device",
                    "cpu",
                ],
            ):
                with redirect_stdout(buffer):
                    cli.main()

        self.assertTrue(buffer.getvalue().strip().startswith("the "))