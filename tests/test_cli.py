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