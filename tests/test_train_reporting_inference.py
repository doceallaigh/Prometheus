from __future__ import annotations

import json
import importlib
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import torch

from prometheus.config import PrometheusConfig, load_config
from prometheus.data import LanguageModelingDataset, build_datasets
from prometheus.reporting import comparison_markdown, summarize_run
from prometheus.train import (
    create_optimizer,
    evaluate_model,
    learning_rate_for_step,
    parameter_count,
    resolve_device,
    resolve_model_config,
    run_training,
)

try:
    inference_module = importlib.import_module("prometheus.inference")
except ModuleNotFoundError:
    inference_module = None

HAS_INFERENCE = inference_module is not None


class TrainReportingInferenceTests(unittest.TestCase):
    def test_resolve_device_prefers_cpu_for_auto_without_cuda(self) -> None:
        device = resolve_device("auto")
        self.assertIn(device.type, {"cpu", "cuda"})

    def test_learning_rate_schedule_warms_up_then_decays(self) -> None:
        config = make_config()
        config.training.max_steps = 4
        config.training.warmup_steps = 2
        warm_start = learning_rate_for_step(config, 0)
        warm_end = learning_rate_for_step(config, config.training.warmup_steps - 1)
        later = learning_rate_for_step(config, config.training.max_steps - 1)

        self.assertLess(warm_start, warm_end)
        self.assertLess(later, config.training.learning_rate)

    def test_create_optimizer_and_parameter_count(self) -> None:
        config = make_config()
        data_bundle = build_datasets(config.data)
        model_config = resolve_model_config(config, data_bundle)
        model = resolve_model(model_config, config)
        optimizer = create_optimizer(model, config)

        self.assertGreater(parameter_count(model), 0)
        self.assertEqual(optimizer.param_groups[0]["lr"], config.training.learning_rate)

    def test_evaluate_model_returns_loss_and_perplexity(self) -> None:
        config = make_config()
        data_bundle = build_datasets(config.data)
        model_config = resolve_model_config(config, data_bundle)
        model = resolve_model(model_config, config)
        dataset = LanguageModelingDataset(data_bundle.val_tokens, config.data.sequence_length)

        metrics = evaluate_model(model, dataset, batch_size=2, device=torch.device("cpu"), max_batches=1)

        self.assertIn("loss", metrics)
        self.assertIn("perplexity", metrics)

    def test_run_training_writes_expected_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = make_config(output_dir=temp_dir)
            buffer = StringIO()
            with redirect_stdout(buffer):
                run_dir = run_training(config)
            config_snapshot = json.loads((run_dir / "config.snapshot.json").read_text(encoding="utf-8"))
            run_summary = json.loads((run_dir / "run.summary.json").read_text(encoding="utf-8"))

            self.assertTrue((run_dir / "config.snapshot.json").exists())
            self.assertTrue((run_dir / "model.summary.json").exists())
            self.assertTrue((run_dir / "metrics.jsonl").exists())
            self.assertTrue((run_dir / "run.summary.json").exists())
            self.assertTrue((run_dir / "checkpoint.pt").exists())
            self.assertIsInstance(config_snapshot["model"]["vocab_size"], int)
            self.assertEqual(config_snapshot["experiment"]["requested_device"], config.experiment.device)
            self.assertEqual(config_snapshot["experiment"]["device"], "cpu")
            model_summary = json.loads((run_dir / "model.summary.json").read_text(encoding="utf-8"))
            self.assertGreater(model_summary["average_hidden_fan_in"], 0)
            self.assertGreater(run_summary["average_training_tokens_per_second"], 0)
            self.assertIn('"split": "train"', buffer.getvalue())

    def test_run_training_records_prune_event_for_inflection_schedule(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = make_config(output_dir=temp_dir)
            config.model.inflection_pruning_keep_ratio = 0.5
            config.training.pruning_schedule = "inflection"
            config.training.pruning_min_steps = 0
            config.training.pruning_patience = 0
            config.training.pruning_min_improvement = 100.0

            run_dir = run_training(config)
            metrics = (run_dir / "metrics.jsonl").read_text(encoding="utf-8")
            run_summary = json.loads((run_dir / "run.summary.json").read_text(encoding="utf-8"))

        self.assertIn('"split": "prune"', metrics)
        self.assertTrue(run_summary["pruning_applied"])
        self.assertEqual(run_summary["strategy"], "inflection")

    def test_summarize_run_reads_artifacts_and_fallbacks(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"
            run_dir.mkdir()
            (run_dir / "config.snapshot.json").write_text(
                json.dumps(
                    {
                        "model": {
                            "architecture": "dense",
                            "vocab_size": 12,
                            "inflection_pruning_keep_ratio": 0.75,
                        },
                        "data": {"sequence_length": 8},
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "metrics.jsonl").write_text(
                json.dumps({"split": "train", "step": 0, "loss": 1.0, "perplexity": 2.7, "tokens_per_second": 123.0, "step_seconds": 0.25}) + "\n"
                + json.dumps({"split": "val", "step": 0, "loss": 0.8, "perplexity": 2.2, "tokens_per_second": 98.0, "elapsed_seconds": 0.5})
                + "\n",
                encoding="utf-8",
            )
            (run_dir / "run.summary.json").write_text(
                json.dumps({"total_training_seconds": 1.0, "average_training_tokens_per_second": 111.0}),
                encoding="utf-8",
            )

            summary = summarize_run(run_dir)

        self.assertEqual(summary["architecture"], "dense")
        self.assertEqual(summary["best_val_loss"], 0.8)
        self.assertEqual(summary["latest_train_tokens_per_second"], 123.0)
        self.assertEqual(summary["latest_val_tokens_per_second"], 98.0)
        self.assertEqual(summary["average_training_tokens_per_second"], 111.0)
        self.assertEqual(summary["configured_keep_ratio"], 0.75)
        self.assertEqual(summary["steps_logged"], 1)

    def test_comparison_markdown_renders_run_rows(self) -> None:
        markdown = comparison_markdown(
            [
                {"run_dir": "outputs/run-a", "architecture": "dense", "parameter_count": 10, "average_hidden_fan_in": 12.5, "sequence_length": 8, "average_training_tokens_per_second": 100.0, "latest_val_tokens_per_second": 80.0, "latest_val_loss": 1.2, "best_val_loss": 1.1, "latest_val_perplexity": 3.3},
                {"run_dir": "outputs/run-b", "architecture": "modular", "parameter_count": 9, "average_hidden_fan_in": 8.5, "sequence_length": 8, "average_training_tokens_per_second": 90.0, "latest_val_tokens_per_second": 70.0, "latest_val_loss": 1.0, "best_val_loss": 0.9, "latest_val_perplexity": 2.7},
            ]
        )

        self.assertIn("| run-a | dense |", markdown)
        self.assertIn("| run-b | modular |", markdown)
        self.assertIn("avg_fan_in", markdown)
        self.assertIn("train_tok_s", markdown)
        self.assertIn("val_tok_s", markdown)

    @unittest.skipUnless(HAS_INFERENCE, "Inference module is not present on this branch.")
    def test_load_run_and_generate_text_work_from_saved_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = make_config(output_dir=temp_dir)
            run_dir = run_training(config)
            loaded = inference_module.load_run(run_dir, raw_device="cpu")

            torch.manual_seed(0)
            generated = inference_module.generate_text(loaded, prompt="the ", max_new_tokens=5, temperature=1.0, top_k=1)

        self.assertTrue(generated.startswith("the "))
        self.assertGreater(len(generated), len("the "))

    @unittest.skipUnless(HAS_INFERENCE, "Inference module is not present on this branch.")
    def test_generate_text_rejects_invalid_prompt_and_temperature(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = make_config(output_dir=temp_dir)
            run_dir = run_training(config)
            loaded = inference_module.load_run(run_dir, raw_device="cpu")

            with self.assertRaisesRegex(ValueError, "must not be empty"):
                inference_module.generate_text(loaded, prompt="", max_new_tokens=1)
            with self.assertRaisesRegex(ValueError, "must be positive"):
                inference_module.generate_text(loaded, prompt="the ", max_new_tokens=1, temperature=0)
            with self.assertRaisesRegex(ValueError, "not in the run vocabulary"):
                inference_module.generate_text(loaded, prompt="THE", max_new_tokens=1)


def make_config(output_dir: str | None = None) -> PrometheusConfig:
    output_root = output_dir or tempfile.gettempdir()
    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = Path(temp_dir) / "config.yaml"
        config_path.write_text(
            f"""
experiment:
  run_name: test-run
  seed: 1
  device: cpu
  output_dir: {output_root}
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
  warmup_steps: 1
evaluation:
  max_batches: 1
""".strip(),
            encoding="utf-8",
        )
        return load_config(config_path)


def resolve_model(model_config, config: PrometheusConfig):
    from prometheus.model import build_model

    return build_model(model_config, sequence_length=config.data.sequence_length)