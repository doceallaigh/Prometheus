from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from prometheus.config import PrometheusConfig, _read_yaml, load_config
from prometheus.data import CharacterTokenizer, LanguageModelingDataset, _load_text, build_datasets, synthetic_corpus


class ConfigAndDataTests(unittest.TestCase):
    def test_read_yaml_returns_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.yaml"
            config_path.write_text("name: prometheus\n", encoding="utf-8")
            parsed = _read_yaml(config_path)

        self.assertEqual(parsed, {"name": "prometheus"})

    def test_read_yaml_rejects_non_mapping_documents(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.yaml"
            config_path.write_text("- one\n- two\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "did not parse to a mapping"):
                _read_yaml(config_path)

    def test_load_config_builds_typed_config(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.yaml"
            config_path.write_text(
                """
experiment:
  run_name: unit-test
  seed: 1
  device: cpu
  output_dir: outputs
data:
  dataset_type: synthetic
  sequence_length: 8
  batch_size: 2
model:
  vocab_size: auto
  embedding_dim: 16
  num_heads: 4
  num_layers: 2
  dropout: 0.1
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
            config = load_config(config_path)

        self.assertIsInstance(config, PrometheusConfig)
        self.assertEqual(config.experiment.run_name, "unit-test")
        self.assertEqual(config.data.sequence_length, 8)
        self.assertEqual(config.to_dict()["model"]["embedding_dim"], 16)

    def test_synthetic_corpus_contains_multiple_repeats(self) -> None:
        corpus = synthetic_corpus(2)
        self.assertGreater(corpus.count("the modular network routes local signals"), 1)

    def test_character_tokenizer_round_trip(self) -> None:
        tokenizer = CharacterTokenizer.build("abc cab")
        encoded = tokenizer.encode("cab")
        decoded = tokenizer.decode(encoded)

        self.assertEqual(decoded, "cab")
        self.assertEqual(tokenizer.vocab_size, len(set("abc cab")))

    def test_language_modeling_dataset_rejects_short_sequences(self) -> None:
        with self.assertRaisesRegex(ValueError, "too small"):
            LanguageModelingDataset(torch.tensor([1, 2, 3]), sequence_length=3)

    def test_language_modeling_dataset_samples_aligned_batches(self) -> None:
        dataset = LanguageModelingDataset(torch.arange(0, 40), sequence_length=5)
        inputs, targets = dataset.sample_batch(batch_size=3, device=torch.device("cpu"))

        self.assertEqual(inputs.shape, (3, 5))
        self.assertEqual(targets.shape, (3, 5))
        self.assertTrue(torch.equal(inputs[:, 1:], targets[:, :-1]))

    def test_load_text_reads_text_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            text_path = Path(temp_dir) / "corpus.txt"
            text_path.write_text("hello world", encoding="utf-8")
            config = load_config_from_inline(
                output_dir=temp_dir,
                dataset_type="text",
                path=str(text_path),
            )

            loaded = _load_text(config.data)

        self.assertEqual(loaded, "hello world")

    def test_build_datasets_splits_tokens(self) -> None:
        config = load_config_from_inline(output_dir=tempfile.gettempdir())
        bundle = build_datasets(config.data)

        self.assertGreater(bundle.train_tokens.numel(), 0)
        self.assertGreater(bundle.val_tokens.numel(), 0)
        self.assertGreater(bundle.tokenizer.vocab_size, 0)


def load_config_from_inline(output_dir: str, dataset_type: str = "synthetic", path: str | None = None) -> PrometheusConfig:
    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = Path(temp_dir) / "config.yaml"
        data_path_line = f"  path: {path}\n" if path else ""
        config_path.write_text(
            f"""
experiment:
  run_name: unit-test
  seed: 1
  device: cpu
  output_dir: {output_dir}
data:
  dataset_type: {dataset_type}
  sequence_length: 8
  batch_size: 2
  train_split: 0.8
{data_path_line}  synthetic_repeats: 20
model:
  vocab_size: auto
  embedding_dim: 16
  num_heads: 4
  num_layers: 2
  dropout: 0.1
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
        return load_config(config_path)