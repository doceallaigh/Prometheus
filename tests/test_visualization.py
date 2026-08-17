from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from prometheus.config import load_config
from prometheus.visualization import build_structure_payload, write_structure_html


class VisualizationTests(unittest.TestCase):
    def test_build_structure_payload_for_dense_model_chains_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = _write_inline_config(
                temp_dir,
                """
experiment:
  run_name: dense-visual
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
""",
            )

            payload = build_structure_payload(load_config(config))

        dense_nodes = [node for node in payload["nodes"] if node["kind"] == "dense"]
        self.assertEqual(len(dense_nodes), 2)
        self.assertEqual(sum(1 for edge in payload["edges"] if edge["kind"] == "flow"), 3)
        self.assertEqual(len(payload["stage_profiles"]), 2)
        self.assertEqual(payload["routing_matrices"], [])

    def test_build_structure_payload_for_cluster_graph_contains_route_edges(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = _write_inline_config(
                temp_dir,
                """
experiment:
  run_name: modular-visual
  seed: 1
  device: cpu
  output_dir: outputs
data:
  dataset_type: synthetic
  sequence_length: 8
  batch_size: 2
model:
  architecture: modular
  vocab_size: auto
  embedding_dim: 16
  num_heads: 4
  num_layers: 0
  dropout: 0.0
  stage_groups: [4, 1]
  stage_depths: [1, 1]
  routing_topology: cluster_graph
  routing_top_k: 1
training:
  max_steps: 2
  eval_interval: 1
  log_interval: 1
  learning_rate: 0.001
  weight_decay: 0.01
  grad_clip: 1.0
evaluation:
  max_batches: 1
""",
            )

            payload = build_structure_payload(load_config(config))

        route_edges = [edge for edge in payload["edges"] if edge["kind"] == "route"]
        self.assertGreater(len(route_edges), 0)
        self.assertTrue(any(edge["source"] == "stage-0-group-2" and edge["target"] == "stage-0-group-0" for edge in route_edges))
        self.assertEqual(len(payload["routing_matrices"]), 2)
        self.assertEqual(payload["routing_matrices"][0]["values"][0][2], 1)

    def test_build_structure_payload_respects_fixed_group_size(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = _write_inline_config(
                temp_dir,
                """
experiment:
  run_name: fixed-group-size-visual
  seed: 1
  device: cpu
  output_dir: outputs
data:
  dataset_type: synthetic
  sequence_length: 8
  batch_size: 2
model:
  architecture: modular
  vocab_size: auto
  embedding_dim: 24
  num_heads: 4
  num_layers: 0
  dropout: 0.0
  stage_groups: [2, 4, 1]
  stage_depths: [1, 1, 1]
  fixed_group_size: 6
  routing_topology: small_world
training:
  max_steps: 2
  eval_interval: 1
  log_interval: 1
  learning_rate: 0.001
  weight_decay: 0.01
  grad_clip: 1.0
evaluation:
  max_batches: 1
""",
            )

            payload = build_structure_payload(load_config(config))

        self.assertEqual([stage["group_dim"] for stage in payload["stage_profiles"]], [6, 6, 6])
        self.assertEqual([stage["stage_dim"] for stage in payload["stage_profiles"]], [12, 24, 6])

    def test_write_structure_html_writes_expected_content(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = _write_inline_config(
                temp_dir,
                """
experiment:
  run_name: html-visual
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
  num_layers: 1
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
""",
            )
            output_path = Path(temp_dir) / "viz" / "model.html"

            result = write_structure_html(load_config(config), output_path)

            html = result.read_text(encoding="utf-8")

        self.assertEqual(result, output_path)
        self.assertIn("Plotly.react", html)
        self.assertIn("html-visual", html)
        self.assertIn("Routing Matrix", html)
        self.assertIn("Stage Profile", html)
        self.assertIn("Preview Controls", html)
        self.assertIn("Apply Preview", html)


def _write_inline_config(temp_dir: str, yaml_text: str) -> Path:
    config_path = Path(temp_dir) / "config.yaml"
    config_path.write_text(yaml_text.strip(), encoding="utf-8")
    return config_path