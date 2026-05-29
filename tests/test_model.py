from __future__ import annotations

import unittest

import torch

from prometheus.config import ModelConfig
from prometheus.model import DenseTransformerLM, ModularTransformerLM, StaticRouter, _valid_head_count, build_model


class ModelTests(unittest.TestCase):
    def test_valid_head_count_falls_back_to_divisible_value(self) -> None:
        self.assertEqual(_valid_head_count(10, 6), 5)
        self.assertEqual(_valid_head_count(7, 4), 1)

    def test_dense_model_forward_returns_logits_and_loss(self) -> None:
        config = ModelConfig(vocab_size=20, embedding_dim=16, num_heads=4, num_layers=2, dropout=0.0)
        model = DenseTransformerLM(config, sequence_length=8)
        tokens = torch.randint(0, 20, (2, 8))
        output = model(tokens, tokens)

        self.assertEqual(output.logits.shape, (2, 8, 20))
        self.assertIsNotNone(output.loss)

    def test_language_model_rejects_unresolved_vocab_size(self) -> None:
        config = ModelConfig(vocab_size="auto", embedding_dim=16, num_heads=4, num_layers=2, dropout=0.0)
        with self.assertRaisesRegex(ValueError, "resolved to an integer"):
            DenseTransformerLM(config, sequence_length=8)

    def test_dense_model_rejects_inputs_longer_than_context(self) -> None:
        config = ModelConfig(vocab_size=20, embedding_dim=16, num_heads=4, num_layers=2, dropout=0.0)
        model = DenseTransformerLM(config, sequence_length=4)
        tokens = torch.randint(0, 20, (1, 5))
        with self.assertRaisesRegex(ValueError, "configured model context window"):
            model(tokens)

    def test_router_mask_for_local_topology_matches_neighbors(self) -> None:
        mask = StaticRouter._build_mask(4, "local")
        self.assertTrue(mask[0, 0])
        self.assertTrue(mask[1, 0])
        self.assertFalse(mask[0, 2])

    def test_router_rejects_unknown_topology(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported routing topology"):
            StaticRouter._build_mask(3, "unknown")

    def test_modular_stage_model_forward_matches_expected_shape(self) -> None:
        config = ModelConfig(
            vocab_size=30,
            embedding_dim=24,
            num_heads=6,
            num_layers=0,
            dropout=0.0,
            architecture="modular",
            stage_groups=[3, 1],
            stage_depths=[1, 1],
            routing_topology="small_world",
            routing_top_k=2,
        )
        model = ModularTransformerLM(config, sequence_length=6)
        tokens = torch.randint(0, 30, (2, 6))
        output = model(tokens, tokens)

        self.assertEqual(output.logits.shape, (2, 6, 30))
        self.assertIsNotNone(output.loss)

    def test_modular_model_rejects_mismatched_stage_lengths(self) -> None:
        config = ModelConfig(
            vocab_size=20,
            embedding_dim=16,
            num_heads=4,
            num_layers=0,
            dropout=0.0,
            architecture="modular",
            stage_groups=[2, 1],
            stage_depths=[1],
        )
        with self.assertRaisesRegex(ValueError, "same length"):
            ModularTransformerLM(config, sequence_length=8)

    def test_modular_model_rejects_invalid_group_division(self) -> None:
        config = ModelConfig(
            vocab_size=20,
            embedding_dim=18,
            num_heads=3,
            num_layers=0,
            dropout=0.0,
            architecture="modular",
            stage_groups=[4],
            stage_depths=[1],
        )
        with self.assertRaisesRegex(ValueError, "divisible by each group count"):
            ModularTransformerLM(config, sequence_length=8)

    def test_build_model_dispatches_to_requested_variant(self) -> None:
        dense = build_model(ModelConfig(vocab_size=10, embedding_dim=8, num_heads=2, num_layers=1, dropout=0.0), sequence_length=4)
        modular = build_model(
            ModelConfig(
                vocab_size=10,
                embedding_dim=8,
                num_heads=2,
                num_layers=0,
                dropout=0.0,
                architecture="modular",
                stage_groups=[2, 1],
                stage_depths=[1, 1],
            ),
            sequence_length=4,
        )

        self.assertIsInstance(dense, DenseTransformerLM)
        self.assertIsInstance(modular, ModularTransformerLM)

    def test_build_model_rejects_unknown_architecture(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported architecture"):
            build_model(
                ModelConfig(vocab_size=10, embedding_dim=8, num_heads=2, num_layers=1, dropout=0.0, architecture="other"),
                sequence_length=4,
            )