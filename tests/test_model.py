from __future__ import annotations

import unittest

import torch

from prometheus.config import ModelConfig
from prometheus.model import ClusteredDenseTransformerLM, DenseRingMemoryTransformerLM, DenseTransformerLM, FractalClusteredDenseTransformerLM, ModularStage, ModularTransformerLM, RecurrentLoopTransformerLM, RingFractalClusteredDenseTransformerLM, StaticRouter, _valid_head_count, build_model, resolve_fractal_cluster_levels, resolve_modular_layout, resolve_modular_stage_specs, resolve_ring_fractal_cluster_shape, structural_connectivity_summary


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

    def test_dense_connectivity_summary_reports_expected_fan_in(self) -> None:
        config = ModelConfig(vocab_size=20, embedding_dim=16, num_heads=4, num_layers=1, dropout=0.0)
        model = DenseTransformerLM(config, sequence_length=8)

        summary = structural_connectivity_summary(model)

        self.assertAlmostEqual(summary["average_hidden_fan_in"], 3072 / 144)
        self.assertEqual(summary["max_hidden_fan_in"], 64)

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

    def test_recurrent_loop_forward_returns_logits_and_loss(self) -> None:
        config = ModelConfig(
            vocab_size=20,
            embedding_dim=16,
            num_heads=4,
            num_layers=1,
            dropout=0.0,
            architecture="recurrent_loop",
            recurrent_steps=4,
            recurrent_state_blend=0.5,
        )
        model = RecurrentLoopTransformerLM(config, sequence_length=8)
        tokens = torch.randint(0, 20, (2, 8))
        output = model(tokens, tokens)

        self.assertEqual(output.logits.shape, (2, 8, 20))
        self.assertIsNotNone(output.loss)

    def test_recurrent_loop_rejects_invalid_runtime_settings(self) -> None:
        invalid_steps = ModelConfig(
            vocab_size=20,
            embedding_dim=16,
            num_heads=4,
            num_layers=1,
            dropout=0.0,
            architecture="recurrent_loop",
            recurrent_steps=0,
            recurrent_state_blend=0.5,
        )
        invalid_blend = ModelConfig(
            vocab_size=20,
            embedding_dim=16,
            num_heads=4,
            num_layers=1,
            dropout=0.0,
            architecture="recurrent_loop",
            recurrent_steps=2,
            recurrent_state_blend=1.5,
        )

        with self.assertRaisesRegex(ValueError, "recurrent_steps"):
            RecurrentLoopTransformerLM(invalid_steps, sequence_length=8)
        with self.assertRaisesRegex(ValueError, "recurrent_state_blend"):
            RecurrentLoopTransformerLM(invalid_blend, sequence_length=8)

    def test_dense_ring_memory_forward_returns_logits_and_loss(self) -> None:
        config = ModelConfig(
            vocab_size=20,
            embedding_dim=16,
            num_heads=4,
            num_layers=2,
            dropout=0.0,
            architecture="dense_ring_memory",
            cluster_copies=2,
            cluster_bridge_percent=25.0,
            cluster_base_embedding_dim=8,
            cluster_levels=2,
            cluster_top_count=2,
            memory_fusion_blend=0.35,
            memory_update_interval=1,
        )
        model = DenseRingMemoryTransformerLM(config, sequence_length=8)
        tokens = torch.randint(0, 20, (2, 8))
        output = model(tokens, tokens)

        self.assertEqual(output.logits.shape, (2, 8, 20))
        self.assertIsNotNone(output.loss)

    def test_dense_ring_memory_rejects_invalid_memory_settings(self) -> None:
        invalid_blend = ModelConfig(
            vocab_size=20,
            embedding_dim=16,
            num_heads=4,
            num_layers=2,
            dropout=0.0,
            architecture="dense_ring_memory",
            cluster_copies=2,
            cluster_bridge_percent=25.0,
            cluster_base_embedding_dim=8,
            cluster_levels=2,
            cluster_top_count=2,
            memory_fusion_blend=1.5,
            memory_update_interval=1,
        )
        invalid_interval = ModelConfig(
            vocab_size=20,
            embedding_dim=16,
            num_heads=4,
            num_layers=2,
            dropout=0.0,
            architecture="dense_ring_memory",
            cluster_copies=2,
            cluster_bridge_percent=25.0,
            cluster_base_embedding_dim=8,
            cluster_levels=2,
            cluster_top_count=2,
            memory_fusion_blend=0.35,
            memory_update_interval=0,
        )

        with self.assertRaisesRegex(ValueError, "memory_fusion_blend"):
            DenseRingMemoryTransformerLM(invalid_blend, sequence_length=8)
        with self.assertRaisesRegex(ValueError, "memory_update_interval"):
            DenseRingMemoryTransformerLM(invalid_interval, sequence_length=8)

    def test_clustered_dense_model_forward_returns_logits_and_loss(self) -> None:
        config = ModelConfig(
            vocab_size=20,
            embedding_dim=16,
            num_heads=4,
            num_layers=2,
            dropout=0.0,
            architecture="clustered_dense",
            cluster_copies=3,
            cluster_bridge_percent=25.0,
        )
        model = ClusteredDenseTransformerLM(config, sequence_length=8)
        tokens = torch.randint(0, 20, (2, 8))
        output = model(tokens, tokens)

        self.assertEqual(model.bridge_width, 4)
        self.assertEqual(output.logits.shape, (2, 8, 20))
        self.assertIsNotNone(output.loss)

    def test_clustered_dense_requires_at_least_two_copies(self) -> None:
        config = ModelConfig(
            vocab_size=20,
            embedding_dim=16,
            num_heads=4,
            num_layers=1,
            dropout=0.0,
            architecture="clustered_dense",
            cluster_copies=1,
            cluster_bridge_percent=10.0,
        )

        with self.assertRaisesRegex(ValueError, "cluster_copies"):
            ClusteredDenseTransformerLM(config, sequence_length=8)

    def test_clustered_dense_requires_valid_bridge_percent(self) -> None:
        config = ModelConfig(
            vocab_size=20,
            embedding_dim=16,
            num_heads=4,
            num_layers=1,
            dropout=0.0,
            architecture="clustered_dense",
            cluster_copies=2,
            cluster_bridge_percent=0.0,
        )

        with self.assertRaisesRegex(ValueError, "cluster_bridge_percent"):
            ClusteredDenseTransformerLM(config, sequence_length=8)

    def test_fractal_clustered_dense_forward_returns_logits_and_loss(self) -> None:
        config = ModelConfig(
            vocab_size=20,
            embedding_dim=16,
            num_heads=4,
            num_layers=1,
            dropout=0.0,
            architecture="fractal_clustered_dense",
            cluster_copies=2,
            cluster_bridge_percent=25.0,
            cluster_base_embedding_dim=8,
            cluster_levels=3,
        )
        model = FractalClusteredDenseTransformerLM(config, sequence_length=8)
        tokens = torch.randint(0, 20, (2, 8))
        output = model(tokens, tokens)

        self.assertEqual(model.cluster_levels, 3)
        self.assertEqual(output.logits.shape, (2, 8, 20))
        self.assertIsNotNone(output.loss)

    def test_resolve_fractal_cluster_levels_matches_requested_budget(self) -> None:
        explicit = ModelConfig(
            vocab_size=20,
            embedding_dim=16,
            num_heads=4,
            num_layers=1,
            dropout=0.0,
            architecture="fractal_clustered_dense",
            cluster_copies=2,
            cluster_bridge_percent=25.0,
            cluster_base_embedding_dim=8,
            cluster_levels=2,
        )
        target_parameter_count = sum(parameter.numel() for parameter in FractalClusteredDenseTransformerLM(explicit, sequence_length=8).parameters())
        budgeted = ModelConfig(
            vocab_size=20,
            embedding_dim=16,
            num_heads=4,
            num_layers=1,
            dropout=0.0,
            architecture="fractal_clustered_dense",
            cluster_copies=2,
            cluster_bridge_percent=25.0,
            cluster_base_embedding_dim=8,
            cluster_target_parameter_count=target_parameter_count,
            cluster_max_levels=4,
        )

        self.assertEqual(resolve_fractal_cluster_levels(budgeted, sequence_length=8), 2)

    def test_fractal_clustered_dense_requires_base_embedding_dim(self) -> None:
        config = ModelConfig(
            vocab_size=20,
            embedding_dim=16,
            num_heads=4,
            num_layers=1,
            dropout=0.0,
            architecture="fractal_clustered_dense",
            cluster_copies=2,
            cluster_bridge_percent=25.0,
            cluster_levels=2,
        )

        with self.assertRaisesRegex(ValueError, "cluster_base_embedding_dim"):
            FractalClusteredDenseTransformerLM(config, sequence_length=8)

    def test_ring_fractal_clustered_dense_forward_returns_logits_and_loss(self) -> None:
        config = ModelConfig(
            vocab_size=20,
            embedding_dim=16,
            num_heads=4,
            num_layers=1,
            dropout=0.0,
            architecture="ring_fractal_clustered_dense",
            cluster_copies=3,
            cluster_bridge_percent=25.0,
            cluster_base_embedding_dim=8,
            cluster_levels=2,
            cluster_top_count=2,
        )
        model = RingFractalClusteredDenseTransformerLM(config, sequence_length=8)
        tokens = torch.randint(0, 20, (2, 8))
        output = model(tokens, tokens)

        self.assertEqual(model.cluster_levels, 2)
        self.assertEqual(model.cluster_top_count, 2)
        self.assertEqual(output.logits.shape, (2, 8, 20))
        self.assertIsNotNone(output.loss)

    def test_resolve_ring_fractal_cluster_shape_matches_requested_budget(self) -> None:
        explicit = ModelConfig(
            vocab_size=20,
            embedding_dim=16,
            num_heads=4,
            num_layers=1,
            dropout=0.0,
            architecture="ring_fractal_clustered_dense",
            cluster_copies=3,
            cluster_bridge_percent=25.0,
            cluster_base_embedding_dim=8,
            cluster_levels=2,
            cluster_top_count=2,
        )
        target_parameter_count = sum(parameter.numel() for parameter in RingFractalClusteredDenseTransformerLM(explicit, sequence_length=8).parameters())
        budgeted = ModelConfig(
            vocab_size=20,
            embedding_dim=16,
            num_heads=4,
            num_layers=1,
            dropout=0.0,
            architecture="ring_fractal_clustered_dense",
            cluster_copies=3,
            cluster_bridge_percent=25.0,
            cluster_base_embedding_dim=8,
            cluster_target_parameter_count=target_parameter_count,
            cluster_max_levels=4,
        )

        self.assertEqual(resolve_ring_fractal_cluster_shape(budgeted, sequence_length=8), (2, 2))

    def test_ring_fractal_clustered_dense_requires_valid_top_count(self) -> None:
        config = ModelConfig(
            vocab_size=20,
            embedding_dim=16,
            num_heads=4,
            num_layers=1,
            dropout=0.0,
            architecture="ring_fractal_clustered_dense",
            cluster_copies=3,
            cluster_bridge_percent=25.0,
            cluster_base_embedding_dim=8,
            cluster_levels=2,
            cluster_top_count=4,
        )

        with self.assertRaisesRegex(ValueError, "cluster_top_count"):
            resolve_ring_fractal_cluster_shape(config, sequence_length=8)

    def test_dense_inflection_pruning_reduces_active_hidden_units(self) -> None:
        config = ModelConfig(vocab_size=20, embedding_dim=16, num_heads=4, num_layers=2, dropout=0.0, inflection_pruning_keep_ratio=0.5)
        model = DenseTransformerLM(config, sequence_length=8)

        summary = model.apply_inflection_pruning()

        self.assertIsNotNone(summary)
        self.assertEqual(summary["pruned_blocks"], 2)
        self.assertLess(summary["active_hidden_units"], summary["previous_hidden_units"])

    def test_router_mask_for_local_topology_matches_neighbors(self) -> None:
        mask = StaticRouter._build_mask(4, "local")
        self.assertTrue(mask[0, 0])
        self.assertTrue(mask[1, 0])
        self.assertFalse(mask[0, 2])

    def test_router_mask_for_cluster_graph_preserves_local_clusters_and_bridges(self) -> None:
        mask = StaticRouter._build_mask(8, "cluster_graph")

        self.assertTrue(mask[0, 0])
        self.assertTrue(mask[0, 1])
        self.assertTrue(mask[0, 2])
        self.assertTrue(mask[0, 6])
        self.assertFalse(mask[0, 4])

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

    def test_modular_inflection_pruning_sets_runtime_top_k(self) -> None:
        config = ModelConfig(
            vocab_size=30,
            embedding_dim=24,
            num_heads=6,
            num_layers=0,
            dropout=0.0,
            architecture="modular",
            stage_groups=[3, 1],
            stage_depths=[1, 1],
            routing_topology="cluster_graph",
            inflection_pruning_top_k=1,
        )
        model = ModularTransformerLM(config, sequence_length=6)

        summary = model.apply_inflection_pruning()

        self.assertIsNotNone(summary)
        self.assertEqual(summary["active_top_k"], 1)
        self.assertTrue(all(stage.router.runtime_top_k == 1 for stage in model.stages))

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

    def test_resolve_modular_stage_specs_supports_fixed_group_size(self) -> None:
        specs = resolve_modular_stage_specs(
            ModelConfig(
                vocab_size=20,
                embedding_dim=24,
                num_heads=6,
                num_layers=0,
                dropout=0.0,
                architecture="modular",
                stage_groups=[2, 4, 1],
                stage_depths=[1, 1, 1],
                fixed_group_size=6,
            )
        )

        self.assertEqual([spec.group_count for spec in specs], [2, 4, 1])
        self.assertEqual([spec.group_dim for spec in specs], [6, 6, 6])
        self.assertEqual([spec.stage_dim for spec in specs], [12, 24, 6])

    def test_modular_model_supports_fixed_group_size_with_variable_stage_widths(self) -> None:
        config = ModelConfig(
            vocab_size=30,
            embedding_dim=24,
            num_heads=6,
            num_layers=0,
            dropout=0.0,
            architecture="modular",
            stage_groups=[2, 4, 1],
            stage_depths=[1, 1, 1],
            fixed_group_size=6,
            routing_topology="small_world",
            routing_top_k=2,
        )
        model = ModularTransformerLM(config, sequence_length=6)
        tokens = torch.randint(0, 30, (2, 6))
        output = model(tokens, tokens)

        self.assertEqual(output.logits.shape, (2, 6, 30))
        self.assertIsNotNone(output.loss)
        self.assertEqual(model.stages[0].stage_dim, 12)
        self.assertEqual(model.stages[1].stage_dim, 24)
        self.assertEqual(model.stages[2].stage_dim, 6)

    def test_modular_model_rejects_nonpositive_fixed_group_size(self) -> None:
        config = ModelConfig(
            vocab_size=20,
            embedding_dim=16,
            num_heads=4,
            num_layers=0,
            dropout=0.0,
            architecture="modular",
            stage_groups=[2, 1],
            stage_depths=[1, 1],
            fixed_group_size=0,
        )

        with self.assertRaisesRegex(ValueError, "fixed_group_size must be a positive integer"):
            ModularTransformerLM(config, sequence_length=8)

    def test_cortical_columns_layout_normalizes_column_fields(self) -> None:
        layout = resolve_modular_layout(
            ModelConfig(
                vocab_size=20,
                embedding_dim=24,
                num_heads=6,
                num_layers=0,
                dropout=0.0,
                architecture="cortical_columns",
                column_counts=[2, 3],
                column_depths=[2, 1],
                fixed_column_size=8,
                column_recombination="summary_router",
                column_routing_topology="small_world",
                column_routing_top_k=2,
            ),
            sequence_length=8,
        )

        self.assertEqual(layout.group_schedule, [2, 3])
        self.assertEqual(layout.depth_schedule, [2, 1])
        self.assertEqual(layout.fixed_group_size, 8)
        self.assertEqual(layout.routing_topology, "small_world")
        self.assertEqual(layout.routing_top_k, 2)
        self.assertEqual(layout.recombination_mode, "summary_router")
        self.assertEqual(layout.unit_label, "Cortical column")

    def test_cortical_columns_layout_supports_binary_tree_recombination(self) -> None:
        layout = resolve_modular_layout(
            ModelConfig(
                vocab_size=20,
                embedding_dim=24,
                num_heads=6,
                num_layers=0,
                dropout=0.0,
                architecture="cortical_columns",
                column_counts=[2, 4],
                column_depths=[2, 1],
                column_recombination="binary_tree",
                column_routing_topology="small_world",
                column_routing_top_k=2,
            ),
            sequence_length=8,
        )

        self.assertEqual(layout.recombination_mode, "binary_tree")
        self.assertEqual(layout.routing_topology, "dense")
        self.assertIsNone(layout.routing_top_k)

    def test_cortical_columns_layout_generates_fractal_schedule_to_budget(self) -> None:
        layout = resolve_modular_layout(
            ModelConfig(
                vocab_size=27,
                embedding_dim=64,
                num_heads=4,
                num_layers=0,
                dropout=0.0,
                architecture="cortical_columns",
                column_input_count=2,
                column_branching_factor=4,
                target_parameter_count=120000,
                max_column_stages=4,
                fixed_column_size=8,
                column_depths=[1],
            ),
            sequence_length=64,
        )

        self.assertEqual(layout.group_schedule, [2, 8, 32])
        self.assertEqual(layout.depth_schedule, [1, 1, 1])

    def test_fractal_cortical_columns_require_fixed_column_size(self) -> None:
        config = ModelConfig(
            vocab_size=27,
            embedding_dim=64,
            num_heads=4,
            num_layers=0,
            dropout=0.0,
            architecture="cortical_columns",
            column_input_count=2,
            column_branching_factor=4,
            target_parameter_count=120000,
        )

        with self.assertRaisesRegex(ValueError, "fixed_column_size"):
            resolve_modular_layout(config, sequence_length=64)

    def test_cortical_columns_forward_matches_expected_shape(self) -> None:
        config = ModelConfig(
            vocab_size=30,
            embedding_dim=24,
            num_heads=6,
            num_layers=0,
            dropout=0.0,
            architecture="cortical_columns",
            column_counts=[2, 3],
            column_depths=[1, 1],
            column_routing_topology="small_world",
            column_routing_top_k=2,
        )
        model = ModularTransformerLM(config, sequence_length=6)
        tokens = torch.randint(0, 30, (2, 6))
        output = model(tokens, tokens)

        self.assertEqual(output.logits.shape, (2, 6, 30))
        self.assertIsNotNone(output.loss)

    def test_binary_tree_group_context_combines_pairs_then_parents(self) -> None:
        summaries = torch.tensor(
            [
                [
                    [1.0, 10.0],
                    [3.0, 30.0],
                    [5.0, 50.0],
                    [7.0, 70.0],
                ]
            ]
        )

        context = ModularStage._binary_tree_group_context(summaries)

        expected = torch.tensor(
            [
                [
                    [6.0, 60.0],
                    [6.0, 60.0],
                    [10.0, 100.0],
                    [10.0, 100.0],
                ]
            ]
        )
        self.assertTrue(torch.equal(context, expected))

    def test_cortical_columns_forward_supports_binary_tree_recombination(self) -> None:
        config = ModelConfig(
            vocab_size=30,
            embedding_dim=24,
            num_heads=6,
            num_layers=0,
            dropout=0.0,
            architecture="cortical_columns",
            column_counts=[2, 4],
            column_depths=[1, 1],
            column_recombination="binary_tree",
        )
        model = ModularTransformerLM(config, sequence_length=6)
        tokens = torch.randint(0, 30, (2, 6))
        output = model(tokens, tokens)

        self.assertEqual(output.logits.shape, (2, 6, 30))
        self.assertIsNotNone(output.loss)

    def test_cortical_columns_rejects_stage_fields(self) -> None:
        config = ModelConfig(
            vocab_size=20,
            embedding_dim=16,
            num_heads=4,
            num_layers=0,
            dropout=0.0,
            architecture="cortical_columns",
            stage_groups=[2, 1],
            column_counts=[2, 1],
        )

        with self.assertRaisesRegex(ValueError, r"Use column_\* fields"):
            ModularTransformerLM(config, sequence_length=8)

    def test_modular_connectivity_summary_reflects_grouped_blocks(self) -> None:
        dense = DenseTransformerLM(ModelConfig(vocab_size=20, embedding_dim=16, num_heads=4, num_layers=1, dropout=0.0), sequence_length=8)
        modular = ModularTransformerLM(
            ModelConfig(
                vocab_size=20,
                embedding_dim=16,
                num_heads=4,
                num_layers=0,
                dropout=0.0,
                architecture="modular",
                stage_groups=[4],
                stage_depths=[1],
                routing_topology="dense",
            ),
            sequence_length=8,
        )

        dense_summary = structural_connectivity_summary(dense)
        modular_summary = structural_connectivity_summary(modular)

        self.assertGreater(modular_summary["average_hidden_fan_in"], 0)
        self.assertLess(modular_summary["average_hidden_fan_in"], dense_summary["average_hidden_fan_in"])

    def test_build_model_dispatches_to_requested_variant(self) -> None:
        dense = build_model(ModelConfig(vocab_size=10, embedding_dim=8, num_heads=2, num_layers=1, dropout=0.0), sequence_length=4)
        dense_ring_memory = build_model(
            ModelConfig(
                vocab_size=10,
                embedding_dim=8,
                num_heads=2,
                num_layers=1,
                dropout=0.0,
                architecture="dense_ring_memory",
                cluster_copies=2,
                cluster_bridge_percent=25.0,
                cluster_base_embedding_dim=4,
                cluster_levels=2,
                cluster_top_count=2,
                memory_fusion_blend=0.35,
                memory_update_interval=1,
            ),
            sequence_length=4,
        )
        recurrent = build_model(
            ModelConfig(
                vocab_size=10,
                embedding_dim=8,
                num_heads=2,
                num_layers=1,
                dropout=0.0,
                architecture="recurrent_loop",
                recurrent_steps=3,
                recurrent_state_blend=0.5,
            ),
            sequence_length=4,
        )
        clustered = build_model(
            ModelConfig(
                vocab_size=10,
                embedding_dim=8,
                num_heads=2,
                num_layers=1,
                dropout=0.0,
                architecture="clustered_dense",
                cluster_copies=2,
                cluster_bridge_percent=25.0,
            ),
            sequence_length=4,
        )
        fractal_clustered = build_model(
            ModelConfig(
                vocab_size=10,
                embedding_dim=8,
                num_heads=2,
                num_layers=1,
                dropout=0.0,
                architecture="fractal_clustered_dense",
                cluster_copies=2,
                cluster_bridge_percent=25.0,
                cluster_base_embedding_dim=4,
                cluster_levels=2,
            ),
            sequence_length=4,
        )
        ring_fractal_clustered = build_model(
            ModelConfig(
                vocab_size=10,
                embedding_dim=8,
                num_heads=2,
                num_layers=1,
                dropout=0.0,
                architecture="ring_fractal_clustered_dense",
                cluster_copies=2,
                cluster_bridge_percent=25.0,
                cluster_base_embedding_dim=4,
                cluster_levels=2,
                cluster_top_count=2,
            ),
            sequence_length=4,
        )
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
        cortical = build_model(
            ModelConfig(
                vocab_size=10,
                embedding_dim=8,
                num_heads=2,
                num_layers=0,
                dropout=0.0,
                architecture="cortical_columns",
                column_counts=[2, 1],
                column_depths=[1, 1],
            ),
            sequence_length=4,
        )

        self.assertIsInstance(dense, DenseTransformerLM)
        self.assertIsInstance(dense_ring_memory, DenseRingMemoryTransformerLM)
        self.assertIsInstance(recurrent, RecurrentLoopTransformerLM)
        self.assertIsInstance(clustered, ClusteredDenseTransformerLM)
        self.assertIsInstance(fractal_clustered, FractalClusteredDenseTransformerLM)
        self.assertIsInstance(ring_fractal_clustered, RingFractalClusteredDenseTransformerLM)
        self.assertIsInstance(modular, ModularTransformerLM)
        self.assertIsInstance(cortical, ModularTransformerLM)

    def test_build_model_rejects_unknown_architecture(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported architecture"):
            build_model(
                ModelConfig(vocab_size=10, embedding_dim=8, num_heads=2, num_layers=1, dropout=0.0, architecture="other"),
                sequence_length=4,
            )