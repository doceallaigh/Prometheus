from __future__ import annotations

import math
import unittest

import torch

from prometheus.retrofit import _pearson, _rank_values, _spearman
from prometheus.retrofit_baselines import _sidecar_high_training_gate_mask
from prometheus.ontogeny_content import _aggregate_trace_rows, _summarize_counts
from prometheus.ontogeny_experiments import _intervene_training_hidden, _token_training_gate_mask, _training_arm_basis, _transform_generation_row, _transform_hidden


class OntogenyHelperTests(unittest.TestCase):
    def test_token_training_gates_align_with_next_token(self) -> None:
        class Tokenizer:
            pieces = {1: "prompt", 2: " 12", 3: " +", 4: " words"}

            def decode(self, token_ids) -> str:
                return self.pieces[token_ids[0]]

        input_ids = [1, 1, 2, 3, 4]
        digit = _token_training_gate_mask(input_ids, 2, Tokenizer(), "digit")
        operator = _token_training_gate_mask(input_ids, 2, Tokenizer(), "operator")
        combined = _token_training_gate_mask(input_ids, 2, Tokenizer(), "digit-or-operator")
        self.assertEqual(digit.tolist(), [False, True, False, False, False])
        self.assertEqual(operator.tolist(), [False, False, True, False, False])
        self.assertEqual(combined.tolist(), [False, True, True, False, False])

    def test_sidecar_gate_uses_completion_excursions_only(self) -> None:
        class Corrector:
            def __call__(self, hidden: torch.Tensor) -> torch.Tensor:
                scores = torch.tensor([[100.0, 1.0, 1.0, 10.0, 1.0, 100.0]])
                return scores.unsqueeze(-1).expand_as(hidden)

        hidden = torch.zeros(1, 6, 1)
        mask = _sidecar_high_training_gate_mask(hidden, Corrector(), prompt_len=2, threshold_z=1.0)
        self.assertEqual(mask.tolist(), [[False, False, False, True, False, False]])

    def test_average_tie_ranks(self) -> None:
        self.assertEqual(_rank_values([10.0, 20.0, 20.0, 30.0]), [1.0, 2.5, 2.5, 4.0])

    def test_correlations(self) -> None:
        self.assertAlmostEqual(_pearson([1.0, 2.0, 3.0], [2.0, 4.0, 6.0]), 1.0)
        self.assertAlmostEqual(_spearman([1.0, 3.0, 2.0], [10.0, 30.0, 20.0]), 1.0)
        self.assertTrue(math.isnan(_pearson([1.0, 1.0], [2.0, 3.0])))

    def test_content_count_summary(self) -> None:
        counts = {
            "tokens": 10, "gold_digits": 4, "full_correct": 6,
            "complement_correct": 5, "noise_correct": 1, "full_wrong": 4,
            "complement_recovers": 2, "noise_recovers": 0,
            "complement_disagrees": 5, "noise_disagrees": 9,
            "complement_digit_at_gold_digit": 3, "noise_digit_at_gold_digit": 1,
            "complement_contending_digit": 2, "noise_contending_digit": 1,
            "complement_contextual_digit": 2, "noise_contextual_digit": 0,
        }
        summary = _summarize_counts(counts)
        self.assertEqual(summary["full_token_accuracy"], 0.6)
        self.assertEqual(summary["complement_recovery_rate"], 0.5)
        self.assertEqual(summary["complement_recovery_precision"], 0.4)
        self.assertEqual(summary["complement_digit_structure"], 0.75)

    def test_trace_aggregation_weights_tokens(self) -> None:
        first = {key: 0 for key in (
            "tokens", "gold_digits", "full_correct", "complement_correct", "noise_correct",
            "full_wrong", "complement_recovers", "noise_recovers", "complement_disagrees",
            "noise_disagrees", "complement_digit_at_gold_digit", "noise_digit_at_gold_digit",
            "complement_contending_digit", "noise_contending_digit", "complement_contextual_digit",
            "noise_contextual_digit",
        )}
        first.update({"tokens": 1, "full_correct": 1, "activation_sum": 0.5, "random_activation_sum": 0.9, "activation_count": 1})
        second = dict(first)
        second.update({"tokens": 3, "full_correct": 0, "activation_sum": 1.5, "random_activation_sum": 2.7, "activation_count": 3})
        summary = _aggregate_trace_rows([first, second])
        self.assertEqual(summary["full_token_accuracy"], 0.25)
        self.assertEqual(summary["activation_complement_fraction"], 0.5)

    def test_complement_transform_invariants(self) -> None:
        generator = torch.Generator().manual_seed(7)
        hidden = torch.randn(2, 3, 8, generator=generator)
        basis = torch.linalg.qr(torch.randn(8, 3, generator=generator)).Q.T
        dominant = _transform_hidden(hidden, basis, "dominant")
        complement = _transform_hidden(hidden, basis, "complement")
        noise = torch.randn(2, 3, 8, generator=generator)
        randomized = _transform_hidden(hidden, basis, "random-complement", noise)
        self.assertTrue(torch.allclose(dominant + complement, hidden, atol=1e-5))
        self.assertTrue(torch.allclose(complement @ basis.T, torch.zeros(2, 3, 3), atol=1e-5))
        randomized_complement = randomized - _transform_hidden(randomized, basis, "dominant")
        self.assertTrue(torch.allclose(randomized_complement.norm(dim=-1), complement.norm(dim=-1), atol=1e-5))

    def test_generation_rows_match_direct_transforms(self) -> None:
        generator = torch.Generator().manual_seed(11)
        hidden = torch.randn(1, 4, 8, generator=generator)
        basis = torch.linalg.qr(torch.randn(8, 3, generator=generator)).Q.T
        expected_dominant = _transform_hidden(hidden, basis, "dominant")

        full = _transform_generation_row(hidden, basis, "full", torch.Generator().manual_seed(1), None)
        complement_zero = _transform_generation_row(hidden, basis, "complement-zero", torch.Generator().manual_seed(2), None)
        scaled = _transform_generation_row(hidden, basis, "complement-scale-0.75", torch.Generator().manual_seed(3), None)

        self.assertTrue(torch.equal(full, hidden))
        self.assertTrue(torch.allclose(complement_zero, expected_dominant, atol=1e-5))
        self.assertTrue(torch.allclose(scaled, expected_dominant + 0.75 * (hidden - expected_dominant), atol=1e-5))

    def test_training_interventions_block_complement_gradients(self) -> None:
        generator = torch.Generator().manual_seed(19)
        basis = torch.linalg.qr(torch.randn(8, 3, generator=generator)).Q.T
        for mode in ("complement-zero", "complement-randomized"):
            hidden = torch.randn(2, 4, 8, generator=generator, requires_grad=True)
            transformed = _intervene_training_hidden(hidden, basis, mode, generator)
            transformed.sum().backward()
            complement_gradient = hidden.grad - (hidden.grad @ basis.T) @ basis
            self.assertTrue(torch.allclose(complement_gradient, torch.zeros_like(complement_gradient), atol=1e-5))
            if mode == "complement-randomized":
                randomized = transformed.detach() - _transform_hidden(transformed.detach(), basis, "dominant")
                original = hidden.detach() - _transform_hidden(hidden.detach(), basis, "dominant")
                self.assertTrue(torch.allclose(randomized.norm(dim=-1), original.norm(dim=-1), atol=1e-5))

    def test_masked_training_intervention_preserves_non_events(self) -> None:
        generator = torch.Generator().manual_seed(21)
        basis = torch.linalg.qr(torch.randn(8, 3, generator=generator)).Q.T
        hidden = torch.randn(2, 4, 8, generator=generator, requires_grad=True)
        mask = torch.tensor([[False, True, False, True], [True, False, False, False]])
        transformed = _intervene_training_hidden(hidden, basis, "complement-zero", generator, mask)
        self.assertTrue(torch.equal(transformed[~mask], hidden[~mask]))
        transformed.sum().backward()
        self.assertTrue(torch.equal(hidden.grad[~mask], torch.ones_like(hidden.grad[~mask])))
        event_complement_gradient = hidden.grad[mask] - (hidden.grad[mask] @ basis.T) @ basis
        self.assertTrue(torch.allclose(event_complement_gradient, torch.zeros_like(event_complement_gradient), atol=1e-5))

    def test_masked_dominant_deletion_preserves_only_event_complement(self) -> None:
        generator = torch.Generator().manual_seed(22)
        basis = torch.linalg.qr(torch.randn(8, 3, generator=generator)).Q.T
        hidden = torch.randn(2, 4, 8, generator=generator, requires_grad=True)
        mask = torch.tensor([[False, True, False, True], [True, False, False, False]])
        transformed = _intervene_training_hidden(hidden, basis, "dominant-zero", generator, mask)
        self.assertTrue(torch.equal(transformed[~mask], hidden[~mask]))
        self.assertTrue(torch.allclose(transformed[mask] @ basis.T, torch.zeros(3, 3), atol=1e-5))
        transformed.sum().backward()
        self.assertTrue(torch.equal(hidden.grad[~mask], torch.ones_like(hidden.grad[~mask])))
        event_dominant_gradient = (hidden.grad[mask] @ basis.T) @ basis
        self.assertTrue(torch.allclose(event_dominant_gradient, torch.zeros_like(event_dominant_gradient), atol=1e-5))

    def test_random_training_control_is_seeded_and_orthonormal(self) -> None:
        basis = torch.eye(8)[:3]
        first, mode = _training_arm_basis(basis, "random-zero", 23)
        second, _ = _training_arm_basis(basis, "random-zero", 23)
        self.assertEqual(mode, "complement-zero")
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.allclose(first @ first.T, torch.eye(3), atol=1e-5))
        self.assertFalse(torch.equal(first, basis))
        _, dominant_mode = _training_arm_basis(basis, "random-dominant-zero", 23)
        self.assertEqual(dominant_mode, "dominant-zero")


if __name__ == "__main__":
    unittest.main()