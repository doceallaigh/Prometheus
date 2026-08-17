from __future__ import annotations

import unittest

import torch

from prometheus.config import DataConfig, ModelConfig
from prometheus.data import CharacterTokenizer, generate_reasoning_problems, reasoning_chain_corpus
from prometheus.latent_reasoning import JSpaceCfCLoop, RRSJCfCModel, _parse_answer
from prometheus.model import DenseTransformerLM


def _data_config(**overrides) -> DataConfig:
    values = {
        "dataset_type": "reasoning_chain",
        "sequence_length": 64,
        "batch_size": 4,
        "chain_length_min": 2,
        "chain_length_max": 4,
        "num_problems": 50,
        "reasoning_seed": 7,
    }
    values.update(overrides)
    return DataConfig(**values)


class ReasoningChainDataTests(unittest.TestCase):
    def test_problem_generation_is_deterministic(self) -> None:
        first = generate_reasoning_problems(_data_config(), split="train")
        second = generate_reasoning_problems(_data_config(), split="train")
        self.assertEqual([p.expression for p in first], [p.expression for p in second])

    def test_train_and_val_splits_are_disjoint(self) -> None:
        train = {p.expression for p in generate_reasoning_problems(_data_config(), split="train")}
        val = {p.expression for p in generate_reasoning_problems(_data_config(num_problems=20), split="val")}
        self.assertTrue(train)
        self.assertTrue(val)
        self.assertFalse(train & val)

    def test_intermediates_track_left_to_right_mod_100(self) -> None:
        for problem in generate_reasoning_problems(_data_config(), split="train")[:10]:
            self.assertEqual(problem.intermediates[-1], problem.answer)
            self.assertEqual(len(problem.intermediates), problem.chain_length)
            self.assertTrue(all(0 <= value < 100 for value in problem.intermediates))

    def test_mixed_corpus_contains_both_formats(self) -> None:
        corpus = reasoning_chain_corpus(_data_config())
        self.assertIn("=T", corpus)
        self.assertIn("=A", corpus)

    def test_parse_answer_extracts_final_answer(self) -> None:
        self.assertEqual(_parse_answer("T19,57,52:A52;"), 52)
        self.assertEqual(_parse_answer("A7;"), 7)
        self.assertIsNone(_parse_answer("T19,57;"))


class RRSJCfCModelTests(unittest.TestCase):
    def _build_model(self) -> tuple[RRSJCfCModel, DenseTransformerLM]:
        model_config = ModelConfig(
            vocab_size=21,
            embedding_dim=32,
            num_heads=4,
            num_layers=4,
            dropout=0.0,
            architecture="dense",
        )
        base = DenseTransformerLM(model_config, sequence_length=32)
        base.eval()
        loop = JSpaceCfCLoop(d_model=32, d_cfc=16, vocab_size=21, max_steps=16)
        return RRSJCfCModel(base=base, jspace_layer_index=2, loop=loop), base

    def test_correction_is_zero_at_initialization(self) -> None:
        torch.manual_seed(0)
        model, base = self._build_model()
        tokens = torch.randint(0, 21, (2, 12))
        with torch.no_grad():
            corrected = model.corrected_logits(tokens)
            base_logits = base(tokens).logits
        self.assertTrue(torch.allclose(corrected, base_logits, atol=1e-6))

    def test_base_parameters_are_frozen_and_loop_receives_gradients(self) -> None:
        torch.manual_seed(0)
        model, _ = self._build_model()
        tokens = torch.randint(0, 21, (2, 12))
        logits = model.corrected_logits(tokens)
        loss = logits.sum()
        loss.backward()
        self.assertTrue(all(not p.requires_grad for p in model.base.parameters()))
        head_grads = [p.grad for p in model.loop.logit_head.parameters()]
        self.assertTrue(all(g is not None for g in head_grads))

    def test_step_and_sequence_forward_agree(self) -> None:
        torch.manual_seed(0)
        model, _ = self._build_model()
        tokens = torch.randint(0, 21, (2, 6))
        h_j, _ = model.states_and_logits(tokens)
        sequence_bias = model.loop(h_j)
        state = model.loop.initial_state(2, tokens.device)
        for position in range(6):
            step_bias, state = model.loop.step(h_j[:, position], state)
            self.assertTrue(torch.allclose(step_bias, sequence_bias[:, position], atol=1e-6))

    def test_gru_cell_variant_is_zero_at_initialization(self) -> None:
        torch.manual_seed(0)
        _, base = self._build_model()
        loop = JSpaceCfCLoop(d_model=32, d_cfc=16, vocab_size=21, max_steps=16, cell_type="gru")
        model = RRSJCfCModel(base=base, jspace_layer_index=2, loop=loop)
        tokens = torch.randint(0, 21, (2, 12))
        with torch.no_grad():
            corrected = model.corrected_logits(tokens)
            base_logits = base(tokens).logits
        self.assertTrue(torch.allclose(corrected, base_logits, atol=1e-6))

    def test_embedding_tap_layer_zero_is_accepted(self) -> None:
        torch.manual_seed(0)
        _, base = self._build_model()
        loop = JSpaceCfCLoop(d_model=32, d_cfc=16, vocab_size=21, max_steps=16)
        model = RRSJCfCModel(base=base, jspace_layer_index=0, loop=loop)
        tokens = torch.randint(0, 21, (2, 8))
        h_j, logits = model.states_and_logits(tokens)
        self.assertEqual(h_j.shape, (2, 8, 32))
        self.assertEqual(logits.shape, (2, 8, 21))
        with torch.no_grad():
            expected = base(tokens).logits
        self.assertTrue(torch.allclose(logits, expected, atol=1e-6))

    def test_invalid_cell_type_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            JSpaceCfCLoop(d_model=32, d_cfc=16, vocab_size=21, max_steps=16, cell_type="lstm")


if __name__ == "__main__":
    unittest.main()
