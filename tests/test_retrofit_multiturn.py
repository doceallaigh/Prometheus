from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from prometheus.retrofit import (
    _chat_turn_suffix,
    _context_key,
    _dependent_turn,
    _generate_stateful_turn,
    _multiturn_suffix,
    _sample_stateful_vote,
    _surface_answer,
)


class _Tokenizer:
    chat_template = None
    eos_token_id = 4

    def decode(self, token_ids, skip_special_tokens=True):
        return " ".join(str(token_id) for token_id in token_ids if token_id != self.eos_token_id)


class _ChatTokenizer(_Tokenizer):
    chat_template = "test"

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        rendered = "".join(f"<{message['role']}>{message['content']}</{message['role']}>" for message in messages)
        return rendered + ("<assistant>" if add_generation_prompt else "")


class _Model:
    def __init__(self, next_tokens):
        self.next_tokens = iter(next_tokens)
        self.calls = []

    def get_output_embeddings(self):
        return lambda hidden: hidden

    def __call__(self, tokens, past_key_values, output_hidden_states, use_cache):
        self.calls.append((tokens.tolist(), past_key_values))
        next_token = next(self.next_tokens)
        hidden = torch.zeros((1, tokens.size(1), 5))
        hidden[:, -1, next_token] = 1.0
        return SimpleNamespace(hidden_states=(hidden, hidden), past_key_values=len(self.calls))


class _Corrector:
    def __init__(self):
        self.initializations = 0

    def initial_state(self, batch_size, device):
        self.initializations += 1
        return torch.tensor([[0.0]], device=device)

    def step(self, tap, state):
        return torch.zeros_like(tap), state + 1


class RetrofitMultiturnTests(unittest.TestCase):
    def test_stateful_vote_stops_and_returns_majority_branch(self) -> None:
        model = _Model([2, 4, 2, 4, 3, 4])
        tokenizer = _Tokenizer()

        text, answer, generated, past, state, rollouts, internal = _sample_stateful_vote(
            model,
            tokenizer,
            None,
            tap_layer=0,
            input_ids=torch.tensor([[1]]),
            max_new_tokens=2,
            device=torch.device("cpu"),
            past=None,
            state=None,
            samples=4,
            temperature=0.0,
            stop_agreement=2,
        )

        self.assertEqual(answer, "2")
        self.assertEqual(text, "2")
        self.assertEqual(generated, [2, 4])
        self.assertEqual(len(rollouts), 2)
        self.assertEqual(internal, 4)
        self.assertEqual(past, 2)
        self.assertIsNone(state)

    def test_dependent_turns_withhold_key_from_user_context(self) -> None:
        question = "A box has 17 red and 23 blue balls."
        key = _context_key(question, episode_index=0)

        first_instruction, first_gold, prefix = _dependent_turn(question, key, 0)
        recall_instruction, recall_gold, _ = _dependent_turn(question, key, 1)
        transform_instruction, transform_gold, _ = _dependent_turn(question, key, 2)

        self.assertNotIn(str(key), first_instruction)
        self.assertNotIn(str(key), recall_instruction)
        self.assertEqual(first_gold, None)
        self.assertEqual(prefix, f"CONTEXT KEY: {key}\n\n")
        self.assertEqual(recall_gold, str(key))
        self.assertEqual(transform_gold, str(key + 7))
        self.assertNotIn(key, {17, 23})

    def test_stateful_turn_reuses_cache_and_corrector_state(self) -> None:
        model = _Model([3, 4, 2, 4])
        tokenizer = _Tokenizer()
        corrector = _Corrector()

        text, generated, past, state = _generate_stateful_turn(
            model,
            tokenizer,
            corrector,
            tap_layer=0,
            input_ids=torch.tensor([[1, 2]]),
            max_new_tokens=3,
            device=torch.device("cpu"),
        )
        second_text, second_generated, second_past, second_state = _generate_stateful_turn(
            model,
            tokenizer,
            corrector,
            tap_layer=0,
            input_ids=torch.tensor([[generated[-1], 1]]),
            max_new_tokens=3,
            device=torch.device("cpu"),
            past=past,
            state=state,
        )

        self.assertEqual(text, "3")
        self.assertEqual(second_text, "2")
        self.assertEqual(generated, [3, 4])
        self.assertEqual(second_generated, [2, 4])
        self.assertEqual(model.calls[2], ([[4, 1]], 2))
        self.assertEqual(second_past, 4)
        self.assertEqual(second_state.item(), 4.0)
        self.assertEqual(corrector.initializations, 1)

    def test_latent_surface_hides_chain_and_suffix_starts_next_turn(self) -> None:
        completion = "We calculate 6 * 7 = 42. #### 42"

        self.assertEqual(_surface_answer(completion), "#### 42")
        self.assertEqual(_surface_answer("no final marker"), "#### INVALID")
        self.assertIn("Problem: What is 3 + 4?", _multiturn_suffix(_Tokenizer(), "What is 3 + 4?"))

    def test_chat_suffix_uses_template_boundary_without_sentinel(self) -> None:
        suffix = _chat_turn_suffix(_ChatTokenizer(), "Recall the earlier value.")

        self.assertTrue(suffix.startswith("</assistant><user>"))
        self.assertTrue(suffix.endswith("</user><assistant>"))
        self.assertIn("Recall the earlier value.", suffix)
        self.assertNotIn("PROMETHEUS_ASSISTANT_CONTENT", suffix)


if __name__ == "__main__":
    unittest.main()