from __future__ import annotations

import unittest

import torch

from prometheus.latent_composition import (
    CompositionTrunk,
    ContinuousHopSidecar,
    LatentCompositionModel,
    encode_proofs,
    generate_transitive_proofs,
)


class LatentCompositionTests(unittest.TestCase):
    def test_generated_proofs_are_valid(self) -> None:
        proofs = generate_transitive_proofs(40, 2, 5, entities=16, distractors=3, seed=7)
        self.assertEqual(len(proofs), 40)
        for proof in proofs:
            self.assertEqual(proof.depth + 1, len(proof.chain))
            self.assertEqual(proof.start, proof.chain[0])
            self.assertEqual(proof.answer, proof.chain[-1])
            self.assertTrue(all((proof.chain[i], proof.chain[i + 1]) in proof.edges for i in range(proof.depth)))

    def test_zero_steps_exactly_matches_frozen_trunk(self) -> None:
        proofs = generate_transitive_proofs(4, 2, 3, entities=12, distractors=2, seed=11)
        tokens, _ = encode_proofs(proofs, entities=12, max_edges=6)
        trunk = CompositionTrunk(12, tokens.size(1), 32, heads=4, lower_layers=1, upper_layers=1)
        model = LatentCompositionModel(trunk, ContinuousHopSidecar(32, 12))
        with torch.no_grad():
            self.assertTrue(torch.equal(trunk(tokens), model(tokens, 0)))

    def test_latent_steps_reach_formal_endpoint(self) -> None:
        entities = 12
        proof = generate_transitive_proofs(1, 5, 5, entities=entities, distractors=2, seed=19)[0]
        tokens, _ = encode_proofs([proof], entities=entities, max_edges=7)
        sidecar = ContinuousHopSidecar(entities, entities)
        embedding = torch.nn.Embedding(entities + 3, entities)
        with torch.no_grad():
            embedding.weight.zero_()
            embedding.weight[:entities].copy_(torch.eye(entities))
            sidecar.output.weight.copy_(torch.eye(entities))
            sidecar.output.bias.zero_()
            state = sidecar(torch.zeros(1, tokens.size(1), entities), tokens, embedding, proof.depth)
        expected = torch.nn.functional.one_hot(torch.tensor([proof.answer]), entities).float()
        self.assertTrue(torch.equal(state, expected))


if __name__ == "__main__":
    unittest.main()