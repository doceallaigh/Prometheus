from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from prometheus.model import TransformerBlock


@dataclass(frozen=True)
class TransitiveProof:
    start: int
    answer: int
    edges: tuple[tuple[int, int], ...]
    chain: tuple[int, ...]
    depth: int

    def formal_text(self) -> str:
        premises = ", ".join(f"R(e{source},e{target})" for source, target in self.edges)
        proof = " -> ".join(f"e{entity}" for entity in self.chain)
        return f"{premises} |- Reach(e{self.start},e{self.answer}); proof: {proof}"


def generate_transitive_proofs(
    count: int,
    min_depth: int,
    max_depth: int,
    entities: int,
    distractors: int,
    seed: int,
) -> list[TransitiveProof]:
    """Generate unique directed-chain proofs with unrelated distractor edges."""

    if min_depth < 1 or max_depth < min_depth:
        raise ValueError("invalid proof-depth range")
    if entities < max_depth + 2:
        raise ValueError("entities must exceed maximum proof depth")
    generator = random.Random(seed)
    proofs: list[TransitiveProof] = []
    seen: set[tuple] = set()
    while len(proofs) < count:
        depth = generator.randint(min_depth, max_depth)
        chain = tuple(generator.sample(range(entities), depth + 1))
        chain_edges = [(chain[index], chain[index + 1]) for index in range(depth)]
        used = set(chain_edges)
        edges = list(chain_edges)
        attempts = 0
        while len(edges) < depth + distractors and attempts < 1000:
            attempts += 1
            source, target = generator.sample(range(entities), 2)
            edge = (source, target)
            if edge in used or source in chain[:-1]:
                continue
            used.add(edge)
            edges.append(edge)
        generator.shuffle(edges)
        key = (chain[0], chain[-1], tuple(sorted(edges)))
        if key in seen:
            continue
        seen.add(key)
        proofs.append(
            TransitiveProof(
                start=chain[0],
                answer=chain[-1],
                edges=tuple(edges),
                chain=chain,
                depth=depth,
            )
        )
    return proofs


def encode_proofs(proofs: list[TransitiveProof], entities: int, max_edges: int) -> tuple[torch.Tensor, torch.Tensor]:
    query_token = entities
    separator_token = entities + 1
    pad_token = entities + 2
    sequence_length = 3 + 2 * max_edges
    tokens = torch.full((len(proofs), sequence_length), pad_token, dtype=torch.long)
    answers = torch.empty(len(proofs), dtype=torch.long)
    for row, proof in enumerate(proofs):
        for edge_index, (source, target) in enumerate(proof.edges[:max_edges]):
            tokens[row, 2 * edge_index] = source
            tokens[row, 2 * edge_index + 1] = target
        tokens[row, -3:] = torch.tensor([separator_token, proof.start, query_token])
        answers[row] = proof.answer
    return tokens, answers


class CompositionTrunk(nn.Module):
    def __init__(self, entities: int, sequence_length: int, d_model: int, heads: int, lower_layers: int, upper_layers: int):
        super().__init__()
        self.entities = entities
        self.sequence_length = sequence_length
        self.token_embedding = nn.Embedding(entities + 3, d_model)
        self.position_embedding = nn.Embedding(sequence_length, d_model)
        self.lower = nn.ModuleList([TransformerBlock(d_model, heads, 4, 0.0) for _ in range(lower_layers)])
        self.upper = nn.ModuleList([TransformerBlock(d_model, heads, 4, 0.0) for _ in range(upper_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, entities)

    def lower_states(self, tokens: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(tokens.size(1), device=tokens.device)
        hidden = self.token_embedding(tokens) + self.position_embedding(positions)
        for block in self.lower:
            hidden = block(hidden)
        return hidden

    def upper_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        for block in self.upper:
            hidden = block(hidden)
        return self.classifier(self.norm(hidden[:, -1]))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.upper_logits(self.lower_states(tokens))


class ContinuousHopSidecar(nn.Module):
    """Propagate a continuous entity distribution over the proof graph."""

    def __init__(self, d_model: int, entities: int):
        super().__init__()
        self.entities = entities
        self.output = nn.Linear(d_model, d_model)

    def forward(self, lower: torch.Tensor, tokens: torch.Tensor, embedding: nn.Embedding, steps: int) -> torch.Tensor:
        edge_end = tokens.size(1) - 3
        source_tokens = tokens[:, :edge_end:2]
        target_tokens = tokens[:, 1:edge_end:2]
        valid = (source_tokens < self.entities) & (target_tokens < self.entities)
        safe_sources = source_tokens.clamp_max(self.entities - 1)
        safe_targets = target_tokens.clamp_max(self.entities - 1)
        source_one_hot = F.one_hot(safe_sources, self.entities).float() * valid.unsqueeze(-1)
        target_one_hot = F.one_hot(safe_targets, self.entities).float() * valid.unsqueeze(-1)
        adjacency = torch.einsum("bei,bej->bij", source_one_hot, target_one_hot)
        start = tokens[:, -2].clamp_max(self.entities - 1)
        state = F.one_hot(start, self.entities).float()
        for _ in range(steps):
            next_state = torch.bmm(state.unsqueeze(1), adjacency).squeeze(1)
            normalizer = next_state.sum(-1, keepdim=True)
            state = torch.where(normalizer > 0, next_state / normalizer.clamp_min(1e-9), state)
        entity_state = state @ embedding.weight[: self.entities]
        return self.output(entity_state)


class LatentCompositionModel(nn.Module):
    def __init__(self, trunk: CompositionTrunk, sidecar: ContinuousHopSidecar):
        super().__init__()
        self.trunk = trunk
        self.sidecar = sidecar
        self.trunk.requires_grad_(False)
        self.trunk.eval()

    def forward(self, tokens: torch.Tensor, latent_steps: int) -> torch.Tensor:
        with torch.no_grad():
            lower = self.trunk.lower_states(tokens)
        if latent_steps == 0:
            return self.trunk.upper_logits(lower)
        composed = self.sidecar(lower, tokens, self.trunk.token_embedding, latent_steps)
        adapted = lower.clone()
        adapted[:, -1] = lower[:, -1] + composed
        return self.trunk.upper_logits(adapted)


def _batches(tokens: torch.Tensor, answers: torch.Tensor, batch_size: int, generator: torch.Generator):
    order = torch.randperm(tokens.size(0), generator=generator)
    for begin in range(0, tokens.size(0), batch_size):
        indices = order[begin: begin + batch_size]
        yield tokens[indices], answers[indices]


@torch.no_grad()
def _accuracy(model, tokens: torch.Tensor, answers: torch.Tensor, device: torch.device, batch_size: int, steps: int | None = None) -> float:
    correct = 0
    for begin in range(0, tokens.size(0), batch_size):
        batch_tokens = tokens[begin: begin + batch_size].to(device)
        logits = model(batch_tokens) if steps is None else model(batch_tokens, steps)
        correct += int((logits.argmax(-1).cpu() == answers[begin: begin + batch_size]).sum())
    return correct / tokens.size(0)


def _plot_html(rows: list[dict]) -> str:
    depths = sorted({row["test_depth"] for row in rows})
    traces = []
    for depth in depths:
        selected = [row for row in rows if row["test_depth"] == depth]
        traces.append({
            "x": [row["latent_steps"] for row in selected],
            "y": [row["accuracy"] for row in selected],
            "name": f"proof depth {depth}",
            "mode": "lines+markers",
        })
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Multi-Hop Latent Composition</title><script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>body{{margin:0;padding:20px;background:#101820;color:#f2f4f3;font-family:Segoe UI,sans-serif}}#plot{{height:720px}}</style>
</head><body><h1>Multi-Hop Latent Composition</h1><div id="plot"></div><script>
Plotly.newPlot('plot',{json.dumps(traces)},{{template:'plotly_dark',title:'Accuracy versus continuous latent steps',
xaxis:{{title:'latent sidecar steps',dtick:1}},yaxis:{{title:'endpoint accuracy',range:[0,1]}}}},{{responsive:true,displaylogo:false}});
</script></body></html>"""


def run_latent_composition(
    output_dir: str | Path,
    train_proofs: int = 1000,
    test_per_depth: int = 200,
    train_min_depth: int = 2,
    train_max_depth: int = 4,
    test_max_depth: int = 10,
    entities: int = 24,
    distractors: int = 4,
    d_model: int = 128,
    trunk_steps: int = 1200,
    sidecar_steps: int = 1800,
    batch_size: int = 64,
    learning_rate: float = 3e-4,
    device_str: str = "auto",
    seed: int = 20260720,
) -> dict:
    device = torch.device(device_str if device_str != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(seed)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    max_edges = test_max_depth + distractors

    train = generate_transitive_proofs(train_proofs, train_min_depth, train_max_depth, entities, distractors, seed)
    tests = {
        depth: generate_transitive_proofs(test_per_depth, depth, depth, entities, distractors, seed + 1000 + depth)
        for depth in range(train_min_depth, test_max_depth + 1)
    }
    with (output / "formal_proofs.jsonl").open("w", encoding="utf-8") as sink:
        for split, proofs in [("train", train)] + [(f"test_depth_{depth}", proofs) for depth, proofs in tests.items()]:
            for proof in proofs:
                sink.write(json.dumps({"split": split, **asdict(proof), "formal": proof.formal_text()}) + "\n")

    train_tokens, train_answers = encode_proofs(train, entities, max_edges)
    encoded_tests = {depth: encode_proofs(proofs, entities, max_edges) for depth, proofs in tests.items()}
    trunk = CompositionTrunk(entities, train_tokens.size(1), d_model, heads=4, lower_layers=2, upper_layers=2).to(device)
    optimizer = torch.optim.AdamW(trunk.parameters(), lr=learning_rate, weight_decay=0.01)
    generator = torch.Generator().manual_seed(seed)
    start = time.time()
    trunk.train()
    for step in range(trunk_steps):
        indices = torch.randint(train_tokens.size(0), (batch_size,), generator=generator)
        tokens = train_tokens[indices].to(device)
        answers = train_answers[indices].to(device)
        loss = F.cross_entropy(trunk(tokens), answers)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trunk.parameters(), 1.0)
        optimizer.step()
        if step % 200 == 0 or step == trunk_steps - 1:
            print(json.dumps({"stage": "trunk", "step": step, "loss": float(loss), "elapsed": time.time() - start}), flush=True)
    trunk.eval()
    torch.save(trunk.state_dict(), output / "trunk.pt")

    sidecar = ContinuousHopSidecar(d_model, entities).to(device)
    latent = LatentCompositionModel(trunk, sidecar).to(device)
    optimizer = torch.optim.AdamW(sidecar.parameters(), lr=learning_rate, weight_decay=0.01)
    for step in range(sidecar_steps):
        indices = torch.randint(train_tokens.size(0), (batch_size,), generator=generator)
        tokens = train_tokens[indices].to(device)
        answers = train_answers[indices].to(device)
        depths = torch.tensor([train[int(index)].depth for index in indices], device=device)
        losses = []
        for depth in range(train_min_depth, train_max_depth + 1):
            mask = depths == depth
            if mask.any():
                losses.append(F.cross_entropy(latent(tokens[mask], depth), answers[mask]))
        loss = torch.stack(losses).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(sidecar.parameters(), 1.0)
        optimizer.step()
        if step % 200 == 0 or step == sidecar_steps - 1:
            print(json.dumps({"stage": "sidecar", "step": step, "loss": float(loss), "elapsed": time.time() - start}), flush=True)
    latent.eval()
    torch.save(sidecar.state_dict(), output / "sidecar.pt")

    rows = []
    for depth, (tokens, answers) in encoded_tests.items():
        direct_accuracy = _accuracy(trunk, tokens, answers, device, batch_size)
        for latent_steps in range(0, test_max_depth + 3):
            accuracy = direct_accuracy if latent_steps == 0 else _accuracy(latent, tokens, answers, device, batch_size, latent_steps)
            rows.append({"test_depth": depth, "latent_steps": latent_steps, "accuracy": accuracy})
    (output / "metrics.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    (output / "accuracy_vs_latent_steps.html").write_text(_plot_html(rows), encoding="utf-8")

    lines = [
        "# Multi-Hop Latent Composition",
        "",
        f"Programmatic formal proofs: {train_proofs} train plus {test_per_depth} per test depth. "
        f"Sidecar trained only on depths {train_min_depth}-{train_max_depth}; depths {train_max_depth + 1}-{test_max_depth} are compositional OOD.",
        "",
        "No chain-of-thought tokens are generated. The sidecar propagates a continuous entity distribution through the encoded relation graph for N recurrent steps, then hands one learned hidden vector to the frozen upper trunk.",
        "",
        "| test depth | direct (0 steps) | best latent accuracy | best steps | matched-depth accuracy |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    summaries = []
    for depth in sorted(tests):
        selected = [row for row in rows if row["test_depth"] == depth]
        best = max(selected[1:], key=lambda row: row["accuracy"])
        matched = next(row for row in selected if row["latent_steps"] == depth)
        direct = selected[0]
        summaries.append({"depth": depth, "direct": direct["accuracy"], "best": best["accuracy"], "best_steps": best["latent_steps"], "matched": matched["accuracy"]})
        lines.append(f"| {depth} | {direct['accuracy']:.3f} | {best['accuracy']:.3f} | {best['latent_steps']} | {matched['accuracy']:.3f} |")
    ood = [row for row in summaries if row["depth"] > train_max_depth]
    mean_direct = sum(row["direct"] for row in ood) / len(ood)
    mean_matched = sum(row["matched"] for row in ood) / len(ood)
    lines.extend([
        "",
        f"Mean OOD direct accuracy: {mean_direct:.3f}; matched-step latent accuracy: {mean_matched:.3f}; gain: {mean_matched - mean_direct:+.3f}.",
        "",
        "Interactive plot: `accuracy_vs_latent_steps.html`.",
    ])
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary = {
        "train_proofs": train_proofs,
        "test_proofs": test_per_depth * len(tests),
        "train_depths": [train_min_depth, train_max_depth],
        "test_depths": [train_min_depth, test_max_depth],
        "mean_ood_direct_accuracy": mean_direct,
        "mean_ood_matched_accuracy": mean_matched,
        "mean_ood_gain": mean_matched - mean_direct,
        "seconds": time.time() - start,
        "output_dir": str(output),
    }
    (output / "run.summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)
    return summary