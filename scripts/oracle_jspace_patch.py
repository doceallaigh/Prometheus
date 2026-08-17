"""Oracle diagnostic: patch J-space states with true CoT-derived target activations
and measure exact-match accuracy through the frozen upper layers.

Modes:
- single (default): patch one vector at the answer position (legacy design).
- --span: patch the whole THINK span (positions between 'T' and 'A') at the input
  to block L with the true CoT states, using placeholder tokens below layer L.
  This bounds what a loop that emits virtual THINK states can achieve.

Usage:
    python scripts/oracle_jspace_patch.py <base-run-dir> <layer_index> [num_problems] [--span] [--placeholder C]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

from prometheus.config import DataConfig
from prometheus.data import generate_reasoning_problems
from prometheus.latent_reasoning import _parse_answer, load_base_checkpoint


@torch.no_grad()
def main() -> None:
    base_run = Path(sys.argv[1])
    layer_index = int(sys.argv[2])
    num_problems = int(sys.argv[3]) if len(sys.argv) > 3 else 300
    span_mode = "--span" in sys.argv
    slots = 0
    if "--slots" in sys.argv:
        slots = int(sys.argv[sys.argv.index("--slots") + 1])
    broadcast = "--broadcast" in sys.argv
    align_positions = "--align-positions" in sys.argv
    placeholder = "\n"
    if "--placeholder" in sys.argv:
        placeholder = sys.argv[sys.argv.index("--placeholder") + 1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    snapshot = json.loads((base_run / "config.snapshot.json").read_text(encoding="utf-8"))
    data_config = DataConfig(**snapshot["data"])
    # Generate problems before loading the torch checkpoint: on this machine,
    # loading torch checkpoints intermittently corrupts the heap in ways that
    # crash subsequent pure-Python random number generation (0xC0000005).
    problems = generate_reasoning_problems(data_config, split="val")[:num_problems]
    base, tokenizer = load_base_checkpoint(base_run / "checkpoint.pt", device)

    def hidden_at(text: str, position: int) -> torch.Tensor:
        tokens = torch.tensor([tokenizer.encode(text)], dtype=torch.long, device=device)
        x = base._embed(tokens)
        for block in base.blocks[:layer_index]:
            x = block(x)
        return x[0, position]

    def lower(text: str) -> torch.Tensor:
        tokens = torch.tensor([tokenizer.encode(text)], dtype=torch.long, device=device)
        x = base._embed(tokens)
        for block in base.blocks[:layer_index]:
            x = block(x)
        return x

    correct_by_key: dict[str, list[bool]] = {}
    stop_id = tokenizer.stoi[";"]

    def decode_with_patch(prompt: str, patch_slice: slice, patch: torch.Tensor) -> str:
        tokens = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
        generated: list[int] = []
        for _ in range(8):
            x = base._embed(tokens)
            for block in base.blocks[:layer_index]:
                x = block(x)
            x[0, patch_slice] = patch
            for block in base.blocks[layer_index:]:
                x = block(x)
            logits = base.lm_head(base.norm(x))
            next_id = int(logits[0, -1].argmax().item())
            generated.append(next_id)
            if next_id == stop_id:
                break
            tokens = torch.cat([tokens, torch.tensor([[next_id]], device=device)], dim=1)
        return tokenizer.decode(generated)

    padded_mode = "--padded" in sys.argv

    for problem in problems:
        cot_text = problem.cot_text()
        if padded_mode:
            # Exact dress rehearsal of the planned training layout: teacher text is the
            # CoT span right-padded to a fixed slot count, student text has all slots as
            # placeholders; ALL slot states are patched from the padded teacher.
            think_start = cot_text.index("T") + 1
            a_position = cot_text.rindex("A")
            span = cot_text[think_start:a_position]
            width = max(slots, len(span))
            teacher = cot_text[:think_start] + span + placeholder * (width - len(span))
            h_slots = lower(teacher)[0, think_start : think_start + width].clone()
            prompt = cot_text[:think_start] + placeholder * width + "A"
            generated = decode_with_patch(prompt, slice(think_start, think_start + width), h_slots)
            parsed = _parse_answer("A" + generated)
            is_correct = parsed == problem.answer
            correct_by_key.setdefault("overall", []).append(is_correct)
            correct_by_key.setdefault(f"chain_{problem.chain_length}", []).append(is_correct)
            continue
        if span_mode:
            think_start = cot_text.index("T") + 1
            a_position = cot_text.rindex("A")
            span_len = a_position - think_start
            h_span = lower(cot_text)[0, think_start:a_position].clone()
            width = max(slots, span_len)
            prompt = cot_text[:think_start] + placeholder * width + "A"
            generated = decode_with_patch(prompt, slice(think_start, think_start + span_len), h_span)
            parsed = _parse_answer("A" + generated)
            is_correct = parsed == problem.answer
            correct_by_key.setdefault("overall", []).append(is_correct)
            correct_by_key.setdefault(f"chain_{problem.chain_length}", []).append(is_correct)
            continue
        a_position = cot_text.rindex("A")
        h_target = hidden_at(cot_text, a_position)

        prompt = f"Q{problem.expression}=A"
        if align_positions:
            # Left-pad with newline filler so the direct "A" position matches the
            # CoT "A" position, neutralizing positional-embedding mismatch.
            padding = "\n" * (a_position - (len(prompt) - 1))
            prompt = padding + prompt
        inject_position = len(prompt) - 1
        tokens = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
        generated: list[int] = []
        for _ in range(8):
            x = base._embed(tokens)
            for block in base.blocks[:layer_index]:
                x = block(x)
            if broadcast:
                x[0, inject_position:] = x[0, inject_position:] + h_target
            else:
                x[0, inject_position] = h_target
            for block in base.blocks[layer_index:]:
                x = block(x)
            logits = base.lm_head(base.norm(x))
            next_id = int(logits[0, -1].argmax().item())
            generated.append(next_id)
            if next_id == stop_id:
                break
            tokens = torch.cat([tokens, torch.tensor([[next_id]], device=device)], dim=1)
        parsed = _parse_answer("A" + tokenizer.decode(generated))
        is_correct = parsed == problem.answer
        correct_by_key.setdefault("overall", []).append(is_correct)
        correct_by_key.setdefault(f"chain_{problem.chain_length}", []).append(is_correct)

    for key in sorted(correct_by_key):
        values = correct_by_key[key]
        print(f"{key}: accuracy={sum(values)/len(values):.4f} n={len(values)}")


if __name__ == "__main__":
    main()
