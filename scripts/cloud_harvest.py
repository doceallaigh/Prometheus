"""Cloud harvest: vLLM-based CoT trace generation for 32B-class trunks.

Drop-in producer for the same JSONL trace format as `retrofit-harvest`
({"prompt", "completion", "answer"}), but using vLLM's continuous batching —
harvest is pure generation (no J-space hooks), so a single A100-80/H100 turns
a multi-day 3090 harvest into a couple of hours.

Run on any GPU cloud box (RunPod / Lambda / GCP):

    pip install vllm datasets
    python cloud_harvest.py --model Qwen/QwQ-32B --dataset math \
        --num-problems 300 --max-new-tokens 4096 \
        --output traces-pilot.jsonl

Then scp/rsync the JSONL back to outputs/retrofit-qwq32b/ for corrector
training. Keep/score logic mirrors src/prometheus/retrofit.py exactly
(#### marker, numeric \\boxed{} fallback).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

COT_PROMPT = (
    "Solve the math problem step by step. End your response with the final "
    "numeric answer on its own line in the form '#### <answer>'.\n\nProblem: "
)
ANSWER_RE = re.compile(r"####\s*\$?(-?[\d,]+(?:\.\d+)?)")
BOXED_RE = re.compile(r"\\boxed\{\s*\$?(-?[\d,]+(?:\.\d+)?)\s*\}")


def _normalize_number(raw: str) -> str:
    value = raw.replace(",", "").replace("$", "").rstrip(".")
    if value.endswith(".0"):
        value = value[:-2]
    return value


def extract_answer(text: str) -> str | None:
    match = ANSWER_RE.search(text)
    if match is not None:
        return _normalize_number(match.group(1))
    boxed = BOXED_RE.findall(text)
    if boxed:
        return _normalize_number(boxed[-1])
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", choices=["gsm8k", "math"], default="math")
    parser.add_argument("--num-problems", type=int, default=300)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tensor-parallel", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--chunk-size", type=int, default=128, help="Problems per generate call; resume checkpoint granularity (spot preemption loses at most one chunk)")
    args = parser.parse_args()

    from datasets import load_dataset
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    if args.dataset == "math":
        raw = load_dataset("DigitalLearningGmbH/MATH-lighteval", "default", split="train")
        rows = [
            {"question": row["problem"], "answer": f"#### {gold}"}
            for row in raw
            if (gold := extract_answer(row["solution"])) is not None
        ]
        print(json.dumps({"math_numeric_problems": len(rows)}), flush=True)
    else:
        rows = list(load_dataset("openai/gsm8k", "main", split="train"))
    rows = rows[: args.num_problems]

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": COT_PROMPT + row["question"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for row in rows
    ]

    llm = LLM(
        model=args.model,
        dtype=args.dtype,
        tensor_parallel_size=args.tensor_parallel,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path = output_path.with_name(output_path.name + ".progress")
    processed = 0
    kept = 0
    if progress_path.exists():
        state = json.loads(progress_path.read_text(encoding="utf-8"))
        processed, kept = state["processed"], state["kept"]
        print(json.dumps({"resumed_at": processed, "kept": kept}), flush=True)
    mode = "a" if processed else "w"
    with output_path.open(mode, encoding="utf-8") as sink:
        for begin in range(processed, len(rows), args.chunk_size):
            chunk_rows = rows[begin : begin + args.chunk_size]
            chunk_prompts = prompts[begin : begin + args.chunk_size]
            outputs = llm.generate(chunk_prompts, sampling)
            for row, prompt, result in zip(chunk_rows, chunk_prompts, outputs):
                completion = result.outputs[0].text
                gold = extract_answer(row["answer"])
                predicted = extract_answer(completion)
                if gold is not None and predicted == gold:
                    kept += 1
                    sink.write(json.dumps({"prompt": prompt, "completion": completion, "answer": gold}) + "\n")
            sink.flush()
            done = begin + len(chunk_rows)
            progress_path.write_text(json.dumps({"processed": done, "kept": kept}), encoding="utf-8")
            print(json.dumps({"processed": done, "total": len(rows), "kept": kept}), flush=True)
    summary = {"model": args.model, "total": len(rows), "kept": kept, "keep_rate": kept / max(len(rows), 1)}
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
