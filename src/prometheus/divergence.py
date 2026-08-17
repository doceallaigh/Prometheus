"""Divergence-onset labeling for divergence-gated computation (roadmap item 5a).

Uses latent self-consistency dumps (k sampled rollouts per problem) as free
supervision: within a problem, wrong rollouts diverge from correct siblings at
a measurable point. The labels feed a probe on the corrector's tap/state that
predicts imminent divergence.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

_EQ_RESULT_RE = re.compile(r"=\s*\$?\s*(-?\d[\d,]*(?:\.\d+)?)")
_ANNOTATION_RE = re.compile(r"<<[^>]*=\s*\$?\s*(-?\d[\d,]*(?:\.\d+)?)>>")
_NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def _to_float(text: str) -> float | None:
    try:
        return float(text.replace(",", ""))
    except ValueError:
        return None


def _allowed_values(question: str, solution: str, gold: str | None) -> set[float]:
    """Values sanctioned by the problem: question numbers + gold `<<..=c>>` intermediates + final answer."""

    allowed: set[float] = set()
    for match in _NUMBER_RE.finditer(question):
        value = _to_float(match.group())
        if value is not None:
            allowed.add(value)
    for match in _ANNOTATION_RE.finditer(solution):
        value = _to_float(match.group(1))
        if value is not None:
            allowed.add(value)
    if gold is not None:
        value = _to_float(gold)
        if value is not None:
            allowed.add(value)
    return allowed


def _first_error_site(text: str, allowed: set[float]) -> int | None:
    """Char offset of the first computed (`= x`) value not sanctioned by the gold solution."""

    for match in _EQ_RESULT_RE.finditer(text):
        value = _to_float(match.group(1))
        if value is None:
            continue
        if not any(abs(value - a) <= 1e-6 * max(1.0, abs(a)) for a in allowed):
            return match.start(1)
    return None


def _lenient_answer(text: str) -> str | None:
    from prometheus.retrofit import extract_answer_lenient

    return extract_answer_lenient(text)


def _common_prefix_len(a: str, b: str) -> int:
    limit = min(len(a), len(b))
    for i in range(limit):
        if a[i] != b[i]:
            return i
    return limit


def _iter_pairs(dump_path: Path):
    """Yield (index, question, gold, wrong_text, correct_sibling, onset_char) pairs."""

    sc_key = None
    for line in dump_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if sc_key is None:
            sc_key = next((k for k in row if k.startswith("latent_sc")), None)
            if sc_key is None:
                raise ValueError(f"No latent_sc* field found in {dump_path}")
        if sc_key not in row:
            continue
        rollouts = json.loads(row[sc_key])
        gold = row["gold"]
        judged = [(text, _lenient_answer(text) == gold) for text in rollouts]
        correct = [text for text, ok in judged if ok]
        incorrect = [text for text, ok in judged if not ok]
        if not correct or not incorrect:
            continue
        for text in incorrect:
            onset, sibling = max(
                ((_common_prefix_len(text, ref), ref) for ref in correct), key=lambda pair: pair[0]
            )
            yield row["index"], row["question"], gold, text, sibling, onset


def label_divergence(dump_path: str | Path, output_path: str | Path) -> dict:
    """Extract divergence-onset labels from a latent-SC completions dump.

    For every problem with at least one correct and one incorrect sampled
    rollout, each incorrect rollout is labeled with the character offset at
    which it departs from its nearest (longest-common-prefix) correct sibling.
    """

    dump_path = Path(dump_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sc_key = None
    problems = 0
    mixed = 0
    all_correct = 0
    all_wrong = 0
    records: list[dict] = []

    for line in dump_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if sc_key is None:
            sc_key = next((k for k in row if k.startswith("latent_sc")), None)
            if sc_key is None:
                raise ValueError(f"No latent_sc* field found in {dump_path}")
        if sc_key not in row:
            continue
        problems += 1
        rollouts = json.loads(row[sc_key])
        gold = row["gold"]
        judged = [(text, _lenient_answer(text) == gold) for text in rollouts]
        correct = [text for text, ok in judged if ok]
        incorrect = [text for text, ok in judged if not ok]
        if not correct or not incorrect:
            all_correct += not incorrect
            all_wrong += not correct
            continue
        mixed += 1
        for text in incorrect:
            onset, sibling = max(
                ((_common_prefix_len(text, ref), ref) for ref in correct), key=lambda pair: pair[0]
            )
            records.append(
                {
                    "index": row["index"],
                    "question": row["question"],
                    "gold": gold,
                    "wrong_answer": _lenient_answer(text),
                    "onset_char": onset,
                    "onset_frac": onset / max(len(text), 1),
                    "rollout_len": len(text),
                    "prefix": text[:onset],
                    "wrong_continuation": text[onset : onset + 80],
                    "correct_continuation": sibling[onset : onset + 80],
                }
            )

    with output_path.open("w", encoding="utf-8") as sink:
        for record in records:
            sink.write(json.dumps(record) + "\n")

    fractions = sorted(r["onset_frac"] for r in records)
    summary = {
        "dump": str(dump_path),
        "problems": problems,
        "mixed_problems": mixed,
        "all_correct_problems": all_correct,
        "all_wrong_problems": all_wrong,
        "labels": len(records),
        "onset_frac_median": fractions[len(fractions) // 2] if fractions else None,
        "onset_frac_p25": fractions[len(fractions) // 4] if fractions else None,
        "onset_frac_p75": fractions[3 * len(fractions) // 4] if fractions else None,
    }
    print(json.dumps(summary), flush=True)
    return summary


def label_error_sites(dump_path: str | Path, output_path: str | Path, split: str = "test") -> dict:
    """Relabel divergence pairs at the arithmetic error site (roadmap 5a, iteration 2).

    The sampling fork is where a wrong rollout *departs* from a correct
    sibling, not where it goes *wrong* — the probe's null result at the fork
    suggested exactly this gap. GSM8K gold solutions carry `<<a op b = c>>`
    calculator annotations; the first computed value in a wrong rollout that
    is not sanctioned by {question numbers, gold intermediates, gold answer}
    marks the error site. Correct siblings score the false-positive rate of
    the same detector.
    """

    from datasets import load_dataset

    dataset = load_dataset("openai/gsm8k", "main", split=split)
    records: list[dict] = []
    wrong_total = 0
    sibling_flagged = 0
    sibling_total = 0
    seen_siblings: set[tuple[int, int]] = set()

    for index, question, gold, wrong, sibling, onset in _iter_pairs(Path(dump_path)):
        solution = dataset[index]["answer"]
        allowed = _allowed_values(question, solution, gold)
        wrong_total += 1
        site = _first_error_site(wrong, allowed)
        sibling_key = (index, hash(sibling))
        if sibling_key not in seen_siblings:
            seen_siblings.add(sibling_key)
            sibling_total += 1
            sibling_flagged += _first_error_site(sibling, allowed) is not None
        if site is None:
            continue
        records.append(
            {
                "index": index,
                "gold": gold,
                "error_char": site,
                "error_frac": site / max(len(wrong), 1),
                "fork_char": onset,
                "error_minus_fork": site - onset,
                "context": wrong[max(0, site - 60) : site + 40],
            }
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as sink:
        for record in records:
            sink.write(json.dumps(record) + "\n")

    fractions = sorted(r["error_frac"] for r in records)
    deltas = sorted(r["error_minus_fork"] for r in records)
    summary = {
        "dump": str(dump_path),
        "wrong_rollouts": wrong_total,
        "labeled": len(records),
        "coverage": len(records) / max(wrong_total, 1),
        "sibling_false_positive_rate": sibling_flagged / max(sibling_total, 1),
        "error_frac_median": fractions[len(fractions) // 2] if fractions else None,
        "error_minus_fork_median_chars": deltas[len(deltas) // 2] if deltas else None,
    }
    print(json.dumps(summary), flush=True)
    return summary


def _auc(scores, labels) -> float:
    """Rank-based AUC (Mann-Whitney)."""

    pairs = sorted(zip(scores, labels))
    rank_sum = 0.0
    positives = 0
    for rank, (_, label) in enumerate(pairs, start=1):
        if label == 1:
            rank_sum += rank
            positives += 1
    negatives = len(pairs) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def simulate_adaptive_sc(dump_path: str | Path, output_path: str | Path) -> dict:
    """Simulate agreement-based early stopping over existing latent-SC rollouts.

    Detector-free precursor to latent rollback (roadmap 5c): draw the already-
    sampled rollouts sequentially and stop as soon as any answer has k_agree
    votes; otherwise fall back to majority over all samples. Reports accuracy
    and mean rollouts consumed per policy — the compute/accuracy frontier for
    adaptive internal sampling, measured with zero new inference.
    """

    from collections import Counter

    dump_path = Path(dump_path)
    sc_key = None
    problems = []
    for line in dump_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if sc_key is None:
            sc_key = next((k for k in row if k.startswith("latent_sc")), None)
            if sc_key is None:
                raise ValueError(f"No latent_sc* field found in {dump_path}")
        if sc_key not in row:
            continue
        answers = [_lenient_answer(text) for text in json.loads(row[sc_key])]
        problems.append((row["gold"], answers))

    def run_policy(k_agree: int) -> dict:
        correct = 0
        used_total = 0
        for gold, answers in problems:
            votes: Counter = Counter()
            used = 0
            decided = None
            for answer in answers:
                used += 1
                if answer is not None:
                    votes[answer] += 1
                    if votes[answer] >= k_agree:
                        decided = answer
                        break
            if decided is None:
                decided = votes.most_common(1)[0][0] if votes else None
            used_total += used
            correct += decided == gold
        return {
            "policy": f"stop_at_{k_agree}_agreeing",
            "accuracy": correct / len(problems),
            "mean_rollouts": used_total / len(problems),
        }

    total = len(problems[0][1]) if problems else 0
    rows = [run_policy(k) for k in (1, 2, 3, 4)]
    baseline_correct = sum(
        (Counter(a for a in answers if a is not None).most_common(1) or [(None, 0)])[0][0] == gold
        for gold, answers in problems
    )
    rows.append(
        {"policy": f"full_majority@{total}", "accuracy": baseline_correct / len(problems), "mean_rollouts": float(total)}
    )

    lines = [
        "# Adaptive latent self-consistency simulation",
        "",
        f"Dump: `{dump_path}`, problems: {len(problems)}, samples available: {total}",
        "",
        "Sequential draw over existing rollouts; stop when k answers agree.",
        "",
        "| policy | accuracy | mean internal rollouts |",
        "| --- | --- | --- |",
    ]
    for row in rows:
        lines.append(f"| {row['policy']} | {row['accuracy']:.4f} | {row['mean_rollouts']:.2f} |")
        print(json.dumps(row), flush=True)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"problems": len(problems), "policies": rows, "report": str(output_path)}


def oracle_rollback(
    model_name: str,
    corrector_path: str | Path,
    dump_path: str | Path,
    output_path: str | Path,
    device_str: str = "cuda",
    budget: int = 4,
    temperature: float = 0.6,
    rewind_margin_tokens: int = 8,
    max_new_tokens: int = 512,
    split: str = "test",
) -> dict:
    """Oracle-ceiling rollback experiment (roadmap 5c).

    For each problem whose greedy latent rollout is wrong, locate the
    arithmetic error site (oracle localization via gold `<<>>` annotations),
    rewind the rollout to rewind_margin_tokens before the error, and resample
    the continuation up to `budget` times at `temperature` ΓÇö the corrector
    state warm-started over the shared prefix, exactly as a KV-cache
    truncation would at inference.

    Reports two acceptance rules per re-roll set:
    - ceiling: any re-roll reaches the gold answer (upper bound on rollback);
    - detector-accepted: first re-roll that passes the same unsanctioned-value
      detector (deployable rule, no gold answer needed), scored for accuracy.
    """

    import torch

    from datasets import load_dataset

    from prometheus.retrofit import (
        COT_PROMPT,
        HiddenDeltaCorrector,
        _chat_prompt,
        _generate_with_corrector,
        load_trunk,
    )

    device = torch.device(device_str if device_str != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model, tokenizer = load_trunk(model_name, device, dtype)
    checkpoint = torch.load(corrector_path, map_location=device, weights_only=True)
    cfg = checkpoint["config"]
    corrector = HiddenDeltaCorrector(
        d_model=cfg["d_model"], d_cfc=cfg["d_cfc"], cell=cfg.get("cell", "cfc")
    ).to(device=device, dtype=torch.float32)
    corrector.load_state_dict(checkpoint["corrector_state"])
    corrector.eval()
    tap_layer = cfg["tap_layer"]

    gsm8k = load_dataset("openai/gsm8k", "main", split=split)

    total = 0
    already_correct = 0
    wrong_no_site = 0
    attempted = 0
    ceiling_recovered = 0
    detector_correct = 0
    detector_accepted = 0
    reroll_tokens_total = 0
    rerolls_total = 0

    for line in Path(dump_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "latent" not in row:
            continue
        total += 1
        gold = row["gold"]
        text = row["latent"]
        if _lenient_answer(text) == gold:
            already_correct += 1
            continue
        question = row["question"]
        allowed = _allowed_values(question, gsm8k[row["index"]]["answer"], gold)
        site = _first_error_site(text, allowed)
        if site is None:
            wrong_no_site += 1
            continue
        attempted += 1
        prompt = _chat_prompt(tokenizer, COT_PROMPT + question)
        prefix_ids = tokenizer(text[:site], add_special_tokens=False)["input_ids"]
        keep = max(0, len(prefix_ids) - rewind_margin_tokens)
        prefix = tokenizer.decode(prefix_ids[:keep], skip_special_tokens=True)

        any_correct = False
        detector_pick: str | None = None
        for _ in range(budget):
            continuation = _generate_with_corrector(
                model, tokenizer, corrector, tap_layer, prompt, max_new_tokens, device,
                temperature=temperature, prefix_text=prefix,
            )
            rerolls_total += 1
            reroll_tokens_total += len(tokenizer(continuation, add_special_tokens=False)["input_ids"])
            candidate = prefix + continuation
            if _lenient_answer(candidate) == gold:
                any_correct = True
            if detector_pick is None and _first_error_site(candidate, allowed) is None:
                detector_pick = candidate
        ceiling_recovered += any_correct
        if detector_pick is not None:
            detector_accepted += 1
            detector_correct += _lenient_answer(detector_pick) == gold
        if attempted % 10 == 0:
            print(
                json.dumps(
                    {
                        "attempted": attempted,
                        "ceiling_rate": round(ceiling_recovered / attempted, 4),
                        "detector_acc": round(detector_correct / max(detector_accepted, 1), 4),
                    }
                ),
                flush=True,
            )

    wrong = total - already_correct
    base_acc = already_correct / max(total, 1)
    ceiling_acc = (already_correct + ceiling_recovered) / max(total, 1)
    detector_acc_overall = (already_correct + detector_correct) / max(total, 1)
    summary = {
        "problems": total,
        "greedy_latent_accuracy": base_acc,
        "wrong": wrong,
        "wrong_without_error_site": wrong_no_site,
        "rollback_attempted": attempted,
        "ceiling_recovered": ceiling_recovered,
        "ceiling_accuracy": ceiling_acc,
        "detector_accepted": detector_accepted,
        "detector_accepted_correct": detector_correct,
        "detector_rule_accuracy": detector_acc_overall,
        "mean_rerolls_per_problem": rerolls_total / max(total, 1),
        "mean_reroll_tokens_per_problem": reroll_tokens_total / max(total, 1),
    }
    print(json.dumps(summary), flush=True)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Oracle rollback experiment",
        "",
        f"Model: `{model_name}`, dump: `{dump_path}`, budget: {budget}, "
        f"temperature: {temperature}, rewind margin: {rewind_margin_tokens} tokens",
        "",
        "Wrong greedy latent rollouts are rewound to just before the arithmetic",
        "error site (oracle localization) and resampled with a warm-started",
        "corrector state over the shared prefix.",
        "",
        "| metric | value |",
        "| --- | --- |",
        f"| problems | {total} |",
        f"| greedy latent accuracy | {base_acc:.4f} |",
        f"| wrong rollouts | {wrong} |",
        f"| wrong without detectable error site | {wrong_no_site} |",
        f"| rollback attempted | {attempted} |",
        f"| ceiling: recovered (any re-roll correct) | {ceiling_recovered} |",
        f"| **ceiling accuracy** | **{ceiling_acc:.4f}** |",
        f"| detector-accepted re-rolls | {detector_accepted} |",
        f"| detector-accepted correct | {detector_correct} |",
        f"| **detector-rule accuracy** | **{detector_acc_overall:.4f}** |",
        f"| mean re-rolls per problem (all problems) | {rerolls_total / max(total, 1):.2f} |",
        f"| mean re-roll tokens per problem | {reroll_tokens_total / max(total, 1):.1f} |",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def probe_divergence(
    model_name: str,
    corrector_path: str | Path,
    dump_path: str | Path,
    output_path: str | Path,
    device_str: str = "cuda",
    max_pairs: int = 400,
    offsets: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64),
    anchor: str = "fork",
) -> dict:
    """Measure the divergence recognition curve (roadmap 5a/5c).

    For each (wrong rollout, correct sibling) pair, teacher-force both texts
    through the frozen trunk + corrector, extract h_tap and CfC state s_t at
    k tokens past the anchor point, and fit a logistic probe per offset k.
    anchor="fork" uses the token-level sampling fork; anchor="error" uses the
    first arithmetically-unsanctioned computed value (see label_error_sites),
    with the correct sibling probed at the matched token position. The
    resulting AUC-vs-tokens-past-anchor curve quantifies how quickly the
    latent state betrays a derailment — the design parameter for a rollback
    trigger.
    """

    import torch

    from prometheus.retrofit import COT_PROMPT, HiddenDeltaCorrector, _chat_prompt, load_trunk

    device = torch.device(device_str if device_str != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model, tokenizer = load_trunk(model_name, device, dtype)
    checkpoint = torch.load(corrector_path, map_location=device, weights_only=True)
    cfg = checkpoint["config"]
    tap_layer = cfg["tap_layer"]
    corrector = HiddenDeltaCorrector(
        d_model=cfg["d_model"], d_cfc=cfg["d_cfc"], cell=cfg.get("cell", "cfc")
    ).to(device=device, dtype=torch.float32)
    corrector.load_state_dict(checkpoint["corrector_state"])
    corrector.eval()

    features: dict[int, list] = {k: [] for k in offsets}  # offset -> [(problem, label, h_tap, s_t)]
    pairs_used = 0
    skipped_unlabeled = 0

    gsm8k = None
    if anchor == "error":
        from datasets import load_dataset

        gsm8k = load_dataset("openai/gsm8k", "main", split="test")

    @torch.no_grad()
    def collect(problem: int, label: int, prompt_ids: list[int], text: str, fork_tok: int) -> None:
        text_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        batch = torch.tensor([prompt_ids + text_ids], device=device)
        outputs = model(batch, output_hidden_states=True)
        h_tap_seq = outputs.hidden_states[tap_layer][0, len(prompt_ids) - 1 : -1, :].float()
        state = corrector.initial_state(1, device)
        states = []
        for t in range(h_tap_seq.size(0)):
            _, state = corrector.step(h_tap_seq[t : t + 1], state)
            states.append(state[0] if isinstance(state, tuple) else state)
        for k in offsets:
            position = fork_tok + k - 1
            if position >= h_tap_seq.size(0):
                continue
            features[k].append(
                (
                    problem,
                    label,
                    h_tap_seq[position].cpu(),
                    states[position].reshape(-1).cpu(),
                )
            )

    for index, question, _gold, wrong, sibling, _onset in _iter_pairs(Path(dump_path)):
        if pairs_used >= max_pairs:
            break
        prompt = _chat_prompt(tokenizer, COT_PROMPT + question)
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        if anchor == "error":
            allowed = _allowed_values(question, gsm8k[index]["answer"], _gold)
            site = _first_error_site(wrong, allowed)
            if site is None:
                skipped_unlabeled += 1
                continue
            anchor_tok = len(tokenizer(wrong[:site], add_special_tokens=False)["input_ids"])
        else:
            wrong_ids = tokenizer(wrong, add_special_tokens=False)["input_ids"]
            sibling_ids = tokenizer(sibling, add_special_tokens=False)["input_ids"]
            anchor_tok = 0
            for a, b in zip(wrong_ids, sibling_ids):
                if a != b:
                    break
                anchor_tok += 1
        collect(index, 1, prompt_ids, wrong, anchor_tok)
        collect(index, 0, prompt_ids, sibling, anchor_tok)
        pairs_used += 1
        if pairs_used % 50 == 0:
            print(json.dumps({"pairs": pairs_used}), flush=True)

    def fit_auc(rows: list, feature: str) -> float:
        """5-fold CV logistic probe; returns mean held-out AUC."""

        import torch as t

        problems = sorted({r[0] for r in rows})
        folds = {p: i % 5 for i, p in enumerate(problems)}
        aucs = []
        for fold in range(5):
            train = [r for r in rows if folds[r[0]] != fold]
            test = [r for r in rows if folds[r[0]] == fold]
            if not train or not test or len({r[1] for r in test}) < 2:
                continue
            pick = (lambda r: r[2]) if feature == "h_tap" else (lambda r: r[3]) if feature == "s_t" else (lambda r: t.cat([r[2], r[3]]))
            x_train = t.stack([pick(r) for r in train])
            y_train = t.tensor([float(r[1]) for r in train])
            mean, std = x_train.mean(0), x_train.std(0) + 1e-6
            x_train = (x_train - mean) / std
            weight = t.zeros(x_train.size(1), requires_grad=True)
            bias = t.zeros(1, requires_grad=True)
            optimizer = t.optim.Adam([weight, bias], lr=0.05)
            for _ in range(300):
                loss = t.nn.functional.binary_cross_entropy_with_logits(
                    x_train @ weight + bias, y_train
                ) + 1e-3 * weight.pow(2).sum()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            with t.no_grad():
                x_test = (t.stack([pick(r) for r in test]) - mean) / std
                scores = (x_test @ weight + bias).tolist()
            aucs.append(_auc(scores, [r[1] for r in test]))
        valid = [a for a in aucs if a == a]
        return sum(valid) / len(valid) if valid else float("nan")

    curve = []
    for k in offsets:
        rows = features[k]
        entry = {"offset": k, "n": len(rows)}
        for feature in ("h_tap", "s_t", "both"):
            entry[f"auc_{feature}"] = fit_auc(rows, feature) if len(rows) >= 40 else None
        curve.append(entry)
        print(json.dumps(entry), flush=True)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Divergence recognition curve",
        "",
        f"Model: `{model_name}`, corrector: `{corrector_path}`, pairs: {pairs_used}, anchor: {anchor}"
        + (f" (skipped {skipped_unlabeled} unlabeled)" if anchor == "error" else ""),
        "",
        "Probe: 5-fold CV logistic regression, wrong rollout vs correct sibling,",
        f"features taken k tokens past the {'arithmetic error site' if anchor == 'error' else 'token-level fork point'}.",
        "",
        "| tokens past anchor | n | AUC h_tap | AUC s_t | AUC both |",
        "| --- | --- | --- | --- | --- |",
    ]
    for entry in curve:
        fmt = lambda v: f"{v:.3f}" if isinstance(v, float) and v == v else "-"
        lines.append(
            f"| {entry['offset']} | {entry['n']} | {fmt(entry['auc_h_tap'])} | {fmt(entry['auc_s_t'])} | {fmt(entry['auc_both'])} |"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary = {"pairs": pairs_used, "curve": curve, "report": str(output_path)}
    return summary


def probe_rollback(
    model_name: str,
    corrector_path: str | Path,
    dump_path: str | Path,
    output_path: str | Path,
    device_str: str = "cuda",
    budget: int = 4,
    temperature: float = 0.6,
    rewind_margin_tokens: int = 8,
    max_new_tokens: int = 512,
    threshold_fpr: float = 0.05,
    split: str = "test",
) -> dict:
    """Deployable probe-triggered rollback (roadmap 5c, closes the oracle gap).

    No gold labels at inference. A logistic probe on h_tap — trained on
    error-site-labeled sibling pairs from problems whose GREEDY rollout is
    correct (problem-disjoint from every rollback target) — scans each
    greedy rollout token by token. First score above threshold triggers a
    rewind to rewind_margin_tokens before the trigger and up to `budget`
    warm-started re-rolls; the first candidate whose rescan stays below
    threshold is accepted, else the original text is kept. The threshold is
    calibrated so at most `threshold_fpr` of correct training rollouts would
    trigger (rollout-level max-score quantile).
    """

    import torch

    from datasets import load_dataset

    from prometheus.retrofit import (
        COT_PROMPT,
        HiddenDeltaCorrector,
        _chat_prompt,
        _generate_with_corrector,
        extract_answer_lenient,
        load_trunk,
    )

    device = torch.device(device_str if device_str != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model, tokenizer = load_trunk(model_name, device, dtype)
    checkpoint = torch.load(corrector_path, map_location=device, weights_only=True)
    cfg = checkpoint["config"]
    tap_layer = cfg["tap_layer"]
    corrector = HiddenDeltaCorrector(
        d_model=cfg["d_model"], d_cfc=cfg["d_cfc"], cell=cfg.get("cell", "cfc")
    ).to(device=device, dtype=torch.float32)
    corrector.load_state_dict(checkpoint["corrector_state"])
    corrector.eval()

    gsm8k = load_dataset("openai/gsm8k", "main", split=split)

    rows = []
    for line in Path(dump_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "latent" in row:
            rows.append(row)
    greedy_correct = {row["index"] for row in rows if extract_answer_lenient(row["latent"]) == row["gold"]}

    @torch.no_grad()
    def h_tap_sequence(prompt_ids: list[int], text: str):
        text_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        batch = torch.tensor([prompt_ids + text_ids], device=device)
        outputs = model(batch, output_hidden_states=True)
        return outputs.hidden_states[tap_layer][0, len(prompt_ids) - 1 : -1, :].float(), text_ids

    # ---- Phase 1: train the trigger probe on greedy-correct problems only.
    train_x, train_y = [], []
    calibration_rollouts = []  # h_tap sequences of correct siblings (for threshold)
    train_pairs = 0
    for index, question, gold, wrong, sibling, _onset in _iter_pairs(Path(dump_path)):
        if index not in greedy_correct:
            continue
        allowed = _allowed_values(question, gsm8k[index]["answer"], gold)
        site = _first_error_site(wrong, allowed)
        if site is None:
            continue
        prompt = _chat_prompt(tokenizer, COT_PROMPT + question)
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        anchor_tok = len(tokenizer(wrong[:site], add_special_tokens=False)["input_ids"])
        wrong_seq, _ = h_tap_sequence(prompt_ids, wrong)
        sibling_seq, _ = h_tap_sequence(prompt_ids, sibling)
        for k in (1, 2, 3, 4):
            position = anchor_tok + k - 1
            if position < wrong_seq.size(0):
                train_x.append(wrong_seq[position].cpu())
                train_y.append(1.0)
            if position < sibling_seq.size(0):
                train_x.append(sibling_seq[position].cpu())
                train_y.append(0.0)
        calibration_rollouts.append(sibling_seq.cpu())
        train_pairs += 1
        if train_pairs % 25 == 0:
            print(json.dumps({"probe_pairs": train_pairs}), flush=True)

    x = torch.stack(train_x)
    y = torch.tensor(train_y)
    mean, std = x.mean(0), x.std(0) + 1e-6
    x = (x - mean) / std
    weight = torch.zeros(x.size(1), requires_grad=True)
    bias = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.Adam([weight, bias], lr=0.05)
    for _ in range(300):
        loss = torch.nn.functional.binary_cross_entropy_with_logits(x @ weight + bias, y) + 1e-3 * weight.pow(2).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    weight = weight.detach()
    bias = bias.detach()

    def scan(seq_cpu: "torch.Tensor") -> "torch.Tensor":
        return ((seq_cpu - mean) / std) @ weight + bias

    max_scores = torch.stack([scan(seq).max() for seq in calibration_rollouts])
    threshold = torch.quantile(max_scores, 1.0 - threshold_fpr).item()
    print(json.dumps({"probe_pairs": train_pairs, "train_rows": len(train_y), "threshold": round(threshold, 4)}), flush=True)

    # ---- Phase 2: deployable pipeline over every problem's greedy rollout.
    total = 0
    baseline_correct = 0
    final_correct = 0
    triggered_on_correct = 0
    triggered_on_wrong = 0
    accepted_rerolls = 0
    flips_up = 0
    flips_down = 0
    rerolls_total = 0
    reroll_tokens_total = 0

    for row in rows:
        total += 1
        gold = row["gold"]
        text = row["latent"]
        was_correct = extract_answer_lenient(text) == gold
        baseline_correct += was_correct
        question = row["question"]
        prompt = _chat_prompt(tokenizer, COT_PROMPT + question)
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        seq, text_ids = h_tap_sequence(prompt_ids, text)
        scores = scan(seq.cpu())
        above = (scores > threshold).nonzero()
        final_text = text
        if above.numel() > 0:
            if was_correct:
                triggered_on_correct += 1
            else:
                triggered_on_wrong += 1
            trigger_pos = int(above[0])
            keep = max(0, trigger_pos - rewind_margin_tokens)
            prefix = tokenizer.decode(text_ids[:keep], skip_special_tokens=True)
            for _ in range(budget):
                continuation = _generate_with_corrector(
                    model, tokenizer, corrector, tap_layer, prompt, max_new_tokens, device,
                    temperature=temperature, prefix_text=prefix,
                )
                rerolls_total += 1
                reroll_tokens_total += len(tokenizer(continuation, add_special_tokens=False)["input_ids"])
                candidate = prefix + continuation
                candidate_seq, _ = h_tap_sequence(prompt_ids, candidate)
                if scan(candidate_seq.cpu()).max().item() <= threshold:
                    final_text = candidate
                    accepted_rerolls += 1
                    break
        now_correct = extract_answer_lenient(final_text) == gold
        final_correct += now_correct
        flips_up += (not was_correct) and now_correct
        flips_down += was_correct and (not now_correct)
        if total % 20 == 0:
            print(
                json.dumps({"scanned": total, "baseline": round(baseline_correct / total, 4), "final": round(final_correct / total, 4)}),
                flush=True,
            )

    summary = {
        "problems": total,
        "probe_train_pairs": train_pairs,
        "threshold": threshold,
        "baseline_accuracy": baseline_correct / max(total, 1),
        "final_accuracy": final_correct / max(total, 1),
        "triggered_on_correct": triggered_on_correct,
        "triggered_on_wrong": triggered_on_wrong,
        "accepted_rerolls": accepted_rerolls,
        "flips_up": flips_up,
        "flips_down": flips_down,
        "mean_rerolls_per_problem": rerolls_total / max(total, 1),
        "mean_reroll_tokens_per_problem": reroll_tokens_total / max(total, 1),
    }
    print(json.dumps(summary), flush=True)

    wrong_count = total - baseline_correct
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Probe-triggered rollback (deployable rule)",
        "",
        f"Model: `{model_name}`, dump: `{dump_path}`, budget: {budget}, temperature: {temperature}, "
        f"rewind margin: {rewind_margin_tokens} tokens, calibrated rollout FPR: {threshold_fpr}",
        "",
        "Trigger: logistic probe on h_tap, trained on error-site sibling pairs from",
        "problems whose greedy rollout is correct (problem-disjoint from all rollback",
        "targets). No gold labels are used at inference time.",
        "",
        "| metric | value |",
        "| --- | --- |",
        f"| problems | {total} |",
        f"| probe training pairs | {train_pairs} |",
        f"| baseline greedy accuracy | {baseline_correct / max(total, 1):.4f} |",
        f"| **final accuracy** | **{final_correct / max(total, 1):.4f}** |",
        f"| wrong rollouts | {wrong_count} |",
        f"| triggered on wrong (recall) | {triggered_on_wrong}/{wrong_count} |",
        f"| triggered on correct (false alarms) | {triggered_on_correct}/{baseline_correct} |",
        f"| re-rolls accepted | {accepted_rerolls} |",
        f"| flips wrong→correct | {flips_up} |",
        f"| flips correct→wrong | {flips_down} |",
        f"| mean re-rolls per problem | {rerolls_total / max(total, 1):.2f} |",
        f"| mean re-roll tokens per problem | {reroll_tokens_total / max(total, 1):.1f} |",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def steer_inject(
    model_name: str,
    corrector_path: str | Path,
    dump_path: str | Path,
    output_path: str | Path,
    device_str: str = "cuda",
    trigger: str = "oracle",
    alphas: tuple[float, ...] = (0.0, 1.0, 2.0, 4.0),
    steer_window: int = 24,
    inject_offset: int = 4,
    max_new_tokens: int = 512,
    threshold_fpr: float = 0.20,
    gate: str = "none",
    split: str = "test",
) -> dict:
    """Steering-vector repair: inject a 'wait-' correction instead of rolling back.

    Alternative to rollback (roadmap 5c): keep the erroneous tokens in
    context and add a contrastive repair direction
    v = mean(h_tap[correct sibling] - h_tap[wrong]) at error-site offsets
    +1..+4 (deployable supervision: pairs drawn only from problems whose
    greedy rollout is correct) to the tap layer's residual stream for
    `steer_window` decode steps after the trigger, then regenerate the
    suffix greedily. Because the original rollout was greedy, alpha=0
    reproduces it exactly — any answer change at alpha>0 is causally the
    injected vector. trigger="oracle" fires at annotation error sites
    (mechanism ceiling, comparable to oracle rollback); trigger="probe"
    uses the deployable h_tap probe scanner, where false alarms are cheap
    by construction so the threshold can be relaxed (default 20% FPR vs
    rollback's 5%); trigger="cusum" runs the same probe through the
    trigger-lab winner — a per-rollout self-normalized 8-token one-sided
    CUSUM — which nearly doubles recall at matched FPR with ~6-token
    median delay, inside the measured repair window. gate="margin" scales
    alpha per rollout by how far the trigger statistic clears the
    threshold (normalized by the calibration tail), so borderline
    triggers — the population false alarms live in — receive a weaker
    push than confident detections: a deployment-legal precision lever.
    """

    import torch

    from datasets import load_dataset

    from prometheus.retrofit import (
        COT_PROMPT,
        HiddenDeltaCorrector,
        _chat_prompt,
        _generate_with_corrector,
        extract_answer_lenient,
        load_trunk,
    )

    device = torch.device(device_str if device_str != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model, tokenizer = load_trunk(model_name, device, dtype)
    checkpoint = torch.load(corrector_path, map_location=device, weights_only=True)
    cfg = checkpoint["config"]
    tap_layer = cfg["tap_layer"]
    corrector = HiddenDeltaCorrector(
        d_model=cfg["d_model"], d_cfc=cfg["d_cfc"], cell=cfg.get("cell", "cfc")
    ).to(device=device, dtype=torch.float32)
    corrector.load_state_dict(checkpoint["corrector_state"])
    corrector.eval()

    gsm8k = load_dataset("openai/gsm8k", "main", split=split)

    rows = []
    for line in Path(dump_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "latent" in row:
            rows.append(row)
    greedy_correct = {row["index"] for row in rows if extract_answer_lenient(row["latent"]) == row["gold"]}

    @torch.no_grad()
    def h_tap_sequence(prompt_ids: list[int], text: str):
        text_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        batch = torch.tensor([prompt_ids + text_ids], device=device)
        outputs = model(batch, output_hidden_states=True)
        return outputs.hidden_states[tap_layer][0, len(prompt_ids) - 1 : -1, :].float(), text_ids

    # ---- Phase 1: steering vector (and probe, if needed) from greedy-correct
    # problems only — same deployable supervision as probe_rollback.
    vec_sum = None
    vec_count = 0
    tap_norm_sum = 0.0
    tap_norm_count = 0
    train_x, train_y = [], []
    calibration_rollouts = []
    train_pairs = 0
    for index, question, gold, wrong, sibling, _onset in _iter_pairs(Path(dump_path)):
        if index not in greedy_correct:
            continue
        allowed = _allowed_values(question, gsm8k[index]["answer"], gold)
        site = _first_error_site(wrong, allowed)
        if site is None:
            continue
        prompt = _chat_prompt(tokenizer, COT_PROMPT + question)
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        anchor_tok = len(tokenizer(wrong[:site], add_special_tokens=False)["input_ids"])
        wrong_seq, _ = h_tap_sequence(prompt_ids, wrong)
        sibling_seq, _ = h_tap_sequence(prompt_ids, sibling)
        for k in (1, 2, 3, 4):
            position = anchor_tok + k - 1
            if position < wrong_seq.size(0) and position < sibling_seq.size(0):
                diff = sibling_seq[position] - wrong_seq[position]
                vec_sum = diff if vec_sum is None else vec_sum + diff
                vec_count += 1
                tap_norm_sum += wrong_seq[position].norm().item()
                tap_norm_count += 1
            if position < wrong_seq.size(0):
                train_x.append(wrong_seq[position].cpu())
                train_y.append(1.0)
            if position < sibling_seq.size(0):
                train_x.append(sibling_seq[position].cpu())
                train_y.append(0.0)
        calibration_rollouts.append(sibling_seq.cpu())
        train_pairs += 1
        if train_pairs % 25 == 0:
            print(json.dumps({"steer_pairs": train_pairs}), flush=True)

    steer_vec = (vec_sum / max(vec_count, 1)).to(device)
    tap_norm_mean = tap_norm_sum / max(tap_norm_count, 1)
    vec_norm = steer_vec.norm().item()
    print(
        json.dumps(
            {
                "steer_pairs": train_pairs,
                "vec_positions": vec_count,
                "vec_norm": round(vec_norm, 3),
                "mean_tap_norm": round(tap_norm_mean, 3),
                "vec_to_tap_ratio": round(vec_norm / max(tap_norm_mean, 1e-6), 4),
            }
        ),
        flush=True,
    )

    threshold = None
    scan = None
    trigger_stat = None
    if trigger in ("probe", "cusum"):
        x = torch.stack(train_x)
        y = torch.tensor(train_y)
        mean, std = x.mean(0), x.std(0) + 1e-6
        x = (x - mean) / std
        weight = torch.zeros(x.size(1), requires_grad=True)
        bias = torch.zeros(1, requires_grad=True)
        optimizer = torch.optim.Adam([weight, bias], lr=0.05)
        for _ in range(300):
            loss = torch.nn.functional.binary_cross_entropy_with_logits(x @ weight + bias, y) + 1e-3 * weight.pow(2).sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        weight = weight.detach()
        bias = bias.detach()

        def scan(seq_cpu: "torch.Tensor") -> "torch.Tensor":
            return ((seq_cpu - mean) / std) @ weight + bias

        def trigger_stat(seq_cpu: "torch.Tensor") -> "torch.Tensor":
            scores = scan(seq_cpu)
            if trigger == "probe":
                return scores
            z = (scores - scores.mean()) / (scores.std() + 1e-6)
            window = 8
            padded = torch.nn.functional.pad(z.clamp(min=0.0).unsqueeze(0).unsqueeze(0), (window - 1, 0))
            return torch.nn.functional.avg_pool1d(padded, window, stride=1).squeeze() * window

        max_scores = torch.stack([trigger_stat(seq).max() for seq in calibration_rollouts])
        threshold = torch.quantile(max_scores, 1.0 - threshold_fpr).item()
        gate_scale = max(torch.quantile(max_scores, 0.95).item() - threshold, 1e-6)
        print(json.dumps({"threshold": round(threshold, 4), "threshold_fpr": threshold_fpr, "gate_scale": round(gate_scale, 4)}), flush=True)

    # ---- Steering hook: add alpha*v to the residual stream at the tap layer
    # (last position only) for the first steer_window forwards of a rollout.
    steer_state = {"vec": None, "remaining": 0}
    layer_module = model.model.layers[tap_layer - 1]

    def _steer_hook(_module, _inputs, output):
        if steer_state["remaining"] <= 0 or steer_state["vec"] is None:
            return output
        hidden = output[0] if isinstance(output, tuple) else output
        hidden = hidden.clone()
        hidden[:, -1, :] += steer_state["vec"].to(hidden.dtype)
        steer_state["remaining"] -= 1
        if isinstance(output, tuple):
            return (hidden,) + tuple(output[1:])
        return hidden

    hook_handle = layer_module.register_forward_hook(_steer_hook)

    # ---- Phase 2: find trigger positions, then regenerate per alpha.
    targets = []  # (row, trigger_tok, was_correct, gate_value)
    total = 0
    baseline_correct = 0
    no_site = 0
    no_room = 0
    triggered_on_correct = 0
    triggered_on_wrong = 0
    gates_on_correct = []
    gates_on_wrong = []
    for row in rows:
        total += 1
        gold = row["gold"]
        text = row["latent"]
        was_correct = extract_answer_lenient(text) == gold
        baseline_correct += was_correct
        question = row["question"]
        prompt = _chat_prompt(tokenizer, COT_PROMPT + question)
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        if trigger == "oracle":
            if was_correct:
                continue
            allowed = _allowed_values(question, gsm8k[row["index"]]["answer"], gold)
            site = _first_error_site(text, allowed)
            if site is None:
                no_site += 1
                continue
            trigger_tok = len(tokenizer(text[:site], add_special_tokens=False)["input_ids"])
            gate_value = 1.0
        else:
            seq, _ = h_tap_sequence(prompt_ids, text)
            stat = trigger_stat(seq.cpu())
            above = (stat > threshold).nonzero()
            if above.numel() == 0:
                continue
            trigger_tok = int(above[0])
            gate_value = min(1.0, (stat.max().item() - threshold) / gate_scale) if gate == "margin" else 1.0
            if was_correct:
                triggered_on_correct += 1
                gates_on_correct.append(gate_value)
            else:
                triggered_on_wrong += 1
                gates_on_wrong.append(gate_value)
        text_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        keep = trigger_tok + inject_offset
        if keep >= len(text_ids):
            no_room += 1
            continue
        targets.append((row, keep, was_correct, gate_value))

    wrong_count = total - baseline_correct
    print(
        json.dumps(
            {
                "problems": total,
                "baseline": round(baseline_correct / max(total, 1), 4),
                "targets": len(targets),
                "no_site": no_site,
                "no_room": no_room,
                "triggered_on_correct": triggered_on_correct,
                "triggered_on_wrong": triggered_on_wrong,
                "mean_gate_false_alarms": round(sum(gates_on_correct) / len(gates_on_correct), 4) if gates_on_correct else None,
                "mean_gate_hits": round(sum(gates_on_wrong) / len(gates_on_wrong), 4) if gates_on_wrong else None,
            }
        ),
        flush=True,
    )

    results = []
    for alpha in alphas:
        steered_correct = 0
        flips_up = 0
        flips_down = 0
        reproduced = 0
        regen_tokens = 0
        for row, keep, was_correct, gate_value in targets:
            gold = row["gold"]
            text = row["latent"]
            question = row["question"]
            prompt = _chat_prompt(tokenizer, COT_PROMPT + question)
            text_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            prefix = tokenizer.decode(text_ids[:keep], skip_special_tokens=True)
            steer_state["vec"] = steer_vec * (alpha * gate_value)
            steer_state["remaining"] = steer_window if alpha != 0.0 else 0
            continuation = _generate_with_corrector(
                model, tokenizer, corrector, tap_layer, prompt, max_new_tokens, device,
                temperature=0.0, prefix_text=prefix,
            )
            steer_state["vec"] = None
            steer_state["remaining"] = 0
            regen_tokens += len(tokenizer(continuation, add_special_tokens=False)["input_ids"])
            candidate = prefix + continuation
            now_correct = extract_answer_lenient(candidate) == gold
            steered_correct += now_correct
            reproduced += candidate.strip() == text.strip()
            flips_up += (not was_correct) and now_correct
            flips_down += was_correct and (not now_correct)
        final_correct = baseline_correct + flips_up - flips_down
        entry = {
            "alpha": alpha,
            "final_accuracy": round(final_correct / max(total, 1), 4),
            "flips_up": flips_up,
            "flips_down": flips_down,
            "reproduced_exactly": reproduced,
            "regen_tokens_per_problem": round(regen_tokens / max(total, 1), 1),
        }
        results.append(entry)
        print(json.dumps(entry), flush=True)

    hook_handle.remove()

    summary = {
        "problems": total,
        "baseline_accuracy": baseline_correct / max(total, 1),
        "wrong": wrong_count,
        "trigger": trigger,
        "targets": len(targets),
        "steer_pairs": train_pairs,
        "vec_norm": vec_norm,
        "mean_tap_norm": tap_norm_mean,
        "steer_window": steer_window,
        "inject_offset": inject_offset,
        "results": results,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Steering-vector repair (inject, don't rewind)",
        "",
        f"Model: `{model_name}`, dump: `{dump_path}`, trigger: {trigger}, "
        f"steer window: {steer_window} tokens, inject offset: {inject_offset} tokens"
        + (f", rollout FPR: {threshold_fpr}, gate: {gate}" if trigger != "oracle" else ""),
        "",
        "v = mean(h_tap[correct sibling] − h_tap[wrong]) at error-site offsets +1..+4,",
        f"from {train_pairs} greedy-correct-problem pairs ({vec_count} positions).",
        f"‖v‖ = {vec_norm:.2f}, mean ‖h_tap‖ = {tap_norm_mean:.2f} "
        f"(ratio {vec_norm / max(tap_norm_mean, 1e-6):.4f}).",
        "The erroneous tokens stay in context; alpha·v is added to the tap layer's",
        "residual stream for the steer window, then greedy decoding continues.",
        "alpha=0 is the determinism control (must reproduce the original rollout).",
        "",
        f"Problems: {total}, baseline greedy accuracy: {baseline_correct / max(total, 1):.4f}, "
        f"wrong: {wrong_count}, steer targets: {len(targets)}"
        + (f" (no site: {no_site}, no room: {no_room})" if trigger == "oracle" else
           f" (false alarms: {triggered_on_correct}, on wrong: {triggered_on_wrong}, no room: {no_room}"
           + (f", mean gate false-alarm/hit: {sum(gates_on_correct) / max(len(gates_on_correct), 1):.3f}/"
              f"{sum(gates_on_wrong) / max(len(gates_on_wrong), 1):.3f}" if gate == "margin" else "")
           + ")"),
        "",
        "| alpha | final accuracy | flips up | flips down | reproduced exactly | regen tokens/problem |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for entry in results:
        lines.append(
            f"| {entry['alpha']} | **{entry['final_accuracy']:.4f}** | {entry['flips_up']} | "
            f"{entry['flips_down']} | {entry['reproduced_exactly']}/{len(targets)} | {entry['regen_tokens_per_problem']} |"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def trigger_lab(
    model_name: str,
    corrector_path: str | Path,
    dump_path: str | Path,
    output_path: str | Path,
    device_str: str = "cuda",
    split: str = "test",
    basis_path: str | Path | None = None,
) -> dict:
    """Offline detection-rule bake-off (the trigger is the bottleneck).

    Computes every greedy rollout's h_tap sequence once, then evaluates
    candidate trigger rules with no regeneration cost. Rules:
    - probe/sibling: logistic probe trained on error-site vs matched sibling
      positions (the rule that collapsed in probe_rollback);
    - probe/incontext: same positives, but negatives drawn from all other
      positions of the same wrong rollouts plus all sibling positions —
      matches the deployment marginal;
    - meandiff: raw projection onto v = mean(correct − wrong) at error
      sites — the steering vector doubling as the landmark detector;
    - complement-energy / complement-frac (with basis_path): norm and
      energy fraction of h_tap outside the trunk's dominant Jacobian
      subspace — the "intrusive thoughts" hypothesis, that contending
      wrong answers surface as complement excursions;
    each scored raw and per-rollout self-normalized (z-score within the
    rollout, which cancels rollout-level score offsets), plus a windowed
    cumulative-excursion variant (rolling sum of positive z over 8 tokens,
    a CUSUM-style sequential statistic). Metrics per rule: rollout-level
    recall on wrong rollouts at fixed false-alarm rates (calibrated on
    correct greedy rollouts, problem-disjoint from probe training), and
    median trigger delay in tokens past the oracle error site.
    """

    import torch

    from datasets import load_dataset

    from prometheus.retrofit import COT_PROMPT, _chat_prompt, extract_answer_lenient, load_trunk

    device = torch.device(device_str if device_str != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model, tokenizer = load_trunk(model_name, device, dtype)
    checkpoint = torch.load(corrector_path, map_location=device, weights_only=True)
    tap_layer = checkpoint["config"]["tap_layer"]

    gsm8k = load_dataset("openai/gsm8k", "main", split=split)

    rows = []
    for line in Path(dump_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "latent" in row:
            rows.append(row)
    greedy_correct = {row["index"] for row in rows if extract_answer_lenient(row["latent"]) == row["gold"]}

    @torch.no_grad()
    def h_tap_sequence(prompt_ids: list[int], text: str):
        text_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        batch = torch.tensor([prompt_ids + text_ids], device=device)
        outputs = model(batch, output_hidden_states=True)
        return outputs.hidden_states[tap_layer][0, len(prompt_ids) - 1 : -1, :].float().cpu()

    # ---- Phase 1: training material from greedy-correct problems only.
    pos_x = []          # error-site features (offsets +1..+4 in wrong rollouts)
    sib_x = []          # matched sibling positions (sibling-rule negatives)
    ctx_x = []          # in-context negatives (other wrong-rollout + sibling positions)
    train_pairs = 0
    for index, question, gold, wrong, sibling, _onset in _iter_pairs(Path(dump_path)):
        if index not in greedy_correct:
            continue
        allowed = _allowed_values(question, gsm8k[index]["answer"], gold)
        site = _first_error_site(wrong, allowed)
        if site is None:
            continue
        prompt = _chat_prompt(tokenizer, COT_PROMPT + question)
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        anchor_tok = len(tokenizer(wrong[:site], add_special_tokens=False)["input_ids"])
        wrong_seq = h_tap_sequence(prompt_ids, wrong)
        sibling_seq = h_tap_sequence(prompt_ids, sibling)
        site_positions = set()
        for k in (1, 2, 3, 4):
            position = anchor_tok + k - 1
            if position < wrong_seq.size(0):
                pos_x.append(wrong_seq[position])
                site_positions.add(position)
            if position < sibling_seq.size(0):
                sib_x.append(sibling_seq[position])
        stride = 7
        for t in range(0, wrong_seq.size(0), stride):
            if all(abs(t - p) > 8 for p in site_positions):
                ctx_x.append(wrong_seq[t])
        for t in range(0, sibling_seq.size(0), stride):
            ctx_x.append(sibling_seq[t])
        train_pairs += 1
        if train_pairs % 25 == 0:
            print(json.dumps({"lab_pairs": train_pairs}), flush=True)

    def fit_probe(negatives: list) -> tuple:
        x = torch.stack(pos_x + negatives)
        y = torch.tensor([1.0] * len(pos_x) + [0.0] * len(negatives))
        mean, std = x.mean(0), x.std(0) + 1e-6
        x = (x - mean) / std
        weight = torch.zeros(x.size(1), requires_grad=True)
        bias = torch.zeros(1, requires_grad=True)
        optimizer = torch.optim.Adam([weight, bias], lr=0.05)
        # Class weights: in-context negatives heavily outnumber positives.
        w_pos = len(y) / (2.0 * max(y.sum().item(), 1.0))
        w_neg = len(y) / (2.0 * max((1 - y).sum().item(), 1.0))
        sample_w = torch.where(y > 0.5, torch.tensor(w_pos), torch.tensor(w_neg))
        for _ in range(300):
            raw = torch.nn.functional.binary_cross_entropy_with_logits(
                x @ weight + bias, y, weight=sample_w
            ) + 1e-3 * weight.pow(2).sum()
            optimizer.zero_grad()
            raw.backward()
            optimizer.step()
        return weight.detach(), bias.detach(), mean, std

    sib_w, sib_b, sib_mean, sib_std = fit_probe(sib_x)
    ctx_w, ctx_b, ctx_mean, ctx_std = fit_probe(ctx_x)
    v_dir = torch.stack(pos_x).mean(0) - torch.stack(sib_x).mean(0)  # wrong − correct: high = error
    v_dir = v_dir / (v_dir.norm() + 1e-6)
    print(json.dumps({"lab_pairs": train_pairs, "pos": len(pos_x), "sib_neg": len(sib_x), "ctx_neg": len(ctx_x)}), flush=True)

    scorers = {
        "probe-sibling": lambda seq: ((seq - sib_mean) / sib_std) @ sib_w + sib_b,
        "probe-incontext": lambda seq: ((seq - ctx_mean) / ctx_std) @ ctx_w + ctx_b,
        "meandiff-v": lambda seq: seq @ v_dir,
    }

    if basis_path is not None:
        bases = torch.load(basis_path, map_location="cpu", weights_only=True)
        b_full = bases["basis_full"].to(torch.float32)  # (rank, d)

        def _complement(seq):
            return seq - (seq @ b_full.T) @ b_full

        scorers["complement-energy"] = lambda seq: _complement(seq).norm(dim=-1)
        scorers["complement-frac"] = lambda seq: _complement(seq).pow(2).sum(-1) / seq.pow(2).sum(-1).clamp_min(1e-9)

    def variants(scores: "torch.Tensor") -> dict:
        z = (scores - scores.mean()) / (scores.std() + 1e-6)
        window = 8
        padded = torch.nn.functional.pad(z.clamp(min=0.0).unsqueeze(0).unsqueeze(0), (window - 1, 0))
        cusum = torch.nn.functional.avg_pool1d(padded, window, stride=1).squeeze() * window
        out = {"raw": scores, "selfnorm": z, "cusum8": cusum}
        # Periodic trajectory audit: sample a windowed mean drift statistic
        # every `stride` tokens — ~T/stride comparisons instead of T, which
        # lowers the calibrated threshold at equal rollout-level FPR.
        for stride in (16, 32):
            audit = torch.full_like(scores, -1e9)
            ends = list(range(stride - 1, scores.size(0), stride))
            if not ends or ends[-1] != scores.size(0) - 1:
                ends.append(scores.size(0) - 1)
            for end in ends:
                begin = max(0, end - stride + 1)
                audit[end] = z[begin : end + 1].mean()
            out[f"audit{stride}"] = audit
        return out

    # ---- Phase 2: score all greedy rollouts once.
    records = []  # (is_wrong, oracle_site_tok or None, {rule: {variant: scores}})
    scanned = 0
    for row in rows:
        gold = row["gold"]
        text = row["latent"]
        is_wrong = extract_answer_lenient(text) != gold
        question = row["question"]
        prompt = _chat_prompt(tokenizer, COT_PROMPT + question)
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        seq = h_tap_sequence(prompt_ids, text)
        site_tok = None
        if is_wrong:
            allowed = _allowed_values(question, gsm8k[row["index"]]["answer"], gold)
            site = _first_error_site(text, allowed)
            if site is not None:
                site_tok = len(tokenizer(text[:site], add_special_tokens=False)["input_ids"])
        scored = {name: variants(fn(seq)) for name, fn in scorers.items()}
        records.append((is_wrong, site_tok, scored))
        scanned += 1
        if scanned % 40 == 0:
            print(json.dumps({"scanned": scanned}), flush=True)

    # ---- Phase 3: recall @ calibrated rollout-level false-alarm rates.
    wrong_total = sum(1 for r in records if r[0])
    results = []
    for name in scorers:
        for variant in ("raw", "selfnorm", "cusum8", "audit16", "audit32"):
            correct_max = torch.tensor([r[2][name][variant].max() for r in records if not r[0]])
            for fpr in (0.05, 0.10, 0.20):
                threshold = torch.quantile(correct_max, 1.0 - fpr).item()
                hits = 0
                delays = []
                for is_wrong, site_tok, scored in records:
                    if not is_wrong:
                        continue
                    above = (scored[name][variant] > threshold).nonzero()
                    if above.numel() == 0:
                        continue
                    hits += 1
                    if site_tok is not None:
                        delays.append(int(above[0]) - site_tok)
                delays.sort()
                entry = {
                    "rule": name,
                    "variant": variant,
                    "fpr": fpr,
                    "recall": round(hits / max(wrong_total, 1), 4),
                    "hits": hits,
                    "median_delay": delays[len(delays) // 2] if delays else None,
                }
                results.append(entry)
                print(json.dumps(entry), flush=True)

    summary = {
        "problems": len(records),
        "wrong": wrong_total,
        "train_pairs": train_pairs,
        "results": results,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Trigger lab: offline detection-rule bake-off",
        "",
        f"Model: `{model_name}`, dump: `{dump_path}`, problems: {len(records)}, "
        f"wrong rollouts: {wrong_total}, training pairs: {train_pairs}",
        "",
        "Rollout-level recall on wrong greedy rollouts at thresholds calibrated to",
        "fixed false-alarm rates over correct greedy rollouts (max-over-positions",
        "statistic). Delay = tokens from oracle error site to first trigger",
        "(negative = fired before the site).",
        "",
        "| rule | variant | FPR | recall | median delay (tok) |",
        "| --- | --- | --- | --- | --- |",
    ]
    for entry in results:
        delay = entry["median_delay"] if entry["median_delay"] is not None else "-"
        lines.append(
            f"| {entry['rule']} | {entry['variant']} | {entry['fpr']:.2f} | "
            f"**{entry['recall']:.3f}** ({entry['hits']}/{wrong_total}) | {delay} |"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def consensus_probe(
    model_name: str,
    corrector_path: str | Path,
    dump_path: str | Path,
    output_path: str | Path,
    device_str: str = "cuda",
) -> dict:
    """State-space consensus: do the k vote rollouts agree in h-space early?

    For each problem's latent-SC rollouts, teacher-force each rollout through
    the frozen trunk, mean-pool h_tap over the first t tokens, and measure
    trajectory dispersion = 1 - mean pairwise cosine of the pooled vectors.
    If early dispersion predicts final answer disagreement (or a wrong vote),
    the vote outcome is legible in state space before any answer commits —
    a consensus signal that needs no answer extraction and could gate
    adaptive sampling earlier than answer-level agreement.
    """

    import torch

    from prometheus.retrofit import COT_PROMPT, _chat_prompt, extract_answer_lenient, load_trunk

    device = torch.device(device_str if device_str != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model, tokenizer = load_trunk(model_name, device, dtype)
    checkpoint = torch.load(corrector_path, map_location=device, weights_only=True)
    tap_layer = checkpoint["config"]["tap_layer"]

    rows = []
    for line in Path(dump_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "latent_sc8" in row:
            rows.append(row)

    checkpoints = (8, 16, 32, 64, 128)

    @torch.no_grad()
    def pooled_prefixes(prompt_ids: list[int], text: str) -> dict | None:
        text_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if not text_ids:
            return None
        batch = torch.tensor([prompt_ids + text_ids], device=device)
        outputs = model(batch, output_hidden_states=True)
        seq = outputs.hidden_states[tap_layer][0, len(prompt_ids) - 1 : -1, :].float()
        pooled = {t: seq[: min(t, seq.size(0))].mean(0).cpu() for t in checkpoints}
        pooled["full"] = seq.mean(0).cpu()
        return pooled

    def dispersion(vectors: list) -> float:
        stacked = torch.stack(vectors)
        stacked = stacked / (stacked.norm(dim=1, keepdim=True) + 1e-6)
        sim = stacked @ stacked.T
        k = stacked.size(0)
        off_diagonal = (sim.sum() - sim.diagonal().sum()) / (k * (k - 1))
        return 1.0 - off_diagonal.item()

    records = []  # (vote_correct, unanimous, {t: dispersion})
    scanned = 0
    for row in rows:
        rollouts = json.loads(row["latent_sc8"])
        answers = [extract_answer_lenient(text) for text in rollouts]
        counts: dict = {}
        for answer in answers:
            counts[answer] = counts.get(answer, 0) + 1
        vote, vote_count = max(counts.items(), key=lambda item: item[1])
        vote_correct = vote == row["gold"]
        unanimous = vote_count == len(answers)
        prompt = _chat_prompt(tokenizer, COT_PROMPT + row["question"])
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        pooled_all = [p for p in (pooled_prefixes(prompt_ids, text) for text in rollouts) if p is not None]
        if len(pooled_all) < 2:
            continue
        disp = {t: dispersion([p[t] for p in pooled_all]) for t in (*checkpoints, "full")}
        records.append((vote_correct, unanimous, disp))
        scanned += 1
        if scanned % 20 == 0:
            print(json.dumps({"consensus_scanned": scanned}), flush=True)

    results = []
    for t in (*checkpoints, "full"):
        scores = [r[2][t] for r in records]
        split_labels = [0 if r[1] else 1 for r in records]
        wrong_labels = [0 if r[0] else 1 for r in records]
        unanimous_disp = [s for s, l in zip(scores, split_labels) if l == 0]
        split_disp = [s for s, l in zip(scores, split_labels) if l == 1]
        results.append(
            {
                "t": t,
                "mean_disp_unanimous": sum(unanimous_disp) / max(len(unanimous_disp), 1),
                "mean_disp_split": sum(split_disp) / max(len(split_disp), 1),
                "auc_split": _auc(scores, split_labels),
                "auc_vote_wrong": _auc(scores, wrong_labels),
            }
        )
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in results[-1].items()}), flush=True)

    unanimous_total = sum(1 for r in records if r[1])
    vote_correct_total = sum(1 for r in records if r[0])
    summary = {
        "problems": len(records),
        "unanimous": unanimous_total,
        "vote_correct": vote_correct_total,
        "results": results,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# State-space consensus probe",
        "",
        f"Model: `{model_name}`, dump: `{dump_path}`, problems: {len(records)}, "
        f"unanimous: {unanimous_total}, vote correct: {vote_correct_total}",
        "",
        "Dispersion = 1 − mean pairwise cosine of h_tap mean-pooled over the",
        "first t tokens of each of the 8 vote rollouts. AUCs: does early",
        "dispersion predict final answer disagreement (split) or a wrong",
        "majority vote?",
        "",
        "| t | mean disp (unanimous) | mean disp (split) | AUC split | AUC vote-wrong |",
        "| --- | --- | --- | --- | --- |",
    ]
    for entry in results:
        lines.append(
            f"| {entry['t']} | {entry['mean_disp_unanimous']:.4f} | {entry['mean_disp_split']:.4f} | "
            f"**{entry['auc_split']:.3f}** | **{entry['auc_vote_wrong']:.3f}** |"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary
