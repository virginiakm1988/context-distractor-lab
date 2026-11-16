"""Paired, token-controlled hard-distractor experiments inspired by the paper."""

from collections import defaultdict
import random, time, math
from .runtime import ask, client_for, parse_object, wilson, save_json
from .tokenization import get_tokenizer, pad_exact

TITLE = "First Drop of Ink with NVIDIA SDKs"
DEFAULT_CONFIG = {
    "seed": 42,
    "replicates": 3,
    "distractors": 10,
    "hard_fractions": [0, 0.1, 0.5, 1.0],
    "slot_tokens": 160,
    "gold_slot": 5,
    "model": "nvidia/nemotron-3.5-lightning-30b-a3b",
    "tokenizer_model": "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16",
    "thinking": False,
    "base_url": "https://integrate.api.nvidia.com/v1",
    "max_requests": 24,
    "max_tokens": 512,
    "temperature": 0,
}


def make_document(question, kind, index):
    entity, answer = question["entity"], question["answer"]
    if kind == "gold":
        return question["evidence"]
    if kind == "hard":
        return question["distractor"].replace("{INDEX}", str(index))
    return f"Unrelated record: botanical sample BOT{index:02d} has green leaves."


def make_question(repetition, seed):
    """Three small, independently answerable application scenarios."""
    rng = random.Random(seed)
    entity = f"project_{repetition:03d}"
    if repetition % 3 == 0:
        answer = f"KEY{rng.randrange(10000,99999)}"
        return {
            "entity": entity,
            "answer": answer,
            "scenario": "record_retrieval",
            "query": f"What is the verified delivery code for {entity}?",
            "evidence": f"Verified record: the current delivery code for {entity} is {answer}.",
            "distractor": f"Unverified draft: {entity} uses code WRONG{{INDEX}}. This entry is not verified.",
        }
    if repetition % 3 == 1:
        answer = f"city_{rng.randrange(100,999)}"
        owner = f"owner_{rng.randrange(100,999)}"
        return {
            "entity": entity,
            "answer": answer,
            "scenario": "two_hop_entity_join",
            "query": f"In which city is the owner of {entity} based?",
            "evidence": f"Verified ownership record: {entity} belongs to {owner}. Verified directory: {owner} is based in {answer}.",
            "distractor": f"Unverified draft: {entity} belongs to alternate_owner{{INDEX}}, based in wrong_city{{INDEX}}.",
        }
    units, price = rng.randint(2, 9), rng.randint(11, 29)
    return {
        "entity": entity,
        "answer": str(units * price),
        "scenario": "numeric_aggregation",
        "query": f"What is the verified total price for {entity}? Return a decimal integer without currency.",
        "evidence": f"Verified order: {entity} has {units} items. Verified pricing: every item costs {price} dollars.",
        "distractor": f"Unverified draft estimate: total price for {entity} is 999{{INDEX}} dollars.",
    }


def prepare(config, mode, output):
    if type(config["distractors"]) is not int or not 1 <= config["distractors"] <= 100:
        raise ValueError("distractors must be 1..100")
    if not 0 <= config["gold_slot"] <= config["distractors"]:
        raise ValueError("gold_slot outside document list")
    if any(not 0 <= f <= 1 for f in config["hard_fractions"]):
        raise ValueError("hard fractions must be in [0,1]")
    counts = [round(f * config["distractors"]) for f in config["hard_fractions"]]
    if len(counts) != len(set(counts)):
        raise ValueError(
            "Hard fractions round to duplicate counts; choose distinct experimental conditions"
        )
    tokenizer = get_tokenizer(mode, config)
    rows = []
    for repetition in range(config["replicates"]):
        rng = random.Random(config["seed"] + repetition)
        question = make_question(repetition, config["seed"] + repetition)
        order = list(range(config["distractors"]))
        rng.shuffle(order)
        prefix = (
            'Use only verified records. Return JSON {"answer":"..."}. '
            + question["query"]
            + "\n"
        )
        prefix_ids = tokenizer.encode(prefix)
        for fraction in config["hard_fractions"]:
            hard_count = round(fraction * config["distractors"])
            hard_indices = set(order[:hard_count])
            documents = []
            for i in range(config["distractors"]):
                kind = "hard" if i in hard_indices else "easy"
                text = make_document(question, kind, i)
                if (
                    kind == "hard"
                    and question["scenario"] == "record_retrieval"
                    and config.get("variants_path")
                ):
                    from .designer import load_approved

                    templates = load_approved(config["variants_path"], "distractor")
                    text = (
                        templates[i % len(templates)]
                        .replace("{ENTITY}", question["entity"])
                        .replace("{WRONG}", f"WRONG{i:02d}")
                    )
                documents.append({"kind": kind, "text": text})
            documents.insert(
                config["gold_slot"],
                {"kind": "gold", "text": make_document(question, "gold", 0)},
            )
            # Concatenate TOKEN ID slots, then decode and re-encode to verify boundaries.
            all_ids = list(prefix_ids)
            gold_start = None
            gold_end = None
            for index, document in enumerate(documents):
                text = pad_exact(
                    f"[Document {index}] " + document["text"] + "\nPadding:",
                    config["slot_tokens"],
                    tokenizer,
                )
                if document["kind"] == "gold":
                    gold_start = len(all_ids)
                all_ids.extend(tokenizer.encode(text))
                if document["kind"] == "gold":
                    gold_end = len(all_ids)
            prompt = tokenizer.decode(all_ids)
            if tokenizer.encode(prompt) != all_ids:
                # Reject silently drifting context length/position rather than label a false control.
                raise ValueError(
                    "Token boundaries changed on decode; adjust slot formatting before experiment"
                )
            rows.append(
                {
                    "id": f"ink-r{repetition}-h{hard_count}",
                    "replicate": repetition,
                    "question": question,
                    "answer": question["answer"],
                    "hard_fraction_requested": fraction,
                    "hard_fraction_actual": hard_count / config["distractors"],
                    "hard_count": hard_count,
                    "n_distractors": config["distractors"],
                    "prompt": prompt,
                    "documents": documents,
                    "input_tokens": len(all_ids),
                    "gold_token_start": gold_start,
                    "gold_token_end": gold_end,
                    "tokenizer": tokenizer.label,
                }
            )
    for repetition in range(config["replicates"]):
        group = [r for r in rows if r["replicate"] == repetition]
        if (
            len(
                {
                    (r["input_tokens"], r["gold_token_start"], r["gold_token_end"])
                    for r in group
                }
            )
            != 1
        ):
            raise AssertionError("Context or gold position not controlled")
    return rows


def run(config, rows, mode, output):
    if mode == "live" and config["model"] != "nvidia/nemotron-3.5-lightning-30b-a3b":
        raise ValueError(
            "Add a verified model/tokenizer mapping before changing models"
        )
    client = client_for(config, output) if mode == "live" else None
    result = []
    for row in rows:
        started = time.perf_counter()
        error = None
        raw = ""
        prediction = ""
        try:
            raw = (
                __import__("json").dumps({"answer": row["answer"]})
                if mode == "fixture"
                else ask(
                    client,
                    "Answer the question from the supplied records.",
                    row["prompt"],
                    config["model"],
                    config,
                    config["thinking"],
                )
            )
            prediction = str(parse_object(raw)["answer"]).strip()
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        result.append(
            {
                **row,
                "prediction": prediction,
                "raw": raw,
                "correct": prediction == row["answer"],
                "elapsed_seconds": time.perf_counter() - started,
                "mode": mode,
                "error": error,
            }
        )
    return result


def evaluate(rows, config):
    groups = defaultdict(list)
    for row in rows:
        groups[row["hard_fraction_actual"]].append(row)
    curve = []
    for ratio, items in sorted(groups.items()):
        n = len(items)
        correct = sum(
            r["prediction"] == r["answer"] and not r.get("error") for r in items
        )
        # Exact complete plain answers and parsed JSON are scored separately from
        # strict JSON protocol compliance. Never extract an answer from prose.
        content_correct = sum(
            (r["prediction"] == r["answer"] and not r.get("error"))
            or r.get("raw", "").strip() == r["answer"]
            for r in items
        )
        curve.append(
            {
                "hard_fraction": ratio,
                "n": n,
                "accuracy": correct / n,
                "accuracy_definition": "final_json_answer_accuracy",
                "answer_exact_match": content_correct / n,
                "answer_wilson95": wilson(content_correct, n),
                "wilson95": wilson(correct, n),
                "errors": sum(bool(r.get("error")) for r in items),
            }
        )
    return {
        "curve": curve,
        "n_predictions": len(rows),
        "scenarios": sorted({r["question"]["scenario"] for r in rows}),
        "interpretation": "Synthetic retrieval, two-hop joining and arithmetic are method demonstrations, not reproduction of the paper's QA datasets or its nonlinear effect. No degradation is imposed on fixture outputs.",
        "format_note": "accuracy requires a correct answer in a final JSON object (a fenced object or explanatory prefix is accepted; trailing prose is rejected). answer_exact_match additionally accepts a complete plain answer exactly equal to the oracle; it never mines text from prose. Report both to separate QA errors from output-format errors.",
    }


def design(config, mode, output):
    from .designer import generate_variants

    return generate_variants(config, mode, output, "distractor")
