"""Recompute response-format diagnostics from saved NIM evidence, without calls.

The original predictions and metrics remain untouched. Matching the exact request
hash protects against accidentally pairing a provider response with another case.
"""

import argparse
import hashlib
import json
from pathlib import Path
from . import core
from .runtime import parse_object, read_jsonl, save_json, save_jsonl


def rescore(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "run_manifest.json").read_text(encoding="utf-8"))
    if manifest["mode"] != "live":
        raise ValueError("Audit rescoring requires saved live NIM evidence")
    config = manifest["config"]
    responses = {}
    for file in sorted((directory / "api").glob("attempt-*.json")):
        attempt = json.loads(file.read_text(encoding="utf-8"))
        request_hash = attempt["request_sha256"]
        if request_hash in responses:
            raise ValueError(
                "Multiple attempts have the same request hash; select an explicit attempt before rescoring"
            )
        responses[request_hash] = attempt
    rows = read_jsonl(directory / "predictions.jsonl")
    audited = []
    for row in rows:
        payload = {
            "model": config["model"],
            "messages": [
                {
                    "role": "system",
                    "content": "Answer the question from the supplied records.",
                },
                {"role": "user", "content": row["prompt"]},
            ],
            "max_tokens": config["max_tokens"],
            "temperature": config["temperature"],
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": config["thinking"]},
        }
        request_hash = hashlib.sha256(
            json.dumps(payload, ensure_ascii=False).encode()
        ).hexdigest()
        attempt = responses[request_hash]
        updated = dict(row)
        if attempt.get("status") == "succeeded":
            raw = attempt["response"]["choices"][0]["message"].get("content", "")
            updated["raw"] = raw
            try:
                updated["prediction"] = str(parse_object(raw)["answer"]).strip()
                updated["error"] = None
            except (ValueError, KeyError) as error:
                updated["prediction"] = ""
                updated["error"] = f"{type(error).__name__}: {error}"
        updated["audited_attempt_id"] = attempt["attempt_id"]
        updated["request_sha256"] = request_hash
        audited.append(updated)
    metrics = core.evaluate(audited, config)
    metrics.update(
        result_kind="actual_nim_calls_rescored_no_new_requests",
        source="Original provider responses matched by request SHA-256. Original predictions and metrics are retained.",
    )
    save_jsonl(directory / "predictions-audited.jsonl", audited)
    save_json(directory / "metrics-audited.json", metrics)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(rescore(args.output_dir), indent=2))
