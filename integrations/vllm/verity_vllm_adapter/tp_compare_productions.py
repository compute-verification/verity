"""Bit-compare production records (tokens, chosen logprobs) across profiles (lane TP, Level 1).

``python -m verity_vllm_adapter.tp_compare_productions --a <production.jsonl> --b <production.jsonl> [--out x.json]``

Two runs of one profile answer "is the profile deterministic"; TP=2 vs TP=1 answers "does tensor
parallelism change the function". Per request: whether the token sequences and the chosen-logprob
sequences are identical, the first differing output index, and the largest logprob difference over
the common prefix before the first token divergence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load(p: str | Path) -> dict[str, dict[str, Any]]:
    return {json.loads(l)["request_id"]: json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()}


def compare(a: dict[str, dict[str, Any]], b: dict[str, dict[str, Any]]) -> dict[str, Any]:
    per = []
    for k in a:
        if k not in b:
            continue
        ta, tb = a[k]["output_token_ids"], b[k]["output_token_ids"]
        la, lb = a[k]["chosen_logprobs"], b[k]["chosen_logprobs"]
        first_tok = next((i for i, (x, y) in enumerate(zip(ta, tb)) if x != y), None)
        if first_tok is None and len(ta) != len(tb):
            first_tok = min(len(ta), len(tb))
        first_lp = next((i for i, (x, y) in enumerate(zip(la, lb)) if x != y), None)
        common = first_tok if first_tok is not None else min(len(ta), len(tb))
        diffs = [abs(x - y) for x, y in zip(la[:common], lb[:common]) if x is not None and y is not None]
        per.append({
            "request_id": k, "prompt_len": a[k]["prompt_len"], "output_len_a": len(ta), "output_len_b": len(tb),
            "tokens_identical": ta == tb, "logprobs_identical": la == lb,
            "first_differing_output_index": first_tok, "first_differing_logprob_index": first_lp,
            "logprob_positions_before_token_divergence": common,
            "logprob_positions_equal_before_token_divergence": sum(1 for x, y in zip(la[:common], lb[:common]) if x == y),
            "max_abs_logprob_diff_before_token_divergence": max(diffs, default=0.0),
        })
    return {
        "profile_a": next(iter(a.values()))["profile_id"] if a else None,
        "profile_b": next(iter(b.values()))["profile_id"] if b else None,
        "requests_compared": len(per),
        "tokens_identical": sum(p["tokens_identical"] for p in per),
        "logprobs_identical": sum(p["logprobs_identical"] for p in per),
        "output_tokens_a": sum(p["output_len_a"] for p in per),
        "logprob_positions_compared_before_divergence": sum(p["logprob_positions_before_token_divergence"] for p in per),
        "logprob_positions_equal_before_divergence": sum(p["logprob_positions_equal_before_token_divergence"] for p in per),
        "max_abs_logprob_diff_before_divergence": max((p["max_abs_logprob_diff_before_token_divergence"] for p in per), default=0.0),
        "per_request": per,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    doc = compare(load(args.a), load(args.b))
    doc["a"], doc["b"] = str(args.a), str(args.b)
    if args.out:
        Path(args.out).write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps({k: v for k, v in doc.items() if k != "per_request"}, indent=1))
    for p in doc["per_request"]:
        print(f" {p['request_id'][:8]} plen={p['prompt_len']:5d} n={p['output_len_a']:3d}/{p['output_len_b']:3d} tokens={'=' if p['tokens_identical'] else '!'} "
              f"logprobs={'=' if p['logprobs_identical'] else '!'} first_tok_diff={p['first_differing_output_index']} first_lp_diff={p['first_differing_logprob_index']} "
              f"lp_equal_before_div={p['logprob_positions_equal_before_token_divergence']}/{p['logprob_positions_before_token_divergence']} "
              f"max|dlp|={p['max_abs_logprob_diff_before_token_divergence']:.3e}")


if __name__ == "__main__":
    main()
