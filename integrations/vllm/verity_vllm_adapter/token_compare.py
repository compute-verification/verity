"""Stock-vs-stock production token comparison across declared profiles (contract §15.5, diagnostic step 1).

CPU-only, no vLLM import: reads two or more ``production.jsonl`` records of the *same* workload and reports,
per unordered pair of profiles and per request, whether ``output_token_ids`` are identical, the first
differing output index and its absolute position ``t = prompt_len - 1 + index`` (the position whose
``sampler.select`` chose it), and every differing position with the token each profile committed. Prompt
ids must agree request by request (same workload) or the script aborts. Diagnostic only: it never touches a
verdict.

    python -m verity_vllm_adapter.token_compare --out cmp.json eager=<dir>/production.jsonl compiled=<dir>/production.jsonl
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path


def load(path: str) -> tuple[str | None, dict[str, dict]]:
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    pid = rows[0].get("profile_id") if rows else None
    return pid, {r["request_id"]: r for r in rows}


def compare_pair(a: dict[str, dict], b: dict[str, dict], la: str, lb: str) -> dict:
    per = []
    for rid in a:
        if rid not in b:
            continue
        ra, rb = a[rid], b[rid]
        if list(ra["prompt_token_ids"]) != list(rb["prompt_token_ids"]):
            raise SystemExit(f"{rid}: prompt ids differ between {la} and {lb}; not the same workload")
        ta, tb = list(ra["output_token_ids"]), list(rb["output_token_ids"])
        n = ra["prompt_len"]
        diffs = [
            {"output_index": i, "position": n - 1 + i, la: x, lb: y}
            for i, (x, y) in enumerate(itertools.zip_longest(ta, tb))
            if x != y
        ]
        per.append({
            "request_id": rid,
            "prompt_len": n,
            "max_tokens": len(ta),
            "identical": not diffs,
            "first_differing_output_index": diffs[0]["output_index"] if diffs else None,
            "first_differing_position": diffs[0]["position"] if diffs else None,
            "differing_positions": len(diffs),
            "differences": diffs,
        })
    return {
        "profiles": {la: None, lb: None},
        "requests_compared": len(per),
        "identical": sum(p["identical"] for p in per),
        "positions_compared": sum(p["max_tokens"] for p in per),
        "positions_differing": sum(p["differing_positions"] for p in per),
        "per_request": per,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("inputs", nargs="+", help="LABEL=path/to/production.jsonl (two or more)")
    args = ap.parse_args()
    labelled = {}
    for spec in args.inputs:
        label, _, path = spec.partition("=")
        labelled[label] = (path, *load(path))
    out = {"inputs": {k: {"path": v[0], "profile_id": v[1]} for k, v in labelled.items()}, "pairs": {}}
    for (la, (_, pa, a)), (lb, (_, pb, b)) in itertools.combinations(labelled.items(), 2):
        c = compare_pair(a, b, la, lb)
        c["profiles"] = {la: pa, lb: pb}
        out["pairs"][f"{la}_vs_{lb}"] = c
        print(f"{la}({pa}) vs {lb}({pb}): identical requests {c['identical']}/{c['requests_compared']}, "
              f"differing positions {c['positions_differing']}/{c['positions_compared']}; first divergences: "
              + ", ".join(f"{p['request_id'][:8]} idx={p['first_differing_output_index']} t={p['first_differing_position']}"
                          for p in c["per_request"] if not p["identical"]))
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
