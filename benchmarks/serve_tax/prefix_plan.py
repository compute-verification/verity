"""`prefix_plan.jsonl`, the full-size table of the load that repeats a prefix (`serve_tax.read_plan`), made from a serve
plan's two files: its rows, a line a request (its place ``i``, ``kind``, ``book``, ``prompt_len``, ``max_tokens`` and
``gap_draw``, beside prompt ids that this never keeps), and its meta (``rate``, ``max_model_len``,
``sent_within_span``, and ``long_rows``: each long request's ``i`` and the prefix-cache hit its model gives it).

The table is the requests the plan sends within its span, a line each:

- ``at_s``, its arrival: the running sum of the rows' ``gap_draw`` over the meta's ``rate``, in seconds, to the
  millisecond (the rule of ``benchmarks/network_e2e/traces/traffic.py``'s ``arrivals``);
- ``prompt`` and ``output``: its prompt's and its output's tokens;
- ``after`` and ``blocks``, on a row whose kind starts ``long128k`` and whose model's ``prefix_hit`` is not 0: the
  earlier ``long128k`` row of the same ``book``, counted from 0, and the whole blocks of 64 tokens the two share.

`prefix_plan.source.json` names the plan (its id in the evidence store and its two files there) and the table's
SHA-256.

    python prefix_plan.py PLAN.jsonl PLAN.meta.json              # make it and check it is the tree's table
    python prefix_plan.py PLAN.jsonl PLAN.meta.json OUT.jsonl    # write it to OUT.jsonl

It prints counts and sums only, never a row: the plan's rows hold their requests' prompt ids. Keep the plan's copy
outside any run's directory and remove it after.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "prefix_plan.source.json"
BLOCK = 64


def table(rows: list[dict], meta: dict) -> list[dict]:
    """The table's lines from a plan's rows and its meta. Refused where the rows are not numbered in order, a request
    does not fit the plan's context limit, or a hit is not whole blocks of one earlier row of its book."""
    rows = rows[:meta["sent_within_span"]]
    if [r["i"] for r in rows] != list(range(len(rows))):
        raise SystemExit("the plan's rows are not numbered in order")
    hits = {r["i"]: r["model"].get("prefix_hit") or 0 for r in meta["long_rows"]}
    out, t = [], 0.0
    for r in rows:
        t += r["gap_draw"] / meta["rate"]
        line = {"at_s": round(t, 3), "prompt": r["prompt_len"], "output": r["max_tokens"]}
        if line["prompt"] + line["output"] > meta["max_model_len"]:
            raise SystemExit(f"row {r['i']} does not fit the context of {meta['max_model_len']}")
        hit = hits.get(r["i"], 0) if r["kind"].startswith("long128k") else 0
        if hit:
            before = [q["i"] for q in rows[:r["i"]] if q["kind"] == "long128k" and q.get("book") == r.get("book")]
            if len(before) != 1 or hit % BLOCK or hit > min(r["prompt_len"], rows[before[0]]["prompt_len"]):
                raise SystemExit(f"row {r['i']}: its hit of {hit} is not whole blocks of one earlier row of its book")
            line |= {"after": before[0], "blocks": hit // BLOCK}
        out.append(line)
    return out


def text(lines: list[dict]) -> str:
    return "".join(json.dumps(x, separators=(",", ":")) + "\n" for x in lines)


def main(argv: list[str]) -> int:
    if len(argv) not in (2, 3):
        raise SystemExit(__doc__)
    with open(argv[0]) as f:
        rows = [json.loads(x) for x in f if x.strip()]
    lines = table(rows, json.loads(Path(argv[1]).read_text()))
    made = text(lines)
    digest, source = hashlib.sha256(made.encode()).hexdigest(), json.loads(SOURCE.read_text())
    print(json.dumps({"requests": len(lines), "prompt_tokens": sum(x["prompt"] for x in lines),
                      "output_tokens": sum(x["output"] for x in lines),
                      "repeated_tokens": sum(x.get("blocks", 0) * BLOCK for x in lines),
                      "repeats": [(k, x["after"], x["blocks"]) for k, x in enumerate(lines) if "after" in x],
                      "last_arrival_s": lines[-1]["at_s"], "sha256": digest}, indent=1))
    if len(argv) == 3:
        Path(argv[2]).write_text(made)
        return 0
    same = digest == source["sha256"] and (HERE / source["table"]).read_text() == made
    print(f"{source['table']}: {'the same' if same else 'NOT the same'} as the table made from this plan "
          f"({source['plan']}: {', '.join(source['files'])})")
    return 0 if same else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
