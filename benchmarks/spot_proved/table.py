"""Whether a capture's history tables (`capture.py --table`) join: per (step, layer, request), the n latent rows below the row's
keys, its n index scores and the top-k its attention read.

    table.py steps --tree TREE --arm-dir ARM --load NAME [--rep 1] [--request ID] [--outputs 2] [--out STEPS.json]
    table.py check CAPTURE [--steps STEPS.json] [--out FILE]

``steps`` names, from the served window (TREE's ``benchmarks/serve_tax/replay.py`` `window`), the steps at which a
request's outputs 0, 1, ... were sampled: output i from the row attending to ``len(prompt) + i`` keys, at the step whose row
of the request ends there (``first position + tokens``). The window's step indices count from its first step, as
``capture.py --steps`` counts a replay's steps from an empty engine. Without ``--request`` it takes the longest prompt.

``check`` reads every ``rank<r>/step<s>/table/table.json`` under CAPTURE and holds each table to:

* its files' SHA-512s and shapes: n rows of the cache's words, n scores, one top-k, the rows attention read;
* n = position + 1, past the step's ``keys`` (the indexer's top-k, so the row's top-k keeps only some keys);
* the top-k is the canonical top-k of the scores (`dsa_topk.canonical_ok` over their f32 bits): the 2,048 positions whose
  scores rank first, in ascending order, the members of the counted mask ``DsaTopKMaskCounted`` computes from them;
* each row attention read is the table's row at that top-k position;
* every rank holds the same table, scores and top-k (the latent cache is replicated across TP ranks);
* with ``--steps``, each named (step, request) has a table at n = ``len(prompt) + i`` on every layer it has one on.

It prints a summary and exits 1 when any table fails.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from verity.ml.definitions.registry.dsa_topk import canonical_ok

DTYPES = {"torch.bfloat16": "<u2", "torch.float32": "<u4", "torch.int32": "<i4"}


def decode_steps(w: dict, request: str | None = None, outputs: int = 2) -> dict:
    """The window's steps sampling ``request``'s first ``outputs`` outputs: output i at the step whose row of the request
    attends to ``len(prompt) + i`` keys."""
    reqs = {q["engine_id"]: q for q in w["requests"]}
    rid = request or max(reqs, key=lambda k: len(reqs[k]["prompt_ids"]))
    prompt = len(reqs[rid]["prompt_ids"])
    at: dict[int, int] = {}
    for s in w["steps"]:
        for r in s["rows"]:
            if r[0] == rid:
                at.setdefault(r[1] + r[2], s["step"])
    return {"request": rid, "prompt": prompt,
            "steps": [{"output": i, "n": prompt + i, "step": at.get(prompt + i)} for i in range(outputs)]}


def _array(td: Path, k: int, name: str, f: dict) -> tuple[np.ndarray | None, str | None]:
    raw = (td / f"{k}.{name}").read_bytes() if (td / f"{k}.{name}").is_file() else None
    if raw is None:
        return None, f"no {k}.{name}"
    if hashlib.sha512(raw).hexdigest() != f["sha512"]:
        return None, f"{k}.{name}'s SHA-512 is not table.json's"
    a = np.frombuffer(raw, DTYPES[f["dtype"]])
    if a.size != int(np.prod(f["shape"])):
        return None, f"{k}.{name} holds {a.size} values, not {f['shape']}"
    return a.reshape(f["shape"]), None


def check_table(td: Path, k: int, e: dict, keys: int) -> list[str]:
    """Why table ``k`` of ``td`` doesn't join, or nothing."""
    files, n, why = e["files"], e["n"], []
    got = {}
    for name in ("kv", "scores", "topk", "read"):
        if name not in files:
            why.append(f"no {name}")
            continue
        got[name], w = _array(td, k, name, files[name])
        if w:
            why.append(w)
    if why:
        return why
    kv, scores, topk, read = got["kv"], got["scores"], got["topk"], got["read"]
    if n != e["position"] + 1 or n <= keys:
        why.append(f"n = {n} at position {e['position']}, with {keys} keys a top-k")
    if kv.shape[0] != n or scores.shape != (n,):
        why.append(f"{kv.shape[0]} rows and {scores.shape[0]} scores for n = {n}")
    if topk.ndim != 1 or read.shape != (topk.shape[0], kv.shape[1]):
        why.append(f"a top-k of {topk.shape} and read rows of {read.shape}")
    if why:
        return why
    if not canonical_ok([int(x) for x in scores], [int(x) for x in topk], topk.shape[0]):
        why.append(f"its top-k is not the canonical top-{topk.shape[0]} of its scores")
    elif not np.array_equal(kv[topk], read):
        bad = np.nonzero((kv[topk] != read).any(axis=1))[0]
        why.append(f"{bad.size} rows attention read are not the table's rows at their top-k positions (first: top-k entry "
                   f"{int(bad[0])}, position {int(topk[bad[0]])})")
    return why


def check(capture: Path, steps: dict | None = None) -> dict:
    tables, failed, ranks = [], [], {}
    for doc_file in sorted(capture.glob("rank*/step*/table/table.json")):
        doc, td = json.loads(doc_file.read_text()), doc_file.parent
        rank = int(td.parents[1].name.removeprefix("rank"))
        for k, e in enumerate(doc["tables"]):
            why = check_table(td, k, e, doc["keys"])
            if doc["rank"] != rank or td.parent.name != f"step{doc['step']}":
                why.append(f"table.json names rank {doc['rank']} step {doc['step']} under {td.parents[1].name}/{td.parent.name}")
            key = (doc["step"], e["layer"], e["request"])
            row = {"rank": rank, "step": doc["step"], "layer": e["layer"], "request": e["request"], "n": e["n"],
                   "op": e.get("op"), "by": e.get("by")}
            tables.append(row)
            if why:
                failed.append({**row, "why": why})
            ranks.setdefault(key, {})[rank] = tuple(e["files"][x]["sha512"] for x in ("kv", "scores", "topk")
                                                          if x in e["files"])
    for (step, layer, rid), by in sorted(ranks.items()):
        if len(set(by.values())) > 1:
            failed.append({"step": step, "layer": layer, "request": rid, "why": [f"ranks {sorted(by)} hold different tables"]})
    missing = []
    for s in (steps or {}).get("steps", []):
        rid = steps["request"]
        mine = [t for t in tables if t["step"] == s["step"] and (t["request"] == rid or str(t["request"]).startswith(f"{rid}-"))]
        if s["step"] is None or not mine:
            missing.append({**s, "why": "no table"})
        elif any(t["n"] != s["n"] for t in mine):
            missing.append({**s, "why": f"tables at n = {sorted({t['n'] for t in mine})}"})
    return {"tables": len(tables), "ranks": sorted({t["rank"] for t in tables}),
            "rows": sorted({(t["step"], t["layer"], t["request"], t["n"]) for t in tables}), "failed": failed,
            "missing": missing, "ok": bool(tables) and not failed and not missing}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("steps", help="the steps sampling a request's first outputs, from the served window")
    s.add_argument("--tree", type=Path, required=True)
    s.add_argument("--arm-dir", type=Path, required=True)
    s.add_argument("--load", required=True)
    s.add_argument("--rep", type=int, default=1)
    s.add_argument("--request")
    s.add_argument("--outputs", type=int, default=2)
    s.add_argument("--out", type=Path)
    c = sub.add_parser("check", help="whether a capture's history tables join")
    c.add_argument("capture", type=Path)
    c.add_argument("--steps", type=Path, help="steps' output: each named step must hold the request's table")
    c.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    if a.cmd == "steps":
        sys.path.insert(0, str(a.tree / "benchmarks" / "serve_tax"))
        import replay
        res = decode_steps(replay.window(a.arm_dir, a.load, a.rep), a.request, a.outputs)
    else:
        res = check(a.capture, json.loads(a.steps.read_text()) if a.steps else None)
    text = json.dumps(res, indent=1)
    if a.out:
        a.out.write_text(text)
    print(text)
    return 0 if a.cmd == "steps" or res["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
