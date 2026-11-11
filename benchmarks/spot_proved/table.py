"""Whether a capture's history tables (`capture.py --table`) join: per (step, layer, request), the n latent rows below the row's
keys, its n index scores and the top-k its attention read.

    table.py steps --tree TREE --arm-dir ARM --load NAME [--rep 1] [--outputs 2] --out STEPS.json
    table.py check CAPTURE [--steps STEPS.json] --out FILE

``steps`` names, from the served window (TREE's ``benchmarks/serve_tax/replay.py`` `window`), the steps at which the
longest prompt's outputs 0, 1, ... were sampled: output i from the row attending to ``len(prompt) + i`` keys, at the step
whose row of the request ends there (``first position + tokens``). The window's step indices count from its first step, as
``capture.py --steps-from STEPS.json`` counts a replay's steps from an empty engine.

``check`` reads every ``rank<r>/step<s>/table/table.json`` under CAPTURE and holds each table to:

* its files' SHA-512s, dtypes and shapes: n rows of the cache's bf16 words, n f32 scores, one i32 top-k, the bf16 rows
  attention read;
* n = position + 1, past the step's ``keys`` (the indexer's top-k, so the row's top-k keeps only some keys);
* the top-k is the canonical top-k of the scores (`dsa_topk.canonical_ok` over their f32 bits): the 2,048 positions whose
  scores rank first, in ascending order, the members of the counted mask ``DsaTopKMaskCounted`` computes from them;
* each row attention read is the table's row at that top-k position;
* every rank holds the same table, scores and top-k (the latent cache is replicated across TP ranks);
* with ``--steps``, each named (step, request) has a table at n = ``len(prompt) + i`` on every layer it has one on.

Both write their result (request ids, steps, lengths) only to ``--out``, a file of mode 0600 outside any run directory
(`capture.run_dir_of`), and print counts; ``check`` exits 1 when any table fails.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np

from verity.definitions.registry.dsa_topk import canonical_ok

DTYPES = {"torch.bfloat16": "<u2", "torch.float32": "<u4", "torch.int32": "<i4"}
#: each file's dtype: the cache's and attention's bf16 rows, the f32 scores the counted mask ranks, the i32 top-k
WANT = {"kv": "torch.bfloat16", "scores": "torch.float32", "topk": "torch.int32", "read": "torch.bfloat16"}


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
        if files[name]["dtype"] != WANT[name]:
            why.append(f"{k}.{name} is {files[name]['dtype']}, not {WANT[name]}")
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


def _run_dir_of(p: Path) -> Path | None:
    spec = importlib.util.spec_from_file_location("spot_capture", Path(__file__).with_name("capture.py"))
    capture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(capture)
    return capture.run_dir_of(p)


def write_private(path: Path, text: str) -> None:
    """``text`` to ``path`` with mode 0600; refused (SystemExit) inside a run directory, whose files custody uploads."""
    if (run := _run_dir_of(path.parent)) is not None:
        raise SystemExit(f"--out {path} lies inside the run directory {run}, whose files custody uploads to the store: "
                         "give it a private path outside the run directory")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(text)
    path.chmod(0o600)


def summary(cmd: str, res: dict) -> dict:
    """What ``res`` says that may be printed: counts, no request id, step or length."""
    if cmd == "steps":
        return {"outputs": len(res["steps"]), "found": sum(s["step"] is not None for s in res["steps"])}
    return {"ok": res["ok"], "tables": res["tables"], "ranks": len(res["ranks"]), "failed": len(res["failed"]),
            "missing": len(res["missing"])}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("steps", help="the steps sampling the longest prompt's first outputs, from the served window")
    s.add_argument("--tree", type=Path, required=True)
    s.add_argument("--arm-dir", type=Path, required=True)
    s.add_argument("--load", required=True)
    s.add_argument("--rep", type=int, default=1)
    s.add_argument("--outputs", type=int, default=2)
    s.add_argument("--out", type=Path, required=True, help="the result, private: outside any run directory, mode 0600")
    c = sub.add_parser("check", help="whether a capture's history tables join")
    c.add_argument("capture", type=Path)
    c.add_argument("--steps", type=Path, help="steps' output: each named step must hold the request's table")
    c.add_argument("--out", type=Path, required=True, help="the result, private: outside any run directory, mode 0600")
    a = ap.parse_args(argv)
    if a.cmd == "steps":
        sys.path.insert(0, str(a.tree / "benchmarks" / "serve_tax"))
        import replay
        res = decode_steps(replay.window(a.arm_dir, a.load, a.rep), None, a.outputs)
    else:
        res = check(a.capture, json.loads(a.steps.read_text()) if a.steps else None)
    write_private(a.out, json.dumps(res, indent=1))
    print(json.dumps(summary(a.cmd, res)))
    return 0 if a.cmd == "steps" or res["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
