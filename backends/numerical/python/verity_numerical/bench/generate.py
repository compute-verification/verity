"""Synthetic input sets from the IR: seeded draws of a subcircuit's inputs, each with the output the IR evaluator computes.

    python -m verity_numerical.bench.generate --list
    python -m verity_numerical.bench.generate --template gemm-coordinate --param K=2048 --param semantics=sm80.mma.m16n8k16.bf16 \\
        --n 4096 --seed 20260926 --out DIR [--procs P] [--register]
    python -m verity_numerical.bench.generate --subcircuit rope-head/d64/neox-bf16 --n 1024 --seed 1 --out DIR

The key is ``verity.randomness.derive(SOURCE, DOMAIN, {"subcircuit": id, "seed": seed})``; draw ``j`` is
``Subcircuit.draw(key, j)`` (``templates``: every port's words from the counter stream at ``(port, j)``), evaluated by
``verity.evaluation.evaluate_batch`` on the subcircuit's IR target.  A draw is admitted when its outputs are inside the
subcircuit's domain (``Subcircuit.admitted``: every BF16 output word finite), and instance ``i`` is the ``i``-th admitted draw
(``index.json`` ``sub.draw``), so a larger set extends a smaller one of the same key: the first ``n`` instances are the same.
The set is written in the captured sets' layout (``input_sets``, schema ``input-set/v1``) under ``<out>/sets/<name>/``, its
provenance holding the recipe, the Definition's Program digests and the software versions.  ``--register`` puts it in the
evidence store as an ``input-set/v1`` artifact (``research data put --tree ... --preserve``) and prints its id.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from verity.randomness import Key, derive

from . import input_sets as IS
from . import templates as TM

SOURCE = b"verity input-set synthetic"
DOMAIN = "verity/input-sets/synthetic/v1"
MAX_DRAW_FACTOR = 4
_SOURCES = ("templates.py", "generate.py", "input_sets.py")


def key_for(sub: TM.Subcircuit, seed: int) -> Key:
    return derive(SOURCE, DOMAIN, {"subcircuit": sub.id, "seed": int(seed)})


_CTX: dict[str, Any] = {}


def _chunk(lo_hi: tuple[int, int]) -> tuple[list[TM.Draw], dict[str, list[np.ndarray]]]:
    sub, key = _CTX["sub"], _CTX["key"]
    draws = [sub.draw(key, j) for j in range(*lo_hi)]
    outs = sub.evaluate({p.name: [d.inputs[p.name] for d in draws] for p in sub.inputs}, [d.instance_statics for d in draws])
    return draws, outs


def _versions() -> dict[str, Any]:
    here = Path(__file__).resolve().parent
    head = os.environ.get("RESEARCH_SOURCE_COMMIT")
    if not head:
        try:
            head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=here, capture_output=True, text=True, timeout=10).stdout.strip() or None
        except (OSError, subprocess.SubprocessError):
            head = None
    return {"git_head": head, "python": platform.python_version(), "numpy": np.__version__,
            "source_sha256": {f"verity_numerical/bench/{f}": hashlib.sha256((here / f).read_bytes()).hexdigest() for f in _SOURCES}}


def generate(sub: TM.Subcircuit, n: int, seed: int, out: str | Path, *, procs: int = 1, chunk: int = 64,
             provenance: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Write the first ``n`` admitted draws of ``sub`` under key (``sub``, ``seed``) as set ``<out>/sets/<sub.set_name>``; returns the
    summary (``input_sets.write``'s, plus the path, draws and rejected draws)."""
    if n <= 0:
        raise ValueError("n must be positive")
    key = key_for(sub, seed)
    t0 = time.time()
    kept: list[tuple[int, TM.Draw, dict[str, np.ndarray]]] = []
    rejected: list[int] = []
    limit = MAX_DRAW_FACTOR * n + 64
    _CTX.update(sub=sub, key=key)
    pool = ProcessPoolExecutor(max_workers=procs, mp_context=__import__("multiprocessing").get_context("fork")) if procs > 1 else None
    try:
        j = 0
        while len(kept) < n:
            if j >= limit:
                raise RuntimeError(f"{sub.id}: {len(kept)} of {n} draws admitted after {j} (the sampler leaves the domain too often)")
            want = n - len(kept)
            span = max(chunk, want + want // 8 + 1) if pool is None else chunk * procs
            ranges = [(a, min(j + span, a + chunk)) for a in range(j, j + span, chunk)]
            results = list(pool.map(_chunk, ranges)) if pool is not None else [_chunk(r) for r in ranges]
            for (a, _b), (draws, outs) in zip(ranges, results):
                ok = sub.admitted(outs)
                for r, d in enumerate(draws):
                    if len(kept) >= n:
                        break
                    if ok[r]:
                        kept.append((a + r, d, {p.name: outs[p.name][r] for p in sub.outputs}))
                    else:
                        rejected.append(a + r)
            j += span
    finally:
        _CTX.clear()
        if pool is not None:
            pool.shutdown()
    draws_used = kept[-1][0] + 1
    rejected = [r for r in rejected if r < draws_used]
    ports: dict[str, list[np.ndarray]] = {p.name: [d.inputs[p.name] for _j, d, _o in kept] for p in sub.inputs}
    ports.update({p.name: [o[p.name] for _j, _d, o in kept] for p in sub.outputs})
    rows = [[i, None, None, None, None, None, None, "synthetic", {"draw": jj, **d.sub}] for i, (jj, d, _o) in enumerate(kept)]
    recipe = {"generator": "verity_numerical.bench.generate", "template": sub.template.id, "subcircuit": sub.id, "parameters": sub.params,
              "seed": int(seed), "n": n, "key": f"verity.randomness.derive({SOURCE!r}, {DOMAIN!r}, {{'subcircuit': {sub.id!r}, 'seed': {int(seed)}}})",
              "draw": "templates.Subcircuit.draw(key, j): each port's words from key.stream((port, j), 4 bytes per word)",
              "samplers": sub.recipe, "admission": "every BF16 output word finite (Subcircuit.admitted); instance i = the i-th admitted draw",
              "draws": draws_used, "rejected": len(rejected), "rejected_draws": rejected[:64]}
    prov = {k: None for k in IS.PROVENANCE_FIELDS}
    prov.update({"source": "synthetic", "recipe": recipe, "research_run": os.environ.get("RESEARCH_RUN_ID"),
                 "program_digests": sub.program_digests([d.instance_statics for _j, d, _o in kept]),
                 "selection": "the first n admitted draws, in draw order", "versions": _versions()})
    prov.update(provenance or {})
    summary = IS.write(Path(out) / "sets", sub.set_name, sub.relation, ports, rows, prov, schema=IS.SCHEMA)
    summary.update(path=str(Path(out) / "sets" / sub.set_name), subcircuit=sub.id, draws=draws_used, rejected=len(rejected),
                   seconds=round(time.time() - t0, 3))
    return summary


def register_command(set_dir: str | Path, meta_file: str | Path) -> list[str]:
    """The ``research data put`` that registers a set (``input-set/v1``, preserved)."""
    return ["research", "data", "put", "--kind", IS.KIND, "--tree", str(set_dir), "--meta", f"@{meta_file}", "--preserve"]


def register(set_dir: str | Path) -> str:
    d = Path(set_dir)
    meta = d.parent.parent / f"{d.name}.meta.json"
    meta.write_text(json.dumps(IS.InputSet.open(d).meta(), indent=1, default=str) + "\n")
    cmd = register_command(d, meta)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"{' '.join(cmd)} failed: {r.stderr[-800:]}")
    art = next((w for w in r.stdout.split() if w.startswith("art:")), None)
    if art is None:
        raise RuntimeError(f"no art id in: {r.stdout[-400:]}")
    return art


def parse_value(v: str) -> Any:
    for f in (int, float):
        try:
            return f(v)
        except ValueError:
            pass
    return v


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="verity_numerical.bench.generate")
    ap.add_argument("--list", action="store_true", help="the templates, their parameters and an example binding")
    ap.add_argument("--template")
    ap.add_argument("--subcircuit", help="a census-style subcircuit id instead of --template/--param")
    ap.add_argument("--param", action="append", default=[], metavar="NAME=VALUE")
    ap.add_argument("--n", type=int)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--out")
    ap.add_argument("--procs", type=int, default=1)
    ap.add_argument("--register", action="store_true", help="research data put --kind input-set/v1 --preserve; prints the art id")
    a = ap.parse_args(list(argv) if argv is not None else None)
    if a.list:
        for t in TM.TEMPLATES.values():
            ex = TM.subcircuit(t.id, **t.example)
            print(json.dumps({"template": t.id, "name": t.name, "parameters": t.parameters, "example": t.example, "subcircuit": ex.id,
                              "set": ex.set_name, "ports": [[p.name, p.dtype, p.words] for p in ex.ports]}))
        return 0
    if a.n is None or a.seed is None or not a.out or not (a.template or a.subcircuit):
        ap.error("--n, --seed, --out and --template or --subcircuit are required")
    if a.subcircuit:
        sub = TM.parse_subcircuit_id(a.subcircuit)
        if a.param:
            sub = TM.subcircuit(sub.template.id, **{**sub.params, **dict((k, parse_value(v)) for k, _, v in (p.partition("=") for p in a.param))})
    else:
        sub = TM.subcircuit(a.template, **{k: parse_value(v) for k, _, v in (p.partition("=") for p in a.param)})
    s = generate(sub, a.n, a.seed, a.out, procs=a.procs)
    if a.register:
        s["art"] = register(s["path"])
    print(json.dumps(s, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
