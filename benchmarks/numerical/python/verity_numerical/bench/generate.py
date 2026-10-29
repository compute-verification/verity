"""Synthetic input sets from the IR: seeded draws of a subcircuit's inputs, each with the output the IR evaluator computes.

    python -m verity_numerical.bench.generate --list
    python -m verity_numerical.bench.generate --template gemm-coordinate --param K=2048 --param semantics=sm80.mma.m16n8k16.bf16 \\
        --n 4096 --seed 20260926 --out DIR [--procs P] [--register]
    python -m verity_numerical.bench.generate --subcircuit rope-head/d64/neox-bf16 --n 1024 --seed 1 --out DIR

The key is ``verity.ml.randomness.derive(SOURCE, DOMAIN, {"subcircuit": id, "seed": seed})``; draw ``j`` is
``Subcircuit.draw(key, j)`` (``templates``: every port's words from the counter stream at ``(port, j)``), evaluated by
``verity.ml.circuits.evaluation.evaluate_batch`` on the subcircuit's IR target.  A draw is admitted when its outputs are inside the
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

from verity.ml.randomness import Key, derive

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
            need = (n - len(kept)) + (n - len(kept)) // 8 + 1
            per = chunk if pool is None else max(1, min(chunk, -(-need // procs)))
            span = max(per, need) if pool is None else per * procs
            ranges = [(a, min(j + span, a + per)) for a in range(j, j + span, per)]
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
              "seed": int(seed), "n": n, "key": f"verity.ml.randomness.derive({SOURCE!r}, {DOMAIN!r}, {{'subcircuit': {sub.id!r}, 'seed': {int(seed)}}})",
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


#: Named suites: (template, parameters, n).  ``101``: row #101's captured subcircuits (Llama-3.2-1B), plus its GEMMs on Hopper wgmma.
SUITES: dict[str, list[tuple[str, dict[str, Any], int]]] = {
    "101": [("gemm-coordinate", {"K": 2048, "semantics": "sm80.mma.m16n8k16.bf16"}, 4096),
            ("gemm-coordinate", {"K": 8192, "semantics": "sm80.mma.m16n8k16.bf16"}, 4096),
            ("gemm-coordinate", {"K": 2048, "semantics": "sm90.wgmma.m64n8k16.bf16"}, 4096),
            ("gemm-coordinate", {"K": 8192, "semantics": "sm90.wgmma.m64n8k16.bf16"}, 4096),
            ("attention-head", {"D": 64, "BN": 128, "T": "1:512"}, 1024),
            ("rope-head", {"D": 64}, 1024),
            ("rmsnorm-fused-cuda", {"N": 2048, "EPS": 1e-5}, 256),
            ("rmsnorm-triton", {"N": 2048, "EPS": 1e-5}, 256),
            ("silu-mul", {"I": 8192}, 256),
            ("gumbel-top-p-token-select", {"V": 128256}, 32)],
    #: The FP8 GEMM lines at vLLM's reduction lengths: the captured #101 sizes (6,272 at K = 2048, 1,920 at K = 8192) and the
    #: largest batches the RTX 4090's 24 GB carried (16,384 and 8,192); each smaller set is the larger one's prefix.
    "fp8": [("gemm-coordinate", {"K": K, "semantics": sem}, n)
            for sem in ("sm89.mma.m16n8k32.e4m3", "sm90.wgmma.m64n8k32.e4m3")
            for K, sizes in ((2048, (6272, 16384)), (8192, (1920, 8192))) for n in sizes],
    #: NVFP4 on the RTX 5090 at K = 1536 for the two Flock cells (Fp4 <= 32,768 and ShaFp4 <= 16,384 VUs per proof); 8,192 is the prefix.
    "nvfp4": [("gemm-coordinate", {"K": 1536, "semantics": "sm120.mma.m16n8k64.e2m1.nvf4"}, n) for n in (8192, 16384)],
    #: The registered compiler's dead-free Definitions a served window draws, at Llama-3.2-1B's shapes (N 2048, D 64) and
    #: Llama-3.1-8B's (N 4096, D 128): RMSNormTriton_v5, AttentionHead_v18, and TokenSelect_v4's two calls at Llama 3's vocabulary.
    "dead-free": [("rmsnorm-triton-v5", {"N": 2048, "EPS": 1e-5}, 256),
                  ("rmsnorm-triton-v5", {"N": 4096, "EPS": 1e-5}, 256),
                  ("attention-head-v18", {"D": 64, "BN": 128, "T": "1:512"}, 1024),
                  ("attention-head-v18", {"D": 128, "BN": 64, "T": "1:512"}, 128),
                  ("token-select-chunk", {"C": 64}, 1024),
                  ("token-select-combine", {"V": 128256, "C": 64}, 64)],
}


#: Instances per set of the workloads suite, by template: enough for a sweep's first points, bounded by bytes (a set of GEMM
#: coordinates at K = 14336 is 117 MB) and by the evaluator (an embedding input materialises the V x H table).
WORKLOAD_N = {"gemm-coordinate": 2048, "attention-head": 128, "rope-head": 1024, "rmsnorm-fused-cuda": 256, "rmsnorm-triton": 256,
              "silu-mul": 256, "gumbel-top-p-token-select": 32, "token-select-greedy": 64, "moe-router-topk": 4096,
              "moe-router-topk-norm": 4096, "moe-sum": 1024, "bias-add": 1024, "bf16-mul-scalar": 1024, "embedding-row": 16,
              "moe-expert-coordinate": 2048, "moe-expert-coordinate-weighted": 2048, "scaled-mm-fp8-block-coordinate": 2048}
WORKLOAD_ROWS = (39, 57, 60, 67, 73, 74)


def workload_suite(rows: tuple[int, ...] = WORKLOAD_ROWS) -> list[tuple[str, dict[str, Any], int]]:
    """One set per census-bound subcircuit of the workloads of vLLM ``rows`` (``benchmarks/census/workloads.json``: a mix entry's
    template bound to exactly its parameters), in id order; an attention head's T range is its census ``variable_ranges.T``."""
    from . import census

    bound = {(s.template, json.dumps(vars(s.parameters), sort_keys=True)): s.id for s in census.SUBCIRCUITS.values()}
    sids = sorted({sid for w in census.raw("workloads") if (w.get("source") or {}).get("row") in rows for m in w["template_mix"]
                   if (sid := bound.get((m["subcircuit_template"], json.dumps(m["subcircuit_parameters"] or {}, sort_keys=True))))})
    out = []
    for sid in sids:
        sub = TM.parse_subcircuit_id(sid)
        params = dict(sub.params)
        if sub.template.id == "attention-head":
            rng = census.SUBCIRCUITS[sid].variable_ranges.T
            params["T"] = f"{rng.min}:{rng.max}"
        out.append((sub.template.id, params, WORKLOAD_N[sub.template.id]))
    return out


def suite(name: str, seed: int, out: str | Path, procs: int = 1, digests_only: bool = False) -> list[dict[str, Any]]:
    """Generate every set of suite ``name``, each under ``<out>/n<n>/sets/<name>/``; with ``digests_only`` each set's port arrays
    are removed once written (its manifest, with every file's sha256, and index stay): enough to compare a regeneration byte
    for byte."""
    out_sets = []
    for tid, params, n in (workload_suite() if name == "workloads" else SUITES[name]):
        sub = TM.subcircuit(tid, **params)
        s = generate(sub, n, seed, Path(out) / f"n{n}", procs=1 if tid == "embedding-row" else procs)   # one V x H table, not one per worker
        s["manifest_sha256"] = IS.InputSet.open(s["path"]).manifest_sha256()
        if digests_only:
            d = Path(s["path"])
            for f in d.iterdir():
                if f.name not in ("manifest.json", "index.json"):
                    f.unlink()
        out_sets.append(s)
        print(json.dumps({k: s[k] for k in ("set", "subcircuit", "n", "content_digest", "draws", "rejected", "seconds")}), flush=True)
    return out_sets


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
    ap.add_argument("--suite", choices=sorted([*SUITES, "workloads"]),
                    help="every set of a named suite (SUITES; workloads: one per bound subcircuit of WORKLOAD_ROWS) under --out, with --seed")
    ap.add_argument("--summary", default=None, help="--suite: write the sets' summaries (digests included) to this JSON file")
    ap.add_argument("--digests-only", action="store_true", help="--suite: keep each set's manifest and index, not its arrays")
    a = ap.parse_args(list(argv) if argv is not None else None)
    if a.suite:
        if a.seed is None or not a.out:
            ap.error("--suite needs --seed and --out")
        sets = suite(a.suite, a.seed, a.out, a.procs, a.digests_only)
        if a.register:
            for s in sets:
                s["art"] = register(s["path"])
        doc = {"suite": a.suite, "seed": a.seed, "sets": sets, "versions": _versions()}
        if a.summary:
            Path(a.summary).write_text(json.dumps(doc, indent=1, default=str) + "\n")
        print(json.dumps({"suite": a.suite, "sets": [{k: s.get(k) for k in ("set", "content_digest", "art")} for s in sets]}, indent=1))
        return 0
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
