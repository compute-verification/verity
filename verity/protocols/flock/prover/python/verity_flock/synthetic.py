"""Synthetic input sets from the IR: seeded samples of a subcircuit's inputs, each with the output the IR evaluator computes.

The key is ``verity.commitments.randomness.derive(SOURCE, DOMAIN, {"subcircuit": id, "seed": seed})``; sample ``j`` is
``Subcircuit.draw(key, j)`` (``templates``: every port's words from the counter stream at ``(port, j)``), evaluated by
``verity.circuits.evaluation.evaluate_batch`` on the subcircuit's IR target.  A sample is admitted when its outputs are inside the
subcircuit's domain (``Subcircuit.admitted``: every BF16 output word finite), and instance ``i`` is the ``i``-th admitted sample
(``index.json`` ``sub.draw``), so a larger set extends a smaller one of the same key: the first ``n`` instances are the same.
The set is written in the captured sets' layout (``input_sets``, schema ``input-set/v1``) under ``<out>/sets/<name>/``, its
provenance holding the recipe, the Definition's Program digests and the software versions.
"""
from __future__ import annotations

import hashlib
import platform
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from verity.commitments.randomness import Key, derive

from verity_flock import input_sets as IS
from verity_flock import subcircuit_templates as TM

SOURCE = b"verity input-set synthetic"
DOMAIN = "verity/input-sets/synthetic/v1"
MAX_DRAW_FACTOR = 4
#: the sources a set's recipe depends on, each by its path under its import package
_SOURCES = {"verity_flock/subcircuit_templates.py": TM.__file__, "verity_flock/synthetic.py": __file__,
            "verity_flock/input_sets.py": IS.__file__}


def key_for(sub: TM.Subcircuit, seed: int) -> Key:
    return derive(SOURCE, DOMAIN, {"subcircuit": sub.id, "seed": int(seed)})


_CTX: dict[str, Any] = {}


def _chunk(lo_hi: tuple[int, int]) -> tuple[list[TM.Draw], dict[str, list[np.ndarray]]]:
    sub, key = _CTX["sub"], _CTX["key"]
    draws = [sub.draw(key, j) for j in range(*lo_hi)]
    outs = sub.evaluate({p.name: [d.inputs[p.name] for d in draws] for p in sub.inputs}, [d.instance_statics for d in draws])
    return draws, outs


def versions(head: str | None = None) -> dict[str, Any]:
    """The software a set was generated with: its source commit ``head``, Python's and numpy's versions, and the sha256 of each
    of the recipe's sources."""
    return {"git_head": head, "python": platform.python_version(), "numpy": np.__version__,
            "source_sha256": {k: hashlib.sha256(Path(f).read_bytes()).hexdigest() for k, f in _SOURCES.items()}}


def generate(sub: TM.Subcircuit, n: int, seed: int, out: str | Path, *, procs: int = 1, chunk: int = 64,
             provenance: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Write the first ``n`` admitted samples of ``sub`` under key (``sub``, ``seed``) as set ``<out>/sets/<sub.set_name>``, with
    ``provenance`` over the recipe's (the run that wrote it, the source commit's ``versions``); returns the summary
    (``input_sets.write``'s, plus the path, samples and rejected samples)."""
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
    recipe = {"generator": "verity_flock.synthetic", "template": sub.template.id, "subcircuit": sub.id, "parameters": sub.params,
              "seed": int(seed), "n": n, "key": f"verity.commitments.randomness.derive({SOURCE!r}, {DOMAIN!r}, {{'subcircuit': {sub.id!r}, 'seed': {int(seed)}}})",
              "draw": "templates.Subcircuit.draw(key, j): each port's words from key.stream((port, j), 4 bytes per word)",
              "samplers": sub.recipe, "admission": "every BF16 output word finite (Subcircuit.admitted); instance i = the i-th admitted draw",
              "draws": draws_used, "rejected": len(rejected), "rejected_draws": rejected[:64]}
    prov = {k: None for k in IS.PROVENANCE_FIELDS}
    prov.update({"source": "synthetic", "recipe": recipe,
                 "program_digests": sub.program_digests([d.instance_statics for _j, d, _o in kept]),
                 "selection": "the first n admitted draws, in draw order", "versions": versions()})
    prov.update(provenance or {})
    summary = IS.write(Path(out) / "sets", sub.set_name, sub.relation, ports, rows, prov, schema=IS.SCHEMA)
    summary.update(path=str(Path(out) / "sets" / sub.set_name), subcircuit=sub.id, draws=draws_used, rejected=len(rejected),
                   seconds=round(time.time() - t0, 3))
    return summary
