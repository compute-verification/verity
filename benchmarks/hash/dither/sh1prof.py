"""SERVE-H1 (`benchmarks/pouw/served_h1/serve.py`) as it stands, with every phase of its served step under an NVTX range and a
host timer: the served GEMM (`int8_ops.linear`), the A tables and their roots (`commit.s8_table`, `commit.levels`,
`commit.leaf`), the step roots, the live coin's round and salt, and the receipts (`receipts.gated_receipts`: its seeds,
noise and tile receipts).  Nothing of serve.py changes; the wrappers are installed on the modules it calls through.

    nsys profile -t cuda,nvtx python sh1prof.py --phases OUT/phases.json -- <serve.py's arguments>
"""
from __future__ import annotations

import functools
import json
import os
import sys
import time

import torch

PHASES: dict[str, list] = {}


def wrap(obj, name: str, label) -> None:
    """LABEL a phase's name, or a function of the call's arguments to it."""
    f = getattr(obj, name)

    @functools.wraps(f)
    def g(*a, **k):
        lab = label(*a, **k) if callable(label) else label
        torch.cuda.nvtx.range_push(lab)
        t = time.perf_counter()
        try:
            return f(*a, **k)
        finally:
            dt = time.perf_counter() - t
            torch.cuda.nvtx.range_pop()
            c = PHASES.setdefault(lab, [0, 0.0])
            c[0] += 1
            c[1] += dt

    setattr(obj, name, g)


def main() -> int:
    i = sys.argv.index("--")
    phases = sys.argv[sys.argv.index("--phases") + 1]
    argv = sys.argv[i + 1:]
    import receipts as RC
    import serve as SV
    from verity.ml.protocols.pouw.circuit import h1tile as HT
    from verity.core.protocols.sampled_proofs import challenger as Ch
    from tests.engine import int8_exact_gpu as G
    from verity_vllm.engine import int8_ops

    wrap(int8_ops, "linear", "gemm")
    wrap(RC, "gated_receipts", "receipts")
    wrap(RC, "unit_seeds", "receipts.seeds")
    wrap(RC, "noise_blocks", "receipts.noise")
    wrap(RC, "tile_receipts", "receipts.tile")
    wrap(SV.CM, "s8_table", lambda dom, key, label, *a, **k: "a_table_root" if label.startswith("a/") else "b_table_root")
    wrap(SV.CM, "levels", "admit_levels")
    wrap(SV, "step_roots", "step_roots")
    wrap(HT, "live_salt", "live_salt")
    wrap(Ch.LiveRecord, "record", "coin_round")
    wrap(G, "generate", "generate")
    t = time.perf_counter()
    rc = SV.main(argv)
    doc = {"wall_s": round(time.perf_counter() - t, 1), "rc": rc,
           "phases": {k: {"calls": v[0], "s": round(v[1], 3)} for k, v in sorted(PHASES.items())}}
    with open(phases, "w") as f:
        json.dump(doc, f, indent=1)
    print(json.dumps(doc), flush=True)
    return rc


if __name__ == "__main__":
    try:
        rc = main()
    except BaseException:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        rc = 2
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)
