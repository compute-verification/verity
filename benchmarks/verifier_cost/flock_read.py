"""flock_read.py K WORK [--circuit-only]: the H1 tile unit's flat statement circuit at contraction K (`h1_prove/h1unit.Shape(K)`,
#3002's 4×8 tile, staged flat as `h1_prove/job.sh` stages it), and one instance on it (not with
`--circuit-only`, which `flock_tmpl.sh` passes), for `flock_read.sh` to time the Lean verifier's read of.

Writes WORK/circuit.txt, WORK/stmt/{inst.bin,pub.bin} and WORK/build.json (the unit's ANDs, nnz, k_log, the circuit's bytes
and the build's seconds). The instance's codes, salt and roots are random: the read's cost depends on the circuit only.
The stager's check of the outputs against the IR reference is skipped (hours at K = 4096); the lowering's own evaluation
still has to satisfy the unit.
"""
from __future__ import annotations

import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np

H1 = Path(__file__).resolve().parents[1] / "pouw" / "h1_prove"
sys.path.insert(0, str(H1))
sys.path.insert(0, str(H1.parent / "hidden_zk"))

import h1unit as U  # noqa: E402


def description(definition) -> dict:
    """The unit's Program descriptor (`codec.encode_program`), the description a templated read would be bounded by: its
    canonical bytes, its Definitions and their nodes, and the Program's gates."""
    from verity.ml.circuits import codec
    from verity.ml.circuits.program import Program

    prog = Program(definition)
    desc = codec.encode_program(prog)
    defs = desc.get("definitions", {})
    nodes = sum(len((d.get("body") or {}).get("nodes", [])) for d in defs.values() if isinstance(d, dict))
    return {"descriptor_bytes": codec.descriptor_size(desc), "definitions": len(defs), "nodes": nodes, "gates": prog.gates}


def main() -> None:
    k, work = int(sys.argv[1]), Path(sys.argv[2])
    work.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    import fast_stage
    fast_stage.install(work / "scratch")
    shape = U.Shape(k)
    stager = shape.stager("unit/pouw-h1-tile", circuit=work / "circuit.txt", flat=True)
    t1 = time.time()
    if "--circuit-only" in sys.argv[3:]:
        head = {"k": k, "ands": stager.low.layout.counts["ands"], "circuit_bytes": (work / "circuit.txt").stat().st_size,
                "build_seconds": round(t1 - t0, 1),
                "maxrss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2)}
        head |= description(shape.definition())
        (work / "build.json").write_text(json.dumps(head, indent=1))
        print("BUILT " + json.dumps(head), flush=True)
        return
    rng = np.random.default_rng(k)
    a = rng.integers(-128, 128, size=(shape.r, k), dtype=np.int16)
    b = rng.integers(-128, 128, size=(shape.c, k), dtype=np.int16)
    rows = {f"a{i}": U.row_words(a[i:i + 1]) for i in range(shape.r)}
    rows |= {f"b{j}": U.row_words(b[j:j + 1]) for j in range(shape.c)}
    rows["pub"] = U.pub_words(os.urandom(64), 0, (os.urandom(64), os.urandom(64))).reshape(1, -1)
    stager.reference_check = lambda _rows, _outs: None
    got = stager.stage({p: rows[p] for p in stager.row_ports}, work / "stmt", os.urandom(32))
    counts = stager.low.layout.counts
    head = {"k": k, "r": shape.r, "c": shape.c, "definition": shape.definition().id, "ands": counts["ands"],
            "unit_nnz": got["unit_nnz"], "k_log": stager.meta["k_log"], "vus_per_block": stager.meta["vus_per_block"],
            "pin": stager.pin, "circuit_bytes": (work / "circuit.txt").stat().st_size,
            "build_seconds": round(t1 - t0, 1), "stage_seconds": round(time.time() - t1, 1),
            "maxrss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2)}
    head |= description(shape.definition())
    (work / "build.json").write_text(json.dumps(head, indent=1))
    print("BUILT " + json.dumps(head), flush=True)


if __name__ == "__main__":
    main()
