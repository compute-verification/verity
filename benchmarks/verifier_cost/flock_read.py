"""flock_read.py K WORK: the H1 tile unit's flat statement circuit at contraction K (`prover/lowering/h1unit.Shape(K)`, the
4×8 tile, staged flat as `h1_prove/job.sh` stages it), for `flock_tmpl.sh` and `flock_unroll.sh`. Writes WORK/circuit.txt and
WORK/build.json (the unit's ANDs, the circuit's bytes, the build's seconds and its Program descriptor's size).
"""
from __future__ import annotations

import json
import resource
import sys
import time
from pathlib import Path

H1 = Path(__file__).resolve().parents[1] / "pouw" / "h1_prove"
sys.path.insert(0, str(H1))
sys.path.insert(0, str(H1.parent / "hidden_zk"))

import h1form as F  # noqa: E402


def description(definition) -> dict:
    """The unit's Program descriptor (`codec.encode_program`), the description a templated read would be bounded by: its
    canonical bytes, its Definitions and their nodes, and the Program's gates."""
    from verity.circuits import codec
    from verity.circuits.program import Program

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
    shape = F.shape(k)
    stager = shape.stager("unit/pouw-h1-tile", circuit=work / "circuit.txt")
    t1 = time.time()
    head = {"k": k, "ands": stager.low.layout.counts["ands"], "circuit_bytes": (work / "circuit.txt").stat().st_size,
            "build_seconds": round(t1 - t0, 1),
            "maxrss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2)}
    head |= description(shape.definition())
    (work / "build.json").write_text(json.dumps(head, indent=1))
    print("BUILT " + json.dumps(head), flush=True)


if __name__ == "__main__":
    main()
