"""A block of the served circuit of the class of record (Llama 3.2 1B, class 4,27,2, four requests, the RTX target, seed
7: circuits' `build_descriptor.py`) through the door, `proofs.verify`, on its one partition (`Q_template_instances`, the
lowering's own query) with blocks ``[0, u), [u, u + 1), [u + 1, population)``, ``u`` a template's first unit.

  python served_door.py build --tree TREE --out DIR                     # DIR/C.json, DIR/query.json
  python served_door.py step block|restrict|lower|door|one --dir DIR --template ID

``block``, ``restrict`` and ``lower`` are the door's own steps (`one_circuit.block`, `.restrict`, `.lower`), each in its
own process; ``door`` is `proofs.verify` on the drawn block, whose proof files (a public file naming the block's unit and
an empty session) and registered file (empty) carry the call to the first step that refuses, and are read by no step
before it; ``one`` is `TemplatePartition` under ``Q_template_instance`` of the template alone. Each prints one
``STEP {json}`` line."""
from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import resource
import sys
import time
from pathlib import Path


def _peak_gib() -> float:
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 3)


def build(a) -> int:
    tree, out = Path(a.tree), Path(a.out)
    sys.path.insert(0, str(tree / "tests/adapters/vllm"))
    from verity.circuits.codec import descriptor_chunks
    from verity_vllm.program.predict import window_circuit as WCirc
    from verity_vllm.program.predict import window_class as WC
    from tests.program.predict import test_steps as S
    t = time.perf_counter()
    cfg = json.loads((tree / "tests/adapters/vllm/tests/program/predict/fixtures/llama32-1b.config.json").read_text())
    reg = S.Registered.of(S.Phi(model={"repo": "org/m", "revision": "r", "hf_config": cfg}, target=dict(S.RTXPRO6000)),
                          default_seed=7)
    circuit = WCirc.lower(reg, WC.WindowClass(4, 27, 2), None, requests=[WC.Request(0, j, 27, 2) for j in range(4)]).circuit
    program, query = circuit.program()
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out / "C.json", "wb") as f:
        for ch in descriptor_chunks(program):
            f.write(ch)
            n += len(ch)
    (out / "query.json").write_text(json.dumps(query))
    print("STEP " + json.dumps({"step": "build", "wall_s": round(time.perf_counter() - t, 1), "peak_gib": _peak_gib(),
                                "C_bytes": n, "root_nodes": len(program.root.body.nodes)}), flush=True)
    return 0


def _program(c: bytes):
    from verity.circuits import restrict as R
    return R.program(c)


def _partition(d: Path, template: str):
    """The served partition with the template's first unit a block of its own, and that block's index."""
    from verity.circuits import partition_object as PO
    p = d / f"partition-{hashlib.sha256(template.encode()).hexdigest()[:16]}.json"
    if p.exists():
        j = json.loads(p.read_text())
        return j["partition"], j["k"], j["u"], j["population"]
    c = (d / "C.json").read_bytes()
    query = json.loads((d / "query.json").read_text())
    program = _program(c)
    tp = PO.TemplatePartition(program, query)
    u = next(a for _k, t, a, n in tp.ranges if t == template and n)
    blocks = [b for b in ((0, u), (u, u + 1), (u + 1, tp.population)) if b[0] < b[1]]
    part = PO.build(json.loads(c), query, blocks)
    k = blocks.index((u, u + 1))
    p.write_text(json.dumps({"partition": part, "k": k, "u": u, "population": tp.population}))
    return part, k, u, tp.population


def _ports(c: bytes, query: dict, u: int) -> dict:
    """Unit ``u``'s parameter leaves by what each reads: another unit's output, a registered input, a run input."""
    from verity.circuits import partition_object as PO
    from verity.circuits.program import REGISTERED_ID, is_input
    program = _program(c)
    tp = PO.TemplatePartition(program, query)
    body = program.root.body
    offs = list(body.offsets)
    unit_of = {k: (a, t) for k, t, a, _n in tp.ranges}
    counts = {"wired": 0, "registered_input": 0, "run_input": 0}
    writers: dict[str, int] = {}
    for g in tp.inputs(u):
        k = bisect.bisect_right(offs, g) - 1
        fn = body.nodes[k].fn
        if is_input(fn):
            counts["registered_input" if REGISTERED_ID.fullmatch(fn.id) else "run_input"] += 1
        else:
            a, t = unit_of[k]
            counts["wired"] += 1
            writers[t] = writers.get(t, 0) + 1
    return {**counts, "writers": writers, "leaves": len(tp.inputs(u))}


def step(a) -> int:
    from verity.protocols.proofs.spot.verifier.one_stage import one_circuit as OC
    d = Path(a.dir)
    c = (d / "C.json").read_bytes()
    t = time.perf_counter()
    rec: dict = {"step": a.name, "template": a.template}
    if a.name == "one":
        from verity.circuits import partition_object as PO
        try:
            PO.TemplatePartition(_program(c), PO.template_instance_query(a.template))
            rec["refused"] = None
        except PO.QueryInapplicable as e:
            rec["refused"] = str(e)[:600]
    else:
        part, k, u, pop = _partition(d, a.template)
        rec |= {"block": k, "unit": u, "population": pop, "partition_s": round(time.perf_counter() - t, 1)}
        t = time.perf_counter()
        if a.name == "block":
            rec["units"] = OC.block(c, part, k)
            rec["ports"] = _ports(c, json.loads((d / "query.json").read_text()), u)
        elif a.name in ("restrict", "lower"):
            c_s = OC.restrict(c, part, [(u, u + 1)])
            rec |= {"restrict_s": round(time.perf_counter() - t, 1), "c_s_bytes": len(c_s)}
            (d / f"c_s-{u}.json").write_bytes(c_s)
            if a.name == "lower":
                from verity.protocols.proofs.spot.verifier.one_stage import proofs as PF
                t = time.perf_counter()
                try:
                    rec["circuit_bytes"] = len(OC.lower(c_s, part, PF.lower_executable))
                    rec["refused"] = None
                except (OC.NotDerived, OC.Refused) as e:
                    rec["refused"] = f"{type(e).__name__}: {e}"
        elif a.name == "door":
            from verity.protocols.proofs.spot.verifier.one_stage import proofs as PF
            f = d / f"door-{u}"
            (f / "session").mkdir(parents=True, exist_ok=True)
            (f / "session" / "session.json").write_text("{}")
            (f / "pub.bin").write_text(json.dumps({"units": {"indices": [u]}}) + "\n")
            (f / "circuit.txt").write_text("")
            found, refusal = PF.verify(PF.Proof(f / "circuit.txt", f / "pub.bin", (f / "session",)), program=c,
                                       partition=part, draw={"units": [k]}, registered={})
            rec |= {"findings": found, "refusal": refusal}
    rec |= {"wall_s": round(time.perf_counter() - t, 1), "peak_gib": _peak_gib()}
    print("STEP " + json.dumps(rec, default=str), flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--tree", required=True)
    b.add_argument("--out", required=True)
    s = sub.add_parser("step")
    s.add_argument("name", choices=("block", "restrict", "lower", "door", "one"))
    s.add_argument("--dir", required=True)
    s.add_argument("--template", required=True)
    a = ap.parse_args()
    return build(a) if a.cmd == "build" else step(a)


if __name__ == "__main__":
    sys.exit(main())
