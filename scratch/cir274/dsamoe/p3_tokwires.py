"""CIR-274 DSAMOE: why BatchedGemm's and SparseMlaSplit's wire patterns grow with prompt ids (4 at 43, 16,387 at 86 for
BatchedGemm in `p3_keycost.py`'s run), at 4 layers, rank 0, 2 outputs. For each prompt-id count given, lowers the joint
window as keycost does and writes, for the wires whose reader is of those two families: counts by (reader, port, writer,
out), by radix and step shape, by step, and a few examples of each group; and the same families' wires before coalescing.
Run from the repository root:

    .venv/bin/python p3_tokwires.py OUT 43 86

Writes OUT/tokwires_p<P>.json and one DSAMOE line each."""
import bisect
import importlib.util
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
spec = importlib.util.spec_from_file_location("compile_cost", "benchmarks/compile_cost/run.py")
cc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cc)

from verity_vllm.program.predict import node_window as NW  # noqa: E402
from verity_vllm.program.predict import node_window_dense as ND  # noqa: E402
from verity_vllm.program.predict import node_window_sparse as NS  # noqa: E402
from verity_vllm.program.predict.steps import plan_steps, request_program  # noqa: E402

FAMS = ("BatchedGemm", "SparseMlaSplit")
reg = cc.registered(0, 4, "eager")
families = {**NW.FAMILIES, **ND.FAMILIES, **NS.FAMILIES}
for LP in map(int, sys.argv[2:]):
    advice, ingress = cc.session([(0, LP, 2)])
    u = [x for x in plan_steps(reg, advice, ingress).requests if x is not None][0]
    p = request_program(reg, u)
    w = NW.NodeWindow(p.program.fn, families, NW._GRAIN, NW.NAME, 1, True).lower()
    circuit = w.circuit()

    def step_of(tid: str, at: int) -> int:
        spans = w.b.spans[tid]
        return spans[bisect.bisect_right([s[0] for s in spans], at) - 1][3]

    def summary(ws) -> dict:
        mine = [x for x in ws if w.family_of(x.reader, x.at) in FAMS]
        groups: dict = defaultdict(list)
        for x in mine:
            groups[(w.family_of(x.reader, x.at), x.reader, x.port, x.writer, x.out)].append(x)
        rows = []
        for k, xs in sorted(groups.items(), key=lambda kv: -len(kv[1])):
            rows.append({"family": k[0], "reader": k[1], "port": k[2], "writer": k[3], "out": k[4], "n": len(xs),
                         "by_step": dict(sorted(Counter(step_of(x.reader, x.at) for x in xs).items())),
                         "radix_step": dict(Counter(f"{x.radix}/{x.step}/cut{tuple(x.cut)}" for x in xs).most_common(4)),
                         "examples": [x.doc() for x in xs[:4]]})
        return {"n": len(mine), "groups": rows[:30]}

    nodes = [n for n in p.program.fn.body.nodes if NW._family(n) in FAMS]
    rec = {"LP": LP, "program": p.digest, "steps": w.steps,
           "nodes": dict(Counter(NW._family(n) for n in nodes)),
           "templates": {t: n for t, n in w.b.units.items() if any(f in t for f in ("BatchedGemm", "SparseMla", "Mla"))},
           "after": summary(circuit.wires), "before": summary(w.b.wires)}
    (out / f"tokwires_p{LP}.json").write_text(json.dumps(rec, indent=1, default=str) + "\n")
    print("DSAMOE tokwires", LP, json.dumps({"after": rec["after"]["n"], "before": rec["before"]["n"], "steps": w.steps,
          "top": [(g["family"], g["reader"], g["port"], g["writer"], g["out"], g["n"]) for g in rec["after"]["groups"][:8]]},
          default=str)[:3000], flush=True)
