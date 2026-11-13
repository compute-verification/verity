"""CIR-274 DENSE: rank 0's class-G request Program at 43 prompt ids and 2 outputs in the registered rank form, lowered into a
node window with the family sets named (frame: `node_window.FAMILIES`, dense: `node_window_dense.FAMILIES`, sparse:
`node_window_sparse.FAMILIES`), its census, and beside it: each writer -> reader wire pattern by family (a gather by the
families it gathers), the free inputs by writer family and reader port, units by family and template, and each template's
rows a unit. COLS (default 0: the module's own) cuts the dense half's COLS families and fused all-reduce at that width
(`node_window_dense._cols` patched here only). Run from the repository root:

    .venv/bin/python p3_census.py OUT [LAYERS] [RANK] [SETS] [COLS]

Writes OUT/p3_l<LAYERS>_r<RANK>_<SETS>_c<COLS>.json and one DENSE line a stage."""
import importlib.util
import json
import resource
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

out = Path(sys.argv[1])
layers = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2] else 4
rank = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3] else 0
sets = sys.argv[4] if len(sys.argv) > 4 and sys.argv[4] else "frame,dense,sparse"
cols = int(sys.argv[5]) if len(sys.argv) > 5 and sys.argv[5] else 0
out.mkdir(parents=True, exist_ok=True)
spec = importlib.util.spec_from_file_location("compile_cost", "benchmarks/compile_cost/run.py")
cc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cc)

from verity_vllm.program.predict import node_window as NW, node_window_dense as ND  # noqa: E402
from verity_vllm.program.predict.steps import plan_steps, request_program  # noqa: E402
from verity_vllm.program.predict.window_builder import _rows  # noqa: E402

SETS = {"frame": NW.FAMILIES, "dense": ND.FAMILIES}
if "sparse" in sets:
    from verity_vllm.program.predict import node_window_sparse as NS  # noqa: E402
    SETS["sparse"] = NS.FAMILIES
families: dict = {}
for s in sets.split(","):
    families.update(SETS[s])
if cols:
    _orig = ND._cols
    ND._cols = lambda N, most: _orig(N, cols)

LP, T = 43, 1
path = out / f"p3_l{layers}_r{rank}_{sets.replace(',', '+')}_c{cols}.json"
rec: dict = {"LP": LP, "outputs": T + 1, "layers": layers, "rank": rank, "sets": sets, "cols": cols}


def peak() -> float:
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576, 3)


def save(label: str, d: dict) -> None:
    path.write_text(json.dumps(rec, indent=1, default=str) + "\n")
    print("DENSE p3", label, json.dumps(d, default=str)[:3000], flush=True)


def pairs(w: NW.NodeWindow) -> dict[str, int]:
    """Wire patterns (before coalescing) by writer and reader family; a gather's writer named by the families its own
    allocation's wires read, through any gathers below it."""
    import bisect
    fam = w.family_of

    def alloc(tid: str, u: int) -> tuple[str, int]:
        a = w.allocs[tid]
        return tid, a[bisect.bisect_right([x[0] for x in a], u) - 1][0]

    src: dict[tuple[str, int], set] = {}
    for x in w.b.wires:
        if fam(x.reader, x.at) == NW.GATHER:
            v = alloc(x.writer, x.base) if fam(x.writer, x.base) == NW.GATHER else fam(x.writer, x.base)
            src.setdefault(alloc(x.reader, x.at), set()).add(v)
    for _ in range(4):
        for t, s in src.items():
            src[t] = {y for v in s for y in (src.get(v, {v}) if isinstance(v, tuple) else {v})}
    got: Counter[str] = Counter()
    for x in w.b.wires:
        wf = fam(x.writer, x.base)
        if wf == NW.GATHER:
            wf = "gather<" + "+".join(sorted(map(str, src.get(alloc(x.writer, x.base), ())))) + ">"
        got[f"{wf} -> {fam(x.reader, x.at)} {x.reader.split('{')[0]}.{x.port}"] += 1
    return dict(sorted(got.items()))


t0 = time.perf_counter()
reg = cc.registered(rank, layers, "eager")
advice, ingress = cc.session([(0, LP, T + 1)])
u = [x for x in plan_steps(reg, advice, ingress).requests if x is not None][0]
p = request_program(reg, u)
rec["program"] = {"digest": p.digest, "nodes": len(p.program.fn.body.nodes), "wall_s": round(time.perf_counter() - t0, 1),
                  "peak_gib": peak()}
save("program", rec["program"])
t0 = time.perf_counter()
try:
    circuit, w = NW.node_window(p.program.fn, families=families)
    cen = w.census(circuit)
    free: dict[str, list[int]] = {}
    for f in w.free:
        k = f"{'peers' if f.kind == 'peer' else f.family} -> {f.tid.split('{')[0]}.{f.port} rows {f.rows[0]}/{f.rows[2]}"
        free.setdefault(k, [0, 0])[0] += f.count
        free[k][1] += 1
    by_fam: Counter[str] = Counter()
    for tid, allocs in w.allocs.items():
        for _at, n, fm in allocs:
            by_fam[f"{fm} {tid}"] += n
    rec["window"] = {**cen, "templates_total": len(circuit.units), "units_total": sum(circuit.units.values()),
                     "units_by_template": dict(sorted(circuit.units.items())), "units_by_family_template": dict(sorted(by_fam.items())),
                     "rows_a_unit": {tid: _rows(fn, circuit.cuts.get(tid)) for tid, fn in w.b.fns.items()},
                     "cuts": circuit.cuts, "pairs": pairs(w), "free_by_port": dict(sorted(free.items())),
                     "free_list": [vars(f) for f in w.free]}
except BaseException as e:  # noqa: BLE001 - a refusal is a result
    rec["window_error"] = {"error": f"{type(e).__name__}: {e}"[:3000], "trace": traceback.format_exc()[-4000:]}
rec["window_wall_s"], rec["peak_gib"] = round(time.perf_counter() - t0, 1), peak()
save("window", {k: v for k, v in (rec.get("window") or rec.get("window_error")).items()
                if k not in ("peers", "free_list", "units_by_template", "units_by_family_template", "rows_a_unit", "pairs",
                             "free_by_port", "trace")} | {"wall_s": rec["window_wall_s"], "peak_gib": rec["peak_gib"]})
