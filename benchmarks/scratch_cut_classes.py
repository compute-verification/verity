import json
import resource
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tests/adapters/vllm"), str(ROOT), str(ROOT / "adapters/vllm"), str(ROOT / "adapters/torch"),
                str(ROOT / "benchmarks/harness"), str(ROOT / "tools/config")]

import numpy as np

from verity.circuits.codec import program_digest
from verity.circuits.population import cut_wires, wired_population_program
from verity.circuits.program import is_input
from verity.circuits.refs import runs


def rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20


def build(name: str):
    from tests.program.predict import test_window_circuit_fa3 as F
    from verity_vllm.program.model import INT8_EXACT_QUANT
    from verity_vllm.program.predict.window_circuit import PearlTables, WindowClass, class_circuit
    llama = json.loads(F.LLAMA_1B.read_text())
    if name == "tied":
        return class_circuit(F._registered(config=llama), WindowClass(4, 27, 2), tables=PearlTables(F.PEARL))
    if name == "untied":
        return class_circuit(F._registered(config=llama), WindowClass(4, 27, 2))
    if name == "int8":
        config = {**llama, "quantization_config": dict(INT8_EXACT_QUANT)}
        return class_circuit(F._registered({**F.H200, "int8_exact": "linears"}, config), WindowClass(4, 27, 2))
    if name == "tiny":
        from tests.program.predict import test_window_circuit as W
        return class_circuit(F._registered(), W.CLASS, None, W.GRAIN)
    raise SystemExit(f"no class {name}")


def expected(circuit) -> dict:
    out: dict = {}
    for w in circuit.wires:
        i = np.arange(w.units, dtype=np.int64)
        rows = w.cut[0] + i % w.cut[1]
        for c in np.unique(rows).tolist():
            m = rows == c
            out.setdefault((w.reader, w.port, c), {}).setdefault(f"{w.writer}/{w.out}", []).append(
                (w.at + i[m] // w.cut[1], w.source(i[m])))
    return {k: {t: tuple(np.concatenate(x) for x in zip(*ps)) for t, ps in v.items()} for k, v in out.items()}


def same(a: dict, b: dict) -> bool:
    if set(a) != set(b):
        return False
    for k in a:
        if set(a[k]) != set(b[k]):
            return False
        for t in a[k]:
            (ra, wa), (rb, wb) = (tuple(map(np.asarray, a[k][t])), tuple(map(np.asarray, b[k][t])))
            oa, ob = np.argsort(ra, kind="stable"), np.argsort(rb, kind="stable")
            if not (np.array_equal(ra[oa], rb[ob]) and np.array_equal(wa[oa], wb[ob])):
                return False
    return True


def edges(program) -> int:
    nodes = program.root.body.nodes
    return sum(1 for nd in nodes for a in nd.args for s, i, *_ in runs(a.refs) if s == "n" and not is_input(nodes[i].fn))


name = sys.argv[1]
t0 = time.perf_counter()
circuit = build(name)
t1 = time.perf_counter()
record = {**circuit.description(), "external": list(circuit.external)}
wired, _ = wired_population_program(list(circuit.templates), record)
t2 = time.perf_counter()
r_before = rss_gb()
cut, query, positions = cut_wires(wired, record)
t3 = time.perf_counter()
r_after = rss_gb()
want, want_query = wired_population_program(list(circuit.templates), {**record, "wires": []})
t4 = time.perf_counter()
exp = expected(circuit)
cuts = circuit.cuts
rows = sum(len(a) for v in positions.values() for a, _ in v.values())
part = sum(len(a) for (r, p, _c), v in positions.items() for a, _ in v.values() if int(cuts.get(r, {}).get(p, 1)) > 1)
print(json.dumps({
    "class": name, "templates": len(circuit.templates), "units": sum(n for _, n in circuit.templates), "wires": len(circuit.wires),
    "wired_digest": program_digest(wired), "cut_digest": program_digest(cut), "no_wires_call_digest": program_digest(want),
    "digest_equal": program_digest(cut) == program_digest(want), "query_equal": query == want_query,
    "root_nodes": {"wired": sum(not is_input(nd.fn) for nd in wired.root.body.nodes),
                   "cut": sum(not is_input(nd.fn) for nd in cut.root.body.nodes)},
    "root_params": {"wired": len(wired.fn.signature()[0]), "cut": len(cut.fn.signature()[0])},
    "edges": {"wired": edges(wired), "cut": edges(cut)},
    "positions_rows": rows, "positions_rows_of_cut_ports": part, "positions_rows_of_whole_ports": rows - part,
    "wire_closed_form_rows": sum(w.units for w in circuit.wires),
    "positions_equal_wires_closed_form": same(positions, exp),
    "seconds": {"class_build": round(t1 - t0, 1), "wired_program": round(t2 - t1, 1), "cut": round(t3 - t2, 1),
                "no_wires_call": round(t4 - t3, 1)},
    "peak_rss_gb": {"before_cut": round(r_before, 2), "after_cut": round(r_after, 2)},
}, indent=1))
