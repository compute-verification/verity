import json
import resource
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tests/adapters/vllm"), str(ROOT), str(ROOT / "adapters/vllm"), str(ROOT / "adapters/torch"),
                str(ROOT / "benchmarks/harness"), str(ROOT / "tools/config")]

from verity.commitments.periods import interface as I  # noqa: E402


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




def per_leaf(ty) -> int:
    return sum(-(-w // 16) for w in ty.leaf_widths())


def packed(ty) -> int:
    return -(-sum(ty.leaf_widths()) // 16)


for name in sys.argv[1:]:
    t0 = time.perf_counter()
    circuit = build(name)
    fns = dict(zip(circuit.ids, (fn for fn, _ in circuit.templates)))
    fields = {t: I._fields(fn.signature()[1]) for t, fn in fns.items()}
    narrow_templates = sorted(t for t, fs in fields.items() if any(packed(f) != per_leaf(f) for f in fs))
    behind = own = 0
    behind_units = own_units = 0
    examples = []
    outs = {}
    for w in circuit.wires:
        fs, j = fields[w.writer], int(w.out[1:])
        outs[w.out] = outs.get(w.out, 0) + 1
        if any(packed(f) != per_leaf(f) for f in fs[:j]):
            behind += 1
            behind_units += w.units
            examples.append(w.doc())
        if packed(fs[j]) != per_leaf(fs[j]):
            own += 1
            own_units += w.units
    print(json.dumps({
        "class": name, "templates": len(fns), "wires": len(circuit.wires), "wire_rows": sum(w.units for w in circuit.wires),
        "wires_by_out_field": dict(sorted(outs.items(), key=lambda kv: int(kv[0][1:]))),
        "templates_with_a_field_whose_packed_words_differ_from_per_leaf_words": narrow_templates,
        "wires_reading_behind_such_a_field": behind, "their_rows": behind_units, "examples": examples[:3],
        "wires_reading_such_a_field_itself": own, "their_rows_": own_units,
        "seconds": round(time.perf_counter() - t0, 1),
        "peak_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2)}))
