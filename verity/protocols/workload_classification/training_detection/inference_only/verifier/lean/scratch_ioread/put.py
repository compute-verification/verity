import importlib.util, json
from pathlib import Path
LEAN = Path("/tmp/wif/verity/protocols/workload_classification/training_detection/inference_only/verifier/lean")
def load(n):
    s = importlib.util.spec_from_file_location(n, LEAN/"scripts"/f"{n}.py"); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
v = load("record_difftest_vectors"); ref = v.ref
cases = json.load(open("/tmp/ioread/three_cases.json"))
for name in ("2x3 one input run handed to two rows", "composite plain alone"):
    c = cases[name]; inst, tr = c["instance"], c["trusted"]
    i, program, desc, rec, T, templates = ref.front(inst, tr)
    print(name, "own_rows passed (front)")
    wires, cuts = ref.wires_of(templates, rec, desc)
    seen = {}
    for k, w in enumerate(wires):
        for u, r, _ in ref.entries(w):
            key = (w["reader"].split("{")[0], w["port"], u, r)
            if key in seen: print("  wire", k, "and wire", seen[key], "fill", key)
            seen[key] = ("wire", k)
    reg = rec.get("registered", {})
    for r in reg.get("reads", []):
        rr = ref.pattern(r, ref.READ)
        for u, j, _ in ref.entries(rr):
            key = (rr["reader"].split("{")[0], rr["port"], u, j)
            if key in seen: print("  read", rr["param"], "and", seen[key], "fill", key)
            seen[key] = ("read", rr["param"])
    by = {t.tid: t for t in templates}
    for r in reg.get("reads", []):
        if r["reader"] not in by or by[r["reader"]].port(r["port"], True) is None:
            print("  read names no template port:", r["reader"].split("{")[0], r["port"], "templates:", [t.split("{")[0] for t in by])
