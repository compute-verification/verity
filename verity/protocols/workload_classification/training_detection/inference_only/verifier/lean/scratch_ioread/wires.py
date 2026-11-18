import importlib.util, json, math, collections
from pathlib import Path
LEAN = Path("/tmp/wif/verity/protocols/workload_classification/training_detection/inference_only/verifier/lean")
def load(n):
    s = importlib.util.spec_from_file_location(n, LEAN/"scripts"/f"{n}.py"); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
vectors = load("record_difftest_vectors"); ref = vectors.ref
shaped = {m["shape"]: m for m in vectors.shaped()}
case = shaped["2x3 registered"]["case"]; inst, tr = case["instance"], case["trusted"]
e = ref.expand(inst, tr)
by = {t.tid: t for t in e["templates"]}
tot = 0; rows = collections.Counter()
for w in e["wires"]:
    n = math.prod(w["radix"]); k = int(e["cuts"].get(w["reader"], {}).get(w["port"], 1))
    b = 16 * by[w["reader"]].port(w["port"]) // k; tot += n*b
    rows[(w["writer"].split("{")[0], w["out"], "->", w["reader"].split("{")[0], w["port"], f"{b} bits/row")] += n
for r, n in sorted(rows.items()): print(n, "rows", r, "=", n*int(r[-1].split()[0]), "bits")
print("total wired bits", tot, "inside per slot", e["inside"])
for t in e["templates"]:
    print(t.tid.split("{")[0], "units", t.units, "rows(port,words)", [(p, wd) for p, wd, _ in t.rows])
for t in e["templates"]:
    if t.tid.split("{")[0] in ("LowEmbed_v1", "LowScan_v1", "LowFeed_v1"):
        p, r = t.fn.signature(); print(t.tid.split("{")[0], "params", [(n, repr(ty)[:60]) for n, ty in p], "ret", repr(r)[:200])
R = ref.reference
width, is_input, _, reads = R.circuit(inst["program"])
read = set().union(*reads)
print("input gates by width, read/unread:", collections.Counter((width[g], g in read) for g in range(len(width)) if is_input[g]))
