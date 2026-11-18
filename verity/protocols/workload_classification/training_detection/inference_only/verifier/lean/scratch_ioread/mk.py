import importlib.util, json, copy
from pathlib import Path
LEAN = Path("/tmp/wif/verity/protocols/workload_classification/training_detection/inference_only/verifier/lean")
def load(n):
    s = importlib.util.spec_from_file_location(n, LEAN/"scripts"/f"{n}.py"); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
vectors = load("record_difftest_vectors"); ref = vectors.ref; R = ref.reference
line = lambda o: json.dumps(o, separators=(",", ":"), sort_keys=True) + "\n"
shaped = {m["shape"]: m for m in vectors.shaped()}
held = shaped["2x3 registered"]
gate = [[("op", 16, [(0, 0)])]]
plain = vectors.template("Plain7", [("p", vectors.Value(16))], [vectors.Value(16)], gate)
comp = copy.deepcopy(held["case"]); vectors._alone(comp, held["circuit"], plain)
rec = [("2x3 registered", held["case"]), ("4x35 registered", shaped["4x35 registered"]["case"]),
       ("2x3 one input run handed to two rows", shaped["2x3 one input run handed to two rows"]["case"]),
       ("2x3 a template in two root nodes", shaped["2x3 a template in two root nodes"]["case"]),
       ("2x3 composite plain7 alone", comp)]
with open("/tmp/ioread/data/record_cases.jsonl", "w") as a, open("/tmp/ioread/data/record_python.jsonl", "w") as b, open("/tmp/ioread/data/record_names.txt", "w") as c:
    for n, k in rec:
        a.write(line(k)); r = vectors.results(k); b.write(line(r)); c.write(n + "\t" + json.dumps(r["detail"]) + "\n")
with open("/tmp/ioread/data/desc_instances.jsonl", "w") as a, open("/tmp/ioread/data/desc_expanded.jsonl", "w") as e, open("/tmp/ioread/data/desc_python.jsonl", "w") as b:
    for size, blocks in (("2x3", 2), ("4x35", 4)):
        case = shaped[f"{size} registered"]["case"]; text = case["instance"]["program"]
        form = vectors.results(case)["detail"]
        inp, _, _ = R.root_nodes(text)
        inst = {"program": text, "theta": [], "owner": ["input" if i else 0 for i in inp], "gamma": form["gates"] * blocks, "alpha": [vectors.BIG, 1]}
        res = R.check_desc(inst); a.write(line(inst)); b.write(line(res)); e.write(line(None if "error" in res else R.expand(inst)))
        print(size, "desc", {k: v for k, v in res.items() if k != "detail"})
