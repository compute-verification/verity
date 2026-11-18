import importlib.util, json, copy, traceback, sys
from pathlib import Path
LEAN = Path(__import__("os").environ.get("ROOT","/tmp/wif") + "/verity/protocols/workload_classification/training_detection/inference_only/verifier/lean")
def load(n):
    s = importlib.util.spec_from_file_location(n, LEAN/"scripts"/f"{n}.py"); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
vectors = load("record_difftest_vectors"); ref = vectors.ref; codec = vectors.codec
shaped = {m["shape"]: m for m in vectors.shaped()}
def why(label, inst, tr):
    try:
        print(label, "->", ref.check_detail(inst, tr)); 
    except Exception as e:
        print(label, "EXC", repr(e))
    try:
        ref.expand(inst, tr)
    except Exception as e:
        tb = traceback.extract_tb(e.__traceback__)
        print("   raised", type(e).__name__, e.args, "at", [(Path(f.filename).name, f.lineno, f.name) for f in tb[-3:]])
        print("   message:", getattr(e, "message", None) or getattr(e, "detail", None) or str(e)[:300])
cases = {}
for name in ("2x3 one input run handed to two rows", "2x3 a template in two root nodes"):
    c = shaped[name]["case"]; cases[name] = c; why(name, c["instance"], c["trusted"])
held = shaped["2x3 registered"]
gate = [[("op", 16, [(0, 0)])]]
plain = vectors.template("Plain7", [("p", vectors.Value(16))], [vectors.Value(16)], gate)
case = copy.deepcopy(held["case"]); vectors._alone(case, held["circuit"], plain); cases["composite plain alone"] = case
why("composite plain alone", case["instance"], case["trusted"])
json.dump(cases, open("/tmp/ioread/three_cases_" + __import__("os").environ.get("TAG","tip") + ".json", "w"))
