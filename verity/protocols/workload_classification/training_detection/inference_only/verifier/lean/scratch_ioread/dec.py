import json, traceback, importlib.util
from pathlib import Path
LEAN = Path("/tmp/wif/verity/protocols/workload_classification/training_detection/inference_only/verifier/lean")
s = importlib.util.spec_from_file_location("r", LEAN/"scripts"/"inference_only_reference.py"); R = importlib.util.module_from_spec(s); s.loader.exec_module(R)
cases = json.load(open("/tmp/ioread/three_cases.json"))
text = cases["2x3 a template in two root nodes"]["instance"]["program"]
desc = json.loads(text)
try:
    R.codec.decode_program(desc, R.declared(desc))
except Exception as e:
    tb = traceback.extract_tb(e.__traceback__)
    print(type(e).__name__, str(e)[:600]); print([(Path(f.filename).name, f.lineno, f.name) for f in tb[-4:]])
root = desc["definitions"][desc["root"]]["body"]["nodes"]
for i, n in enumerate(root): print(i, n.get("form"), desc["callees"].get(n.get("fn")) if "fn" in n else None, json.dumps({k: v for k, v in n.items() if k not in ("form",)})[:260])
