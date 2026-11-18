import json, importlib.util
from pathlib import Path
LEAN = Path("/tmp/wif/verity/protocols/workload_classification/training_detection/inference_only/verifier/lean")
s = importlib.util.spec_from_file_location("r", LEAN/"scripts"/"inference_only_reference.py"); R = importlib.util.module_from_spec(s); s.loader.exec_module(R)
cases = json.load(open("/tmp/ioread/three_cases.json"))
for name in ("2x3 a template in two root nodes",):
    desc = json.loads(cases[name]["instance"]["program"])
    try: R.codec.decode_program(desc, R.declared(desc))
    except Exception as e: print(str(e)[-400:])
    root = desc["definitions"][desc["root"]]["body"]["nodes"]
    cal = desc["callees"]
    for i, n in enumerate(root):
        print(i, json.dumps(n)[:330])
