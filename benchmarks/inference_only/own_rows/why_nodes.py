"""`served_record_case.why_nodes` on a case file: argv CASE.case.json."""
import json
import sys
import time
from pathlib import Path

tree = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(tree / "verity/protocols/workload_classification/training_detection/inference_only/verifier/lean/scripts"))
import served_record_case as SRC  # noqa: E402

case = json.loads(Path(sys.argv[1]).read_text())
t = time.perf_counter()
print(SRC.why_nodes(case), flush=True)
print("seconds", round(time.perf_counter() - t, 1))
