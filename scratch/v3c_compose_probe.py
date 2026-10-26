"""V3c scratch probe (not for main): with #4813, #4818 and #4820 merged, how far does a TINY V3b window get toward a commit?
Members, the rows window (wiring validation), the verifier's window_of_record, then interface.Window."""
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.getcwd(), "integrations", "vllm"))
from tests.program.predict import test_window_circuit as T  # noqa: E402

from verity.ml.windows import interface as IF, rows as CS  # noqa: E402

LAW, RUN, out = {"law": "stratified", "k": 2}, "ab" * 64, {}


def step(name, f):
    t = time.perf_counter()
    try:
        r = f()
        out[name] = {"ok": True, "s": round(time.perf_counter() - t, 1), **(r or {})}
    except Exception as e:  # noqa: BLE001
        out[name] = {"ok": False, "s": round(time.perf_counter() - t, 1), "error": f"{type(e).__name__}: {e}"[:600]}
    print(name, json.dumps(out[name]), flush=True)
    return out[name]["ok"]


circuit = T._circuit(T.WINDOW)
state = {}
step("members", lambda: state.update(ms=IF.members(circuit, 64)) or {
    "members": len(state["ms"]), "wired_and_tables": sum(1 for m in state["ms"] if m.wired and m.tables)})
step("program", lambda: state.update(pq=circuit.program()) or {})
step("rows.Window", lambda: CS.Window(*state["pq"], LAW, RUN, state["ms"]) and {})
step("descriptor_record", lambda: state.update(doc=IF.descriptor_record(circuit, 64)) or {})
step("window_of_record", lambda: {"same_members": IF.window_of_record(
    {**state["doc"], "run": RUN, "context": ""}, LAW, IF.resolve).members == state["ms"]})
step("interface.Window", lambda: IF.Window(circuit, RUN, LAW, 64) and {})
json.dump(out, open(os.path.join(os.environ.get("OUT", "."), "compose_probe.json"), "w"), indent=1)
