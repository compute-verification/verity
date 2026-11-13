"""CIR-274 DSAMOE: rank 0's window (the families frame, dense and sparse: `p3_keycost.py`'s settings, run
r20261113-041114-3cb4) of the class-G request Program at LAYERS layers and PROMPT_IDS prompt ids, at each OUTPUTS, its stated
description written whole: `Circuit.description` (templates, wires, cuts, registered, order) and the record's ``blocks``
(`Circuit.blocks_record`: proof and inference entries), as canonical JSON (`interface.circuit_bytes`). Run from the
repository root:

    .venv/bin/python p4_describe.py OUT LAYERS PROMPT_IDS OUTPUTS...

Writes OUT/description_<PROMPT_IDS>_<OUTPUTS>.json for each, and OUT/describe.json: each file's bytes and SHA-256, the
description's own digest (SHA-512 of `circuit_bytes(description())`, the window census's) and bytes, and the blocks' bytes."""
import hashlib
import importlib.util
import json
import resource
import sys
import time
from pathlib import Path

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
layers, LP = int(sys.argv[2]), int(sys.argv[3])
spec = importlib.util.spec_from_file_location("compile_cost", "benchmarks/compile_cost/run.py")
cc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cc)

from verity.commitments.windows import interface as IF  # noqa: E402
from verity_vllm.program.predict import node_window as NW  # noqa: E402
from verity_vllm.program.predict import node_window_dense as ND  # noqa: E402
from verity_vllm.program.predict import node_window_sparse as NS  # noqa: E402
from verity_vllm.program.predict.steps import plan_steps, request_program  # noqa: E402

FAM = {**NW.FAMILIES, **ND.FAMILIES, **NS.FAMILIES}
rec: dict = {}
for OUTS in map(int, sys.argv[4:]):
    t0 = time.perf_counter()
    reg = cc.registered(0, layers, "eager")
    advice, ingress = cc.session([(0, LP, OUTS)])
    u = [x for x in plan_steps(reg, advice, ingress).requests if x is not None][0]
    p = request_program(reg, u)
    w = NW.NodeWindow(p.program.fn, FAM, NW._GRAIN, NW.NAME, 1, True).lower()
    c = w.circuit()
    desc = c.description()
    plain = IF.circuit_bytes(desc)
    blocks = c.blocks_record()
    blob = IF.circuit_bytes({**desc, "blocks": blocks})
    name = f"description_{LP}_{OUTS}.json"
    (out / name).write_bytes(blob)
    r = rec[name] = {"layers": layers, "ids": LP, "outputs": OUTS, "program_digest": p.digest, "steps": w.steps,
                     "templates": len(c.templates), "units": sum(c.units.values()), "wires": len(c.wires),
                     "inference_entries": len(blocks["inference"]["entries"]),
                     "bytes": len(blob), "sha256": hashlib.sha256(blob).hexdigest(),
                     "description_bytes": len(plain), "description_sha512": hashlib.sha512(plain).hexdigest(),
                     "blocks_bytes": len(IF.circuit_bytes({"blocks": blocks})),
                     "wall_s": round(time.perf_counter() - t0, 1),
                     "peak_gib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2)}
    (out / "describe.json").write_text(json.dumps(rec, indent=1) + "\n")
    print("DSAMOE describe", name, json.dumps(r), flush=True)
