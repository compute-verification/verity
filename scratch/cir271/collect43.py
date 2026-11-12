"""CIR-271 (REQBLOCK): class G's TP window of collective sites (`pipeline.tp.wires.sites_circuit`) at the smallest class keeping
every Definition, one request of 43 prompt ids prefilled in one step and decoding once, as `test_tp_steps.class_g_tp` does at one
and two prompt ids. Run from adapters/vllm:

    ../../.venv/bin/python ../../scratch/cir271/collect43.py OUT LAYERS LP

Writes OUT/collect_<LP>_l<LAYERS>.json: the sites, combine Definitions, units by template, wires, reads left free (by kind),
units C-Flock's composer refuses (`proof_blocks`) and the first backward wire."""
import json
import re
import resource
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

import importlib.util  # noqa: E402

sys.path.insert(0, str(Path.cwd()))
from tests.pipeline.test_tp_wires import in_order, proof_blocks  # noqa: E402
from tests.program.predict.test_steps import GREEDY, _session  # noqa: E402

# `test_steps._class_g` reads a workload file that isn't in the tree; compile_cost's φ is class G's registration (node1.py's).
spec = importlib.util.spec_from_file_location("compile_cost", Path.cwd().parents[1] / "benchmarks/compile_cost/run.py")
cc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cc)
from verity_vllm.pipeline.tp import wires as TW  # noqa: E402
from verity_vllm.program.predict.tp_steps import compile_tp  # noqa: E402

from verity.circuits.codec import descriptor_id  # noqa: E402

out, layers, lp = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
out.mkdir(parents=True, exist_ok=True)
t0, rec = time.perf_counter(), {"layers": layers, "LP": lp, "outputs": 2}
try:
    once = {**GREEDY, "max_tokens": 2, "ignore_eos": True}
    advice, ingress = _session([(lp, 2, once)], [[(0, 0, lp)], [(0, lp, 1)]])
    c = compile_tp(cc.registered(0, layers, "eager"), advice, ingress)
    rec["compile_s"] = round(time.perf_counter() - t0, 1)
    live = [b for b in c.bindings if b is not None]
    sites = TW.sites_circuit(c.bindings, "class-g")
    try:
        in_order(sites)
        backward = None
    except AssertionError as e:
        backward = str(e)[:2000]
    kinds = Counter(re.sub(r"\[\d+, \d+\)", "[..)", re.sub(r"request \d+'s collective \d+ \[\d+, \d+\): rank \d+'s node \d+ ", "", u))
                    for u in sites.unwired)
    rec.update({"workload": c.workload.digest, "sites": [len(x.sites) for x in live],
                "combines": sorted({b.fns[0][s.combine[0][0]].id for b in live for s in b.sites}),
                "units": dict(sorted(sites.circuit.units.items())), "templates": len(sites.circuit.templates),
                "template_ids": sorted(descriptor_id(fn) for fn, _ in sites.circuit.templates),
                "wires": len(sites.circuit.wires), "unwired": dict(sorted(kinds.items())), "proof_blocks": proof_blocks(sites),
                "backward": backward})
except BaseException as e:  # noqa: BLE001 - a refusal is a result
    rec.update({"error": f"{type(e).__name__}: {e}"[:3000], "trace": traceback.format_exc()[-3000:]})
rec["wall_s"] = round(time.perf_counter() - t0, 1)
rec["peak_gib"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576, 2)
(out / f"collect_{lp}_l{layers}.json").write_text(json.dumps(rec, indent=1) + "\n")
print("REQBLOCK collect", json.dumps({k: v for k, v in rec.items() if k not in ("units", "template_ids", "trace")})[:3000], flush=True)
