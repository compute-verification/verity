"""The served road loads no module of the fold, the match or the replay driver.

A fresh interpreter imports the road's entry modules: serve-openai with its seal, `predict` (`row`, `aot`), `advice`,
`block-counts`, and the adapter modules `prove_path.py` imports for `window-circuit`, `window-commit` and
`weights-registration`. No module under `observe/fold`, `check/match`, `pipeline/match*`, nor `check/replay/driver` or
its eight companions, may then be in `sys.modules`; and no function of a module the road loaded may import one, but
those `HOLDERS` names (a list that only shrinks).
"""
from __future__ import annotations

import json
import subprocess
import sys

ENTRIES = (
    "verity_vllm.pipeline.serve_openai", "verity_vllm.pipeline.serve_seal", "verity_vllm.pipeline.seal_process",
    "verity_vllm.program.predict.cli", "verity_vllm.program.predict.aot", "verity_vllm.pipeline.advice",
    "verity_vllm.program.predict.block_counts", "verity_vllm.program.predict.window_circuit",
    "verity_vllm.pipeline.advice_windows", "verity_vllm.pipeline.deferred_rows", "verity_vllm.pipeline.edge",
    "verity_vllm.pipeline.window_advice", "verity_vllm.pipeline.tp.serving_rows", "verity_vllm.check.weights_of_record",
    "verity_vllm.check.oracle_compare", "verity_vllm.check.replay.index", "verity_vllm.check.replay.opening",
    "verity_vllm.pipeline.cpu_replay", "verity_vllm.program.predict.scan_state", "verity_vllm.program.predict.window_rows",
    "verity_vllm.program.predict.window_weights", "verity_vllm.program.predict.build", "verity_vllm.program.predict.steps",
)
PACKAGES = ("verity_vllm.observe.fold", "verity_vllm.check.match")
DRIVER = frozenset(f"verity_vllm.check.replay.{m}" for m in (
    "driver", "coverage", "population", "linkage", "request_replay", "recomputed_prefix", "replay_record", "fork", "sample"))

#: (file, function) of a module the road loads whose body imports the driver; none of them runs on the road
HOLDERS = frozenset({
    ("pipeline/c2_replay.py", "run"), ("pipeline/c2_replay.py", "arrivals"), ("pipeline/c2_replay.py", "seed_of"),
    ("pipeline/c2_replay.py", "linkage_requests"), ("pipeline/c2_replay.py", "executed_prefix_kw"),
    ("pipeline/c2_replay.py", "serving_rows"), ("pipeline/c2_replay.py", "evaluate"), ("pipeline/c2_replay.py", "live_weights"),
    ("pipeline/cpu_replay.py", "main"), ("pipeline/tp/rank_replay.py", "_load"), ("pipeline/tp/rank_replay.py", "seed_of"),
    ("pipeline/tp/rank_replay.py", "index_query"), ("pipeline/tp/rank_replay.py", "evaluate"),
    ("pipeline/tp/rank_replay.py", "host_slot"), ("pipeline/tp/rank_replay.py", "in_process"),
    ("pipeline/tp/rank_replay.py", "defer"), ("pipeline/tp/serving_rows.py", "write"),
})

PROBE = """
import ast, importlib, json, sys
entries, packages, driver = json.loads(sys.argv[1])
def off(m):
    return (m.startswith("verity_vllm.pipeline.match") or m in driver
            or any(m == p or m.startswith(p + ".") for p in packages))
for e in entries:
    importlib.import_module(e)
loaded = sorted(m for m in sys.modules if off(m))
lazy = set()
for name, mod in sorted(sys.modules.items()):
    f = getattr(mod, "__file__", None) or ""
    if not name.startswith("verity_vllm.") or off(name) or not f.endswith(".py"):
        continue
    for fn in ast.walk(ast.parse(open(f).read())):
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for n in ast.walk(fn):
                names = ([n.module, *(f"{n.module}.{a.name}" for a in n.names)] if isinstance(n, ast.ImportFrom) and n.module
                         else [a.name for a in n.names] if isinstance(n, ast.Import)
                         else [n.value] if isinstance(n, ast.Constant) and isinstance(n.value, str) else [])
                if any(off(t) for t in names):
                    lazy.add((f.split("/verity_vllm/", 1)[1], fn.name))
print(json.dumps({"loaded": loaded, "lazy": sorted(lazy)}))
"""


def _road() -> dict[str, list]:
    run = subprocess.run([sys.executable, "-c", PROBE, json.dumps([ENTRIES, PACKAGES, sorted(DRIVER)])],
                         capture_output=True, text=True, check=False)
    assert run.returncode == 0, run.stderr[-4000:]
    return json.loads(run.stdout.splitlines()[-1])


def test_the_road_loads_no_module_of_the_fold_the_match_or_the_driver() -> None:
    road = _road()
    assert not road["loaded"], ("the road's entries loaded " + ", ".join(road["loaded"])
                                + ": import what they use from a module outside the fold, the match and the driver")
    lazy = {tuple(x) for x in road["lazy"]}
    assert lazy <= HOLDERS, f"a function on the road imports the fold, the match or the driver: {sorted(lazy - HOLDERS)}"
    assert HOLDERS <= lazy, f"HOLDERS lists functions that no longer import them; delete {sorted(HOLDERS - lazy)}"
