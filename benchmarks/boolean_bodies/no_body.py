"""Every word primitive with no Boolean body (`versions.boolean_version` refuses it), by where it is registered: every module
of `ROOTS` imported first, then the catalog and the Boolean versions loaded (an earlier load misses bodies a later import
registers); and the primitives `benchmarks/`, `tests/` and `tools/` define, by name from their source (none is imported).
`uv run --all-packages --extra torch-cpu python benchmarks/boolean_bodies/no_body.py > benchmarks/boolean_bodies/no_body.json`."""
import importlib
import json
import pkgutil
import re
import sys
from pathlib import Path

from verity.circuits.defs import REGISTRY, PrimitiveDefinition
from verity.circuits.ops.boolean import versions as V
from verity.circuits.program import is_word_primitive

ROOTS = ["verity.circuits", "verity.commitments", "verity.protocols", "verity_vllm", "verity_torch", "verity_flock"]
TREE = Path(__file__).resolve().parents[2]
failed = {}
for root in ROOTS:
    try:
        m = importlib.import_module(root)
    except ImportError as e:
        failed[root] = repr(e)[:200]
        continue
    for info in pkgutil.walk_packages(getattr(m, "__path__", []), root + "."):
        try:
            importlib.import_module(info.name)
        except ImportError as e:
            failed[info.name] = repr(e)[:200]
from verity_vllm.definitions.registry.catalog import load  # noqa: E402

load()
V.load()

PLACES = [("the circuits' catalogue", "verity.circuits"), ("useful work's prover", "verity.protocols.resource_accounting.compute.pouw"),
          ("protocols, other", "verity.protocols"), ("the vLLM adapter", "verity_vllm"), ("the torch adapter", "verity_torch"),
          ("benchmarks", "verity_bench"), ("tests", "test")]


def place(mod: str) -> str:
    for name, pre in PLACES:
        if mod == pre or mod.startswith(pre + ".") or (pre == "test" and ".test" in "." + mod):
            return name
    return "other: " + mod


rows, n = [], 0
for fid, p in sorted(REGISTRY.defs.items()):
    if not isinstance(p, PrimitiveDefinition) or not is_word_primitive(p):
        continue
    n += 1
    try:
        V.boolean_version(p)
    except LookupError:
        mod = getattr(p.evaluate, "__module__", "") or ""
        rows.append({"id": fid, "place": place(mod), "module": mod})
NAMED = re.compile(r'(?:PrimitiveDefinition\(\s*|@primitive\(\s*)f?"([A-Za-z0-9_{}]+)"')
defined = {top: sorted({m for p in sorted((TREE / top).rglob("*.py")) for m in NAMED.findall(p.read_text(errors="replace"))})
           for top in ("benchmarks", "tests", "tools")}
for name, why in sorted(failed.items()):
    print(f"not imported: {name}: {why}", file=sys.stderr)
out = {"roots": ROOTS, "word_primitives": n, "no_body": len(rows),
       "by_place": {k: [r["id"] for r in rows if r["place"] == k] for k in sorted({r["place"] for r in rows})}, "rows": rows,
       "defined_outside_verity": defined}
json.dump(out, sys.stdout, indent=1)
print()
