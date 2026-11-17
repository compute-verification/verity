"""Scratch: a record-check case (`rows_case_probe.py`) and a5's compose cache as the files of `inference-only-rows check`.

    python rows_from_case.py CASE CACHE OUT [--warm-log FILE] [--alpha NUM] [--gamma N]

Each record template is matched to the cache entry whose made.json META names it (`template`), or, where several do, to
the pin warm's line for it gives. Writes OUT/params.json, commitment.json, evidence.json and blobs.txt (the circuit files,
then the pinned tables, one path a line), and prints each template's rows of one unit (`unit_rows`, read off META and the
unit circuit's header) and the predicted Γ: a block's rows, Σ units × rows over the templates, over the blocks. Without
--gamma and --alpha, Γ is that prediction and α is --alpha (default the case's fresh bits a block) over Γ.
"""
import argparse
import hashlib
import importlib.util
import json
import mmap
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path.cwd()
S = ROOT / "verity/protocols/workload_classification/training_detection/inference_only/verifier/lean/scripts"
sys.path[:0] = [str(ROOT / "tools/difftest/python")]
from verity_difftest import canonical  # noqa: E402


def load(name):
    spec = importlib.util.spec_from_file_location(name, S / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


ref = load("inference_only_record_reference")
ap = argparse.ArgumentParser()
ap.add_argument("case", type=Path)
ap.add_argument("cache", type=Path)
ap.add_argument("out", type=Path)
ap.add_argument("--warm-log", type=Path)
ap.add_argument("--alpha", type=int, default=337_786_912)
ap.add_argument("--gamma", type=int)
ap.add_argument("--blocks", type=int, default=4)
a = ap.parse_args()

case = json.loads(a.case.read_text())
inst, tr = case["instance"], case["trusted"]
doc = json.loads(inst["record"])
templates = [(t["id"], t["units"]) if isinstance(t, dict) else (t[0], t[1]) for t in doc["templates"]]

warm = {}
for ln in (a.warm_log.read_text().splitlines() if a.warm_log else []):
    if ln.startswith("{"):
        g = json.loads(ln)
        warm[g["template"]] = g["pin"]
entries = defaultdict(list)
for made in sorted(a.cache.glob("*/made.json")):
    m = json.loads(made.read_text())
    entries[m["meta"].get("template")].append((made.parent, m))

circuits, files, tables, rows, missing = [], [], {}, {}, []
for tid, units in templates:
    got = entries.get(tid, [])
    if len(got) > 1 and tid in warm:
        got = [e for e in got if e[1]["pin"] == warm[tid]]
    if len(got) != 1:
        missing.append((tid, len(got)))
        continue
    d, m = got[0]
    f = d / "circuit.txt"
    with f.open("rb") as fh, mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        rows[tid] = ref.unit_rows(mm[:1 << 26])
        h = hashlib.sha512()
        for i in range(0, len(mm), 1 << 28):
            h.update(mm[i:i + (1 << 28)])
    circuits.append([tid, h.hexdigest()])
    files.append((f, f.stat().st_size))
    for t in sorted((d / "tables").glob("*")) if (d / "tables").is_dir() else ():
        tables[hashlib.sha512(t.read_bytes()).hexdigest()] = t
    print(json.dumps({"template": tid, "units": units, "rows": rows[tid], "bytes": f.stat().st_size,
                      "sha512": circuits[-1][1][:16]}), flush=True)
if missing:
    sys.exit(f"no single cache entry for {len(missing)} templates: {missing[:5]}; META templates in the cache: "
             f"{sorted(map(str, entries))[:10]}")

block = sum(units * rows[tid] for tid, units in templates)
gamma = a.gamma if a.gamma is not None else -(-block // a.blocks)
print("ROWS " + json.dumps({"templates": len(templates), "all_units_rows": block, "blocks": a.blocks, "gamma": gamma,
                            "alpha": [a.alpha, gamma], "circuit_bytes": sum(s for _, s in files),
                            "tables": len(tables)}), flush=True)
a.out.mkdir(parents=True, exist_ok=True)
params = {"alpha": [a.alpha, gamma], "circuits": circuits, "gamma": gamma, "log": tr["log"][:-1], "pins": inst["pins"],
          "program": inst["program"]}
(a.out / "params.json").write_text(canonical(params))
(a.out / "commitment.json").write_text(canonical(tr["log"][-1]))
(a.out / "evidence.json").write_text(canonical({"record": inst["record"], **({"order": inst["order"]} if "order" in inst
                                                                             else {})}))
(a.out / "blobs.txt").write_text("".join(f"{p}\n" for p in [*(f for f, _ in files), *tables.values()]))
