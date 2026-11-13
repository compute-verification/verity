"""CIR-274 DSAMOE: do the per-expert tables the sparse lowering declares carry the names serving's registration gives?

At LAYERS layers, rank 0, 43 prompt ids and 2 outputs (the census's Build, `benchmarks/compile_cost/run.py`'s `registered`):
the set of `IF.Table`s the node window declares for the experts (`node_window_sparse._table_reads`: name, words a row,
rows), against `pipeline.expert_tables.tables_of` at ``--block w13=1 --block w2=32`` over the same Build's weight fields,
each table's row width and row count from `expert_rows.table_words` on a zero stack of the field's shape. Serving's fields
two ways: the weights record as `window_weights._fields` reads a Program's (on the request Program: the class Program is
refused for this Build), and `weights_of_record.program_weight_fields` over ``[["weights", str(type)]]`` (a Program dir's
``instances["params"]``, as `program/workload.py` writes it), which `expert_tables.register` reads. Run from
the repository root:

    .venv/bin/python p3_names.py OUT [LAYERS]

Writes OUT/names_l<LAYERS>.json."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

out = Path(sys.argv[1])
layers = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2] else 4
out.mkdir(parents=True, exist_ok=True)
spec = importlib.util.spec_from_file_location("compile_cost", "benchmarks/compile_cost/run.py")
cc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cc)

from verity_vllm.check import weights_of_record as WR  # noqa: E402
from verity_vllm.pipeline import expert_tables as ET  # noqa: E402
from verity_vllm.pipeline.expert_rows import table_words  # noqa: E402
from verity_vllm.program.predict import node_window as NW  # noqa: E402
from verity_vllm.program.predict import node_window_dense as ND  # noqa: E402
from verity_vllm.program.predict import node_window_sparse as NS  # noqa: E402
from verity_vllm.program.predict.steps import plan_steps, request_program  # noqa: E402

BLOCKS = {"w13": 1, "w2": 32}
reg = cc.registered(0, layers, "eager")
advice, ingress = cc.session([(0, 43, 2)])
u = [x for x in plan_steps(reg, advice, ingress).requests if x is not None][0]
p = request_program(reg, u)
w = NW.NodeWindow(p.program.fn, {**NW.FAMILIES, **ND.FAMILIES, **NS.FAMILIES}, NW._GRAIN, NW.NAME, 1, True).lower()
mine = {t.param: {"name": t.name, "words": t.words, "rows": t.rows, "leaves": w.b.params.get(t.param)}
        for t in w.b.tables.values() if "/b" in t.param and t.param.endswith("/table")}


def serving(fields: list) -> dict:
    shape = {f: [int(x) for x in s] for f, s, _b in fields}
    bits = {f: int(b) for f, _s, b in fields}
    got = {}
    for label, ts in ET.tables_of(fields, BLOCKS).items():
        E, N, K = shape[label]
        stack = np.zeros((E, N, K), np.uint8 if bits[label] == 8 else np.uint32)
        for b, param, what, n in ts:
            rows = table_words(stack, what, E, N, K, 1, n, b)
            got[param] = {"words": int(rows.shape[1]), "rows": int(rows.shape[0]), "field": label, "shape": [E, N, K],
                          "bits": bits[label]}
    return got


def weights_type(fn) -> str:
    return str(dict(fn.signature()[0])[WR.PARAM])


def record_fields(fn) -> list:
    """`window_weights._fields`' reading of a Program's weights record, on ``fn`` (its class Program is refused for this
    Build, which has no sweep: `launch_context._requests`)."""
    from verity_vllm.program.predict.common import _arr_shape
    return [(n, list(_arr_shape(t)), next(iter(t.width_set()))) for n, t in dict(fn.signature()[0])[WR.PARAM].fields]


sides = {
    "record_fields": record_fields(p.program.fn),
    "program_dir_params": WR.program_weight_fields({"params": [[WR.PARAM, weights_type(p.program.fn)]]}),
}
rec = {"layers": layers, "rank": 0, "blocks": BLOCKS, "program": p.digest,
       "mine": {"count": len(mine), "tables": mine}}
for k, fields in sides.items():
    theirs = serving(fields)
    diff_width = {n: (mine[n]["words"], mine[n]["rows"], theirs[n]["words"], theirs[n]["rows"]) for n in set(mine) & set(theirs)
                  if (mine[n]["words"], mine[n]["rows"]) != (theirs[n]["words"], theirs[n]["rows"])}
    rec[k] = {"fields": len(fields), "expert_fields": sorted(f for f, _s, _b in fields if WR.EXPERTS_RE.match(f)),
              "count": len(theirs), "equal_names": set(mine) == set(theirs),
              "only_mine": sorted(set(mine) - set(theirs)), "only_theirs": sorted(set(theirs) - set(mine)),
              "width_or_rows_differ": diff_width, "tables": theirs}
    print("DSAMOE names", k, json.dumps({x: rec[k][x] for x in ("fields", "count", "equal_names")}),
          "only_mine", len(rec[k]["only_mine"]), rec[k]["only_mine"][:3], "only_theirs", len(rec[k]["only_theirs"]),
          rec[k]["only_theirs"][:3], "width_or_rows_differ", len(diff_width), flush=True)
rec["leaves_check"] = all(m["leaves"] == m["words"] * m["rows"] for m in mine.values())
print("DSAMOE names mine", len(mine), "leaves = words x rows:", rec["leaves_check"],
      sorted({(m["words"], m["rows"]) for m in mine.values()}), flush=True)
(out / f"names_l{layers}.json").write_text(json.dumps(rec, indent=1, default=str) + "\n")
