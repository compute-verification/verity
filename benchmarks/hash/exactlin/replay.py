"""The int8 Definitions' replay evaluators on the fused kernels' outputs: `Int8RowQuant_v1` (int8_rows.int8_row_quant_rows)
on each dumped x against the kernel's xq and s_a, and `ScaledMmS8_v1` (int8_rows.scaled_mm, DOT=BlackwellS8ImmaDot32_v1)
on the kernel's xq and s_a, row by row, against its y.  The npz files are `kgate.py --dump`'s (OUT/replay/), words
throughout; CPU only.

`--evaluators` is an int8_rows.py to load in place of the installed `verity.ml.definitions.numerics.int8_rows`, e.g. a
commit's:
    git show REV:verity/ml/definitions/numerics/int8_rows.py > /tmp/int8_rows.py
    python replay.py RUN/out/replay --evaluators /tmp/int8_rows.py
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


def load(path: str | None):
    if path is None:
        try:
            from verity.ml.definitions.numerics import int8_rows
        except ImportError as e:
            raise SystemExit(f"replay.py: {e}; install verity, or pass --evaluators (see the docstring)") from e
        return int8_rows
    spec = importlib.util.spec_from_file_location("exactlin_int8_rows", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def check(ev, f: Path) -> dict:
    d = np.load(f)
    x, xq, s, w, sw, y = (d[k] for k in ("x", "xq", "s", "w", "sw", "y"))
    m, k = x.shape
    n = w.shape[0]
    codes, s_ev = ev.int8_row_quant_rows(x)
    doc = {"file": f.name, "m": m, "k": k, "n_checked": n,
           "row_quant": {"xq_diff": int((codes.reshape(m, k) != xq).sum()), "s_diff": int((s_ev.reshape(m) != s).sum())}}
    if hasattr(ev, "scaled_mm"):
        statics = {"K": k, "N": n, "DOT": ev.DOT}
        y_diff = 0
        for r in range(m):
            out = ev.scaled_mm(statics, [xq[r], s[r:r + 1], w, sw])["0"]
            y_diff += int((np.asarray(out, dtype=np.uint16) != y[r]).sum())
        doc["scaled_mm"] = {"y_diff": y_diff, "cells": m * n}
    doc["ok"] = (doc["row_quant"]["xq_diff"] == 0 and doc["row_quant"]["s_diff"] == 0
                 and doc.get("scaled_mm", {"y_diff": 0})["y_diff"] == 0)
    return doc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dump", help="a directory of kgate.py --dump's npz files")
    ap.add_argument("--evaluators", help="an int8_rows.py to load (default: verity.ml.definitions.numerics.int8_rows)")
    a = ap.parse_args()
    ev = load(a.evaluators)
    rows = [check(ev, f) for f in sorted(Path(a.dump).glob("*.npz"))]
    for r in rows:
        print(json.dumps(r), flush=True)
    doc = {"ok": bool(rows) and all(r["ok"] for r in rows), "files": len(rows), "scaled_mm": hasattr(ev, "scaled_mm"),
           "evaluators": a.evaluators or "verity.ml.definitions.numerics.int8_rows"}
    print(json.dumps(doc), flush=True)
    return 0 if doc["ok"] else 4


if __name__ == "__main__":
    sys.exit(main())
