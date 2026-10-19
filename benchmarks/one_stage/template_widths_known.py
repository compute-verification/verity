"""CIR-57 (measurement only, scratch): `template_widths.py`'s measurement of Definitions named by id, for the over-16
Definitions a served path offers outside the Programs of record (a Boolean form the records don't build yet, a unit of the
one-stage audit's static pair form, a Definition on an unmerged branch).

    python benchmarks/one_stage/template_widths_known.py --out KNOWN.json [--where LABEL] ID{STATICS}...

Each id resolves through the vLLM registry (`program_view._registry`), statics read as Python literals; a
``GatherE4m3x{V}_v2`` / ``GatherF32x{V}_v2`` through `boolean_fp8_moe`. Run it from the tree that carries the Definition
(``PYTHONPATH=<tree>:<tree>/integrations/vllm``, that tree the working directory), whose pins it reads and whose
`partition.py` must carry the held rule; ``--where`` names that tree in the output.
"""
from __future__ import annotations

import argparse
import ast
import inspect
import json
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import template_widths as TW  # noqa: E402

GATHER = re.compile(r"^Gather(E4m3|F32)x(\d+)_v2$")


def split(body: str) -> list[str]:
    out, depth, cur = [], 0, ""
    for ch in body + ",":
        if ch in "{([":
            depth += 1
        elif ch in "})]":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += ch
    return out


def bind(tid: str, reg):
    base, _, rest = tid.partition("{")
    statics = {}
    for part in split(rest[:-1]) if rest else []:
        k, _, v = part.partition("=")
        statics[k] = ast.literal_eval(v)
    if m := GATHER.match(base):
        from verity_vllm.program.registry import boolean_fp8_moe as BFM
        return (BFM.gather_e4m3 if m[1] == "E4m3" else BFM.gather_f32)(int(m[2])).bind()
    return reg.defs[base].bind(**statics)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="+")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--where", default="main")
    ap.add_argument("--literal-max", type=int, default=400_000)
    ap.add_argument("--cone-max", type=int, default=3_000_000)
    ap.add_argument("--literal-s", type=float, default=60)
    ap.add_argument("--recut-s", type=float, default=120)
    a = ap.parse_args()
    if "an instance of" not in inspect.getsource(TW.P._check_units):
        sys.exit(f"{TW.P.__file__} has no held rule (9431e36d39's `_check_units`): apply it to that tree first")
    from verity_vllm.query.program_view import _registry
    reg, _ = _registry()
    head = subprocess.run(["git", "-C", str(TW.ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    out = {"where": a.where, "tree": head, "rule": TW.P.__file__, "templates": {}}
    for tid in a.ids:
        t = time.time()
        try:
            fn = bind(tid, reg)
            rec = TW.describe(fn, {})
            TW.deep(fn, rec, a, recut_too=True)
        except Exception as e:  # noqa: BLE001 - a probe reports what it could not measure
            rec = {"id": tid, "family": tid.split("{")[0], "bits": None, "describe_error": f"{type(e).__name__}: {e}"}
        rec["where"], rec["seconds"] = a.where, round(time.time() - t, 2)
        out["templates"][rec["id"]] = rec
        print(rec["id"], rec.get("bits"), (rec.get("verdict") or rec.get("describe_error") or "")[:80], file=sys.stderr)
    a.out.write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
