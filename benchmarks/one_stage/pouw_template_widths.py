"""CIR-57 (measurement only, scratch): PoUW's template-query units on main, with the bits an instance commits under
9431e36d39's held rule, as `template_widths.py` measures a served row's.

    python benchmarks/one_stage/pouw_template_widths.py --out OUT.json

The units are (1) the tile templates the verifier pins (`core_check.TILES`, the served ZK window's), bound here where
their Definition is on main, and (2) every template a PoUW root of record instantiates: each PoUW circuit-check sidecar's
`_pouw_roots()` (`verity/core/protocols/pouw/circuit/*.circuit-check.py`), where a root is a call, check, weight or
window Program (`ROOT_FAMILY`; a `Q_template_instances` root holds only template instances) and its children are its
templates. A sidecar target that is neither a root nor a root's child is listed as a catalogued unit (`PouwH1Tile`, the
compile pass's tile unit, is one).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import template_widths as TW  # noqa: E402

from verity.core.protocols.pouw import core_check as CC  # noqa: E402
from verity.core.protocols.pouw.circuit import pc8  # noqa: E402
from verity.ml.circuits.defs import PrimitiveDefinition  # noqa: E402
from verity.ml.circuits.program import is_input  # noqa: E402

POUW = TW.ROOT / "verity/core/protocols/pouw/circuit"
#: PoUW's `Q_template_instances` roots: a call, its check, its weight side, a window
ROOT_FAMILY = re.compile(r"^(Ncp(Linear|Weight|Window)|Pc[48](Call|Check|Weight)\w*)_v\d+$")


def sidecar_targets() -> dict[str, list]:
    out = {}
    for f in sorted(POUW.glob("*.circuit-check.py")):
        spec = importlib.util.spec_from_file_location(f"pouw_sidecar_{f.name.split('.')[0]}", f)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        out[f.name.split(".")[0]] = list(m._pouw_roots()) if hasattr(m, "_pouw_roots") else []
    return out


def pinned_tiles() -> dict[str, object | None]:
    """`core_check.TILES`' ids, each bound where its Definition is on main."""
    out: dict[str, object | None] = {}
    for tid in CC.TILES:
        fam = TW.K.family(tid)
        k = 4096 if "K=4096" in tid else 128
        f = None
        if fam == "Pc8TileSide_v1":
            f = pc8.side_tile(k, (1, 1), "sm120")
        elif fam == "Pc8TileHidden_v1":
            f = pc8.templates(k, (1, 1), "sm120", True)["tile"]
        out[tid] = f if f is not None and f.id == tid else None
    return out


def children(fn, fns: dict) -> Counter:
    c: Counter = Counter()
    if isinstance(fn, PrimitiveDefinition) or not hasattr(fn, "body"):
        return c
    for nd in fn.body.nodes:
        if not is_input(nd.fn):
            c[nd.fn.id] += TW.K.mult(nd)
            fns.setdefault(nd.fn.id, nd.fn)
    return c


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--literal-max", type=int, default=400_000)
    ap.add_argument("--cone-max", type=int, default=3_000_000)
    a = ap.parse_args()
    t0 = time.time()
    by_mod = sidecar_targets()
    fns: dict[str, object] = {}
    mods: dict[str, set] = {}
    for mod, ts in by_mod.items():
        for t in ts:
            fns.setdefault(t.id, t)
            mods.setdefault(t.id, set()).add(mod)
    roots = {tid for tid in fns if ROOT_FAMILY.match(TW.K.family(tid))}
    kids = {tid: children(fns[tid], fns) for tid in sorted(roots)}
    role: dict[str, dict] = {}
    for r in sorted(roots):
        for k, n in kids[r].items():
            if k not in roots:
                role.setdefault(k, {"roles": set(), "used_by": {}})
                role[k]["roles"].add("root template")
                role[k]["used_by"][r] = n
    for tid in list(mods):
        if tid not in roots and tid not in role and not isinstance(fns[tid], PrimitiveDefinition):
            role.setdefault(tid, {"roles": {"catalogued unit"}, "used_by": {}})
    pinned = pinned_tiles()
    for tid, f in pinned.items():
        if f is not None:
            fns[tid] = f
        role.setdefault(tid, {"roles": set(), "used_by": {}})
        role[tid]["roles"].add("verifier-pinned tile (core_check.TILES)")
    memo: dict = {}
    recs = {}
    for tid in sorted(role):
        f = fns.get(tid)
        rec = {"id": tid, "roles": sorted(role[tid]["roles"]), "used_by": role[tid]["used_by"],
               "sidecars": sorted(mods.get(tid, ()))}
        if f is None:
            rec["unbound"] = "Definition not on main"
        else:
            try:
                rec.update(TW.describe(f, memo))
                TW.deep(f, rec, a, recut_too=True)
            except Exception as e:  # noqa: BLE001
                rec["error"] = repr(e)[:300]
        recs[tid] = rec
        print(f"[pouw] {tid}: bits {rec.get('bits')} gates {rec.get('gates')} {rec.get('verdict', '')[:60]}", flush=True)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps({"templates": recs, "roots": sorted(roots), "partial": True}, indent=1, default=str))
    refused = sorted((r["id"] for r in recs.values() if (r.get("bits") or 0) > TW.P.UNIT_BITS), key=lambda t: -recs[t]["bits"])
    doc = {"rule": "9431e36d39", "templates": recs, "roots": sorted(roots), "refused": refused,
           "disagreements": [r["id"] for r in recs.values() if r.get("disagree")], "seconds": round(time.time() - t0, 1)}
    a.out.write_text(json.dumps(doc, indent=1, default=str))
    print(f"[pouw] {len(recs)} units, {len(refused)} over 16 bits, {doc['seconds']} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
