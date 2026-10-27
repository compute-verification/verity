"""The atoms `frontier.py` composes, counted: each Definition of the exact-int8 profile a composition reads (vLLM's registry:
`int8_exact`, `moe`, `b1`, `act`), bound at a model's statics and counted by property/v1 (`verity.ml.circuits.stacking.count`)
with the run dependence `frontier.DYN` gives its family. An atom's record: its units by class, the gates property/v1 refuses,
the s8 family's proven steps and their products, the proven affine maps, and its units by ``primitive|class``
(`units.by_primitive`, every composite expanded). A fitted family (`frontier.FITTED`, the attention row and the KV window) is
counted at each S of `frontier.FIT_S` and written ``const`` + ``slope`` S, held to the last S. Then `frontier.check_anchor`
holds the atoms to the int8 census's measured Llama-3.1-8B, and `frontier.table` gives the table.

Runs where `verity_vllm` imports (the registry modules need numpy, not torch):

    python atoms.py --out atoms.json [--only REGEX]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping, Sequence

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import frontier as F  # noqa: E402
import units as U  # noqa: E402
from verity.ml.circuits import stacking as P  # noqa: E402
from verity.ml.circuits.defs import REGISTRY, bind  # noqa: E402


def registry() -> Any:
    """The registry modules whose Definitions the atoms bind (imported for their registration), and `int8_exact`."""
    from verity.ml.definitions.registry import act, b1, int8_exact, moe  # noqa: F401

    return int8_exact


def definition(atom: str, s: int | None = None) -> tuple[Any, list[bool]]:
    """The Definition an atom key names, bound at its statics (and S = `s`), and its run dependence."""
    ie = registry()
    fam, st = F.parse(atom)
    dyn = list(F.DYN[fam])
    if atom.endswith(F.EMB_DYN_TABLE):
        dyn = [True] * len(dyn)
    if fam == "ScaledMmS8_v1":
        return ie.scaled_mm(st["K"], st["N"]), dyn
    if s is not None:
        st = {**st, "S": s}
    return bind(REGISTRY.defs[fam], **st), dyn


def tally(fn: Any, dyn: Sequence[bool]) -> dict:
    t = P.count(fn, dyn)
    rec = {"units": {c: n for c, n in t.units.items() if n}, "refused": {f"{c}:{p}": n for (c, p), n in t.refused.items()},
           "family_steps": t.family, "macs": t.macs, "affine": t.affine}
    try:
        rec["primitives"] = U.by_primitive(fn, dyn)
    except (AssertionError, ValueError) as e:
        rec["primitives"] = {f"(unexpanded)|{c}": n for c, n in rec["units"].items()}
        rec["primitives_error"] = str(e)
    return rec


def linear_fit(points: Mapping[int, Mapping[str, int]]) -> tuple[dict, dict]:
    """const + slope S through the first two points, held to every other (raises when one misses)."""
    (s0, a), (s1, b) = list(points.items())[:2]
    keys = set(a) | set(b)
    slope = {k: Fraction(b.get(k, 0) - a.get(k, 0), s1 - s0) for k in keys}
    const = {k: a.get(k, 0) - slope[k] * s0 for k in keys}
    for s, u in points.items():
        model = {k: const.get(k, 0) + slope.get(k, 0) * s for k in keys | set(u)}
        if any(model[k] != u.get(k, 0) for k in model):
            raise AssertionError(f"not linear in S: at S={s} the fit gives {U._ints(model)}, the count {dict(u)}")
    return ({k: v for k, v in const.items() if v}, {k: v for k, v in slope.items() if v})


def count_atom(atom: str) -> dict:
    fam, st = F.parse(atom)
    if fam not in F.FITTED:
        fn, dyn = definition(atom)
        return {"family": fam, "statics": st, "dyn": dyn, "id": fn.id, **tally(fn, dyn)}
    at = {}
    for s in F.FIT_S:
        fn, dyn = definition(atom, s)
        at[s] = tally(fn, dyn)
    out: dict[str, Any] = {"family": fam, "statics": st, "dyn": dyn, "S": list(F.FIT_S), "at": at, "const": {}, "slope": {}}
    for what in ("units", "primitives"):
        out["const"][what], out["slope"][what] = linear_fit({s: at[s][what] for s in F.FIT_S})
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--models", type=Path, default=HERE / "frontier_models.json")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--only", help="count only the atoms whose key matches this regex (no anchor check)")
    a = ap.parse_args(argv)
    spec = json.loads(a.models.read_text())
    S, P_ = spec["workload"]["context"], spec["workload"]["prompt"]
    keys = F.needed(spec["models"], S, P_)
    if a.only:
        keys = [k for k in keys if re.search(a.only, k)]
    atoms: dict[str, Any] = {}
    for i, k in enumerate(keys):
        atoms[k] = count_atom(k)
        print(f"[atoms] {i + 1}/{len(keys)} {k}: {atoms[k].get('units') or atoms[k]['slope']['units']}", file=sys.stderr, flush=True)
        a.out.write_text(json.dumps(U.jsonable({"schema": "verity-stacked-share/atoms/v1", "classes": P.digest(), "atoms": atoms}),
                                    indent=1))
    refused = {k: r["refused"] for k, r in atoms.items() if r.get("refused")}
    if refused:
        print(f"[atoms] refused gates: {refused}", file=sys.stderr)
    if not a.only:
        print(json.dumps(U.jsonable(F.check_anchor(spec["models"], atoms, spec["anchor"], S, P_))), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
