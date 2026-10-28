"""GD-MODEL's J (golden/model.py of cursor/compute-research-gd-model-6eb4) as a model for search.py.

    python search.py run --model gdm --grid coarse --shard 0/16 --scenario nominal --out DIR

GD-MODEL is vendored at `GD_MODEL_COMMIT` under vendor/golden (model.py and its own adapter, search_adapter.py), so a
`research run` checkout of this branch carries it; `GD_MODEL=PATH` imports another copy (a directory holding golden/).

GD-MODEL's adapter maps a design onto its model and states feasibility (credit floor, the block cap, prover memory,
lowering, retention). This module adds only:
- the grids (`KNOBS`, `GRIDS`): the adapter's knobs, and these, each set in the model's design dict:
  `cbits` (commit.bits_per_word), `mmax` (prover.m_max), `retain` (device: the retain_GB input; host: `HOST_RETAIN_GB`,
  an estimate), and JR2's H1 host levers taken (`H1_LEVERS`);
- the hiding arm (`hiding`): hm96 (salt, c, h: two compressions in circuit, seven a 4 KiB row served) or SHA-512 as a
  PRF, H(salt || row), which drops c: hm96_ands at 0 and a row's served compressions its data's plus the frame leaf's;
- the demo's window class: at t512 a window is one step (`steps` 1), since the brief pins its class.
"""
from __future__ import annotations

import math
import os
import sys
from dataclasses import replace
from itertools import product
from pathlib import Path

GD_MODEL_COMMIT = "56c63004acf82c858c0d6ab07a48f74eae364d21"
sys.path.insert(0, os.environ.get("GD_MODEL") or str(Path(__file__).resolve().parent / "vendor"))
import golden.model as M  # noqa: E402
import golden.search_adapter as A  # noqa: E402

#: GD-MODEL's inputs and ranges; the H200's drawn-AND cost is measured, but its 29x over sm_120 has an unchecked cause
#: (GD-PROVER), so it is swept over GD-MODEL's own range for it
INPUTS = {**A.INPUTS, "o_per_and_h200": replace(A.INPUTS["o_per_and_h200"], uncertain=True)}
POINTS = A.POINTS
#: GB a window may retain on the host (VERITY_RETAIN=host): an estimate, the top of GD-MODEL's retain_GB range
HOST_RETAIN_GB = 1000.0
#: JR2's served H1 path at every host lever (as the interim took them)
H1_LEVERS = {"h1_host": "stack", "kdhash": True, "hash_at_bar": True, "coin_off_loop": True}

KNOBS = ("fuse", "cbits", "row", "seal", "host", "kernels", "overlap", "open", "scheme", "lp", "t", "noise", "drow",
         "credit", "joint", "form", "ptile", "tile", "akeys", "kmax", "mmax", "reps", "device", "sampling", "stmt",
         "hiding", "steps", "retain", "period")
_DITHER_ONLY = ("lp", "t", "noise", "drow", "credit", "joint")
_JR2_ONLY = ("form",)
_ADAPTER = ("fuse", "row", "seal", "host", "kernels", "overlap", "open", "scheme", "ptile", "tile", "akeys", "kmax",
            "reps", "device", "sampling", "stmt", "steps", "period")

_BASE = {
    "fuse": ("all", "fuse", "e2"),
    "cbits": (8,),
    "row": (1024, 4096, 16384, 65536),
    "seal": ("roots",),
    "host": ("fixed",),
    "kernels": ("bar",),
    "overlap": ("corun",),
    "open": ("whole", "exact"),
    "scheme": ("dither", "jr2"),
    "lp": (32, 128, 512),
    "t": (1, 2, 4),
    "noise": ("perm",),
    "drow": (1024, 4096, 16384),
    "credit": ("all4", "k2048"),
    "joint": ("separate", "joint"),
    "form": ("tile", "lanes"),
    "ptile": ((1, 1), (4, 8), (32, 32)),
    "tile": ((1, 1),),
    "akeys": (32, 64, 128),
    "kmax": (27, 30),
    "mmax": (35, 40),
    "reps": (2, 1),
    "device": ("h200", "sm120"),
    "sampling": ("work", "stratified", "uniform"),
    "stmt": ("template", "universal"),
    "hiding": ("hm96",),
    "steps": (1,),
    "retain": ("device",),
    "period": (1,),
}
#: the full product (18 M designs)
GRIDS: dict[str, dict[str, tuple]] = {
    "full": {**_BASE, "steps": (1, 4, 16, 32), "retain": ("device", "host")},
}
GRIDS["smoke"] = {k: v[:2] for k, v in GRIDS["full"].items()}
#: the knobs that interact, in full; what's committed and its word bits fixed (each moves J by under 1 part in 10^4,
#: the serve term alone), and Dither's credit and joint at today's (`flips` prices them at the optimum)
GRIDS["core"] = {**GRIDS["full"], "fuse": ("fuse",), "credit": ("all4",), "joint": ("separate",),
                 "ptile": ((1, 1), (4, 8), (32, 32))}
#: the core grid near its optimum, for the one-at-a-time and Monte Carlo sweeps
GRIDS["sweep"] = {**GRIDS["core"], "open": ("exact",), "lp": (32, 128, 512), "t": (1, 2), "drow": (1024, 4096),
                  "ptile": ((1, 1), (4, 8)), "row": (1024, 4096, 16384), "mmax": (40,), "reps": (1,),
                  "sampling": ("work",), "stmt": ("template", "universal")}
#: the accounting period n over the sweep grid, both hiding arms and both reps
GRIDS["period"] = {**GRIDS["sweep"], "kmax": (27,), "akeys": (32,), "reps": (2, 1), "hiding": ("hm96", "prf"),
                   "period": (1, 10, 100, 1000, 10_000, 100_000)}
#: the core grid with every lever that needs new work held at today's: the H200 proving, two reps, m <= 35, whole-row
#: openings, a statement a template, k_log <= 27, hm96, retention on the device
GRIDS["today"] = {**GRIDS["core"], "open": ("whole",), "kmax": (27,), "mmax": (35,), "reps": (2,), "device": ("h200",),
                  "stmt": ("template",), "retain": ("device",)}
#: today's grid against the accounting period n
GRIDS["today-period"] = {**GRIDS["today"], "period": (1, 10, 100, 1000, 10_000, 100_000)}


def size(name: str) -> int:
    return sum(1 for _ in grid(name))


def grid(name: str):
    """The product of GRIDS[name], each JR2 design once (at the first values of Dither's knobs) and each Dither design
    once (at the first values of JR2's)."""
    g = GRIDS[name]
    first = {k: g[k][0] for k in _DITHER_ONLY + _JR2_ONLY}
    j = 0
    for values in product(*(g[k] for k in KNOBS)):
        d = dict(zip(KNOBS, values))
        skip = _DITHER_ONLY if d["scheme"] == "jr2" else _JR2_ONLY
        if any(d[k] != first[k] for k in skip):
            continue
        yield j, d
        j += 1


def shard(name: str, i: int, n: int):
    for j, d in grid(name):
        if j % n == i:
            yield j, d


def key(d: dict) -> str:
    pt = "x".join(map(str, d["ptile"]))
    pouw = (f"dither/L'{d['lp']}/t{d['t']}/{d['noise']}/drow{d['drow']}/{d['credit']}/{d['joint']}/pt{pt}"
            if d["scheme"] == "dither" else f"jr2/{d['form']}/pt{pt}")
    s = (f"{d['fuse']}/b{d['cbits']}/row{d['row']}/{d['seal']}/{d['host']}/{d['kernels']}/{d['overlap']}/{d['open']}/"
         f"{pouw}/vt{'x'.join(map(str, d['tile']))}/ak{d['akeys']}/k{d['kmax']}/m{d['mmax']}/r{d['reps']}/"
         f"{d['device']}/{d['sampling']}/{d['stmt']}/{d['hiding']}/S{d['steps']}/{d['retain']}")
    return s if d["period"] == 1 else f"{s}/n{d['period']}"


def nominal() -> dict[str, float]:
    return {k: i.value for k, i in INPUTS.items()}


_extra: dict[tuple[str, str], object] = {}
_to_design = A.to_design


def _design(k: dict, point: str = "t8192") -> dict:
    d = _to_design(k, point)
    for (sec, name), val in _extra.items():
        d[sec][name] = val
    return d


A.to_design = _design
_compressions_per_row = M.compressions_per_row


#: every value `flips` tries for each knob, one knob at a time from a design
FLIPS = {
    "fuse": ("all", "dedupe", "fuse", "e1", "e2"), "cbits": (8, 12, 16), "row": (512, 1024, 2048, 4096, 16384, 65536),
    "seal": ("tree", "roots"), "host": ("today", "fixed"), "kernels": ("served", "bar"),
    "overlap": ("none", "corun", "corun4"), "open": ("whole", "exact"), "scheme": ("dither", "jr2"),
    "lp": (32, 64, 128, 256, 512), "t": (1, 2, 3, 4, 8), "noise": ("perm", "periodic", "fresh"),
    "drow": (512, 1024, 2048, 4096, 16384, 32768), "credit": ("all4", "k2048"), "joint": ("separate", "joint"),
    "form": ("tile", "lanes"), "ptile": ((1, 1), (1, 8), (4, 8), (8, 8), (32, 32)), "tile": ((1, 1), (1, 4), (4, 4)),
    "akeys": (32, 64, 128), "kmax": (27, 28, 29, 30), "mmax": (35, 36, 37, 38, 39, 40), "reps": (1, 2),
    "device": ("h200", "sm120"), "sampling": ("work", "stratified", "uniform"), "stmt": ("template", "universal"),
    "hiding": ("hm96", "prf"), "steps": (1, 2, 4, 8, 16, 32, 64), "retain": ("device", "host"),
    "period": (1, 10, 100, 1000, 10_000, 100_000),
}


def flips(d: dict, point: str, v: dict[str, float] | None = None) -> list[dict]:
    """J at d, and at each other value of each knob with the rest held."""
    out = []
    for k, vals in FLIPS.items():
        for x in vals:
            r = evaluate({**d, k: x}, v, point)
            out.append({"knob": k, "value": x, "at_d": x == d[k], "J": r["J"], "serve": r["serve"],
                        "prove_ratio": r["prove_ratio"], "feasible": r["feasible"], "reasons": r["reasons"]})
    return out


def _prf_compressions_per_row(row_bytes: int) -> int:
    return math.ceil((row_bytes + 17) / 128) + 2


def _refused(why: str) -> dict:
    return {"J": math.inf, "serve": math.inf, "prove_ratio": math.inf, "feasible": False, "reasons": [why]}


def evaluate(d: dict, v: dict[str, float] | None = None, point: str = "t8192") -> dict:
    if point == "t512" and d["steps"] != 1:
        return _refused("window class")
    v = dict(v or nominal())
    knobs = {k: d[k] for k in _ADAPTER}
    if d["scheme"] == "dither":
        knobs.update({k: d[k] for k in _DITHER_ONLY})
    else:
        knobs["form"] = d["form"]
    _extra.clear()
    _extra[("commit", "bits_per_word")] = d["cbits"]
    _extra[("prover", "m_max")] = d["mmax"]
    if d["retain"] == "host":
        _extra[("sampling", "retain_GB")] = HOST_RETAIN_GB
    if d["scheme"] != "dither":
        _extra.update({("pouw", k): x for k, x in H1_LEVERS.items()})
    if d["hiding"] == "prf":
        v["hm96_ands"] = 0.0
        M.compressions_per_row = _prf_compressions_per_row
    try:
        return A.evaluate(knobs, v, point)
    finally:
        M.compressions_per_row = _compressions_per_row
        _extra.clear()


def main(argv: list[str]) -> int:
    """`gdm.py flips SEARCH_JSON GRID POINT [OUT]`: `flips` at the nominal best design of a reduced search.json."""
    import json
    if len(argv) < 4 or argv[0] != "flips":
        print(main.__doc__)
        return 2
    _, path, grid_name, point = argv[:4]
    best = json.loads(Path(path).read_text())["scenarios"][grid_name][point]["nominal"]["top"][0]
    d = {k: tuple(x) if isinstance(x, list) else x for k, x in best["design"].items()}
    rows = flips(d, point)
    text = json.dumps({"point": point, "design": d, "key": key(d), "flips": rows}, indent=1, default=str)
    if len(argv) > 4:
        Path(argv[4]).write_text(text)
    for r in rows:
        print(f"{r['knob']:9} {str(r['value']):12} {'*' if r['at_d'] else ' '} J {r['J']:.6g} serve {r['serve']:.4g} "
              f"{'' if r['feasible'] else '+'.join(r['reasons'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
