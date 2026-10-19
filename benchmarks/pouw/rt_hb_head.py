"""`rt_hb_head`: RT-HB-HEAD's witness, the H200's and B200's headlines when ♭ credits only what the checks read.

    python benchmarks/pouw/rt_hb_head.py [--json F]

For #2699's designs and the repaired ones (B200 `dither-2s-licpw-redux-ldtm0-l32`: honest's plain LI-CP-W fold, the one
`PouwWarpCheck` recomputes, and no LDTM credit; H200 `dither-2s-licpw-redux-p1-l32`: ffill out of ♭), each reading's
Lean-form bound (`lean_headline_dc.reading`, ♭ as honest's program), ε_skip at its own ♭, the machine model's core ε
at A6's `q = ½` and at `q* = 1.22` with its floor program and cheater, and each sheet's fold-cost cliff: eps.py with
the design's fold scaled to R lane instructions a word, the largest R at which ε_machine rises above its value at the
design's own fold.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lean_headline_dc as LD  # noqa: E402
from verity_machine_model import eps as E  # noqa: E402

REPAIRED = {"Sm90": "dither-2s-licpw-redux-p1-l32", "Sm100": "dither-2s-licpw-redux-ldtm0-l32"}
GRID = (1.0, 0.995, 0.99, 0.98, 0.95, 0.93, 0.92, 0.90)


def _per_unit(n: dict[str, float], Nw: int) -> dict[str, float]:
    return {k: round(v * Nw, 4) for k, v in sorted(n.items())}


def readings() -> list[dict]:
    out = []
    for fam in LD.FAMILIES:
        for design in (fam.design, REPAIRED[fam.ns]):
            f = dataclasses.replace(fam, design=design)
            d = E.design(design, f.data)
            for v in (f.sheet, f.measured):
                r = LD.reading(f, v, floor_check=False)
                pi = E.machine(v, f.data)
                s = E.setup(d, pi, f.data, operands="free", credit="floor")
                row = {"device": fam.gpu, "design": design, "vector": v, "menu": r.menu,
                       "lean_bound": r.bound, "folded": r.folded, "unfolded": r.unfolded, "one_minus_Y": r.one_minus_Y,
                       "B_flat": r.B, "flat_facet": r.flat_facet, "certificate": r.certificate, "wstar_kind": r.wstar,
                       "w": r.w, "phi": r.phi, "chi": r.chi, "eps_skip": r.eps_skip, "exact_bound": r.exact["bound"],
                       "setup_floor": _per_unit(s.baseline.n, r.Nw), "setup_floor_T": s.baseline.T * r.Nw,
                       "flat_is_setup_floor": abs(s.baseline.T * r.Nw - (r.B * (1 - r.eps_skip))) < 1e-3}
                for q in (0.5, 1.22):
                    e = E.eps(d, pi, q, f.data, operands="free", credit="floor")
                    row[f"machine_q{q}"] = {"core": e.core(), "machine_eps": e.machine_eps, "route": e.route,
                                            "luck": e.luck, "cheat": _per_unit(e.cheat.n, r.Nw),
                                            "cheat_T": e.cheat.T * r.Nw, "cheat_binding": list(e.cheat.binding),
                                            "cheat_words": {k: round(x * r.Nw / 4096, 3) for k, x in
                                                            e.cheat.words.items()}}
                out.append(row)
    return out


def cliffs() -> list[dict]:
    out = []
    for fam in LD.FAMILIES:
        pi = E.machine(fam.sheet, fam.data)
        for design in (fam.design, REPAIRED[fam.ns]):
            d0 = E.design(design, fam.data)
            R0 = sum(d0.fold.values())

            def at(R: float) -> E.Eps:
                d = dataclasses.replace(d0, fold={k: x * R / R0 for k, x in d0.fold.items()})
                return E.eps(d, pi, 0.5, fam.data, operands="free", credit="floor")

            lo, hi = 0.02 * R0, R0
            m0 = at(R0).machine_eps
            for _ in range(40):
                mid = (lo + hi) / 2
                lo, hi = (lo, mid) if at(mid).machine_eps <= m0 + 1e-7 else (mid, hi)
            grid = {f"{R0 * g:.4f}": round(at(R0 * g).core(), 6) for g in GRID}
            out.append({"device": fam.gpu, "design": design, "vector": fam.sheet, "fold": R0, "cliff": hi,
                        "slack": 1 - hi / R0, "core_by_fold": grid})
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--json", help="write the readings and cliffs here")
    a = ap.parse_args(argv)
    rs, cs = readings(), cliffs()
    for r in rs:
        m, m2 = r["machine_q0.5"], r["machine_q1.22"]
        print(f"{r['device']} {r['design']} on {r['vector']} ({r['menu']}): Lean bound {r['lean_bound']:.4%} "
              f"(folded {r['folded']:.4%}, unfolded {r['unfolded']:.4%}; B_♭ {r['B_flat']:.4f} on {r['flat_facet']}, "
              f"ε_skip {r['eps_skip']:.4%}); machine core {m['core']:.4%} (ε_machine {m['machine_eps']:.4%}, route "
              f"{m['route']:.4%}) at q ½, {m2['core']:.4%} at q* 1.22; ♭ is setup's floor: {r['flat_is_setup_floor']}")
    for c in cs:
        print(f"{c['device']} {c['design']} on {c['vector']}: fold {c['fold']:.6f} a word, cliff {c['cliff']:.6f}, "
              f"slack {c['slack']:.2%}; core by fold {c['core_by_fold']}")
    if a.json:
        Path(a.json).write_text(json.dumps({"readings": rs, "cliffs": cs}, indent=1, default=str) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
