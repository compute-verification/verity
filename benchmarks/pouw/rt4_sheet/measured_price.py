"""RT4-SHEET-1 on both sm_120 readings: fold_bulk.cu's timed loops (sm_120a SASS) priced as
`test_lean_headline.test_bit_exact_folds_meet_fold_opt_at_pi_star` prices a fold, at the sheet's π* (DS_issue on DS-120,
`lean_headline.load`) and the measured reading's (JI3t_0 on VF3s-16, `lean_headline.mload`), against φ* there (the
cheapest of the instance's fold routes).

A unit's class counts are fold_bulk.sh's split: what every warp issues, and the bodies of forward predicated branches
once a block of W = BT / 32 warps, over 8 K units a set; IMMA and NOP dropped, an unclassed opcode in NOROW (A2's
catch-all). `loads.class_of` classes UBLKRED as UNI (its uniform rule fires first), so the route is priced again with
the bulk reductions in the census's own class for them, TMAred512 (`loads.SIDE_BY_NAME`: "a 4 KB bulk op is several
TMAred512"): each word of the slot is reduced twice, 4 bytes each, so 1024 B, 2 TMAred512, a unit.

    PYTHONPATH=TREE/benchmarks/pouw/machine_model python benchmarks/pouw/rt4_sheet/measured_price.py --tree TREE SASS

TREE is the statement under test (#2791's head). Prints each route and its time/credit on each reading for a full-work
run that produces its words as ♭ does (`Instance.nw`) and folds by the route.
"""
from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import sys
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from dc_price import loops  # noqa: E402


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tree", type=Path, required=True)
    ap.add_argument("sass", type=Path)
    ap.add_argument("--json", type=Path)
    a = ap.parse_args(argv)
    pouw = a.tree.resolve() / "benchmarks" / "pouw"
    sys.path.insert(0, str(pouw))
    H = _load(pouw / "lean_headline.py", "rt4_lean_headline")
    L = H.loads if hasattr(H, "loads") else _load(pouw / "facet_break" / "loads.py", "rt4_loads")
    i = H.instance()
    m = H.measured(i)
    catch = H.CATCH_ALL["sm_120"]
    nw = i.Nw
    phi = nw * min(H.load(i, i.pistar, f) for f in i.folds)
    phi_m = nw * min(H.mload(m, m.pistar, f) for f in i.folds)
    words, words_m = H.load(i, i.pistar, i.nw), H.mload(m, m.pistar, i.nw)
    rows = ["DS_issue", "DS_tensor", "DS_smem", *i.fam.pipes]
    print(f"sheet: π* {i.pistar}, N_w φ* {phi} a unit, ♭ {float(i.B):.6f}, bound {float(i.bound):.6%}, "
          f"word production {float(words):.6f}")
    print(f"measured: π* {m.pistar}, N_w φ* {float(phi_m):.6f} a unit, ♭ {float(m.B):.6f}, bound {float(m.bound):.6%}, "
          f"word production {float(words_m):.6f}")
    out = {"sheet": {"pistar": i.pistar, "phi": float(phi), "B": float(i.B), "bound": float(i.bound)},
           "measured": {"pistar": m.pistar, "phi": float(phi_m), "B": float(m.B), "bound": float(m.bound)},
           "routes": {}}
    under = []
    for name, K, BT, outside, inside in loops(a.sass.read_text()):
        W, units = BT // 32, 8 * K
        ops = collections.Counter()
        for op in outside:
            ops[op] += Fraction(1, units)
        for op in inside:
            ops[op] += Fraction(1, units * W)
        n: dict[str, Fraction] = collections.defaultdict(Fraction)
        blk = Fraction(0)
        for op, v in ops.items():
            if op == "NOP" or op.startswith(("IMMA", "HMMA", "QMMA", "OMMA")):
                continue
            k = L.class_of(op)
            n[k if k in i.classes else catch] += v
            if op.startswith("UBLKRED"):
                blk += v
        nb = dict(n)
        nb["UNI"] -= blk
        nb["TMAred512"] = nb.get("TMAred512", Fraction(0)) + 2
        r = {}
        for tag, cnt in (("as_classed", dict(n)), ("bulk_as_TMAred512", nb)):
            s, s_m = H.load(i, i.pistar, cnt), H.mload(m, m.pistar, cnt)
            tot = {k: i.nw.get(k, 0) + cnt.get(k, 0) for k in {*i.nw, *cnt}}
            top = max(rows, key=lambda q: H.load(i, q, tot))
            top_m = max(m.rows, key=lambda q: H.mload(m, q, tot))
            r[tag] = {"sheet": float(s), "sheet_over_phi": float(s / phi), "measured": float(s_m),
                      "measured_over_phi": float(s_m / phi_m),
                      "sheet_time_over_credit": float((words + s) / i.B),
                      "measured_time_over_credit": float((words_m + s_m) / m.B),
                      "sheet_max_row": [top, float(H.load(i, top, tot) / i.B)],
                      "measured_max_row": [top_m, float(H.mload(m, top_m, tot) / m.B)]}
            if s < phi:
                under.append(f"{name} {tag} sheet")
            if s_m < phi_m:
                under.append(f"{name} {tag} measured")
        r["bulk_reductions_a_unit"] = float(blk)
        out["routes"][name] = r
        c, b = r["as_classed"], r["bulk_as_TMAred512"]
        print(f"  {name}: sheet {c['sheet']:.4f} ({c['sheet_over_phi']:.4f} φ*, time/credit "
              f"{c['sheet_time_over_credit']:.5f} vs {1 - float(i.bound):.5f}); measured {c['measured']:.4f} "
              f"({c['measured_over_phi']:.4f} φ*, time/credit {c['measured_time_over_credit']:.5f} vs "
              f"{1 - float(m.bound):.5f}); bulk as TMAred512: sheet {b['sheet']:.4f}, measured {b['measured']:.4f} "
              f"({b['measured_over_phi']:.4f} φ*)")
        print(f"    the most-loaded row over ♭: sheet {c['sheet_max_row']}, measured {c['measured_max_row']}, "
              f"measured with bulk as TMAred512 {b['measured_max_row']}")
    print(f"under φ*: {under or 'none'}")
    if a.json:
        a.json.write_text(json.dumps(out, indent=1, sort_keys=True))
    return 1 if under else 0


if __name__ == "__main__":
    raise SystemExit(main())
