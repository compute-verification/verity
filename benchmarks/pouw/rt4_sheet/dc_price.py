"""RT4-SHEET on the H200 and B200: fold_bulk.cu's timed loops (fold_bulk_sass.sh's SASS for sm_90a and sm_100a) priced
on each family's data sheet (DS-90, DS-100) at π*, the certificate `lean_headline_dc` proves the sheet bound at, against
its fold floor φ* a word.

Each SASS opcode is classed by the tree's `facet_break/loads.class_of` and priced π*·c[class]; MMA forms and NOP are
dropped (the word route's, not the fold's). An opcode whose class is None or not among the sheet's classes is priced 0:
neither family has a catch-all class (`lean_headline.CATCH_ALL` names sm_120 only), so the sheet's model counts no
such instruction. Each is listed, and the route is priced again with each at π*'s issue weight alone, as an A2
catch-all would price it. The timed loop splits as fold_bulk.sh's: what every warp issues, and the bodies of forward
predicated branches (thread 0's, or warp 0's), once a block of W = BT / 32 warps; a word's price is
(outside + bodies / W) / (1024 K), 8 units of 128 words a logical warp a set.

    uv run python benchmarks/pouw/rt4_sheet/dc_price.py --tree TREE SASS_SM90 SASS_SM100 [--json OUT]

Run from a tree whose `benchmarks/pouw/lean_headline_dc.py` is the statement under test (DC-CENSUS's head of #2699).
Exit 0, or 1 when a route prices under φ* on a sheet.
"""
from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import re
import sys
from fractions import Fraction
from pathlib import Path

INS = re.compile(r"/\*([0-9a-f]{4,})\*/\s+((?:@!?U?P[T0-9]+\s+)?)([A-Z][A-Z0-9_.]*)([^;]*);")
NAME = re.compile(r"(fb[kuw]|fcp)ILi(\d+)ELi(\d+)E(?:Li(\d+)E)?")


def use_tree(pouw: Path) -> None:
    """Import the tree's own machine model, not the environment's editable install of another checkout's."""
    sys.path[:0] = [str(pouw / "machine_model"), str(pouw)]
    for k in [m for m in sys.modules if m.startswith("verity_machine_model")]:
        del sys.modules[k]


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def targets(rest: str) -> list[int]:
    return [int(t, 16) for t in re.findall(r"0x([0-9a-f]+)", rest)]


def loops(sass: str):
    """Each fold_bulk kernel: (name, K, BT, every-warp opcodes, body opcodes) of its timed loop."""
    parts = re.split(r"\n\s*Function : (\S+)\n", "\n" + sass)
    for name, body in zip(parts[1::2], parts[2::2]):
        m = NAME.search(name)
        if not m:
            continue
        kern, K, BT = m[1], int(m[2]), int(m[3])
        tag = {"fbk": f"fbk{'256' if m[4] == '1' else '128'}", "fbu": "fbv" if m[4] == "1" else "fbu",
               "fbw": f"fbw{m[4]}", "fcp": f"fcp{m[4]}"}[kern]
        ins = [(int(a, 16), p.strip(), op, rest) for a, p, op, rest in INS.findall(body)]
        back = [(t, a) for a, p, op, rest in ins if op.startswith("BRA") for t in targets(rest) if t < a]
        head, tail = max(back, key=lambda lp: lp[1] - lp[0])
        loop = [x for x in ins if head <= x[0] <= tail]
        bodies = [(a, targets(rest)[0]) for a, p, op, rest in loop
                  if op in ("BRA", "BRA.U") and p and targets(rest) and targets(rest)[0] > a]
        inside = [op for a, p, op, r in loop if any(b < a < t for b, t in bodies)]
        outside = [op for a, p, op, r in loop if not any(b < a < t for b, t in bodies)]
        yield f"{tag}_K{K}_BT{BT}", K, BT, outside, inside


def price_family(LHD, L, fam, sass: str) -> dict:
    r = LHD.reading(fam, fam.sheet)
    x = r.data
    lam, c = x.lam, x.c
    phi, w, B, bound = (Fraction(r.exact[k]) for k in ("phi", "w", "B", "bound"))
    issue = lam.get("DS_issue", Fraction(0))
    mma = set(L.MMA_FORMS)
    out = {"family": fam.ns, "gpu": fam.gpu, "sheet": fam.sheet, "design": fam.design, "pistar": r.lam,
           "phi_star": str(phi), "phi_star_float": float(phi), "Nw": r.Nw, "w_star": str(w), "B": float(B),
           "bound": float(bound), "target": r.target, "routes": {}}
    for name, K, BT, outside, inside in loops(sass):
        W = BT // 32
        words = 1024 * K
        free: collections.Counter = collections.Counter()

        def cost(ops: list[str], catch_all: bool) -> Fraction:
            tot = Fraction(0)
            for op in ops:
                if op == "NOP":
                    continue
                k = L.class_of(op)
                if k in mma or op.startswith(("IMMA", "HMMA", "QMMA", "OMMA")):
                    continue
                if k is None or k not in c:
                    if not catch_all:
                        free[f"{op} ({k})" if k else op] += 1
                    tot += issue if catch_all else 0
                    continue
                tot += sum((lam_r * c[k].get(row, Fraction(0)) for row, lam_r in lam.items()), Fraction(0))
            return tot

        per_word = (cost(outside, False) + cost(inside, False) / W) / words
        per_word_ca = (cost(outside, True) + cost(inside, True) / W) / words
        # the model's time/credit for a full-work run that writes its words as ♭'s tensor route does (w* a word) and
        # folds every word by this route: (N_w w* + N_w route) / B against 1 − the sheet bound
        tc = (r.Nw * w + r.Nw * per_word) / B
        out["routes"][name] = {
            "K": K, "BT": BT, "warps": W, "per_word": float(per_word), "per_word_with_catch_all": float(per_word_ca),
            "over_phi": float(per_word / phi), "over_phi_with_catch_all": float(per_word_ca / phi),
            "below_phi": per_word < phi, "time_over_credit": float(tc), "one_minus_bound": float(1 - bound),
            "breaks_conclusion": tc < 1 - bound, "unpriced": dict(free.most_common())}
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tree", type=Path, default=Path.cwd())
    ap.add_argument("sass90", type=Path)
    ap.add_argument("sass100", type=Path)
    ap.add_argument("--json", type=Path)
    a = ap.parse_args(argv)
    pouw = a.tree.resolve() / "benchmarks" / "pouw"
    use_tree(pouw)
    LHD = _load(pouw / "lean_headline_dc.py", "rt4_lean_headline_dc")
    L = _load(pouw / "facet_break" / "loads.py", "rt4_loads")
    res = [price_family(LHD, L, LHD.SM90, a.sass90.read_text()),
           price_family(LHD, L, LHD.SM100, a.sass100.read_text())]
    if a.json:
        a.json.write_text(json.dumps(res, indent=1, sort_keys=True))
    under = []
    for f in res:
        print(f"{f['family']} ({f['gpu']}, {f['sheet']}, {f['design']}): π* {f['pistar']}, φ* {f['phi_star']} "
              f"({f['phi_star_float']:.6f}) a word, N_w {f['Nw']}, bound {f['bound']:.6%} (target {f['target']:.4%})")
        for n, v in f["routes"].items():
            print(f"  {n}: {v['per_word']:.6f} a word = {v['over_phi']:.4f} φ* (with a catch-all "
                  f"{v['over_phi_with_catch_all']:.4f} φ*); time/credit {v['time_over_credit']:.5f} vs "
                  f"{v['one_minus_bound']:.5f}{'  BREAKS' if v['breaks_conclusion'] else ''}")
            if v["unpriced"]:
                print(f"    unpriced on {f['sheet']}: {v['unpriced']}")
            if v["below_phi"]:
                under.append(f"{f['family']} {n}")
    print(f"routes under φ*: {under or 'none'}")
    return 1 if under else 0


if __name__ == "__main__":
    raise SystemExit(main())
