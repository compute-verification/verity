"""`rt_hb_head`: RT-HB-HEAD's witness, the H200's and B200's headlines when ♭ credits only what the checks read.

    python benchmarks/pouw/rt_hb_head.py [--json F]

For #2699's designs and the repaired ones (B200 `dither-2s-licpw-redux-ldtm0-l32`: honest's plain LI-CP-W fold, the one
`PouwWarpCheck` recomputes, and no LDTM credit; H200 `dither-2s-licpw-redux-p1-l32`: ffill out of ♭), each reading's
Lean-form bound (`lean_headline_dc.reading`, ♭ as honest's program), ε_skip at its own ♭, the machine model's core ε
at A6's `q = ½` and at `q* = 1.22` with its floor program and cheater, and each sheet's fold-cost cliff: eps.py with
the design's fold scaled to R lane instructions a word, the largest R at which ε_machine rises above its value at the
design's own fold. `lean_recompute` reads each statement's bound from its Lean data text alone, in #2699's form and in
the proved `headline_of_branches`' (whose unfolded branch has no χ*).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import re
import sys
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lean_headline_dc as LD  # noqa: E402
from verity_machine_model import eps as E  # noqa: E402

ORIGINAL = {"Sm90": "dither-2s-licpw-redux-l32", "Sm100": "dither-2s-licpw-redux-heavy-ldtm0-l32"}  # #2699 @ f3c3237e8a
REPAIRED = {"Sm90": "dither-2s-licpw-redux-p1-l32", "Sm100": "dither-2s-licpw-redux-ldtm0-l32"}
GRID = (1.0, 0.995, 0.99, 0.98, 0.95, 0.93, 0.92, 0.90)


def _per_unit(n: dict[str, float], Nw: int) -> dict[str, float]:
    return {k: round(v * Nw, 4) for k, v in sorted(n.items())}


def readings() -> list[dict]:
    out = []
    for fam in LD.FAMILIES:
        for design in (ORIGINAL[fam.ns], REPAIRED[fam.ns]):
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
        for design in (ORIGINAL[fam.ns], REPAIRED[fam.ns]):
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


LEAN = Path(__file__).resolve().parents[2] / "Security" / "Definitions" / "Accounting" / "HeadlineDc"
STATEMENTS = Path(__file__).resolve().parents[2] / "Security" / "Models" / "Accounting" / "Guarantees" / "HeadlineDc.lean"


def stated(name: str) -> Fraction:
    """The ε `HeadlineDc.lean`'s statement for data file `name` states (`… < 1 - N / 100000`)."""
    text = STATEMENTS.read_text()
    stmt = name.replace("Measured", "") + ("Measured" if "Measured" in name else "Sheet") + "Headline"
    m = re.search(rf"^def {stmt} : Prop :=.*?< 1 - (\d+) / (\d+)\)", text, re.S | re.M)
    return Fraction(int(m.group(1)), int(m.group(2)))
LN2_HI = Fraction(6931471808, 10**10)   # Real.log_two_lt_d9
LN_P_LO = Fraction(1, 8)                # 1 − 7/8 ≤ ln(8/7)


def _block(text: str, name: str) -> str:
    """The text of `def NAME ... :=` up to the next declaration, comments dropped."""
    m = re.search(rf"^(?:noncomputable )?def {re.escape(name)}\b.*?(?=^/--|^(?:noncomputable )?def |^end |\Z)", text,
                  re.S | re.M)
    if not m:
        raise SystemExit(f"no `def {name}`")
    return re.sub(r"--[^\n]*", "", m.group(0))


def _list(s: str) -> list:
    """The first bracketed Lean list in `s`, nested lists and tuples of naturals, as Python."""
    i = s.index("[", s.index(":="))
    depth, j = 0, i
    while True:
        depth += {"[": 1, "]": -1}.get(s[j], 0)
        if depth == 0:
            break
        j += 1
    return eval(s[i:j + 1].replace("\n", " "))  # naturals, commas, brackets and parentheses only


def _field(s: str, f: str) -> str:
    m = re.search(rf"\b{f} :=\s*(.*?)(?=\n  \w+ :=|\n\n|\Z)", s, re.S)
    return m.group(1).strip()


def _sparse(v: list) -> dict[int, Fraction]:
    out: dict[int, Fraction] = {}
    for i, num, den in v:
        out[i] = out.get(i, Fraction(0)) + Fraction(num, den)
    return out


def _design(text: str, base: str) -> dict:
    """`d2s`'s fields, a field written as `SmX.d2s.f` read from the base file's `d2s`."""
    own, under = _block(text, "d2s"), _block(base, "d2s")
    pick = lambda f: under if _field(own, f).startswith("Sm") else own
    lst = lambda f: _list(pick(f)[pick(f).index(f" {f} :="):])
    return {"writers": lst("writers"), "vec": lst("vec"), "folds": [_sparse(f) for f in lst("folds")],
            "q": Fraction(_field(pick("q"), "q").replace(" ", "")),
            "qT": Fraction(_field(pick("qT"), "qT").replace(" ", "")), "Nw": int(_field(pick("Nw"), "Nw"))}


def lean_recompute() -> list[dict]:
    """Each statement's bound from the Lean data text alone: `Accounting.TS.Model`'s `demand`, `wStar`, `phiStar`,
    `chiStar` and `bnd` on the file's machine, ♭ and π*, the two branches `headline_of_branches` takes (its unfolded
    one with no χ*, `Proofs/Accounting/Headline/Whole.lean:296`) and #2699's (χ* in both)."""
    out = []
    for dev in ("Sm90", "Sm100"):
        base = (LEAN / f"{dev}.lean").read_text()
        for name in (dev, dev + "Measured"):
            text = (LEAN / f"{name}.lean").read_text()
            classes = eval(_block(base, "classes")[_block(base, "classes").index("["):])
            nk = len(classes)
            if name == dev:
                sd = _block(text, f"ds{dev[2:]}")
                macs, peak, smem = (eval(_field(sd, f)) for f in ("macs", "peak", "smem"))
                pipe = [None if p == "none" else p.split(".")[1] for p in
                        re.findall(r"some \.\w+|none", _field(sd, "pipe"))]
                width = {p: int(w) for p, w in re.findall(r"\| \.(\w+) => (\d+)", _block(text, "Pipe.width"))}
                rows = {"issue": {k: Fraction(1) for k in range(nk)},
                        "tensor": {k: (Fraction(macs[k]) / (Fraction(peak[k]) / 4) if peak[k] else Fraction(0))
                                   for k in range(nk)}}
                for p in width:
                    rows[f"pipe {p}"] = {k: (Fraction(128, width[p]) if pipe[k] == p else Fraction(0))
                                         for k in range(nk)}
                rows["smem"] = {k: Fraction(smem[k], 32) for k in range(nk)}
                lam: dict[str, Fraction] = {}
                for r, num, den in re.findall(r"\| \.(issue|tensor|smem|pipe \.\w+) => (\d+) / (\d+)",
                                              _block(text, "lamStar")):
                    lam[r.replace(".", "")] = Fraction(int(num), int(den))
                d = _design(text, base)
                own = text
            else:
                vec_name = re.search(r"^def (dc\w+) : List Sparse", text, re.M).group(1)
                rlist = _list(_block(text, vec_name))
                names = eval(_block(text, "rowNames")[_block(text, "rowNames").index("["):])
                rows = {names[i]: {k: v for k, v in _sparse(r).items()} for i, r in enumerate(rlist)}
                cert = _list(_block(text, "lamStar"))
                lam = {names[i]: Fraction(num, den) for i, num, den in cert}
                d = _design(text, base)
                own = text
            m = re.search(r"d2s\.model(Fill)? (\d+) (\d+) (\d+) [\w.]+( \w+\.flatFill| flatFill)?", own)
            fill_on, nt, nf = bool(m.group(1)), int(m.group(3)), int(m.group(4))
            if (int(m.group(2)), len(d["writers"]), len(d["folds"])) != (nk, nt, nf):
                raise SystemExit(f"{name}: Fin sizes {m.groups()} against {nk} classes, {len(d['writers'])} writers, "
                                 f"{len(d['folds'])} folds")
            src = lambda part: _sparse(_list(_block(text if f"def {part}" in text else base, part)))
            nw, nfo, nx, ns = src("flatWords"), src("flatFold"), src("flatFill"), src("flatSkip")
            flat = {k: nw.get(k, 0) + nfo.get(k, 0) + nx.get(k, 0) + ns.get(k, 0) for k in range(nk)}
            load = lambda n, r: sum((rows[r].get(k, Fraction(0)) * v for k, v in n.items()), Fraction(0))
            B = max(load(flat, r) for r in rows)
            if sum(lam.values()) != 1 or any(x < 0 for x in lam.values()) or not set(lam) <= set(rows):
                raise SystemExit(f"{name}: π* {lam} is no certificate over {list(rows)}")
            price = lambda dem: sum((x * dem[r] for r, x in lam.items()), Fraction(0))
            vecmin = {r: min(rows[r].get(k, Fraction(0)) for k in d["vec"]) for r in rows}
            routes = {f"writer {classes[c]}": {r: (rows[r].get(c, Fraction(0)) / om if om else Fraction(0))
                                               for r in rows} for c, om in d["writers"]}
            routes["vector"] = {r: d["q"] / 32 * vecmin[r] for r in rows}
            routes["table"] = {r: max(d["qT"] / 32 * vecmin[r], Fraction(0)) for r in rows}
            wk = min(routes, key=lambda h: price(routes[h]))
            w = price(routes[wk])
            phis = [price({r: load(f, r) for r in rows}) for f in d["folds"]]
            phi = min(phis)
            chi = price({r: load(nx, r) for r in rows}) if fill_on else Fraction(0)
            Nw, S = d["Nw"], 48000
            Y = (Nw * w + Nw * phi + chi) / B
            Y0 = (Nw * w + Nw * phi) / B
            kappa = Nw * w / B
            folded = 1 - Y + kappa * 41 * LN2_HI / (LN_P_LO * S)
            unf_2699 = 1 - Y * (1 - Fraction(41, S))
            unf_proved = 1 - Y0 * (1 - Fraction(41, S))
            out.append({"statement": name, "stated": float(stated(name)), "B_flat": float(B),
                        "flat_facet": max(rows, key=lambda r: load(flat, r)), "certificate": {r: float(x) for r, x in
                                                                                             lam.items()},
                        "wstar_route": wk, "Nw_w": float(Nw * w), "Nw_phi": float(Nw * phi),
                        "fold_prices": [float(x) for x in phis], "chi": float(chi), "folded": float(folded),
                        "unfolded_2699": float(unf_2699), "bound_2699": float(max(folded, unf_2699)),
                        "unfolded_proved": float(unf_proved), "bound_proved": float(max(folded, unf_proved)),
                        "meets_2699": max(folded, unf_2699) <= stated(name),
                        "meets_proved": max(folded, unf_proved) <= stated(name)})
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--json", help="write the readings and cliffs here")
    a = ap.parse_args(argv)
    rs, cs, ls = readings(), cliffs(), lean_recompute()
    for r in rs:
        m, m2 = r["machine_q0.5"], r["machine_q1.22"]
        print(f"{r['device']} {r['design']} on {r['vector']} ({r['menu']}): Lean bound {r['lean_bound']:.4%} "
              f"(folded {r['folded']:.4%}, unfolded {r['unfolded']:.4%}; B_♭ {r['B_flat']:.4f} on {r['flat_facet']}, "
              f"ε_skip {r['eps_skip']:.4%}); machine core {m['core']:.4%} (ε_machine {m['machine_eps']:.4%}, route "
              f"{m['route']:.4%}) at q ½, {m2['core']:.4%} at q* 1.22; ♭ is setup's floor: {r['flat_is_setup_floor']}")
    for c in cs:
        print(f"{c['device']} {c['design']} on {c['vector']}: fold {c['fold']:.6f} a word, cliff {c['cliff']:.6f}, "
              f"slack {c['slack']:.2%}; core by fold {c['core_by_fold']}")
    for x in ls:
        print(f"{x['statement']} from its Lean data: B_♭ {x['B_flat']:.4f} on {x['flat_facet']}, N_w w* {x['Nw_w']:.4f} "
              f"({x['wstar_route']}), N_w φ* {x['Nw_phi']:.4f}, χ* {x['chi']:.4f}; folded {x['folded']:.5%}; "
              f"#2699's form {x['bound_2699']:.5%} (meets {x['stated']:.3%}: {x['meets_2699']}); "
              f"headline_of_branches' form {x['bound_proved']:.5%} (meets: {x['meets_proved']})")
    if a.json:
        Path(a.json).write_text(json.dumps({"readings": rs, "cliffs": cs, "lean": ls}, indent=1, default=str) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
