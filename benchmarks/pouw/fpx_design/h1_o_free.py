"""On the data-centre rows: the served words H1's and Dither-2S's receipts leave unread, priced.

    uv run --project benchmarks/pouw/machine_model --with python-sat --with-editable benchmarks/pouw/exact_kit \
      python benchmarks/pouw/fpx_design/h1_o_free.py OUT

Run from a tree's root: it reads that tree's benchmarks/pouw, so a copy of this file runs on another branch's tree.

H1 (co32c-u16-cohalf, `dither-c-peel-fx1`) receipts T_1..T_last, O' and D_1..D_{m-1} on its linears and lm_head
(`cgemm_ref.py`'s co32c gate, `h1tile`'s CP-W receipt); its served output O = O' - sum 128^d D_d is no word and no check
reads it. `eps_budget.TS_EPS` takes co32c at `checked_strip`'s ts_co32c_recombination case "O committed" (O one more
folded word, recombined or at A = B = 0 one IMMA O = T_last - U.Y), within 1e-5 of "charged". As the receipts stand,
O is free: the cheater skips the recombination's 4(m - 1) IMADs a tile as well as the residue sums. So TS_EPS leaves
the recombination out of ♭'s credit, which is sound: pricing it as skippable (O free) raises the credit and ε_skip
together, and the floor (1 - bound) B_♭ the Lean builders give is the same either way. This prices co32c both ways on Llama-3.1 8B and 70B (linears at TS_SHAPES, lm_head at (h, 128256)), fails
unless the committed case reproduces TS_EPS, and gives sm_120's ε_skip (`design_skip`) and ε (`verity_machine_model`'s
eps of the design, floor credit, operands free, its skip moved by ε_skip's change, so `core` is exact) at the sheet
(DS-120, q 1/2), the measured reading (VF3r-16, 1.22), the measured vector at 1/2, and the Lean measured instance's
(VF3s-16, 9/8).

g1 on H1 serves 4 O - 2 sum X - 2 sum P^ + k (`g1h_eps.py`), so its correction is free beside O. The fix commits that
centred word S as one more receipted word beside O' and the D_d: the cheater then computes S, through the recombination
or the one-IMMA route, and pays g1's correction too.

The data-centre g1 rows (`fpx-g1s-i8`: the H200's and B200's Dither-2S records on R + b) serve
4 T_last - 2 sum X - 2 sum P^ + k, and their checks read T_last. Honest's correction, a token: `per_cell` lane
instructions a cell (1, an IADD3 of the row's and the column's terms on the integer word; 2, an IMAD and an IADD3), sum
X's one add an X entry a linear (k/32 warp instructions), and sum P^'s k/32 popcounts and adds a weight column shared by
2048 tokens (n k / (512 * 2048)); each warp instruction one SMSP-cycle on the device's issue row, as `floor_skip` prices
the peel's sums and epilogue. It is added to floor_skip's prefill pieces for linears and lm_head (decode's are products
alone there). The fix commits S in T_last's place, an injective affine image of it given the committed operands and
coins: no word or fold more, and the cheater pays the cell work and sum P^; sum X is left free to it (b's popcounts at
the zero instance), an upper bound on what stays open.

Writes OUT/h1_o_free.json and OUT/h1_o_free.md; exit 1 if a check fails."""
import argparse
import importlib.util
import json
import sys
import tomllib
from dataclasses import replace
from pathlib import Path

POUW = Path.cwd() / "benchmarks" / "pouw"
sys.path.insert(0, str(POUW))
import eps_budget as eb  # noqa: E402
from verity_machine_model import eps as E  # noqa: E402


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = sys.modules.setdefault(name, importlib.util.module_from_spec(spec))
    spec.loader.exec_module(m)
    return m


cs = _load("checked_strip", POUW / "theory" / "checked_strip.py")

DESIGN = "dither-c-peel-fx1"
SHAPES = {m: {"linears": [(k, n) for _, k, n in s.linears], "lm_head": [(s.h, s.vocab)]} for m, s in eb.MODELS.items()}
TOKENS = 2048
#: (title, co32c pricing, g1's lane instructions a cell, the cheater pays g1's correction)
CASES = [("H1, O committed (the record's TS_EPS)", "O committed", 0, False),
         ("H1, O charged (checked downstream)", "charged", 0, False),
         ("H1, O free (its receipts)", "free", 0, False),
         ("g1 on H1, O free, 1 a cell", "free", 1, False),
         ("g1 on H1, O free, 2 a cell", "free", 2, False),
         ("fix: g1's S one more receipted word, 1 a cell", "O committed", 1, True),
         ("fix: g1's S one more receipted word, 2 a cell", "O committed", 2, True)]
DC = {"sm_90": ("H200", "dither-2s-licpw-redux-l32"), "sm_100": ("B200", "dither-2s-licpw-redux-heavy-l32")}
#: (title, g1's lane instructions a cell, the parts honest pays and the cheater skips)
DC_CASES = [("the record (Dither-2S, served T_last)", 0, ()),
            ("g1 unfixed, 1 a cell", 1, ("cells", "sum X", "sum P^")),
            ("g1 unfixed, 2 a cell", 2, ("cells", "sum X", "sum P^")),
            ("fix: S in T_last's place (sum X free to the cheater)", 1, ("sum X",))]


def digits(k: int) -> int:
    return cs.n_digits((k // 32) * -128, (k // 32) * 127)


def g1_ops(k: int, per_cell: int) -> float:
    """g1's correction on H1, warp instructions a 128-cell tile: the cells and sum P^'s popcounts (sum X is the residue
    sums')."""
    return 4 * per_cell + k / (4 * TOKENS) if per_cell else 0.0


def co32c_tile(k: int, n: int, case: str, g: float = 0.0, g_cheat: bool = False,
               f: float = eb.TS_FOLD) -> tuple[float, float]:
    """(honest, the cheapest cheater) SMSP-cycles of a co32c tile, as checked_strip's ts_co32c_recombination prices the
    case, with g1's correction g on honest's side (and the cheater's when its word is committed)."""
    m = digits(k)
    (Mh, Oh), (Mc, Oc) = cs.ts_tile("co32c", k, n, f)
    gc, r = (g if g_cheat else 0.0), 4 * (m - 1)
    if case == "free":
        return cs.ts_time(Mh, Oh + g), cs.ts_time(Mc, Oc + gc)
    if case == "charged":
        return cs.ts_time(Mh, Oh + g), cs.ts_time(Mc, Oc + r + gc)
    tile = ((Mh, Oh + 4 * f + g), (Mc, Oc + 4 * f + r + gc))
    return cs.ts_time(*tile[0]), cs.ts_best("co32c", k, n, f, [(1.0, -4.0 * (m - 1))], tile)[1]


def ts_eps(shapes, case, per_cell=0, g_cheat=False) -> float:
    h = c = 0.0
    for k, n in shapes:
        th, tc = co32c_tile(k, n, case, g1_ops(k, per_cell), g_cheat)
        h, c = h + th * n, c + tc * n
    return h / c - 1


def honest_ratio(case, per_cell, g_cheat) -> float:
    """Honest's tensor-scalar time on 8B's linears under the case over H1's as built (O free, no correction)."""
    sh = SHAPES["llama-3.1-8b"]["linears"]
    w = sum(co32c_tile(k, n, case, g1_ops(k, per_cell), g_cheat)[0] * n for k, n in sh)
    return w / sum(co32c_tile(k, n, "free")[0] * n for k, n in sh)


def receipt_bits(K: int, extra: int = 0) -> int:
    """h1tile's CP-W receipt a unit: 2 words a receipted set, T_0..T_(K/32 - 1), O', D_1..D_(m-1), and `extra` more."""
    return 2 * (K // 32 + digits(K) + extra) * 32


def g1_dc(s, ctx: int, per_cell: int, parts) -> dict:
    """Honest's g1 correction a token-layer in prefill, SMSP-cycles at one a warp instruction, by component."""
    def one(k, n):
        p = {"cells": n * per_cell / 32, "sum X": k / 32, "sum P^": n * k / (512 * TOKENS)}
        return sum(p[x] for x in parts)
    return {"linears": sum(one(k, n) for _, k, n in s.linears), "lm_head": one(s.h, s.vocab) / (s.layers * ctx)}


def dc_skip(d, data, vec, per_cell, parts) -> dict:
    best = None
    for m, s in eb.MODELS.items():
        for c in eb.CONTEXTS:
            for ph in eb.PHASES:
                words, free, _ = eb.floor_skip(m, c, d, data=data, vector=vec, phase=ph)
                pieces = dict(free)
                if ph == "prefill" and parts:
                    for comp, v in g1_dc(s, c, per_cell, parts).items():
                        pieces[f"g1 {comp}"] = v
                tot = sum(pieces.values())
                cell = {"skip": tot / (words + tot), "at": f"{m} {c // 1024}K {ph}",
                        "pieces": {k: v / (words + tot) for k, v in pieces.items()}, "words": words}
                if best is None or cell["skip"] > best["skip"]:
                    best = cell
    return best


def main(out: Path) -> int:
    out.mkdir(parents=True, exist_ok=True)
    rc, lines = 0, []
    saved = {m: dict(v) for m, v in eb.TS_EPS["co32c"].items()}
    head = tomllib.loads((eb.DATA / "headline.toml").read_text())
    fam, q = head["family"]["sm_120"], {r: x["q"] for r, x in head["reading"].items()}
    readings = [(r, fam[r]["vector"], q[r]) for r in q] + [("measured-half", fam["measured"]["vector"], q["sheet"]),
                                                            ("Lean measured", "VF3s-16", 9 / 8)]
    d = E.design(DESIGN)
    record = {r: E.eps(d, E.machine(vec), qr, operands="free", credit="floor") for r, vec, qr in readings}

    cases, pricings = [], {"the record (eps_budget's TS_EPS as written)": saved}
    for title, case, per_cell, g_cheat in CASES:
        ts = {m: {c: ts_eps(sh, case, per_cell, g_cheat) for c, sh in comps.items()} for m, comps in SHAPES.items()}
        eb.TS_EPS["co32c"] = ts
        skips = {}
        for r, vec, _ in readings:
            s = eb.design_skip(d, vector=vec)
            skips[r] = {"skip": s.eps_time, "at": f"{s.model} {s.ctx // 1024}K {s.phase}"}
        cases.append({"case": title, "pricing": case, "per_cell": per_cell, "cheater_pays_g1": g_cheat, "ts_eps": ts,
                      "skip": skips, "honest_ratio": honest_ratio(case, per_cell, g_cheat)})
        if per_cell == 0 and case != "charged":
            pricings[title] = ts
    eb.TS_EPS["co32c"] = saved
    for m, comps in saved.items():
        for c, v in comps.items():
            if abs(cases[0]["ts_eps"][m][c] - v) > 5e-7:
                print(f"TS_EPS {m} {c}: {cases[0]['ts_eps'][m][c]:.4e} here, {v:.4e} in eps_budget", file=sys.stderr)
                rc = 1
    base = cases[0]["skip"]
    lines += ["## sm_120, H1 (dither-c-peel-fx1): co32c's recombination priced each way", "",
              "| case | TS ε 8B linears | 70B linears | lm_head | honest TS time (H1 = 1) | " +
              " | ".join(f"{r} ε_skip / ε" for r, _, _ in readings) + " |",
              "|---|---:|---:|---:|---:|" + "---:|" * len(readings)]
    for c in cases:
        c["eps"] = {}
        cells = []
        for r, _, _ in readings:
            e = record[r]
            c["eps"][r] = replace(e, skip=e.skip - base[r]["skip"] + c["skip"][r]["skip"]).core()
            cells.append(f"{100 * c['skip'][r]['skip']:.4f}% / **{100 * c['eps'][r]:.4f}%**")
        ts = c["ts_eps"]
        lines.append(f"| {c['case']} | {100 * ts['llama-3.1-8b']['linears']:.4f}% | "
                     f"{100 * ts['llama-3.1-70b']['linears']:.4f}% | {100 * ts['llama-3.1-8b']['lm_head']:.4f}% | "
                     f"{c['honest_ratio']:.5f} | " + " | ".join(cells) + " |")
    lines += ["", "The record's own skip term and core by reading (registry `[skip]`, against design_skip at O committed): "
              + "; ".join(f"{r} {vec}@{qr:g}: registry {100 * record[r].skip:.4f}%, live {100 * base[r]['skip']:.4f}% "
                          f"({base[r]['at']}), core {100 * record[r].core():.4f}%" for r, vec, qr in readings)]
    for r, _, _ in readings[:2]:
        if abs(record[r].skip - base[r]["skip"]) > 5e-6:
            print(f"{r}: the registry's skip {record[r].skip} is not design_skip's {base[r]['skip']}", file=sys.stderr)
            rc = 1
    rb = {K: (receipt_bits(K), receipt_bits(K, 1)) for K in sorted({k for sh in SHAPES.values() for k, _ in sh["linears"]})}
    lines += ["", "h1tile's receipt a unit, bits, as built and with S one more receipted word: " + "; ".join(
        f"K {K}: {a} -> {b} (+{100 * (b / a - 1):.2f}%)" for K, (a, b) in rb.items())]

    dc = {}
    lines += ["", "## Data-centre g1 rows (#2710's fpx-g1s-i8, the Dither-2S records on R + b): g1's correction", ""]
    for dev, (gpu, name) in DC.items():
        data = eb.DATA / "devices" / dev
        dd = E.design(name, data)
        famd = head["family"][dev]
        rds = [(r, famd[r]["vector"], q[r]) for r in q] + [("measured-half", famd["measured"]["vector"], q["sheet"])]
        recd = {r: E.eps(dd, E.machine(vec, data), qr, data, operands="free", credit="floor") for r, vec, qr in rds}
        rows = []
        for title, per_cell, parts in DC_CASES:
            sk = {r: dc_skip(dd, data, vec, per_cell, parts) for r, vec, _ in rds}
            rows.append({"case": title, "per_cell": per_cell, "parts": list(parts), "skip": sk})
        for r, vec, _ in rds:
            live = eb.design_skip(dd, data=data, vector=vec).eps_time
            if abs(rows[0]["skip"][r]["skip"] - live) > 1e-12:
                print(f"{dev} {r}: dc_skip at no g1 {rows[0]['skip'][r]['skip']} is not design_skip's {live}",
                      file=sys.stderr)
                rc = 1
        lines += [f"### {gpu} ({dev}, {name})", "",
                  "| case | " + " | ".join(f"{r} ({vec}@{qr:g}) ε_skip (at; g1's share) / ε" for r, vec, qr in rds) + " |",
                  "|---|" + "---:|" * len(rds)]
        for row in rows:
            row["eps"], cells = {}, []
            for r, _, _ in rds:
                e, s = recd[r], row["skip"][r]
                row["eps"][r] = replace(e, skip=e.skip - rows[0]["skip"][r]["skip"] + s["skip"]).core()
                g1 = sum(v for k, v in s["pieces"].items() if k.startswith("g1"))
                cells.append(f"{100 * s['skip']:.4f}% ({s['at']}; {100 * g1:.4f}%) / **{100 * row['eps'][r]:.4f}%**")
            lines.append(f"| {row['case']} | " + " | ".join(cells) + " |")
        lines += ["", "The record: " + "; ".join(
            f"{r} registry skip {100 * recd[r].skip:.4f}%, live {100 * rows[0]['skip'][r]['skip']:.4f}%, "
            f"core {100 * recd[r].core():.4f}%" for r, _, _ in rds), ""]
        dc[dev] = {"gpu": gpu, "design": name, "readings": rds, "rows": rows,
                   "record": {r: {"skip": e.skip, "core": e.core()} for r, e in recd.items()}}

    (out / "h1_o_free.md").write_text("\n".join(lines) + "\n")
    (out / "h1_o_free.json").write_text(json.dumps({
        "design": DESIGN, "readings": readings, "ts_eps_record": saved, "cases": cases,
        "record": {r: {"skip": e.skip, "core": e.core(), "terms": e.terms()} for r, e in record.items()},
        "receipt_bits": {str(K): v for K, v in rb.items()}, "dc": dc}, indent=1,
        default=str) + "\n")
    print("\n".join(lines))
    return rc


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    a = ap.parse_args()
    sys.exit(main(a.out))
