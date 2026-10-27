"""g1 on H1 (co32c-u16; `dither-c-peel-fx1`, the RTX PRO 6000's record) on the machine model:
its ε, with g1's served correction priced as the record prices co32c's recombination, and its γ against plain FP8 and
plain NVFP4.

    uv run --project benchmarks/pouw/machine_model --with python-sat --with-editable benchmarks/pouw/exact_kit \
      python benchmarks/pouw/fpx_design/g1h_eps.py OUT

g1 serves (R_x + b - 1/2)(R_w + c - 1/2): H1's word O = (R_w + c) X less its rank-one terms, 4 served =
4 O - 2 ΣX - 2 ΣP^ + k (W4A8's dithered activations: 2 served = 2 O - ΣX). ΣX is H1's residue sums summed; ΣP^ is the
row's committed levels plus a popcount of its live coins c, k/32 popcounts and adds a row, shared by the call's
tokens. No check reads the correction, so on H1's kernel unchanged it is vector work only the served output needs:
a skip term, priced like co32c's recombination in checked_strip's tensor-scalar model (`eps_budget.TS_EPS`, from
`checked_strip.ts_tile`), at one lane instruction a cell (an FFMA beside the dequantizing FMULs) or two (an IMAD and an
IADD3 on the integer word), plus ΣP^'s popcounts at 2K's tokens (k / (4 * 2048) warp instructions a 128-cell tile, the
most over the contexts). The record's TS_EPS prices co32c with O committed (the cheater recombines O); main's
`cgemm_ref.py` receipts are over T, O' and D, so each case is also priced with O free. ε_skip is `eps_budget.design_skip`
(the most over Llama-3.1 8B and 70B, 2K to 32K, prefill and decode) with TS_EPS's co32c entries replaced; the other
terms are `machine-model eps dither-c-peel-fx1` at fpx_design's readings (`fpx_eps.py`: credit floor, operands free,
the sheet at q* = 1/2, measured at 1.22 and at the proved 1/2). If g1's centred word is the committed served word in
O's place (the check recomputing it from the transcript), the correction is charged and ε is H1's.

γ is H1's honest time a MMA over the plain int8 MMA (`machine-model eps`'s), times the tensor-scalar model's honest
time with the correction over without, on Llama-3.1-8B's linears. Against plain FP8 (and W4A8, which runs at FP8's
rate) and plain NVFP4: the vector's form prices (16 SMSP-cycles a firing for IMMA16832, QMMA16832 and OMMASF16864 on
both, so x1 and x2), and the census's JI3s_20 alone (VF3r-16: IMMA 15.1152, QMMA 15.3456, OMMASF 15.1647). Writes
OUT/g1h_eps.json and OUT/g1h_eps.md; exit 1 if the replaced TS_EPS at no correction doesn't reproduce the record's."""
import csv
import importlib.util
import json
import subprocess
import sys
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent
POUW = HERE.parent
sys.path.insert(0, str(POUW))
import eps_budget as eb  # noqa: E402
from verity_machine_model.eps import machine  # noqa: E402

_spec = importlib.util.spec_from_file_location("checked_strip", POUW / "theory" / "checked_strip.py")
cs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cs)

DESIGN = "dither-c-peel-fx1"
#: (model, component) shapes (k, n): TS_SHAPES' linears, lm_head on the 128256-token vocabulary
SHAPES = {"llama-3.1-8b": {"linears": [(k, n) for _, k, n in cs.TS_SHAPES["Llama-3.1-8B"]], "lm_head": [(4096, 128256)]},
          "llama-3.1-70b": {"linears": [(k, n) for _, k, n in cs.TS_SHAPES["Llama-3.1-70B"]],
                            "lm_head": [(8192, 128256)]}}
#: tokens sharing a row's coin popcount: 2K prefill's, the fewest the TS terms are priced at
TOKENS = 2048
CASES = [("H1 (O committed, the record's TS_EPS)", True, 0), ("g1 on H1, 1 lane instruction a cell", True, 1),
         ("g1 on H1, 2 lane instructions a cell", True, 2), ("H1 with O free (cgemm_ref's receipts)", False, 0),
         ("g1 on H1 with O free, 1 lane instruction a cell", False, 1),
         ("g1 on H1 with O free, 2 lane instructions a cell", False, 2)]


def g1_ops(k: int, per_cell: int) -> float:
    """g1's correction, warp instructions a 128-cell tile (4 cells a lane), with ΣP^'s popcounts and adds."""
    return 4 * per_cell + (k / (4 * TOKENS) if per_cell else 0.0)


def tile_times(k: int, n: int, per_cell: int, o_committed: bool, f: float = eb.TS_FOLD) -> tuple[float, float]:
    """(honest, the cheapest cheater) SMSP-cycles of a co32c tile, as `checked_strip`'s ts_co32c_recombination prices
    them ("O committed" or "free"), with g1's correction on honest's side only."""
    m = cs.n_digits((k // 32) * -128, (k // 32) * 127)
    (Mh, Oh), (Mc, Oc) = cs.ts_tile("co32c", k, n, f)
    g = g1_ops(k, per_cell)
    if not o_committed:
        return cs.ts_time(Mh, Oh + g), cs.ts_time(Mc, Oc)
    tile = ((Mh, Oh + 4 * f + g), (Mc, Oc + 4 * f + 4 * (m - 1)))
    return cs.ts_time(Mh, Oh + 4 * f + g), cs.ts_best("co32c", k, n, f, [(1.0, -4.0 * (m - 1))], tile)[1]


def ts_eps(shapes, per_cell: int, o_committed: bool) -> float:
    h = c = 0.0
    for k, n in shapes:
        th, tc = tile_times(k, n, per_cell, o_committed)
        h, c = h + th * n, c + tc * n
    return h / c - 1


def gamma_ratio(per_cell: int, o_committed: bool) -> float:
    """Honest's tensor-scalar time on 8B's linears with g1's correction over without."""
    w = sum(tile_times(k, n, per_cell, o_committed)[0] * n for k, n in SHAPES["llama-3.1-8b"]["linears"])
    wo = sum(tile_times(k, n, 0, o_committed)[0] * n for k, n in SHAPES["llama-3.1-8b"]["linears"])
    return w / wo


def census_alone() -> dict:
    rows = csv.DictReader((eb.DATA / "census" / "VF3r-16.csv").open())
    return {r["kind"]: float(r["cycles"]) for r in rows if r["row"] == "JI3s_20"}


def main(out: Path) -> int:
    out.mkdir(parents=True, exist_ok=True)
    head = tomllib.loads((eb.DATA / "headline.toml").read_text())
    fam, q = head["family"]["sm_120"], {r: x["q"] for r, x in head["reading"].items()}
    readings = [(r, fam[r]["vector"], q[r]) for r in q] + [("measured-half", fam["measured"]["vector"], q["sheet"])]
    record = {}
    for reading, vec, qr in readings:
        cmd = ["machine-model", "eps", DESIGN, "--credit", "floor", "--operands", "free", "--q", f"{qr:g}",
               "--vector", vec, "--json"]
        record[reading] = json.loads(subprocess.run(cmd, capture_output=True, text=True, check=True).stdout)
    d = eb.design(DESIGN, eb.DATA)
    saved = {m: dict(v) for m, v in eb.TS_EPS["co32c"].items()}
    rc, cases = 0, []
    for title, oc, per_cell in CASES:
        eb.TS_EPS["co32c"] = {m: {c: ts_eps(sh, per_cell, oc) for c, sh in comps.items()} for m, comps in SHAPES.items()}
        skips = {}
        for reading, vec, _ in readings:
            s = eb.design_skip(d, vector=vec)
            skips[reading] = {"skip": s.eps_time, "at": f"{s.model} {s.ctx} {s.phase}"}
        cases.append({"case": title, "o_committed": oc, "lane_instructions_a_cell": per_cell,
                      "ts_eps": {m: dict(v) for m, v in eb.TS_EPS["co32c"].items()}, "skip": skips,
                      "gamma_ratio": gamma_ratio(per_cell, oc)})
    eb.TS_EPS["co32c"] = saved
    h1 = cases[0]
    for m, comps in saved.items():
        for c, v in comps.items():
            if abs(h1["ts_eps"][m][c] - v) > 5e-7:
                print(f"TS_EPS {m} {c}: {h1['ts_eps'][m][c]:.4e} here, {v:.4e} in eps_budget", file=sys.stderr)
                rc = 1
    alone = census_alone()
    factors = {"FP8 (x the vector's forms)": 1.0, "NVFP4 (x the vector's forms)": 2.0,
               "FP8 (x JI3s_20 alone)": alone["IMMA16832"] / alone["QMMA16832"],
               "NVFP4 (x JI3s_20 alone)": 2 * alone["IMMA16832"] / alone["OMMASF16864"]}
    pi = {v: {f: max(machine(v).load({f: 1}).values()) for f in ("IMMA16832", "QMMA16832", "OMMASF16864")}
          for _, v, _ in readings}
    lines = ["| case | reading | ε_skip (at) | ε | γ (int8) | γ / FP8 | γ / NVFP4 | γ / NVFP4 (JI3s_20) |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for c in cases:
        c["eps"], c["gamma"] = {}, {}
        for reading, vec, _ in readings:
            j = record[reading]
            eps = j["core"] - j["terms"]["skip"] + c["skip"][reading]["skip"]
            g = j["gamma"] * c["gamma_ratio"]
            imma = pi[vec]["IMMA16832"]
            c["eps"][reading] = eps
            c["gamma"][reading] = {"int8": g, "FP8": g * imma / pi[vec]["QMMA16832"],
                                   "NVFP4": g * 2 * imma / pi[vec]["OMMASF16864"],
                                   "NVFP4 (JI3s_20)": g * factors["NVFP4 (x JI3s_20 alone)"]}
            gg = c["gamma"][reading]
            lines.append(f"| {c['case']} | {reading} | {100 * c['skip'][reading]['skip']:.3f}% "
                         f"({c['skip'][reading]['at']}) | **{100 * eps:.3f}%** | {g:.3f} | {gg['FP8']:.3f} | "
                         f"{gg['NVFP4']:.3f} | {gg['NVFP4 (JI3s_20)']:.3f} |")
    lines += ["", f"The record ({DESIGN}): " + "; ".join(
        f"{r} ε {100 * j['core']:.3f}% (skip {100 * j['terms']['skip']:.3f}%, machine {100 * j['terms']['machine']:.3f}%, "
        f"luck {100 * j['terms']['luck']:.3f}%), γ {j['gamma']:.3f}" for r, j in record.items()),
              "JI3s_20 alone (VF3r-16): " + ", ".join(f"{k} {v:.4f}" for k, v in sorted(alone.items())),
              f"Forms on each vector: {json.dumps(pi)}"]
    (out / "g1h_eps.md").write_text("\n".join(lines) + "\n")
    (out / "g1h_eps.json").write_text(json.dumps({"design": DESIGN, "readings": readings, "record": {
        r: {"core": j["core"], "terms": j["terms"], "gamma": j["gamma"]} for r, j in record.items()},
        "cases": cases, "factors": factors, "forms": pi}, indent=1) + "\n")
    print("\n".join(lines))
    return rc


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1])))
