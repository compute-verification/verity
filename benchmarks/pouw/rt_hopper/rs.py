"""rs.sh's reading: each loop's wgmma rate over the SS form's alone rate (the tensor fraction, as `../dc/analyze.py` reads
`wgli_*`), and its cycles a MMA at the plain GEMM's 122.41 SMSP-cycles over that fraction (dc90-licpw-redux-l32's
measured 204.25 is wgli_redux's). RT-HOPPER-1 is a real escape only if the RS (or U8S8) form beside the same LI fold runs
more than 2% faster than SS does; otherwise it is a model gap (the joint rows' default), not a hardware one. The
`wgli_true` loops fold the wgmma's own accumulators after each wait (RT-HOPPER-8): their cycles a MMA against
wgli_redux's say how far the stand-in loop behind honest's measured time, and so γ, is from the receipt's own.

    python rs.py OUT [--json PATH]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PLAIN = 122.41   # the plain GEMM's SMSP-cycles a MMA on DC-90 (dc90-licpw-redux-l32's source)
MARGIN = 0.02
PAIRS = {"li": ("wgli_redux", "wgli_redux_rs", "wgli_redux_u8"), "imad": ("wgmix_imad_r256", "wgmix_imad_r256_rs"),
         "lop3": ("wgmix_lop3_r256", "wgmix_lop3_r256_rs"),
         "true": ("wgli_true_w2", "wgli_true_w3", "wgli_pp_w2", "wgli_pp_w3")}
# sm_90's census, DC-90: IGMMA_S8_N256's line and the line every other warp instruction carries on each joint row
JOINT = {"JI90_0": (62.7407638, 0.98054536), "JI90_1": (66.93849, 0.919487525), "JI90_2": (121.566, 0.0822829491)}
LANE = ("LOP3", "IMAD", "REDUX")   # the lane instructions a facet check counts; control and fences count nothing


def facets(out: Path, rows: dict) -> dict:
    """Each joint row's lower bound on a timed loop, SMSP-cycles a wgmma, over the loop's measured cycles: above 1, the
    row isn't a lower bound there. Counts only the lane instructions LANE names, so a ratio is at most the row's."""
    res = {}
    for v in json.loads((out / "gate-build.json").read_text())["verdicts"]:
        p, loops = v["probe"], v["verdict"].get("loops") or []
        if p not in rows or len(loops) != 1:
            continue
        top = loops[0]["top"]
        n_mma = sum(n for k, n in top.items() if k.startswith("IGMMA"))
        n_lane = sum(n for k, n in top.items() if k.split(".")[0] in LANE)
        cycles = 4 / gmma(rows[p])
        res[p] = {"lane_a_mma": n_lane / n_mma, "cycles": cycles,
                  **{r: (u + b * n_lane / n_mma) / cycles for r, (u, b) in JOINT.items()}}
    return res


def gmma(row: dict) -> float:
    return next(r for k, r in row["rates"].items() if k.split(":", 1)[1].startswith("IGMMA"))


def read(out: Path) -> dict:
    rows = {}
    for ln in (out / "probes" / "rows.jsonl").read_text().splitlines():
        r = json.loads(ln)
        rows[r["probe"]] = r
    alone = gmma(rows["wg_s8_n256_a1"])
    res = {"alone": {n: {"rate": gmma(rows[n]), "of_ss": gmma(rows[n]) / alone, "sm_mhz": rows[n].get("sm_mhz"),
                         "tags": rows[n].get("tags")}
                     for n in ("wg_s8_n256_a1", "wg_s8_n256_rs_a1", "wg_u8s8_n256_a1") if n in rows}}
    for fam, names in PAIRS.items():
        res[fam] = {}
        for n in names:
            if n in rows:
                f = gmma(rows[n]) / alone
                res[fam][n] = {"tensor_fraction": f, "cycles_a_mma": PLAIN / f, "sm_mhz": rows[n].get("sm_mhz"),
                               "tags": rows[n].get("tags")}
    li = res["li"]
    ss = li.get("wgli_redux", {}).get("cycles_a_mma")
    res["verdict"] = {n: ("escape" if v["cycles_a_mma"] < ss * (1 - MARGIN) else "no escape")
                      for n, v in li.items() if n != "wgli_redux" and ss}
    best = min((v["cycles_a_mma"] for v in res["true"].values()), default=None)
    if best and ss:
        res["verdict"]["honest"] = (f"the receipt's own loop runs at {best:.2f} SMSP-cycles a MMA, "
                                    f"{best / ss - 1:+.1%} on wgli_redux's {ss:.2f}")
    if (out / "gate-build.json").is_file():
        res["facets"] = facets(out, rows)
        over = sorted(f"{p} {r} {f[r]:.4f}" for p, f in res["facets"].items() for r in JOINT if f[r] > 1)
        res["verdict"]["joint rows"] = ("not a lower bound at " + ", ".join(over)) if over else "a lower bound at every loop"
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out", type=Path)
    ap.add_argument("--json", type=Path)
    a = ap.parse_args(argv)
    res = read(a.out)
    for n, v in res["alone"].items():
        print(f"{n:<22} alone {v['rate']:.5f} a SM-clock, {v['of_ss']:.4f} of SS")
    for fam in PAIRS:
        for n, v in res[fam].items():
            print(f"{n:<22} tensor fraction {v['tensor_fraction']:.4f}  {v['cycles_a_mma']:7.2f} SMSP-cycles a MMA")
    for n, f in res.get("facets", {}).items():
        print(f"{n:<22} {f['lane_a_mma']:6.2f} lane a MMA, {f['cycles']:7.2f} SMSP-cycles a wgmma; the joint rows' bound "
              "over it: " + "  ".join(f"{r} {f[r]:.4f}" for r in JOINT))
    for n, v in res["verdict"].items():
        print(f"VERDICT {n}: {v}")
    if a.json:
        a.json.write_text(json.dumps(res, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
