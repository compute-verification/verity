"""rs.sh's reading: each loop's wgmma rate over the SS form's alone rate (the tensor fraction, as `../dc/analyze.py`
reads `wgli_*`), and its cycles a MMA at the plain GEMM's 122.41 SMSP-cycles over that fraction (dc90-licpw-redux-l32's
measured 204.25 is wgli_redux's). Offering another form is a real escape only if the RS (or U8S8) form beside the same
LI fold runs more than 2% faster than SS does; otherwise it is a model gap (the joint rows' default), not a hardware
one. The `wgli_true` loops fold the wgmma's own accumulators after each wait: their cycles a MMA against wgli_redux's
say how far the stand-in loop behind honest's measured time, and so γ, is from the receipt's own: unordered (wgli_true),
ping-pong on named barriers (wgli_pp, wgli_pp2) and double-buffered on two N128 tiles (wgli_db). A N128 wgmma counts as
half a N256 one throughout. The b1 form (BGMMA m64n256k256) and sm90_forms.cu's s8-exact siblings (IGMMA N128 and N64,
IMMA.16832) have the same loops as RS. The facet check prices every loop on DC-90's joint rows, each tensor instruction
at IGMMA's line by its MACs, and `fit` gives each form's joint coefficients over IGMMA's, the least over its matched
pairs, which `forms_gap.py` offers the cheater.

    python rs.py OUT [--json PATH]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PLAIN = 122.41   # the plain GEMM's SMSP-cycles a MMA on DC-90 (dc90-licpw-redux-l32's source)
MARGIN = 0.02
# each form's loop is its SS loop's name with this suffix
FORMS = ("rs", "u8", "b1", "n128", "n64", "im", "n192", "n32", "n16", "n8", "im16", "bm",
         "f16", "bf", "tf", "e4", "bf128", "e4128", "e5", "hf16", "hbf", "htf")
PAIRS = {"li": ("wgli_redux", *(f"wgli_redux_{f}" for f in FORMS)),
         "imad": ("wgmix_imad_r256", *(f"wgmix_imad_r256_{f}" for f in FORMS)),
         "lop3": ("wgmix_lop3_r256", *(f"wgmix_lop3_r256_{f}" for f in FORMS)),
         "true": ("wgli_true_w2", "wgli_true_w3", "wgli_pp_w2", "wgli_pp_w3", "wgli_pp2_w2", "wgli_pp2_w3",
                  "wgli_db_w1", "wgli_db_w2")}
# sm_90's census, DC-90: IGMMA_S8_N256's line and the line every other warp instruction carries on each joint row
JOINT = {"JI90_0": (62.7407638, 0.98054536), "JI90_1": (66.93849, 0.919487525), "JI90_2": (121.566, 0.0822829491)}
LANE = ("LOP3", "IMAD", "REDUX")   # the lane instructions a facet check counts; control and fences count nothing
GMMA = ("IGMMA", "BGMMA", "IMMA.16832", "IMMA.16816", "BMMA.168256", "HGMMA", "QGMMA", "HMMA.16816",
        "HMMA.1688")   # BGMMA.64x256x256 has IGMMA_S8_N256's tensor line
ALONE = ("wg_s8_n256_a1", "wg_s8_n256_rs_a1", "wg_u8s8_n256_a1", "wg_b1_n256_a1", "wg_s8_n128_a1", "wg_s8_n64_a1",
         "imma_s8_a1", "wg_s8_n192_a1", "wg_s8_n32_a1", "wg_s8_n16_a1", "wg_s8_n8_a1", "imma16816_a1", "bmma_a1",
         "wg_f16_n256_a1", "wg_bf16_n256_a1", "wg_tf32_n256_a1", "wg_e4m3_n256_a1", "wg_bf16_n128_a1",
         "wg_e4m3_n128_a1", "wg_e5m2_n256_a1", "hmma_f16_a1", "hmma_bf16_a1", "hmma_tf32_a1")


def n256(op: str) -> float:
    """A tensor opcode's N256-equivalents: IGMMA.64x128x32 is half an IGMMA.64x256x32's MACs and words, a warp's
    IMMA.16832 (m16n8k32) a 32nd of its share and IMMA.16816 a 64th, and BMMA.168256 a 32nd of a BGMMA.64x256x256's.
    The FP forms count their MACs too: HGMMA.64x256x16 (f16, bf16) is half, HGMMA.64x256x8 (tf32) a quarter,
    QGMMA.64x256x32 (e4m3, e5m2) one, and a warp's HMMA.16816 a 64th and HMMA.1688 (tf32) a 128th."""
    if op.startswith(("IMMA.16832", "BMMA.168256")):
        return 1 / 32
    if op.startswith(("IMMA.16816", "HMMA.16816")):
        return 1 / 64
    if op.startswith("HMMA.1688"):
        return 1 / 128
    _, n, k = op.split(".")[1].split("x")
    return int(n) / 256 * (int(k) / 32 if op.startswith(("HGMMA", "QGMMA")) else 1)


def facets(out: Path, rows: dict) -> dict:
    """Each joint row's lower bound on a timed loop, SMSP-cycles a N256 wgmma, over the loop's measured cycles: above
    1, the row isn't a lower bound there. Counts only the lane instructions LANE names, so a ratio is at most the
    row's; a narrower wgmma counts as its N256-equivalents, at IGMMA_S8_N256's line."""
    res = {}
    gate = out / "gate-build.json" if (out / "gate-build.json").is_file() else out / "probes" / "gate.json"
    for v in json.loads(gate.read_text())["verdicts"]:
        p, loops = v["probe"], v["verdict"].get("loops") or []
        if p not in rows or len(loops) != 1:
            continue
        top = loops[0]["top"]
        n_mma = sum(n * n256(k) for k, n in top.items() if k.startswith(GMMA))
        n_lane = sum(n for k, n in top.items() if k.split(".")[0] in LANE)
        cycles = 4 / gmma(rows[p])
        res[p] = {"lane_a_mma": n_lane / n_mma, "cycles": cycles,
                  **{r: (u + b * n_lane / n_mma) / cycles for r, (u, b) in JOINT.items()}}
    return res


def fit(fac: dict) -> dict:
    """Each form's joint coefficient over IGMMA_S8_N256's on each row, a N256-equivalent: the least over its matched
    pairs (alone, LI, IMAD, LOP3) of the tensor share the row leaves at the form's loop over the share it leaves at
    SS's, (cycles − b · lane) / (cycles_SS − b · lane_SS). At 1 the form takes IGMMA's lines by MACs."""
    res = {}
    for form, alone in zip(FORMS, ALONE[1:]):
        pairs = {"alone": (ALONE[0], alone),
                 **{fam: (ss, f"{ss}_{form}") for fam, ss in (("li", "wgli_redux"), ("imad", "wgmix_imad_r256"),
                                                              ("lop3", "wgmix_lop3_r256"))}}
        pairs = {fam: p for fam, p in pairs.items() if p[0] in fac and p[1] in fac}
        if pairs:
            res[form] = {r: min(((fac[f]["cycles"] - b * fac[f]["lane_a_mma"]) / (fac[s]["cycles"] - b * fac[s]["lane_a_mma"]),
                                 fam) for fam, (s, f) in pairs.items())
                         for r, (_, b) in JOINT.items()}
    return res


def gmma(row: dict) -> float:
    """The loop's wgmma rate in N256-equivalents a SM-clock."""
    return next(r * n256(k.split(":", 1)[1]) for k, r in row["rates"].items() if k.split(":", 1)[1].startswith(GMMA))


def read(out: Path) -> dict:
    rows = {}
    for ln in (out / "probes" / "rows.jsonl").read_text().splitlines():
        r = json.loads(ln)
        rows[r["probe"]] = r
    alone = gmma(rows["wg_s8_n256_a1"])
    res = {"alone": {n: {"rate": gmma(rows[n]), "of_ss": gmma(rows[n]) / alone, "sm_mhz": rows[n].get("sm_mhz"),
                         "tags": rows[n].get("tags")}
                     for n in ALONE if n in rows}}
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
    for form in FORMS[2:]:
        ratio = {fam: res[fam][f"{ss_n}_{form}"]["cycles_a_mma"] / res[fam][ss_n]["cycles_a_mma"]
                 for fam, ss_n in (("li", "wgli_redux"), ("imad", "wgmix_imad_r256"), ("lop3", "wgmix_lop3_r256"))
                 if f"{ss_n}_{form}" in res[fam] and ss_n in res[fam]}
        if ratio:
            res["verdict"][f"{form} over SS"] = ", ".join(f"{fam} {r:.4f}" for fam, r in ratio.items()) + (
                " (cycles a MMA: at 1 the form takes IGMMA's joint lines by MACs, not the lane default)")
    best = min((v["cycles_a_mma"] for v in res["true"].values()), default=None)
    if best and ss:
        res["verdict"]["honest"] = (f"the receipt's own loop runs at {best:.2f} SMSP-cycles a MMA, "
                                    f"{best / ss - 1:+.1%} on wgli_redux's {ss:.2f}")
    if (out / "gate-build.json").is_file() or (out / "probes" / "gate.json").is_file():
        res["facets"] = facets(out, rows)
        over = sorted(f"{p} {r} {f[r]:.4f}" for p, f in res["facets"].items() for r in JOINT if f[r] > 1)
        res["verdict"]["joint rows"] = ("not a lower bound at " + ", ".join(over)) if over else "a lower bound at every loop"
        res["fit"] = fit(res["facets"])
        for form, rows_fit in res["fit"].items():
            res["verdict"][f"{form} fit"] = "  ".join(f"{r} {v:.4f} ({fam})" for r, (v, fam) in rows_fit.items())
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
