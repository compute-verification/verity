"""rs.sh's reading: each loop's wgmma rate over the SS form's alone rate (the tensor fraction, as `../dc/analyze.py` reads
`wgli_*`), and its cycles a MMA at the plain GEMM's 122.41 SMSP-cycles over that fraction (dc90-licpw-redux-l32's
measured 204.25 is wgli_redux's). RT-HOPPER-1 is a real escape only if the RS (or U8S8) form beside the same LI fold runs
more than 2% faster than SS does; otherwise it is a model gap (the joint rows' default), not a hardware one.

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
         "lop3": ("wgmix_lop3_r256", "wgmix_lop3_r256_rs")}


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
    for n, v in res["verdict"].items():
        print(f"VERDICT {n}: {v} (beside the honest LI loop, against wgli_redux)")
    if a.json:
        a.json.write_text(json.dumps(res, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
