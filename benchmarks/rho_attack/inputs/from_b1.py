"""rho_attack inputs from B1's measured F and beta (schema tvi-b1/F_beta/v2): one S label per A2 adapter size on
Qwen2.5-1.5B-Instruct, with F (forward crossing), G (backward, so F + G is the cheapest boundary's Ffb_min) and the
betas each the median over A2's conditions, seeds and steps at that size.

    python3 from_b1.py F_beta.json --out measured_b1.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

COLS = ("bv", "bv_gx", "bv_rm", "bx", "bw", "bw_min", "Ff_min", "Ff_med", "Fb_min", "Ffb_min")
PRECISIONS = {"fp8": ("fp8t", 8.0), "nvfp4": ("nvfp4", 4.5)}  # name: (B1's key, raw bits per coordinate)
LABELS = {272: "A2-34B", 9920: "A2-1.2kB", 78400: "A2-9.8kB", 802816: "A2-100kB", 295436288: "lora16",
          24699428864: "full"}
EXPLICIT = ("A2-34B", "A2-1.2kB", "A2-9.8kB", "A2-100kB")  # sizes that don't scale with the model


def convert(doc: dict) -> dict:
    if doc.get("schema") != "tvi-b1/F_beta/v2":
        raise ValueError(f"expected schema tvi-b1/F_beta/v2, got {doc.get('schema')!r}")
    a2 = [e for e in doc["entries"] if e["fam"] == "A2"]
    med = lambda xs: float(statistics.median(xs))  # noqa: E731

    def betas(entries: list[dict], key: str, raw: float) -> dict:
        col = lambda name: [e[key][COLS.index(name)] for e in entries]  # noqa: E731
        return {"beta_v": med(col("bv")), "beta_gx": med(col("bv_gx")), "beta_x": med(col("bx")),
                "beta_w": med(col("bw")), "raw_bits": raw}

    out = {"source": "B1 tvi-b1/F_beta/v2, A2 on Qwen2.5-1.5B-Instruct: medians over conditions, seeds and steps",
           "measured_on": {"d_model": a2[0]["d"], "model": a2[0]["model"]}, "F_scaling": "per_dim",
           "precisions": {p: betas(a2, key, raw) for p, (key, raw) in PRECISIONS.items()},
           "F": {}, "G": {}, "S_bits": {}, "beta_by_S": {}}
    for s_bits, label in LABELS.items():
        at = [e for e in a2 if e["S_bits"] == s_bits]
        if not at:
            continue
        col = lambda key, name: [e[key][COLS.index(name)] for e in at]  # noqa: E731
        out["F"][label] = {p: med(col(key, "Ff_min")) for p, (key, _) in PRECISIONS.items()}
        out["G"][label] = {p: med([fb - ff for fb, ff in zip(col(key, "Ffb_min"), col(key, "Ff_min"))])
                           for p, (key, _) in PRECISIONS.items()}
        out["beta_by_S"][label] = {p: betas(at, key, raw) for p, (key, raw) in PRECISIONS.items()}
        if label in EXPLICIT:
            out["S_bits"][label] = s_bits
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("f_beta")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    Path(args.out).write_text(json.dumps(convert(json.loads(Path(args.f_beta).read_text())), indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
