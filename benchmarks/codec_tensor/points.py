"""`verity/codec-frontier-point/v0` points (ra-codec-wide's schema, `benchmarks/word_codec/frontier.py`) from this
lane's runs: a GEMM run's `codec-tensor.json` (layer-weighted y per field, batch and tag) and a rates run's (x and Δ
at k = 1, at the locked clock, so x_boosted is false and the renderer grants the clock).

    python points.py --gemm RUN_DIR [--gemm RUN_DIR ...] --rates RUN_DIR [--art RUN=art:...] --out points.json

y_base is the plain kernel's layer ratio, y_decode the untagged fused kernel's over it, y_tag the tagged (or sketched)
fused kernel's over the untagged one, and y_commit the step's leaves and tree after the last linear. τ is the tag's
compute per stored pass: ro-blake3/v1's measured `b3` rate (0.48), and 0 for the sketch (its MMAs share the decode's
CTA; the renderer's convention for sketch points).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

SCHEMA = "verity/codec-frontier-point/v0"
LANE = "ra-codec-tensor"
MODEL = {"70b": "llama-3.1-70b", "8b": "llama-3.1-8b"}
TAU_B3 = 0.48
TAGS = {"fused_k1_tagged": ("ro-blake3/v1", "commit_step", TAU_B3),
        "fused_k1_sketch16": ("ro-sketch-blake3/v0 k_s=16", "sketch_commit_step", 0.0),
        "fused_k1_sketch64": ("ro-sketch-blake3/v0 k_s=64", "sketch_commit_step", 0.0)}


def points(gemm: dict, run: str, art: str | None, rates: dict, sm_mhz: float) -> list[dict]:
    out = []
    for name, e in gemm["libs"].items():
        if "layer" not in e or name not in rates["libs"] or "analysis" not in rates["libs"][name]:
            continue
        an = rates["libs"][name]["analysis"]["k"]["1"]
        c = e["constants"]
        for mb, row in e["layer"].items():
            for arm, (tag, commit, tau) in TAGS.items():
                if arm not in row or commit not in e:
                    continue
                y_commit = e[commit]["y_commit"][mb]
                p = {"schema": SCHEMA, "lane": LANE, "model": MODEL[gemm["model"]["name"]], "b": c["n"], "k": 1,
                     "arithmetic": f"cuda-int32/wt.cu/{name}/role-split" + (f"/{tag}" if tag else ""),
                     "batch": int(mb), "y_base": row["plain"], "y_decode": row["fused_k1"] - row["plain"],
                     "y_tag": row[arm] - row["fused_k1"], "y_commit": y_commit, "y_total": row[arm] + y_commit,
                     "x": an["x_cuda"], "x_boosted": False, "delta": an["delta_cuda"], "tau": tau, "measured": True,
                     "label": f"{name} k=1 {tag}, M={mb}", "run": run, "gpu": gemm["device"]["name"],
                     "sm_mhz": sm_mhz, "payload_per_stored_bit": c["PB"] / c["S"]}
                if art:
                    p["art"] = art
                out.append(p)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gemm", action="append", required=True, help="a GEMM run's directory (holding out/)")
    ap.add_argument("--rates", required=True, help="a rates run's directory")
    ap.add_argument("--art", action="append", default=[], help="RUN=art:... (the run's run_record)")
    ap.add_argument("--sm-mhz", type=float, default=2092.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    arts = dict(v.split("=", 1) for v in a.art)
    rates = json.loads((Path(a.rates) / "out" / "codec-tensor.json").read_text())
    pts = []
    for d in a.gemm:
        run = Path(d).name
        pts += points(json.loads((Path(d) / "out" / "codec-tensor.json").read_text()), run, arts.get(run), rates,
                      a.sm_mhz)
    Path(a.out).write_text(json.dumps(pts, indent=1))
    print(f"{len(pts)} points -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
