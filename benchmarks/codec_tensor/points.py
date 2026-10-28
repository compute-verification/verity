"""`verity/codec-tensor-rates/v0` rows from this lane's rates runs (`codec-tensor.json`): one row a field, the best
squarings/s on CUDA and tensor cores, decode and encode words/s at each k, and x, Δ and r at the locked clock (so
x_boosted is false).

    python points.py --rates RUN_DIR [--rates RUN_DIR ...] --out rates.json

Each field's rates come from the --rates run with its fastest k = 1 encode.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

RATES_SCHEMA = "verity/codec-tensor-rates/v0"
LANE = "ra-codec-tensor"


def merge_rates(runs: list[tuple[str, dict]]) -> dict:
    """The rates runs' libs that have an analysis, each with its run. A field measured more than once takes the run
    with the fastest k = 1 encode: x is the adversary's best, never a slower squaring's."""
    def encode1(e: dict) -> float:
        return next(v["per_s"] for k, v in e["rates"]["encode"].items() if int(k) == 1)

    libs = {}
    for run, d in runs:
        for name, e in d["libs"].items():
            if "analysis" in e and "rates" in e and (name not in libs or encode1(e) > encode1(libs[name])):
                libs[name] = dict(e, run=run, device=d["device"]["name"])
    return {"libs": libs}


def rates_rows(rates: dict, sm_mhz: float) -> list[dict]:
    def best(kind: dict) -> float | None:
        return max((v["per_s"] for v in kind.values() if "per_s" in v), default=None)

    out = []
    for name, e in rates["libs"].items():
        r, an = e["rates"], e["analysis"]
        out.append({"schema": RATES_SCHEMA, "lane": LANE, "field": name, "n": e["constants"]["n"],
                    "squarings_s_cuda": best(r["square_cuda"]), "squarings_s_tc": best(r["square_tc"]),
                    "decode_words_s": {k: v["per_s"] for k, v in r["decode"].items()},
                    "encode_words_s": {k: v["per_s"] for k, v in r["encode"].items()},
                    "read_gbps": an["read_gbps"], "words_per_plain_byte": an["words_per_plain_byte"],
                    "k": an["k"], "x_boosted": False, "sm_mhz": sm_mhz, "run": e["run"], "gpu": e["device"]})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rates", action="append", required=True, help="a rates run's directory (holding out/)")
    ap.add_argument("--sm-mhz", type=float, default=2092.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rates = merge_rates([(Path(d).name, json.loads((Path(d) / "out" / "codec-tensor.json").read_text()))
                         for d in a.rates])
    rows = rates_rows(rates, a.sm_mhz)
    Path(a.out).write_text(json.dumps(rows, indent=1))
    print(f"{len(rows)} fields -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
