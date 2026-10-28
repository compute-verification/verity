"""Where the card's capacity goes in a HIGHUTIL run: the share of θ x wall (θ at the card's SM-clock ceiling) that the
credited MACs fill, and each term of the rest, from period.py's summary.json and the protocol arm's kernel timeline
(timeline.py over the same settings). The terms sum to 1:

- certified: the credited MACs at θ;
- clock: the run's SM clock below the card's ceiling (the node's lock and the power cap);
- between blocks: the protocol wall the GPU spends outside its blocks (coin round trips, the host between blocks);
- idle in a block: the protocol block's GPU time with no kernel running;
- off the GEMM: the protocol block's busy time outside a GEMM kernel (the coins and form kernels before each GEMM, and
  the hashing where nothing overlaps it), with the tensor pipe idle;
- GEMM beside the hashing: the protocol GEMMs' time past the plain arm's;
- GEMM slack: the plain arm's GEMM time past the MACs at θ at the run's clock.

ε = ε_skip + ε_undecl, with ε_undecl the share no credited MAC fills (every term but `certified`).

    terms.py SUMMARY TIMELINE      prints one JSON object
"""
from __future__ import annotations

import json
import sys


def terms(summary: dict, timeline: dict) -> dict:
    card = summary["card"]
    f = summary["sm_mhz_median"] / card["clock_mhz"]
    wall, gpu, plain = summary["proto_s_per_step"], summary["proto_gpu_s_per_step"], summary["plain_gpu_s_per_step"]
    ideal = summary["workload"]["step_macs"] / summary["peak_macs_s"]["run_clock"]
    gemm = timeline["gemm_share"] * gpu
    busy = timeline["busy_share"] * gpu
    t = {"certified": ideal / wall * f, "clock": 1 - f, "between blocks": (wall - gpu) / wall * f,
         "idle in a block": (gpu - busy) / wall * f, "off the GEMM": (busy - gemm) / wall * f,
         "GEMM beside the hashing": (gemm - plain) / wall * f, "GEMM slack": (plain - ideal) / wall * f}
    undecl = 1 - t["certified"]
    return {"card": card["sku"], "clock_mhz": card["clock_mhz"], "run_mhz": summary["sm_mhz_median"], "terms": t,
            "sum": sum(t.values()), "eps_skip": summary["eps_skip"], "eps_undecl": undecl,
            "eps": summary["eps_skip"] + undecl, "delta": summary["delta"],
            "ms_per_step": {"wall": wall * 1e3, "gpu": gpu * 1e3, "plain_gpu": plain * 1e3, "ideal_at_run_clock": ideal * 1e3}}


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2:
        print(__doc__.split("\n\n")[-1].strip(), file=sys.stderr)
        return 2
    s, tl = (json.loads(open(x).read()) for x in argv)
    print(json.dumps(terms(s, tl), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
