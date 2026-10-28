"""Every analysis of aggregated PoUW words, each writing its JSON under OUT (default $OUT): the AND counts, the spectra,
the exact attacks on Dither's coins, the guessing, partial-computation and one-direction measurements, the 8-bit-cell spectra and
the cost model. `run_all.py OUT [step ...]` runs only the steps named."""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import agg_ands  # noqa: E402
import agg_attacks  # noqa: E402
import agg_cost  # noqa: E402
import agg_entropy  # noqa: E402
import agg_line  # noqa: E402
import agg_partial  # noqa: E402
import agg_partial2  # noqa: E402
import agg_spec  # noqa: E402
import agg_spec8  # noqa: E402

STEPS = (("ands", agg_ands.main), ("cost", agg_cost.main), ("attacks", agg_attacks.main), ("partial", agg_partial.main),
         ("partial2", agg_partial2.main), ("line", agg_line.main), ("spec", agg_spec.main), ("spec8", agg_spec8.main),
         ("entropy", agg_entropy.main))


def main() -> None:
    out = sys.argv[1] if len(sys.argv) > 1 else os.environ["OUT"]
    only = set(sys.argv[2:])
    os.makedirs(out, exist_ok=True)
    for name, fn in STEPS:
        if only and name not in only:
            continue
        t = time.monotonic()
        print(f"== {name}", flush=True)
        fn(os.path.join(out, f"{name}.json"))
        print(f"== {name} done in {time.monotonic() - t:.0f} s", flush=True)


if __name__ == "__main__":
    main()
