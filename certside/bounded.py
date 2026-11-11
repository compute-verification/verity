#!/usr/bin/env python3
"""`warden_serve.py` with its id requests' prompt ids below a bound, for a model whose vocabulary is smaller than
the 128,256 `id_traffic` draws below:

    python certside/bounded.py N rehearse ...      # as `python benchmarks/network_traces/warden_serve.py rehearse ...`

Each prompt id x of `id_traffic` becomes ⌊x·N / 128,256⌋, so the ids stay uniform, now below N, and N = 128,256 changes
none. The arrivals, the prompts' lengths, `max_tokens`, the sampling and the seeds are the seed's as before. Only the
process that writes the traffic reads this: `rehearse` runs `serve`, `client` and `check` from `warden_serve.py` itself.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks" / "network_traces"))

import warden_serve  # noqa: E402

DRAWN_BELOW = 128_256


def main(argv: list[str]) -> int:
    if not argv or not argv[0].isdigit() or not 1 <= int(argv[0]) <= DRAWN_BELOW:
        raise SystemExit(f"bounded.py N ...: N from 1 to {DRAWN_BELOW}, the bound of the prompt ids")
    bound, drawn = int(argv[0]), warden_serve.id_traffic

    def id_traffic(*args, **kwargs) -> list[dict]:
        traffic = drawn(*args, **kwargs)
        for item in traffic:
            item["ids"]["prompt"] = [x * bound // DRAWN_BELOW for x in item["ids"]["prompt"]]
        return traffic

    warden_serve.id_traffic = id_traffic
    return warden_serve.main(argv[1:])


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
