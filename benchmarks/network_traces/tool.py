"""Store declarations for the network traces' harness (research.store README §6.1), each the ``--tool`` of a job:

    benchmarks.network_traces.tool:COVERT_SYNC, for
      bash benchmarks/network_traces/covert.sh last cover-3-2 envelope
    benchmarks.network_traces.tool:COVERT_STEADY, for
      python benchmarks/network_traces/covert.py --calibration C --out $D/steady.json --steady last --windows 3000

Key params: the sync policies (and the windows, for the steady state). The traces and the calibration are inputs.
Imports nothing but ``research.store.tool``.
"""

from __future__ import annotations

import argparse

from research.store.tool import Tool


def parse_covert(argv: list[str]) -> dict:
    return {"policies": " ".join(a for a in argv[2:] if not a.startswith("-"))}


def parse_steady(argv: list[str]) -> dict:
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    ap.add_argument("--steady", action="append", default=[])
    ap.add_argument("--windows", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ns, _ = ap.parse_known_args(argv)
    return {"policies": " ".join(ns.steady), "windows": ns.windows, "seed": ns.seed}


COVERT_SYNC = Tool(name="network_covert_sync", version="1", command=["bash", "benchmarks/network_traces/covert.sh"],
                   closure=["benchmarks/network_traces/**", "verity/core/protocols/certifier/**"],
                   key_params=["policies"], produces=["run-files/v1"], parse=parse_covert)
COVERT_STEADY = Tool(name="network_covert_steady", version="1", command=["python", "benchmarks/network_traces/covert.py"],
                     closure=["benchmarks/network_traces/covert.py", "verity/core/protocols/certifier/**"],
                     key_params=["policies", "windows", "seed"], produces=["run-files/v1"], parse=parse_steady)
