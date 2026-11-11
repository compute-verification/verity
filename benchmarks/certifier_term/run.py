"""`verity bench certifier_term [ARGS]`: the network certifier's term for the served model at the load of a node, the
census's reference datacenter and the world, over a week, a month and a year, in bits and GB (the README beside this
file). It runs the calculator of K, `benchmarks/network_traces/leak_bounds.py`, as its own command with ARGS."""
from __future__ import annotations

import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parents[1] / "network_traces" / "leak_bounds.py"), run_name="__main__")
