"""Store declaration for the repository's check (research.store README §6.1; `tools/check/check.py`).

    research run --tool check --project verity --cwd source -- uv run --locked --extra torch-cpu python tools/check/check.py

No key params and no closure: the attempt's source (a clean commit) is the whole input, and `research merge` reads the attempt by
that commit.  A merge that changes backends/flock/, other than a README.md or PROTOCOL.md (which nothing the agreement runs
reads), needs an attempt whose `lean-agreement` step passed (`merge_requires`; the step runs when the run has the pinned
upstream build, which `check.py --record` sends then).  `research run --on` refuses a machine that fails `preflight.py` (the pod
setup, the Lean dependencies, a store read, AVX-512 when needed, free disk) before the run is claimed.  Nothing here imports
anything but `research.store.tool`.
"""
from __future__ import annotations

from research.store.tool import Tool

CHECK = Tool(name="check", version="1", command=["uv", "run", "--locked", "--extra", "torch-cpu", "python", "tools/check/check.py"],
             produces=["check/v1"],
             merge_requires={"lean-agreement": ["backends/flock/", "!README.md", "!PROTOCOL.md"]},
             preflight=["tools/check/preflight.py"])
