"""Store declaration for the repository's check (research.store README §6.1; `tools/verity/check/check.py`).

    research run --tool check --project verity --cwd source -- uv run --locked --extra torch-cpu python tools/verity/check/check.py

No key params and no closure: the attempt's source (a clean commit) is the whole input, and `research merge` reads the attempt by
that commit.  A merge that changes verity/ml/flock/, other than a README.md or PROTOCOL.md (which nothing the agreement runs
reads), needs an attempt whose `lean-agreement` step passed (`merge_requires`; the step runs when the run has the pinned
upstream build, which `check.py --record` sends then).  `research run --on` refuses a machine that fails `preflight.py` (the pod
setup, the Lean dependencies, a store read, AVX-512 when needed, free disk) before the run is claimed.  Once `research merge --push`
has pushed a landing, it starts `landed.py` on it, detached (`after_land`): `post_train.py`, the vLLM reference regression's
manifest rebuilds, which check leaves out, and their label on the attempt; and `check.py --after-landing`, ci.toml's
`after_landing` jobs, which the check a merge requires skips, with a failure posted to the merge thread.  Nothing here imports anything but `research.store.tool`.
"""
from __future__ import annotations

from research.store.tool import Tool

CHECK = Tool(name="check", version="1", command=["uv", "run", "--locked", "--extra", "torch-cpu", "python", "tools/verity/check/check.py"],
             produces=["check/v1"],
             merge_requires={"lean-agreement": ["verity/ml/flock/", "verity/core/service/flock/", "!README.md", "!PROTOCOL.md"]},
             preflight=["tools/verity/check/preflight.py"],
             after_land=["uv", "run", "-q", "python", "tools/verity/check/landed.py"])
