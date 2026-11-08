"""Store declaration for the repository's check (research.store README §6.1; `tools/verity/check/check.py`).

    research run --tool check --project verity --cwd clone -- uv run --locked --all-packages --extra torch-cpu python tools/verity/check/check.py

No key params and no closure: the attempt's source (a clean commit) is the whole input, and `research merge` reads the attempt by
that commit.  A merge that changes verity/ml/flock/ or the Lean verifier's package verity/core/service/flock/, other than a
README.md or PROTOCOL.md (which nothing the agreement runs reads), needs an attempt whose `lean-agreement` step passed
(`merge_requires`; the step runs when the run has the pinned upstream build, which `check.py --record` sends then).  A merge
that changes V*'s stager (`rec_vstage`, `rec_outer`, `rec_open`, `rec_residuals`, `rec_cap`, `circuit`, `one_stage.registered`)
or its Lean mirror (`VStar.In`, `core_vstar`) needs `lean-suites` passed, whose vstar suite holds one to the other.
`research run --on` refuses a machine that fails `preflight.py` (the pod setup, the Lean dependencies, a store read, AVX-512
when needed, free disk) before the run is claimed.  Once `research merge --push`
has pushed a landing, it starts `post_train.py` on it, detached (`after_land`): the vLLM reference regression's manifest rebuilds,
which check leaves out, and their label on the attempt.  Nothing here imports anything but `research.store.tool`.
"""
from __future__ import annotations

from research.store.tool import Tool

CHECK = Tool(name="check", version="1", command=["uv", "run", "--locked", "--all-packages", "--extra", "torch-cpu", "python",
                                                     "tools/verity/check/check.py"],
             produces=["check/v1"],
             merge_requires={"lean-agreement": ["verity/ml/flock/", "verity/core/service/flock/", "!README.md", "!PROTOCOL.md"],
                             "lean-suites": ["verity/ml/flock/python/verity_flock/rec_vstage.py",
                                             "verity/ml/flock/python/verity_flock/rec_outer.py",
                                             "verity/ml/flock/python/verity_flock/rec_open.py",
                                             "verity/ml/flock/python/verity_flock/rec_residuals.py",
                                             "verity/ml/flock/python/verity_flock/rec_cap.py",
                                             "verity/ml/flock/python/verity_flock/circuit.py",
                                             "verity/core/protocols/sampled_proofs/one_stage/registered.py",
                                             "verity/core/service/flock/VStar/In.lean", "verity/core/service/vstar/"]},
             preflight=["tools/verity/check/preflight.py"],
             after_land=["uv", "run", "-q", "python", "tools/verity/check/post_train.py"])
