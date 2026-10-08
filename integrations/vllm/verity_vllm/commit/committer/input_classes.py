"""The Program inputs the committer's classes cover, named by binding and leaf width (`verity.ml.circuits.program.input_prim`)."""
from __future__ import annotations

import re

from verity.ml.circuits.program import RUN, input_prim

TOKEN_IDS_INPUT = input_prim(32, RUN).id

#: the binding time of the Program's `weights` parameter: per run until the weights are registered inputs
WEIGHTS_BINDING = RUN


def weights_inputs(weights_tree: dict | None, binding: str = WEIGHTS_BINDING) -> list[str]:
    """The input ids the registered weights are leaves of: one per leaf width of the registered tensors' dtypes."""
    names = (weights_tree or {}).get("names") or []
    widths = {int(m[0]) for _n, dt, _s, _b in names if (m := re.search(r"\d+", str(dt)))} or {16}
    return sorted(input_prim(w, binding).id for w in widths)


def weights_class(weights: dict, weights_tree: dict | None) -> dict:
    return {"class": "weights", "how": "registered-input", "covers_families": weights_inputs(weights_tree), "acquisitions": ["registered-input"],
            "rule": "register_weights(): per-tensor chunked pos_leaf subtree, fold of subtree roots bound with names/dtypes/shapes (once per process)",
            "count": None, "bytes": weights["bytes"], "tensors": weights["tensors"], "evidence": f"root {weights['root'][:16]}"}
