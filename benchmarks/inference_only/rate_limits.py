"""The rate rule's limits fixed from the inference-only policy alone (`inference-only-limits.json`): α = k/c and Γ = c of
the policy's own c and k, in the unit the policy counts c in. The one writer of those default limits. It loads only the
standard library, so a Python with no packages loads it by path, as a shell step's interpreter does."""
from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from typing import Any

FORMAT = "verity-bench/inference-only-limits/v1"


def of_policy(body: bytes) -> dict[str, Any]:
    """The limits of the policy whose bytes are ``body``; ValueError where its c and k give none."""
    try:
        p = json.loads(body)
        a, g = Fraction(p["k"], p["c"]), int(p["c"])
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as e:
        raise ValueError(f"the policy's c and k give no limits ({type(e).__name__}: {e})") from None
    if a < 0 or g < 1:
        raise ValueError(f"alpha {a} and gamma {g} are no limits")
    return {"format": FORMAT, "alpha": [a.numerator, a.denominator], "gamma": g,
            "unit": "Boolean gates by the policy's gate table, the unit of its c",
            "count": {x: p.get(x) for x in ("c", "k", "n")},
            "from": "the policy's c and k (alpha = k/c, gamma = c)",
            "policy_sha512": hashlib.sha512(body).hexdigest(), "written_by": "the developer"}
