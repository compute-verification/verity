"""The shape a derived request-wrapper directory was built for, from its construction manifest or derive report."""

from __future__ import annotations

import json
import os

from verity_vllm.program import prefix_plan as PP

__all__ = ["derived_shape"]


def derived_shape(d: str) -> tuple[int, ...] | None:
    """(LP, T) of a derived request-wrapper dir, (LP, T, C) for a cached prefix of C rows (`program.prefix_plan.shape`): derive_step's
    construction_manifest.json `structural_inputs`."""
    for fn in ("construction_manifest.json", "artifact.json"):
        p = os.path.join(d, fn)
        if os.path.exists(p):
            with open(p) as f:
                si = json.load(f).get("structural_inputs")   # the shape the directory was built for; a manifest without it states none
            if si and si.get("LP") is not None and si.get("T") is not None:
                return PP.shape(int(si["LP"]), int(si["T"]), int(si.get("C") or 0))
    rep = os.path.join(d, "derive-report.json")
    if os.path.exists(rep):
        with open(rep) as f:
            r = json.load(f)
        prov = r.get("provenance")   # the report states the shape at its top or under its provenance
        for k in ("structural_inputs", "shape"):
            si = r.get(k) or (prov.get(k) if prov else None)
            if si and si.get("LP") is not None and si.get("T") is not None:
                return int(si["LP"]), int(si["T"])
    return None
