"""[CAPCORR] The DERIVED program as a `python:` program source for `derive_corr.py` / `derive_prove.py`:

    --program python:capcorr_derived_program:program        (LP, STEPS taken from the record's pins)

`program(LP, STEPS)` runs lane DERIVE's importer (`verity_vllm.frontend.derive.derive_report`, S3 of GUIDANCE §43.5) on
`GreedyServe(B0_CONFIG, steps=STEPS)` — an unannotated serving-loop `nn.Module` over vLLM-style kernel-interface
classes — with a `(prompt int32[LP],)` example input in profile mode `b1-eager`, and returns the derived `Program`.
Nothing here names `Serve_v1` or any registered composite: the identity the program carries is whatever the
derivation + the family rule established (recorded in `Program.annotations["derive"]`), and `derive_corr.py`'s binding
stage compares its digest to the record's pin.  A derivation that refuses (`Unsupported`) raises with the reason.
"""
from __future__ import annotations

import time
from typing import Any

DERIVE_SOURCE = {"module": "verity_vllm.frontend.derive_examples", "class": "GreedyServe", "mode": "profile", "profile": "b1-eager",
                 "config": "verity_vllm.registry.b1.B0_CONFIG", "example_inputs": "(prompt: int32[LP],)"}


def derive_report_for(LP: int, STEPS: int, *, final_norm_all_rows: bool = False):
    import torch
    from verity_vllm.frontend.derive import derive_report
    from verity_vllm.frontend.derive_examples import GreedyServe
    from verity_vllm.registry.b1 import B0_CONFIG

    torch.manual_seed(0)
    with torch.device("meta"):
        module = GreedyServe(B0_CONFIG, steps=STEPS, final_norm_all_rows=final_norm_all_rows)
    prompt = torch.zeros(LP, dtype=torch.int32, device="meta")
    t0 = time.time()
    rep = derive_report(module, (prompt,), mode="profile", profile="b1-eager", config=B0_CONFIG, name="GreedyServe")
    rep.provenance["capcorr_wall_seconds"] = round(time.time() - t0, 2)
    return rep


def program(LP: int = 32, STEPS: int = 15) -> Any:
    rep = derive_report_for(LP, STEPS)
    if not rep.ok:
        u = rep.unsupported
        raise RuntimeError(f"derivation refused: {u.to_json() if u is not None else 'unknown'}")
    p = rep.program
    ann = dict(getattr(p, "annotations", {}) or {})
    ann["capcorr_derive_summary"] = {"digest": rep.digest, "root": rep.root, "gates": rep.gates, "flat_digest": rep.flat_digest, "flat_root": rep.flat_root,
                                     "family": {k: v for k, v in (rep.family or {}).items() if k in ("rule", "matched", "identity", "leaves", "family_id", "target")},
                                     "source": DERIVE_SOURCE, "wall_seconds": rep.provenance.get("capcorr_wall_seconds")}
    try:
        p.annotations = ann
    except (AttributeError, TypeError):
        pass
    return p


def program_all_rows(LP: int = 32, STEPS: int = 15) -> Any:
    """The variant that applies the final norm to every prefill row (vLLM's model forward): a DIFFERENT program
    (DERIVE's caveat 2) — used as a negative for the binding stage, never as the deliverable."""
    rep = derive_report_for(LP, STEPS, final_norm_all_rows=True)
    if not rep.ok:
        raise RuntimeError(f"derivation refused: {rep.unsupported.to_json() if rep.unsupported else 'unknown'}")
    return rep.program


if __name__ == "__main__":
    import json
    import sys
    sys.path[:0] = [".", "src"]
    rep = derive_report_for(int(sys.argv[1]) if len(sys.argv) > 1 else 32, int(sys.argv[2]) if len(sys.argv) > 2 else 15)
    print(json.dumps({k: v for k, v in rep.to_json().items() if k != "provenance"}, indent=1)[:3000])
