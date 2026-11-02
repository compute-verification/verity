"""A run's submissions, one per checkpoint, and the run's block size c in MACs.

c = `C_PER_FORWARD_TOKEN` times the start model's forward MACs per token is the bound a counterexample family's rebuild
must meet (te_core.counterexample) and the record's `c`. The score itself is the training formalism v1's
(te_core.formalism, at te_core.policy's (c, k) in gates); te_core before 4.0 scored by ρ = φcβ/(k(k + β)) over a k
grid, retired with formalism v1.
"""
from __future__ import annotations

from .annotate import annotate

C_PER_FORWARD_TOKEN = 4096


def block_c(forward_macs_per_token: float) -> float:
    return C_PER_FORWARD_TOKEN * forward_macs_per_token


def checkpoints(record: dict) -> list[dict]:
    """Every checkpoint of a run as its own submission: the prefix of C through its step, with its MACs and the
    annotation's summary."""
    out = []
    last = record["steps"][-1]["step"] if record["steps"] else -1
    for cp in record.get("checkpoints", []):
        ann = annotate(record, through=None if cp["step"] >= last else cp["step"])
        out.append({"checkpoint": cp, "macs": ann.macs, "annotation": ann.summary()})
    return out
