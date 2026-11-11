"""GP-01 at tensor parallelism: a TP workload composes the W rank workloads (`workload.WorkloadProgram`, by digest) with each
request's binding of its W rank Programs (`tp_binding.RequestBinding`)."""
from __future__ import annotations

import hashlib
import json
from typing import Any

TP_WORKLOAD_TAG = "verity/tp-workload/v1"


class TpWorkload:
    """GP-01 at tensor parallelism: the W rank workloads in rank order, each GP-01's composition of that rank's request Programs
    (by digest), and each request's binding of its W rank Programs (`tp_binding.RequestBinding`; None for a request no rank ran).
    Its digest is SHA-512 over the canonical `record` of W, the rank workloads' digests and the bindings' digests under
    `TP_WORKLOAD_TAG`. No Program-level edge joins the ranks: each rank's ``peers_*`` stay request-local Inputs
    (`workload._REQUEST_LOCAL_PREFIXES`), and the bindings say which producer output each one is, which a window wires
    (`pipeline.tp.wires`)."""

    def __init__(self, ranks: list[str], bindings: list[Any]) -> None:
        if len(ranks) < 2 or len(set(ranks)) != len(ranks):
            raise ValueError(f"a TP workload composes two or more ranks' distinct workloads, not {len(ranks)} ({len(set(ranks))} distinct)")
        for i, b in enumerate(bindings):
            if b is not None and b.world != len(ranks):
                raise ValueError(f"request {i}'s binding pairs {b.world} ranks' Programs, not the workload's {len(ranks)}")
        self.world, self.ranks, self.bindings = len(ranks), tuple(ranks), tuple(bindings)
        self.record = {"tag": TP_WORKLOAD_TAG, "world": self.world, "ranks": list(self.ranks),
                       "bindings": [None if b is None else b.digest for b in self.bindings]}
        self.digest = hashlib.sha512(json.dumps(self.record, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def artifact(self) -> dict:
        return {**self.record, "tp_workload_digest": self.digest,
                "requests": [None if b is None else {"ranks": list(b.ranks), "binding": b.digest, "sites": len(b.sites)}
                             for b in self.bindings]}
