"""Served domains: the frame domain of a root a serving window registers.

Each port's domain binds the program, the partition, the port, its schema, its leaf count and the run, so the roots can't be
read as anything else: ``binding = identity_digest_sha512(SERVED_DOMAIN, {program, partition, port, schema, leaves, run})``,
under ``CommitmentDomain(binding, -1, [0, leaves), "sha512")``.  Sampled proofs' registration recomputes every served root's
domain with it (``one_stage.registration``, which names it too), and a serving stack commits each port under it.
"""
from __future__ import annotations

from .identity import identity_digest_sha512
from .indexed import RangeIndexedDomain
from .merkle import CommitmentDomain

SERVED_DOMAIN = "verity/one-stage/served-domain/v1"


def served_domain(program: str, partition: str, port: str, schema: str, leaves: int, run: str) -> str:
    binding = identity_digest_sha512(SERVED_DOMAIN, {"program": program, "partition": partition, "port": port, "schema": schema,
                                                     "leaves": int(leaves), "run": run})
    return CommitmentDomain(binding, -1, RangeIndexedDomain(0, int(leaves)), hash="sha512").domain_id.hex()


__all__ = ["SERVED_DOMAIN", "served_domain"]
