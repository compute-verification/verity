"""A run commitment's roots (`acquire.committer_api.RunCommitment.roots()`): the run root and each step's root and sizes, without the
committer's measurements, Merkle roots or layout digests.  A committed store keeps this much of a run (`commit.store_dump`)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class StepRoot:
    """A `StepCommitment`'s step root and the leaves, value bytes and tensors it binds."""
    step: int
    root: bytes
    n_leaves: int
    n_bytes: int
    n_tensors: int


@dataclass
class RunRoots:
    """The run root and each step's `StepRoot`."""
    run_root: bytes
    steps: list[StepRoot]
