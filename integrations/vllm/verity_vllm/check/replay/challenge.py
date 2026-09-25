"""Challenge randomness: the seeds the replay drivers sample from, and the generators over them.

A challenge fixes what a driver checks: sampled replay's VUs, replay's instances and slices, the compiled-kernel
check's steps, recompute's reference rows, difftest's cases, the relation checkers' nodes.  Each derivation below is
the one of record for its driver; changing one moves every sample drawn from it, so they differ where the records do.
Only this module constructs random generators.
"""
from __future__ import annotations

import hashlib
import random
from typing import Any

import numpy as np

__all__ = ["case_seed", "challenge_seed", "generator", "kernel_check_seed", "reference_rows_seed", "root_seed", "stream"]


def root_seed(run_root: bytes) -> int:
    """The seed with no verifier challenge, drawn after commitment: the run root's first 8 bytes, big-endian."""
    return int.from_bytes(bytes(run_root)[:8], "big")


def challenge_seed(challenge: str, run_root: bytes) -> int:
    """The seed under a verifier-issued challenge: sha256(challenge | run_root), first 8 bytes, big-endian.  Bound to the
    root, so a challenge issued before the root was fixed, or for another run, selects another sample."""
    return int.from_bytes(hashlib.sha256(challenge.encode("utf-8") + b"|" + bytes(run_root)).digest()[:8], "big")


def kernel_check_seed(run_root: bytes) -> int:
    """The compiled-kernel check's step seed: the run root's first 4 bytes, ``int(run_root.hex()[:8], 16)``."""
    return int(bytes(run_root).hex()[:8], 16)


def reference_rows_seed(rows_total: int, rows_pending: int) -> int:
    """Recompute's reference-row seed: rows sampled in total x 7919 + rows pending a value check."""
    return int(rows_total) * 7919 + int(rows_pending)


def case_seed(seed: int, i: int) -> int:
    """Difftest's seed for the inputs of case ``i``."""
    return int(seed) * 1_000_003 + int(i)


def stream(seed: Any) -> random.Random:
    """A ``random.Random`` over ``seed``, as given."""
    return random.Random(seed)


def generator(seed: Any) -> np.random.Generator:
    """A numpy ``Generator`` over ``seed``, as given."""
    return np.random.default_rng(seed)
