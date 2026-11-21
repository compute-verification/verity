"""The integration's one challenge module: every sample over committed values.

A sample on `verity.commitments.randomness` derives through its `derive` from a `source`, the run's live coin
(`pipeline.commit_coins`: the auditor device's challenger opens it after the run's root), with the run root and the sample's
other bindings as context.  Such a sample without its coin is refused by name:

* opening positions (`challenge_positions`): `verity/vllm-commit/opening-positions/v1`.

The only generator constructors (`stream`, `generator`) are here too: changing one moves every sample taken from it.
"""
from __future__ import annotations

import random
from bisect import bisect_right
from itertools import accumulate
from typing import Any

import numpy as np
from verity.commitments.randomness import derive

__all__ = ["challenge_positions", "generator", "run_root_of", "stream"]

POSITIONS_DOMAIN = "verity/vllm-commit/opening-positions/v1"


def _source(source: bytes | None) -> bytes:
    if not source:
        raise ValueError("a challenge needs a source: the run's live coin, an auditor key")
    return source


class NoRunRoot(ValueError):
    """A run whose committer left a step uncommitted has no run root to sample from."""


def run_root_of(run: Any) -> bytes:
    """``run``'s root (a `committer_api.RunCommitment`: rows-sha512's 64 bytes, or vllm-v1's 32), the one its challenge
    derives from.  A run with a step its committer did not commit (a placeholder root, the step's error in
    ``run.extra["errors"]``) has none: `NoRunRoot`, naming those steps and the first error."""
    root = bytes(run.run_root)
    if len(root) in (32, 64):
        return root
    bad = [s.step for s in run.steps if len(s.root) not in (32, 64)]
    extra = run.extra
    listed = extra.get("errors") if extra else None
    errors = list(listed) if listed else []
    raise NoRunRoot(f"no run root: {len(bad)} of {len(run.steps)} steps not committed (steps {bad[:8]}"
                    f"{' ...' if len(bad) > 8 else ''}); first error: {errors[0] if errors else 'none recorded'}")


def challenge_positions(run_root: bytes, leaves_per_step: list[int], k: int, salt: bytes = b"", *,
                        source: bytes | None) -> list[tuple[int, int]]:
    """`k` committed leaves as (step, leaf within the step), uniform over all leaves with replacement.  Deterministic, so
    the harness and a verifier derive the same list from (source, root, layout sizes, k, salt).  The root is vllm-v1's 32
    bytes or a SHA-512 root's 64 (rows-sha512, the TP run root)."""
    _source(source)
    if len(run_root) not in (32, 64):
        raise ValueError(f"a run root is 32 bytes (vllm-v1) or 64 (SHA-512), not {len(run_root)}")
    total = sum(leaves_per_step)
    if total == 0 or k <= 0:
        return []
    cum = list(accumulate(leaves_per_step))

    def at(g: int) -> tuple[int, int]:
        s = bisect_right(cum, g)
        return s, g - (cum[s - 1] if s else 0)

    key = derive(source, POSITIONS_DOMAIN, {"run_root": run_root, "salt": salt, "layout": ",".join(map(str, leaves_per_step))})
    return [at(key.uniform(total, i)) for i in range(k)]


def stream(seed: Any) -> random.Random:
    """A ``random.Random`` over ``seed``, as given."""
    return random.Random(seed)


def generator(seed: Any) -> np.random.Generator:
    """A numpy ``Generator`` over ``seed``, as given."""
    return np.random.default_rng(seed)
