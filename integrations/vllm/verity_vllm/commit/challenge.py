"""The integration's one challenge module: every sample over committed values, on `verity.ml.randomness`.

Each sample derives through `verity.ml.randomness.derive` from a `source`, the run's live coin (`pipeline.commit_coins`:
the auditor host's challenger opens it after the run's root), with the run root and the sample's other bindings as
context.  A sample without its coin is refused by name:

* opening positions (`challenge_positions`): `verity/vllm-commit/opening-positions/v1`;
* opening identities from the expected population (`identity_picker`): `verity/vllm-commit/opening-identities/v1`;
* the replay sample (`replay_key`): `verity/sampled-proofs/vu-sample/v1`, the domain of sampled proofs' former derived
  proof-block draw, with `per_stratum` units of each stratum drawn uniformly at the stratum's index (`stratum_picks`).

The legacy seeds of the other drivers (replay's tiers, the compiled-kernel check, recompute's reference rows,
difftest, the relation checkers) are here too (`root_seed` .. `case_seed`), with the only generator constructors
(`stream`, `generator`): changing one moves every sample drawn from it.
"""
from __future__ import annotations

import hashlib
import random
from bisect import bisect_right
from collections.abc import Callable, Sequence
from itertools import accumulate
from typing import Any, TypeVar

import numpy as np

from verity.ml.randomness import Key, derive

POSITIONS_DOMAIN = "verity/vllm-commit/opening-positions/v1"
IDENTITIES_DOMAIN = "verity/vllm-commit/opening-identities/v1"
REPLAY_DOMAIN = "verity/sampled-proofs/vu-sample/v1"

T = TypeVar("T")


def _source(source: bytes | None) -> bytes:
    if not source:
        raise ValueError("a challenge needs a source: the run's live coin, an auditor key")
    return source


class NoRunRoot(ValueError):
    """A run whose committer left a step uncommitted has no run root to draw from."""


def run_root_of(run: Any) -> bytes:
    """``run``'s root (a `committer_api.RunCommitment`: rows-sha512's 64 bytes, or vllm-v1's 32), the one its challenge
    derives from.  A run with a step its committer did not commit (a placeholder root, the step's error in
    ``run.extra["errors"]``) has none: `NoRunRoot`, naming those steps and the first error."""
    root = bytes(run.run_root)
    if len(root) in (32, 64):
        return root
    bad = [s.step for s in run.steps if len(s.root) not in (32, 64)]
    errors = list((run.extra or {}).get("errors") or [])
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


def identity_picker(run_root: bytes, map_digest: str, salt: bytes = b"", *, source: bytes | None) -> Callable[[int, int], int]:
    """`pick(i, n)`: the i-th sample in [0, n) of an opening-identity challenge bound to (source, root, binding-map digest,
    salt)."""
    key = derive(_source(source), IDENTITIES_DOMAIN, {"run_root": run_root, "map_digest": bytes.fromhex(map_digest), "salt": salt})
    return lambda i, n: key.uniform(n, i)


def legacy_replay_seed(challenge: str, run_root: bytes) -> int:
    """The legacy replay seed under a verifier-issued challenge: sha256(challenge || "|" || run_root)[:8].  Its non-legacy
    form is `replay_key`."""
    return int.from_bytes(hashlib.sha256(challenge.encode("utf-8") + b"|" + bytes(run_root)).digest()[:8], "big")


def replay_key(source: bytes, run_root: bytes) -> Key:
    """The replay sample's key, derived after the run root."""
    return derive(_source(source), REPLAY_DOMAIN, {"run_root": bytes(run_root)})


def stratum_picks(pools: Sequence[Sequence[T]], per_stratum: int, seed: int | None, key: Key | None = None) -> list[list[T]]:
    """`per_stratum` members of each pool (all of a smaller pool), in pool order.

    With `seed` (legacy; pass None with a key): one `random.Random(seed)` samples the pools in turn.  With `key`: pool i's
    members at the positions of `key.subset(min(per_stratum, len(pool)), len(pool), ("vu", i))`."""
    if (seed is None) == (key is None):
        raise ValueError("stratum_picks takes a legacy seed or a key, not both")
    if key is not None:
        if int(per_stratum) < 0:
            raise ValueError("stratum_picks needs per_stratum >= 0")
        return [[pool[j] for j in key.subset(min(int(per_stratum), len(pool)), len(pool), ("vu", i))]
                for i, pool in enumerate(pools)]
    rng = random.Random(int(seed))
    out = []
    for pool in pools:
        n = min(int(per_stratum), len(pool))
        out.append(rng.sample(list(pool), n) if n < len(pool) else list(pool))
    return out


def root_seed(run_root: bytes) -> int:
    """The legacy seed with no verifier challenge, drawn after commitment: the run root's first 8 bytes, big-endian."""
    return int.from_bytes(bytes(run_root)[:8], "big")


def challenge_seed(challenge: str, run_root: bytes) -> int:
    """The legacy seed under a verifier-issued challenge (`legacy_replay_seed`): bound to the root, so a challenge issued
    before the root was fixed, or for another run, selects another sample."""
    return legacy_replay_seed(challenge, run_root)


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
