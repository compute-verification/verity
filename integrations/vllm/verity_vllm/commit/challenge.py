"""The integration's one challenge module: every draw over committed values, on `verity.ml.randomness`.

Today's derivations hash the run root, the prover's own commitment (Fiat-Shamir), and their outputs are recorded in
verdicts.  They stay byte-identical while `LEGACY` is true; decision 42 retires them at the re-baseline epoch.  The
non-legacy forms draw through `verity.ml.randomness.derive` from a `source` that is an auditor key, with the run root
and the draw's other bindings as context:

* opening positions (`challenge_positions`): `verity/vllm-commit/opening-positions/v1`;
* opening identities from the expected population (`identity_picker`): `verity/vllm-commit/opening-identities/v1`;
* the replay sample (`replay_key`): `verity/sampled-proofs/vu-sample/v1`, the domain of sampled proofs' former derived
  proof-unit draw, with `per_stratum` units of each stratum drawn uniformly at the stratum's index (`stratum_picks`).

A run of record (`commit.record`: a 64-byte root of ``frame-v3-sha512`` / ``hm96-sha512`` rows) has no legacy draw.  Its source is
drawn once its root is fixed (`draw_live`: the auditor's key, else this process's own coins) and recorded with the run, and every draw
over it is a non-legacy form from that source: `record_positions`, `identity_picker(legacy=False)`, `replay_key` and
`record_kernel_check_seed`, through `positions_of`, `identities_kw` and `kernel_seed_of`, which read a recorded ``vllm-v1`` run with
the legacy forms.

The legacy seeds of the other drivers (replay's tiers, the compiled-kernel check, recompute's reference rows,
difftest, the relation checkers) are here too (`root_seed` .. `case_seed`), with the only generator constructors
(`stream`, `generator`): changing one moves every sample drawn from it.
"""
from __future__ import annotations

import hashlib
import os
import random
from bisect import bisect_right
from collections.abc import Callable, Sequence
from itertools import accumulate
from typing import Any, TypeVar

import numpy as np

from verity.ml.randomness import Key, derive

#: Every derivation's default until the re-baseline epoch (decision 42).
LEGACY = True

POSITIONS_DOMAIN = "verity/vllm-commit/opening-positions/v1"
IDENTITIES_DOMAIN = "verity/vllm-commit/opening-identities/v1"
REPLAY_DOMAIN = "verity/sampled-proofs/vu-sample/v1"
_LEGACY_POSITIONS_TAG = b"verity/cmt-integ/challenge/v0"

T = TypeVar("T")


def _source(legacy: bool, source: bytes | None) -> None:
    if legacy and source is not None:
        raise ValueError("a legacy challenge derives from the run root alone; pass legacy=False with a source")
    if not legacy and not source:
        raise ValueError("a non-legacy challenge needs a source: an auditor key")


def challenge_positions(run_root: bytes, leaves_per_step: list[int], k: int, salt: bytes = b"", *, legacy: bool = LEGACY,
                        source: bytes | None = None) -> list[tuple[int, int]]:
    """`k` committed leaves as (step, leaf within the step), uniform over all leaves with replacement.  Deterministic, so
    the harness and a verifier derive the same list from (root, layout sizes, k, salt) and, off legacy, the source.

    Legacy: 8-byte words of sha256(tag || root || salt || counter), each reduced mod the leaf total."""
    _source(legacy, source)
    if len(run_root) != 32:
        raise ValueError("run root must be 32 bytes")
    total = sum(leaves_per_step)
    if total == 0 or k <= 0:
        return []
    cum = list(accumulate(leaves_per_step))

    def at(g: int) -> tuple[int, int]:
        s = bisect_right(cum, g)
        return s, g - (cum[s - 1] if s else 0)

    if not legacy:
        key = derive(source, POSITIONS_DOMAIN, {"run_root": run_root, "salt": salt, "layout": ",".join(map(str, leaves_per_step))})
        return [at(key.uniform(total, i)) for i in range(k)]
    out: list[tuple[int, int]] = []
    ctr = 0
    while len(out) < k:
        d = hashlib.sha256(_LEGACY_POSITIONS_TAG + run_root + salt + ctr.to_bytes(8, "big")).digest()
        ctr += 1
        for i in range(0, 32, 8):
            out.append(at(int.from_bytes(d[i:i + 8], "big") % total))
            if len(out) >= k:
                break
    return out


def identity_picker(run_root: bytes, map_digest: str, salt: bytes = b"", *, legacy: bool = LEGACY,
                    source: bytes | None = None) -> Callable[[int, int], int]:
    """`pick(i, n)`: the i-th draw in [0, n) of an opening-identity challenge bound to (root, binding-map digest, salt).

    Legacy: sha256(sha256(root || map digest || salt) || i as 4 bytes)[:8] mod n."""
    _source(legacy, source)
    if not legacy:
        key = derive(source, IDENTITIES_DOMAIN, {"run_root": run_root, "map_digest": bytes.fromhex(map_digest), "salt": salt})
        return lambda i, n: key.uniform(n, i)
    seed = hashlib.sha256(run_root + bytes.fromhex(map_digest) + salt).digest()
    return lambda i, n: int.from_bytes(hashlib.sha256(seed + i.to_bytes(4, "big")).digest()[:8], "big") % n


def legacy_replay_seed(challenge: str, run_root: bytes) -> int:
    """The legacy replay seed under a verifier-issued challenge: sha256(challenge || "|" || run_root)[:8].  Its non-legacy
    form is `replay_key`."""
    return int.from_bytes(hashlib.sha256(challenge.encode("utf-8") + b"|" + bytes(run_root)).digest()[:8], "big")


def replay_key(source: bytes, run_root: bytes) -> Key:
    """The replay sample's key off legacy, derived after the run root."""
    _source(False, source)
    return derive(source, REPLAY_DOMAIN, {"run_root": bytes(run_root)})


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


# --- draws over a run of record (`commit.record`): live coins only ---------------------------------------------------------

RECORD_ROOT_BYTES = 64
KERNEL_CHECK_DOMAIN = "verity/vllm-commit/kernel-check-steps/v1"


def of_record(run_root: bytes) -> bool:
    """A run of record has the record's 64-byte root; a run recorded before it, a 32-byte ``vllm-v1`` root (read with the legacy
    derivations above)."""
    return len(run_root) == RECORD_ROOT_BYTES


def live_source(given: str | bytes | None = None) -> tuple[bytes, str]:
    """The source of every draw over a run of record, taken once its root is fixed: ``(key, origin)``.  The auditor's key when one is
    ``given`` (hex or bytes; origin ``auditor``), else 32 fresh OS bytes this process draws as the run's own auditor (origin ``own``,
    the Commit's rule for PoUW's coins: ``protocol_options.own_auditor``).  Never derived from the run root."""
    if given:
        key = bytes.fromhex(given) if isinstance(given, str) else bytes(given)
        if len(key) < 16:
            raise ValueError(f"a challenge key is at least 16 bytes, not {len(key)}")
        return key, "auditor"
    return os.urandom(32), "own"


def record_positions(record_root: bytes, leaves_per_step: Sequence[int], k: int, salt: bytes = b"", *, source: bytes) -> list[tuple[int, int]]:
    """`challenge_positions` off legacy over a root of record: `k` leaves uniform over all leaves with replacement, drawn from
    ``derive(source, POSITIONS_DOMAIN, {run_root, salt, layout})``."""
    _source(False, source)
    if not of_record(record_root):
        raise ValueError(f"a root of record is {RECORD_ROOT_BYTES} bytes, not {len(record_root)}")
    total = sum(leaves_per_step)
    if total == 0 or k <= 0:
        return []
    cum = list(accumulate(leaves_per_step))
    key = derive(source, POSITIONS_DOMAIN, {"run_root": bytes(record_root), "salt": salt, "layout": ",".join(map(str, leaves_per_step))})
    out = []
    for i in range(k):
        g = key.uniform(total, i)
        s = bisect_right(cum, g)
        out.append((s, g - (cum[s - 1] if s else 0)))
    return out


def record_kernel_check_seed(record_root: bytes, *, source: bytes) -> int:
    """`kernel_check_seed` over a root of record: a 32-bit seed drawn from ``derive(source, KERNEL_CHECK_DOMAIN, {run_root})``."""
    _source(False, source)
    return derive(source, KERNEL_CHECK_DOMAIN, {"run_root": bytes(record_root)}).uniform(1 << 32)


def draw_live(run: Any, given: str | bytes | None = None) -> bytes | None:
    """Draw a run of record's live source (`live_source`) once its root is fixed, and put it on the run's record (``run.extra
    ["challenge"]``: the key, its origin and the root it was drawn after; a verifier's coins are public once drawn).  None for a run
    recorded before the record, whose draws stay the legacy ones."""
    if not of_record(bytes(run.run_root)):
        return None
    key, origin = live_source(given)
    run.extra["challenge"] = {"key": key.hex(), "origin": origin, "after_run_root": bytes(run.run_root).hex()}
    return key


def source_of(run: Any) -> bytes:
    """The live source a run of record drew (`draw_live`), from a `RunCommitment` or a row's ``commit`` record; refused by name when it
    drew none or drew it after another root."""
    extra = run.extra if hasattr(run, "extra") else (run.get("extra") or {})
    root = bytes(run.run_root).hex() if hasattr(run, "run_root") else str(run.get("run_root"))
    ch = (extra or {}).get("challenge") or {}
    if not ch.get("key") or ch.get("after_run_root") != root:
        raise ValueError(f"run of record {root[:16]}: no live key drawn after its root (draw_live) -- no challenge without one")
    return bytes.fromhex(ch["key"])


def positions_of(run: Any, k: int, salt: bytes = b"") -> list[tuple[int, int]]:
    """``k`` opening positions of ``run``: `record_positions` from its live source over a root of record, else the recorded legacy
    `challenge_positions` over its ``vllm-v1`` root."""
    leaves = [s.n_leaves for s in run.steps]
    if of_record(bytes(run.run_root)):
        return record_positions(bytes(run.run_root), leaves, k, salt, source=source_of(run))
    return challenge_positions(bytes(run.run_root), leaves, k, salt)


def identities_kw(run: Any) -> dict[str, Any]:
    """`identity_picker`'s (and `binding.challenge_identities`') keywords for ``run``: off legacy with its live source over a root of
    record, else none (the recorded legacy form)."""
    return {"legacy": False, "source": source_of(run)} if of_record(bytes(run.run_root)) else {}


def kernel_seed_of(run: Any) -> int:
    """The compiled-kernel check's step seed for ``run``: `record_kernel_check_seed` over a root of record, else `kernel_check_seed`."""
    root = bytes(run.run_root)
    return record_kernel_check_seed(root, source=source_of(run)) if of_record(root) else kernel_check_seed(root)
