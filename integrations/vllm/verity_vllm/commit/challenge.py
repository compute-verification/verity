"""The integration's one challenge module: every draw over committed values, on `verity.randomness`.

Today's derivations hash the run root, the prover's own commitment (Fiat-Shamir), and their outputs are recorded in
verdicts.  They stay byte-identical while `LEGACY` is true; decision 42 retires them at the re-baseline epoch.  The
non-legacy forms are the verifier's (deployment decision A24; no beacon, as sampled proofs has none since #132): every coin
is `verity.randomness.derive` keyed by the verifier's secret, never by a root, with the receipt of the registration it
follows and the draw's other bindings as context:

* the run's salt (`run_salt`): `verity/vllm-commit/run-salt/v1` over the weights root and the run id, issued after the
  weights root's receipt; the prover commits it as an Input, anchored in the registration record;
* opening positions (`challenge_positions`): `verity/vllm-commit/opening-positions/v1`;
* opening identities from the expected population (`identity_picker`): `verity/vllm-commit/opening-identities/v1`;
* the replay sample (`replay_key`): `verity/vllm-commit/replay-sample/v1`, each stratum a replay unit selected with
  p = 1 and `per_stratum` of its units checked (`stratum_picks`).

`RunWindows` is the verifier's side of a run's registrations: each window's call indices registered once, a window whose
indices include one an earlier window registered refused (X-SPC-18), and an aborted window a rejected run.  Its receipts
are what the draws follow.

The legacy seeds of the other drivers (replay's tiers, the compiled-kernel check, recompute's reference rows,
difftest, the relation checkers) are here too (`root_seed` .. `case_seed`), with the only generator constructors
(`stream`, `generator`): changing one moves every sample drawn from it.
"""
from __future__ import annotations

import hashlib
import random
from bisect import bisect_right
from collections.abc import Callable, Sequence
from fractions import Fraction
from itertools import accumulate
from typing import Any, TypeVar

import numpy as np

from verity.randomness import Key, derive
from verity_sampled_proofs.law import ReplayUnit, Stage, TwoStageLaw, select_verification_units

#: Every derivation's default until the re-baseline epoch (decision 42).
LEGACY = True

POSITIONS_DOMAIN = "verity/vllm-commit/opening-positions/v1"
IDENTITIES_DOMAIN = "verity/vllm-commit/opening-identities/v1"
REPLAY_DOMAIN = "verity/vllm-commit/replay-sample/v1"
SALT_DOMAIN = "verity/vllm-commit/run-salt/v1"
RECEIPT_TAG = b"verity/vllm-commit/window-receipt/v1"
_LEGACY_POSITIONS_TAG = b"verity/cmt-integ/challenge/v0"

T = TypeVar("T")


def _source(legacy: bool, source: bytes | None, receipt: bytes | None = b"") -> None:
    if legacy and (source is not None or receipt):
        raise ValueError("a legacy challenge derives from the run root alone; pass legacy=False with a source and a receipt")
    if not legacy and not source:
        raise ValueError("a non-legacy challenge needs a source: the verifier's secret")
    if not legacy and not receipt:
        raise ValueError("a non-legacy challenge is drawn after a receipt: pass the registration receipt it follows")


def run_salt(secret: bytes, weights_root: bytes, run_id: str) -> bytes:
    """The run's 32-byte salt, the verifier's: derived from its secret over the weights root and the run id, after the weights
    root's receipt.  Never from a prover's value alone."""
    _source(False, secret, b"-")
    return derive(secret, SALT_DOMAIN, {"weights_root": bytes(weights_root), "run_id": str(run_id)}).stream("salt", 32)


def challenge_positions(run_root: bytes, leaves_per_step: list[int], k: int, salt: bytes = b"", *, legacy: bool = LEGACY,
                        source: bytes | None = None, receipt: bytes | None = None) -> list[tuple[int, int]]:
    """`k` committed leaves as (step, leaf within the step), uniform over all leaves with replacement.  Deterministic, so
    the harness and a verifier derive the same list from (root, layout sizes, k, salt) and, off legacy, the verifier's secret
    and the receipt the draw follows.

    Legacy: 8-byte words of sha256(tag || root || salt || counter), each reduced mod the leaf total."""
    _source(legacy, source, receipt)
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
        key = derive(source, POSITIONS_DOMAIN, {"run_root": run_root, "receipt": bytes(receipt), "salt": salt,
                                                "layout": ",".join(map(str, leaves_per_step))})
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
                    source: bytes | None = None, receipt: bytes | None = None) -> Callable[[int, int], int]:
    """`pick(i, n)`: the i-th draw in [0, n) of an opening-identity challenge bound to (root, binding-map digest, salt), and off
    legacy to the receipt it follows.

    Legacy: sha256(sha256(root || map digest || salt) || i as 4 bytes)[:8] mod n."""
    _source(legacy, source, receipt)
    if not legacy:
        key = derive(source, IDENTITIES_DOMAIN, {"run_root": run_root, "receipt": bytes(receipt),
                                                 "map_digest": bytes.fromhex(map_digest), "salt": salt})
        return lambda i, n: key.uniform(n, i)
    seed = hashlib.sha256(run_root + bytes.fromhex(map_digest) + salt).digest()
    return lambda i, n: int.from_bytes(hashlib.sha256(seed + i.to_bytes(4, "big")).digest()[:8], "big") % n


def legacy_replay_seed(challenge: str, run_root: bytes) -> int:
    """The legacy replay seed under a verifier-issued challenge: sha256(challenge || "|" || run_root)[:8].  Its non-legacy
    form is `replay_key`."""
    return int.from_bytes(hashlib.sha256(challenge.encode("utf-8") + b"|" + bytes(run_root)).digest()[:8], "big")


def replay_key(source: bytes, run_root: bytes, receipt: bytes) -> Key:
    """The replay sample's key off legacy: the verifier's secret, over the run root and the receipt of its registration."""
    _source(False, source, receipt)
    return derive(source, REPLAY_DOMAIN, {"run_root": bytes(run_root), "receipt": bytes(receipt)})


class RunWindows:
    """The verifier's record of one run's registrations (X-SPC-18).  `register` takes a window's call indices (hashable
    positions: forward, layer, site) and its root, and returns the window's receipt.  It refuses a window already registered
    (a window is never re-registered), duplicate indices within it, any index an earlier window of the run registered, and
    everything after an abort (`abort`: the run is rejected)."""

    def __init__(self, run_id: str) -> None:
        self.run_id = str(run_id)
        self.windows: dict[int, bytes] = {}
        self.owner: dict[Any, int] = {}
        self.aborted: int | None = None

    def register(self, window: int, indices: Sequence[Any], root: bytes) -> bytes:
        if self.aborted is not None:
            raise ValueError(f"run {self.run_id} was rejected: window {self.aborted} aborted")
        if window in self.windows:
            raise ValueError(f"window {window} of run {self.run_id} is already registered; a window is never re-registered")
        if len(set(indices)) != len(indices):
            raise ValueError(f"window {window} lists a call index twice")
        again = sorted((str(i), self.owner[i]) for i in indices if i in self.owner)
        if again:
            raise ValueError(f"window {window} re-registers call indices of earlier windows of run {self.run_id}: {again[:5]}")
        digest = hashlib.sha256(b"".join(hashlib.sha256(repr(i).encode()).digest() for i in sorted(indices, key=repr))).digest()
        receipt = hashlib.sha256(RECEIPT_TAG + self.run_id.encode() + int(window).to_bytes(8, "big") + bytes(root) + digest).digest()
        self.windows[window] = receipt
        self.owner.update({i: window for i in indices})
        return receipt

    def abort(self, window: int) -> None:
        """An aborted window rejects the run: nothing registers after it."""
        self.aborted = int(window)


def stratum_picks(pools: Sequence[Sequence[T]], per_stratum: int, seed: int | None, key: Key | None = None) -> list[list[T]]:
    """`per_stratum` members of each pool (all of a smaller pool), in pool order.

    With `seed` (legacy; pass None with a key): one `random.Random(seed)` samples the pools in turn.  With `key`: the two-stage law with every
    pool a replay unit at p = 1 and k = `per_stratum`, pool i drawn at index i."""
    if (seed is None) == (key is None):
        raise ValueError("stratum_picks takes a legacy seed or a key, not both")
    if key is not None:
        law = TwoStageLaw({"stratum": Stage(Fraction(1), int(per_stratum))})
        vus = select_verification_units(key, law, [ReplayUnit(i, "stratum", len(p)) for i, p in enumerate(pools)])
        return [[pool[j] for j in vus[i]] for i, pool in enumerate(pools)]
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
