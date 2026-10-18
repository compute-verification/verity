"""The integration's one challenge module: every draw over committed values, on `verity.ml.randomness`.

The legacy derivations hash the run root, the prover's own commitment (Fiat-Shamir), and their outputs are recorded in
verdicts.  They read runs recorded before the record (a 32-byte ``vllm-v1`` root) and stay byte-identical for them while `LEGACY`
is true; decision 42 retires them, and a run of record never takes them.  The non-legacy forms draw through `verity.ml.randomness.derive` from a `source` that is an auditor key, with the run root
and the draw's other bindings as context:

* opening positions (`challenge_positions`): `verity/vllm-commit/opening-positions/v1`;
* opening identities from the expected population (`identity_picker`): `verity/vllm-commit/opening-identities/v1`;
* the replay sample (`replay_key`): `verity/sampled-proofs/vu-sample/v1`, the domain of sampled proofs' former derived
  proof-unit draw, with `per_stratum` units of each stratum drawn uniformly at the stratum's index (`stratum_picks`).

A run of record (`commit.record`: a 64-byte root of ``frame-v3-sha512`` / ``hm96-sha512`` rows) has no legacy draw.  Its sources
are a `LiveChallenge` a `Challenger` answers its root with, once the root is fixed (`draw_live`), recorded with the run: a key per
label of `LABELS`, and every draw over the root is a non-legacy form from its label's key: `record_positions` (`POSITIONS`),
`identity_picker(legacy=False)` (`IDENTITIES`), `replay_key` (`REPLAY`) and `record_kernel_check_seed` (`KERNEL_CHECK`), through
`positions_of`, `identities_kw` and `kernel_seed_of`, which read a recorded ``vllm-v1`` run with the legacy forms.  The Commit is a
client of the proof service's challenger (`verity.core.protocols.sampled_proofs.challenger`, scheme `SCHEME`, `layout`): coin i of each
label opens once, after the Commit's i-th root (`service_challenger`).  Only a challenge whose coins open against a commitment the
verifier recorded from its own challenger is post-commitment of record (`attested_reason`, `trust`); a key handed in
(`key_challenger`), the run's own bytes (`own_challenger`) or a challenger the prover's process holds draw measurement samples.

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
from dataclasses import dataclass
from itertools import accumulate
from typing import Any, TypeVar

import numpy as np

from verity.ml.randomness import Key, derive

#: The default of the legacy-capable draws: True, so a recorded ``vllm-v1`` run reads its recorded picks (decision 42).
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


REFERENCE_ROWS_DOMAIN = "verity/vllm-check/reference-rows/v1"


def reference_rows(source: bytes, rows_total: int, pending: int, k: int) -> list[int]:
    """Recompute's reference rows off legacy: ``min(k, pending)`` of the ``pending`` value-checked rows, ascending, drawn from
    ``derive(source, REFERENCE_ROWS_DOMAIN, {rows_total, pending})``."""
    _source(False, source)
    key = derive(source, REFERENCE_ROWS_DOMAIN, {"rows_total": int(rows_total), "pending": int(pending)})
    return sorted(key.subset(min(int(k), int(pending)), int(pending), "rows"))


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


#: The Commit as a client of the proof service's challenger (`verity.core.protocols.sampled_proofs.challenger`): its scheme, and one
#: coin family per draw over a run of record, coin i of each opened after the Commit's i-th root of record.
SCHEME = "vllm/commit"
POSITIONS, IDENTITIES, REPLAY, KERNEL_CHECK = "opening-positions", "opening-identities", "replay", "kernel-check"
LABELS = (POSITIONS, IDENTITIES, REPLAY, KERNEL_CHECK)
COIN_BYTES = 64

#: the one origin whose challenge can be post-commitment of record: an answer of the service's challenger, with its coin openings
ATTESTED = "challenger"

#: the coin commitments the verifier recorded from its own challenger before their epochs (`trust`; `LiveCommitment.to_json`, by root hex)
TRUSTED: dict[str, dict[str, Any]] = {}


def layout(runs: int) -> tuple[tuple[str, int], ...]:
    """The coin layout of a Commit of at most ``runs`` runs of record: each label's family holds a coin per run."""
    return tuple((label, int(runs)) for label in LABELS)


@dataclass(frozen=True)
class LiveChallenge:
    """A live challenge answering a committed root: a ``key`` per label in `LABELS`, every draw of that label over the root derives from
    it; the root it came ``after``; its ``origin`` (`ATTESTED`; ``unattested``: a key handed to this process; ``own``: this process's
    own bytes) and, from the service's challenger, its ``provenance``: the coin commitment and the label's coin openings."""

    keys: dict[str, bytes]
    after: bytes
    origin: str
    provenance: dict[str, Any] | None = None

    def to_json(self) -> dict[str, Any]:
        return {"keys": {label: k.hex() for label, k in self.keys.items()}, "origin": self.origin, "after_run_root": self.after.hex(),
                **({"provenance": self.provenance} if self.provenance is not None else {})}


#: The openings' seam to the proof service's challenger: a committed root in, the live challenge answering it out.  Positions are
#: then `record_positions(root, layout, k, salt, source=challenge.keys[POSITIONS])`, a function of the root and the challenge alone.
Challenger = Callable[[bytes], LiveChallenge]


def service_challenger(challenger: Any) -> Challenger:
    """A `Challenger` over the service's challenger (a `challenger.LiveChallenger` holding `layout(runs)` under `SCHEME`): each label's
    `LiveRecord` records the Commit's next root and only then opens that label's next coin after it, once.  Origin `ATTESTED`; whether
    the commitment is the auditor's is the verifier's check (`attested_reason`)."""
    from verity.core.protocols.sampled_proofs.challenger import LiveRecord

    if challenger.commitment.scheme != SCHEME:
        raise ValueError(f"the Commit's coins are under scheme {SCHEME!r}, not {challenger.commitment.scheme!r}")
    records = {label: LiveRecord(challenger, label) for label in LABELS}

    def ask(root: bytes) -> LiveChallenge:
        openings = {label: r.record(bytes(root)) for label, r in records.items()}
        return LiveChallenge({label: o.coin for label, o in openings.items()}, bytes(root), ATTESTED,
                             {"commitment": challenger.commitment.to_json(), "openings": {label: o.to_json() for label, o in openings.items()}})
    return ask


def key_challenger(given: str | bytes) -> Challenger:
    """A challenger over a key handed to this process (hex or bytes, at least 16), the key of every label: origin ``unattested``, since
    nothing shows the key came from the service's challenger after the root."""
    key = bytes.fromhex(given) if isinstance(given, str) else bytes(given)
    if len(key) < 16:
        raise ValueError(f"a challenge key is at least 16 bytes, not {len(key)}")
    return lambda root: LiveChallenge(dict.fromkeys(LABELS, key), bytes(root), "unattested")


def own_challenger(root: bytes) -> LiveChallenge:
    """The run as its own auditor: `COIN_BYTES` OS bytes per label once the root is fixed, origin ``own``.  A measurement draw, never
    post-commitment of record."""
    return LiveChallenge({label: os.urandom(COIN_BYTES) for label in LABELS}, bytes(root), "own")


def commit_challenger(given: str | bytes | None, runs: int) -> Challenger:
    """The Commit's challenger: a key handed in (`key_challenger`), else the service's challenger (a `challenger.LiveChallenger` over
    `layout(runs)`, epoch 0) held by this process.  Its commitment is this process's own, so its draws are measurement samples unless the
    verifier recorded that commitment from its own challenger (`trust`)."""
    if given:
        return key_challenger(given)
    from verity.core.protocols.sampled_proofs.challenger import LiveChallenger

    return service_challenger(LiveChallenger(SCHEME, 0, layout(runs)))


def trust(*commitments: dict[str, Any]) -> None:
    """Record coin commitments (`LiveCommitment.to_json`) as the verifier's challenger's, published before their epochs."""
    from verity.core.protocols.sampled_proofs.challenger import LiveCommitment

    for doc in commitments:
        c = LiveCommitment.from_json(doc)
        TRUSTED[c.root.hex()] = c.to_json()


def attested_reason(challenge: dict[str, Any] | None) -> str | None:
    """Why a recorded challenge (``extra.challenge``) is not the service's challenger's after its root; None when it is: origin
    `ATTESTED`, a coin commitment the verifier recorded from its own challenger (`trust`), and per label a `COIN_BYTES` coin that opens
    against it after the recorded root (`challenger.live_reason`) and is the label's key."""
    ch = challenge or {}
    if ch.get("origin") != ATTESTED:
        return f"its live keys' origin is {ch.get('origin')!r}, not the service challenger's"
    from verity.core.protocols.sampled_proofs import challenger as LC

    prov, keys = ch.get("provenance") or {}, ch.get("keys") or {}
    try:
        com = LC.LiveCommitment.from_json(prov.get("commitment"))
        after = bytes.fromhex(str(ch.get("after_run_root")))
        openings = {label: LC.LiveOpening.from_json(o) for label, o in (prov.get("openings") or {}).items()}
    except (LC.CoinError, ValueError, TypeError, KeyError) as e:
        return f"its coin commitment or openings are malformed ({e})"
    if com.scheme != SCHEME:
        return f"its coins are under scheme {com.scheme!r}, not the Commit's {SCHEME!r}"
    if TRUSTED.get(com.root.hex()) != com.to_json():
        return f"its coin commitment {com.root.hex()[:16]} is not one the verifier recorded from its challenger (challenge.trust)"
    if sorted(openings) != sorted(LABELS) or sorted(keys) != sorted(LABELS):
        return f"it does not carry one opened coin and one key per label ({', '.join(LABELS)})"
    for label in LABELS:
        o = openings[label]
        if len(o.coin) != COIN_BYTES:
            return f"its {label} coin is {len(o.coin)} bytes, not a {COIN_BYTES}-byte live coin"
        if com.position(label, o.index) is None:
            return f"its {label} coin {o.index} is not in the epoch's layout"
        why = LC.live_reason(com, o, label, o.index, after)
        if why is not None:
            return f"its {label} {why}"
        if keys[label] != o.coin.hex():
            return f"its {label} key is not the coin the challenger opened"
    return None


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


def draw_live(run: Any, challenger: Challenger | str | bytes | None = None) -> LiveChallenge | None:
    """Ask ``challenger`` for a run of record's live challenge once its root is fixed, and put it on the run's record (``run.extra
    ["challenge"]``, `LiveChallenge.to_json`; a verifier's coins are public once drawn).  A key (hex or bytes) is `key_challenger`'s,
    None `own_challenger`.  None for a run recorded before the record, whose draws stay the legacy ones."""
    root = bytes(run.run_root)
    if not of_record(root):
        return None
    ask = challenger if callable(challenger) else key_challenger(challenger) if challenger else own_challenger
    ch = ask(root)
    if (not isinstance(ch, LiveChallenge) or ch.after != root or sorted(ch.keys) != sorted(LABELS)
            or any(len(k) < 16 for k in ch.keys.values())):
        raise ValueError(f"run of record {root.hex()[:16]}: the challenger's answer is not a live challenge after its root")
    run.extra["challenge"] = ch.to_json()
    return ch


def source_of(run: Any, label: str) -> bytes:
    """The live key of ``label`` a run of record drew (`draw_live`), from a `RunCommitment` or a row's ``commit`` record; refused by name
    when it drew none or drew it after another root."""
    extra = run.extra if hasattr(run, "extra") else (run.get("extra") or {})
    root = bytes(run.run_root).hex() if hasattr(run, "run_root") else str(run.get("run_root"))
    ch = (extra or {}).get("challenge") or {}
    key = (ch.get("keys") or {}).get(label)
    if not key or ch.get("after_run_root") != root:
        raise ValueError(f"run of record {root[:16]}: no live {label} key drawn after its root (draw_live) -- no challenge without one")
    return bytes.fromhex(key)


def positions_of(run: Any, k: int, salt: bytes = b"") -> list[tuple[int, int]]:
    """``k`` opening positions of ``run``: `record_positions` from its live `POSITIONS` key over a root of record, else the recorded
    legacy `challenge_positions` over its ``vllm-v1`` root."""
    leaves = [s.n_leaves for s in run.steps]
    if of_record(bytes(run.run_root)):
        return record_positions(bytes(run.run_root), leaves, k, salt, source=source_of(run, POSITIONS))
    return challenge_positions(bytes(run.run_root), leaves, k, salt)


def identities_kw(run: Any) -> dict[str, Any]:
    """`identity_picker`'s (and `binding.challenge_identities`') keywords for ``run``: off legacy with its live `IDENTITIES` key over a
    root of record, else none (the recorded legacy form)."""
    return {"legacy": False, "source": source_of(run, IDENTITIES)} if of_record(bytes(run.run_root)) else {}


def kernel_seed_of(run: Any) -> int:
    """The compiled-kernel check's step seed for ``run``: `record_kernel_check_seed` from its live `KERNEL_CHECK` key over a root of
    record, else `kernel_check_seed`."""
    root = bytes(run.run_root)
    return record_kernel_check_seed(root, source=source_of(run, KERNEL_CHECK)) if of_record(root) else kernel_check_seed(root)
