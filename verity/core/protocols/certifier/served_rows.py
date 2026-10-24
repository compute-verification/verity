"""The served rows: the step rows a served window ran, committed on the Commit side before the draw, which the compile
slice's ``rows`` bit holds the replay to. With ``pub.digest`` the prover's own word, a prover who served user-1's admission
at the sixth ask can register six passes instead, and the slice accepts it at 0x1F with a 5-bit code against the honest 8
(REDTEAM phase 10, finding 2); under the digest published here, ``rows`` refuses it.

**The rows.** The server keeps each step's rows as the model runner laid them out (`verity_vllm.commit.served_record`'s
`SamplerInputsSource.rows`: the runner's request id, its token rows [lo, hi) of the step's flattened batch, its first
position and its prefill length). `step_rows` gives one step's in `replay_step`'s form, (slot, request, kind, start, n) in
batch order: the request its certified arrival id, n = hi − lo, kind decode iff n = 1 and start ≥ the prefill
(ReplayStep_v1's rule). A value wider than its field is refused, since `compile_slice.rows_bits` would truncate it. A window
is the slice's P steps in turn, each packed as `rows_bits` packs it (an idle step zero): `window_bits` is what the slice's
``rows`` hashes.

**The commitment** (`register`), made when the window closes and published with its run root before the advice check and
the draw:
- window w's digest SHA-512(salt ‖ bits) under a fresh 32-byte salt, the slice's ``pub.digest`` form; the salt is the
  slice witness's ``salt``;
- window w as row w of one hm96-sha512/row/v2 registration in a frame-v3-sha512 tree (`verity.ml.commitments.advice`'s
  construction), its row the window's bits as u16 words, as a proof commits a row, under `binding` of the run root.

`Served.record` is the public part: the root, the count, the row width and the digests. The rows and both salts stay the
prover's.

**The verifier's side.** `pub` is its own copy of a unit's ``pub`` row: the declared charge and window w's published
digest, never the prover's words. `leaf_ok` holds a committed row's b‖c to window w's registered leaf, as the advice's
``adv`` row is held, for a slice that reads the served rows as a committed row instead of hashing them.

That the served rows are the rows the served compute ran is the step tree's to bind (each row's start and n against the
step's committed positions and input ids); nothing here checks it.
"""
from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from verity.core.protocols.certifier import compile_slice as CS, replay_step as RS
from verity.ml.commitments import advice as ADV
from verity.ml.commitments.frame_v3 import BitRowLeaf
from verity.ml.commitments.rowleaf import bit_row, row_leaf

SCHEMA = "verity/certifier/served-rows/v1"
DOMAIN = b"verity/certifier/served-rows/v1"

Row = tuple[int, int, int, int, int]


def binding(run_root: bytes) -> bytes:
    """The registration's binding: the window's run root, so a registration can't be carried to another run."""
    return hashlib.sha512(DOMAIN + bytes(run_root)).digest()


def step_rows(S, served: Sequence[tuple], ids: Mapping[str, int]) -> tuple[Row, ...]:
    """One step's served rows (request, lo, hi, start, prefill_len), in batch order, as the slice's (slot, request, kind,
    start, n); `ids` maps a runner's request id to its certified arrival id."""
    Z = RS.widths(CS.step_statics(S))
    if len(served) > S.R:
        raise ValueError(f"a step of {len(served)} rows is over R = {S.R}")
    out = []
    for slot, (req, lo, hi, start, prefill) in enumerate(served):
        if req not in ids:
            raise ValueError(f"row {slot}: request {req!r} is no certified arrival")
        n = hi - lo
        row = (slot, ids[req], int(n == 1 and start >= prefill), start, n)
        for name, v, w in (("request", row[1], S.IW), ("start", start, Z.LW), ("n", n, Z.TW)):
            if not 0 <= v < 1 << w:
                raise ValueError(f"row {slot}: {name} {v} doesn't fit its {w} bits")
        out.append(row)
    return tuple(out)


def windows(S, by_step: Mapping[int, Sequence[tuple]], ids: Mapping[str, int], steps: int) -> list[list[tuple[Row, ...] | None]]:
    """The session's ``steps`` executed steps (`SamplerInputsSource.rows`' form, by step) in windows of P, a step with no
    rows idle (None)."""
    rows = [step_rows(S, by_step[u], ids) if by_step.get(u) else None for u in range(steps)]
    return [rows[k:k + S.P] for k in range(0, steps, S.P)]


def window_bits(S, window: Sequence[tuple[Row, ...] | None]) -> list[int]:
    if len(window) > S.P:
        raise ValueError(f"a window of {len(window)} steps is over P = {S.P}")
    return [b for s in list(window) + [None] * (S.P - len(window)) for b in CS.rows_bits(S, s)]


def _row(bits: Sequence[int]) -> tuple[bytes, int]:
    words = CS.words(list(bits) + [0] * (-len(bits) % CS.WORD))
    return bit_row((words, CS.WORD))[0], CS.WORD * len(words)


@dataclass(frozen=True)
class Served:
    """The prover's side: the registration (rows and their salts), the digests and their salts."""

    registration: ADV.Registration
    digest_salts: tuple[bytes, ...]
    digests: tuple[bytes, ...]

    def record(self) -> dict:
        r = self.registration
        return {"schema": SCHEMA, "binding": r.binding.hex(), "root": r.root.hex(), "count": len(r.rows),
                "n_bits": r.n_bits, "digests": [d.hex() for d in self.digests]}

    def bc(self, w: int) -> bytes:
        """Window w's row commitment b‖c, as a proof that holds the row under its salt commits it."""
        r = self.registration
        return ADV.INST.commit_string(r.key, BitRowLeaf(r.n_bits).layout().digest(r.rows[w]), r.salts[w])


def register(S, session: Sequence[Sequence[tuple[Row, ...] | None]], run_root: bytes, *,
             salts: Sequence[bytes] | None = None, digest_salts: Sequence[bytes] | None = None) -> Served:
    """Commit the session's windows (`windows`' form) under the run root; fresh salts unless given."""
    bits = [window_bits(S, w) for w in session]
    rows = [_row(b) for b in bits]
    ds = tuple(bytes(s) for s in digest_salts) if digest_salts is not None else tuple(os.urandom(CS.SALT) for _ in bits)
    if len(ds) != len(bits) or any(len(s) != CS.SALT for s in ds):
        raise ValueError(f"one {CS.SALT}-byte digest salt per window")
    reg = ADV.register([r for r, _n in rows], rows[0][1], binding(run_root), salts)
    return Served(reg, ds, tuple(hashlib.sha512(s + CS._bytes(b)).digest() for s, b in zip(ds, bits)))


def pub(S, record: Mapping, w: int, charge: int) -> list[int]:
    """The verifier's copy of window w's ``pub`` row (u16 words): the declared charge and the published digest."""
    return CS.words(CS._pack(S, "pub", {"c": [charge], "digest": list(bytes.fromhex(record["digests"][w]))}))


def leaf_ok(record: Mapping, run_root: bytes, w: int, bc: bytes, path: Sequence[bytes]) -> bool:
    """Whether b‖c, with `path`, is window w's row of the registration `record` publishes under the run root."""
    b, n, n_bits = binding(run_root), record["count"], record["n_bits"]
    if record["binding"] != b.hex() or not 0 <= w < n:
        return False
    leaf = row_leaf(ADV.domain(b, n), w, ADV.INST.tree_leaf(ADV.INST.default_key, bytes(bc)), BitRowLeaf(n_bits).hiding_schema)
    return ADV._scheme(b, n, n_bits).verify_path(ADV.PORT, bytes.fromhex(record["root"]), n, w, leaf, tuple(path))
