"""PoUS's session units (`pous.prover.session_circuit`): the signature's unit agrees with `lms.verify` on a toy key's
signatures and on each with one thing wrong, the round's unit agrees with its reference on an honest round and on each
wrong row, record, digest, round and index, and both read the rows the memory challenger writes."""

from __future__ import annotations

import random

import numpy as np
import pytest

from verity.circuits import evaluation as E
from verity.circuits.defs import bind
from verity.circuits.evaluation.reference import resolve
from verity.commitments.crypto import lms as R
from verity.commitments.rowleaf import sha512_row_v2_digest
from verity.protocols.accounting.pous.device.memory_challenger import Round
from verity.protocols.accounting.pous.prover import session_circuit as S

RNG = random.Random(11)
#: a toy key's tree height: four leaves, so q's low bits and the top bits' check each refuse something
H = 2


def _bits(data: bytes) -> list[int]:
    return [(b >> k) & 1 for b in data for k in range(8)]


def _int(v: int, width: int) -> list[int]:
    return [(v >> t) & 1 for t in range(width)]


def _flip(b: bytes, at: int = 0, bit: int = 0) -> bytes:
    return b[:at] + bytes([b[at] ^ 1 << bit]) + b[at + 1:]


def _run(d, rows: list[list[list[int]]]) -> list[int]:
    cols = [np.array([r[a] for r in rows], dtype=np.uint8) for a in range(len(rows[0]))]
    return np.asarray(E.evaluate_batch(d, *cols)).reshape(-1).tolist()


def test_the_rows_at_the_deployed_scheme() -> None:
    """The deployed signature's unit reads the 80-byte key, the 64-byte message and the 34,756-byte signature, and the
    round's unit a 2,056-byte row, a records row of two 92-byte records and the public u32 j and u64 index; each outputs
    one bit."""
    p = R.DEPLOYED
    params, ret = resolve(bind(S.PousSessionSig, N=p.N, W=p.W, H=p.H)).signature()
    assert [(n, t.leaves) for n, t in params] == [("key", 640), ("m", 512), ("sig", 8 * p.signature_bytes())]
    assert ret.leaves == 1
    params, ret = resolve(bind(S.PousRound, N=2056, SLOT=1)).signature()
    assert [(n, t.leaves) for n, t in params] == [("row", 16448), ("records", 1472), ("j", 32), ("i", 64)]
    assert ret.leaves == 1
    assert len(Round(0, 0, 0, 0, bytes(64)).value()) == S.RECORD_BYTES


@pytest.mark.slow
def test_the_signature_unit_agrees_with_lms_verify() -> None:
    """About 20 s (the toy verifier's first evaluation). A toy key's signature (N = 8, `lms.TOY_N`) verifies; a wrong
    message, a wrong I, a wrong root, a wrong q, a q past 2^H whose low bits are the signature's own, a set top bit of q, and
    a flipped bit of C, of a chain value or of a path node each give 0, exactly where `lms.verify` refuses. The deployed
    N = 64 runs the same wiring on eight words a value; `gates.lms`' tests check the verifier itself at N = 64."""
    p = R.Params(R.TOY_N, 1, H, toy=True)
    key = R.keygen(p, bytes(range(R.TOY_N)))
    pk = R.public_key(key)
    m = R.memory_challenger_message(key.I, bytes(32), b"records")
    sig = R.sign_at(key, 1, m).encode()
    q = lambda v: v.to_bytes(4, "big") + sig[4:]  # noqa: E731
    n = p.N
    cases = [(pk, m, sig), (pk, _flip(m, 9), sig), (_flip(pk, 3), m, sig), (_flip(pk, 16 + n - 1, 7), m, sig),
             (pk, m, q(0)), (pk, m, q(1 | 1 << H)), (pk, m, q(1 | 1 << 31)), (pk, m, _flip(sig, 4 + 2, 5)),
             (pk, m, _flip(sig, 4 + n * 7)), (pk, m, _flip(sig, len(sig) - 1, 3))]
    want = [S.session_sig_ok(n, 1, H, *c) for c in cases]
    assert want == [1] + [0] * (len(cases) - 1)
    assert _run(bind(S.PousSessionSig, N=n, W=1, H=H), [[_bits(x) for x in c] for c in cases]) == want


def _rand(n: int) -> bytes:
    return bytes(RNG.getrandbits(8) for _ in range(n))


@pytest.mark.slow
@pytest.mark.parametrize("slot", [0, 1])
def test_the_round_unit_agrees_with_its_reference(slot) -> None:
    """About 9 s at the first slot (the row digest's ends traced over a 136-byte row: a head and a tail with its rest).
    Round j's record at slot j mod 2 of its row, holding the row's digest, gives 1; another row, the other record of the
    pair at the unit's slot, a flipped bit of the record's digest, the wrong round and the wrong index each give 0."""
    N = 136
    row, other = _rand(N), _rand(N)
    j, i = 2 * 5 + slot, 0x0123_4567_89AB
    mine = Round(j, i, 10, 20, sha512_row_v2_digest(row, 8 * N)).value()
    pair = Round(j ^ 1, 77, 11, 21, sha512_row_v2_digest(other, 8 * N)).value()
    recs = mine + pair if slot == 0 else pair + mine
    swapped = recs[S.RECORD_BYTES:] + recs[:S.RECORD_BYTES]
    cases = [(row, recs, j, i), (other, recs, j, i), (row, swapped, j, i),
             (row, _flip(recs, S.RECORD_BYTES * slot + 60, 2), j, i), (row, recs, j ^ 1, i), (row, recs, j, i ^ 1 << 40)]
    want = [S.round_ok(N, slot, *c) for c in cases]
    assert want == [1, 0, 0, 0, 0, 0]
    rows = [[_bits(r), _bits(rs), _int(jj, 32), _int(ii, 64)] for r, rs, jj, ii in cases]
    assert _run(bind(S.PousRound, N=N, SLOT=slot), rows) == want


def test_a_slot_past_the_row_is_refused() -> None:
    with pytest.raises(ValueError, match="SLOT"):
        resolve(bind(S.PousRound, N=2056, SLOT=2)).signature()
