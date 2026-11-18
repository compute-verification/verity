"""PoUS's session units (`pous.prover.session_circuit`): the signature's unit agrees with `lms.verify` on a toy key's
signatures and on each with one thing wrong, the round's unit agrees with its reference on an honest round and on each
wrong row, record, digest, round and index, and both read the rows the memory attester writes."""

from __future__ import annotations

import json
import random

import numpy as np
import pytest

from verity.circuits import evaluation as E
from verity.circuits.defs import bind
from verity.circuits.evaluation.reference import resolve
from verity.circuits.ops.boolean import versions  # noqa: F401 - registers WordBits' Boolean bodies, which the round unit's row reads through
from verity.commitments.crypto import lms as R
from verity.commitments.rowleaf import sha512_row_v2_digest
from verity.protocols.resource_accounting.memory.pous.attester import memory_attester as mc
from verity.protocols.resource_accounting.memory.pous.attester.memory_attester import Round
from verity.protocols.resource_accounting.memory.pous.prover import session_circuit as S

RNG = random.Random(11)
#: a toy key's tree height: four leaves, so q's low bits and the top bits' check each refuse something
H = 2


def _bits(data: bytes) -> list[int]:
    return [(b >> k) & 1 for b in data for k in range(8)]


def _flip(b: bytes, at: int = 0, bit: int = 0) -> bytes:
    return b[:at] + bytes([b[at] ^ 1 << bit]) + b[at + 1:]


def _run(d, rows: list[list[list[int]]]) -> list[int]:
    cols = [np.array([r[a] for r in rows], dtype=np.uint8) for a in range(len(rows[0]))]
    return np.asarray(E.evaluate_batch(d, *cols)).reshape(-1).tolist()


def test_the_rows_at_the_deployed_scheme() -> None:
    """The deployed signature's unit reads the 80-byte key, the 64-byte message and the 34,756-byte signature, and the
    round's unit a 2,056-byte row as 1,028 words, a records row of two 92-byte records and the 12 bytes u32 j || u64 i;
    each outputs one bit."""
    p = R.DEPLOYED
    params, ret = resolve(bind(S.PousSessionSig, N=p.N, W=p.W, H=p.H)).signature()
    assert [(n, t.leaves) for n, t in params] == [("key", 640), ("m", 512), ("sig", 8 * p.signature_bytes())]
    assert ret.leaves == 1
    params, ret = resolve(bind(S.PousRound, N=2056, SLOT=1)).signature()
    assert [(n, t.leaves, t.leaf_widths()[0]) for n, t in params] == [("row", 1028, 16), ("records", 1472, 1),
                                                                        ("ji", 96, 1)]
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
    m = R.memory_attester_message(key.I, bytes(32), b"records")
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
    pair at the unit's slot, a flipped bit of the record's digest, the wrong round and the wrong index each give 0. The
    row is read as its words, each two bytes little-endian."""
    N = 136
    row, other = _rand(N), _rand(N)
    j, i = 2 * 5 + slot, 0x0123_4567_89AB
    mine = Round(j, i, 10, 20, sha512_row_v2_digest(row, 8 * N)).value()
    pair = Round(j ^ 1, 77, 11, 21, sha512_row_v2_digest(other, 8 * N)).value()
    recs = mine + pair if slot == 0 else pair + mine
    swapped = recs[S.RECORD_BYTES:] + recs[:S.RECORD_BYTES]
    ji = S.round_ji(j, i)
    cases = [(row, recs, ji), (other, recs, ji), (row, swapped, ji),
             (row, _flip(recs, S.RECORD_BYTES * slot + 60, 2), ji), (row, recs, S.round_ji(j ^ 1, i)),
             (row, recs, S.round_ji(j, i ^ 1 << 40))]
    want = [S.round_ok(N, slot, *c) for c in cases]
    assert want == [1, 0, 0, 0, 0, 0]
    cols = [np.array([np.frombuffer(r, dtype="<u2") for r, _, _ in cases], dtype=np.uint16),
            np.array([_bits(rs) for _, rs, _ in cases], dtype=np.uint8),
            np.array([_bits(x) for _, _, x in cases], dtype=np.uint8)]
    assert np.asarray(E.evaluate_batch(bind(S.PousRound, N=N, SLOT=slot), *cols)).reshape(-1).tolist() == want


def test_a_slot_past_the_row_is_refused() -> None:
    with pytest.raises(ValueError, match="SLOT"):
        resolve(bind(S.PousRound, N=2056, SLOT=2)).signature()


def _session(K: int, late: int | None = None):
    """A session of K rounds as the memory attester signs it: its header length and row, the signed context, the records
    rows and the digest the signature signs (`lms.memory_attester_message` of `mc.message`), round
    ``late``, if any, signed past the limit."""
    c = mc.SessionCommitment(_rand(32), _rand(64), _rand(32), 1 << 20, K, 854_000, 2056, _rand(64))
    rounds = [Round(j, RNG.randrange(1 << 20), 1000 * j, 1000 * j + 500, _rand(64)) for j in range(K)]
    if late is not None:
        rounds[late] = Round(late, rounds[late].index, 0, c.limit_ns + 1, rounds[late].digest)
    ident = _rand(16)
    m = R.memory_attester_message(ident, c.session, mc.message(c, mc.COMPLETE, rounds))
    head = json.dumps({**c.header(), "ended": mc.COMPLETE, "rounds": K}, sort_keys=True, separators=(",", ":")).encode()
    flat = b"".join(r.value() for r in rounds) + bytes(S.RECORD_BYTES * (K % 2))
    rows = [flat[184 * t:184 * (t + 1)] for t in range(S.records_rows(K))]
    return c, len(head), head + bytes(S.head_bytes(len(head)) - len(head)), S.signed_context(ident, c.session), rows, m


def test_the_signed_digest_reference_is_the_memory_attesters() -> None:
    """`session_digest` gives the digest the memory attester's signature signs, at an odd and an even K and at a K
    past one record per compression, the constant it copies is the memory attester's, and a late round's ok is 0."""
    assert S.SESSION_TAG == mc.SESSION_TAG and S.RECORD_BYTES == mc.RECORD_BYTES
    for K in (1, 4, 140):
        c, n, head, ctx, rows, m = _session(K)
        assert S.session_digest(K, n, c.limit_ns, head, ctx, rows) == (1, m, tuple(rows))
    c, n, head, ctx, rows, m = _session(3, late=1)
    assert S.session_digest(3, n, c.limit_ns, head, ctx, rows) == (0, m, tuple(rows))
    with pytest.raises(ValueError, match="K"):
        resolve(bind(S.PousSessionDigest, K=0, HEAD=753, LIMIT=1)).signature()


@pytest.mark.slow
def test_the_signed_digest_unit_agrees_with_its_reference() -> None:
    """About 30 s (the records' stream digest, three compressions, and the two message hashes, at K = 3). On the honest
    session the unit's ok is 1, its m the digest the signature signs and its records rows the rows it read; a flipped bit
    of the context, of the header or of a record's digest changes m, and a late round sets ok to 0."""
    K = 3
    c, n, head, ctx, rows, m = _session(K)
    late = _session(K, late=2)
    cases = [(head, ctx, rows), (head, _flip(ctx, 50), rows), (_flip(head, 100), ctx, rows),
             (head, ctx, [_flip(rows[0], 92 + 40)] + rows[1:])]
    want = [S.session_digest(K, n, c.limit_ns, *cs) for cs in cases]
    assert want[0] == (1, m, tuple(rows)) and all(w[1] != m for w in want[1:])
    assert late[1] == n
    cases.append(late[2:5])
    want.append(S.session_digest(K, n, c.limit_ns, *cases[-1]))
    assert want[-1][0] == 0
    params = [[_bits(h), _bits(x), *[_bits(r) for r in rs]] for h, x, rs in cases]
    cols = [np.array([p[a] for p in params], dtype=np.uint8) for a in range(len(params[0]))]
    out = np.asarray(E.evaluate_batch(bind(S.PousSessionDigest, K=K, HEAD=n, LIMIT=c.limit_ns), *cols))
    assert out.reshape(len(cases), -1).tolist() == [[ok, *_bits(mm), *(b for r in rs for b in _bits(r))]
                                                     for ok, mm, rs in want]


@pytest.mark.slow
def test_the_door_takes_the_sessions_units_in_wire_order() -> None:
    """The session's units as `window.session_units` lays them out (digest, signature, the store's openings, the rounds)
    make one Program through the door (`wired_population_program`), and the signature ranked before the digest it reads
    is refused. Building the Program takes about two minutes at toy size; nothing cheaper runs the door's rank and port
    rules over these units."""
    from verity.commitments.periods.interface import Circuit
    from verity.protocols.resource_accounting.memory.pous.prover import window as W

    c = W.session_units(3, 8, 1000, 264, 2, R.Params(R.TOY_N, 1, 1, toy=True))
    assert [t.split("{")[0] for t in c.ids] == ["PousSessionDigest_v1", "PousSessionSig_v1", "EmbeddingRead_v2",
                                                "PousRound_v2", "PousRound_v2"]
    assert list(c.units.values()) == [1, 1, 3, 2, 1]
    c.program()
    (dig, _), (sig, _), *rest = c.templates
    with pytest.raises(ValueError, match="at or after reader"):
        Circuit("pous-session", ((sig, 1), (dig, 1), *rest), wires=c.wires, registered=c.registered).program()
