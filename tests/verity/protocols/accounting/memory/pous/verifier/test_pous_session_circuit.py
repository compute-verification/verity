"""PoUS's session units (`pous.prover.session_circuit`): the signature's unit agrees with `lms.verify` on a toy key's
signatures and on each with one thing wrong, the round's unit agrees with its reference on an honest round and on each
wrong row, record, digest, round and index, the signed digest's unit holds the salted m to the records and refuses each
message that isn't theirs, and each reads the rows the memory attester writes."""

from __future__ import annotations

import hashlib
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
from verity.commitments import session_coins as SC
from verity.protocols.accounting.memory.pous.attester import memory_attester as mc
from verity.protocols.accounting.memory.pous.attester.memory_attester import Round
from verity.protocols.accounting.memory.pous.prover import session_circuit as S

RNG = random.Random(11)
#: a toy key's tree height: four leaves, so q's low bits and the top bits' check each refuse something
H = 2


def _bits(data: bytes) -> list[int]:
    return [(b >> k) & 1 for b in data for k in range(8)]


def _words(data: bytes) -> list[int]:
    return np.frombuffer(data, dtype="<u2").tolist()


def _flip(b: bytes, at: int = 0, bit: int = 0) -> bytes:
    return b[:at] + bytes([b[at] ^ 1 << bit]) + b[at + 1:]


def _run(d, rows: list[list[list[int]]]) -> list[int]:
    cols = [np.array([r[a] for r in rows], dtype=np.uint8) for a in range(len(rows[0]))]
    return np.asarray(E.evaluate_batch(d, *cols)).reshape(-1).tolist()


def test_the_rows_at_the_deployed_scheme() -> None:
    """The deployed signature's unit reads the 80-byte key, the 64-byte message and the 34,756-byte signature, and the
    round's unit a 2,056-byte row as 1,028 words, its opening's verdict word, a records row of two 92-byte records as 92 words
    and the 12 bytes u32 j || u64 i; the signed digest's unit its header, context, salt, m and records rows as words. Each
    outputs one verdict word first."""
    p = R.DEPLOYED
    params, ret = resolve(bind(S.PousSessionSig, N=p.N, W=p.W, H=p.H)).signature()
    assert [(n, t.leaves) for n, t in params] == [("key", 640), ("m", 512), ("sig", 8 * p.signature_bytes())]
    assert ret.leaf_widths() == [16]
    params, ret = resolve(bind(S.PousRound, N=2056, SLOT=1)).signature()
    assert [(n, t.leaves, t.leaf_widths()[0]) for n, t in params] == [("row", 1028, 16), ("ok", 1, 16), ("records", 92, 16),
                                                                        ("j", 1, 32), ("ihi", 1, 32),
                                                                        ("ilo", 1, 32)]
    assert ret.leaf_widths() == [16]
    params, ret = resolve(bind(S.PousSessionDigest, K=3, HEAD=753, LIMIT=1)).signature()
    assert [(n, t.leaves, set(t.leaf_widths())) for n, t in params] == [
        ("head", 377, {16}), ("ctx", 53, {16}), ("salt", 32, {16}), ("m", 32, {16}), ("r0", 92, {16}), ("r1", 92, {16})]
    assert ret.leaf_widths() == [16] * (1 + 2 * 92)
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
    pair at the unit's slot, a flipped bit of the record's digest, the wrong round and the wrong index each give 0; so do
    an opening's verdict word of 0 (a key past the store or a row that doesn't open) and an index past 2^32, its ihi
    half nonzero, though the record holds it. The row is read as its words, each two bytes little-endian."""
    N = 136
    row, other = _rand(N), _rand(N)
    j, i = 2 * 5 + slot, 0x89AB_CDEF
    mine = Round(j, i, 10, 20, sha512_row_v2_digest(row, 8 * N)).value()
    pair = Round(j ^ 1, 77, 11, 21, sha512_row_v2_digest(other, 8 * N)).value()
    recs = mine + pair if slot == 0 else pair + mine
    swapped = recs[S.RECORD_BYTES:] + recs[:S.RECORD_BYTES]
    def coin(j: int, i: int) -> bytes:
        return j.to_bytes(4, "big") + i.to_bytes(8, "big")

    ji = coin(j, i)
    wide = Round(j, i | 1 << 40, 10, 20, sha512_row_v2_digest(row, 8 * N)).value()
    cases = [(row, recs, ji, 1), (other, recs, ji, 1), (row, swapped, ji, 1),
             (row, _flip(recs, S.RECORD_BYTES * slot + 60, 2), ji, 1), (row, recs, coin(j ^ 1, i), 1),
             (row, recs, coin(j, i ^ 1 << 40), 1), (row, recs, coin(j, i ^ 1 << 3), 1), (row, recs, ji, 0),
             (row, recs, ji, 1 << 8), (row, wide + pair if slot == 0 else pair + wide, coin(j, i | 1 << 40), 1)]
    want = [S.round_ok(N, slot, *c) for c in cases]
    assert want == [1] + [0] * 9
    halves = [[int.from_bytes(x[h:h + 4], "big") for h in (0, 4, 8)] for _, _, x, _ in cases]
    cols = [np.array([np.frombuffer(r, dtype="<u2") for r, _, _, _ in cases], dtype=np.uint16),
            np.array([ok for *_, ok in cases], dtype=np.uint16),
            np.array([_words(rs) for _, rs, _, _ in cases], dtype=np.uint16),
            *(np.array([hs[h] for hs in halves], dtype=np.uint32) for h in range(3))]
    assert np.asarray(E.evaluate_batch(bind(S.PousRound, N=N, SLOT=slot), *cols)).reshape(-1).tolist() == want


def test_a_slot_past_the_row_is_refused() -> None:
    with pytest.raises(ValueError, match="SLOT"):
        resolve(bind(S.PousRound, N=2056, SLOT=2)).signature()


def _session(K: int, late: int | None = None):
    """A session of K rounds as the memory attester signs it: its header length and row, the signed context, the record's
    salt, the records rows, the rounds and the message the signature signs (`lms.memory_attester_message` of
    `mc.message`), round ``late``, if any, signed past the limit."""
    c = mc.SessionCommitment(_rand(32), _rand(64), _rand(32), 1 << 20, K, 854_000, 2056, _rand(64))
    rounds = [Round(j, RNG.randrange(1 << 20), 1000 * j, 1000 * j + 500, _rand(64)) for j in range(K)]
    if late is not None:
        rounds[late] = Round(late, rounds[late].index, 0, c.limit_ns + 1, rounds[late].digest)
    ident, salt = _rand(16), _rand(mc.RECORD_SALT_BYTES)
    m = R.memory_attester_message(ident, c.session, mc.message(c, salt, SC.COMPLETE, rounds))
    head = json.dumps({**c.header(), "ended": SC.COMPLETE, "rounds": K}, sort_keys=True, separators=(",", ":")).encode()
    rows = _rows(rounds)
    return (c, len(head), head + bytes(S.head_bytes(len(head)) - len(head)), S.signed_context(ident, c.session), salt, rows,
            rounds, m)


def _rows(rounds) -> list[bytes]:
    flat = b"".join(r.value() for r in rounds) + bytes(S.RECORD_BYTES * (len(rounds) % 2))
    return [flat[184 * t:184 * (t + 1)] for t in range(S.records_rows(len(rounds)))]


def _unsalted(c, ident: bytes, rounds) -> bytes:
    """The message under the unsalted tag, without a salt: what the memory attester signed before its record was salted."""
    head = json.dumps({**c.header(), "ended": SC.COMPLETE, "rounds": len(rounds)}, sort_keys=True,
                      separators=(",", ":")).encode()
    record = hashlib.sha512(b"verity-pous/memory-challenger/session/v1\0" + len(head).to_bytes(8, "big") + head
                            + mc.records_digest(rounds)).digest()
    return R.memory_attester_message(ident, c.session, record)


def _negatives(K: int):
    """The honest session's digest unit's inputs, and beside them each with one thing that isn't the signed session's: a
    record one word off, another salt, another m, and the unsalted message; each by name."""
    c, n, head, ctx, salt, rows, rounds, m = _session(K)
    word = _flip(rows[0], 2 * 7)
    unsalted = _unsalted(c, ctx[8 + len(S._ROLE):8 + len(S._ROLE) + 16], rounds)
    return c, n, m, {
        "honest": (head, ctx, salt, m, rows),
        "a record one word off": (head, ctx, salt, m, [word, *rows[1:]]),
        "another salt": (head, ctx, _flip(salt, 5), m, rows),
        "another m": (head, ctx, salt, _flip(m, 63, 7), rows),
        "the unsalted message": (head, ctx, salt, unsalted, rows),
    }


def test_the_signed_digest_reference_is_the_memory_attesters() -> None:
    """`session_message` gives the message the memory attester's signature signs, at an odd and an even K and at a K past
    one record per compression, and `session_digest` accepts it; the constants it copies are the memory attester's; a late
    round's ok is 0; and a record one word off, another salt, another m and the unsalted message each give 0."""
    assert S.SESSION_TAG == mc.SESSION_TAG and S.RECORD_BYTES == mc.RECORD_BYTES and S.SALT_BYTES == mc.RECORD_SALT_BYTES
    for K in (1, 4, 140):
        c, n, head, ctx, salt, rows, _rounds, m = _session(K)
        assert S.session_message(K, n, head, ctx, salt, rows) == m
        assert S.session_digest(K, n, c.limit_ns, head, ctx, salt, m, rows) == (1, tuple(rows))
    c, n, head, ctx, salt, rows, _rounds, m = _session(3, late=1)
    assert S.session_digest(3, n, c.limit_ns, head, ctx, salt, m, rows) == (0, tuple(rows))
    c, n, m, cases = _negatives(3)
    assert {k: S.session_digest(3, n, c.limit_ns, *v)[0] for k, v in cases.items()} == {
        "honest": 1, "a record one word off": 0, "another salt": 0, "another m": 0, "the unsalted message": 0}
    with pytest.raises(ValueError, match="K"):
        resolve(bind(S.PousSessionDigest, K=0, HEAD=753, LIMIT=1)).signature()
    with pytest.raises(ValueError, match="64-byte salt"):
        S.session_digest(3, n, c.limit_ns, cases["honest"][0], cases["honest"][1], bytes(63), m, cases["honest"][4])


@pytest.mark.slow
def test_the_signed_digest_unit_agrees_with_its_reference() -> None:
    """About 30 s (the records' stream digest, three compressions, and the two message hashes, at K = 3). On the honest
    session the unit's ok is 1 and its records rows the rows it read; a record one word off, another salt, another m, the
    unsalted message, a flipped bit of the context or of the header, and a late round each set ok to 0, exactly where
    `session_digest` does."""
    K = 3
    c, n, m, named = _negatives(K)
    cases = list(named.values())
    head, ctx, salt, _, rows = named["honest"]
    cases += [(head, _flip(ctx, 50), salt, m, rows), (_flip(head, 100), ctx, salt, m, rows)]
    late = _session(K, late=2)
    assert late[1] == n
    cases.append((late[2], late[3], late[4], late[7], late[5]))
    want = [S.session_digest(K, n, c.limit_ns, *cs) for cs in cases]
    assert [w[0] for w in want] == [1] + [0] * (len(cases) - 1)
    params = [[_words(h), _words(x), _words(sl), _words(mm), *[_words(r) for r in rs]] for h, x, sl, mm, rs in cases]
    cols = [np.array([p[a] for p in params], dtype=np.uint16) for a in range(len(params[0]))]
    out = np.asarray(E.evaluate_batch(bind(S.PousSessionDigest, K=K, HEAD=n, LIMIT=c.limit_ns), *cols))
    assert out.reshape(len(cases), -1).tolist() == [[ok, *(w for r in rs for w in _words(r))] for ok, rs in want]


@pytest.mark.slow
def test_the_door_takes_the_sessions_units_in_wire_order() -> None:
    """The session's units as `window.session_units` lays them out (the digest, the store's openings, the rounds), their
    coins, header, context and message supplied under their marks and the salt a registered read, make one Program
    through the door (`wired_population_program`), and a round ranked before the digest it reads is refused. Building the Program takes about two minutes at toy size; nothing cheaper runs the door's rank and port
    rules over these units."""
    from verity.commitments.periods.interface import Circuit
    from verity.protocols.accounting.memory.pous.prover import window as W

    c = W.session_units(3, 8, 1000, 264, 2)
    assert [t.split("{")[0] for t in c.ids] == ["PousSessionDigest_v2", "EmbeddingRead_v3", "PousRound_v3", "PousRound_v3"]
    assert list(c.units.values()) == [1, 3, 2, 1]
    o = c.ids[1]
    assert sorted((w.radix, w.base, w.step) for w in c.wires if (w.port, w.writer, w.out) == ("ok", o, "o1")) == [
        ((1,), 1, (2,)), ((2,), 0, (2,))]
    reads = {(r.port, r.param): (r.at, r.base) for r in c.registered.reads}
    assert [reads[("tok", f"{W.ROUND_COIN}/{j}")] for j in range(3)] == [(0, 4), (1, 4), (2, 4)]
    assert [reads[(h, f"{W.ROUND_COIN}/2")] for h in ("j", "ihi", "ilo")] == [(1, 0), (1, 2), (1, 4)]
    assert c.registered.supplied[f"{W.ROUND_COIN}/2"] == {"supplied": "pous.session.round_coin/v1", "args": {"round": 2},
                                                           "layout": "halves", "bits": 96}
    assert sorted(c.registered.supplied) == sorted([W.SESSION_CONTEXT, W.SESSION_HEADER, W.SESSION_MESSAGE,
                                                    *(f"{W.ROUND_COIN}/{j}" for j in range(3))])
    assert reads[("salt", W.SESSION_SALT)] == (0, 0) and reads[("m", W.SESSION_MESSAGE)] == (0, 0)
    c.program()
    (dig, _), opener, *rounds = c.templates
    with pytest.raises(ValueError, match="at or after reader"):
        Circuit("pous-session", (opener, *rounds, (dig, 1)), wires=c.wires, registered=c.registered).program()
