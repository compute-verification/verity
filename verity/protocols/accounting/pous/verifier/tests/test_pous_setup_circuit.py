"""PoUS's setup unit (`pous.setup_circuit`) against `schemes.p2`'s decode and the row's slot (`setup_circuit.slot_payload`)."""

from __future__ import annotations

import random

import numpy as np
import pytest
import settings

from verity.circuits.defs import bind
from verity.circuits.evaluation import bits as EB
from verity.protocols.accounting.pous.prover import decode_circuit as D
from verity.protocols.accounting.pous.prover import setup_circuit as S
from verity.protocols.accounting.pous.verifier.schemes import p2 as P2

RNG = random.Random(17)
PARAMS = P2.P2Params(64, 189, 4)
PB = PARAMS.payload_bytes
SALT = bytes(RNG.getrandbits(8) for _ in range(24))
PREFIX = P2.expand_prefix(PARAMS, SALT)
L = len(PREFIX) + 16


def _bits(rows: list[bytes]) -> np.ndarray:
    return np.array([[(b >> k) & 1 for b in row for k in range(8)] for row in rows], dtype=np.uint8)


def _bytes(row) -> bytes:
    return bytes(int(sum(int(row[8 * i + k]) << k for k in range(8))) for i in range(len(row) // 8))


def _slots(ks: list[int], width: int) -> np.ndarray:
    return np.array([[k >> i & 1 for i in range(width)] for k in ks], dtype=np.uint8)


def _base(j: int, s: int = 0) -> bytes:
    return PREFIX + s.to_bytes(8, "big") + j.to_bytes(8, "big")


def _decode(j: int, block: bytes) -> bytes | None:
    """`schemes.p2`'s decode of block j of segment 0 (`decode_segment`'s step): its payload, or None where it refuses."""
    f, c = P2._field(PARAMS), int.from_bytes(block, "big")
    k, t = P2.sha512_keys(PARAMS, PREFIX, 0, j)
    m = (f.sq(f.sigma(f.sq(c), k)) - t) % f.p
    return None if c >= f.p or m >> 8 * PB else m.to_bytes(PB, "big")


def _store(row: bytes) -> list[bytes]:
    """The row's blocks of C: each slot's payload encoded as block j = slot of segment 0."""
    n = S.slots(len(row), PB)
    return P2.encode_blocks(PARAMS, SALT, 0, 0, b"".join(S.slot_payload(row, k, PB) for k in range(n)))


def _unit(N: int, cases: list[tuple[bytes, int, bytes, int]]) -> list[int]:
    """The unit's bit for each (block, j, row, slot)."""
    fn = bind(S.PousP2SetupCheck, W=PARAMS.w, C0=PARAMS.c0, L=L, N=N)
    out = EB.evaluate_bits(fn, _bits([c[0] for c in cases]), _bits([_base(c[1]) for c in cases]),
                           _bits([c[2] for c in cases]), _slots([c[3] for c in cases], S.slot_bits(N, PB)))
    return [int(r[0]) for r in out]


def _reference(block: bytes, j: int, row: bytes, slot: int) -> int:
    payload = _decode(j, block)
    return int(payload is not None and payload == S.slot_payload(row, slot, PB))


@pytest.mark.parametrize("N", [4, 12, 15])
def test_honest_blocks_give_1(N):
    """Each slot of a row shorter than a payload (one slot, zero tail), an exact fit (two) and three slots with a zero
    tail: its block of C, under its own base, gives 1 at its slot."""
    row = bytes(RNG.getrandbits(8) for _ in range(N))
    blocks = _store(row)
    assert _unit(N, [(b, k, row, k) for k, b in enumerate(blocks)]) == [1] * len(blocks)


def test_each_wrong_input_gives_0():
    """A wrong block, a wrong byte of W in the slot, a wrong slot, a slot past the row's last, a wrong base and a nonzero
    tail past the row's end each give 0; a changed byte of W in another slot does not."""
    N = 15
    row = bytes(RNG.getrandbits(8) for _ in range(N))
    blocks = _store(row)
    flip = lambda b, i: b[:i] + bytes([b[i] ^ 1]) + b[i + 1:]  # noqa: E731
    tail = bytearray(S.slot_payload(row, 2, PB))
    tail[-1] = 1
    (bad_tail,) = P2.encode_blocks(PARAMS, SALT, 0, 2, bytes(tail))
    cases = {
        "honest": ((blocks[1], 1, row, 1), 1),
        "wrong block": ((flip(blocks[1], 5), 1, row, 1), 0),
        "wrong byte of W": ((blocks[1], 1, flip(row, PB + 2), 1), 0),
        "other slot's byte": ((blocks[1], 1, flip(row, 2), 1), 1),
        "wrong slot": ((blocks[1], 1, row, 0), 0),
        "slot past the row": ((blocks[1], 1, row, 3), 0),
        "wrong base": ((blocks[1], 2, row, 1), 0),
        "nonzero tail": ((bad_tail, 2, row, 2), 0),
    }
    got = _unit(N, [c for c, _ in cases.values()])
    assert dict(zip(cases, got)) == {k: want for k, (_, want) in cases.items()}
    assert got == [_reference(*c) for c, _ in cases.values()]


def test_a_decode_that_is_not_ok_gives_0():
    """A block at or above p, with W's row holding exactly the payload the circuit's decode computes for it: only ok
    tells them apart, and the unit gives 0."""
    f = P2._field(PARAMS)
    blocks = [c.to_bytes(PARAMS.block_bytes, "big") for c in (f.p, f.p + 1, (1 << PARAMS.w) - 1)]
    dec = EB.evaluate_bits(bind(D.PousP2Decode, W=PARAMS.w, C0=PARAMS.c0, L=L), _bits(blocks), _bits([_base(0)] * 3))
    assert dec[:, -1].tolist() == [0, 0, 0]
    rows = [_bytes(r[:-1]) for r in dec]
    assert _unit(PB, [(b, 0, r, 0) for b, r in zip(blocks, rows)]) == [0, 0, 0]


def test_unit_is_the_reference_on_random_inputs():
    """Random blocks, rows and slots beside honest ones, over every slot value the slot's bits hold."""
    N = 15
    rows = [bytes(RNG.getrandbits(8) for _ in range(N)) for _ in range(4)]
    cases = [(b, k, row, k) for row in rows for k, b in enumerate(_store(row))]
    cases += [(RNG.getrandbits(PARAMS.w).to_bytes(PARAMS.block_bytes, "big"), RNG.randrange(3), RNG.choice(rows),
               RNG.randrange(4)) for _ in range(12)]
    cases += [(b, k, row, RNG.randrange(4)) for row in rows[:2] for k, b in enumerate(_store(row))]
    assert _unit(N, cases) == [_reference(*c) for c in cases]


@pytest.mark.parametrize(("P", "N"), [(6, 4), (6, 6), (6, 13), (6, 24), (6, 31)])
def test_slot_match_is_slot_payload(P, N):
    """The slot's selection and compare alone, on payloads equal to a slot's (ok set or not) and payloads one bit off,
    over every slot value: 1 iff ok, the slot is the row's and the payload is `slot_payload`'s."""
    fn = bind(S.PousP2SlotMatch, P=P, N=N)
    width = S.slot_bits(N, P)
    row = bytes(RNG.getrandbits(8) for _ in range(N))
    cases = []
    for k in range(1 << width):
        want = S.slot_payload(row, k, P) or bytes(RNG.getrandbits(8) for _ in range(P))
        for payload in (want, want[:-1] + bytes([want[-1] ^ 0x80])):
            cases += [(payload, ok, k) for ok in (0, 1)]
    decoded = np.concatenate([_bits([c[0] for c in cases]), np.array([[c[1]] for c in cases], dtype=np.uint8)], axis=1)
    out = EB.evaluate_bits(fn, decoded, _bits([row] * len(cases)), _slots([c[2] for c in cases], width))
    assert [int(r[0]) for r in out] == [int(ok == 1 and payload == S.slot_payload(row, k, P)) for payload, ok, k in cases]


def test_slot_match_counts_at_p2():
    """The unit's ANDs beyond the decode at P2's payload, per row width served: int8 rows of 2,048 and 8,192 bytes, bf16
    rows of 4,096 and 16,384 (one, four, two and eight slots); a multiplexer merge per slot past the first, the compare
    and the slot's bound."""
    pb = (P2.POINT[0] - 16) // 8
    got = {N: EB.compile(bind(S.PousP2SlotMatch, P=pb, N=N)).counts["and"] for N in (2048, 4096, 8192, 16384)}
    assert got == {2048: 16433, 4096: 32864, 8192: 65728, 16384: 131456}


def test_refuses_what_the_decode_refuses():
    with pytest.raises(ValueError, match="whole number of bytes"):
        bind(S.PousP2SetupCheck, W=60, C0=189, L=L, N=15).signature()
    with pytest.raises(ValueError, match="at most"):
        bind(S.PousP2SetupCheck, W=64, C0=189, L=128 + D.MAX_REST + 1, N=15).signature()
    with pytest.raises(ValueError, match="N is an int"):
        bind(S.PousP2SetupCheck, W=64, C0=189, L=L, N=0).signature()


SLOW = pytest.mark.skipif(not settings.SLOW, reason="minutes at P2's width: VERITY_POUS_SLOW=1")


@SLOW
def test_counts_at_p2():
    """The unit at P2's point over a bf16 row of 16,384 bytes is the decode's ANDs and the slot match's, nothing more:
    the composition adds no gate, which the slot match's own count can't show."""
    at = dict(W=P2.POINT[0], C0=P2.POINT[1], L=D.P2_BASE_BYTES)
    assert EB.compile(bind(S.PousP2SetupCheck, N=16384, **at)).counts["and"] == 31960530 + 131456
