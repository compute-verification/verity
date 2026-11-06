"""`verity.ml.commitments.gates.sha512_stream`: a stream's rows and its last row in gates give `hashlib.sha512` of the stream,
at an odd header (every row's carry ending in a pad byte), at whole-block phases (no carry), with rows of several blocks and
a last row whose end is one block and two; a flipped bit, two rows swapped and another length each change the digest; the
signatures carry a carry exactly where bytes pend; malformed bindings and states are refused; and a binding costs its
compressions' `Sha512Compress_v1` ANDs and nothing else."""
from __future__ import annotations

import hashlib
import random

import numpy as np
import pytest

from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation.bits import compile as compile_bits
from verity.ml.circuits.evaluation.bits import evaluate_bits
from verity.ml.circuits.types import Tuple
from verity.ml.commitments.gates import sha512_stream as S
from verity.ml.commitments.gates.lms import Sha512Compress
from verity.ml.commitments.gates.sha512 import IV

#: `Sha512Compress_v1`'s And gates (`test_boolean_lms.PARTS`)
COMPRESS_ANDS = 57947


def _words(row: bytes) -> list[int]:
    return np.frombuffer(row, "<u2").tolist()


def _unpack(fn, out: np.ndarray, rest: int) -> tuple[list[int], bytes]:
    """A row's or the last row's output bits as (cv, pending bytes)."""
    flat = out.reshape(-1)
    cv = [sum(int(flat[64 * k + j]) << j for j in range(64)) for k in range(8)]
    words = [sum(int(flat[512 + 16 * k + j]) << j for j in range(16)) for k in range((rest + 1) // 2)]
    return cv, b"".join(w.to_bytes(2, "little") for w in words)[:rest]


def _args(cv: list[int], pend: bytes, data: list[int]) -> list[np.ndarray]:
    return [*[np.array([a], dtype=np.uint8) for a in S.state_bits(cv, pend)],
            np.array([[(w >> j) & 1 for w in data for j in range(16)]], dtype=np.uint8)]


def _digest(out: np.ndarray) -> bytes:
    return b"".join(v.to_bytes(8, "big") for v in _unpack(None, out, 0)[0])


def _gates_stream(msg: bytes, start: int, width: int, rows: int) -> tuple[bytes, list[tuple[list[int], bytes]]]:
    """``msg``'s first ``start`` bytes as the state the verifier computes, then ``rows`` rows of ``width`` words, each but
    the last through `Sha512StreamRow_v1` and the last through `Sha512StreamLast_v1`: (the digest, each row's state in
    gates but the last's)."""
    cv, pend = S.state(msg[:start])
    at, states = start, []
    for _ in range(rows - 1):
        data = _words(msg[at:at + 2 * width])
        fn = S.row(len(pend), width)
        rest = (len(pend) + 2 * width) % 128
        out = evaluate_bits(fn, *_args(cv, pend, data))
        got = _unpack(fn, out, rest)
        if not rest:
            assert (out.size, got[1]) == (512, b"")
        assert got == S.row_reference(cv, pend, data)
        cv, pend = got
        states.append(got)
        at += 2 * width
    data = _words(msg[at:at + 2 * width])
    out = evaluate_bits(S.last(len(pend), width, at + 2 * width), *_args(cv, pend, data))
    assert out.size == 512
    return _digest(out), states


def test_rows_and_the_last_give_the_streams_sha512_at_every_phase() -> None:
    """An odd 37-byte header and rows of 24 words (48 bytes) walk the phase through odd pending counts, one row of them
    completing a block; a 256-byte header and rows of 64 words keep it at 0 (no carry anywhere); rows of 160 words are
    2.5 blocks each; and the last rows end in one block (pending < 112) and two (pending >= 112)."""
    rng = random.Random("sha512-stream/phases")
    msg = rng.randbytes(4096)
    for start, width, rows in ((37, 24, 6), (256, 64, 3), (11, 160, 3), (3, 56, 1)):
        n = start + 2 * width * rows
        digest, states = _gates_stream(msg, start, width, rows)
        assert digest == hashlib.sha512(msg[:n]).digest(), (start, width)
        if states:
            assert states[-1] == S.state(msg[:n - 2 * width]), (start, width)
    assert (37 + 48) % 128 == 85 and (3 + 112) % 128 == 115


def test_the_last_row_alone_is_sha512_at_its_length() -> None:
    """A last row after every kind of state: no carry, an odd carry, a carry of 127 bytes; a row that completes a block
    and one that doesn't; an end of one block (111 bytes left) and of two (112), and the padding a block of its own."""
    rng = random.Random("sha512-stream/last")
    for start, width in ((0, 8), (0, 56), (1, 55), (100, 6), (37, 48), (0, 64), (127, 1), (64, 64), (1000, 12)):
        m = rng.randbytes(start + 2 * width)
        cv, pend = S.state(m[:start])
        data = _words(m[start:])
        out = evaluate_bits(S.last(len(pend), width, len(m)), *_args(cv, pend, data))
        assert _digest(out) == hashlib.sha512(m).digest(), (start, width)
        assert S.last_reference(cv, pend, data, len(m)) == hashlib.sha512(m).digest()
        assert S.end_reference(*S.state(m), len(m)) == hashlib.sha512(m).digest()


def test_a_flipped_bit_swapped_rows_or_another_length_change_the_digest() -> None:
    rng = random.Random("sha512-stream/controls")
    msg = bytearray(rng.randbytes(37 + 96 * 4))
    honest = _gates_stream(bytes(msg), 37, 48, 4)[0]
    flipped = bytearray(msg)
    flipped[37 + 96 * 2 + 5] ^= 0x10
    swapped = msg[:37] + msg[37 + 96:37 + 192] + msg[37:37 + 96] + msg[37 + 192:]
    got = [_gates_stream(bytes(flipped), 37, 48, 4)[0], _gates_stream(bytes(swapped), 37, 48, 4)[0],
           _gates_stream(bytes(msg) + bytes(96), 37, 48, 5)[0]]
    assert len({honest, *got}) == 4 and honest == hashlib.sha512(bytes(msg)).digest()


def test_the_signature_carries_what_pends() -> None:
    def names(fn):
        return [n for n, _ in fn.signature()[0]]

    assert names(S.row(0, 64)) == ["cv", "data"] and S.row(0, 64).signature()[1] == S.DIGEST
    assert names(S.row(37, 48)) == ["cv", "carry", "data"]
    assert [t.n for _, t in S.row(37, 48).signature()[0]] == [8, 19, 48]
    ret = S.row(37, 48).signature()[1]
    assert isinstance(ret, Tuple) and [f.leaves for f in ret.fields] == [512, 16 * 3]
    assert names(S.last(0, 64, 128)) == ["cv", "data"] and S.last(37, 48, 133).signature()[1] == S.DIGEST
    assert [t.n for _, t in S.last(37, 48, 133).signature()[0]] == [8, 19, 48]
    assert S.carry_words(b"\x01\x02\x03") == [0x0201, 0x0003] and S.state(b"")[0] == IV


def test_malformed_bindings_and_states_are_refused() -> None:
    for H, W in ((128, 1), (-1, 1), (True, 1), (0, 0), (0, "8")):
        with pytest.raises(ValueError, match="a stream's (H|W) is an int"):
            S.row(H, W).signature()  # type: ignore[arg-type]
    for n in (-1, 1 << 64, "8"):
        with pytest.raises(ValueError, match="a stream's LEN is an int"):
            S.last(0, 1, n).signature()  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="ends with 6 pending bytes, not the 5 a row of 48 words leaves after 37"):
        S.last(37, 48, 134).signature()
    with pytest.raises(ValueError, match="ends with 3 pending bytes, not 2"):
        S.end_reference(IV, b"ab", 131)
    with pytest.raises(ValueError, match="16 bits"):
        S.row_reference(IV, b"", [1 << 16])
    with pytest.raises(ValueError, match="a stream's H is an int"):
        S.row_reference(IV, bytes(128), [0])


def test_a_binding_costs_its_compressions_and_nothing_else() -> None:
    """Every compression is a `Sha512Compress_v1` call, its message private: a row of no whole block is wiring, of one block
    one call, of 2.5 blocks two or three by its phase, the last row's its row's and one or two more for the end; the body's
    own gates are none (the constants the padding and an odd tail's pad byte write are constant bits, not gates)."""
    assert bind(Sha512Compress).gates and compile_bits(bind(Sha512Compress)).counts["and"] == COMPRESS_ANDS
    for fn, blocks in ((S.row(2, 3), 0), (S.row(0, 64), 1), (S.row(37, 48), 1), (S.row(0, 160), 2), (S.row(100, 160), 3),
                       (S.last(0, 8, 16), 1), (S.last(37, 48, 133), 2), (S.last(1, 56, 113), 2), (S.last(0, 64, 128), 2)):
        assert compile_bits(fn).counts["and"] == blocks * COMPRESS_ANDS, fn.id
