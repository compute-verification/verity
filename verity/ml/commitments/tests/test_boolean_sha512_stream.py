"""`verity.ml.commitments.gates.sha512_stream`: a stream's rows and its end in gates give `hashlib.sha512` of the stream, at
an odd header (every row's carry ending in a pad byte), at whole-block phases (no carry), with rows of several blocks and an
end of one block and of two; a flipped bit, two rows swapped and another length each change the digest; the signatures carry
a carry exactly where bytes pend; malformed bindings and states are refused; and a binding costs its compressions'
`Sha512Compress_v1` ANDs and nothing else."""
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
    """A row's or the end's output bits as (cv, pending bytes)."""
    flat = out.reshape(-1)
    cv = [sum(int(flat[64 * k + j]) << j for j in range(64)) for k in range(8)]
    words = [sum(int(flat[512 + 16 * k + j]) << j for j in range(16)) for k in range((rest + 1) // 2)]
    return cv, b"".join(w.to_bytes(2, "little") for w in words)[:rest]


def _gates_stream(msg: bytes, start: int, width: int, rows: int) -> tuple[bytes, list[tuple[list[int], bytes]]]:
    """``msg``'s first ``start`` bytes as the state the verifier computes, then ``rows`` rows of ``width`` words through
    `Sha512StreamRow_v1` and the end through `Sha512StreamEnd_v1`: (the digest, each row's state in gates)."""
    cv, pend = S.state(msg[:start])
    at, states = start, []
    for _ in range(rows):
        data = _words(msg[at:at + 2 * width])
        fn = S.row(len(pend), width)
        rest = (len(pend) + 2 * width) % 128
        out = evaluate_bits(fn, *[np.array([a], dtype=np.uint8) for a in S.state_bits(cv, pend)],
                            np.array([[(w >> j) & 1 for w in data for j in range(16)]], dtype=np.uint8))
        got = _unpack(fn, out, rest)
        if not rest:
            assert (out.size, got[1]) == (512, b"")
        assert got == S.row_reference(cv, pend, data)
        cv, pend = got
        states.append(got)
        at += 2 * width
    out = evaluate_bits(S.end(at), *[np.array([a], dtype=np.uint8) for a in S.state_bits(cv, pend)])
    return b"".join(v.to_bytes(8, "big") for v in _unpack(None, out, 0)[0]), states


def test_rows_and_the_end_give_the_streams_sha512_at_every_phase() -> None:
    """An odd 37-byte header and rows of 24 words (48 bytes) walk the phase through odd pending counts, one row of them
    completing a block; a 256-byte header and rows of 64 words keep it at 0 (no carry anywhere); rows of 160 words are
    2.5 blocks each; and the ends take one block (pending < 112) and two (pending >= 112)."""
    rng = random.Random("sha512-stream/phases")
    msg = rng.randbytes(4096)
    for start, width, rows in ((37, 24, 6), (256, 64, 3), (11, 160, 3), (3, 56, 1)):
        n = start + 2 * width * rows
        digest, states = _gates_stream(msg, start, width, rows)
        assert digest == hashlib.sha512(msg[:n]).digest(), (start, width)
        assert states[-1] == S.state(msg[:n]), (start, width)
    assert (37 + 48) % 128 == 85 and (3 + 112) % 128 == 115


def test_the_end_alone_is_sha512_at_its_length() -> None:
    rng = random.Random("sha512-stream/end")
    for n in (0, 1, 111, 112, 127, 128, 255, 1000):
        m = rng.randbytes(n)
        cv, pend = S.state(m)
        out = evaluate_bits(S.end(n), *[np.array([a], dtype=np.uint8) for a in S.state_bits(cv, pend)])
        assert b"".join(v.to_bytes(8, "big") for v in _unpack(None, out, 0)[0]) == hashlib.sha512(m).digest(), n
        assert S.end_reference(cv, pend, n) == hashlib.sha512(m).digest()


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
    assert names(S.end(1024)) == ["cv"] and [t.n for _, t in S.end(1000).signature()[0]] == [8, 52]
    assert S.carry_words(b"\x01\x02\x03") == [0x0201, 0x0003] and S.state(b"")[0] == IV


def test_malformed_bindings_and_states_are_refused() -> None:
    for H, W in ((128, 1), (-1, 1), (True, 1), (0, 0), (0, "8")):
        with pytest.raises(ValueError, match="a stream's (H|W) is an int"):
            S.row(H, W).signature()  # type: ignore[arg-type]
    for n in (-1, 1 << 64, "8"):
        with pytest.raises(ValueError, match="a stream's LEN is an int"):
            S.end(n).signature()  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="ends with 3 pending bytes, not 2"):
        S.end_reference(IV, b"ab", 131)
    with pytest.raises(ValueError, match="16 bits"):
        S.row_reference(IV, b"", [1 << 16])
    with pytest.raises(ValueError, match="a stream's H is an int"):
        S.row_reference(IV, bytes(128), [0])


def test_a_binding_costs_its_compressions_and_nothing_else() -> None:
    """Every compression is a `Sha512Compress_v1` call, its message private: a row of no whole block is wiring, of one block
    one call, of 2.5 blocks two or three by its phase, an end one or two; the body's own gates are none (the constants
    the end's padding and an odd tail's pad byte write are constant bits, not gates)."""
    assert bind(Sha512Compress).gates and compile_bits(bind(Sha512Compress)).counts["and"] == COMPRESS_ANDS
    for fn, blocks in ((S.row(2, 3), 0), (S.row(0, 64), 1), (S.row(37, 48), 1), (S.row(0, 160), 2), (S.row(100, 160), 3),
                       (S.end(1000), 1), (S.end(1024), 1), (S.end(1015), 2)):
        assert compile_bits(fn).counts["and"] == blocks * COMPRESS_ANDS, fn.id
