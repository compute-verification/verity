"""The SHA-512 compression and the ``hm96-sha512/v1`` row circuit against their references (``hashlib``, core's hm96)."""
import hashlib
import os
import random

from verity.commitments import hm96
from verity_flock import sha512_circuit as S


def _sliced(vals, nbits):
    return [sum(((v >> i) & 1) << k for k, v in enumerate(vals)) for i in range(nbits)]


def _unsliced(bits, t):
    return [sum(((b >> k) & 1) << i for i, b in enumerate(bits)) for k in range(t)]


def _pack(words):
    return sum(w << (64 * i) for i, w in enumerate(words))


def test_reference_is_sha512():
    for m in (b"", b"abc", bytes(range(111))):
        assert S.compress_ref(S.IV, S.words_be(m + S.pad_tail(len(m)))) == S.words_be(hashlib.sha512(m).digest())
    long = os.urandom(300)
    cv = S.IV
    padded = long + S.pad_tail(len(long))
    for i in range(0, len(padded), 128):
        cv = S.compress_ref(cv, S.words_be(padded[i:i + 128]))
    assert cv == S.words_be(hashlib.sha512(long).digest())


def test_compression_circuit():
    C = S.compression()
    counts = C.counts()
    assert counts["inputs"] == 1536 and counts["output_copies"] == 512
    rng = random.Random(7)
    cvs = [[rng.getrandbits(64) for _ in range(8)] for _ in range(3)]
    msgs = [[rng.getrandbits(64) for _ in range(16)] for _ in range(3)]
    outs, _ = C.evaluate({"cv": _sliced([_pack(c) for c in cvs], 512), "msg": _sliced([_pack(m) for m in msgs], 1024)}, 3)
    assert _unsliced(outs["out"], 3) == [_pack(S.compress_ref(c, m)) for c, m in zip(cvs, msgs)]


def test_hm96_row_circuit():
    inst = hm96.SHA512
    H = S.hm96_row()
    xs, ys = [os.urandom(64) for _ in range(2)], [os.urandom(192) for _ in range(2)]
    outs, _ = H.evaluate({"x": _sliced([_pack(S.words_be(x)) for x in xs], 512),
                          "y": _sliced([int.from_bytes(y, "little") for y in ys], 1536)}, 2)
    got = [v.to_bytes(128, "little") for v in _unsliced(outs["bc"], 2)]
    assert got == [inst.commit_string(inst.default_key, x, y) for x, y in zip(xs, ys)]
