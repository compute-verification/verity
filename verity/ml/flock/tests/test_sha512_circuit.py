"""The SHA-512 compression and the ``hm96-sha512/v1`` row circuit against their references (core's `compress_ref`, core's hm96)."""
import os
import random

from verity.ml.commitments import hm96
from verity_flock import sha512_circuit as S


def _sliced(vals, nbits):
    return [sum(((v >> i) & 1) << k for k, v in enumerate(vals)) for i in range(nbits)]


def _unsliced(bits, t):
    return [sum(((b >> k) & 1) << i for i, b in enumerate(bits)) for k in range(t)]


def _pack(words):
    return sum(w << (64 * i) for i, w in enumerate(words))


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
    from verity.ml.commitments.rowleaf import ROLE_X, sha512_row_digest

    H = S.hm96_row()
    words = [[random.Random(i).getrandbits(16) for _ in range(64)] for i in range(2)]
    prefix = S.sha512_row_prefix(64)
    cvs = [S.midstate(prefix + b"".join(w.to_bytes(2, "little") for w in ws)) for ws in words]
    pad = S.words_be(S.pad_tail(128 + 128))
    ys = [os.urandom(192) for _ in range(2)]
    outs, _ = H.evaluate({"cv": _sliced([_pack(cv) for cv in cvs], 512), "pad": _sliced([_pack(pad)] * 2, 1024),
                          "y": _sliced([int.from_bytes(y, "little") for y in ys], 1536)}, 2)
    got = [v.to_bytes(128, "little") for v in _unsliced(outs["bc"], 2)]
    assert got == [inst.commit_string(inst.default_key, sha512_row_digest(ws, 16, ROLE_X), y) for ws, y in zip(words, ys)]
