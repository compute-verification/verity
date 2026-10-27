"""Known-answer vectors for `F`, `Π₂`, `P_T`, `H` and whole segments of both codecs (`fixtures/vectors.json`).

`python tests/pous_vectors.py` rewrites the file, including P3's and the band's 64 KB-label segments (about a minute);
the tests recompute
everything but the two whole 64 KB segments by default and compare.  The dense scheme's 64 KB entry is the first 16
labels of a segment (136 calls of `Π₂`; the whole segment is 131,328); the band's are the first 16 labels and the whole
secure-first segment (6,578 calls).
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from verity.core.protocols.pous import band, dense  # noqa: E402
from verity.core.protocols.pous.labels import salt_value, tag  # noqa: E402
from verity.core.protocols.pous.oracle import Tweak, run  # noqa: E402
from verity.core.protocols.pous.params import OPERATING_POINT, Params  # noqa: E402
from verity.core.protocols.pous.primitives import heuristic_arx, key_program, key_query, reference  # noqa: E402
from verity.core.protocols.pous.scheme import encode  # noqa: E402

SMALL = Params(4, 96, 2, 2)


def _bytes(label: str, n: int) -> bytes:
    return hashlib.shake_256(b"verity-pous/vectors/" + label.encode()).digest(n)


def _int(label: str, nbytes: int) -> int:
    return int.from_bytes(_bytes(label, nbytes), "little")


def _digest(x: int, nbytes: int) -> str:
    return hashlib.sha256(x.to_bytes(nbytes, "little")).hexdigest()


def band_segment_input() -> tuple[bytes, bytes]:
    """One whole secure-first band segment's `W` (32 MiB) and salt."""
    return _bytes("band-segment-w", band.SECURE_FIRST.segment_bytes), _bytes("salt", 24)


def segment_input(params: Params) -> tuple[bytes, bytes]:
    return _bytes(f"w/{params.n}/{params.wc}/{params.k}/{params.segments}", params.w_bytes), _bytes("salt", 24)


def _primitive_vectors(params: Params, prims, name: str) -> dict:
    mb = params.m // 8
    t = Tweak(_int("tag", 32), 1, min(params.n - 1, 12))
    x = _int("p-input", mb)
    y = prims.p.forward(t, x)
    s = _int("pi2-input", 2 * mb + 64)
    parents = {u: _int(f"parent/{u}", mb) for u in params.parents(t.node)}
    key, _ = run(key_program(prims.h, key_query(params, t.tag, 1, t.node, parents)), prims)
    return {f"{name}/tweak": t.to_bytes().hex(), f"{name}/p": _digest(y, mb),
            f"{name}/pi2": _digest(prims.h.pi2.forward(s), 2 * mb + 64), f"{name}/h": _digest(key, mb)}


def vectors(operating_point_segment: bool = True) -> dict:
    op = reference(OPERATING_POINT)
    half = (2 * OPERATING_POINT.m // 8 + 64) // 2
    out = {"F/pi2/round1": hashlib.sha256(op.h.pi2.core.f(1, b"", _bytes("f-input", half))).hexdigest(),
           "F/p/round1": hashlib.sha256(op.p.core.f(1, Tweak(0, 0, 0).to_bytes(),
                                                    _bytes("f-input", OPERATING_POINT.m // 16))).hexdigest()}
    out.update(_primitive_vectors(OPERATING_POINT, op, "64kb"))
    out.update(_primitive_vectors(SMALL, reference(SMALL), "small"))
    out.update(_primitive_vectors(SMALL, heuristic_arx(SMALL), "small-arx"))
    w, salt = segment_input(SMALL)
    enc = encode(SMALL, reference(SMALL), w, salt)
    out["small/blocks"] = [enc.block(i).hex() for i in range(SMALL.blocks)]
    small = dense.DenseParams(8, 64, 2)
    w, salt = _bytes("dense-small-w", small.w_bytes), _bytes("salt", 24)
    prims = dense.reference(small)
    out["dense/small/blocks"] = [b.hex() for s in range(2)
                                 for b in dense.encode_segment(small, prims, salt, s, w[s * 64:(s + 1) * 64])]
    sf = dense.POINT_64K
    w, salt = _bytes("dense-w", 16 * sf.m // 8), _bytes("salt", 24)
    labels = [int.from_bytes(w[i * sf.m // 8:(i + 1) * sf.m // 8], "little") for i in range(16)]
    prims = dense.reference(sf)
    c, _ = run(dense.encode_segment_prog(sf, prims.h, tag(salt_value(salt), 0), labels, count=16), prims)
    out["dense/64kb/prefix16"] = hashlib.sha256(b"".join(x.to_bytes(sf.m // 8, "little") for x in c)).hexdigest()
    small = band.BandParams(16, 64, 4, 2)
    w, salt = _bytes("band-small-w", small.w_bytes), _bytes("salt", 24)
    prims = band.reference(small)
    out["band/small/blocks"] = [b.hex() for s in range(2)
                                for b in small.codec.encode_segment(small, prims, salt, s, w[s * 128:(s + 1) * 128])]
    bf = band.SECURE_FIRST
    labels = [int.from_bytes(_bytes("band-w", 16 * bf.m // 8)[i * bf.m // 8:(i + 1) * bf.m // 8], "little")
              for i in range(16)]
    prims = band.reference(bf)
    c, _ = run(dense.encode_segment_prog(bf, prims.h, tag(salt_value(salt), 0), labels, count=16), prims)
    out["band/64kb/prefix16"] = hashlib.sha256(b"".join(x.to_bytes(bf.m // 8, "little") for x in c)).hexdigest()
    if operating_point_segment:
        w, salt = band_segment_input()
        blocks = bf.codec.encode_segment(bf, prims, salt, 0, w)
        out["band/64kb/segment"] = hashlib.sha256(b"".join(blocks)).hexdigest()
    if operating_point_segment:
        w, salt = segment_input(OPERATING_POINT)
        enc = encode(OPERATING_POINT, op, w, salt)
        out["segment_64kb"] = hashlib.sha256(b"".join(enc.block(i) for i in range(OPERATING_POINT.blocks))).hexdigest()
    return out


if __name__ == "__main__":
    path = Path(__file__).parent / "fixtures" / "vectors.json"
    path.write_text(json.dumps(vectors(), indent=1, sort_keys=True) + "\n")
    print(f"wrote {path}")
