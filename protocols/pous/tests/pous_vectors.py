"""Known-answer vectors for `F`, `Π₂`, `P_T`, `H` and whole segments (`fixtures/vectors.json`).

`python tests/pous_vectors.py` rewrites the file, including the 64 KB-label segment (about 30 s); the tests recompute
everything but that segment by default and compare.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from verity_pous.oracle import Tweak, run  # noqa: E402
from verity_pous.params import OPERATING_POINT, Params  # noqa: E402
from verity_pous.primitives import heuristic_arx, key_program, key_query, reference  # noqa: E402
from verity_pous.scheme import encode  # noqa: E402

SMALL = Params(4, 96, 2, 2)


def _bytes(label: str, n: int) -> bytes:
    return hashlib.shake_256(b"verity-pous/vectors/" + label.encode()).digest(n)


def _int(label: str, nbytes: int) -> int:
    return int.from_bytes(_bytes(label, nbytes), "little")


def _digest(x: int, nbytes: int) -> str:
    return hashlib.sha256(x.to_bytes(nbytes, "little")).hexdigest()


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
    if operating_point_segment:
        w, salt = segment_input(OPERATING_POINT)
        enc = encode(OPERATING_POINT, op, w, salt)
        out["segment_64kb"] = hashlib.sha256(b"".join(enc.block(i) for i in range(OPERATING_POINT.blocks))).hexdigest()
    return out


if __name__ == "__main__":
    path = Path(__file__).parent / "fixtures" / "vectors.json"
    path.write_text(json.dumps(vectors(), indent=1, sort_keys=True) + "\n")
    print(f"wrote {path}")
