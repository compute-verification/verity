"""Regenerate the pinned NCP vectors: ``uv run python verity/core/protocols/pouw/tests/pouw_vectors.py`` rewrites
``tests/vectors/ncp.json``.  A change to that file is a change to the scheme and needs a reason in the PR.

Inputs are derived, not stored: row i of operand ``name`` in case ``c`` is SHAKE256("verity/pouw/test-rows/v1", c, name, i)
read as int7 (each byte mod 128, minus 64)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from verity.core.protocols.pouw.protocol import Context, Matmul, Shape, Tile
from verity.core.protocols.pouw.schemes import ncp

OUT = Path(__file__).with_name("vectors") / "ncp.json"
P_KS = (1088, 2048, 4096, 11008)
#: (case, perm, shape, tile rows, tile cols, index, weight): small test-instance shapes and one lead-domain tile
CASES = (("small-word", "word", (16, 64, 64), (0, 16), (16, 32), 7, 3),
         ("small-shift24", "shift24", (16, 64, 64), (0, 16), (0, 16), 1 << 40, 2),
         ("lead-word", "word", (16, 1088, 1600), (0, 16), (48, 64), (1 << 64) - 1, 11))


def rows(case: str, name: str, count: int, k: int, first: int = 0) -> list[bytes]:
    return [bytes((b % 128) - 64 & 0xFF for b in hashlib.shake_256(f"verity/pouw/test-rows/v1\0{case}\0{name}\0{i}".encode()).digest(k))
            for i in range(first, first + count)]


def ctx_of(case: str, shape: tuple[int, int, int], index: int, weight: int) -> Context:
    return Context(hashlib.sha256(f"salt/{case}".encode()).digest(), Matmul(index, Shape(*shape), weight),
                   hashlib.sha256(f"root_a/{case}".encode()).digest(), hashlib.sha256(f"root_b/{case}".encode()).digest())


def scheme(perm: str, shape: tuple[int, int, int]) -> ncp.NCP:
    return ncp.NCP(perm) if shape[1] >= 1088 else ncp.NCP(perm, k_min=64, n_min=64)


def case_vector(case, perm, shape, r, c, index, weight) -> dict:
    s, ctx = scheme(perm, shape), ctx_of(case, shape, index, weight)
    s.check(ctx.matmul)
    tile = Tile(tuple(range(*r)), tuple(range(*c)))
    ops = s.encode(ctx, s.noise(ctx), rows(case, "A", len(tile.rows), shape[1], r[0]), rows(case, "B", len(tile.cols), shape[1], c[0]), tile)
    checked = s.checked(ctx, ops)
    useful = s.useful(ctx, None, ops, checked)
    return {"case": case, "scheme": s.name, "shape": list(shape), "tile": {"rows": list(r), "cols": list(c)}, "index": index,
            "weight": weight, "salt": ctx.salt.hex(), "root_a": ctx.root_a.hex(), "root_b": ctx.root_b.hex(),
            "x_row0": ops.x[0][:16], "y_col0": ops.y[0][:16],
            "digest": s.digest(ctx, None, checked).hex(),
            "useful_sha256": hashlib.sha256(json.dumps(useful).encode()).hexdigest(), "useful_00": useful[0][0]}


def vectors() -> dict:
    return {"schema": "verity-pouw/ncp-vectors/v1",
            "word_p_sha256": {str(k): ncp.perm_digest(ncp.word_perm(k)[0]) for k in P_KS},
            "cases": [case_vector(*c) for c in CASES]}


if __name__ == "__main__":
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(vectors(), indent=1) + "\n")
    print(f"wrote {OUT}")
