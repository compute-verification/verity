"""Regenerate the pinned ``fp8-is-h100-*`` vectors: ``uv run python protocols/pouw/tests/fp8_is_vectors.py`` rewrites
``tests/vectors/fp8_is.json``.  A change to that file is a change to a scheme and needs a reason in the PR.

Inputs are derived, not stored: row i of operand ``name`` in case ``c`` is SHAKE256("verity/pouw/test-rows/fp8/v1", c,
name, i), each byte b read as the E4M3 code ``(b & 0x80) | ((b & 0x7F) % (cap + 1))`` for the forming layer's cap on
that side (0x7E, the largest finite code, when it has none)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from verity_pouw.protocol import Context, Matmul, Shape, Tile
from verity_pouw.schemes import fp8_is

OUT = Path(__file__).with_name("vectors") / "fp8_is.json"
#: (case, forming, shape, tile rows, tile cols, index, weight): test-instance shapes and one lead-domain k
CASES = (("small-d2", "d2-draft", (64, 128, 16), (0, 2), (8, 10), 7, 3),
         ("small-fpncp", "fpncp-v0", (64, 128, 16), (0, 2), (8, 10), 7, 3),
         ("lead-k-d2", "d2-draft", (8192, 8192, 8192), (0, 1), (0, 1), (1 << 64) - 1, 11))


def caps(forming: fp8_is.Forming) -> tuple[int, int]:
    return (getattr(forming, "a_max", 0x7E), getattr(forming, "b_max", 0x7E))


def rows(case: str, name: str, count: int, k: int, cap: int, first: int = 0) -> list[bytes]:
    return [bytes((b & 0x80) | ((b & 0x7F) % (cap + 1))
                  for b in hashlib.shake_256(f"verity/pouw/test-rows/fp8/v1\0{case}\0{name}\0{i}".encode()).digest(k))
            for i in range(first, first + count)]


def ctx_of(case: str, shape: tuple[int, int, int], index: int, weight: int) -> Context:
    return Context(hashlib.sha256(f"salt/{case}".encode()).digest(), Matmul(index, Shape(*shape), weight),
                   hashlib.sha256(f"root_a/{case}".encode()).digest(), hashlib.sha256(f"root_b/{case}".encode()).digest())


def scheme(forming: str, shape: tuple[int, int, int]) -> fp8_is.FP8IS:
    layer = fp8_is.FORMINGS[forming]()
    return fp8_is.FP8IS(layer) if min(shape) >= 8192 else fp8_is.FP8IS(layer, mnk_min=16)


def case_vector(case, forming, shape, r, c, index, weight) -> dict:
    s, ctx = scheme(forming, shape), ctx_of(case, shape, index, weight)
    s.check(ctx.matmul)
    tile = Tile(tuple(range(*r)), tuple(range(*c)))
    cap_a, cap_b = caps(s.forming)
    a = rows(case, "A", len(tile.rows), shape[1], cap_a, r[0])
    b = rows(case, "B", len(tile.cols), shape[1], cap_b, c[0])
    ops = s.encode(ctx, s.noise(ctx), a, b, tile)
    checked = s.checked(ctx, ops)
    useful = s.useful(ctx, None, ops, checked)
    return {"case": case, "scheme": s.name, "shape": list(shape), "tile": {"rows": list(r), "cols": list(c)}, "index": index,
            "weight": weight, "salt": ctx.salt.hex(), "root_a": ctx.root_a.hex(), "root_b": ctx.root_b.hex(),
            "x_row0": ops.x[0][:16], "y_col0": ops.y[0][:16],
            "trace_sha256": hashlib.sha256(checked.trace).hexdigest(),
            "cell00_words": [list(w) for w in checked.cells[0][0][:4]],
            "digest": s.digest(ctx, None, checked).hex(),
            "useful_sha256": hashlib.sha256(json.dumps(useful).encode()).hexdigest(), "useful_00": useful[0][0]}


def vectors() -> dict:
    return {"schema": "verity-pouw/fp8-is-vectors/v1",
            "perm_sha256": {name: hashlib.sha256(b"".join(x.to_bytes(2, "little") for x in fp8_is.ncp_perm(8192, layer.P_SEED)[0]))
                            .hexdigest() for name, layer in fp8_is.FORMINGS.items()},
            "cases": [case_vector(*c) for c in CASES]}


if __name__ == "__main__":
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(vectors(), indent=1) + "\n")
    print(f"wrote {OUT}")
