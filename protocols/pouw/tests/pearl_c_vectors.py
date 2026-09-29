"""Pinned vectors for ``pearl-c-h100-v0``: ``uv run python protocols/pouw/tests/pearl_c_vectors.py`` rewrites
``tests/vectors/pearl_c.json``.  A change to that file is a change to the scheme and needs a reason in the PR.

Inputs are derived, not stored: row i of operand ``name`` in case ``c`` is an FP10 row whose k codes are the signed bytes
of SHAKE256("verity/pouw/pearl-c-vectors/v1", c, name, "codes", i) clamped to ±127 and whose k/8 BF16 scales are
2^(−4 − (b mod 5)) for the bytes of SHAKE256(…, "scales", i).  The context's salt and roots are SHA-256 of their labels."""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

from verity_pouw.protocol import Context, Matmul, Shape, Tile
from verity_pouw.schemes import pearl_c as C
from verity_pouw.schemes import pearl_kw as P

OUT = Path(__file__).with_name("vectors") / "pearl_c.json"
#: (case, shape, tile rows, tile cols, unit index, weight id, test instance)
CASES = (("test-16", (16, 128, 16), (0, 16), (0, 16), 3, 9, True),
         ("prod-64", (64, 1024, 64), (0, 64), (0, 64), 7, 2, False))


def shake(case: str, name: str, part: str, i: int, n: int) -> bytes:
    return hashlib.shake_256(f"verity/pouw/pearl-c-vectors/v1\0{case}\0{name}\0{part}\0{i}".encode()).digest(n)


def rows(case: str, name: str, count: int, k: int) -> list[bytes]:
    out = []
    for i in range(count):
        codes = [max(-127, b - 256 if b >= 128 else b) for b in shake(case, name, "codes", i, k)]
        scales = [P.f32_to_bf16(2.0 ** (-4 - b % 5)) for b in shake(case, name, "scales", i, k // 8)]
        out.append(P.fp10_row(codes, scales))
    return out


def ctx_of(case: str, shape: tuple[int, int, int], index: int, weight: int) -> Context:
    h = lambda s: hashlib.sha256(f"{s}/{case}".encode()).digest()
    return Context(h("salt"), Matmul(index, Shape(*shape), weight), h("root_a"), h("root_b"))


def scheme(test: bool) -> C.PearlC:
    return C.PearlC(k_min=32, tile=(16, 16)) if test else C.PearlC()


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def words(grid: list[list[int]]) -> bytes:
    return b"".join(struct.pack(f"<{len(r)}I", *r) for r in grid)


def halves(grid: list[list[int]]) -> bytes:
    return b"".join(struct.pack(f"<{len(r)}H", *r) for r in grid)


def case_vector(case, shape, r, c, index, weight, test) -> dict:
    s, ctx = scheme(test), ctx_of(case, shape, index, weight)
    s.check(ctx.matmul)
    tile = Tile(tuple(range(*r)), tuple(range(*c)))
    a, b = rows(case, "A", len(tile.rows), shape[1]), rows(case, "B", len(tile.cols), shape[1])
    noise = s.noise(ctx)
    ops = s.encode(ctx, noise, a, b, tile)
    checked = s.checked(ctx, ops)
    useful = s.useful(ctx, noise, ops, checked)
    return {
        "case": case, "scheme": s.name, "shape": list(shape), "tile": {"rows": list(r), "cols": list(c)}, "index": index,
        "weight": weight, "test_instance": test,
        "seed_a": noise.seed_a.hex(), "seed_b": noise.seed_b.hex(),
        "a_codes_sha256": sha(bytes(x for row in ops.a.codes for x in row)),
        "b_codes_sha256": sha(bytes(x for row in ops.b.codes for x in row)),
        "alpha_a": ops.a.alpha[:4], "beta_a": ops.a.beta[:4], "alpha_b": ops.b.alpha[:4], "beta_b": ops.b.beta[:4],
        "pa_sha256": sha(halves(ops.pa)), "pb_sha256": sha(halves(ops.pb)), "pa_0": ops.pa[0][:8], "pb_0": ops.pb[0][:8],
        "c_sha256": sha(words(checked.c)), "u_sha256": sha(words(checked.u)),
        "c_0": checked.c[0][:4], "u_0": checked.u[0][:4],
        "leaf": s.digest(ctx, noise, checked).hex(), "tickets": [t.hex() for t in s.tickets(noise, checked)],
        "useful_sha256": sha(json.dumps(useful).encode()), "useful_00": useful[0][0],
    }


def vectors() -> dict:
    return {"schema": "verity-pouw/pearl-c-vectors/v1", "cases": [case_vector(*c) for c in CASES]}


if __name__ == "__main__":
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(vectors(), indent=1) + "\n")
    print(f"wrote {OUT}")
