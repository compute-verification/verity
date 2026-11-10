"""Pinned vectors for ``pearl-c-nvfp4-v0`` and ``pearl-c-mxfp4-v0``: ``uv run python verity/core/protocols/pouw/tests/pearl_c4_vectors.py``
rewrites ``tests/vectors/pearl_c4.json``.  A change to that file is a change to the scheme and needs a reason in the PR.

Inputs are derived, not stored: word t of row i of operand ``name`` in case ``c`` is the BF16 value with sign bit b0 >> 7,
mantissa b0 & 0x7F and exponent −4 + (b1 mod 6), where (b0, b1) are bytes 2t and 2t + 1 of
SHAKE256("verity/pouw/pearl-c4-vectors/v1", c, name, "words", i).  The context's salt and roots are SHA-256 of their
labels.  A ``-h3`` case (hashing format h3, SHA-512 trees, leaf and tickets) reads its h0 twin's inputs (its label without
``-h3``), its context's roots are 64 bytes (SHA-512 of their labels), and it adds row 0's seed and E_A line and a hash of
every row seed (`pearl_c4.row_seed` over each row's SHA-512 digest, `pearl_c.row_digest`)."""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

from verity.core.protocols.pouw.protocol import Context, Matmul, Shape
from verity.ml.protocols.pouw.schemes import pearl_c4 as F
from verity.ml.protocols.pouw.schemes import pearl_c_work as W

OUT = Path(__file__).with_name("vectors") / "pearl_c4.json"
#: (case, format, shape, unit index, weight id, test instance[, hashing]); the tile is the whole test shape, or the first
#: 64 x 64
CASES = (("test-16-nvfp4", "nvfp4", (16, 128, 16), 3, 9, True),
         ("test-16-mxfp4", "mxfp4", (16, 128, 16), 3, 9, True),
         ("prod-64-nvfp4", "nvfp4", (64, 1024, 128), 7, 2, False),
         ("test-16-nvfp4-h3", "nvfp4", (16, 128, 16), 3, 9, True, "h3"))


def shake(case: str, name: str, part: str, i: int, n: int) -> bytes:
    return hashlib.shake_256(f"verity/pouw/pearl-c4-vectors/v1\0{case}\0{name}\0{part}\0{i}".encode()).digest(n)


def row_codes(case: str, name: str, i: int, k: int) -> list[int]:
    raw = shake(case, name, "words", i, 2 * k)
    return [(b0 >> 7) << 15 | (123 + b1 % 6) << 7 | (b0 & 0x7F) for b0, b1 in zip(raw[0::2], raw[1::2])]


def rows(case: str, name: str, count: int, k: int) -> list[bytes]:
    return [F.bf16_row(row_codes(case, name, i, k)) for i in range(count)]


def ctx_of(case: str, shape: tuple[int, int, int], index: int, weight: int, wide: bool = False) -> Context:
    h = lambda s: hashlib.sha256(f"{s}/{case}".encode()).digest()  # noqa: E731
    r = (lambda s: hashlib.sha512(f"{s}/{case}".encode()).digest()) if wide else h  # noqa: E731
    return Context(h("salt"), Matmul(index, Shape(*shape), weight), r("root_a"), r("root_b"))


def scheme(fmt: str, test: bool, hashing: str = "h0") -> F.PearlC4:
    return F.PearlC4(fmt, k_min=128, tile=(16, 16), n_min=16, hashing=hashing) if test else F.PearlC4(fmt, hashing=hashing)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def words(grid: list[list[int]]) -> bytes:
    return b"".join(struct.pack(f"<{len(r)}I", *r) for r in grid)


def halves(grid: list[list[int]]) -> bytes:
    return b"".join(struct.pack(f"<{len(r)}H", *r) for r in grid)


def case_vector(case, fmt, shape, index, weight, test, hashing="h0") -> dict:
    base = case.removesuffix("-h3")
    s, ctx = scheme(fmt, test, hashing), ctx_of(base, shape, index, weight, hashing == "h3")
    s.check(ctx.matmul)
    tile = s.tiles(ctx.matmul.shape)[0]
    a, b = rows(base, "A", len(tile.rows), shape[1]), rows(base, "B", len(tile.cols), shape[1])
    noise = s.noise(ctx)
    ops = s.encode(ctx, noise, a, b, tile)
    checked = s.checked(ctx, ops)
    useful = s.useful(ctx, noise, ops, checked)
    _, cap = s.tile_cap(W.UnitRecord(ctx.matmul, shape[0], shape[1]), ops, range(len(tile.rows)), range(len(tile.cols)))
    dead = F.salt_dead(s.fmt, ops.a.x[0], ops.a.alpha[0], ops.a.ae[0])
    p, modes = F.change_probabilities(s.fmt, ops.a.x[0], ops.a.alpha[0], ops.e_a[0], ops.a.ae[0], dead, ops.a.scales[0])
    lo, pmf = F.noise_law(tuple(ops.e_a[0]))
    out = {
        "case": case, "scheme": s.name, "shape": list(shape),
        "tile": {"rows": [tile.rows[0], tile.rows[-1] + 1], "cols": [tile.cols[0], tile.cols[-1] + 1]},
        "index": index, "weight": weight, "test_instance": test, "admissible": s.admissible(ctx, ops),
        "seed_a": noise.seed_a.hex(), "seed_b": noise.seed_b.hex(),
        "a_codes_sha256": sha(bytes(x for row in ops.a.codes for x in row)),
        "b_codes_sha256": sha(bytes(x for row in ops.b.codes for x in row)),
        "a_scales_sha256": sha(bytes(x for row in ops.a.scales for x in row)),
        "b_scales_sha256": sha(bytes(x for row in ops.b.scales for x in row)),
        "alpha_a": ops.a.alpha[:4], "beta_a": ops.a.beta[:4], "ae_a": ops.a.ae[:4],
        "alpha_b": ops.b.alpha[:4], "beta_b": ops.b.beta[:4], "ae_b": ops.b.ae[:4],
        "pa_sha256": sha(halves(ops.pa)), "pb_sha256": sha(halves(ops.pb)), "pa_0": ops.pa[0][:8], "pb_0": ops.pb[0][:8],
        "c_sha256": sha(words(checked.c)), "u_sha256": sha(words(checked.u)),
        "c_0": checked.c[0][:4], "u_0": checked.u[0][:4],
        "leaf": s.digest(ctx, noise, checked).hex(), "tickets": [t.hex() for t in s.tickets(noise, checked)],
        "useful_sha256": sha(json.dumps(useful).encode()), "useful_00": useful[0][0],
        "debit": s.last_debit.to_json(), "cap": str(cap),
        "f1_a0": {"noise_law_lo": lo, "noise_law_sha256": sha(struct.pack(f"<{len(pmf)}d", *pmf)),
                  "modal_bytes_sha256": sha(bytes(modes)), "modal_bytes": modes[:8],
                  "p_sha256": sha(struct.pack(f"<{len(p)}d", *p)), "p": [v.hex() for v in p[:8]],
                  "split_units": str(F.split_units(s.fmt, p, modes, ops.a.scales[0]))},
    }
    if hashing == "h3":
        seeds = s.row_seeds(ctx, a, tile.rows)
        out.update({"hashing": hashing, "row_seed_a0": seeds[0].hex(), "row_seeds_sha256": sha(b"".join(seeds)), "e_a0": ops.e_a[0]})
    return out


def vectors() -> dict:
    return {"schema": "verity-pouw/pearl-c4-vectors/v1", "cases": [case_vector(*c) for c in CASES]}


if __name__ == "__main__":
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(vectors(), indent=1) + "\n")
    print(f"wrote {OUT}")
