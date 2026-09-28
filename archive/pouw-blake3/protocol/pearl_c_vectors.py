"""Pinned vectors for ``pearl-c-h100-v0``: ``uv run python verity/core/protocols/pouw/tests/pearl_c_vectors.py`` rewrites
``tests/vectors/pearl_c.json``.  A change to that file is a change to the scheme and needs a reason in the PR.

Inputs are derived, not stored: row i of operand ``name`` in case ``c`` is an FP10 row whose k codes are the signed bytes
of SHAKE256("verity/pouw/pearl-c-vectors/v1", c, name, "codes", i) clamped to ±127 and whose k/8 BF16 scales are
2^(−4 − (b mod 5)) for the bytes of SHAKE256(…, "scales", i).  A v1 row (FP32 words) is that row opened, each nonzero word's
low 16 mantissa bits replaced by the 16-bit words of SHAKE256(…, "fraction", i).  The context's salt and roots are SHA-256
of their labels.  A ``-h1`` case (hashing format h1) reads its h0 twin's inputs (its label without ``-h1``), and adds A's
``frame-b3`` row root over the tile's A rows, row 0's leaf and the first message digest.  ``decode-32-v1-h1`` is decode's
partial tile: 32 rows of a 64-column tile, its leaf over 64 digests.  A ``-h2`` case is its ``-h1`` twin with the trees
``frame-b3s``: the same inputs, digests and leaf, another row root and row leaf.  A ``-h3`` case is its ``-h2`` twin with E_A
seeded per row (`pearl_c.row_seed` over each row's frame-b3s leaf): it adds row 0's seed and E_A line and a hash of every
row seed, and its operands, peel factors, words, leaf and tickets are its own.  A ``-h3s`` case is its ``-h3`` twin with each
row's seed keyed on its SHA-512 row digest (`pearl_c.row_digest`, ``sha512/row-seg/v1``) and SHA-512 in BLAKE3's every place:
its context's roots are 64 bytes (SHA-512 of their labels), its row tree `audit.HM96`'s under a salt key SHA-256 of its label,
and it adds row 0's digest and a hash of every row digest, and no message digest (its leaf hashes the messages)."""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

from verity.core.protocols.pouw.audit import HM96, ROW_SCHEMA, activation_binding, commit_rows, hm96_salt
from verity.core.protocols.pouw.protocol import Context, Matmul, Shape, Tile
from verity.core.protocols.pouw.schemes import pearl_c as C
from verity.core.protocols.pouw.schemes import pearl_kw as P

OUT = Path(__file__).with_name("vectors") / "pearl_c.json"
#: (case, shape, tile rows, tile cols, unit index, weight id, test instance, forming[, hashing])
CASES = (("test-16", (16, 128, 16), (0, 16), (0, 16), 3, 9, True, "v0"),
         ("prod-64", (64, 1024, 64), (0, 64), (0, 64), 7, 2, False, "v0"),
         ("test-16-v1", (16, 128, 16), (0, 16), (0, 16), 3, 9, True, "v1"),
         ("prod-64-v1", (64, 1024, 64), (0, 64), (0, 64), 7, 2, False, "v1"),
         ("test-16-v1-h1", (16, 128, 16), (0, 16), (0, 16), 3, 9, True, "v1", "h1"),
         ("prod-64-v1-h1", (64, 1024, 64), (0, 64), (0, 64), 7, 2, False, "v1", "h1"),
         ("decode-32-v1-h1", (32, 1024, 64), (0, 32), (0, 64), 7, 2, False, "v1", "h1"),
         ("test-16-v1-h2", (16, 128, 16), (0, 16), (0, 16), 3, 9, True, "v1", "h2"),
         ("prod-64-v1-h2", (64, 1024, 64), (0, 64), (0, 64), 7, 2, False, "v1", "h2"),
         ("decode-32-v1-h2", (32, 1024, 64), (0, 32), (0, 64), 7, 2, False, "v1", "h2"),
         ("test-16-v1-h3", (16, 128, 16), (0, 16), (0, 16), 3, 9, True, "v1", "h3"),
         ("decode-32-v1-h3", (32, 1024, 64), (0, 32), (0, 64), 7, 2, False, "v1", "h3"),
         ("test-16-v1-h3s", (16, 128, 16), (0, 16), (0, 16), 3, 9, True, "v1", "h3s"),
         ("decode-32-v1-h3s", (32, 1024, 64), (0, 32), (0, 64), 7, 2, False, "v1", "h3s"))


def shake(case: str, name: str, part: str, i: int, n: int) -> bytes:
    return hashlib.shake_256(f"verity/pouw/pearl-c-vectors/v1\0{case}\0{name}\0{part}\0{i}".encode()).digest(n)


def rows(case: str, name: str, count: int, k: int) -> list[bytes]:
    out = []
    for i in range(count):
        codes = [max(-127, b - 256 if b >= 128 else b) for b in shake(case, name, "codes", i, k)]
        scales = [P.f32_to_bf16(2.0 ** (-4 - b % 5)) for b in shake(case, name, "scales", i, k // 8)]
        out.append(P.fp10_row(codes, scales))
    return out


def rows_f32(case: str, name: str, count: int, k: int) -> list[bytes]:
    out = []
    for i, row in enumerate(rows(case, name, count, k)):
        opened = P.open_row(*P.parse_row(row, k))
        frac = struct.unpack(f"<{k}H", shake(case, name, "fraction", i, 2 * k))
        out.append(C.f32_row([(x << 16) | f if x & 0x7FFF else x << 16 for x, f in zip(opened, frac)]))
    return out


def ctx_of(case: str, shape: tuple[int, int, int], index: int, weight: int, wide: bool = False) -> Context:
    h = lambda s: hashlib.sha256(f"{s}/{case}".encode()).digest()  # noqa: E731
    r = (lambda s: hashlib.sha512(f"{s}/{case}".encode()).digest()) if wide else h  # noqa: E731
    return Context(h("salt"), Matmul(index, Shape(*shape), weight), r("root_a"), r("root_b"))


def salt_key(case: str) -> bytes:
    return hashlib.sha256(f"salt-key/{case}".encode()).digest()


def scheme(test: bool, forming: str = "v0", hashing: str = "h0") -> C.PearlC:
    return C.PearlC(k_min=32, tile=(16, 16), forming=forming, hashing=hashing) if test else C.PearlC(forming=forming, hashing=hashing)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def words(grid: list[list[int]]) -> bytes:
    return b"".join(struct.pack(f"<{len(r)}I", *r) for r in grid)


def halves(grid: list[list[int]]) -> bytes:
    return b"".join(struct.pack(f"<{len(r)}H", *r) for r in grid)


def case_vector(case, shape, r, c, index, weight, test, forming, hashing="h0") -> dict:
    base = case.removesuffix("-h1").removesuffix("-h2").removesuffix("-h3s").removesuffix("-h3")   # a hashed case reads its h0 twin's inputs
    s, ctx = scheme(test, forming, hashing), ctx_of(base, shape, index, weight, hashing == "h3s")
    s.check(ctx.matmul)
    tile = Tile(tuple(range(*r)), tuple(range(*c)))
    make = rows if forming == "v0" else rows_f32
    a, b = make(base, "A", len(tile.rows), shape[1]), make(base, "B", len(tile.cols), shape[1])
    noise = s.noise(ctx)
    ops = s.encode(ctx, noise, a, b, tile)
    checked = s.checked(ctx, ops)
    useful = s.useful(ctx, noise, ops, checked)
    out = {
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
    if hashing in ("h1", "h2", "h3", "h3s"):
        key = salt_key(case) if s.commitment_hash == HM96 else None
        tree = commit_rows(activation_binding(ctx.salt, index), index, a, s.commitment_hash, key, index)
        value0 = a[0] + (hm96_salt(key, index, 0) if key else b"")
        out.update({"hashing": hashing, "rows_root_a": tree.commitment.root.hex(),
                    "row_leaf_a0": tree.domain.leaf(0, 0, ROW_SCHEMA, value0).hex()})
        if hashing != "h3s":
            msg = words([checked.c[0][:C.CHUNK]]) + words([checked.u[0][:C.CHUNK]])
            out["message_00"] = C.message_digest(msg, hashing).hex()
    if hashing in C.ROW_SEEDED:
        seeds = s.row_seeds(ctx, a, tile.rows)
        out.update({"row_seed_a0": seeds[0].hex(), "row_seeds_sha256": sha(b"".join(seeds)), "e_a0": ops.e_a[0]})
    if hashing == "h3s":
        digests = [C.row_digest(x) for x in a]
        out.update({"row_digest_a0": digests[0].hex(), "row_digests_sha256": sha(b"".join(digests))})
    return out


def vectors() -> dict:
    return {"schema": "verity-pouw/pearl-c-vectors/v1", "cases": [case_vector(*c) for c in CASES]}


if __name__ == "__main__":
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(vectors(), indent=1) + "\n")
    print(f"wrote {OUT}")
