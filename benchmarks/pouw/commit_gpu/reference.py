"""commit_gpu's reference side: its parameter file, its salts, and every leaf, level and opening by the repository's own
functions (``hm96.SHA512``, ``rowleaf``, ``merkle``), so a check compares commit_gpu's output with them bit for bit.

A leaf at position ``pos`` of an n-leaf domain (rank = position) is ``rowleaf.row_leaf(domain, pos, h,
"hm96-sha512/row/v2")`` with ``h = hm96.SHA512.tree_leaf(key, b || c)``, ``b || c = hm96.SHA512.commit_string(key, x,
salt)`` and ``x = rowleaf.sha512_row_v2_digest(row, n_bits)`` (in ``seg`` mode ``x = rowleaf.sha512_row_seg_digest(row,
n_bits)`` under ``"hm96-sha512/row-seg/v1"``); the levels are ``merkle.tree_levels``.  The domain is the e2e
slice's operand domain: ``CommitmentDomain(identity_digest_sha512(
BINDING_TAG, label), -1, RangeIndexedDomain(0, n), hash="sha512")``.  The tile modes commit an m x k int8
operand's :func:`permute` rows, leaf pos the block of r rows x 32 values at (pos % ceil(m/r), pos // ceil(m/r)), so the
spot check's 128-row tile at one 32-step is the 128/r adjacent leaves :func:`tile_leaves` names.

Salts are the committer's secret.  commit_gpu takes them from a file or derives them on the device by SHA-512 alone, the
one hash: ``y = SHA-512(Ks || u64be(pos) || u64be(0)) || ... || SHA-512(Ks || u64be(pos) || u64be(2))`` (:func:`prf_salt`),
Ks a 64-byte salt key.  Either way the root and openings are the reference's for those salts.

    python3 reference.py params OUT N N_BITS [--fresh-key] [--seg] [--ks-seed S]   the CGP1 file for an n-leaf
                                                              domain of n_bits rows (--seg: sha512/row-seg/v1 rows; --ks-seed:
                                                              Ks from seed S, so domains can share it, else random)
    python3 reference.py prepare WORK [quick|full]            every case's inputs and the reference's levels, b || c and leaf
                                                              digests (WORK/cNNN; CPU only, before a GPU lease)
    python3 reference.py check BIN WORK TAG                   BIN's `check` on every prepared case, compared with the
                                                              reference (JSON lines; exit 1 on any difference)

Run from the source tree's root with ``PYTHONPATH=.``; stdlib only.
"""
from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
import random
import struct
import subprocess
import sys
import time

from verity.ml.commitments import hm96
from verity.ml.commitments.identity import identity_digest_sha512
from verity.ml.commitments.indexed import RangeIndexedDomain
from verity.ml.commitments.limits import VerificationLimits
from verity.ml.commitments.merkle import Commitment, CommitmentDomain, MerkleTree, Opening, tree_levels, verify_opening
from verity.ml.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_SEG as SEG
from verity.ml.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_V2 as V2
from verity.ml.commitments.rowleaf import (bit_row, row_leaf, sha512_row_seg_digest, sha512_row_seg_prefix, sha512_row_v2_digest,
                                           sha512_row_v2_prefix)

INST = hm96.SHA512
KEY = INST.default_key
BINDING_TAG = "verity/e2e-slice/operand/v1"
OWNER = -1
E2E_BITS = 1024
#: an int8 value's bf16 bit pattern (exact: |v| < 256)
BF16 = [struct.unpack(">I", struct.pack(">f", float(b - 256 if b > 127 else b)))[0] >> 16 for b in range(256)]


def label(n: int) -> dict:
    return {"session": "commit-gpu", "operand": "A", "leaves": n}


def domain(n: int) -> CommitmentDomain:
    """The operand domain of n leaves."""
    return CommitmentDomain(identity_digest_sha512(BINDING_TAG, label(n)), OWNER, RangeIndexedDomain(0, n), hash="sha512")


def t_table(key: bytes) -> bytes:
    """T_v = xor of (K >> s) over the set bits s of v, K the key as a little-endian integer: 256 rows of 256 bytes.

    M_key is Hankel (column j is key bits j .. j + 511), so ``byte_tables(key)[p][v]`` is bytes [p, p + 64) of T_v;
    :func:`check_t_table` holds the table to that."""

    k = int.from_bytes(INST.check_key(key), "little")
    rows = [0] * 256
    for v in range(1, 256):
        rows[v] = rows[v & (v - 1)] ^ (k >> ((v & -v).bit_length() - 1))
    return b"".join(r.to_bytes(256, "little") for r in rows)


def check_t_table(key: bytes, t: bytes) -> None:
    tables = INST.byte_tables(key)
    for p in range(INST.salt_bytes):
        for v in range(256):
            if tables[p][v] != int.from_bytes(t[256 * v + p:256 * v + p + 64], "little"):
                raise AssertionError(f"T table differs from byte_tables at p {p}, v {v}")


def params(n: int, n_bits: int, ks: bytes, key: bytes = KEY, seg: bool = False, domain_id: bytes | None = None) -> bytes:
    """commit_gpu's CGP1 file: P(n_bits) (row-seg: Q(n_bits)), the salt prefix, leaf_prefix(key), the domain id (default
    :func:`domain`'s; a caller's own ``CommitmentDomain(...).domain_id`` commits under its label), Ks and T."""

    t = t_table(key)
    check_t_table(key, t)
    p = sha512_row_seg_prefix(n_bits) if seg else sha512_row_v2_prefix(n_bits)
    did = domain(n).domain_id if domain_id is None else domain_id
    assert len(did) == 64
    out = b"CGP1" + p + INST.salt_prefix + INST.leaf_prefix(key) + did + ks + t
    assert len(out) == 4 + 128 * 3 + 64 * 2 + 65536
    return out


def prf_salt(ks: bytes, pos: int) -> bytes:
    return b"".join(hashlib.sha512(ks + pos.to_bytes(8, "big") + r.to_bytes(8, "big")).digest() for r in range(3))


def tile_count(m: int, k: int, r: int) -> int:
    return -(-m // r) * (k // 32)


def permute(a: bytes, m: int, k: int, r: int) -> bytes:
    """The tile layout of an m x k int8 operand (row-major, k a multiple of 32) as rows of 32 r bytes: leaf pos
    is block (pos % ceil(m/r), pos // ceil(m/r)) of r rows x 32 values, its bytes the block's in row-major order, rows past
    m zero.  commit_gpu's ``tile`` and ``tileseg`` modes read the operand in place in this order; the reference commits
    these rows as sha512/row/v2 or sha512/row-seg/v1 rows of 256 r bits."""

    assert m > 0 and k > 0 and k % 32 == 0 and r > 0 and len(a) == m * k
    pad = bytes(32 * (-(-m // r) * r - m))
    return b"".join(b"".join(a[i * k + 32 * c:i * k + 32 * c + 32] for i in range(m)) + pad for c in range(k // 32))


def tile_leaves(m: int, r: int, m0: int, c: int) -> range:
    """The leaves a check's 128-row tile (rows m0 .. m0 + 127, m0 and 128 multiples of r) at the 32-step c opens."""

    assert m0 % r == 0 and 128 % r == 0
    mr = -(-m // r)
    return range(c * mr + m0 // r, c * mr + min(mr, (m0 + 128) // r))


def check_footprint(a: bytes, rows: bytes, m: int, k: int, r: int, rng: random.Random, tiles: int = 16) -> int:
    """Each sampled check tile's leaves hold exactly the tile's values (row-major, then zero rows past m)."""

    if 128 % r:
        return 0
    for _ in range(tiles):
        m0, c = 128 * rng.randrange(-(-m // 128)), rng.randrange(k // 32)
        want = b"".join(a[i * k + 32 * c:i * k + 32 * c + 32] if i < m else bytes(32) for i in range(m0, m0 + 128))
        span = tile_leaves(m, r, m0, c)
        got = rows[32 * r * span.start:32 * r * span.stop]
        assert got == want[:len(got)] and not want[len(got):].strip(b"\0"), f"tile ({m0}, {c}) at r {r}"
    return tiles


def e2e_row(a: bytes, pos: int) -> bytes:
    """The e2e slice's row at ``pos``: the 32 int8 at a[32 pos ..] as bf16 words, twice (1024 bits)."""

    words = [BF16[b] for b in a[32 * pos:32 * pos + 32]]
    row, bits = bit_row((words + words, 16))
    assert bits == E2E_BITS
    return row


# --- one check case ----------------------------------------------------------------------------------------------

_CASE: dict = {}


def _leaves(span: tuple[int, int]) -> tuple[bytes, list[bytes], list[bytes]]:
    c = _CASE
    dom = domain(c["n"])
    bc_all, leaves, hs = [], [], []
    for pos in range(*span):
        if c["mode"] == "e2e":
            row = e2e_row(c["rows"], pos)
        else:
            row = c["rows"][pos * c["stride"]:pos * c["stride"] + (c["n_bits"] + 7) // 8]
        salt = prf_salt(c["ks"], pos) if c["salts"] is None else c["salts"][192 * pos:192 * pos + 192]
        x = (sha512_row_seg_digest if c["mode"] == "seg" else sha512_row_v2_digest)(row, c["n_bits"])
        bc = INST.commit_string(c["key"], x, salt)
        h = INST.tree_leaf(c["key"], bc)
        bc_all.append(bc)
        hs.append(h)
        leaves.append(row_leaf(dom, pos, h, schema(c["mode"])))
    return b"".join(bc_all), leaves, hs


def schema(mode: str) -> str:
    return SEG if mode in ("seg", "tileseg") else V2


def _rows_for(mode: str, n: int, n_bits: int, stride: int, rng: random.Random) -> bytes:
    if mode == "e2e":
        return rng.randbytes(32 * n)
    lb = (n_bits + 7) // 8
    rows = bytearray(rng.randbytes(n * stride))
    if n_bits % 8:
        for pos in range(n):
            rows[pos * stride + lb - 1] &= (1 << n_bits % 8) - 1
    return bytes(rows)


def prepare_case(work: str, mode: str, n: int, n_bits: int, stride: int, salts: str, key: bytes, seed: int) -> dict:
    """The case's inputs (params, rows, salts) and the reference's levels, b || c and hm96 leaf digests, under ``work``.
    The tile modes take (m, k, r) for (n, n_bits, stride): rows.bin is the m x k operand, and the reference commits its
    :func:`permute` rows (sha512/row/v2 for ``tile``, sha512/row-seg/v1 for ``tileseg``)."""

    rng = random.Random(seed)
    tile, footprint = {}, 0
    if mode in ("tile", "tileseg"):
        m, k, r = n, n_bits, stride
        operand = rng.randbytes(m * k)
        t0 = time.perf_counter()
        rows = permute(operand, m, k, r)
        footprint = check_footprint(operand, rows, m, k, r, rng)
        tile = {"m": m, "k": k, "r": r, "permute_s": round(time.perf_counter() - t0, 3)}
        n, n_bits, stride, base = tile_count(m, k, r), 256 * r, 32 * r, "rows" if mode == "tile" else "seg"
    else:
        rows = operand = _rows_for(mode, n, n_bits, stride, rng)
        base = mode
    ks = rng.randbytes(64)
    salt_bytes = None if salts == "prf" else rng.randbytes(192 * n)
    t0 = time.perf_counter()
    _CASE.clear()
    _CASE.update(mode=base, n=n, n_bits=n_bits, stride=stride, rows=rows, salts=salt_bytes, ks=ks, key=key)
    step = max(1, min(4096, -(-n // (4 * (os.cpu_count() or 1)))))
    spans = [(a, min(n, a + step)) for a in range(0, n, step)]
    if n < 2048:
        parts = [_leaves(s) for s in spans]
    else:
        with mp.get_context("fork").Pool() as pool:   # forked after _CASE is set: the workers read it
            parts = pool.map(_leaves, spans)
    levels = tree_levels(domain(n), [x for p in parts for x in p[1]])
    files = {"params.bin": params(n, n_bits, ks, key, base == "seg"), "rows.bin": operand, "salts.bin": salt_bytes or b"",
             "ref_levels.bin": b"".join(b"".join(level) for level in levels), "ref_bc.bin": b"".join(p[0] for p in parts),
             "ref_h.bin": b"".join(x for p in parts for x in p[2])}
    os.makedirs(work, exist_ok=True)
    for name, data in files.items():
        with open(os.path.join(work, name), "wb") as f:
            f.write(data)
    case = {"mode": mode, "n": n, "n_bits": n_bits, "stride": stride, "salts": salts, "key": "default" if key == KEY else "fresh",
            "seed": seed, "ref_s": round(time.perf_counter() - t0, 3), **tile}
    if tile:
        case["footprint_tiles"] = footprint
    with open(os.path.join(work, "case.json"), "w") as f:
        json.dump(case, f)
    return case


def check_case(binary: str, work: str, tag: str, openings: int = 64) -> dict:
    """``binary check`` on a prepared case (its output under work/tag), compared with the reference: every level, every
    b || c, and openings: commit_gpu's levels give the path, the reference verifier takes it, a changed value fails, and up
    to 4096 leaves ``MerkleTree.open`` gives the same opening."""

    with open(os.path.join(work, "case.json")) as f:
        case = json.load(f)
    n, out = case["n"], os.path.join(work, tag)
    os.makedirs(out, exist_ok=True)
    t0 = time.perf_counter()
    shape = (case["m"], case["k"], case["r"]) if "r" in case else (n, case["n_bits"], case["stride"])
    gpu = subprocess.run([binary, "check", f"{work}/params.bin", out, case["mode"], *map(str, shape), f"{work}/rows.bin",
                          "prf" if case["salts"] == "prf" else f"{work}/salts.bin"], capture_output=True, text=True)
    result = {**case, "tag": tag, "gpu_s": round(time.perf_counter() - t0, 3)}
    if gpu.returncode:
        return {**result, "ok": False, "error": gpu.stderr.strip()[-400:]}
    read = lambda name: open(name, "rb").read()  # noqa: E731
    g_levels, g_bc = read(f"{out}/levels.bin"), read(f"{out}/bc.bin")
    r_levels, r_bc, hs_flat = read(f"{work}/ref_levels.bin"), read(f"{work}/ref_bc.bin"), read(f"{work}/ref_h.bin")
    for name in ("levels.bin", "bc.bin"):
        os.remove(f"{out}/{name}")
    hs = [hs_flat[64 * i:64 * i + 64] for i in range(n)]
    dom = domain(n)
    depth = (n - 1).bit_length() if n > 1 else 0
    offs = [0]
    for d in range(depth + 1):
        offs.append(offs[-1] + 64 * ((1 << depth) >> d))
    bad_levels = [d for d in range(depth + 1) if g_levels[offs[d]:offs[d + 1]] != r_levels[offs[d]:offs[d + 1]]]
    root = r_levels[offs[depth]:offs[depth + 1]]
    checks = {"levels": not bad_levels and len(g_levels) == len(r_levels) == offs[-1], "bc": g_bc == r_bc,
              "root": g_levels[offs[depth]:offs[depth + 1]] == root}

    sib = lambda flat, d, pos: flat[offs[d] + 64 * ((pos >> d) ^ 1):offs[d] + 64 * ((pos >> d) ^ 1) + 64]  # noqa: E731
    rng = random.Random(case["seed"])
    picks = sorted(set(range(n)) if n <= openings else {0, 1, n - 2, n - 1, n // 2, *rng.sample(range(n), openings - 5)})
    limits, commitment, sch = VerificationLimits(), Commitment(g_levels[offs[depth]:offs[depth + 1]], n), schema(case["mode"])
    tree = MerkleTree(dom, dict(enumerate(hs)), lambda _p: sch) if n <= 4096 else None
    bad_open = []
    for pos in picks:
        path = tuple(sib(g_levels, d, pos) for d in range(depth))
        ok = verify_opening(dom, commitment, Opening(pos, hs[pos], path), sch, limits)
        ok &= not verify_opening(dom, commitment, Opening(pos, bytes([hs[pos][0] ^ 1]) + hs[pos][1:], path), sch, limits)
        ok &= path == tuple(sib(r_levels, d, pos) for d in range(depth))
        if tree is not None:
            ok &= tree.open(pos) == Opening(pos, hs[pos], path) and tree.commitment.root == commitment.root
        if not ok:
            bad_open.append(pos)
    checks["openings"] = not bad_open
    return {**result, "ok": all(checks.values()), "checks": checks, "bad_levels": bad_levels[:8], "bad_openings": bad_open[:8],
            "openings_checked": len(picks), "depth": depth, "root": root.hex()}


def cases(kind: str) -> list[tuple]:
    """(mode, n, n_bits, stride, salts, key), the tile modes (mode, m, k, r, salts, key): ragged leaf counts (pads), every
    padding boundary of a row's last block, given
    and derived salts, a fresh key; row-seg rows from the empty row (no segment) through a short, a full and a one-bit
    last segment to 64 segments; ``full`` adds the e2e slice's o_proj call (2^20 leaves) and int8 rows of 4096 values as
    v2 and as row-seg rows."""

    fresh = INST.fresh_key()
    out = [("e2e", n, E2E_BITS, 0, s, KEY) for n in (1, 2, 3, 5, 7, 8, 255, 256, 257, 511, 513, 1000) for s in ("prf", "given")]
    out += [("e2e", n, E2E_BITS, 0, "prf", KEY) for n in (4095, 4097, 65537)]
    out += [("e2e", 777, E2E_BITS, 0, "given", fresh)]
    for n_bits in (0, 1, 7, 8, 13, 256, 888, 889, 896, 903, 960, 1016, 1024, 1025, 1913, 8192, 32768):
        lb = (n_bits + 7) // 8
        stride = max(16, -(-lb // 16) * 16)
        out += [("rows", 37, n_bits, stride, "given", KEY), ("rows", 300, n_bits, stride + 16, "prf", KEY)]
    out += [("rows", 1025, 256, 32, "prf", fresh)]
    for n_bits in (0, 1, 13, 1016, 1024, 8184, 8191, 8192, 8193, 9095, 16384, 3 * 8192 + 5, 32768, 65536):
        stride = max(16, -(-((n_bits + 7) // 8) // 16) * 16)
        out += [("seg", 37, n_bits, stride, "given", KEY), ("seg", 300, n_bits, stride + 16, "prf", KEY)]
    out += [("seg", 64, 64 * 8192, 65536, "given", KEY), ("seg", 513, 8192 + 128, 1040, "prf", fresh)]
    # tile leaves (m, k, r): ragged m, r not dividing 128, a row/v2 leaf whose padding fits its last block (r 3) or not
    # (r 4), row-seg leaves of one partial, one full, 2, 3, 4 and 64 segments, one leaf, a fresh key
    out += [("tile", m, k, r, s, KEY) for m, k, r, s in ((130, 96, 16, "given"), (7, 32, 4, "prf"), (7, 32, 3, "given"),
                                                        (200, 64, 64, "prf"), (1, 32, 1, "prf"), (1000, 128, 128, "prf"),
                                                        (256, 256, 32, "given"), (129, 64, 128, "given"))]
    out += [("tile", 64, 32, 5, "given", fresh)]
    out += [("tileseg", m, k, r, s, KEY) for m, k, r, s in ((31, 64, 31, "given"), (130, 96, 32, "prf"), (100, 64, 33, "prf"),
                                                           (130, 96, 64, "given"), (300, 32, 65, "prf"), (64, 32, 2048, "given"))]
    out += [("tileseg", 257, 32, 128, "prf", fresh)]
    if kind == "full":
        out += [("e2e", 1 << 20, E2E_BITS, 0, "prf", KEY), ("e2e", (1 << 20) - 3, E2E_BITS, 0, "given", KEY),
                ("rows", 8192, 32768, 4096, "prf", KEY), ("rows", 100003, 256, 32, "prf", KEY),
                ("seg", 8192, 32768, 4096, "prf", KEY), ("seg", 8189, 65536, 8192, "given", KEY)]
        # o_proj's operand at 2K and 8K prefill chunks, as the sweep commits it
        out += [("tile", 2048, 4096, r, "prf", KEY) for r in (16, 32, 64, 128)]
        out += [("tile", 8192, 4096, 16, "given", KEY), ("tile", 8192, 4096, 64, "prf", KEY), ("tile", 8191, 4096, 128, "given", fresh)]
        out += [("tileseg", 2048, 4096, r, "prf", KEY) for r in (64, 128)]
        out += [("tileseg", 8192, 4096, 128, "prf", KEY), ("tileseg", 2047, 4096, 128, "given", KEY)]
    return out


def main(argv: list[str]) -> int:
    if len(argv) >= 5 and argv[1] == "params":
        key = INST.fresh_key() if "--fresh-key" in argv else KEY
        ks = random.Random(int(argv[argv.index("--ks-seed") + 1])).randbytes(64) if "--ks-seed" in argv else os.urandom(64)
        with open(argv[2], "wb") as f:
            f.write(params(int(argv[3]), int(argv[4]), ks, key, "--seg" in argv, domain(int(argv[3])).domain_id))
        return 0
    if len(argv) >= 3 and argv[1] == "prepare":
        kind = argv[3] if len(argv) > 3 else "quick"
        for i, case in enumerate(cases(kind)):
            print(json.dumps(prepare_case(os.path.join(argv[2], f"c{i:03d}"), *case, seed=1000 + i)), flush=True)
        return 0
    if len(argv) >= 5 and argv[1] == "check":
        binary, work, tag = argv[2:5]
        failed = total = 0
        for d in sorted(x for x in os.listdir(work) if x.startswith("c")):
            r = check_case(binary, os.path.join(work, d), tag)
            failed += not r["ok"]
            total += 1
            print(json.dumps(r), flush=True)
        print(json.dumps({"summary": {"tag": tag, "cases": total, "failed": failed}}), flush=True)
        return 1 if failed or not total else 0
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
