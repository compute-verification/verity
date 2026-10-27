"""libvc's conformance on this machine's GPU: every entry point bit for bit against its owner's reference, on fixed vectors
and seeded random cases, and whole commits captured into a CUDA graph and replayed on new inputs.

    python benchmarks/hash/lib/conformance.py [--lib PATH] [--quick]

JSON lines on stdout, one a case ({"case", "ok", ...}), then a summary; exit status 1 if any case differs. The references:
hashlib (SHA-512 and its compression), ``rowleaf`` (sha512/row/v2, sha512/row-seg/v1), ``hm96.SHA512`` and ``merkle``
(the hm96 leaf, frame-v3-sha512's leaf, pad, node, empty root and levels), ``audit`` (A's tree, the transcript tree and
their ChaCha20 salts), ``pearl_c`` (row digests, row seeds, tile leaves), commit_gpu's ``reference.prf_salt`` (the PRF salt
source) and verity_vllm's ``serving_rows.salts`` (RFC 8439 salts; the audit identity below when it doesn't import). Openings
are checked by ``merkle.verify_opening`` and ``multiproof.verify_multiproof``, and against ``MerkleTree.open`` and ``multiproof``.

Every output and scratch buffer (salt records, b || c, h, leaves, every tree level and the root, opening records) is 0xA5
before each call whose output is checked, a repeated call or graph replay included, and every byte of it is compared with the
reference. Each buffer has 64 guard bytes of 0xA5 past its end, checked after each case. Given salts are the reference salt
source's, never a constant. Controls the same checks must reject: no call; the leaves of half the rows; leaves over salt
records nobody wrote; opening records nobody wrote.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

import vclib  # noqa: E402
from verity.core.protocols.pouw import audit  # noqa: E402
from verity.core.protocols.pouw import weights as Wt  # noqa: E402
from verity.core.protocols.pouw.schemes import pearl_c  # noqa: E402
from verity.core.protocols.sampled_proofs.one_stage import registered as G  # noqa: E402
from verity.ml.commitments import RangeIndexedDomain, hm96  # noqa: E402
from verity.ml.commitments.limits import VerificationLimits  # noqa: E402
from verity.ml.commitments.merkle import CommitmentDomain, MerkleTree, _message, _uint, tree_levels, verify_opening  # noqa: E402
from verity.ml.commitments.merkle import Opening as MerkleOpening  # noqa: E402
from verity.ml.commitments.multiproof import multiproof, multiproof_from_paths, verify_multiproof  # noqa: E402
from verity.ml.commitments.rowleaf import (ROLE_X, SCHEMA_HM96_SHA512_ROW, row_bytes, sha512_row_digest,  # noqa: E402
                                           sha512_row_seg_digest, sha512_row_v2_digest)

KEY = hm96.SHA512.default_key
SALT = hm96.SHA512.salt_bytes
LIMITS = VerificationLimits()
POISON, GUARD = 0xA5, 64


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


CG_REF = _load("commit_gpu_reference", ROOT / "benchmarks/pouw/commit_gpu/reference.py")


#: the split-K rows' vectors' generator: its cases (`SV.CASES`), their rows and keys, and credit's reading of a case
#: (`SV.commit`: `h1zy.chunk_words` under `Registered.commit`); `SV.OUT` the pinned vectors, every leaf's words_sha512,
#: row_digest and frame_leaf and each value's domain and root
SV = _load("splitk_rows_vectors", ROOT / "verity/core/protocols/pouw/tests/splitk_rows_vectors.py")
#: a K at or below weights.CHUNK_MAX, whose rows are their own leaves (no vector: the layout there is unchanged)
WHOLE = ("k-4096", "a/4096@0", 4096, 64, SV.A_NONCE, True)


def splitk_vectors() -> dict:
    """The pinned vectors by case name."""
    return {c["case"]: c for c in json.loads(SV.OUT.read_text())["cases"]}


def rfc_salts_identity(key: bytes, nonce: int, n: int) -> bytes:
    """RFC 8439's block (32-bit counter, 12-byte nonce n0 n1 n2) is audit's (64-bit counter, 64-bit nonce) at counter = ctr +
    2^32 n0, nonce = n1 + 2^32 n2: the n salts of serving_rows.salts(key, nonce, n) from audit.chacha20_block."""
    n0, n1 = nonce & 0xFFFFFFFF, nonce >> 32
    return b"".join(audit.chacha20_block(key, b + (n0 << 32), n1) for b in range(3 * n))


def rfc_salts(key: bytes, nonce: int, n: int) -> tuple[bytes, str]:
    try:
        sys.path.insert(0, str(ROOT / "integrations/vllm"))
        from verity.ml.windows.rows import salts
        return salts(key, nonce, n), "serving_rows.salts"
    except Exception:  # noqa: BLE001 - its numpy-side dependencies may be absent on a node
        return rfc_salts_identity(key, nonce, n), "audit.chacha20_block identity"


class Dev:
    def __init__(self, lib_path: str | None) -> None:
        self.cuda = vclib.Cuda()
        self.lib = vclib.Lib(lib_path)
        self.s = self.cuda.stream()
        self.held: dict[int, int] = {}   # each buffer's bytes, before its guard

    def up(self, data: bytes) -> int:
        p = self.buf(len(data))
        self.cuda.h2d(p, data, self.s)
        return p

    def buf(self, n: int) -> int:
        p = self.cuda.malloc(n + GUARD)
        self.cuda.memset(p, POISON, n + GUARD, self.s)
        self.held[p] = n
        return p

    def poison(self, *ps: int) -> None:
        for p in ps:
            self.cuda.memset(p, POISON, self.held[p], self.s)

    def guards_intact(self) -> bool:
        self.cuda.sync(self.s)
        return all(self.cuda.d2h(p + n, GUARD, self.s) == bytes([POISON]) * GUARD for p, n in self.held.items())

    def down(self, p: int, n: int) -> bytes:
        self.cuda.sync(self.s)
        return self.cuda.d2h(p, n, self.s)

    def tables(self, plan: vclib.Plan) -> int:
        t = self.buf(plan.tables_bytes)
        self.lib.tables_launch(plan.ptr, t, self.s)
        return t

    def release(self) -> None:
        self.cuda.sync(self.s)
        for p in self.held:
            self.cuda.free(p)
        self.held.clear()


RESULTS: list[dict] = []


def report(case: str, ok: bool, **kw) -> None:
    rec = {"case": case, "ok": bool(ok), **kw}
    RESULTS.append(rec)
    print(json.dumps(rec), flush=True)


def first_diff(a: list[bytes], b: list[bytes]) -> int | None:
    for i, (x, y) in enumerate(zip(a, b, strict=True)):
        if x != y:
            return i
    return None


def chunks(raw: bytes, size: int) -> list[bytes]:
    return [raw[i:i + size] for i in range(0, len(raw), size)]


def domain(rng: random.Random, n: int) -> CommitmentDomain:
    return CommitmentDomain(rng.randbytes(64), rng.randrange(0, 1000), RangeIndexedDomain(n), hash="sha512")


def rows_of(rng: random.Random, n: int, n_bits: int) -> list[bytes]:
    lb = (n_bits + 7) // 8
    out = []
    for _ in range(n):
        r = bytearray(rng.randbytes(lb))
        if n_bits % 8:
            r[-1] &= (1 << (n_bits % 8)) - 1
        out.append(bytes(r))
    return out


def packed(rows: list[bytes], stride: int) -> bytes:
    return b"".join(r.ljust(stride, b"\0") for r in rows)


def stride_of(lb: int) -> int:
    return (lb + 15) // 16 * 16 + 16


# --- primitives ------------------------------------------------------------------------------------------------------


def case_sha512(d: Dev, rng: random.Random, quick: bool) -> None:
    for ln in (0, 1, 55, 111, 112, 113, 127, 128, 129, 239, 240, 255, 256, 1000, 4096, 4097) + (() if quick else (70000,)):
        n = 7
        msgs = [rng.randbytes(ln) for _ in range(n)]
        st = stride_of(ln)
        src, out = d.up(packed(msgs, st)), d.buf(64 * n)
        d.lib.sha512_launch(src, st, ln, n, out, d.s)
        got = chunks(d.down(out, 64 * n), 64)
        want = [hashlib.sha512(m).digest() for m in msgs]
        report("sha512", got == want, len=ln, n=n, first_diff=first_diff(got, want))


def _pad1(m: bytes) -> bytes:
    return m + b"\x80" + bytes(128 - len(m) - 17) + (8 * len(m)).to_bytes(16, "big")


IV = bytes.fromhex("6a09e667f3bcc908bb67ae8584caa73b3c6ef372fe94f82ba54ff53a5f1d36f1510e527fade682d19b05688c2b3e6c1f1f83d9abfb41bd6b5be0cd19137e2179")


def case_compress(d: Dev, rng: random.Random, quick: bool) -> None:
    n = 64
    msgs = [rng.randbytes(rng.randrange(0, 112)) for _ in range(n)]
    states, blocks, out = d.up(IV * n), d.up(b"".join(_pad1(m) for m in msgs)), d.buf(64 * n)
    d.lib.compress_launch(states, blocks, n, out, d.s)
    got = chunks(d.down(out, 64 * n), 64)
    want = [hashlib.sha512(m).digest() for m in msgs]
    report("compress/one-block", got == want, n=n, first_diff=first_diff(got, want))
    msgs = [rng.randbytes(rng.randrange(128, 240)) for _ in range(n)]
    tails = []
    for m in msgs:
        full = m + b"\x80" + bytes((-len(m) - 17) % 128) + (8 * len(m)).to_bytes(16, "big")
        tails.append((full[:128], full[128:256]))
    mid = d.buf(64 * n)
    d.lib.compress_launch(states, d.up(b"".join(t[0] for t in tails)), n, mid, d.s)
    d.lib.compress_launch(mid, d.up(b"".join(t[1] for t in tails)), n, out, d.s)
    got = chunks(d.down(out, 64 * n), 64)
    want = [hashlib.sha512(m).digest() for m in msgs]
    report("compress/chained", got == want, n=n, first_diff=first_diff(got, want))


# --- row digests -----------------------------------------------------------------------------------------------------


def case_rows(d: Dev, rng: random.Random, quick: bool) -> None:
    did = rng.randbytes(64)
    for n_bits in (8, 3, 8 * 111, 8 * 112, 8 * 128, 8 * 128 + 3, 2048, 8 * 1000, 32768, 8 * 4096 * 3 + 8):
        n = 9
        rows = rows_of(rng, n, n_bits)
        plan = d.lib.plan(vclib.Spec(**vclib.row_v2_spec(did, n_bits)))
        st = stride_of(len(rows[0]))
        out = d.buf(64 * n)
        d.lib.rows_launch(plan.ptr, None, d.up(packed(rows, st)), st, n, out, None, d.s)
        got = chunks(d.down(out, 64 * n), 64)
        want = [sha512_row_v2_digest(r, n_bits) for r in rows]
        report("row/v2", got == want, n_bits=n_bits, n=n, first_diff=first_diff(got, want))
    # one launch packs blockDim / s rows a block up to 128 segments, loops past it, and takes two launches past 768
    shapes = [(b, 5) for b in (8, 5, 8 * 1024, 8 * 1025, 8 * 1024 * 3 + 5, 32768, 8 * 4 * 256, 8 * 4 * 4096, 8 * 1024 * 7 + 8 * 513)]
    shapes += [(8 * 1024 * 3 + 5, 70), (8 * 1024 * 56, 9), (8 * 1024 * 200 + 56, 3), (8 * 1024 * 769 + 8, 2)]
    shapes += [] if quick else [(8 * 4 * 32768, 5)]
    # served row counts (a prefill step's 4,096 rows, decode's 128 and 32), odd and even segment counts: 57 with a short last
    # segment, 6 and 7 packed 5 and 4 rows a block (4,101 leaves a ragged last block), 112 and 129 either side of the loop,
    # 767 and 768 the one launch's last, 770 the two launches'
    shapes += [(8 * (1024 * 56 + 40), 4096), (8 * 1024 * 6, 4096), (8 * 1024 * 7, 4101), (8 * 1024 * 112, 128),
               (8 * 1024 * 129, 128), (8 * 1024 * 767, 32), (8 * 1024 * 768, 32), (8 * 1024 * 770, 32)]
    for n_bits, n in shapes:
        rows = rows_of(rng, n, n_bits)
        plan = d.lib.plan(vclib.Spec(**vclib.row_seg_spec(did, n_bits)))
        st = stride_of(len(rows[0]))
        out, scratch = d.buf(64 * n), d.buf(d.lib.rows_scratch_bytes(plan.ptr, n))
        d.lib.rows_launch(plan.ptr, d.tables(plan), d.up(packed(rows, st)), st, n, out, scratch, d.s)
        got = chunks(d.down(out, 64 * n), 64)
        want = [sha512_row_seg_digest(r, n_bits) for r in rows]
        report("row-seg/v1", got == want, n_bits=n_bits, n=n, first_diff=first_diff(got, want))


# --- hm96 leaves in frame-v3-sha512 trees -----------------------------------------------------------------------------


def salts_for(source: tuple, positions: list[int]) -> tuple[list[bytes], str]:
    kind = source[0]
    if kind == "prf":
        return [CG_REF.prf_salt(source[1], p) for p in positions], "commit_gpu reference.prf_salt"
    if kind == "chacha64":
        return [audit.hm96_salt(source[1], source[2], p) for p in positions], "audit.hm96_salt"
    if kind == "chacha-rfc":
        raw, ref = rfc_salts(source[1], source[2], max(positions) + 1)
        return [raw[SALT * p:SALT * (p + 1)] for p in positions], ref
    raise ValueError(kind)


def hm_leaf(dom: CommitmentDomain, pos: int, schema: str, x: bytes, y: bytes) -> tuple[bytes, bytes, bytes]:
    bc = hm96.SHA512.commit_string(KEY, x, y)
    h = hm96.SHA512.tree_leaf(KEY, bc)
    return dom.leaf(pos, pos, schema, h), bc, h


def levels_of(raw: bytes, width: int) -> list[list[bytes]]:
    out, off, w = [], 0, width
    while w >= 1:
        out.append(chunks(raw[64 * off:64 * (off + w)], 64))
        off += w
        if w == 1:
            break
        w //= 2
    return out


def same_levels(got: list[list[bytes]], want) -> tuple[bool, str | None]:
    if len(got) != len(want):
        return False, f"{len(got)} levels, want {len(want)}"
    for lv, (g, w) in enumerate(zip(got, want)):
        i = first_diff(g, list(w))
        if i is not None:
            return False, f"level {lv} node {i}"
    return True, None


FORMS = ((vclib.FORM_SPLIT, "split"), (vclib.FORM_FUSED, "fused"), (vclib.FORM_FOLD, "fold"), (vclib.FORM_SPEC, "spec"),
         (vclib.FORM_AUTO, "auto"))


def case_hm96_rows(d: Dev, rng: random.Random, quick: bool) -> None:
    n_bits, schema = 32768, "hm96-sha512/row/v2"
    sources = [("prf", rng.randbytes(64)), ("given",), ("chacha64", rng.randbytes(32), rng.randrange(1 << 64)),
               ("chacha-rfc", rng.randbytes(32), rng.randrange(1 << 40))]
    for src in sources:
        for n in ((1, 3, 300) if quick else (1, 2, 3, 300, 1025)):
            dom = domain(rng, n)
            rows = rows_of(rng, n, n_bits)
            ys, ref = salts_for(src if src[0] != "given" else ("chacha64", rng.randbytes(32), 7), list(range(n)))
            want_leaves, want_bc, want_h = [], [], []
            for i, (r, y) in enumerate(zip(rows, ys)):
                lf, bc, h = hm_leaf(dom, i, schema, sha512_row_v2_digest(r, n_bits), y)
                want_leaves.append(lf), want_bc.append(bc), want_h.append(h)
            want = tree_levels(dom, want_leaves)
            plan = d.lib.plan(vclib.hm96_rows(dom.domain_id, n_bits, salt=src, schema=schema))
            tab = d.tables(plan)
            st = stride_of(n_bits // 8)
            rp = d.up(packed(rows, st))
            sp = d.up(b"".join(ys)) if src[0] == "given" else None
            width, size = d.lib.tree_width(n), d.lib.tree_bytes(n)

            def judged(lv: int, bc: int) -> tuple[bool, str | None, int | None]:
                ok, where = same_levels(levels_of(d.down(lv, size), width), want)
                got_bc = chunks(d.down(bc, 128 * n), 128)
                return ok and got_bc == want_bc, where, first_diff(got_bc, want_bc)
            for form, fname in FORMS:
                lv, scratch, bc = d.buf(size), d.buf(d.lib.commit_scratch_bytes(n)), d.buf(128 * n)
                for call in range(2):   # each from poison, scratch included: neither call may pass on the other's writes
                    d.poison(lv, bc, scratch)
                    d.lib.commit_rows_launch(plan.ptr, tab, rp, st, sp, n, lv, bc, scratch, form, d.s)
                    ok, where, bc_diff = judged(lv, bc)
                    report("hm96/row-v2/commit", ok, salt=src[0], salt_ref=ref, n=n, form=fname, call=call, where=where, bc_diff=bc_diff)
            # the phases one by one, with b || c and h
            lv, scratch = d.buf(size), d.buf(d.lib.salt_scratch_bytes(width))
            bc, hs = d.buf(128 * n), d.buf(64 * n)
            d.lib.salts_launch(plan.ptr, tab, sp, 0, width, n, scratch, lv, d.s)
            d.lib.leaves_rows_launch(plan.ptr, rp, st, 0, n, n, scratch, lv, bc, hs, d.s)
            d.lib.tree_launch(plan.ptr, lv, n, d.s)
            ok, where, bc_diff = judged(lv, bc)
            got_h = chunks(d.down(hs, 64 * n), 64)
            report("hm96/row-v2/phases", ok and got_h == want_h, salt=src[0], n=n, where=where, bc_diff=bc_diff,
                   h_diff=first_diff(got_h, want_h))
            if n < 2:
                continue
            # controls the same judgment must reject: no call, the leaves of half the rows, leaves over unwritten salt records
            d.poison(lv, scratch, bc, hs)
            no_call = judged(lv, bc)[0]
            d.lib.salts_launch(plan.ptr, tab, sp, 0, width, n, scratch, lv, d.s)
            d.lib.leaves_rows_launch(plan.ptr, rp, st, 0, n // 2, n, scratch, lv, bc, hs, d.s)
            d.lib.tree_launch(plan.ptr, lv, n, d.s)
            half = judged(lv, bc)[0]
            d.poison(lv, scratch, bc, hs)
            d.lib.leaves_rows_launch(plan.ptr, rp, st, 0, n, n, scratch, lv, bc, hs, d.s)
            d.lib.tree_launch(plan.ptr, lv, n, d.s)
            no_salt = judged(lv, bc)[0]
            no_salt_bc = chunks(d.down(bc, 128 * n), 128) == want_bc
            report("hm96/row-v2/controls-rejected", not (no_call or half or no_salt or no_salt_bc), salt=src[0], n=n,
                   accepted=[k for k, v in (("no call", no_call), ("half the rows", half), ("no salt write", no_salt),
                                            ("no salt write, b||c", no_salt_bc)) if v] or None)


def case_fold_wide(d: Dev, rng: random.Random, quick: bool) -> None:
    """The commit over trees of many CTAs (FOLD's climb over up to 10 levels above its 256-leaf subtrees and SPEC's over 13
    above its warps' 32, siblings racing; node indices of 3 bytes at width 2^18; AUTO on both sides of its switch), short rows
    to keep the reference cheap, twice in a row, levels and scratch (the climb's counters included) poisoned before each."""
    n_bits, schema = 1024, "hm96-sha512/row/v2"
    for n in ((5000, 70000) if quick else (20000, 32768, 140000)):
        dom = domain(rng, n)
        rows = rows_of(rng, n, n_bits)
        ks = rng.randbytes(64)
        leaves = [hm_leaf(dom, i, schema, sha512_row_v2_digest(r, n_bits), CG_REF.prf_salt(ks, i))[0] for i, r in enumerate(rows)]
        want = tree_levels(dom, leaves)
        plan = d.lib.plan(vclib.hm96_rows(dom.domain_id, n_bits, salt=("prf", ks), schema=schema))
        tab = d.tables(plan)
        st = stride_of(n_bits // 8)
        rp = d.up(packed(rows, st))
        width, size = d.lib.tree_width(n), d.lib.tree_bytes(n)
        for form, fname in ((vclib.FORM_FOLD, "fold"), (vclib.FORM_SPEC, "spec"), (vclib.FORM_SPLIT, "split"),
                            (vclib.FORM_AUTO, "auto")):
            lv, sc = d.buf(size), d.buf(d.lib.commit_scratch_bytes(n))
            ok_all, where = True, None
            for _ in range(2):
                d.poison(lv, sc)
                d.lib.commit_rows_launch(plan.ptr, tab, rp, st, None, n, lv, None, sc, form, d.s)
                ok, w = same_levels(levels_of(d.down(lv, size), width), want)
                ok_all, where = ok_all and ok, where or w
            report("hm96/row-v2/wide", ok_all, form=fname, n=n, width=width, where=where)


def case_hm96_x(d: Dev, rng: random.Random, quick: bool) -> None:
    """Given inner digests, positions in windows that cross each length of uint(pos): 1 -> 2, 2 -> 3, 3 -> 4 bytes."""
    dom = CommitmentDomain(rng.randbytes(64), 3, RangeIndexedDomain(1 << 32), hash="sha512")
    for schema in ("hm96-sha512/row-seg/v1", "hm96-sha512/row/v1"):
        for src in (("chacha64", rng.randbytes(32), rng.randrange(1 << 64)), ("prf", rng.randbytes(64)), ("given",)):
            plan = d.lib.plan(vclib.hm96_digests(dom.domain_id, salt=src, schema=schema))
            tab = d.tables(plan)
            for first in (0, 250, (1 << 16) - 6, (1 << 24) - 6, (1 << 32) - 12):
                m, n_total = 12, first + 8
                pos = list(range(first, first + m))
                xs = [rng.randbytes(64) for _ in range(m)]
                ys, ref = salts_for(src if src[0] != "given" else ("chacha64", rng.randbytes(32), 9), pos)
                want = [hm_leaf(dom, p, schema, x, y)[0] if p < n_total else dom.pad(p) for p, x, y in zip(pos, xs, ys)]
                xp = d.up(b"".join(xs))
                sp = d.up(b"".join(ys)) if src[0] == "given" else None
                lv, scratch = d.buf(64 * m), d.buf(d.lib.salt_scratch_bytes(m))
                d.lib.salts_launch(plan.ptr, tab, sp, first, m, n_total, scratch, lv, d.s)
                d.lib.leaves_x_launch(plan.ptr, xp, first, m, n_total, scratch, lv, None, None, d.s)
                got = chunks(d.down(lv, 64 * m), 64)
                report("hm96/x/split", got == want, schema=schema, salt=src[0], salt_ref=ref, first=first, first_diff=first_diff(got, want))
                lv2 = d.buf(64 * m)
                d.lib.leaves_fused_launch(plan.ptr, tab, None, 0, xp, sp, first, m, n_total, lv2, None, None, d.s)
                got = chunks(d.down(lv2, 64 * m), 64)
                report("hm96/x/fused", got == want, schema=schema, salt=src[0], first=first, first_diff=first_diff(got, want))


# --- the frame alone --------------------------------------------------------------------------------------------------


def case_frame(d: Dev, rng: random.Random, quick: bool) -> None:
    for schema in ("hm96-sha512/row/v2", "hm96-sha512/row-seg/v1"):
        for n in ((0, 1, 2, 5, 513) if quick else (0, 1, 2, 5, 513, 1100, 70000)):
            dom = domain(rng, n)
            plan = d.lib.plan(vclib.frame_spec(dom.domain_id, schema))
            if n == 0:
                lv = d.buf(64)
                d.lib.tree_launch(plan.ptr, lv, 0, d.s)
                got = d.down(lv, 64)
                report("frame/empty-root", got == dom.empty_root() and plan.empty_root() == dom.empty_root(), schema=schema)
                continue
            vals = [rng.randbytes(64) for _ in range(n)]
            want = tree_levels(dom, [dom.leaf(i, i, schema, v) for i, v in enumerate(vals)])
            width = d.lib.tree_width(n)
            lv = d.buf(d.lib.tree_bytes(n))
            d.lib.frame_leaves_launch(plan.ptr, d.up(b"".join(vals)), 0, width, n, lv, d.s)
            d.lib.tree_launch(plan.ptr, lv, n, d.s)
            ok, where = same_levels(levels_of(d.down(lv, d.lib.tree_bytes(n)), width), want)
            report("frame/leaves+pads+tree", ok, schema=schema, n=n, where=where)
    dom = CommitmentDomain(rng.randbytes(64), -1, RangeIndexedDomain(1 << 32), hash="sha512")
    plan = d.lib.plan(vclib.frame_spec(dom.domain_id, "hm96-sha512/row/v2"))
    for level, first in ((0, 0), (3, 250), (7, (1 << 16) - 3), (20, (1 << 24) - 3), (31, 1)):
        m = 6
        kids = [rng.randbytes(64) for _ in range(2 * m)]
        out = d.buf(64 * m)
        d.lib.nodes_launch(plan.ptr, d.up(b"".join(kids)), out, m, level, first, d.s)
        got = chunks(d.down(out, 64 * m), 64)
        want = [dom.node(level, first + i, kids[2 * i], kids[2 * i + 1]) for i in range(m)]
        report("frame/nodes", got == want, level=level, first_index=first, first_diff=first_diff(got, want))
    for first in (0, 250, (1 << 16) - 3, (1 << 24) - 3):
        out = d.buf(64 * 6)
        d.lib.frame_leaves_launch(plan.ptr, None, first, 6, 0, out, d.s)
        got = chunks(d.down(out, 64 * 6), 64)
        want = [dom.pad(first + i) for i in range(6)]
        report("frame/pads", got == want, first=first, first_diff=first_diff(got, want))


def case_arity(d: Dev, rng: random.Random, quick: bool) -> None:
    """The what-if tree: node(level, index, c_0 .. c_(a-1)) as frame-v3's node message with a child parts."""
    schema = "hm96-sha512/row/v2"
    for arity in (2, 4, 8):
        for n in (1, 5, 70):
            dom = domain(rng, n)
            plan = d.lib.plan(vclib.frame_spec(dom.domain_id, schema))
            vals = [rng.randbytes(64) for _ in range(n)]
            width = d.lib.tree_arity_width(n, arity)
            level = [dom.leaf(i, i, schema, v) for i, v in enumerate(vals)] + [dom.pad(r) for r in range(n, width)]
            want, lv_i = [level], 0
            while len(level) > 1:
                level = [hashlib.sha512(_message(b"node", dom.domain_id, _uint(lv_i), _uint(j), *level[arity * j:arity * (j + 1)])).digest()
                         for j in range(len(level) // arity)]
                want.append(level)
                lv_i += 1
            total = d.lib.tree_arity_bytes(n, arity)
            lv = d.buf(total)
            d.lib.frame_leaves_launch(plan.ptr, d.up(b"".join(vals)), 0, width, n, lv, d.s)
            d.lib.tree_arity_launch(plan.ptr, lv, n, arity, d.s)
            raw, got, off, w = d.down(lv, total), [], 0, width
            while True:
                got.append(chunks(raw[64 * off:64 * (off + w)], 64))
                off += w
                if w == 1:
                    break
                w //= arity
            ok, where = same_levels(got, want)
            if arity == 2:
                ok = ok and same_levels(got, tree_levels(dom, [dom.leaf(i, i, schema, v) for i, v in enumerate(vals)]))[0]
            report("what-if/arity", ok, arity=arity, n=n, where=where)


# --- Pearl-C -h3 ----------------------------------------------------------------------------------------------------


def f32_rows(rng: random.Random, n: int, k: int) -> list[bytes]:
    return [rng.randbytes(4 * k) for _ in range(n)]


def case_pearl(d: Dev, rng: random.Random, quick: bool) -> None:
    # Llama-3.1-8B's k (4,096 and 14,336: 16 and 56 segments) at decode's m (32, 128) and a prefill step's 4,096 rows
    served = [(k, m) for k in (4096, 14336) for m in (32, 128, 4096)]
    for k, n in [(256, 1), (256, 7), (1024, 64), (4096, 9), *served]:
        did = rng.randbytes(64)
        rows = f32_rows(rng, n, k)
        plan = d.lib.plan(vclib.seeds_spec(did, 32 * k))
        out, scratch = d.buf(64 * n), d.buf(d.lib.rows_scratch_bytes(plan.ptr, n))
        rp = d.up(b"".join(rows))
        d.lib.rows_launch(plan.ptr, d.tables(plan), rp, 4 * k, n, out, scratch, d.s)
        got = chunks(d.down(out, 64 * n), 64)
        want = [pearl_c.row_digest(r) for r in rows]
        report("pearl-c/row_digest", got == want, k=k, n=n, first_diff=first_diff(got, want))
        salt, root_b, unit = rng.randbytes(32), rng.randbytes(64), rng.randrange(1 << 32)
        seeds = d.buf(32 * n)
        d.lib.row_seeds_launch(plan.ptr, out, n, d.up(salt), d.up(root_b), unit, seeds, d.s)
        got = chunks(d.down(seeds, 32 * n), 32)
        want = [pearl_c.row_seed(salt, dg, root_b, unit, i, "h3") for i, dg in enumerate(want)]
        report("pearl-c/row_seed", got == want, k=k, n=n, unit=unit, first_diff=first_diff(got, want))
    for rows, cols in ((64, 64), (128, 192), (80, 128), (2, 64), (16, 64)):
        did = rng.randbytes(64)
        C = [[rng.getrandbits(32) for _ in range(cols)] for _ in range(rows)]
        U = [[rng.getrandbits(32) for _ in range(cols)] for _ in range(rows)]
        plan = d.lib.plan(vclib.tiles_spec(did, rows, cols))
        nt = d.lib.tiles_count(plan.ptr)
        out, scratch = d.buf(64 * nt), d.buf(d.lib.tiles_scratch_bytes(plan.ptr))
        cp = d.up(b"".join(struct.pack(f"<{cols}I", *r) for r in C))
        up = d.up(b"".join(struct.pack(f"<{cols}I", *r) for r in U))
        d.lib.tiles_launch(plan.ptr, d.tables(plan), cp, up, out, scratch, d.s)
        got = chunks(d.down(out, 64 * nt), 64)
        want = []
        for ti in range(0, rows, 64):
            for tj in range(0, cols, 64):
                ck = pearl_c.Checked(c=[r[tj:tj + 64] for r in C[ti:ti + 64]], u=[r[tj:tj + 64] for r in U[ti:ti + 64]])
                want.append(pearl_c.leaf(ck, "h3"))
        report("pearl-c/tile_leaf", got == want, rows=rows, cols=cols, tiles=nt, first_diff=first_diff(got, want))
    # A's tree (audit.commit_rows under HM96: x the row's row-seg digest, ChaCha20 salts with nonce = the unit) and the
    # transcript tree (digests: x the tile leaf itself, nonce HM96_TRANSCRIPT)
    for k, n in ((256, 5), (1024, 300) if not quick else (1024, 40)):
        binding, unit, key = rng.randbytes(64), rng.randrange(1 << 20), rng.randbytes(32)
        rows = f32_rows(rng, n, k)
        tree = audit.commit_rows(binding, unit, rows, hash=audit.HM96, salt_key=key, nonce=unit)
        want = tree._levels
        dom = tree.domain
        plan = d.lib.plan(vclib.hm96_digests(dom.domain_id, salt=("chacha64", key, unit), row_bits=32 * k))
        tab = d.tables(plan)
        width = d.lib.tree_width(n)
        x, rs = d.buf(64 * n), d.buf(d.lib.rows_scratch_bytes(plan.ptr, n))
        lv, sc = d.buf(d.lib.tree_bytes(n)), d.buf(d.lib.salt_scratch_bytes(width))
        rp = d.up(b"".join(rows))

        def a_tree(s: int) -> None:
            d.lib.rows_launch(plan.ptr, tab, rp, 4 * k, n, x, rs, s)
            d.lib.salts_launch(plan.ptr, tab, None, 0, width, n, sc, lv, s)
            d.lib.leaves_x_launch(plan.ptr, x, 0, n, n, sc, lv, None, None, s)
            d.lib.tree_launch(plan.ptr, lv, n, s)
        a_tree(d.s)
        ok, where = same_levels(levels_of(d.down(lv, d.lib.tree_bytes(n)), width), want)
        report("pearl-c/A-tree", ok, k=k, n=n, root=tree.commitment.root.hex()[:16], where=where)
        # the same pipeline captured into a graph, replayed over new rows
        graph = d.cuda.capture(d.s, a_tree)
        ok_all = True
        for _ in range(2):
            rows = f32_rows(rng, n, k)
            d.cuda.h2d(rp, b"".join(rows), d.s)
            d.poison(x, rs, lv, sc)
            d.cuda.replay(graph, d.s)
            want = audit.commit_rows(binding, unit, rows, hash=audit.HM96, salt_key=key, nonce=unit)._levels
            ok_all &= same_levels(levels_of(d.down(lv, d.lib.tree_bytes(n)), width), want)[0]
        report("graph/A-tree-replay", ok_all, k=k, n=n, replays=2)
    n = 37
    binding, key = rng.randbytes(64), rng.randbytes(32)
    leaves = [rng.randbytes(64) for _ in range(n)]
    tree = audit.commit_rows(binding, -1, leaves, hash=audit.HM96, salt_key=key, nonce=audit.HM96_TRANSCRIPT, digests=True)
    plan = d.lib.plan(vclib.hm96_digests(tree.domain.domain_id, salt=("chacha64", key, audit.HM96_TRANSCRIPT)))
    tab, width = d.tables(plan), d.lib.tree_width(n)
    lv, sc = d.buf(d.lib.tree_bytes(n)), d.buf(d.lib.salt_scratch_bytes(width))
    d.lib.salts_launch(plan.ptr, tab, None, 0, width, n, sc, lv, d.s)
    d.lib.leaves_x_launch(plan.ptr, d.up(b"".join(leaves)), 0, n, n, sc, lv, None, None, d.s)
    d.lib.tree_launch(plan.ptr, lv, n, d.s)
    ok, where = same_levels(levels_of(d.down(lv, d.lib.tree_bytes(n)), width), tree._levels)
    report("pearl-c/transcript-tree", ok, n=n, where=where)


def case_graph(d: Dev, rng: random.Random, quick: bool) -> None:
    """A served step's commit (salts, leaves, tree) captured once and replayed over new rows, in both forms."""
    n_bits, schema, n = 32768, "hm96-sha512/row/v2", 300
    dom = domain(rng, n)
    ks = rng.randbytes(64)
    plan = d.lib.plan(vclib.hm96_rows(dom.domain_id, n_bits, salt=("prf", ks), schema=schema))
    tab = d.tables(plan)
    st = n_bits // 8
    rp = d.buf(st * n)
    width = d.lib.tree_width(n)
    ys = [CG_REF.prf_salt(ks, p) for p in range(n)]
    for form, fname in FORMS:
        lv, sc = d.buf(d.lib.tree_bytes(n)), d.buf(d.lib.commit_scratch_bytes(n))
        graph = d.cuda.capture(d.s, lambda s: d.lib.commit_rows_launch(plan.ptr, tab, rp, st, None, n, lv, None, sc, form, s))
        ok_all = True
        for _ in range(2):
            rows = rows_of(rng, n, n_bits)
            d.cuda.h2d(rp, b"".join(rows), d.s)
            d.poison(lv, sc)
            d.cuda.replay(graph, d.s)
            want = tree_levels(dom, [hm_leaf(dom, i, schema, sha512_row_v2_digest(r, n_bits), y)[0] for i, (r, y) in enumerate(zip(rows, ys))])
            ok_all &= same_levels(levels_of(d.down(lv, d.lib.tree_bytes(n)), width), want)[0]
        report("graph/commit-replay", ok_all, form=fname, n=n, replays=2)


# --- openings --------------------------------------------------------------------------------------------------------


def open_records(d: Dev, plan: vclib.Plan, tab, x, bc, sp, lv: int, n: int, positions: list[int], graph: bool = False,
                 call: bool = True) -> list[vclib.Opening]:
    """The records vc_open_launch writes into a poisoned buffer; with call=False, the no-write control's."""
    m, size = len(positions), d.lib.open_bytes(n)
    pp, out = d.up(struct.pack(f"<{m}I", *positions)), d.buf(size * m)

    def launch(s: int) -> None:
        d.lib.open_launch(plan.ptr, tab, x, bc, sp, lv, n, pp, m, out, s)
    if call and graph:
        d.cuda.replay(d.cuda.capture(d.s, launch), d.s)
    elif call:
        launch(d.s)
    return vclib.parse_openings(d.down(out, size * m), n, m)


def check_openings(dom: CommitmentDomain, commitment, schema: str, recs: list[vclib.Opening], n: int, want_leaves, want_salt, want_bc,
                   value_of, asked: list[int] | None = None) -> dict:
    """Each record against the reference (and its position against the one ``asked``); verify_opening on it, and on it with a
    sibling or the value changed; the paths gathered into a multiproof verify_multiproof accepts. What failed, by check (the
    first position each)."""
    bad: dict = {}
    if asked is not None and [r.position for r in recs] != asked:
        bad["positions"] = True
    paths, leaves = {}, {}
    for r in recs:
        p = r.position
        if p >= n:
            if r.offset is not None or any(r.salt + r.bc + r.leaf + b"".join(r.path)):
                bad.setdefault("outside", p)
            continue
        if r.offset != 64 * p:
            bad.setdefault("offset", p)
        if (r.salt != want_salt(p)) if want_salt else any(r.salt):
            bad.setdefault("salt", p)
        if (r.bc != want_bc(p)) if want_bc else any(r.bc):
            bad.setdefault("bc", p)
        if r.leaf != want_leaves[p]:
            bad.setdefault("leaf", p)
        v = value_of(r)
        if not verify_opening(dom, commitment, MerkleOpening(p, v, r.path), schema, LIMITS):
            bad.setdefault("verify_opening", p)
        if verify_opening(dom, commitment, MerkleOpening(p, bytes([v[0] ^ 1]) + v[1:], r.path), schema, LIMITS):
            bad.setdefault("changed value accepted", p)
        if r.path and verify_opening(dom, commitment, MerkleOpening(p, v, (bytes([r.path[0][0] ^ 1]) + r.path[0][1:],) + r.path[1:]),
                                     schema, LIMITS):
            bad.setdefault("changed sibling accepted", p)
        paths[p], leaves[p] = r.path, r.leaf
    if paths and not verify_multiproof(commitment.root, n, dom.domain_id, leaves, multiproof_from_paths(n, paths), hash="sha512"):
        bad["verify_multiproof"] = True
    return bad


def opened_positions(rng: random.Random, n: int) -> list[int]:
    return sorted({0, n - 1, *(rng.randrange(n) for _ in range(6))}) + [n]


def case_open(d: Dev, rng: random.Random, quick: bool) -> None:
    """vc_open_launch over trees the library committed: verity_vllm's serving rows (u16 words, sha512/row/v1, hm96-sha512/row/v1;
    RFC 8439 salts given as serving_rows expands them, or keyed, and b || c retained by the commit), A's tree (b || c from x and
    y in the open kernel) and a frame-only tree. One position past the tree is asked for each time and must come back flagged."""
    n_words, schema = 2048, SCHEMA_HM96_SHA512_ROW
    for n in ((1, 2, 300) if quick else (1, 2, 3, 300, 1025)):
        dom = CommitmentDomain(rng.randbytes(64), -1, RangeIndexedDomain(0, n), hash="sha512")
        words = [[rng.getrandbits(16) for _ in range(n_words)] for _ in range(n)]
        rows = [row_bytes(w, 16) for w in words]
        xs = [sha512_row_digest(w, 16, ROLE_X) for w in words]
        key, port = rng.randbytes(32), rng.randrange(1 << 20)
        raw, ref = rfc_salts(key, port, n)
        ys = chunks(raw, SALT)
        bcs = [hm96.SHA512.commit_string(KEY, x, y) for x, y in zip(xs, ys)]
        tree = MerkleTree(dom, {i: hm96.SHA512.tree_leaf(KEY, b) for i, b in enumerate(bcs)}, lambda _: schema)
        positions = opened_positions(rng, n)
        for src in (("given",), ("chacha-rfc", key, port)):
            plan = d.lib.plan(vclib.served_rows(dom.domain_id, n_words, salt=src))
            tab = d.tables(plan)
            sp = d.up(raw) if src[0] == "given" else None
            lv, bc = d.buf(d.lib.tree_bytes(n)), d.buf(128 * n)
            d.lib.commit_rows_launch(plan.ptr, tab, d.up(b"".join(rows)), 2 * n_words, sp, n, lv, bc, d.buf(d.lib.commit_scratch_bytes(n)),
                                     vclib.FORM_AUTO, d.s)
            ok_tree, where = same_levels(levels_of(d.down(lv, d.lib.tree_bytes(n)), d.lib.tree_width(n)), tree._levels)
            def judge(recs):
                return check_openings(dom, tree.commitment, schema, recs, n, tree._levels[0], lambda p: ys[p], lambda p: bcs[p],
                                      lambda r: hm96.SHA512.tree_leaf(KEY, r.bc), asked=positions)
            recs = open_records(d, plan, tab, None, bc, sp, lv, n, positions, graph=src[0] != "given")
            bad, control = judge(recs), judge(open_records(d, plan, tab, None, bc, sp, lv, n, positions, call=False))
            inside = [r for r in recs if r.offset is not None]
            same_mp = multiproof(tree, [r.position for r in inside]) == multiproof_from_paths(n, {r.position: r.path for r in inside})
            report("open/served-rows", ok_tree and not bad and same_mp and bool(control), salt=src[0], salt_ref=ref, n=n,
                   opened=len(positions), graph=src[0] != "given", where=where, bad=bad or None, multiproof_is_reference=same_mp,
                   control_rejected=bool(control))
    for k, n in ((256, 5), (1024, 40 if quick else 300)):
        binding, unit, key = rng.randbytes(64), rng.randrange(1 << 20), rng.randbytes(32)
        rows = f32_rows(rng, n, k)
        tree = audit.commit_rows(binding, unit, rows, hash=audit.HM96, salt_key=key, nonce=unit)
        plan = d.lib.plan(vclib.hm96_digests(tree.domain.domain_id, salt=("chacha64", key, unit), row_bits=32 * k))
        tab, width = d.tables(plan), d.lib.tree_width(n)
        x, lv, sc = d.buf(64 * n), d.buf(d.lib.tree_bytes(n)), d.buf(d.lib.salt_scratch_bytes(width))
        d.lib.rows_launch(plan.ptr, tab, d.up(b"".join(rows)), 4 * k, n, x, d.buf(d.lib.rows_scratch_bytes(plan.ptr, n)), d.s)
        d.lib.salts_launch(plan.ptr, tab, None, 0, width, n, sc, lv, d.s)
        d.lib.leaves_x_launch(plan.ptr, x, 0, n, n, sc, lv, None, None, d.s)
        d.lib.tree_launch(plan.ptr, lv, n, d.s)
        positions = opened_positions(rng, n)
        ys = {p: audit.hm96_salt(key, unit, p) for p in positions if p < n}

        def judge(recs):
            return check_openings(tree.domain, tree.commitment, audit.ROW_SCHEMA, recs, n, tree._levels[0], lambda p: ys[p],
                                  lambda p: hm96.SHA512.commit_string(KEY, sha512_row_seg_digest(rows[p], 32 * k), ys[p]),
                                  lambda r: rows[r.position] + r.salt, asked=positions)
        recs = open_records(d, plan, tab, x, None, None, lv, n, positions)
        bad, control = judge(recs), judge(open_records(d, plan, tab, x, None, None, lv, n, positions, call=False))
        same = all(tree.open(r.position) == MerkleOpening(r.position, rows[r.position] + r.salt, r.path) for r in recs if r.offset is not None)
        report("open/A-tree", not bad and same and bool(control), k=k, n=n, bad=bad or None, same_as_tree_open=same,
               control_rejected=bool(control))
    for n in (1, 70):
        schema = "hm96-sha512/row/v2"
        dom = domain(rng, n)
        vals = [rng.randbytes(64) for _ in range(n)]
        tree = MerkleTree(dom, dict(enumerate(vals)), lambda _: schema)
        plan = d.lib.plan(vclib.frame_spec(dom.domain_id, schema))
        width = d.lib.tree_width(n)
        lv = d.buf(d.lib.tree_bytes(n))
        d.lib.frame_leaves_launch(plan.ptr, d.up(b"".join(vals)), 0, width, n, lv, d.s)
        d.lib.tree_launch(plan.ptr, lv, n, d.s)
        positions = opened_positions(rng, n)

        def judge(recs):
            return check_openings(dom, tree.commitment, schema, recs, n, tree._levels[0], None, None, lambda r: vals[r.position],
                                  asked=positions)
        bad = judge(open_records(d, plan, None, None, None, None, lv, n, positions))
        control = judge(open_records(d, plan, None, None, None, None, lv, n, positions, call=False))
        report("open/frame", not bad and bool(control), n=n, bad=bad or None, control_rejected=bool(control))


def case_empty(d: Dev, rng: random.Random, quick: bool) -> None:
    """n = 0 through the one-call commit and open, as merkle defines the empty tree: ``tree_levels`` has no levels and the
    commitment's root is the domain's empty root, which the commit writes into the levels' 64 bytes (every form and salt
    source, twice from poison, and captured into a graph), touching neither b || c nor scratch (their guards); and
    ``MerkleTree.open`` refuses every position, as verify_opening rejects it, so every record comes back outside, with or
    without salts, x and b || c."""
    n_bits, schema, n_words = 32768, "hm96-sha512/row/v2", 2048
    positions = [0, 1, 7, (1 << 32) - 1]
    sources = [("prf", rng.randbytes(64)), ("given",), ("chacha64", rng.randbytes(32), rng.randrange(1 << 64)),
               ("chacha-rfc", rng.randbytes(32), rng.randrange(1 << 40))]
    plans = [(f"hm96_rows/{s[0]}", schema, lambda did, s=s: vclib.hm96_rows(did, n_bits, salt=s, schema=schema), n_bits // 8, s)
             for s in sources]
    plans += [(f"served_rows/{s[0]}", SCHEMA_HM96_SHA512_ROW, lambda did, s=s: vclib.served_rows(did, n_words, salt=s), 2 * n_words, s)
              for s in sources if s[0] in ("given", "chacha-rfc")]
    plans += [("A-tree", audit.ROW_SCHEMA, lambda did: vclib.hm96_digests(did, salt=sources[2], row_bits=32 * 256), None, sources[2]),
              ("frame", schema, lambda did: vclib.frame_spec(did, schema), None, None)]
    for name, sch, spec_of, stride, src in plans:
        dom = domain(rng, 0)
        tree = MerkleTree(dom, {}, lambda _, sch=sch: sch)
        refused = []
        for p in positions:
            try:
                tree.open(p)
                refused.append(False)
            except KeyError:
                refused.append(True)
        rejected = not any(verify_opening(dom, tree.commitment, MerkleOpening(p, bytes(64), ()), sch, LIMITS) for p in positions)
        root = tree.commitment.root
        reference = tree_levels(dom, []) == () and root == dom.empty_root() and all(refused) and rejected
        plan = d.lib.plan(spec_of(dom.domain_id))
        tab = d.tables(plan) if plan.tables_bytes else None
        size = d.lib.tree_bytes(0)
        lv = d.buf(size)
        if stride is not None:
            rp, sc_bytes = d.buf(0), d.lib.commit_scratch_bytes(0)
            for form, fname in FORMS:
                lv, bc, sc = d.buf(size), d.buf(0), d.buf(sc_bytes)
                for call in range(2):
                    d.poison(lv, bc, sc)
                    d.lib.commit_rows_launch(plan.ptr, tab, rp, stride, None, 0, lv, bc, sc, form, d.s)
                    report("empty/commit", reference and size == 64 and d.down(lv, size) == root, plan=name, form=fname, call=call)
            sc = d.buf(sc_bytes)
            graph = d.cuda.capture(d.s, lambda s: d.lib.commit_rows_launch(plan.ptr, tab, rp, stride, None, 0, lv, None, sc,
                                                                            vclib.FORM_AUTO, s))
            d.poison(lv)
            no_call = d.down(lv, size) == root
            d.cuda.replay(graph, d.s)
            report("empty/commit-graph", reference and d.down(lv, size) == root and not no_call, plan=name,
                   control_rejected=not no_call)
        else:
            d.lib.tree_launch(plan.ptr, lv, 0, d.s)

        def judge(recs):
            return check_openings(dom, tree.commitment, sch, recs, 0, (), None, None, lambda r: r.bc, asked=positions)
        given = src is not None and src[0] == "given"
        absent = judge(open_records(d, plan, tab, None, None, None, lv, 0, positions, graph=not given))
        present = judge(open_records(d, plan, tab, d.buf(0), d.buf(0), d.buf(0) if given else None, lv, 0, positions))
        control = judge(open_records(d, plan, tab, None, None, None, lv, 0, positions, call=False))
        report("empty/open", reference and not absent and not present and bool(control), plan=name, opened=len(positions),
               record_bytes=d.lib.open_bytes(0), bad=absent or present or None, control_rejected=bool(control))


def case_splitk(d: Dev, rng: random.Random, quick: bool) -> None:
    """An H1 table's chunk leaves (vc_chunk_rows_launch from a row-major [R, K] code buffer and its [R] scales), committed
    in every form under the value's registered domain and audit.hm96_salt's ChaCha20 salts on the device: SPLITK's two
    vectors (`SV.OUT`) leaf for leaf, and at K = 4096 the whole rows' commit byte for byte. The same commit over whole rows
    at a chunked K (the layout before SPLITK) is a control that must miss the vector's root."""
    vec = splitk_vectors()
    for case in SV.CASES + (WHOLE,):
        name, _value, k, R, nonce, _windowed = case
        v = vec.get(name)
        n = Wt.chunks(k)
        rows = SV.rows(name, k, R)
        _words, reg = SV.commit(case)
        N, lb = n * R, k // n + 4
        pinned = v is None or (reg.domain.domain_id.hex(), reg.root.hex(), reg.leaves) == (v["domain"], v["root"], v["leaves"])
        spec, st = vclib.chunk_rows(reg.domain.domain_id, k, n, ("chacha64", SV.key(), nonce))
        cp, sp = d.up(b"".join(r[:k] for r in rows)), d.up(b"".join(r[k:] for r in rows))
        out = d.buf(st * N)
        d.lib.chunk_rows_launch(cp, k, sp, R, k, n, out, st, d.s)
        got = chunks(d.down(out, st * N), st)
        want = [leaf for r in rows for leaf in SV.Z.chunk_leaves(r, k)]
        packed_ok = [x[:lb] for x in got] == want and all(x[lb:] == bytes(st - lb) for x in got)
        words_ok = v is None or [hashlib.sha512(x[:lb]).hexdigest() for x in got] == [e["words_sha512"] for e in v["leaf"]]
        if v is None:
            packed_ok = packed_ok and want == rows
        plan = d.lib.plan(spec)
        tab = d.tables(plan)
        dig = d.buf(64 * N)
        d.lib.rows_launch(plan.ptr, None, out, st, N, dig, None, d.s)
        got_dig = chunks(d.down(dig, 64 * N), 64)
        want_dig = [G.row_digest(w, G.WORD_BITS, G.SCHEMA_V2) for w in reg.rows]
        dig_ok = got_dig == want_dig and (v is None or [x.hex() for x in got_dig] == [e["row_digest"] for e in v["leaf"]])
        report("splitk/leaves", pinned and packed_ok and words_ok and dig_ok, vector=name, k=k, chunks=n, leaves=N,
               leaf_bytes=lb, reference_pinned=pinned, packed=packed_ok, words=words_ok, row_digests=dig_ok,
               dig_diff=first_diff(got_dig, want_dig))
        width, size = d.lib.tree_width(N), d.lib.tree_bytes(N)
        for form, fname in FORMS:
            lv, scratch, bc = d.buf(size), d.buf(d.lib.commit_scratch_bytes(N)), d.buf(128 * N)
            d.poison(lv, bc, scratch)
            d.lib.commit_rows_launch(plan.ptr, tab, out, st, None, N, lv, bc, scratch, form, d.s)
            levels = levels_of(d.down(lv, size), width)
            ok, where = same_levels(levels, reg.levels)
            root = levels[-1][0].hex()
            frames_ok = v is None or [x.hex() for x in levels[0][:N]] == [e["frame_leaf"] for e in v["leaf"]]
            got_bc = chunks(d.down(bc, 128 * N), 128)
            report("splitk/commit", ok and frames_ok and got_bc == list(reg.bcs) and root == reg.root.hex()
                   and (v is None or root == v["root"]), vector=name, form=fname, root=root,
                   want=v["root"] if v else reg.root.hex(), where=where, frame_leaves=frames_ok,
                   bc_diff=first_diff(got_bc, list(reg.bcs)))
        if v is None:
            # the whole rows packed by hand (no vc_chunk_rows_launch) commit to the same root
            hp = d.up(packed(rows, st))
            lv, scratch = d.buf(size), d.buf(d.lib.commit_scratch_bytes(N))
            d.lib.commit_rows_launch(plan.ptr, tab, hp, st, None, N, lv, None, scratch, vclib.FORM_AUTO, d.s)
            same = d.down(lv, size)[-64:] == reg.root
            report("splitk/k-4096-whole-rows", same and packed_ok, root=reg.root.hex())
            continue
        # control: the rows whole at the chunked K, under that layout's own domain, must miss the vector's root
        wdom = G.domain(reg.program, reg.value, Wt.b_words(k), R, G.WORD_BITS, reg.window)
        wplan = d.lib.plan(vclib.hm96_rows(wdom.domain_id, 8 * (k + 4), salt=("chacha64", SV.key(), nonce)))
        wst = stride_of(k + 4)
        whole = d.buf(wst * R)
        d.lib.chunk_rows_launch(cp, k, sp, R, k, 1, whole, wst, d.s)
        wsize = d.lib.tree_bytes(R)
        lv, scratch = d.buf(wsize), d.buf(d.lib.commit_scratch_bytes(R))
        d.lib.commit_rows_launch(wplan.ptr, d.tables(wplan), whole, wst, None, R, lv, None, scratch, vclib.FORM_AUTO, d.s)
        wroot = d.down(lv, wsize)[-64:].hex()
        refused = []
        for bad in ((k, k // 8), (k + 2, n), (k, 0)):   # 8-code chunks, a K past the codes' stride, no chunks
            try:
                d.lib.chunk_rows_launch(cp, k, sp, R, bad[0], bad[1], out, st, d.s)
            except vclib.LaunchError:
                refused.append(bad)
        report("splitk/controls-rejected", wroot != v["root"] and len(refused) == 3, vector=name, whole_root=wroot,
               refused=[list(b) for b in refused])


CASES = [case_sha512, case_compress, case_rows, case_frame, case_arity, case_hm96_x, case_hm96_rows, case_fold_wide, case_pearl, case_graph,
         case_open, case_empty, case_splitk]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    d = Dev(a.lib)
    for case in CASES:
        if a.only and case.__name__ not in a.only:
            continue
        rng = random.Random(case.__name__)
        try:
            case(d, rng, a.quick)
            report("guards", d.guards_intact(), after=case.__name__, buffers=len(d.held))
        except Exception as e:  # noqa: BLE001 - a case that raises is a failed case, and the rest still run
            report(case.__name__, False, error=f"{type(e).__name__}: {e}")
        d.release()
    bad = [r for r in RESULTS if not r["ok"]]
    print(json.dumps({"summary": True, "cases": len(RESULTS), "failed": len(bad), "ok": not bad}), flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
