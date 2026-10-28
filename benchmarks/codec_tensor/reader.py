"""The device reader (`reader.cu`): the remote bandwidth driver's receipts, leaves and receipt tree on the GPU, gated
byte for byte against the driver (`verity/core/protocols/remote_bandwidth/receipts.py`) and timed against a read of
the same bytes.

~~~text
segment     a device reader's part of a pass: count = 16 · 2^h chunks of one region, read in order from position
            p = a multiple of count (`verity_vllm.protocols.remote_bandwidth.registry.aligned`); its root is the node (h + 4 levels above
            the chunks) end_pass takes by segment number, and `open` serves an opening's leaf receipts and leaf path
salts       the store's PRF, `remote_bandwidth.store.store_salt(key, seed, leaf)` (`NodeMemory.salt`), so none is kept
~~~

sketch-sha512 receipts (`receipts.SketchReceipter`) run at k_s = 16 rows (dp4a) and at k_s = 64, a chunk's 2^−64 (int8 tensor
cores, `pack64`'s signs): σ then hash as two kernels, or both in one (`rb_sketch64_receipts`), σ never leaving the SM.

    python reader.py --lib reader.so --out OUT [--gate-blocks 4096] [--gib 8]
"""
from __future__ import annotations

import argparse
import ctypes
import functools
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from benchmarks.word_codec import cuda_lib  # noqa: E402  (after the root's path entry, which it needs)
from verity.core.protocols.remote_bandwidth import receipts as rb, store
from verity.ml.commitments import merkle_depth
from verity.ml.commitments.indexed import RangeIndexedDomain
from verity.ml.commitments.merkle import domain_message
from verity.ml.commitments.rowleaf import sha512_row_v2_prefix

import sketch_receipt as skr

BLOCK = 4096
# reader.cu's LOAD_* modes: the first four compute receipts, the last two are ceilings that read other bytes or none.
LOADS = ("ldg", "l2_256", "prefetch", "warp", "hot", "none")
EXACT = LOADS[:4]
SEGMENTS = (2, 4, 16)
ROWS64, MTS = 64, (1, 2, 4)   # rb_sketch64's rows and its m-tiles a warp
_v, _u64, _i, _b = ctypes.c_void_p, ctypes.c_uint64, ctypes.c_int, ctypes.c_char_p
ABI = {   # reader.cu's functions (`cuda_lib.load`), which session.cu includes
    "rb_set": [_b, _b, _b, _b, _b, _b, _b, _b, _i, _b, _i, _u64, _v],
    "rb_receipts": [_v, _v, _u64, _u64, _v, _i, _v],
    "rb_leaves": [_v, _u64, _u64, _v, _v, _v],
    "rb_pads": [_u64, _u64, _v, _v],
    "rb_level": [_v, _v, _i, _u64, _u64, _v],
    "rb_tree": [_v, _u64, _u64, _v, _v],
    "rb_read": [_v, _u64, _v, _i, _v],
    "rb_sketch_set": [_b, _v],
    "rb_sketch": [_v, _v, _u64, _v, _v, _i, _v],
    "rb_sketch_hash": [_v, _u64, _u64, _v, _v],
    "rb_sketch64": [_v, _v, _u64, _v, _v, _i, _i, _v],
    "rb_sketch64_hash": [_v, _u64, _u64, _v, _v],
    "rb_sketch64_receipts": [_v, _v, _u64, _u64, _v, _v, _v, _i, _v],
    "rb_chain_set": [_b, _i, _b, _b, _b, _b, _b, _b, _i, _i, _b, _i, _i, _b, _i, _b, _i, _i, _u64, _v],
    "rb_chain": [_v, _v],
    "rb_consts": [_v, _u64],
    "rb_consts_bytes": ([], _u64),
}


def pack64(bits: np.ndarray) -> np.ndarray:
    """S's 64 × 512 sign bytes (`sketch_receipt.sign_bits`, bit u of row c at byte u // 8, LSB first) as reader.cu's
    sketch64 B operands, (64, 32, 4) uint32: word w of lane l = 4 g + t at p holds in bit 8 i + r the sign of row
    8 n + g at byte 64 p + 16 t + 8 s + 4 b + i, for n = 2 w + r // 4, s = r // 2 mod 2, b = r mod 2."""
    x = np.unpackbits(np.asarray(bits, np.uint8).reshape(ROWS64, BLOCK // 8), axis=1, bitorder="little")
    x = x.reshape(4, 2, 8, 64, 4, 2, 2, 4).transpose(3, 2, 4, 0, 7, 1, 5, 6)   # p, g, t, w, i, n mod 2, s, b
    return np.packbits(x.reshape(64, 32, 4, 32), axis=-1, bitorder="little").view("<u4").reshape(64, 32, 4).copy()


def chain_messages(plan: rb.Plan, key: bytes) -> dict[str, tuple[bytes, int]]:
    """A pass's seed-dependent messages with the seed's 64 bytes zero, each with where those bytes go (`rb_chain_set`):
    the receipt prefix (`rb.prefix`) and the salts' input (`store.store_salt`) take the seed, the tree's binding
    (`rb.binding`) the seed, and its domain message (`rb.domain`'s) the binding."""
    z, f = bytes(64), b"\xff" * 64
    positions = RangeIndexedDomain(0, rb.leaves(plan.length))

    def dom(binding: bytes) -> bytes:
        return domain_message(binding, rb.OWNER, positions.identity_sha512(), positions.count, "sha512")

    out = {}
    for name, fn in (("receipt", rb.prefix), ("binding", lambda s: rb.TREE_DOMAIN + s + plan.digest),
                     ("domain", dom), ("salt", lambda s: store.SALT_DOMAIN + key + s)):
        m0, m1 = fn(z), fn(f)
        a = next(i for i, (x, y) in enumerate(zip(m0, m1)) if x != y)
        if m0[a:a + 64] != z or m1[a:a + 64] != f or m0[:a] + m0[a + 64:] != m1[:a] + m1[a + 64:]:
            raise RuntimeError(f"the {name} message's seed-derived bytes are not one run of 64")
        out[name] = (m0, a)
    return out


def digest(words) -> bytes:
    """A digest's bytes from its 8 state words (a row of an int64 tensor)."""
    return np.asarray(words.cpu().numpy() if hasattr(words, "cpu") else words).astype(np.uint64).astype(">u8").tobytes()


class Reader:
    """One GPU's reader of the driver's receipt tree: `begin` a pass, receipt and commit its `segment`s, `open` them."""

    def __init__(self, path: str, torch, load: str = "warp") -> None:
        self.torch, self.so, self.load = torch, cuda_lib.load(path, ABI), load
        self.kept: dict[int, tuple[int, object, list]] = {}

    def call(self, fn: str, *args) -> None:
        rc = getattr(self.so, fn)(*args, self.torch.cuda.current_stream().cuda_stream)
        if rc:
            raise RuntimeError(f"{fn}: {rc}")

    def begin(self, seed: bytes, plan: rb.Plan, key: bytes) -> None:
        """Pass `seed`'s constants for its plan, under the store's salt key `key`."""
        n = plan.length
        self.seed, self.plan, self.key, self.positions, self.leaves = seed, plan, key, n, rb.leaves(n)
        last = n - rb.PER_LEAF * (self.leaves - 1)
        self.dom = rb.domain(seed, plan)
        salt_in = store.SALT_DOMAIN + key + seed
        self.call("rb_set", rb.prefix(seed), sha512_row_v2_prefix(8 * rb.DIGEST_BYTES * rb.PER_LEAF),
                  sha512_row_v2_prefix(8 * rb.DIGEST_BYTES * last), rb.HM96.salt_prefix, rb.HM96.leaf_prefix(rb.KEY),
                  rb.KEY, self.dom.domain_id, rb.SCHEMA.encode(), len(rb.SCHEMA), salt_in, len(salt_in), n)
        self.kept = {}

    def chain_begin(self, plan: rb.Plan, key: bytes) -> None:
        """`begin`'s constants for a pass whose seed is on the device (`chain`): the driver's messages with the seed's
        bytes zero, each with where they go."""
        n = plan.length
        self.seed, self.plan, self.key, self.positions, self.leaves = None, plan, key, n, rb.leaves(n)
        last = n - rb.PER_LEAF * (self.leaves - 1)
        m = chain_messages(plan, key)
        self.call("rb_chain_set", *m["receipt"], sha512_row_v2_prefix(8 * rb.DIGEST_BYTES * rb.PER_LEAF),
                  sha512_row_v2_prefix(8 * rb.DIGEST_BYTES * last), rb.HM96.salt_prefix, rb.HM96.leaf_prefix(rb.KEY),
                  rb.KEY, m["binding"][0], len(m["binding"][0]), m["binding"][1], m["domain"][0], len(m["domain"][0]),
                  m["domain"][1], rb.SCHEMA.encode(), len(rb.SCHEMA), m["salt"][0], len(m["salt"][0]), m["salt"][1], n)
        self.kept = {}

    def chain(self, seed_ptr: int) -> None:
        """`chain_begin`'s pass's constants from the 64-byte seed at seed_ptr on the device (wt.cu's g_pass.seed), on
        the stream: those `begin` sets at that seed."""
        self.call("rb_chain", seed_ptr)

    def consts(self) -> bytes:
        """The pass's constants as the device holds them (after the stream's work)."""
        out = ctypes.create_string_buffer(self.so.rb_consts_bytes())
        rc = self.so.rb_consts(ctypes.addressof(out), len(out.raw))
        if rc:
            raise RuntimeError(f"rb_consts: {rc}")
        return out.raw

    def receipts(self, data, j0: int, n: int, ys, zs=None, load: str | None = None) -> None:
        """Receipts of n chunks at `data` (store indices j0 …) into ys (n x 8), each bound to its spread z in zs."""
        self.call("rb_receipts", data.data_ptr(), zs.data_ptr() if zs is not None else None, j0, n, ys.data_ptr(),
                  LOADS.index(load or self.load))

    def sketch_begin(self, seed: bytes) -> None:
        """Pass `seed`'s sketch-sha512 constants: its signs at 16 and at 64 rows on the device and its seed block's
        midstate."""
        self.signs = self.torch.from_numpy(skr.signs(seed)).cuda()
        self.signs64 = self.torch.from_numpy(pack64(skr.sign_bits(seed, ROWS64)).view(np.int32)).cuda()
        self.call("rb_sketch_set", rb.sketch_prefix(seed))

    def sketches64(self, data, n: int, sk, zs=None, mt: int = 2, ctas: int = 0) -> None:
        """σ at 64 rows of n chunks at `data` into sk (n x 64 int32), mt m-tiles a warp."""
        self.call("rb_sketch64", data.data_ptr(), zs.data_ptr() if zs is not None else None, n,
                  self.signs64.data_ptr(), sk.data_ptr(), mt, ctas)

    def sketch64_receipts(self, data, j0: int, n: int, ys, zs=None, sk=None, mt: int = 0) -> None:
        """64-row sketch-sha512 receipts of n chunks at `data` (store indices j0 …) into ys (n x 8): in one kernel at
        mt = 0 (σ into sk too, if given), else rb_sketch64 at mt m-tiles a warp and then the hash."""
        zp = zs.data_ptr() if zs is not None else None
        if not mt:
            self.call("rb_sketch64_receipts", data.data_ptr(), zp, j0, n, self.signs64.data_ptr(),
                      sk.data_ptr() if sk is not None else None, ys.data_ptr(), 0)
            return
        sk = self.torch.empty((n, ROWS64), dtype=self.torch.int32, device="cuda") if sk is None else sk
        self.sketches64(data, n, sk, zs, mt)
        self.call("rb_sketch64_hash", sk.data_ptr(), j0, n, ys.data_ptr())

    def sketches(self, data, n: int, sk, zs=None, ctas: int = 0) -> None:
        """The sketches of n chunks at `data` into sk (n x 16 int32), each chunk XORed with its spread z in zs."""
        self.call("rb_sketch", data.data_ptr(), zs.data_ptr() if zs is not None else None, n, self.signs.data_ptr(),
                  sk.data_ptr(), ctas)

    def sketch_receipts(self, data, j0: int, n: int, ys, zs=None, sk=None) -> None:
        """sketch-sha512 receipts of n chunks at `data` (store indices j0 …) into ys (n x 8)."""
        sk = self.torch.empty((n, skr.ROWS), dtype=self.torch.int32, device="cuda") if sk is None else sk
        self.sketches(data, n, sk, zs)
        self.call("rb_sketch_hash", sk.data_ptr(), j0, n, ys.data_ptr())

    def commit(self, ys, c0: int, nl: int, salts=None):
        dig = self.torch.empty((nl, 8), dtype=self.torch.int64, device="cuda")
        self.call("rb_leaves", ys.data_ptr(), c0, nl, dig.data_ptr(), salts.data_ptr() if salts is not None else None)
        return dig

    def levels(self, dig, c0: int, pad: bool = False) -> list:
        """The tree over leaves [c0, c0 + len(dig)), level by level up to its root; with `pad`, the pass's whole
        tree (c0 = 0, padded to a power of two)."""
        torch, level = self.torch, dig
        if pad and (w := 1 << merkle_depth(len(dig))) > len(dig):
            pads = torch.empty((w - len(dig), 8), dtype=torch.int64, device="cuda")
            self.call("rb_pads", len(dig), w - len(dig), pads.data_ptr())
            level = torch.cat([dig, pads])
        n = len(level)
        if n == 1:
            return [level]
        if n & (n - 1):
            raise ValueError(f"{n} digests are no subtree")
        up = torch.empty((n - 1, 8), dtype=torch.int64, device="cuda")
        self.call("rb_tree", level.data_ptr(), n, c0, up.data_ptr())
        out, at = [level], 0
        while n > 1:
            n //= 2
            out.append(up[at:at + n])
            at += n
        return out

    def segment(self, data, j0: int, position: int, count: int, zs=None) -> bytes:
        """An aligned segment's receipts, leaves and subtree, kept for its openings; its root, for end_pass."""
        if count % rb.PER_LEAF or (count // rb.PER_LEAF) & (count // rb.PER_LEAF - 1) or position % count:
            raise ValueError(f"{count} blocks at position {position} is no subtree of the pass's tree")
        ys = self.torch.empty((count, 8), dtype=self.torch.int64, device="cuda")
        self.receipts(data, j0, count, ys, zs)
        c0 = position // rb.PER_LEAF
        levels = self.levels(self.commit(ys, c0, count // rb.PER_LEAF), c0)
        self.kept[position] = (count, ys, levels)
        return digest(levels[-1][0])

    def open(self, position: int) -> tuple[tuple[bytes, ...], tuple[bytes, ...], bytes]:
        """Position `position`'s leaf receipts, its leaf's path to its segment's root, and its salt."""
        p0 = next(p for p, (n, _, _) in self.kept.items() if p <= position < p + n)
        _, ys, levels = self.kept[p0]
        c, c0 = position // rb.PER_LEAF, p0 // rb.PER_LEAF
        lo = (c - c0) * rb.PER_LEAF
        leaf = tuple(digest(y) for y in ys[lo:lo + rb.PER_LEAF])
        path = tuple(digest(levels[h][((c >> h) ^ 1) - (c0 >> h)]) for h in range(len(levels) - 1))
        return leaf, path, store.store_salt(self.key, self.seed, c)


def gate(reader: Reader, torch, blocks: int, rng: np.random.Generator) -> dict:
    """Byte for byte against the driver: a pass of a partial plan out of store order (runs at odd offsets, a short last
    leaf, pads), every receipt bound to a z of its own length, its salts, leaves and root; then a pass of two aligned
    segments, their roots against the driver's tree, and openings through them verified by `rb.verify_leaf`."""
    host = rng.integers(0, 256, blocks * BLOCK, dtype=np.uint8)
    data = torch.from_numpy(host).cuda()
    key = rng.bytes(32)
    out = {}

    seed = rb.seed(rng.bytes(32), 3, rng.bytes(64))
    runs = ((blocks // 2 + 7, blocks // 8 + 3), (11, blocks // 4 - 29), (blocks - blocks // 8, blocks // 16 + 5))
    plan = rb.Plan(runs, blocks)
    order = list(plan)
    zs = [rng.bytes(int(rng.integers(1, 2 * BLOCK))) for _ in order]
    spread = torch.from_numpy(np.frombuffer(b"".join(rb.spread(z) for z in zs), np.uint8).copy()).cuda()
    reader.begin(seed, plan, key)
    ys = torch.empty((plan.length, 8), dtype=torch.int64, device="cuda")
    p = 0
    for first, count in plan.runs:
        reader.receipts(data[first * BLOCK:], first, count, ys[p:], spread[p * BLOCK:])
        p += count
    salts = torch.empty(reader.leaves * rb.SALT_BYTES, dtype=torch.uint8, device="cuda")
    levels = reader.levels(reader.commit(ys, 0, reader.leaves, salts), 0, pad=True)
    receipter = rb.Receipter(seed)
    ref_ys = [receipter(j, host[j * BLOCK:(j + 1) * BLOCK].tobytes(), z) for j, z in zip(order, zs)]
    ref_salts = [store.store_salt(key, seed, c) for c in range(reader.leaves)]
    tree = rb.tree(seed, plan, ref_ys, ref_salts)
    got = [digest(y) for y in ys]
    sl = salts.cpu().numpy().tobytes()
    # Every exact load mode on the same runs (none a whole number of warps), bound to the z's and to z's of zero bytes.
    ref_plain = [receipter(j, host[j * BLOCK:(j + 1) * BLOCK].tobytes(), bytes(BLOCK)) for j in order]
    by_load = {}
    for load in EXACT:
        with_z, plain = torch.zeros_like(ys), torch.zeros_like(ys)
        p = 0
        for first, count in plan.runs:
            reader.receipts(data[first * BLOCK:], first, count, with_z[p:], spread[p * BLOCK:], load=load)
            reader.receipts(data[first * BLOCK:], first, count, plain[p:], load=load)
            p += count
        by_load[load] = [digest(y) for y in with_z] == ref_ys and [digest(y) for y in plain] == ref_plain
    out["partial"] = {
        "positions": plan.length, "leaves": reader.leaves, "width": len(levels[0]),
        "receipts": got == ref_ys, "receipts_by_load": by_load,
        "salts": [sl[c * rb.SALT_BYTES:(c + 1) * rb.SALT_BYTES] for c in range(reader.leaves)] == ref_salts,
        "leaves_and_pads": [digest(x) for x in levels[0]] == list(tree._levels[0]),
        "levels": all([digest(x) for x in lv] == list(tree._levels[h]) for h, lv in enumerate(levels)),
        "root": digest(levels[-1][0]) == tree.commitment.root,
    }

    seed = rb.seed(rng.bytes(32), 0, rng.bytes(64))
    half = blocks // 2
    plan = rb.Plan(((0, half), (blocks - half // 2, half // 2)), blocks)
    reader.begin(seed, plan, key)
    roots = [reader.segment(data, 0, 0, half), reader.segment(data[(blocks - half // 2) * BLOCK:], blocks - half // 2,
                                                              half, half // 2)]
    receipter = rb.Receipter(seed)
    ref_ys = [receipter(j, host[j * BLOCK:(j + 1) * BLOCK].tobytes(), bytes(BLOCK)) for j in plan]
    tree = rb.tree(seed, plan, ref_ys, [store.store_salt(key, seed, c) for c in range(reader.leaves)])
    h0, h1 = (half // rb.PER_LEAF).bit_length() - 1, (half // 2 // rb.PER_LEAF).bit_length() - 1
    opened = []
    for p in sorted({0, 1, 17, half - 1, half, half + 33, plan.length - 1, *map(int, rng.integers(0, plan.length, 6))}):
        leaf, path, salt = reader.open(p)
        c, level = p // rb.PER_LEAF, h0 if p < half else h1
        full = tree.open(c).path
        opened.append(path == full[:level] and leaf == tuple(ref_ys[c * rb.PER_LEAF:(c + 1) * rb.PER_LEAF])
                      and rb.verify_leaf(seed, plan, tree.commitment.root, p, b"".join(leaf), salt,
                                         path + full[level:]))
    out["segments"] = {
        "positions": plan.length, "segments": [[0, half], [half, half // 2]],
        "roots": roots == [tree._levels[h0][0], tree._levels[h1][half // rb.PER_LEAF >> h1]],
        "openings": len(opened), "openings_ok": all(opened),
    }
    out["ok"] = all(v for part in ("partial", "segments") for k, v in out[part].items() if isinstance(v, bool)) and all(
        by_load.values())
    return out


def chain_gate(reader: Reader, torch, rng: np.random.Generator) -> dict:
    """The chained pass (`chain_begin`, then `chain` from a seed on the device) against `begin` at the same seed: the
    device's constants byte for byte, and its commit of random receipts (salts, root) against `rb.tree` at that seed,
    on a partial plan out of store order with a short last leaf and on a whole one."""
    out, key = {}, rng.bytes(32)
    for name, plan in (("partial", rb.Plan(((901, 37), (5, 300), (2000, 18)), 4096)), ("whole", rb.Plan.whole(4096))):
        seed = rb.seed(rng.bytes(32), int(rng.integers(1 << 32)), rng.bytes(64))
        dev = torch.from_numpy(np.frombuffer(seed, np.uint8).copy()).cuda()
        ys = torch.empty((plan.length, 8), dtype=torch.int64, device="cuda").random_()
        reader.chain_begin(plan, key)
        reader.chain(dev.data_ptr())
        salts = torch.empty(reader.leaves * rb.SALT_BYTES, dtype=torch.uint8, device="cuda")
        levels = reader.levels(reader.commit(ys, 0, reader.leaves, salts), 0, pad=True)
        torch.cuda.synchronize()
        chained, root, sl = reader.consts(), digest(levels[-1][0]), salts.cpu().numpy().tobytes()
        reader.begin(seed, plan, key)
        torch.cuda.synchronize()
        ref_salts = [store.store_salt(key, seed, c) for c in range(reader.leaves)]
        tree = rb.tree(seed, plan, [digest(y) for y in ys.cpu().numpy()], ref_salts)
        out[name] = {"positions": plan.length, "leaves": reader.leaves, "consts": chained == reader.consts(),
                     "salts": sl == b"".join(ref_salts), "root": root == tree.commitment.root}
    out["ok"] = all(v for part in out.values() for v in part.values() if isinstance(v, bool))
    return out


def sketch_gate(reader: Reader, torch, blocks: int, rng: np.random.Generator, rows: int = skr.ROWS) -> dict:
    """sketch-sha512 at k_s = rows byte for byte against `receipts.SketchReceipter`, on every device path (16
    rows: dp4a; 64: rb_sketch64 at each m-tile count then the hash, and the one kernel): a partial plan out of store
    order, every receipt bound to a z of its own length and to z's of zero bytes, their sketches, the device's leaves
    and root against `rb.tree` over the reference's receipts, and openings through the device's tree by
    `rb.verify_leaf`."""
    host = rng.integers(0, 256, blocks * BLOCK, dtype=np.uint8)
    host[:BLOCK], host[BLOCK:2 * BLOCK] = 0x80, 0x7F
    data = torch.from_numpy(host).cuda()
    key = rng.bytes(32)
    seed = rb.seed(rng.bytes(32), 5, rng.bytes(64))
    runs = ((blocks // 2 + 7, blocks // 8 + 3), (0, blocks // 4 - 29), (blocks - blocks // 8, blocks // 16 + 5))
    plan = rb.Plan(runs, blocks)
    order = list(plan)
    zs = [rng.bytes(int(rng.integers(1, 2 * BLOCK))) for _ in order]
    spread = torch.from_numpy(np.frombuffer(b"".join(rb.spread(z) for z in zs), np.uint8).copy()).cuda()
    reader.begin(seed, plan, key)
    reader.sketch_begin(seed)
    if rows == skr.ROWS:
        paths = {"dp4a": reader.sketch_receipts}
    elif rows == ROWS64:
        paths = {**{f"mt{mt}": functools.partial(reader.sketch64_receipts, mt=mt) for mt in MTS},
                 "one_kernel": reader.sketch64_receipts}
    else:
        raise ValueError(f"the device sketches 16 or 64 rows, not {rows}")
    receipter = rb.SketchReceipter(seed, rows)
    blk = [host[j * BLOCK:(j + 1) * BLOCK].tobytes() for j in order]
    ref_sk = [receipter.sketch(b, z) for b, z in zip(blk, zs)]
    ref_ys = [receipter.receipt(j, s) for j, s in zip(order, ref_sk)]
    ref_plain = [receipter(j, b, bytes(BLOCK)) for j, b in zip(order, blk)]
    by_path = {}
    for name, fn in paths.items():
        ys, plain = (torch.zeros((plan.length, 8), dtype=torch.int64, device="cuda") for _ in range(2))
        sk = torch.zeros((plan.length, rows), dtype=torch.int32, device="cuda")
        p = 0
        for first, count in plan.runs:
            fn(data[first * BLOCK:], first, count, ys[p:], spread[p * BLOCK:], sk=sk[p:])
            fn(data[first * BLOCK:], first, count, plain[p:])
            p += count
        by_path[name] = {"sketches": sk.cpu().numpy().astype("<i4").tobytes() == b"".join(ref_sk),
                         "receipts": [digest(y) for y in ys] == ref_ys,
                         "receipts_plain": [digest(y) for y in plain] == ref_plain}
    salts = torch.empty(reader.leaves * rb.SALT_BYTES, dtype=torch.uint8, device="cuda")
    levels = reader.levels(reader.commit(ys, 0, reader.leaves, salts), 0, pad=True)
    tree = rb.tree(seed, plan, ref_ys, [store.store_salt(key, seed, c) for c in range(reader.leaves)])
    root = digest(levels[-1][0])
    got = [digest(y) for y in ys]
    sl = salts.cpu().numpy().tobytes()
    opened = []
    for p in sorted({0, 1, 17, plan.length // 2, plan.length - 1, *map(int, rng.integers(0, plan.length, 6))}):
        c = p // rb.PER_LEAF
        leaf = b"".join(got[c * rb.PER_LEAF:(c + 1) * rb.PER_LEAF])
        path = tuple(digest(levels[h][(c >> h) ^ 1]) for h in range(len(levels) - 1))
        opened.append(rb.verify_leaf(seed, plan, root, p, leaf, sl[c * rb.SALT_BYTES:(c + 1) * rb.SALT_BYTES], path))
    out = {
        "kind": rb.SKETCH_SHA512, "rows": rows, "positions": plan.length, "leaves": reader.leaves, "by_path": by_path,
        "sketches": all(v["sketches"] for v in by_path.values()),
        "receipts": all(v["receipts"] for v in by_path.values()) and got == ref_ys,
        "receipts_plain": all(v["receipts_plain"] for v in by_path.values()),
        "root": root == tree.commitment.root, "openings": len(opened), "openings_ok": all(opened),
        "extremes_in_plan": 0 in order and 1 in order,
    }
    out["ok"] = all(v for v in out.values() if isinstance(v, bool))
    return out


def measure(reader: Reader, torch, gib: float, reps: int) -> dict:
    """One pass over gib GiB as a device reader does it, against reading the same bytes: receipts (z of zero bytes,
    and over half of it each chunk bound to a spread z the other half holds), leaves, the whole tree."""
    n = int(gib * 2 ** 30) // BLOCK // rb.PER_LEAF * rb.PER_LEAF
    for S in SEGMENTS:
        if (n // S) % rb.PER_LEAF or (u := n // S // rb.PER_LEAF) & (u - 1) or n % S:
            raise ValueError(f"{gib} GiB is no {S} aligned segments")
    data = torch.randint(-2 ** 31, 2 ** 31 - 1, (n * BLOCK // 4,), dtype=torch.int32, device="cuda")
    ys = torch.empty((n, 8), dtype=torch.int64, device="cuda")
    xr = torch.zeros(1, dtype=torch.int64, device="cuda")
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    plan = rb.Plan.whole(n)
    seed = rb.seed(bytes(32), 0, bytes(64))
    reader.begin(seed, plan, bytes(32))
    reader.sketch_begin(seed)
    sk = torch.empty((n, skr.ROWS), dtype=torch.int32, device="cuda")
    half = n // 2
    steps = {"read": lambda: reader.call("rb_read", data.data_ptr(), n * BLOCK, xr.data_ptr(), 16 * sms)}
    steps["sketch"] = lambda: reader.sketches(data, n, sk)
    steps["sketch_z_half"] = lambda: reader.sketches(data, half, sk, data[half * BLOCK // 4:])
    steps["sketch_hash"] = lambda: reader.call("rb_sketch_hash", sk.data_ptr(), 0, n, ys.data_ptr())
    sk64 = torch.empty((n, ROWS64), dtype=torch.int32, device="cuda")
    for mt in MTS:
        steps[f"sketch64/{mt}"] = lambda mt=mt: reader.sketches64(data, n, sk64, mt=mt)
        steps[f"sketch64_z_half/{mt}"] = lambda mt=mt: reader.sketches64(data, half, sk64, data[half * BLOCK // 4:],
                                                                         mt=mt)
    steps["sketch64_hash"] = lambda: reader.call("rb_sketch64_hash", sk64.data_ptr(), 0, n, ys.data_ptr())
    steps["sketch64_one_kernel"] = lambda: reader.sketch64_receipts(data, 0, n, ys)
    steps["sketch64_one_kernel_z_half"] = lambda: reader.sketch64_receipts(data, 0, half, ys, data[half * BLOCK // 4:])
    for load in LOADS:
        steps[f"receipts/{load}"] = lambda load=load: reader.receipts(data, 0, n, ys, load=load)
        steps[f"receipts_z_half/{load}"] = lambda load=load: reader.receipts(data, 0, half, ys,
                                                                             data[half * BLOCK // 4:], load=load)
    steps["leaves"] = lambda: reader.commit(ys, 0, n // rb.PER_LEAF)
    dig = reader.commit(ys, 0, n // rb.PER_LEAF)
    steps["tree"] = lambda: reader.levels(dig, 0, pad=True)
    # The pass as the store reads it, in aligned segments: segment s's leaves and subtree on a side stream while
    # segment s + 1 is receipted, its root what end_pass takes. With `lanes`, consecutive segments' receipts go to
    # alternate streams, so one segment's last wave runs beside the next one's first.
    side, roots = torch.cuda.Stream(), {}
    pair = (torch.cuda.Stream(), torch.cuda.Stream())

    # With `sketch`, the receipts are sketch-sha512's, into ys_sk.
    ys_sk = torch.empty_like(ys)

    def segmented(segments: int, lanes: int, sketch: bool = False) -> None:
        count, main, out = n // segments, torch.cuda.current_stream(), ys_sk if sketch else ys
        for st in (*pair, side):
            st.wait_stream(main)
        for s in range(segments):
            lane = pair[s % lanes]
            with torch.cuda.stream(lane):
                at = data[s * count * BLOCK // 4:]
                if sketch:
                    reader.sketch_receipts(at, s * count, count, out[s * count:], sk=sk[s * count:])
                else:
                    reader.receipts(at, s * count, count, out[s * count:])
                done = torch.cuda.Event()
                done.record(lane)
            with torch.cuda.stream(side):
                side.wait_event(done)
                c0 = s * count // rb.PER_LEAF
                roots[sketch, segments, s] = reader.levels(reader.commit(out[s * count:], c0, count // rb.PER_LEAF),
                                                           c0)[-1]
        for st in (*pair, side):
            main.wait_stream(st)

    for segments in SEGMENTS:
        for lanes in (1, 2):
            steps[f"segmented/{segments}/{lanes}"] = lambda segments=segments, lanes=lanes: segmented(segments, lanes)
            steps[f"sketch_segmented/{segments}/{lanes}"] = lambda segments=segments, lanes=lanes: segmented(
                segments, lanes, True)
    ms = {}
    for name, fn in steps.items():
        fn()
        torch.cuda.synchronize()
        best = []
        for _ in range(reps):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            fn()
            b.record()
            torch.cuda.synchronize()
            best.append(a.elapsed_time(b))
        ms[name] = min(best)
    segment_roots = True
    for sketch, out in ((False, ys), (True, ys_sk)):
        full = reader.levels(reader.commit(out, 0, n // rb.PER_LEAF), 0, pad=True)
        segment_roots &= all([digest(roots[sketch, S, s][0]) for s in range(S)]
                             == [digest(x) for x in full[merkle_depth(n // S // rb.PER_LEAF)]] for S in SEGMENTS)
    nbytes = n * BLOCK
    by_load = {load: {"gb_s": nbytes / ms[f"receipts/{load}"] / 1e6,
                      "z_gb_s": nbytes / (ms[f"receipts_z_half/{load}"] * n / half) / 1e6,
                      "sha512_compressions_per_s": n * 33 / ms[f"receipts/{load}"] * 1e3} for load in LOADS}
    best = min(EXACT, key=lambda load: ms[f"receipts/{load}"])
    best_z = min(EXACT, key=lambda load: ms[f"receipts_z_half/{load}"])
    plain, z = ms[f"receipts/{best}"], ms[f"receipts_z_half/{best_z}"] * n / half
    reader_ms = plain + ms["leaves"] + ms["tree"]
    compressions = n * 33 + n // rb.PER_LEAF * 15
    commit, read = ms["leaves"] + ms["tree"], ms["read"]
    sketch_z = ms["sketch_z_half"] * n / half
    sketch = {
        "kind": rb.SKETCH_SHA512, "sketch_gb_s": nbytes / ms["sketch"] / 1e6, "sketch_z_gb_s": nbytes / sketch_z / 1e6,
        "tau_per_pass": (ms["sketch"] + ms["sketch_hash"] + commit) / read,
        "tau_z_per_pass": (sketch_z + ms["sketch_hash"] + commit) / read,
        "tau_sketch": ms["sketch"] / read, "tau_sketch_z": sketch_z / read, "tau_hash": ms["sketch_hash"] / read,
        "tau_leaves_and_tree": commit / read,
        # what a kernel that sketches the bytes it already holds (the weight GEMM's) adds past its sketch warps
        "tau_after_sketch": (ms["sketch_hash"] + commit) / read,
        "tau_segmented_per_pass": {k.split("/", 1)[1]: v / read for k, v in ms.items()
                                   if k.startswith("sketch_segmented/")},
        "compressions_per_block": (n + n // rb.PER_LEAF * 15) / n,
    }
    mt64 = min(MTS, key=lambda mt: ms[f"sketch64/{mt}"])
    mt64_z = min(MTS, key=lambda mt: ms[f"sketch64_z_half/{mt}"])
    s64, s64_z = ms[f"sketch64/{mt64}"], ms[f"sketch64_z_half/{mt64_z}"] * n / half
    one, one_z = ms["sketch64_one_kernel"], ms["sketch64_one_kernel_z_half"] * n / half
    sketch64 = {
        "kind": rb.SKETCH_SHA512, "rows": ROWS64, "best_mt": mt64, "best_mt_z": mt64_z,
        "sketch_gb_s": {mt: nbytes / ms[f"sketch64/{mt}"] / 1e6 for mt in MTS},
        "sketch_z_gb_s": {mt: nbytes / (ms[f"sketch64_z_half/{mt}"] * n / half) / 1e6 for mt in MTS},
        "tau_sketch": s64 / read, "tau_sketch_z": s64_z / read, "tau_hash": ms["sketch64_hash"] / read,
        "tau_per_pass": (s64 + ms["sketch64_hash"] + commit) / read,
        "tau_z_per_pass": (s64_z + ms["sketch64_hash"] + commit) / read,
        "tau_one_kernel": one / read, "tau_one_kernel_z": one_z / read,
        "tau_one_kernel_per_pass": (one + commit) / read, "tau_one_kernel_z_per_pass": (one_z + commit) / read,
        "tau_after_sketch": (ms["sketch64_hash"] + commit) / read,
        "compressions_per_block": (3 * n + n // rb.PER_LEAF * 15) / n,
    }
    best64 = min(sketch64["tau_per_pass"], sketch64["tau_one_kernel_per_pass"])
    sketch64["over_16_rows"] = best64 / sketch["tau_per_pass"]
    return {
        "sketch_sha512": sketch, "sketch_sha512_64": sketch64,
        "blocks": n, "bytes": nbytes, "ms": ms, "by_load": by_load, "best_load": best, "best_load_z": best_z,
        "read_gb_s": nbytes / ms["read"] / 1e6, "receipts_gb_s": nbytes / plain / 1e6,
        "receipts_z_gb_s": nbytes / z / 1e6,
        "sha512_compressions_per_s": n * 33 / plain * 1e3,
        "reader_ms": reader_ms, "reader_z_ms": z + ms["leaves"] + ms["tree"],
        "tau_per_pass": reader_ms / ms["read"], "tau_z_per_pass": (z + ms["leaves"] + ms["tree"]) / ms["read"],
        "tau_ceiling_none": (ms["receipts/none"] + ms["leaves"] + ms["tree"]) / ms["read"],
        "leaves_and_tree_share": (ms["leaves"] + ms["tree"]) / reader_ms,
        "compressions_per_block": compressions / n,
        "segments": SEGMENTS, "segment_roots_match": segment_roots,
        "tau_segmented_per_pass": {k.split("/", 1)[1]: v / ms["read"] for k, v in ms.items()
                                   if k.startswith("segmented/")},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lib", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--gate-blocks", type=int, default=4096)
    ap.add_argument("--gib", type=float, default=8.0)
    ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args()
    import torch

    a.out.mkdir(parents=True, exist_ok=True)
    reader = Reader(a.lib, torch)
    g = gate(reader, torch, a.gate_blocks, np.random.default_rng(20261012))
    print("reader gate:", json.dumps(g), flush=True)
    sg = sketch_gate(reader, torch, a.gate_blocks, np.random.default_rng(20261013))
    print("sketch-sha512 gate:", json.dumps(sg), flush=True)
    sg64 = sketch_gate(reader, torch, a.gate_blocks, np.random.default_rng(20261014), rows=ROWS64)
    print("sketch-sha512 64-row gate:", json.dumps(sg64), flush=True)
    cg = chain_gate(reader, torch, np.random.default_rng(20261015))
    print("chained pass gate:", json.dumps(cg), flush=True)
    g["ok"] = g["ok"] and sg["ok"] and sg64["ok"] and cg["ok"]
    result = {"schema": "verity/device-reader/v0", "device": torch.cuda.get_device_name(0), "gate": g,
              "sketch_gate": sg, "sketch64_gate": sg64, "chain_gate": cg}
    if g["ok"] and a.gib > 0:
        result["measure"] = m = measure(reader, torch, a.gib, a.reps)
        print("reader measure:", json.dumps(m), flush=True)
    (a.out / "reader.json").write_text(json.dumps(result, indent=1))
    return 0 if g["ok"] and result.get("measure", {}).get("segment_roots_match", True) else 7


if __name__ == "__main__":
    sys.exit(main())
