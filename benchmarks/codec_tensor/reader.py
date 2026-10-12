"""The device reader (`reader.cu`): the remote bandwidth driver's receipts, leaves and receipt tree on the GPU, gated
byte for byte against the driver (`verity/core/protocols/remote_bandwidth/receipts.py`) and timed against a read of
the same bytes.

~~~text
segment     a device reader's part of a pass: count = 16 · 2^h blocks of one region, read in order from position
            p = a multiple of count (`verity_vllm.store.registry.aligned`); its root is the node (h + 4 levels above
            the blocks) end_pass takes by segment number, and `open` serves an opening's leaf receipts and leaf path
salts       the store's PRF, SHAKE256(D(store salt) ‖ key ‖ seed ‖ u64be leaf) (`Store.salt`), so none is kept
~~~

    python reader.py --lib reader.so --out OUT [--gate-blocks 4096] [--gib 8]
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import struct
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from verity.core.protocols.remote_bandwidth import receipts as rb
from verity.ml.commitments import merkle_depth
from verity.ml.commitments.rowleaf import sha512_row_v2_prefix

BLOCK = 4096
SALT_DOMAIN = b"verity/store/salt/v0\0"
# reader.cu's LOAD_* modes: the first four compute receipts, the last two are ceilings that read other bytes or none.
LOADS = ("ldg", "l2_256", "prefetch", "warp", "hot", "none")
EXACT = LOADS[:4]
SEGMENTS = 16


def store_salt(key: bytes, seed: bytes, leaf: int) -> bytes:
    return hashlib.shake_256(SALT_DOMAIN + key + seed + struct.pack(">Q", leaf)).digest(rb.SALT_BYTES)


def digest(words) -> bytes:
    """A digest's bytes from its 8 state words (a row of an int64 tensor)."""
    return np.asarray(words.cpu().numpy() if hasattr(words, "cpu") else words).astype(np.uint64).astype(">u8").tobytes()


class Reader:
    """One GPU's reader of the driver's receipt tree: `begin` a pass, receipt and commit its `segment`s, `open` them."""

    def __init__(self, path: str, torch, load: str = "warp") -> None:
        self.torch, self.so, self.load = torch, ctypes.CDLL(path), load
        v, u64, i = ctypes.c_void_p, ctypes.c_uint64, ctypes.c_int
        b = ctypes.c_char_p
        self.so.rb_set.argtypes = [b, b, b, b, b, b, b, b, i, b, i, u64, v]
        self.so.rb_receipts.argtypes = [v, v, u64, u64, v, i, v]
        self.so.rb_leaves.argtypes = [v, u64, u64, v, v, v]
        self.so.rb_pads.argtypes = [u64, u64, v, v]
        self.so.rb_level.argtypes = [v, v, i, u64, u64, v]
        self.so.rb_tree.argtypes = [v, u64, u64, v, v]
        self.so.rb_read.argtypes = [v, u64, v, i, v]
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
        salt_in = SALT_DOMAIN + key + seed
        self.call("rb_set", rb.prefix(seed), sha512_row_v2_prefix(8 * rb.DIGEST_BYTES * rb.PER_LEAF),
                  sha512_row_v2_prefix(8 * rb.DIGEST_BYTES * last), rb.HM96.salt_prefix, rb.HM96.leaf_prefix(rb.KEY),
                  rb.KEY, self.dom.domain_id, rb.SCHEMA.encode(), len(rb.SCHEMA), salt_in, len(salt_in), n)
        self.kept = {}

    def receipts(self, data, j0: int, n: int, ys, zs=None, load: str | None = None) -> None:
        """Receipts of n blocks at `data` (store indices j0 …) into ys (n x 8), each bound to its spread z in zs."""
        self.call("rb_receipts", data.data_ptr(), zs.data_ptr() if zs is not None else None, j0, n, ys.data_ptr(),
                  LOADS.index(load or self.load))

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
        return leaf, path, store_salt(self.key, self.seed, c)


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
    ref_salts = [store_salt(key, seed, c) for c in range(reader.leaves)]
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
    tree = rb.tree(seed, plan, ref_ys, [store_salt(key, seed, c) for c in range(reader.leaves)])
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


def measure(reader: Reader, torch, gib: float, reps: int) -> dict:
    """One pass over gib GiB as a device reader does it, against reading the same bytes: receipts (z of zero bytes,
    and over half of it each block bound to a spread z the other half holds), leaves, the whole tree."""
    n = int(gib * 2 ** 30) // BLOCK // rb.PER_LEAF * rb.PER_LEAF
    if (n // SEGMENTS) % rb.PER_LEAF or (u := n // SEGMENTS // rb.PER_LEAF) & (u - 1) or n % SEGMENTS:
        raise ValueError(f"{gib} GiB is no {SEGMENTS} aligned segments")
    data = torch.randint(-2 ** 31, 2 ** 31 - 1, (n * BLOCK // 4,), dtype=torch.int32, device="cuda")
    ys = torch.empty((n, 8), dtype=torch.int64, device="cuda")
    xr = torch.zeros(1, dtype=torch.int64, device="cuda")
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    plan = rb.Plan.whole(n)
    reader.begin(rb.seed(bytes(32), 0, bytes(64)), plan, bytes(32))
    half = n // 2
    steps = {"read": lambda: reader.call("rb_read", data.data_ptr(), n * BLOCK, xr.data_ptr(), 16 * sms)}
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
    count, side, roots = n // SEGMENTS, torch.cuda.Stream(), {}
    pair = (torch.cuda.Stream(), torch.cuda.Stream())

    def segmented(lanes: int) -> None:
        main = torch.cuda.current_stream()
        for st in (*pair, side):
            st.wait_stream(main)
        for s in range(SEGMENTS):
            lane = pair[s % lanes]
            with torch.cuda.stream(lane):
                reader.receipts(data[s * count * BLOCK // 4:], s * count, count, ys[s * count:])
                done = torch.cuda.Event()
                done.record(lane)
            with torch.cuda.stream(side):
                side.wait_event(done)
                c0 = s * count // rb.PER_LEAF
                roots[s] = reader.levels(reader.commit(ys[s * count:], c0, count // rb.PER_LEAF), c0)[-1]
        for st in (*pair, side):
            main.wait_stream(st)

    steps[f"segmented/{SEGMENTS}"] = lambda: segmented(1)
    steps[f"segmented/{SEGMENTS}/2"] = lambda: segmented(2)
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
    full = reader.levels(reader.commit(ys, 0, n // rb.PER_LEAF), 0, pad=True)
    h = (count // rb.PER_LEAF).bit_length() - 1
    segment_roots = [digest(roots[s][0]) for s in range(SEGMENTS)] == [digest(x) for x in full[h]]
    nbytes = n * BLOCK
    by_load = {load: {"gb_s": nbytes / ms[f"receipts/{load}"] / 1e6,
                      "z_gb_s": nbytes / (ms[f"receipts_z_half/{load}"] * n / half) / 1e6,
                      "sha512_compressions_per_s": n * 33 / ms[f"receipts/{load}"] * 1e3} for load in LOADS}
    best = min(EXACT, key=lambda load: ms[f"receipts/{load}"])
    best_z = min(EXACT, key=lambda load: ms[f"receipts_z_half/{load}"])
    plain, z = ms[f"receipts/{best}"], ms[f"receipts_z_half/{best_z}"] * n / half
    reader_ms = plain + ms["leaves"] + ms["tree"]
    compressions = n * 33 + n // rb.PER_LEAF * 15
    return {
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
        "tau_segmented_per_pass": ms[f"segmented/{SEGMENTS}"] / ms["read"],
        "tau_segmented_2_per_pass": ms[f"segmented/{SEGMENTS}/2"] / ms["read"],
        "commit_marginal_segmented_2": (ms[f"segmented/{SEGMENTS}/2"] - ms["receipts/warp"]) / ms["read"],
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
    result = {"schema": "verity/device-reader/v0", "device": torch.cuda.get_device_name(0), "gate": g}
    if g["ok"] and a.gib > 0:
        result["measure"] = m = measure(reader, torch, a.gib, a.reps)
        print("reader measure:", json.dumps(m), flush=True)
    (a.out / "reader.json").write_text(json.dumps(result, indent=1))
    return 0 if g["ok"] and result.get("measure", {}).get("segment_roots_match", True) else 7


if __name__ == "__main__":
    sys.exit(main())
