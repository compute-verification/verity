"""The GPU tagger's harness API: `libbwtag.so` (`tagger.cu`'s C ABI) through ctypes.

    tagger = Tagger("libbwtag.so")
    ws = tagger.alloc(tagger.workspace_bytes(tag, blocks))      # or any device buffer of that size (torch, cupy)
    commitment = tagger.tag_pass(tag, store_ptr, blocks, seed, ws)   # E in device memory, 2,056-byte rows
    window.commit(commitment.root)  ->  samples
    window.verdict([commitment.open(j, store_tree) for j in samples])

`tag_pass` is the honest prover's timed step (`audit.tag_pass` on the GPU): it returns once the root is on the host.
`open` is untimed: the block is read back from the device, the tags of its leaf come from the device (`gf31-poly/v1`,
`ro-blake3/v1`) or are recomputed from the block (`ro-sha512/v1`, whose kernel keeps only the leaves), and the tag path is RFC 6962's
`PATH(c)` read from the tree levels the pass left in the workspace.  `store_tree` is the store's own tree under `vk`
(`pous.verifier.MerkleTree`), which the prover keeps from setup.  The checks are the core verifier's (`audit.Window`).

    python gpu.py windows --lib libbwtag.so --leaves store_leaves.bin --blocks N --out windows.json

runs honest windows through `audit.Verifier` on the generated test store (`reference.block`, filled on the device) at
`--deadline-ns` (default the largest `audit.Params` allows), negative controls that tag only part of the store, and one
window per tag at a deadline no certificate allows, which checks the openings of a tag whose honest pass is late.  Before
each timed window the prover reads `--scrub-mib` of another buffer (untimed, before the seed), so the window starts with
L2 clean and holding no part of the store.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import time
from collections.abc import Callable, Sequence
from pathlib import Path

from reference import SALT, geometry
from verity.core.protocols.bandwidth import audit, tags
from verity.core.protocols.pous.verifier import DIGEST_BYTES, MerkleTree, VerifierKey

TAG_IDS = {"gf31-poly/v1": 0, "ro-sha512/v1": 1, "ro-blake3/v1": 2}
BLOCK_BYTES = 2056
MAX_LEVELS = 65
LOOSE_NS = 100_000_000


def bottom_up(leaves: Sequence[bytes], node: Callable[[bytes, bytes], bytes]) -> list[list[bytes]]:
    """The GPU's tree on the host: pair neighbours level by level, an odd last node promoted.  Its root is RFC 6962's."""
    levels = [list(leaves)]
    while len(levels[-1]) > 1:
        cur = levels[-1]
        levels.append([node(cur[i], cur[i + 1]) if i + 1 < len(cur) else cur[i] for i in range(0, len(cur), 2)])
    return levels


def path(sizes: Sequence[int], read: Callable[[int, int], bytes], i: int) -> tuple[bytes, ...]:
    """RFC 6962 `PATH(i)` from a bottom-up tree with level sizes `sizes`, `read(level, index)` giving a node: at each
    level the sibling, none where the node is an odd last one (promoted)."""
    if not 0 <= i < sizes[0]:
        raise IndexError(i)
    out = []
    for level, n in enumerate(sizes[:-1]):
        sibling = i ^ 1
        if sibling < n:
            out.append(read(level, sibling))
        i >>= 1
    return tuple(out)


class TaggerError(RuntimeError):
    pass


class Tagger:
    def __init__(self, lib: str | Path) -> None:
        self.lib = ctypes.CDLL(str(lib))
        L, u64, vp, i32 = self.lib, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_int
        L.bw_workspace_bytes.argtypes, L.bw_workspace_bytes.restype = [i32, u64], ctypes.c_size_t
        L.bw_tag_pass.argtypes = [i32, vp, u64, ctypes.c_char_p, vp, ctypes.c_char_p, vp]
        L.bw_tag_pass_prefix.argtypes = [i32, vp, u64, u64, ctypes.c_char_p, vp, ctypes.c_char_p, vp]
        L.bw_read_tags.argtypes = [i32, u64, vp, u64, u64, vp]
        L.bw_read_node.argtypes = [i32, u64, vp, ctypes.c_uint32, u64, ctypes.c_char_p]
        L.bw_levels.argtypes, L.bw_levels.restype = [i32, u64, ctypes.POINTER(u64)], ctypes.c_uint32
        L.bw_fill_test_store.argtypes = [vp, u64]
        L.bw_read_block.argtypes = [vp, u64, ctypes.c_char_p]
        L.bw_alloc.argtypes, L.bw_alloc.restype = [ctypes.c_size_t], vp
        L.bw_free.argtypes = [vp]
        L.bw_memset.argtypes = [vp, i32, ctypes.c_size_t]
        L.bw_read_scrub.argtypes = [vp, ctypes.c_size_t]
        for f in ("bw_tag_pass", "bw_tag_pass_prefix", "bw_read_tags", "bw_read_node", "bw_fill_test_store",
                  "bw_read_block", "bw_free", "bw_memset", "bw_read_scrub"):
            getattr(L, f).restype = i32

    def _ok(self, rc: int, what: str) -> None:
        if rc:
            raise TaggerError(f"{what} failed ({rc})")

    def alloc(self, nbytes: int) -> int:
        p = self.lib.bw_alloc(nbytes)
        if not p:
            raise TaggerError(f"cudaMalloc of {nbytes} bytes failed")
        return p

    def free(self, ptr: int) -> None:
        self._ok(self.lib.bw_free(ptr), "cudaFree")

    def memset(self, ptr: int, value: int, nbytes: int) -> None:
        self._ok(self.lib.bw_memset(ptr, value, nbytes), "cudaMemset")

    def read_scrub(self, ptr: int, nbytes: int) -> None:
        """Read `nbytes` of a buffer once: before a window, L2 then holds clean lines and no part of the store."""
        self._ok(self.lib.bw_read_scrub(ptr, nbytes), "read scrub")

    def fill_test_store(self, store: int, blocks: int) -> None:
        self._ok(self.lib.bw_fill_test_store(store, blocks), "fill_test_store")

    def read_block(self, store: int, j: int) -> bytes:
        out = ctypes.create_string_buffer(BLOCK_BYTES)
        self._ok(self.lib.bw_read_block(store, j, out), "read_block")
        return out.raw

    def workspace_bytes(self, tag: str, blocks: int) -> int:
        return self.lib.bw_workspace_bytes(TAG_IDS[tag], blocks)

    def levels(self, tag: str, blocks: int) -> tuple[int, ...]:
        sizes = (ctypes.c_uint64 * MAX_LEVELS)()
        return tuple(sizes[:self.lib.bw_levels(TAG_IDS[tag], blocks, sizes)])

    def tag_pass(self, tag: str, store: int, blocks: int, seed: bytes, workspace: int, stream: int = 0) -> Commitment:
        """Every block's tag, the leaves and the root; returns when the root is on the host."""
        if len(seed) != tags.SEED_BYTES:
            raise ValueError(f"a seed is {tags.SEED_BYTES} bytes")
        root = ctypes.create_string_buffer(DIGEST_BYTES)
        self._ok(self.lib.bw_tag_pass(TAG_IDS[tag], store, blocks, seed, workspace, root, stream), "tag pass")
        return Commitment(self, tag, store, blocks, seed, workspace, root.raw)

    def tag_pass_prefix(self, tag: str, store: int, blocks: int, tagged: int, seed: bytes, workspace: int,
                        stream: int = 0) -> Commitment:
        """A dishonest pass (negative controls): only blocks `0..tagged-1` are tagged; the rest of the tree is built
        over whatever the workspace held."""
        root = ctypes.create_string_buffer(DIGEST_BYTES)
        self._ok(self.lib.bw_tag_pass_prefix(TAG_IDS[tag], store, blocks, tagged, seed, workspace, root, stream),
                 "prefix tag pass")
        return Commitment(self, tag, store, blocks, seed, workspace, root.raw)


class Commitment:
    """One pass's root, and its openings from the workspace it left (valid until the workspace is reused)."""

    def __init__(self, tagger: Tagger, tag: str, store: int, blocks: int, seed: bytes, workspace: int,
                 root: bytes) -> None:
        self.tagger, self.tag, self.store, self.blocks = tagger, tags.get(tag), store, blocks
        self.seed, self.workspace, self.root = seed, workspace, root
        self.sizes = tagger.levels(tag, blocks)

    def node(self, level: int, index: int) -> bytes:
        out = ctypes.create_string_buffer(DIGEST_BYTES)
        rc = self.tagger.lib.bw_read_node(TAG_IDS[self.tag.name], self.blocks, self.workspace, level, index, out)
        self.tagger._ok(rc, f"read node {level}/{index}")
        return out.raw

    def leaf_tags(self, c: int) -> tuple[bytes, ...]:
        per = tags.tags_per_leaf(self.tag)
        first, count = c * per, min(per, self.blocks - c * per)
        tag_id = TAG_IDS[self.tag.name]
        if isinstance(self.tag, tags.PolynomialTag):
            out = (ctypes.c_uint32 * count)()
            self.tagger._ok(self.tagger.lib.bw_read_tags(tag_id, self.blocks, self.workspace, first, count, out),
                            "read tags")
            return tuple(int(y).to_bytes(4, "big") for y in out)
        if isinstance(self.tag, tags.Blake3Tag):
            raw = ctypes.create_string_buffer(count * self.tag.tag_bytes)
            self.tagger._ok(self.tagger.lib.bw_read_tags(tag_id, self.blocks, self.workspace, first, count, raw),
                            "read tags")
            return tuple(raw.raw[i * self.tag.tag_bytes:(i + 1) * self.tag.tag_bytes] for i in range(count))
        keyed = self.tag.keyed(self.seed)
        return tuple(keyed(j, self.tagger.read_block(self.store, j)) for j in range(first, first + count))

    def open(self, j: int, store_tree: MerkleTree) -> audit.Opening:
        c = j // tags.tags_per_leaf(self.tag)
        return audit.Opening(j, self.tagger.read_block(self.store, j), store_tree.path(j), self.leaf_tags(c),
                             path(self.sizes, self.node, c))


def _windows(a: argparse.Namespace) -> dict:
    params = geometry(a.blocks)
    raw = a.leaves.read_bytes()
    store_leaves = [raw[i:i + DIGEST_BYTES] for i in range(0, len(raw), DIGEST_BYTES)]
    store_tree = MerkleTree(store_leaves)
    vk = VerifierKey(params, SALT, store_tree.root, tuple(store_leaves))
    t = Tagger(a.lib)
    store = t.alloc(a.blocks * BLOCK_BYTES)
    t.fill_test_store(store, a.blocks)
    scrub_bytes = a.scrub_mib << 20
    scrub = t.alloc(scrub_bytes) if scrub_bytes else 0
    if scrub:
        t.memset(scrub, 0, scrub_bytes)
    source = bytes.fromhex(a.source)
    out: dict = {"blocks": a.blocks, "store_root": vk.root.hex(), "samples": a.samples, "scrub_mib": a.scrub_mib,
                 "results": [], "openings_check": []}
    for name in a.tags:
        ws = t.alloc(t.workspace_bytes(name, a.blocks))
        t.tag_pass(name, store, a.blocks, bytes(tags.SEED_BYTES), ws)  # loads the kernels, untimed
        p = audit.Params(name, a.samples, a.deadline_ns)
        verifier = audit.Verifier(vk, p, source)
        for n, f in enumerate([0.0] * a.windows + a.skip):
            w = verifier.window(n)
            tagged = a.blocks - round(f * a.blocks)
            if f:
                t.memset(ws, 0xA5, t.workspace_bytes(name, a.blocks))
            if scrub:
                t.read_scrub(scrub, scrub_bytes)
            seed = w.start()
            commitment = (t.tag_pass(name, store, a.blocks, seed, ws) if not f
                          else t.tag_pass_prefix(name, store, a.blocks, tagged, seed, ws))
            drawn = w.commit(commitment.root)
            o0 = time.perf_counter_ns()
            openings = [commitment.open(j, store_tree) for j in drawn]
            o1 = time.perf_counter_ns()
            v = w.verdict(openings)
            o2 = time.perf_counter_ns()
            out["results"].append({"tag": name, "window": n, "skipped_fraction": f, "tagged": tagged,
                                   "deadline_ns": p.deadline_ns, "elapsed_ns": v.elapsed_ns, "accepted": v.accepted,
                                   "failure": v.failure, "index": v.index, "open_ms": (o1 - o0) / 1e6,
                                   "verdict_ms": (o2 - o1) / 1e6})
        # The openings alone, at a deadline no certificate allows (the rebuild floor lifted past it): a tag whose
        # honest pass is late still has its openings checked by the verifier.
        loose = audit.Params(name, a.samples, LOOSE_NS, regen_floor_ns=LOOSE_NS + 1)
        w = audit.Verifier(vk, loose, source + b"/openings-check").window(0)
        commitment = t.tag_pass(name, store, a.blocks, w.start(), ws)
        v = w.verdict([commitment.open(j, store_tree) for j in w.commit(commitment.root)])
        out["openings_check"].append({"tag": name, "deadline_ns": LOOSE_NS, "certificate": False,
                                      "elapsed_ns": v.elapsed_ns, "accepted": v.accepted, "failure": v.failure})
        t.free(ws)
    t.free(store)
    if scrub:
        t.free(scrub)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("windows", help="honest windows and skip controls through audit.Verifier")
    w.add_argument("--lib", type=Path, required=True)
    w.add_argument("--leaves", type=Path, required=True)
    w.add_argument("--blocks", type=int, required=True)
    w.add_argument("--samples", type=int, default=audit.SAMPLES)
    w.add_argument("--deadline-ns", type=int, default=audit.REGEN_FLOOR_NS - 1)
    w.add_argument("--scrub-mib", type=int, default=512, help="read this much of another buffer before each window")
    w.add_argument("--windows", type=int, default=20)
    w.add_argument("--skip", type=float, nargs="*", default=[0.01, 0.05])
    w.add_argument("--tags", nargs="*", default=sorted(TAG_IDS))
    w.add_argument("--source", default=b"verity bandwidth gpu windows".hex())
    w.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    result = _windows(a)
    a.out.write_text(json.dumps(result, indent=1) + "\n")
    for r in result["results"]:
        print(json.dumps(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
