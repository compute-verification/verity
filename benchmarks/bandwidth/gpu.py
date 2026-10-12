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
window per tag at a deadline no certificate allows, which checks the openings of a tag whose honest pass is late.  Every
window goes through `Prover`, which prepares before it asks for the seed: one untimed pass under a throwaway seed, then
a read of `SCRUB_BYTES` of another buffer, so the window starts with its launch path warm and L2 clean and holding no
part of the store.  Without the pass, the last window's openings and verdict (seconds of Python for `ro-blake3/v1`)
leave up to 0.2 ms of host time inside the next window; without the read, L2's dirty lines cost the pass 0.6 ms.  Each
window records the pass's device time and the host call's beside the verifier's.
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
SCRUB_BYTES = 512 << 20
"""The read before each window: four times the RTX PRO 6000's 128 MiB of L2."""


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
        L.bw_last_pass_ms.argtypes = [ctypes.POINTER(ctypes.c_float)] * 2
        for f in ("bw_tag_pass", "bw_tag_pass_prefix", "bw_read_tags", "bw_read_node", "bw_fill_test_store",
                  "bw_read_block", "bw_free", "bw_memset", "bw_read_scrub", "bw_last_pass_ms"):
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

    def last_pass_ms(self) -> tuple[float, float]:
        """The last pass's device time from its first tag kernel: to the tags' end and to the root, in ms."""
        tags_ms, pass_ms = ctypes.c_float(), ctypes.c_float()
        self._ok(self.lib.bw_last_pass_ms(ctypes.byref(tags_ms), ctypes.byref(pass_ms)), "last pass time")
        return tags_ms.value, pass_ms.value

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


class Prover:
    """The honest prover of one tag over a store in device memory.  `commit(window)` always prepares before it asks for
    `σ`: one untimed pass under a throwaway seed, then a read of `SCRUB_BYTES` of its own buffer, so the window starts
    with the launch path warm and L2 clean and holding no part of the store.  The bound holds for any state before `σ`,
    so the preparation is the prover's liveness, not the verifier's concern."""

    def __init__(self, tagger: Tagger, tag: str, store: int, blocks: int, workspace: int, scrub: int) -> None:
        self.tagger, self.tag, self.store, self.blocks, self.workspace, self.scrub = (tagger, tag, store, blocks,
                                                                                      workspace, scrub)

    def commit(self, window: audit.Window, tagged: int | None = None) -> tuple[Commitment, tuple[int, ...], float]:
        """One window: prepare, take `σ` (`window.start`), run the pass and commit its root.  Returns the commitment,
        the samples and the host call's time in ms.  `tagged` below the store's blocks is a negative control that tags
        only blocks `0..tagged-1` over a poisoned workspace."""
        t, tagged = self.tagger, self.blocks if tagged is None else tagged
        t.tag_pass(self.tag, self.store, self.blocks, bytes(tags.SEED_BYTES), self.workspace)
        if tagged < self.blocks:
            t.memset(self.workspace, 0xA5, t.workspace_bytes(self.tag, self.blocks))
        t.read_scrub(self.scrub, SCRUB_BYTES)
        seed = window.start()
        c0 = time.perf_counter_ns()
        commitment = (t.tag_pass(self.tag, self.store, self.blocks, seed, self.workspace) if tagged == self.blocks
                      else t.tag_pass_prefix(self.tag, self.store, self.blocks, tagged, seed, self.workspace))
        c1 = time.perf_counter_ns()
        return commitment, window.commit(commitment.root), (c1 - c0) / 1e6


def _windows(a: argparse.Namespace) -> dict:
    params = geometry(a.blocks)
    raw = a.leaves.read_bytes()
    store_leaves = [raw[i:i + DIGEST_BYTES] for i in range(0, len(raw), DIGEST_BYTES)]
    store_tree = MerkleTree(store_leaves)
    vk = VerifierKey(params, SALT, store_tree.root, tuple(store_leaves))
    t = Tagger(a.lib)
    store = t.alloc(a.blocks * BLOCK_BYTES)
    t.fill_test_store(store, a.blocks)
    scrub = t.alloc(SCRUB_BYTES)
    t.memset(scrub, 0, SCRUB_BYTES)
    source = bytes.fromhex(a.source)
    out: dict = {"blocks": a.blocks, "store_root": vk.root.hex(), "samples": a.samples, "scrub_mib": SCRUB_BYTES >> 20,
                 "results": [], "openings_check": []}
    audit.ChallengerState.create(a.state, vk)
    with audit.ChallengerState(a.state, vk) as state:
        for name in a.tags:
            ws = t.alloc(t.workspace_bytes(name, a.blocks))
            prover = Prover(t, name, store, a.blocks, ws, scrub)
            p = audit.Params(name, a.samples, a.deadline_ns)
            verifier = audit.Verifier(vk, p, source, state)
            for n, f in enumerate([0.0] * a.windows + a.skip):
                w = verifier.window()
                tagged = a.blocks - round(f * a.blocks)
                commitment, drawn, call_ms = prover.commit(w, tagged)
                tags_ms, pass_ms = t.last_pass_ms()
                o0 = time.perf_counter_ns()
                openings = [commitment.open(j, store_tree) for j in drawn]
                o1 = time.perf_counter_ns()
                v = w.verdict(openings)
                o2 = time.perf_counter_ns()
                out["results"].append({"tag": name, "window": n, "skipped_fraction": f, "tagged": tagged,
                                       "deadline_ns": p.deadline_ns, "elapsed_ns": v.elapsed_ns,
                                       "accepted": v.accepted, "failure": v.failure, "index": v.index,
                                       "device_tags_ms": tags_ms, "device_pass_ms": pass_ms, "call_ms": call_ms,
                                       "open_ms": (o1 - o0) / 1e6, "verdict_ms": (o2 - o1) / 1e6})
            # The openings alone, at a deadline no certificate allows (the rebuild floor lifted past it): a tag whose
            # honest pass is late still has its openings checked by the verifier.
            loose = audit.Params(name, a.samples, LOOSE_NS, regen_floor_ns=LOOSE_NS + 1)
            w = audit.Verifier(vk, loose, source, state).window()
            commitment, drawn, _ = prover.commit(w)
            v = w.verdict([commitment.open(j, store_tree) for j in drawn])
            out["openings_check"].append({"tag": name, "deadline_ns": LOOSE_NS, "certificate": False,
                                          "elapsed_ns": v.elapsed_ns, "accepted": v.accepted, "failure": v.failure})
            t.free(ws)
    t.free(store)
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
    w.add_argument("--windows", type=int, default=20)
    w.add_argument("--skip", type=float, nargs="*", default=[0.01, 0.05])
    w.add_argument("--tags", nargs="*", default=sorted(TAG_IDS))
    w.add_argument("--source", default=b"verity bandwidth gpu windows".hex())
    w.add_argument("--state", type=Path, required=True,
                   help="the store's challenger state, made here: refused if the file exists")
    w.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    result = _windows(a)
    a.out.write_text(json.dumps(result, indent=1) + "\n")
    for r in result["results"]:
        print(json.dumps(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
