"""The prover's pass over a store where its tensors hold it, on one GPU and with no copy of the store: vLLM's weight
tensors and KV pages as its binder registers them (`verity_vllm.protocols.poub.binder`'s `TorchBacking` and `PagedBacking`).

~~~text
region     store chunks first … first + blocks − 1 (`Region`): a weight tensor's storage, chunk b at addr + 4096 b,
           or a KV layer's pages, plane q = b // plane_blocks (page q // planes, plane q mod planes) at addr
           + page · page_stride + plane · plane_stride with chunk b's 4096 bytes (b mod plane_blocks) · 4096 into it
           (`PagedBacking.where`)
map        a pass's plan (any `receipts.Plan` over the regions) to each position's device address and store index
           j = plan[p], on the device from the plan's runs and the regions (`rs_map`)
pass       its sketch-sha512 receipts on tensor cores, each chunk read through the map, and `receipts.tree`'s leaves
           (the last one short), padding and levels hashed behind them as they are read (`rs_mapped_pass`), leaf c's
           salt `device.store_salt(key, seed, c)`; the pass keeps its levels from h0 up
opening    the 2^h0-leaf subtree a drawn position falls in, recomputed under the pass's seed and plan from the store as
           it is when it opens (`rs_mapped_receipts`, then reader.cu's `rb_leaves`, `rb_pads` and `rb_tree`), and the
           path's levels from h0 up from the pass
~~~

Every region's tensor is on the current GPU, at a 16-byte-aligned address with 16-byte-aligned strides, and stays
where it is for the store's life (a weight tensor isn't reallocated, the KV pool is allocated once). The store grows
as chunks are appended: vLLM's memory map gives a written chunk a fresh index j while its bytes stay in its slot, the
registered chunk it was written in, and `append` maps the store's next indices to their slots' bytes. A pass reads
what the tensors hold, so its receipts are the session's only where they hold the codec's encoding. An opening reads
a chunk as it is when it opens, so a chunk rewritten since its pass opens to what it holds now; the caller's
staleness check (vLLM's `Stale`) is what refuses it.
"""
from __future__ import annotations

import bisect
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from device import BLOCK, PACK16_WORDS, PER_LEAF, SALT_DOMAIN, Lib, digests, rb, store_salt, stream_head

ROW = 64
"""rs_gather64's row: a chunk is BLOCK // ROW of them."""

READY_BLOCKS = 1 << 13
"""session.cu's CONSUMER_READY_BLOCKS: the positions of a group whose receipts `rs_mapped_pass` counts together."""

FUSED_H0 = 8
"""The largest h0 `rs_mapped_pass` takes: it hashes each 2^h0-leaf subtree in shared memory."""


@dataclass(frozen=True)
class Region:
    """Store chunks first … first + blocks − 1 where a tensor holds them on the GPU (the module's docstring): chunk b at
    addr + 4096 b, or, paged (plane_blocks > 0), in plane q = b // plane_blocks at addr + (q // planes) · page_stride +
    (q mod planes) · plane_stride, (b mod plane_blocks) · 4096 bytes in."""

    first: int
    blocks: int
    addr: int
    plane_blocks: int = 0
    planes: int = 1
    page_stride: int = 0
    plane_stride: int = 0

    @classmethod
    def of_backing(cls, first: int, backing) -> Region:
        """A registered region at store chunk `first` from its vLLM backing: a `TorchBacking`'s ``flat`` (its whole
        chunks), or a `PagedBacking`'s pages through its ``physical`` layout, on a GPU."""
        physical = getattr(backing, "physical", None)
        if physical is None:
            return cls._checked(first, backing.nbytes // BLOCK, backing.flat, 0, 0, 1, 0, 0)
        ps, ks = physical.strides()
        return cls._checked(first, backing.nbytes // BLOCK, backing.flat, backing.base, cls._plane_blocks(backing),
                            physical.planes, ps, ks, end=(physical.pages - 1) * ps + (physical.planes - 1) * ks)

    def address(self, j: int) -> int:
        """Store chunk j's device address (`map_kernel`'s)."""
        x = j - self.first
        if not self.plane_blocks:
            return self.addr + BLOCK * x
        q = x // self.plane_blocks
        return (self.addr + q // self.planes * self.page_stride + q % self.planes * self.plane_stride
                + BLOCK * (x % self.plane_blocks))

    @staticmethod
    def _plane_blocks(backing) -> int:
        if backing.plane % BLOCK:
            raise ValueError(f"a {backing.plane}-byte KV plane is not whole {BLOCK}-byte blocks")
        return backing.plane // BLOCK

    @classmethod
    def _checked(cls, first, blocks, flat, base, plane_blocks, planes, page_stride, plane_stride, end=None) -> Region:
        if not getattr(flat, "is_cuda", False):
            raise ValueError("a region read in place is on a GPU")
        if blocks < 1:
            raise ValueError("a region is one or more whole blocks")
        last = (base + blocks * BLOCK) if end is None else base + end + plane_blocks * BLOCK
        if last > flat.numel() * flat.element_size():
            raise ValueError(f"a region's last block ends at byte {last}, past its storage's "
                             f"{flat.numel() * flat.element_size()}")
        return cls(first, blocks, flat.data_ptr() + base, plane_blocks, planes, page_stride, plane_stride)


@dataclass(frozen=True)
class Pass:
    """What a pass keeps for its openings: its seed and plan, its tree's height H and h0, and its levels h0 … H (level
    h at the sum of width >> f over h0 <= f < h, the root last) on the device."""

    seed: bytes
    plan: rb.Plan
    H: int
    h0: int
    kept: object


def level_at(width: int, h0: int, h: int) -> int:
    """Level h's first node in a tree of `width` leaves kept from level h0 (`rb_tree`'s layout, level h0 first)."""
    return sum(width >> f for f in range(h0, h))


class InPlace:
    """Passes over the store `regions` describe (`Region`, in store order from chunk 0) on the current GPU, read in
    place, leaf salts under `key`; `kept_from` is h0 (at most FUSED_H0), and `ctas` the receipt grids' CTAs (0: an
    opening's as many as are resident at once, a pass's one an SM). A pass's tree is hashed on a stream of its own
    beside the caller's."""

    def __init__(self, torch, lib: Lib, regions: Sequence[Region], key: bytes, *, kept_from: int = 6,
                 ctas: int = 0) -> None:
        if not 0 <= kept_from <= FUSED_H0:
            raise ValueError(f"a pass keeps its levels from h0 = 0 … {FUSED_H0} up, the subtree below h0 hashed in "
                             f"shared memory: kept_from {kept_from}")
        self.torch, self.lib, self.key, self.kept_from, self.ctas = torch, lib, key, kept_from, ctas
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.side = torch.cuda.Stream(self.device)
        self.regions: tuple[Region, ...] = ()
        self.blocks = 0
        self._firsts: list[int] = []
        self._rows = np.zeros((0, 7), dtype=np.uint64)
        self.extend(regions)

    def extend(self, regions: Sequence[Region]) -> None:
        """Regions from the store's end on, in store order: its registered regions, or chunks appended (`append`)."""
        at, out = self.blocks, []
        for r in regions:
            if r.first != at or r.blocks < 1:
                raise ValueError(f"regions are the store's blocks in order, one or more each: one starts at block "
                                 f"{r.first}, not {at}")
            if r.addr % 16 or r.page_stride % 16 or r.plane_stride % 16:
                raise ValueError(f"the region at block {r.first} is read in 16-byte words: its address and strides "
                                 f"are multiples of 16")
            if r.plane_blocks and (r.planes < 1 or r.blocks % (r.planes * r.plane_blocks)):
                raise ValueError(f"the region at block {r.first}: {r.blocks} blocks are not whole pages of {r.planes} "
                                 f"planes of {r.plane_blocks} blocks")
            at += r.blocks
            out.append(r)
        if not out and not self.regions:
            raise ValueError("a store is one or more regions")
        rows = np.array([[r.first, r.blocks, r.addr, r.plane_blocks, r.planes, r.page_stride, r.plane_stride]
                         for r in out], dtype=np.uint64).reshape(-1, 7)
        self._rows = np.concatenate([self._rows, rows])
        self.table = self.torch.from_numpy(self._rows.view(np.int64)).to(self.device)
        self.regions, self.blocks = self.regions + tuple(out), at
        self._firsts.extend(r.first for r in out)

    def append(self, slots: Sequence[int]) -> int:
        """The store's next indices, the k-th at the bytes of store chunk slots[k] (an appended chunk's j at its slot,
        vLLM's memory map); slots back to back in memory share a region. Returns the first's index."""
        if any(type(s) is not int or not 0 <= s < self.blocks for s in slots):
            raise ValueError(f"an appended block's slot is one of the store's {self.blocks} blocks")
        first, out = self.blocks, []
        for k, s in enumerate(slots):
            a = self.regions[bisect.bisect_right(self._firsts, s) - 1].address(s)
            if out and out[-1][1] + BLOCK * out[-1][2] == a:
                out[-1][2] += 1
            else:
                out.append([first + k, a, 1])
        self.extend([Region(f, c, a) for f, a, c in out])
        return first

    def map(self, plan: rb.Plan):
        """Each position's device address and store index, two int64 tensors of plan.length (`rs_map`)."""
        torch = self.torch
        runs = np.frombuffer(plan.encode(), dtype=">u8", offset=4).reshape(-1, 2).astype(np.uint64)
        if int((runs[:, 0] + runs[:, 1]).max()) > self.blocks:
            raise ValueError(f"the plan reads past the store's {self.blocks} blocks")
        pairs = np.empty_like(runs)
        pairs[:, 0] = runs[:, 0]
        pairs[0, 1] = 0
        np.cumsum(runs[:-1, 1], out=pairs[1:, 1])
        d = torch.from_numpy(pairs.view(np.int64)).to(self.device)
        n = plan.length
        addrs, js = (torch.empty(n, dtype=torch.int64, device=self.device) for _ in range(2))
        self.lib.call("rs_map", d.data_ptr(), len(pairs), self.table.data_ptr(), len(self.regions), n,
                      addrs.data_ptr(), js.data_ptr())
        return addrs, js

    def set_pass(self, seed: bytes, plan: rb.Plan):
        """A pass's constants on the current GPU (its plan's last leaf short) and its packed signs."""
        n, leaves = plan.length, rb.leaves(plan.length)
        self.lib.set_tree(rb.prefix(seed), rb.DIGEST_BYTES * PER_LEAF, rb.DIGEST_BYTES * (n - PER_LEAF * (leaves - 1)),
                          rb.domain(seed, plan).domain_id, SALT_DOMAIN + self.key + seed, n)
        self.lib.call("rb_sketch_set", rb.sketch_prefix(seed))
        head = stream_head(seed)
        pk = self.torch.empty(PACK16_WORDS, dtype=self.torch.int32, device=self.device)
        self.lib.call("rs_signs16", head, len(head), pk.data_ptr())
        return pk

    def receipts(self, addrs, js, pk, p0: int, n: int):
        """Positions p0 … p0 + n − 1's receipts (n × 8 words) under the pass constants set."""
        ys = self.torch.empty((n, 8), dtype=self.torch.int64, device=self.device)
        self.lib.call("rs_mapped_receipts", addrs.data_ptr() + 8 * p0, js.data_ptr() + 8 * p0, n, pk.data_ptr(),
                      ys.data_ptr(), self.ctas)
        return ys

    def tree(self, ys, c0: int, real: int, width: int):
        """The 2 · width − 1 nodes of the subtree of `width` leaves from leaf c0: `real` leaves from ys, then padding,
        then every level above (`rb_tree`'s layout, level 0 first)."""
        nodes = self.torch.empty((2 * width - 1, 8), dtype=self.torch.int64, device=self.device)
        self.lib.call("rb_leaves", ys.data_ptr(), c0, real, nodes.data_ptr(), None)
        if real < width:
            self.lib.call("rb_pads", c0 + real, width - real, nodes.data_ptr() + ROW * real)
        if width > 1:
            self.lib.call("rb_tree", nodes.data_ptr(), width, c0, nodes.data_ptr() + ROW * width)
        return nodes

    def pass_receipts(self, seed: bytes, plan: rb.Plan):
        """Every receipt of the pass (plan.length × 8 words), for a gate."""
        addrs, js = self.map(plan)
        return self.receipts(addrs, js, self.set_pass(seed, plan), 0, plan.length)

    def run_pass(self, seed: bytes, plan: rb.Plan) -> tuple[bytes, Pass]:
        """One pass under `seed` over `plan`: its root and what its openings need (`Pass`)."""
        addrs, js = self.map(plan)
        H, h0, kept = self.fused(addrs, js, self.set_pass(seed, plan), plan.length)
        root = digests(kept[-1:].cpu())[0]
        return root, Pass(seed, plan, H, h0, kept)

    def fused(self, addrs, js, pk, n: int):
        """The receipts and tree of n positions mapped (`map`) under the pass constants and signs set (`set_pass`),
        the tree hashed behind the receipts (`rs_mapped_pass`): its H, h0 and its levels h0 … H on the device."""
        torch = self.torch
        H = (rb.leaves(n) - 1).bit_length()
        h0 = min(self.kept_from, H)
        top = 1 << (H - h0)
        ys = torch.empty((n, 8), dtype=torch.int64, device=self.device)
        kept = torch.empty((2 * top - 1, 8), dtype=torch.int64, device=self.device)
        ctr = torch.empty(2 + -(-n // READY_BLOCKS) + top, dtype=torch.int32, device=self.device)
        self.lib.call("rs_mapped_pass", addrs.data_ptr(), js.data_ptr(), n, pk.data_ptr(), ctr.data_ptr(), len(ctr),
                      ys.data_ptr(), kept.data_ptr(), h0, self.ctas, self.side.cuda_stream)
        return H, h0, kept

    def open(self, handle: Pass, positions: Sequence[int]) -> list[tuple[bytes, bytes, bytes, tuple[bytes, ...]]]:
        """Each position's (chunk as the store holds it now, its leaf's receipts, the leaf's salt, the leaf's path):
        `receipts.verify_leaf`'s leaf, salt and path."""
        torch, seed, plan, H, h0 = self.torch, handle.seed, handle.plan, handle.H, handle.h0
        n, leaves, width, sub = plan.length, rb.leaves(plan.length), 1 << H, 1 << h0
        if any(type(p) is not int or not 0 <= p < n for p in positions):
            raise IndexError(f"a pass of {n} positions opens positions 0 … {n - 1}")
        addrs, js = self.map(plan)
        pk = self.set_pass(seed, plan)
        subs = {}
        for s0 in sorted({p // PER_LEAF >> h0 << h0 for p in positions}):
            p0 = PER_LEAF * s0
            ys = self.receipts(addrs, js, pk, p0, min(PER_LEAF * sub, n - p0))
            subs[s0] = (ys.cpu(), self.tree(ys, s0, min(sub, leaves - s0), sub).cpu())
        kept = handle.kept.cpu()
        at = addrs[torch.tensor(list(positions), dtype=torch.int64, device=self.device)].cpu().numpy().view(np.uint64)
        rows = (at[:, None] + np.arange(0, BLOCK, ROW, dtype=np.uint64)).reshape(-1)
        got = torch.empty((len(rows), 8), dtype=torch.int64, device=self.device)
        if len(rows):
            self.lib.call("rs_gather64", torch.from_numpy(rows.view(np.int64)).to(self.device).data_ptr(), len(rows),
                          got.data_ptr())
        blocks = got.cpu().numpy().view(np.uint8).reshape(len(positions), BLOCK)
        out = []
        for k, p in enumerate(positions):
            c = p // PER_LEAF
            s0 = c >> h0 << h0
            ys, nodes = subs[s0]
            q = PER_LEAF * (c - s0)
            leaf = b"".join(digests(ys[q:q + min(PER_LEAF, n - PER_LEAF * c)]))
            path = []
            for h in range(H):
                sib = (c >> h) ^ 1
                row = (nodes[level_at(sub, 0, h) + sib - (s0 >> h)] if h < h0
                       else kept[level_at(width, h0, h) + sib])
                path.append(digests(row[None])[0])
            out.append((blocks[k].tobytes(), leaf, store_salt(self.key, seed, c), tuple(path)))
        return out
