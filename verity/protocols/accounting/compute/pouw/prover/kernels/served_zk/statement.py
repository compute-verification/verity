"""A served pass's rows in the ZK statement's format beside the scheme's (-h3's): each decode forward that the engine replays whole
(the device adapter's whole-forward replay) commits, once after the forward and on the replay's stream (so the next
forward waits for it), its calls' A rows and its tile rows as hm96-sha512 leaves of frame-v3-sha512 trees (hm96_rows.cu,
`Commit`).  `PearlCDeviceRun.statement` holds it; it runs while ``enabled`` (e2e.py's ``pearlc-zk`` mode).

- **The rows:** inside the whole capture, each call's A words (m₁₆ FP32 rows, its BF16 x widened, exact) are copied to its
  weight's slot of its k's stage (`stage`).  After the replay (`after`), each k's stage is one commitment of role X rows
  (``hm96-sha512/row-seg/v1``), and the forward's tile leaves (the scheme's, `run.nb` bytes each, standing in for the statement's tile
  digest, TurboSHAKE128, which is not computed) one of ``row/v2`` rows.
- **The salts:** read from the OS on the host for every forward's every tree, as Flock's are (`Commit.fresh_salts`:
  192 bytes a row, copied on the replay's stream before the tree's launch).  Each tree's domain names its k (or the tiles) and m₁₆, not the forward (v0).
- **The roots:** each forward's, copied to a device ring (`ROOTS` forwards), whose SHA-512 over the forwards since the last
  `take` the e2e's decode response reports.
- **What it is not:** the scheme's own A root seeds forming (seed_A), so the statement's roots are in addition to its.  Prefill
  runs eagerly, so its rows are not committed here; nor are the fp8-p rows (the tile's inputs), which a prover commits after
  the pass, before the sample.

`check` is the gate before any timing: a forward of every weight (m = 32) captured whole and replayed from poisoned stages
and outputs must give Python's roots and b ‖ c over the rows it staged, which must be the calls' A words.

The device's tensors (`zeros`, `empty`, `fill_stages`) and the gate (`check`) are benchmark-side code, which verity does not
import: they raise here, and `benchmarks/pouw/served_zk/statement_torch.py`'s Statement supplies them."""
from __future__ import annotations

import ctypes
import hashlib
import os
from pathlib import Path
from typing import Any

from verity.commitments import hm96
from verity.commitments import merkle as MK
from verity.commitments import rowleaf as RL
from verity.commitments.indexed import RangeIndexedDomain

PREFIX = "verity/pouw/served-zk/statement/v0"
#: forwards whose roots the device ring holds
ROOTS = 1 << 12
POISON, SLOT, SEG_TPB = 0xA5, 256, 256
INST, KEY = hm96.SHA512, hm96.SHA512.default_key
SEG, V2 = RL.SCHEMA_HM96_SHA512_ROW_SEG, RL.SCHEMA_HM96_SHA512_ROW_V2
ON_THE_DEVICE = ("Statement.{} is benchmark-side code, which verity does not import (compute-accounting/pouw-package-holds-kernels): "
                 "benchmarks/pouw/served_zk/statement_torch.py's Statement supplies it")


def slots(heads: list[bytes]) -> bytes:
    out = b""
    for h in heads:
        if len(h) > SLOT - 4:
            raise ValueError(f"a message head of {len(h)} bytes does not fit a slot")
        out += (len(h).to_bytes(4, "little") + h).ljust(SLOT, b"\0")
    return out


def node_head(dom: MK.CommitmentDomain, level: int, index: int) -> bytes:
    """The bytes of a frame-v3-sha512 node's message before its left child: the node is SHA-512(head ‖ left ‖ u64be 64 ‖
    right) (merkle's `_message`)."""
    return MK._message(MK._NODE, dom.domain_id, MK._uint(level), MK._uint(index)) + (64).to_bytes(8, "big")


def constants(h: Any) -> dict:
    """hm96_rows.cu's constants on the device: M_key's byte tables and the midstates of hm96's salt and leaf prefixes."""
    tables = b"".join(v.to_bytes(64, "little") for row in INST.byte_tables(KEY) for v in row)
    pre = h.put(INST.salt_prefix + INST.leaf_prefix(KEY))
    consts = {"tables": h.put(tables), "hm_mid": h.alloc(128)}
    h.launch("sz_mid", 1, 32, p=(pre, consts["hm_mid"]), i=(2,))
    return consts


class Commit:
    """One call's rows (`rows` of `nbytes` each at device address `data`, stride `stride`) as hm96-sha512 leaves of a
    frame-v3-sha512 tree under schema `schema` (SEG: row-seg/v1 rows, any whole number of 128-byte chunks a segment; V2: rows of
    at most 111 bytes), with its constants on the device, and its rows' salts read from the OS (`fresh_salts` reads new ones)."""

    def __init__(self, dev, h, consts: dict, tag: str, data: int, rows: int, nbytes: int, stride: int, schema: str,
                 prefix: str = PREFIX):
        self.dev, self.h, self.data, self.rows, self.nbytes, self.stride, self.schema = dev, h, data, rows, nbytes, stride, schema
        self.c = consts
        self.salts = os.urandom(192 * rows)
        self.d_salts = h.put(self.salts)
        n_bits = 8 * nbytes
        if schema == SEG:
            segs = RL.sha512_row_seg_segments(n_bits)
            if any(m % 1024 for m in segs):
                raise ValueError(f"hm96_rows.cu takes segments of whole 128-byte chunks, not {segs}")
            self.s, self.last = len(segs), segs[-1] // 8
            blocks = [RL.sha512_row_seg_segment_prefix(j, self.s, m) for j, m in enumerate(segs)] + [RL.sha512_row_seg_prefix(n_bits)]
        else:
            if nbytes > 111:
                raise ValueError("a v2 row of hm96_rows.cu is at most 111 bytes")
            self.s, self.last = 0, 0
            blocks = [RL.sha512_row_v2_prefix(n_bits)]
        mids = h.alloc(64 * len(blocks))
        h.launch("sz_mid", 1, 32 * ((len(blocks) + 31) // 32), p=(h.put(b"".join(blocks)), mids), i=(len(blocks),))
        self.seg_mids, self.q_mid = mids, mids + 64 * (len(blocks) - 1)
        self.dom = MK.CommitmentDomain(hashlib.sha256(f"{prefix}/{tag}".encode()).digest(), -1,
                                       RangeIndexedDomain(0, rows), hash="sha512")
        self.width = 1 << max(rows - 1, 0).bit_length()
        self.leaf_heads = h.put(slots([self.dom.leaf_prefix(r, r, schema, 64) for r in range(rows)]))
        heads, w, lvl = [], self.width, 0
        while w > 1:
            heads += [node_head(self.dom, lvl, j) for j in range(w // 2)]
            w, lvl = w // 2, lvl + 1
        self.node_heads = h.put(slots(heads) if heads else bytes(SLOT))
        self.digests = h.alloc(64 * rows * max(self.s, 1))
        self.x, self.bc = h.alloc(64 * rows), h.alloc(128 * rows)
        self.tree_bytes = 64 * (2 * self.width - 1)
        self.tree = h.alloc(self.tree_bytes)
        self.pads = b"".join(self.dom.pad(r) for r in range(rows, self.width))
        self.write_pads()

    def fresh_salts(self) -> None:
        """New salts from the OS for the next commitment, copied on the device's stream (after every launch already on it)."""
        self.salts = os.urandom(192 * self.rows)
        self.h.ck("cuMemcpyHtoDAsync_v2", ctypes.c_uint64(self.d_salts), ctypes.c_char_p(self.salts), len(self.salts), self.h.stream)

    def write_pads(self) -> None:
        if self.pads:
            self.dev.ck("cuMemcpyHtoD_v2", ctypes.c_uint64(self.tree + 64 * self.rows), ctypes.c_char_p(self.pads), len(self.pads))

    def segments(self) -> None:
        if self.s:
            total = self.rows * self.s
            blocks = -(-total // SEG_TPB)
            self.h.launch("sz_segments", blocks, SEG_TPB, (SEG_TPB // 32) * 32 * 33 * 4, p=(self.data, self.seg_mids, self.digests),
                          i=(self.rows, self.stride, self.s, self.last))

    def finish(self, mode: int, one_cta: bool = True) -> None:
        c = self.c
        block = min(256, 32 * (-(-self.rows // 32))) if one_cta else 128
        grid = 1 if one_cta else -(-self.rows // block)
        self.h.launch("sz_finish", grid, block, p=(self.digests if self.s else self.data, self.q_mid, c["tables"], self.d_salts,
                                                    c["hm_mid"], self.leaf_heads, self.node_heads, self.x, self.bc, self.tree),
                      i=(self.rows, self.s, self.stride, mode, 0, 0, self.width, self.nbytes))
        if mode == 2 and not one_cta:
            w, off, heads = self.width, 0, self.node_heads
            while w > 1:
                self.h.launch("sz_level", -(-(w // 2) // 128), 128, p=(self.tree, heads), i=(off, w // 2))
                off, heads, w = off + w, heads + SLOT * (w // 2), w // 2

    def run(self, mode: int, one_cta: bool = True) -> None:
        self.segments()
        self.finish(mode, one_cta)

    def reference(self, rows: list[bytes]) -> dict:
        """Python's x, b ‖ c, hm96 leaves and root for these row bytes."""
        n_bits = 8 * self.nbytes
        xs = [RL.sha512_row_seg_digest(r, n_bits) if self.schema == SEG else RL.sha512_row_v2_digest(r, n_bits) for r in rows]
        bcs = [INST.commit_string(KEY, x, self.salts[192 * i:192 * (i + 1)]) for i, x in enumerate(xs)]
        hls = [INST.tree_leaf(KEY, bc) for bc in bcs]
        levels = MK.tree_levels(self.dom, [RL.row_leaf(self.dom, i, hl, self.schema) for i, hl in enumerate(hls)])
        return {"x": b"".join(xs), "bc": b"".join(bcs), "hl": b"".join(hls), "root": levels[-1][0]}

    def read_rows(self) -> list[bytes]:
        return [self.dev.read(self.data + r * self.stride, self.nbytes) for r in range(self.rows)]


def slots_of(ks: dict[int, int]) -> tuple[dict[int, tuple[int, int]], dict[int, int]]:
    """Each weight's slot, (its k, its place among that k's weights by id), from `ks` (weight -> k); and each k's weights."""
    pos, count = {}, {}
    for k in sorted(set(ks.values())):
        mine = sorted(w for w, kk in ks.items() if kk == k)
        pos.update({w: (k, i) for i, w in enumerate(mine)})
        count[k] = len(mine)
    return pos, count


class Statement:
    def __init__(self, run: Any, cubin: str | Path) -> None:
        self.run = run
        raw = Path(cubin).read_bytes()
        self.cubin_sha256 = hashlib.sha256(raw).hexdigest()
        self.h = run.kernel.run.Dev(raw)
        self.h.open()
        self.consts = constants(self.h)
        self.h.sync()
        self.enabled = False
        self.pos, self.count = slots_of({w: wt.k for w, wt in run.weights.items()})
        self.ks = sorted(self.count)
        self.stages: dict[tuple[int, int], Any] = {}
        self.commits: dict[tuple[int, int], list] = {}
        self.ring = self.zeros(ROOTS * 64 * (len(self.ks) + 1))
        self.forwards = self.taken = 0

    # ---- the device's tensors ---------------------------------------------------------------------------------------

    def zeros(self, n: int) -> Any:
        """`n` zero bytes on the run's device."""
        raise NotImplementedError(ON_THE_DEVICE.format("zeros"))

    def empty(self, rows: int, k: int) -> Any:
        """A `rows` × `k` FP32 stage on the run's device."""
        raise NotImplementedError(ON_THE_DEVICE.format("empty"))

    def fill_stages(self, byte: int) -> None:
        """Every stage's bytes set to `byte`, the device synchronized after."""
        raise NotImplementedError(ON_THE_DEVICE.format("fill_stages"))

    # ---- inside and around the whole capture ------------------------------------------------------------------------

    def prepare(self, w: int, sh: Any) -> None:
        """Outside the capture (vLLM's warm-up forward): the stage of `w`'s k at `sh`'s m₁₆."""
        k, key = self.pos[w][0], (self.pos[w][0], sh.m16)
        if key not in self.stages:
            self.stages[key] = self.empty(self.count[k] * sh.m16, k)

    def stage(self, w: int, sh: Any) -> None:
        """Inside the capture, after the call's steps: its A words to its weight's slot, before a call of its shape rewrites
        them."""
        (k, i), m16 = self.pos[w], sh.m16
        st = self.stages.get((k, m16))
        if st is None:
            raise RuntimeError(f"served_zk statement: weight {w}'s call of {m16} rows has no stage: run the forward once outside "
                               f"the capture first")
        st[i * m16:(i + 1) * m16].copy_(sh.xa[:m16])

    def _commits(self, m16: int, tiles: int) -> list:
        key = (m16, tiles)
        if key not in self.commits:
            h = self.h
            cms = [Commit(h, h, self.consts, f"x/k{k}/m{m16}", self.stages[(k, m16)].data_ptr(), self.count[k] * m16, 4 * k,
                          4 * k, SEG) for k in self.ks]
            cms.append(Commit(h, h, self.consts, f"t/m{m16}/{tiles}", 0, tiles, self.run.nb, self.run.nb, V2))
            h.sync()
            self.commits[key] = cms
        return self.commits[key]

    def after(self, calls: tuple, tiles_at: int) -> None:
        """After vLLM launches the replay of a forward of `calls`, its first tile at `tiles_at` of the pass's leaves: the
        forward's commitments and their roots to the ring, on the replaying stream."""
        if {c.w for c in calls} != set(self.pos) or len({c.m16 for c in calls}) != 1:
            raise RuntimeError("served_zk statement: a forward replayed whole calls every weight once, at one m16")
        cms = self._commits(calls[0].m16, sum(c.tiles for c in calls))
        cms[-1].data = self.run.leaves.data_ptr() + self.run.nb * tiles_at
        self.h.stream = self.run.stream()
        try:
            at = self.ring.data_ptr() + 64 * (len(cms) * (self.forwards % ROOTS))
            for j, cm in enumerate(cms):
                cm.fresh_salts()
                cm.run(2, one_cta=False)
                self.h.copy_async(at + 64 * j, cm.tree + cm.tree_bytes - 64, 64)
        finally:
            self.h.stream = None
        self.forwards += 1

    def take(self) -> dict:
        """The forwards committed since the last take, and the SHA-512 of their roots in order (read after a sync)."""
        n, first = self.forwards - self.taken, self.taken
        self.taken = self.forwards
        if n > ROOTS:
            return {"forwards": n, "roots_sha512": None, "wrapped": True}
        per = 64 * (len(self.ks) + 1)
        ring = bytes(self.ring.cpu().numpy())
        roots = b"".join(ring[per * ((first + i) % ROOTS):per * ((first + i) % ROOTS + 1)] for i in range(n))
        return {"forwards": n, "roots_sha512": hashlib.sha512(roots).hexdigest()}

    def poison(self) -> None:
        self.fill_stages(POISON)
        for cms in self.commits.values():
            for cm in cms:
                for p, n in ((cm.x, 64 * cm.rows), (cm.bc, 128 * cm.rows), (cm.tree, cm.tree_bytes)):
                    self.h.ck("cuMemsetD8_v2", ctypes.c_uint64(p), POISON, n)
                cm.write_pads()

    # ---- the gate ---------------------------------------------------------------------------------------------------

    def check(self, weights: dict, m: int = 32) -> dict:
        """The gate (the module's docstring), with hashing and graphs on for its length."""
        raise NotImplementedError(ON_THE_DEVICE.format("check"))

__all__ = ["Commit", "INST", "KEY", "SEG", "SLOT", "Statement", "V2", "constants", "node_head", "slots", "slots_of"]
