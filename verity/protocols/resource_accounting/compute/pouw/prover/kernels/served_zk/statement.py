"""A served pass's rows in the ZK statement's format beside the scheme's (-h3's): each decode forward that the engine replays whole
(the device adapter's whole-forward replay) commits, once after the forward and on the replay's stream (so the next
forward waits for it), its calls' A rows and its tile rows as hm96-sha512 leaves of frame-v3-sha512 trees (hm96_rows.cu,
`rows_bench.Commit`).  `PearlCDeviceRun.statement` holds it; it runs while ``enabled`` (e2e.py's ``pearlc-zk`` mode).

- **The rows:** inside the whole capture, each call's A words (m₁₆ FP32 rows, its BF16 x widened, exact) are copied to its
  weight's slot of its k's stage (`stage`).  After the replay (`after`), each k's stage is one commitment of role X rows
  (``hm96-sha512/row-seg/v1``), and the forward's tile leaves (the scheme's, `run.nb` bytes each, standing in for the statement's tile
  digest, TurboSHAKE128, which is not computed) one of ``row/v2`` rows.
- **The salts:** read from the OS on the host for every forward's every tree, as Flock's are (`rows_bench.Commit.fresh_salts`:
  192 bytes a row, copied on the replay's stream before the tree's launch).  Each tree's domain names its k (or the tiles) and m₁₆, not the forward (v0).
- **The roots:** each forward's, copied to a device ring (`ROOTS` forwards), whose SHA-512 over the forwards since the last
  `take` the e2e's decode response reports.
- **What it is not:** the scheme's own A root seeds forming (seed_A), so the statement's roots are in addition to its.  Prefill
  runs eagerly, so its rows are not committed here; nor are the fp8-p rows (the tile's inputs), which a prover commits after
  the pass, before the sample (rows_bench.py's ``batch.p.hm96``).

`check` is the gate before any timing: a forward of every weight (m = 32) captured whole and replayed from poisoned stages
and outputs must give Python's roots and b ‖ c over the rows it staged, which must be the calls' A words; the no-write
control (the same replay with the commitments left out) must not.

The device's tensors (`zeros`, `empty`, `fill_stages`) and the gate (`check`) are benchmark-side code, which verity does not
import: they raise here, and `benchmarks/pouw/served_zk/statement_torch.py`'s Statement supplies them."""
from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

HERE = Path(__file__).resolve().parent
PREFIX = "verity/pouw/served-zk/statement/v0"
#: forwards whose roots the device ring holds
ROOTS = 1 << 12
POISON = 0xA5
ON_THE_DEVICE = ("Statement.{} is benchmark-side code, which verity does not import (compute-accounting/pouw-package-holds-kernels): "
                 "benchmarks/pouw/served_zk/statement_torch.py's Statement supplies it")


def rows_bench() -> ModuleType:
    name = "served_zk_rows_bench"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, HERE / "rows_bench.py")
        mod = sys.modules[name] = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    return sys.modules[name]


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
        self.B, self.run = rows_bench(), run
        raw = Path(cubin).read_bytes()
        self.cubin_sha256 = hashlib.sha256(raw).hexdigest()
        self.h = run.kernel.run.Dev(raw)
        self.h.open()
        self.consts = self.B.constants(self.h)
        self.h.sync()
        self.enabled = self.skip = False
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
            B, h = self.B, self.h
            cms = [B.Commit(h, h, self.consts, f"x/k{k}/m{m16}", self.stages[(k, m16)].data_ptr(), self.count[k] * m16, 4 * k,
                            4 * k, B.SEG, prefix=PREFIX) for k in self.ks]
            cms.append(B.Commit(h, h, self.consts, f"t/m{m16}/{tiles}", 0, tiles, self.run.nb, self.run.nb, B.V2, prefix=PREFIX))
            h.sync()
            self.commits[key] = cms
        return self.commits[key]

    def after(self, calls: tuple, tiles_at: int) -> None:
        """After vLLM launches the replay of a forward of `calls`, its first tile at `tiles_at` of the pass's leaves: the
        forward's commitments and their roots to the ring, on the replaying stream."""
        if {c.w for c in calls} != set(self.pos) or len({c.m16 for c in calls}) != 1:
            raise RuntimeError("served_zk statement: a forward replayed whole calls every weight once, at one m16")
        cms = self._commits(calls[0].m16, sum(c.tiles for c in calls))
        if self.skip:
            return
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
                    self.h.ck("cuMemsetD8_v2", self.B.C.c_uint64(p), POISON, n)
                cm.write_pads()

    # ---- the gate ---------------------------------------------------------------------------------------------------

    def check(self, weights: dict, m: int = 32) -> dict:
        """The gate (the module's docstring), with hashing and graphs on for its length."""
        raise NotImplementedError(ON_THE_DEVICE.format("check"))

__all__ = ["Statement", "slots_of"]
