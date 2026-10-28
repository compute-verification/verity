"""A served pass's rows in the ZK statement's format beside the scheme's (-h3's): each decode forward that vLLM replays whole
(`verity_vllm.protocols.pouw.pearl_c.whole`) commits, once after the forward and on the replay's stream (so the next
forward waits for it), its calls' A rows and its tile rows as hm96-sha512 leaves of frame-v3-sha512 trees (hm96_rows.cu,
`rows_bench.Commit`).  `PearlCDeviceRun.statement` holds it; it runs while ``enabled`` (e2e.py's ``pearlc-zk`` mode).

- **The rows:** inside the whole capture, each call's A words (m₁₆ FP32 rows, its BF16 x widened, exact) are copied to its
  weight's slot of its k's stage (`stage`).  After the replay (`after`), each k's stage is one commitment of role X rows
  (``hm96-sha512/row-seg/v1``), and the forward's tile leaves (the scheme's, `run.nb` bytes each, standing in for the statement's tile
  digest, TurboSHAKE128, which is not computed) one of ``row/v2`` rows.
- **The salts:** read from the OS on the host for every forward's every tree, as C-Flock's are (`rows_bench.Commit.fresh_salts`:
  192 bytes a row, copied on the replay's stream before the tree's launch).  Each tree's domain names its k (or the tiles) and m₁₆, not the forward (v0).
- **The roots:** each forward's, copied to a device ring (`ROOTS` forwards), whose SHA-512 over the forwards since the last
  `take` the e2e's decode response reports.
- **What it is not:** the scheme's own A root seeds forming (seed_A), so the statement's roots are in addition to its.  Prefill
  runs eagerly, so its rows are not committed here; nor are the fp8-p rows (the tile's inputs), which a prover commits after
  the pass, before the draw (rows_bench.py's ``batch.p.hm96``).

`check` is the gate before any timing: a forward of every weight (m = 32) captured whole and replayed from poisoned stages
and outputs must give Python's roots and b ‖ c over the rows it staged, which must be the calls' A words; the no-write
control (the same replay with the commitments left out) must not."""
from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
PREFIX = "verity/pouw/served-zk/statement/v0"
#: forwards whose roots the device ring holds
ROOTS = 1 << 12
POISON = 0xA5


def rows_bench():
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


class StatementBench:
    def __init__(self, run: Any, cubin: str | Path) -> None:
        import torch
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
        self.ring = torch.zeros(ROOTS * 64 * (len(self.ks) + 1), dtype=torch.uint8, device=run.device)
        self.forwards = self.taken = 0

    # ---- inside and around the whole capture ------------------------------------------------------------------------

    def prepare(self, w: int, sh: Any) -> None:
        """Outside the capture (vLLM's warm-up forward): the stage of `w`'s k at `sh`'s m₁₆."""
        import torch
        k, key = self.pos[w][0], (self.pos[w][0], sh.m16)
        if key not in self.stages:
            self.stages[key] = torch.empty((self.count[k] * sh.m16, k), dtype=torch.float32, device=self.run.device)

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
        import torch
        with torch.inference_mode():
            for st in self.stages.values():
                st.view(torch.uint8).fill_(POISON)
        torch.cuda.synchronize(self.run.device)
        for cms in self.commits.values():
            for cm in cms:
                for p, n in ((cm.x, 64 * cm.rows), (cm.bc, 128 * cm.rows), (cm.tree, cm.tree_bytes)):
                    self.h.ck("cuMemsetD8_v2", self.B.C.c_uint64(p), POISON, n)
                cm.write_pads()

    # ---- the gate ---------------------------------------------------------------------------------------------------

    def check(self, weights: dict, m: int = 32) -> dict:
        """The gate (the module's docstring), with hashing and graphs on for its length."""
        import torch
        run = self.run
        dev = run.device
        gen = torch.Generator(device=dev).manual_seed(20261006)
        xs = {w: torch.randn(m, W.shape[1], device=dev, generator=gen).to(torch.bfloat16) for w, W in weights.items()}
        forward = lambda: [run.gemm(w, xs[w], W, None) for w, W in weights.items()]  # noqa: E731
        held = run.graphs, run.hashing, self.enabled
        run.graphs, run.hashing, self.enabled = True, True, True
        res: dict = {}
        try:
            run.begin()
            forward()
            run.end()
            run.begin()
            side, graph = torch.cuda.Stream(device=dev), torch.cuda.CUDAGraph()
            run.whole_capture_begin(run.whole_key())
            try:
                forward()
                side.wait_stream(torch.cuda.current_stream(dev))
                with torch.cuda.graph(graph, stream=side):
                    forward()
            finally:
                calls = run.whole_capture_end()
            run.end()
            for name, skip in (("honest", False), ("control", True)):
                run.begin()
                run.poison()
                self.poison()
                at = run.cur.tiles
                run.whole_replay(calls)
                graph.replay()
                self.skip = skip
                try:
                    run.whole_replayed(calls)
                finally:
                    self.skip = False
                torch.cuda.synchronize(dev)
                cms = self._commits(m, sum(c.tiles for c in calls))
                cms[-1].data = run.leaves.data_ptr() + run.nb * at
                same = []
                for cm in cms:
                    want = cm.reference(cm.read_rows())
                    same.append(self.h.read(cm.tree + cm.tree_bytes - 64, 64) == want["root"]
                                and self.h.read(cm.bc, 128 * cm.rows) == want["bc"])
                staged = all(torch.equal(self.stages[(k, m)][i * m:(i + 1) * m], xs[w].float()) for w, (k, i) in self.pos.items())
                res[name] = {"roots_equal": same, "staged_rows_equal": staged}
                run.end()
        finally:
            run.graphs, run.hashing, self.enabled = held
            self.forwards = self.taken = 0
        rows = {f"x/k{k}": self.count[k] * m for k in self.ks}
        rows["tiles"] = sum(c.tiles for c in calls)
        ok = (all(res["honest"]["roots_equal"]) and res["honest"]["staged_rows_equal"] and res["control"]["staged_rows_equal"]
              and not any(res["control"]["roots_equal"]))
        return {"check": "statement", "cubin_sha256": self.cubin_sha256, "calls": len(calls), "rows": rows, **res, "ok": ok}
