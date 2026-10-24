"""The Pearl-C device path's passes (`pouw_pearl_c_device.PearlCDeviceRun`'s): a pass's state (`_Pass`) and its end
(`PassEnds`): one tile tree over every leaf under `audit.transcript_domain`, A's roots and the screen's counts as the pass's
commitment, a retained pass's files and manifest (`write_manifest`), and a late end's copies (`_Late`), settled at the next
end.
"""
from __future__ import annotations

import hashlib
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from verity.core.protocols.pouw.audit import Epoch, Workload, transcript_domain
from verity.core.protocols.pouw.protocol import Matmul
from verity.core.protocols.pouw.schemes import pearl_c_work as PW
from verity.core.protocols.pouw.serving import Schedule, SchemeId

from .pouw_pearl_c_graphs import _capturing
from .pouw_pearl_c_host import _bytes, _put, _write, _zeros, write_manifest
from .pouw_pearl_c_manifest import MANIFEST_SCHEMA


@dataclass
class _Pass:
    retain: Path | None
    run: str | None = None          # the run id a retained pass's manifest names
    matmuls: list[Matmul] = field(default_factory=list)
    flag_rows: int = 0               # rows of the pass's screen flags so far
    records: list[PW.UnitRecord] = field(default_factory=list)
    files: list[dict] = field(default_factory=list)
    tiles: int = 0
    poison: int | None = None       # the byte every call buffer starts the pass as, those created during it included
    stepped: bool = False            # begun by an engine step's end (`PearlCDeviceRun.end_pass`'s ``step``)


@dataclass
class _Late:
    """A pass `PearlCDeviceRun.end` closed late: its commitment so far, the pass and its tile count, the host copies of A's
    roots, the tree's root and the screen's two sums, the event after them (None off the GPU) on stream `raw`, and the
    device tree they read, held until then."""
    out: dict
    cur: _Pass
    count: int
    host: tuple
    event: Any
    raw: int
    tree: Any


class PassEnds:
    """`PearlCDeviceRun`'s pass ends, as its base class: the run's ``cur``, ``hashing``, ``kernel``, ``nb``, ``roots``,
    ``salt``, ``schedule``, ``leaves``, ``roots_buf``, ``flag_buf``, ``const_keys``, ``tree_hash``, ``details``,
    ``commits``, ``weights``, ``entry``, ``scheme``, ``coins``, ``opening``, ``salt_key``, ``device``, ``_late``,
    ``_late_host`` and ``_late_event``, and its ``stream``, ``torch_stream``, ``join_sides`` and ``begin``."""

    def end(self, late: bool = False) -> dict[str, Any]:
        """Close the pass: its tile tree on the device, every A root, and with `retain` the manifest.  Returns the pass's
        commitment (also in `commits` and `details["passes"]`).  `late`, for an unretained pass of a schedule on the caller's
        stream alone (no side lane, the caller's priority): the host waits for nothing, the device copying the tree's root,
        A's roots and the screen's sums to host memory behind an event, and the commitment returned is filled in, and
        `commits` updated, at `settle`, which the next pass's end, `finish` and a whole capture's start call first.  The next
        pass's buffers are rewritten behind those copies: on their stream in its order, on another after their event
        (`_after_late`)."""
        import torch
        self.settle()
        cur, K = self.cur, self.kernel
        out: dict[str, Any] = {"units": len(cur.matmuls), "tiles": cur.tiles, "hashing": self.hashing, "schedule": self.schedule.name}
        if self.hashing and cur.matmuls:
            self.join_sides()
            epoch = Epoch(Workload(tuple(cur.matmuls)), self.roots, self.salt)
            R, count, nb = K.run, cur.tiles, self.nb
            tree = _zeros(nb * (2 * R.pow2(count) - 1), self.device)
            dom = _zeros(nb, self.device)
            _put(dom, transcript_domain(epoch, count, self.tree_hash).domain_id)
            ns = types.SimpleNamespace(dev=K.dev, b={"tree_t": tree.data_ptr()}, keys={"dom_t": dom.data_ptr(), **self.const_keys})
            K.dev.stream = self.stream()
            try:
                for _, f in R.Pipeline.tree_setup(ns, "t", count, "dom_t"):
                    f()
                R.Pipeline.tree_leaves(ns, "t", self.leaves.data_ptr(), count, "dom_t")
                R.Pipeline.levels(ns, "t", count)
            finally:
                K.dev.stream = None
            if late and cur.retain is None and not self.schedule.side and self.schedule.main_priority != "greatest":
                self._late = self._copy_late(out, cur, tree, count)
                self.details["passes"].append(out)
                return out
            roots_a = _bytes(self.roots_buf[:nb * len(cur.matmuls)])
            flags = self.flag_buf[:cur.flag_rows].to(torch.int64).sum(0).tolist() if cur.flag_rows else [0, 0]
            self._close(out, cur, count, roots_a, _bytes(tree[-nb:]), flags)
            if cur.retain is not None:
                out["manifest"] = str(self._write_retained(cur, epoch, roots_a, tree, count))
        self.details["passes"].append(out)
        return out

    def _close(self, out: dict[str, Any], cur: _Pass, count: int, roots_a: bytes, root: bytes, flags: Any) -> None:
        """The pass's commitment, in `out` and `commits`: the tree's root, A's roots' digest and the screen's counts."""
        out.update(transcript_root=root.hex(), activation_roots=hashlib.sha512(roots_a).hexdigest(),
                   screened_rows=cur.flag_rows, failing_rows=int(flags[0]), unsure_rows=int(flags[1]),
                   excluded_rows=sum(len(r.excluded_a) for r in cur.records) if cur.retain is not None else None)
        self.commits.update(transcript_root=out["transcript_root"], units=len(cur.matmuls), tiles=count)

    def _copy_late(self, out: dict[str, Any], cur: _Pass, tree: Any, count: int) -> _Late:
        """`end`'s ``late`` copies, on the current stream behind the pass's tree, into host buffers made at the first (pinned
        on the GPU), which `settle` reads before the next late end rewrites them."""
        import torch
        n, nb, gpu = self.nb * len(cur.matmuls), self.nb, self.device.type == "cuda"
        if self._late_host is None:
            self._late_host = (torch.empty(self.roots_buf.numel(), dtype=torch.uint8, pin_memory=gpu),
                               torch.empty(nb, dtype=torch.uint8, pin_memory=gpu), torch.zeros(2, dtype=torch.int64, pin_memory=gpu))
        roots, root, sums = self._late_host
        roots[:n].copy_(self.roots_buf[:n], non_blocking=True)
        root.copy_(tree[-nb:], non_blocking=True)
        if cur.flag_rows:
            sums.copy_(self.flag_buf[:cur.flag_rows].to(torch.int64).sum(0), non_blocking=True)
        raw, event = self.stream(), None
        if gpu:
            event = self._late_event = self._late_event or torch.cuda.Event()
            event.record(self.torch_stream(raw))
        return _Late(out, cur, count, (roots[:n], root, sums), event, raw, tree)

    def settle(self) -> None:
        """A late end's commitment (`end`'s ``late``) filled in once the device has copied it, and `commits` updated; nothing
        when no end is late."""
        p, self._late = self._late, None
        if p is None:
            return
        if p.event is not None:
            p.event.synchronize()
        roots, root, sums = p.host
        self._close(p.out, p.cur, p.count, roots.numpy().tobytes(), root.numpy().tobytes(),
                    sums.tolist() if p.cur.flag_rows else [0, 0])

    def _after_late(self) -> None:
        """Before work that rewrites the pass's buffers (a call, a forward replayed whole, a poison): on another stream than
        a late end's copies, the current stream waits for them; on theirs it follows them, and a capture takes no wait."""
        p = self._late
        if p is None or p.event is None:
            return
        raw = self.stream()
        if raw != p.raw and not _capturing():
            self.torch_stream(raw).wait_event(p.event)

    def end_pass(self, retain: str | None = None, run: str | None = None, schedule: str | Schedule | None = None, *,
                 step: bool = False) -> dict[str, Any]:
        """`end` the open pass and `begin` the next (with `begin`'s arguments), so a pass is never left ended: a call after
        this lands in the next pass, and `finish` doesn't end this one twice.  Returns the ended pass's commitment.  `step`:
        an engine step's forward has returned (the PoUW option's hook, `pouw.install`).  Without it, a pass a step began that
        has no call yet is that step's, already ended: nothing ends, the step's commitment is returned, and arguments apply."""
        if not step and self.cur.stepped and not self.cur.matmuls:
            if retain or run or schedule:
                self.begin(retain, run, schedule, stepped=True)
            self.settle()
            return self.details["passes"][-1]
        out = self.end()
        self.begin(retain, run, schedule, stepped=step)
        return out

    def _write_retained(self, cur: _Pass, epoch: Epoch, roots_a: bytes, tree: Any, count: int) -> Path:
        import torch
        d = cur.retain
        wd = d / "weights"
        wd.mkdir(parents=True, exist_ok=True)
        weights = {}
        for w in sorted(self.weights):
            wt = self.weights[w]
            size = self.kernel.run.Pipeline.sizes(self.kernel.run.BM, wt.n, wt.k, self.kernel.run.BM)["tree_b"]
            weights[str(w)] = {"n": wt.n, "k": wt.k, "root": wt.root.hex(), "excluded_b": list(wt.excluded),
                               "rows": _write(wd / f"{w}.rows.bf16", _bytes(wt.tensor.to(torch.bfloat16).contiguous().view(torch.uint8)),
                                              digest=False),
                               "tree": _write(wd / f"{w}.tree", _bytes(wt.b["tree_b"][:size]))}
        matmuls = [{"index": u.index, "m": u.shape.m, "k": u.shape.k, "n": u.shape.n, "weight": u.weight,
                    "real_rows": r.real_rows, "k_kept": r.k_kept, "excluded_a": list(r.excluded_a),
                    "root_a": roots_a[self.nb * u.index:self.nb * (u.index + 1)].hex(), **f}
                   for u, r, f in zip(cur.matmuls, cur.records, cur.files)]
        head = {"schema": MANIFEST_SCHEMA, "run": cur.run, "entry": self.entry, "scheme": self.scheme.name,
                "scheme_args": {k: getattr(SchemeId.parse(self.scheme.name), k) for k in ("forming", "device", "hashing")},
                "coins": self.coins.to_json(), "salt_opening": self.opening.to_json(), "salt": self.salt.hex(),
                "cubin_sha256": self.kernel.cubin_sha256,
                "kernel_source_sha256": self.kernel.source_sha256, "schedule": self.details["schedule"],
                "records_root": PW.records_root(self.salt, epoch.weights_root,
                                                {w: PW.Split(self.weights[w].k) for w in epoch.workload.weights()}, cur.records).hex(),
                "commitment": {"root": _bytes(tree[-self.nb:]).hex(), "count": count},
                "leaves": _write(d / "leaves", _bytes(self.leaves[:self.nb * count])),
                "tree_t": _write(d / "tree_t", _bytes(tree)), "hm96_salt_key": self.salt_key.hex()}
        return write_manifest(d, head, weights, matmuls)

    def finish(self) -> None:
        self.settle()
        if self.cur is not None and self.cur.matmuls:
            self.end()
            self.cur = _Pass(None)
