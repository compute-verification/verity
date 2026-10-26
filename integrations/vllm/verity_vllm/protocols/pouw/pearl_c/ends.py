"""The Pearl-C device path's passes (`pouw_pearl_c_device.PearlCDeviceRun`'s): a pass's state (`_Pass`) and its end
(`PassEnds`): one tile tree over every leaf under `audit.transcript_domain`, A's roots and the screen's counts as the pass's
commitment, and a retained pass's files and manifest (`write_manifest`).  A lagged end's readback and tree aside are
`pouw_pearl_c_lanes.LaneCalls`' (`settle`, `_end_aside`, `_ended`).
"""
from __future__ import annotations

import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from verity.core.protocols.pouw.audit import Epoch, Workload, transcript_domain
from verity.core.protocols.pouw.protocol import Matmul
from verity.core.protocols.pouw.schemes import pearl_c_work as PW
from verity.core.protocols.pouw.serving import Schedule, SchemeId

from verity_vllm.protocols.pouw.pearl_c.host import _bytes, _linked, _put, _write, _zeros, write_manifest
from verity_vllm.protocols.pouw.pearl_c.manifest import MANIFEST_SCHEMA, session_head


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
    base: int | None = None          # its call at position p has index base + p; None until its first call (`call_index`)


class PassEnds:
    """`PearlCDeviceRun`'s pass ends, as its base class: the run's ``cur``, ``hashing``, ``kernel``, ``nb``, ``roots``,
    ``salt``, ``schedule``, ``leaves``, ``roots_buf``, ``flag_buf``, ``const_keys``, ``tree_hash``, ``details``,
    ``commits``, ``weights``, ``entry``, ``scheme``, ``coins``, ``opening``, ``salt_key``, ``device``, ``aside_ends``,
    ``lag_ends``, ``_lagged`` and ``_weight_files``, and its ``stream``, ``join_sides``, ``begin``, ``settle``,
    ``_end_aside`` and ``_ended``."""

    def end(self, lag: bool = False) -> dict[str, Any]:
        """Close the pass: its tile tree on the device, every A root, and with `retain` the manifest.  Returns the pass's
        commitment (also in `commits` and `details["passes"]`).  With `lag` (on CUDA, without `retain`) the root, A's roots
        and the flag counts are copied out behind the pass's work and the host doesn't wait for them: the returned dict gets
        them, and `commits` its root, when the next end, `settle` or `finish` completes the readback."""
        import torch
        self.settle()
        cur, K = self.cur, self.kernel
        out: dict[str, Any] = {"units": len(cur.matmuls), "tiles": cur.tiles, "hashing": self.hashing,
                               "schedule": self.schedule.name} | ({"base": cur.base} if cur.matmuls else {})
        if self.hashing and cur.matmuls:
            self.join_sides()
            epoch = Epoch(Workload(tuple(cur.matmuls)), self.roots, self.salt)
            R, count, nb = K.run, cur.tiles, self.nb
            tree = _zeros(nb * (2 * R.pow2(count) - 1), self.device)
            dom = _zeros(nb, self.device)
            _put(dom, transcript_domain(epoch, count, self.tree_hash).domain_id)
            ns = types.SimpleNamespace(dev=K.dev, b={"tree_t": tree.data_ptr()}, keys={"dom_t": dom.data_ptr(), **self.const_keys})
            units = len(cur.matmuls)
            flags = self.flag_buf[:cur.flag_rows].to(torch.int64).sum(0) if cur.flag_rows else torch.zeros(2, dtype=torch.int64)
            aside = lag and self.aside_ends and cur.retain is None and self.device.type == "cuda"
            leaves, side, got = self._end_aside(count, self.roots_buf[:nb * units], flags) if aside else \
                (self.leaves.data_ptr(), None, None)
            K.dev.stream = side.cuda_stream if aside else self.stream()
            try:
                for _, f in R.Pipeline.tree_setup(ns, "t", count, "dom_t"):
                    f()
                R.Pipeline.tree_leaves(ns, "t", leaves, count, "dom_t")
                R.Pipeline.levels(ns, "t", count)
            finally:
                K.dev.stream = None
            out.update(screened_rows=cur.flag_rows,
                       excluded_rows=sum(len(r.excluded_a) for r in cur.records) if cur.retain is not None else None)
            if aside:
                with torch.cuda.stream(side):
                    root, done = torch.empty(nb, dtype=torch.uint8, pin_memory=True).copy_(tree[-nb:], non_blocking=True), torch.cuda.Event()
                    done.record()
                tree.record_stream(side)
                dom.record_stream(side)
                self._lagged = (out, done, [got[0], root, got[1]], units, count)
            elif lag and cur.retain is None and self.device.type == "cuda":
                got = [torch.empty(t.shape, dtype=t.dtype, pin_memory=True).copy_(t, non_blocking=True)
                       for t in (self.roots_buf[:nb * units], tree[-nb:], flags)]
                done = torch.cuda.Event()
                done.record()
                self._lagged = (out, done, got, units, count)
            else:
                roots_a = _bytes(self.roots_buf[:nb * units])
                self._ended(out, roots_a, _bytes(tree[-nb:]), flags.tolist(), units, count)
                if cur.retain is not None:
                    out["manifest"] = str(self._write_retained(cur, epoch, roots_a, tree, count))
        self.details["passes"].append(out)
        return out

    def end_pass(self, retain: str | None = None, run: str | None = None, schedule: str | Schedule | None = None, *,
                 step: bool = False) -> dict[str, Any]:
        """`end` the open pass and `begin` the next (with `begin`'s arguments), so a pass is never left ended: a call after
        this lands in the next pass, and `finish` doesn't end this one twice.  Returns the ended pass's commitment.  `step`:
        an engine step's forward has returned (the PoUW option's hook, `pouw.install`).  Without it, a pass a step began that
        has no call yet is that step's, already ended: nothing ends, the step's commitment is returned, and arguments apply."""
        if not step and self.cur.stepped and not self.cur.matmuls:
            if retain or run or schedule:
                self.begin(retain, run, schedule, stepped=True)
            return self.details["passes"][-1]
        out = self.end(lag=step and self.lag_ends)
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
                               "rows": _linked(wd / f"{w}.rows.bf16", self._weight_files, digest=False,
                                               data=lambda wt=wt: _bytes(wt.tensor.to(torch.bfloat16).contiguous().view(torch.uint8))),
                               "tree": _linked(wd / f"{w}.tree", self._weight_files, data=lambda wt=wt, size=size: _bytes(wt.b["tree_b"][:size]))}
        matmuls = [{"index": u.index, "m": u.shape.m, "k": u.shape.k, "n": u.shape.n, "weight": u.weight,
                    "real_rows": r.real_rows, "k_kept": r.k_kept, "excluded_a": list(r.excluded_a),
                    "root_a": roots_a[self.nb * p:self.nb * (p + 1)].hex(), **f}
                   for p, (u, r, f) in enumerate(zip(cur.matmuls, cur.records, cur.files))]
        head = {"schema": MANIFEST_SCHEMA, "run": cur.run, "entry": self.entry, "scheme": self.scheme.name,
                "scheme_args": {k: getattr(SchemeId.parse(self.scheme.name), k) for k in ("forming", "device", "hashing")},
                "coins": self.coins.to_json(), "salt_opening": self.opening.to_json(), "salt": self.salt.hex(),
                **session_head(self.commits), "cubin_sha256": self.kernel.cubin_sha256,
                "kernel_source_sha256": self.kernel.source_sha256, "schedule": self.details["schedule"],
                "records_root": PW.records_root(self.salt, epoch.weights_root,
                                                {w: PW.Split(self.weights[w].k) for w in epoch.workload.weights()}, cur.records).hex(),
                "commitment": {"root": _bytes(tree[-self.nb:]).hex(), "count": count}, "hm96_salt_key": self.salt_key.hex(),
                "leaves": _write(d / "leaves", _bytes(self.leaves[:self.nb * count])), "tree_t": _write(d / "tree_t", _bytes(tree))}
        return write_manifest(d, head, weights, matmuls)

    def finish(self) -> None:
        if self.cur is not None and self.cur.matmuls:
            self.end()
            self.cur = _Pass(None)
        self.settle()
