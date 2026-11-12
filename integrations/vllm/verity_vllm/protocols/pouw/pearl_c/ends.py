"""The Pearl-C device path's passes (`pouw_pearl_c_device.PearlCDeviceRun`'s): a pass's state (`_Pass`) and its end
(`PassEnds`): one tile tree over every leaf under `audit.transcript_domain`, A's roots and the screen's counts as the pass's
commitment, a retained call's record and files (`_retained`, written behind the call) and a retained pass's files and
manifest (`write_manifest`), completed behind a step's end (`retained_ends`).  A lagged end's readback and tree aside are
`pouw_pearl_c_lanes.LaneCalls`' (`settle`, `_end_aside`, `_ended`).  Where each pass's seconds went is the run's
``details["seconds"]`` (`parts.Parts`), beside its commitments and in none of them.
"""
from __future__ import annotations

import copy
import types
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from verity.core.protocols.pouw.audit import Epoch, Workload, transcript_domain
from verity.core.protocols.pouw.core_check import MANIFEST_SCHEMA
from verity.core.protocols.pouw.protocol import Matmul
from verity.ml.protocols.pouw.schemes import pearl_c_work as PW
from verity.core.protocols.pouw.serving import Schedule, SchemeId

from verity_vllm.protocols.pouw.pearl_c.host import (
    _bytes,
    _linked,
    _put,
    _zeros,
    joined,
    rows_name,
    write_behind,
    write_manifest,
)
from verity_vllm.protocols.pouw.pearl_c.host import behind as host_behind
from verity_vllm.protocols.pouw.pearl_c.manifest import session_head
from verity_vllm.protocols.pouw.pearl_c.parts import Parts
from verity_vllm.protocols.pouw.pearl_c.screen import excluded, excluded_host
from verity_vllm.protocols.pouw.pearl_c import served_y as SY


@dataclass
class _Pass:
    retain: Path | None
    run: str | None = None          # the run id a retained pass's manifest names
    matmuls: list[Matmul] = field(default_factory=list)
    flag_rows: int = 0               # rows of the pass's screen flags so far
    records: list[PW.UnitRecord] = field(default_factory=list)
    files: list[dict] = field(default_factory=list)
    writes: list[Any] = field(default_factory=list)   # its files' writes still behind its calls (`write_behind`)
    tiles: int = 0
    poison: int | None = None       # the byte every call buffer starts the pass as, those created during it included
    stepped: bool = False            # begun by an engine step's end (`PearlCDeviceRun.end_pass`'s ``step``)
    base: int | None = None          # its call at position p has index base + p; None until its first call (`call_index`)
    dirs: set[str] = field(default_factory=set)   # the directories under ``retain`` its retained calls have made
    parts: Parts = field(default_factory=Parts)   # where its seconds go on the thread that runs it, and behind it


class PassEnds:
    """`PearlCDeviceRun`'s pass ends, as its base class: the run's ``cur``, ``hashing``, ``kernel``, ``nb``, ``roots``,
    ``salt``, ``schedule``, ``leaves``, ``roots_buf``, ``flag_buf``, ``const_keys``, ``tree_hash``, ``details``,
    ``commits``, ``weights``, ``entry``, ``scheme``, ``coins``, ``opening``, ``salt_key``, ``device``, ``aside_ends``,
    ``lag_ends``, ``_lagged``, ``_weight_files``, ``_rows``, ``_rows_at``, ``shapes``, ``_graphs`` and ``_stages``, and its ``stream``,
    ``join_sides``, ``begin``, ``settle``, ``_end_aside`` and ``_ended``."""

    #: the run's y trees on its device (`served_y.DeviceTrees`), made at the first retained call on it
    _y_trees: Any = None
    #: whether a step's retained pass is completed behind it (`end`'s ``behind``)
    retain_behind: bool = True
    #: the retained passes being completed behind their steps, oldest first (`retained_ends`)
    _behind: tuple = ()
    #: the stream a device's retained calls are completed on behind the calls (`_retained`), made at the first
    _retain_side: Any = None

    def end(self, lag: bool = False, behind: bool = False) -> dict[str, Any]:
        """Close the pass: its tile tree on the device, every A root, and with `retain` the manifest.  Returns the pass's
        commitment (also in `commits` and `details["passes"]`).  With `lag` (on CUDA, without `retain`) the root, A's roots
        and the flag counts are copied out behind the pass's work and the host doesn't wait for them: the returned dict gets
        them, and `commits` its root, when the next end, `settle` or `finish` completes the readback.  With `behind` (and
        `retain_behind`) a retained pass's files and manifest are completed behind it, in order (`retained_ends`, which
        `finish` waits for): the dict gets ``excluded_rows`` and ``manifest`` once its manifest is written, after every
        file it names.  A pass with calls adds its seconds by part to ``details["seconds"]`` (`parts.Parts`)."""
        import torch
        cur, K, deferred = self.cur, self.kernel, False
        parts = cur.parts
        parts.switch("tree")
        self.settle()
        self.retained_ends(wait=False)
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
            out.update(screened_rows=cur.flag_rows, excluded_rows=None)
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
                parts.switch("readback")
                roots_a, root = _bytes(self.roots_buf[:nb * units]), _bytes(tree[-nb:])
                self._ended(out, roots_a, root, flags.tolist(), units, count)
                if cur.retain is not None:
                    parts.switch("files")
                    complete = self._write_retained(cur, epoch, roots_a, root, tree, count, out)
                    if behind and self.retain_behind:
                        self._behind = (*self._behind, host_behind(lambda: parts.behind(complete(Parts()))))
                        deferred = True
                    else:
                        self.retained_ends()
                        complete(parts)
        if not deferred:
            joined(cur.writes)
        parts.switch(None)
        if cur.matmuls:
            self.details["seconds"].append(parts.seconds)
        self.details["passes"].append(out)
        return out

    def begin_step(self) -> None:
        """An engine step's forward begins (`passes.step_ends`): the open pass's seconds run from here, as ``forward``."""
        self.cur.parts.switch("forward")

    def end_pass(self, retain: str | None = None, run: str | None = None, schedule: str | Schedule | None = None, *,
                 step: bool = False) -> dict[str, Any]:
        """`end` the open pass and `begin` the next (with `begin`'s arguments), so a pass is never left ended: a call after
        this lands in the next pass, and `finish` doesn't end this one twice.  Returns the ended pass's commitment.  `step`:
        an engine step's forward has returned (the PoUW option's hook, `pouw.install`).  Without it, a pass a step began that
        has no call yet is that step's, already ended: nothing ends, the step's commitment is returned, and arguments apply.
        The next pass's begin is the ended pass's part ``next``: a step's wall holds it."""
        if not step and self.cur.stepped and not self.cur.matmuls:
            if retain or run or schedule:
                self.begin(retain, run, schedule, stepped=True)
            return self.details["passes"][-1]
        parts = self.cur.parts
        out = self.end(lag=step and self.lag_ends, behind=step)
        parts.switch("next")
        self.begin(retain, run, schedule, stepped=step)
        parts.switch(None)
        return out

    def _retained(self, cur: _Pass, mm: Matmul, m: int, flags: Any, xa: Any, tree: Any, y: Any = None) -> None:
        """A retained call's record (the rows its screen excluded, from its flags) and files (its rows as BF16, A's tree,
        and given its real rows of `y`, y's rows and its tree, `served_y.y_tree`, made on the call's device: all written
        behind the call, `write_behind`, joined by the pass's end).  On a device the call's buffers are copied as they are on
        its stream, and the rest runs behind it on the run's retain stream (`_retain_stream`): y's tree, the copies to the
        host, and the record and y's root, decided from the bytes the writes land (`screen.excluded_host`).  Its seconds
        are the pass's part ``retain``, and y's tree's ``y_tree``, or ``y_first`` for a width's first on a device."""
        import torch
        was = cur.parts.switch("retain")
        d = cur.retain / "calls"
        for name in ("calls",) if y is None else ("calls", "y"):
            if name not in cur.dirs:
                cur.retain.mkdir(mode=0o700, parents=True, exist_ok=True)
                (cur.retain / name).mkdir(mode=0o700, exist_ok=True)
                cur.dirs.add(name)
        unit, ex, rec = (mm, m, mm.shape.k), self.weights[mm.weight].excluded, len(cur.records)
        files = {"rows": None, "tree": None} | ({} if y is None else {"y": {"root": None, "rows": None, "tree": None}})
        paths = (d / f"{mm.index}.rows.bf16", d / f"{mm.index}.tree",
                 cur.retain / "y" / f"{mm.index}.rows.bf16", cur.retain / "y" / f"{mm.index}.tree")

        def y_root(host: Any) -> None:
            files["y"]["root"] = host[-self.nb:].numpy().tobytes().hex()

        def y_tree(rows: Any) -> Any:
            first = rows.is_cuda and (self._y_trees is None or rows.shape[1] not in self._y_trees.pipes)
            cur.parts.switch("y_first" if first else "y_tree")
            made = SY.y_tree(self, rows, mm.index)
            cur.parts.switch("retain")
            return made

        if not xa.is_cuda:
            cur.records.append(PW.UnitRecord(*unit, excluded(flags[:, 0].bool(), flags[:, 1].bool(), xa[:m]), ex))
            out = [(paths[0], xa.to(torch.bfloat16).view(torch.uint8), False), (paths[1], tree, True)]
            if y is not None:
                out += [(paths[2], y, False), (paths[3], y_tree(y), True, y_root)]
            got = write_behind(tuple(out), cur.writes, cur.parts)
        else:
            side = self._retain_stream(xa.device)
            kept = [xa.to(torch.bfloat16, copy=True), tree.clone(), flags[:m].clone()] + ([] if y is None else [y.clone()])
            side.wait_stream(torch.cuda.current_stream(xa.device))
            cur.records.append(None)
            with torch.cuda.stream(side):
                for t in kept:
                    t.record_stream(side)
                rows, a_tree, held, *ys = kept

                def record(host: Any, flags: Any) -> None:
                    words = host.numpy().view("<u2").reshape(-1, mm.shape.k)
                    cur.records[rec] = PW.UnitRecord(*unit, excluded_host(flags.numpy().reshape(-1, 2), words), ex)
                out = [(paths[0], rows.view(torch.uint8), False, record, held), (paths[1], a_tree, True)]
                if ys:
                    out += [(paths[2], ys[0], False), (paths[3], y_tree(ys[0]), True, y_root)]
                got = write_behind(tuple(out), cur.writes, cur.parts, fresh=True, here=True)
        files.update(rows=got[0], tree=got[1])
        if y is not None:
            files["y"].update(rows=got[2], tree=got[3])
        cur.files.append(files)
        cur.parts.switch(was)

    def _poison_graphs(self, value: int) -> None:
        """Every buffer the graphs write at `value`: each graph's y and leaves, and each stage, skipping those that hold it
        from an earlier poison, which no call has taken since (a graph's call marks it, a forward replayed whole marks all)."""
        import torch
        for g in self._graphs.values():
            if g.poisoned != value:
                g.y.view(torch.uint8).fill_(value)
                g.leaves.fill_(value)
                g.poisoned = value
        if self._stages and self._stages_poisoned != value:
            for stage in self._stages.values():
                for t in stage:
                    t.view(torch.uint8).fill_(value)
            self._stages_poisoned = value

    def _unpoisoned(self) -> None:
        """Every shape, graph and stage marked written (a forward replayed whole writes them without a call's Python)."""
        for sh in self.shapes.values():
            sh.poisoned = None
        for g in self._graphs.values():
            g.poisoned = None
        self._stages_poisoned = None

    def _retain_stream(self, device: Any) -> Any:
        if self._retain_side is None:
            import torch
            self._retain_side = torch.cuda.Stream(device=device)
        return self._retain_side

    def _write_retained(self, cur: _Pass, epoch: Epoch, roots_a: bytes, root: bytes, tree: Any, count: int,
                        out: dict[str, Any]) -> Callable[[Parts], Parts]:
        """A retained pass's last files (the leaves and the tile tree, written behind the end) and the step that completes it
        once every file is: its records, its weights' files and its manifest, ``out``'s ``excluded_rows`` and ``manifest``.
        A weight's files are written from the device by the first retained pass, on this thread, and hard-linked after;
        its rows are linked by the first too where the run's store holds them (`_rows_kept`).  The completing step takes
        the `Parts` of the thread it runs on, and returns them."""
        import torch
        d, nb = cur.retain, self.nb
        wd = d / "weights"
        wd.mkdir(mode=0o700, parents=True, exist_ok=True)
        last = write_behind(((d / "leaves", self.leaves[:nb * count], True), (d / "tree_t", tree, True)), cur.writes,
                            cur.parts)
        sizes = {w: self.kernel.run.Pipeline.sizes(self.kernel.run.BM, wt.n, wt.k, self.kernel.run.BM)["tree_b"]
                 for w, wt in self.weights.items()}

        was = cur.parts.switch("weights")
        rows = {w: self._rows_kept(w) for w in self.weights}

        def weight_files(read: Callable[[int, str], Callable[[], bytes]]) -> dict[str, dict]:
            return {str(w): {"n": wt.n, "k": wt.k, "root": wt.root.hex(), "excluded_b": list(wt.excluded),
                             "rows": _linked(wd / f"{w}.rows.bf16", rows[w][0], digest=False, data=read(w, "rows.bf16"),
                                             name=rows[w][1]),
                             "tree": _linked(wd / f"{w}.tree", self._weight_files, data=read(w, "tree"))}
                    for w, wt in sorted(self.weights.items())}

        def from_device(w: int, part: str) -> Callable[[], bytes]:
            wt = self.weights[w]
            if part == "tree":
                return lambda: _bytes(wt.b["tree_b"][:sizes[w]])
            return lambda: _bytes(wt.tensor.to(torch.bfloat16).contiguous().view(torch.uint8))

        def from_first(w: int, part: str) -> Callable[[], bytes]:
            if part == "tree":
                return self._weight_files[f"{w}.tree"][0].read_bytes
            return rows[w][0][rows[w][1]][0].read_bytes

        kept = all(rows[w][1] in rows[w][0] and f"{w}.tree" in self._weight_files for w in self.weights)
        weights = None if kept else weight_files(from_device)
        cur.parts.switch(was)
        head = {"schema": MANIFEST_SCHEMA, "run": cur.run, "entry": self.entry, "scheme": self.scheme.name,
                "scheme_args": {k: getattr(SchemeId.parse(self.scheme.name), k) for k in ("forming", "device", "hashing")},
                "coins": self.coins.to_json(), "salt_opening": self.opening.to_json(), "salt": self.salt.hex(),
                **session_head(self.commits), "cubin_sha256": self.kernel.cubin_sha256,
                "kernel_source_sha256": self.kernel.source_sha256, "schedule": copy.deepcopy(self.details["schedule"]),
                "records_root": None, "commitment": {"root": root.hex(), "count": count}, "hm96_salt_key": self.salt_key.hex(),
                "leaves": last[0], "tree_t": last[1]}
        splits = {w: PW.Split(self.weights[w].k) for w in epoch.workload.weights()}

        def complete(own: Parts) -> Parts:
            own.switch("join")
            joined(cur.writes)
            own.switch("manifest")
            head["records_root"] = PW.records_root(self.salt, epoch.weights_root, splits, cur.records).hex()
            matmuls = [{"index": u.index, "m": u.shape.m, "k": u.shape.k, "n": u.shape.n, "weight": u.weight,
                        "real_rows": r.real_rows, "k_kept": r.k_kept, "excluded_a": list(r.excluded_a),
                        "root_a": roots_a[nb * p:nb * (p + 1)].hex(), **f}
                       for p, (u, r, f) in enumerate(zip(cur.matmuls, cur.records, cur.files))]
            path = write_manifest(d, head, weights or weight_files(from_first), matmuls)
            out.update(excluded_rows=sum(len(r.excluded_a) for r in cur.records), manifest=str(path))
            return own
        return complete

    def _rows_kept(self, w: int) -> tuple[dict[str, tuple[Path, dict]], str]:
        """Where weight `w`'s rows file is kept and under what name (`_linked`'s ``kept`` and ``name``): the store the run
        was given, under `rows_name` of the row digests this install's tree was made from, read back from the device
        once, where the store holds that name; else the run's own files, under the file's name, as its tree's is."""
        if w not in self._rows_at:
            wt, name = self.weights[w], None
            if self._rows:
                name = rows_name(w, wt.n, wt.k, _bytes(wt.b["rowdig_b"][:64 * wt.n]))
            self._rows_at[w] = (self._rows, name) if name in self._rows else (self._weight_files, f"{w}.rows.bf16")
        return self._rows_at[w]

    def retained_ends(self, wait: bool = True) -> None:
        """The retained passes ended behind their steps (`end`'s ``behind``) complete, in order, when `wait`; else those
        already done.  Each one's failure is raised here, the first first."""
        while self._behind and (wait or self._behind[0].done()):
            first, self._behind = self._behind[0], self._behind[1:]
            first.result()

    def finish(self) -> None:
        if self.cur is not None and self.cur.matmuls:
            self.end()
            self.cur = _Pass(None)
        self.settle()
        self.retained_ends()
