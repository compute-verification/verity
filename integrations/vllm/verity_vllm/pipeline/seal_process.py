"""serve-openai's seals of its execution blocks in a spawned process of their own (`SealProcess`, ``--seal-process``), so that
no seal's Python holds the serving process's GIL while the engine loop runs. `sealer_of` is where serve-openai's seals run
(`SealingOptions`): on the engine loop, on a `serve_seal.Sealer`'s thread, or by default through a `ProcessSealer`'s process.

An execution block's seal is `serve_seal.seal_view` either way, its files and its result the ones `serve_seal.seal_window`
writes in the serving process; here it runs in two parts. The serving process writes the committed store's step files (its
retained values, tree levels and salt keys) and the weights' files from the pinned memory blocks they sit in into the bundle's
partial dir (`replay_bundle.write_store_files`): the values reach the seal's process as those files, written once, and no
memory block is copied for it. It hands the process the rest of what the seal reads, a message at a time
(`SealProcess.handoff`): per committed step its plan's metas as columns (`PlanMetas.columns`) and the bytes of the port's
committed positions, and once an execution block the committer's retained ranges, its rows, the step roots and the seal's
arguments; the required manifest once a run. While the serving process writes, the seal's process builds the binding map and
runs its checks in `seal_window`'s order; once the files are written, and the memory blocks given back to the pool
(``written``), it writes the store's ``store.json`` (`store_dump.write_doc`), the context, the manifest, ``runs.jsonl`` and
``verdict.json``. A check that fails before the store is written discards the partial dir, which `seal_window` would not
have written.
"""
from __future__ import annotations

import dataclasses
import os
import time
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from verity_vllm.config import option
from verity_vllm.pipeline import serve_seal as SS
from verity_vllm.pipeline import window_records as WR

#: the keys of an execution block's line in windows.jsonl that its seal reads
REC_KEYS = ("window", "run_id", "steps", "passed", "served_rows")


@dataclasses.dataclass(frozen=True, kw_only=True)
class SealingOptions:
    """serve-openai's ``sealing`` group: where each execution block's seal runs (`sealer_of`)."""
    seal_off_loop: bool = option("--seal-off-loop", action="boolean_optional", default=True,
                                 help="finalize and seal each window on a thread of its own while the engine serves the next one "
                                      "(serve_seal.Sealer; default on); --no-seal-off-loop does both on the engine loop, which "
                                      "stops serving for them")
    seal_process: bool = option("--seal-process", action="boolean_optional", default=True,
                                help="with --seal-off-loop: seal each execution block in a spawned process (seal_process.SealProcess; "
                                     "default on), the sealer's second thread writing only its store's files, and run each window's "
                                     "records (its readback, served rows, step trace and requests' checks) in another "
                                     "(window_records.RecordProcess), so no seal's or record's Python takes the engine's GIL; "
                                     "--no-seal-process runs both on the sealer's threads")


def sealer_of(opts: Any) -> SS.Sealer | None:
    """serve-openai's sealer under ``--seal``: None (the seals run on the engine loop) with ``--no-seal-off-loop``, else a
    `serve_seal.Sealer` holding ``--seal-hold-gb``, which seals through a `SealProcess` (`ProcessSealer`) unless ``--no-seal-process``."""
    if not (opts.seal and opts.sealing.seal_off_loop):
        return None
    return (ProcessSealer if opts.sealing.seal_process else SS.Sealer)(int(opts.seal_hold_gb * SS.GB))


class ProcessSealer(SS.Sealer):
    """A `seal_stages.Sealer` whose second thread seals each execution block through ``process``, a `SealProcess`, and whose
    first runs each window's records (`serve_commit.CommitWindows.recorder`) through ``records``, a `window_records.RecordProcess`:
    both spawned before its threads start and closed after they end. Its summary names the processes, and once closed how
    they ended."""

    def __init__(self, hold_bytes: int, clock: Callable[[], float] = time.perf_counter) -> None:
        self.process, self.records, self.ended = SealProcess(), WR.RecordProcess(), None
        super().__init__(hold_bytes, clock)

    def close(self) -> None:
        super().close()
        self.ended = {"process": self.process.close(), "records": self.records.close()}

    def summary(self) -> dict[str, Any]:
        return {**super().summary(), **(self.ended or {"process": {"pid": self.process.pid}, "records": {"pid": self.records.pid}})}


class SealFailed(RuntimeError):
    """What the seal's process raised for an execution block, as ``<type>: <message>``."""


class _Unstored(Exception):
    pass


class SealProcess(WR.Spawned):
    """One spawned process that seals execution blocks one at a time (`seal`), called from one thread (`seal_stages.Sealer`'s second). A process
    that has exited, killed or failed, raises at the next seal by name, and every seal after it fails the same way."""

    def __init__(self) -> None:
        self._manifests: dict[int, Any] = {}             # each manifest sent, by id (held, so no other object takes its id)
        super().__init__(_main, "serve-seal")

    def seal(self, com: Any, rc: Any, d: str | os.PathLike, *, extras: Any = None, written: Callable[[], Any] | None = None,
             **kw: Any) -> dict[str, Any]:
        """`serve_seal.seal_window(com, rc, d, extras=extras, **kw)`, its store's files written here and the rest in the process;
        ``written`` is called once the files are, when nothing here reads ``com``'s memory blocks again. Its result also says how
        long this process took to hand the execution block over and to write the files (``engine_s``)."""
        from verity_vllm.pipeline import replay_bundle as RB

        t0 = time.perf_counter()
        out = Path(d) / "commit"
        for msg in self.handoff(com, rc, d, kw, t0):
            self.conn.send(msg)
        t_sent = time.perf_counter()
        hashed: dict[str, str] = {}
        try:
            _p, doc = RB.write_store_files(out, 0, com, rc, digests=hashed, extras=extras)
        except Exception as e:
            self.conn.send(("unstored",))
            if (got := self._receive("seal"))[0] == "error":     # a check the process ran first: the execution block's error
                RB.discard(out, 0)
                raise SealFailed(got[1]) from e
            raise
        t_written = time.perf_counter()
        self.conn.send(("stored", doc, hashed))
        try:
            if written is not None:
                written()
        finally:
            got = self._receive("seal")
        if got[0] == "unstored":
            raise RuntimeError("the seal process found no store written")
        if got[0] == "error":
            if got[2]:
                RB.discard(out, 0)
            raise SealFailed(got[1])
        return {**got[1], "engine_s": {"handoff": round(t_sent - t0, 3), "store_files": round(t_written - t_sent, 3)}}

    def handoff(self, com: Any, rc: Any, d: str | os.PathLike, kw: dict[str, Any], t0: float) -> list[tuple]:
        """The messages that hand the process what the seal reads of ``com`` and ``rc`` beside the store's files: the manifest the
        first time, the execution block's head, a message per committed step, then ``go``."""
        from verity_vllm.commit import store_dump as SD
        from verity_vllm.commit.committer.plan_cache import PlanMetas
        from verity_vllm.commit.served_record import PORT_NAMES, StepInputPort, committed_raw, served_rows
        from verity_vllm.query.manifest.bind import source_ports

        manifest, sources = getattr(com, "required_manifest", None), getattr(com, "sources", None) or ()
        msgs: list[tuple] = []
        if manifest is not None and id(manifest) not in self._manifests:
            self._manifests[id(manifest)] = manifest
            msgs.append(("manifest", id(manifest), manifest))
        port, fams = SS.port_of(com), sorted(getattr(com, "retain_exclude", None) or ())
        view = {"program_digest": bytes(com.program_digest), "chunk": com.chunk, "name": com.name,
                "scheme": getattr(com, "commit_scheme", SS.V1), "retain": getattr(com, "retain", None),
                "manifest": None if manifest is None else id(manifest), "manifest_path": getattr(com, "required_manifest_path", None),
                "port": port, "ports": source_ports(sources) if manifest is not None else {},
                "has_port": any(isinstance(s, StepInputPort) for s in sources), "rows": served_rows(com)}
        rec = kw["rec"]
        msgs.append(("seal", {"t0": t0, "d": str(d), "view": view, "fams": fams, "held": SD.held_of(com) if fams else None,
                              "rc": {"run_root": rc.run_root, "steps": list(rc.steps), "weight_registration": rc.weight_registration},
                              "kw": {**kw, "rec": {k: rec[k] for k in REC_KEYS if k in rec}}}))
        for s in sorted(com._layouts):
            metas = com._layouts[s]
            lay = ("plan", metas.columns()) if isinstance(metas, PlanMetas) else (
                "metas", [dataclasses.replace(m, host=None, dev_block=None) for m in metas])
            msgs.append(("layout", s, lay, committed_raw(com, s, PORT_NAMES[1]) if port is not None else None))
        msgs.append(("go",))
        return msgs


def _main(conn: Any) -> None:
    from verity_vllm.commit.committer.plan_cache import PlanMetas
    from verity_vllm.commit.served_record import ints_of

    manifests: dict[int, Any] = {}
    conn.send(("ready", os.getpid()))
    while True:
        try:
            msg = conn.recv()
        except EOFError:
            return
        if msg[0] == "close":
            return
        if msg[0] == "manifest":
            manifests[msg[1]] = msg[2]
            continue
        head, layouts, positions = msg[1], {}, {}
        while (m := conn.recv())[0] == "layout":
            _kind, s, (form, metas), pos = m
            layouts[s] = PlanMetas.of_columns(metas) if form == "plan" else metas
            positions[s] = None if pos is None else ints_of(*pos)
        conn.send(_seal(conn, head, layouts, positions, manifests))


def _seal(conn: Any, head: dict[str, Any], layouts: dict[int, Any], positions: dict[int, Any], manifests: dict[int, Any]) -> tuple:
    """One execution block's seal in the process, its store's files written by the serving process: ``("sealed", result)``, else
    ``("error", what was raised, whether the store was still to be written)``, or ``("unstored",)`` when the serving process failed
    to write it and no check failed first."""
    from verity_vllm.commit import store_dump as SD
    from verity_vllm.pipeline import replay_bundle as RB

    v, d, fams, held = head["view"], Path(head["d"]), set(head["fams"]), head["held"]
    view = SS.SealView(**{**v, "manifest": manifests.get(v["manifest"])}, layouts=layouts, positions=lambda: positions,
                       unretained=lambda: SD.not_retained_of(fams, int(v["chunk"]), held, layouts) if fams else None)
    stored: list[tuple] = []

    def store(hashed: dict[str, str]) -> None:
        stored.append(conn.recv())
        if stored[0][0] != "stored":
            raise _Unstored
        hashed.update(stored[0][2])
        SD.write_doc(RB.partial(d / "commit", 0) / "store", stored[0][1], layouts)

    try:
        return ("sealed", SS.seal_view(view, SimpleNamespace(**head["rc"]), d, store=store, t0=head["t0"], **head["kw"]))
    except _Unstored:
        return ("unstored",)
    except Exception as e:  # noqa: BLE001  (the execution block's failure, sent to the serving process, which seals the next one)
        if not stored:
            conn.recv()
        return ("error", f"{type(e).__name__}: {e}", not stored)
