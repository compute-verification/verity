"""The Commit's PoUB session on a served engine (`CommitSession`, an adapter's ``--remote-verifier HOST:PORT``),
registered as the process `PROCESS` (`poub.scheme.POUB.processes`): PoUB's prover's serving loop (`node_store.serving`:
`StoreBackend`, `prove`, `Lockstep`, `KvDevice`) attached to the engine through the adapter's exchange (`POUB.attach`:
the store binder, `binder`), the session in lockstep with the Commit's instrumented run.

It imports no adapter. The adapter looks the process up by name, hands it the exchange of the build it is about to make
(``exchange``, called at `install`), and drives the object it gets back: `before_build` (or `install`), `check`, `begin`,
each engine step through the `Lockstep` that `begin` returns, `end`. What the session knows of the engine is what the
build declared on the exchange (`verity.hooks.Declared`) and the events the binder reads.
"""
from __future__ import annotations

import json
import socket
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from verity.protocols.resource_accounting.memory.poub.prover.node_store import (
    memory_map as store_map,
)
from verity.protocols.resource_accounting.memory.poub.prover.node_store import serving
from verity.protocols.resource_accounting.memory.poub.prover.node_store.registry import (
    NodeMemory,
)
from verity.protocols.resource_accounting.memory.poub.prover.node_store.serving import (
    KvDevice,
    Lockstep,
    StoreBackend,
)
from verity.protocols.resource_accounting.memory.poub.prover.binder import Binder
from verity.protocols.resource_accounting.memory.poub.scheme import GOAL, POUB, SESSION_PROCESS
from verity.protocols.resource_accounting.memory.poub.verifier import audit
from verity.protocols.resource_accounting.memory.poub.verifier import codec as rcodec

from verity import hooks as H

__all__ = ["PROCESS", "RECORD_SCHEMA", "CommitSession"]

#: the name the session is registered and looked up under
PROCESS = SESSION_PROCESS
#: the id of the session's record on disk (``remote_session.json``)
RECORD_SCHEMA = "verity-vllm/remote-session/v0"


class CommitSession:
    """The Commit's PoUB session (``--remote-verifier HOST:PORT``, `pipeline.commit`), on its engine:

    ~~~text
    target       before the build: the row's target declares synchronous scheduling (its Program's LAG 0)
    install      before the build: `POUB` attached on the build's exchange (``Exchange.attach``, so the build declares it):
                 the session's requests (`requests`: one GPU, synchronous steps, no
                 prefix caching, the KV pool at ``kv_pages`` ranges, None: the engine's own, at the row's
                 gpu_memory_utilization), which the build answers or refuses, and
                 ``binder.Binder(store, passes=False, kv=False)``, subscribed to the
                 workload's memory and steps events, registering weight tensors up to
                 ``weight_blocks`` blocks.  The KV pool stays outside
                 the registration (served writes do, and live KV is uncredited for chat),
                 so no step writes a registered block and every pass reads every one
    check        after the build, from what it declared (`verity.hooks.Declared`): the steps taken in this process, one
                 GPU, scheduling synchronous, no prefix caching, and no KV pool registered; then
                 ``filler_gib`` GiB of zeros on the engine's GPU registered as the store's filler (`Binder.register_filler`)
    begin        before the Commit's first instrumented run: the session on its own thread, registered before the run
                 goes on, and its first window armed once the run's first step has run (`Serving.ready`).  Its
                 backend is a host mirror encoded by kv.cu at ``kv_lib`` (`KvDevice`) or on the CPU
    (the run)    each of its engine steps through the returned `Lockstep`: a pass's step while the session lasts, so the
                 memory map, recorded against the registration's digest beside the Commit (``out``), is of the run's
                 steps from its first (a header and no line: no step writes the store); the steps after it unaudited
    end          after the run: the session done (failed if it still asked for a step), its record
                 ``remote_session.json`` with the map's agreement with the store (`store_map.agreement`), taken before the run
                 went on, each pass's ``[window, pass, step]`` as the verifier's record has its steps
                 (``pass_steps``, each a step of the map), and the map stopped there (`store.memory_map.end`)
    ~~~

    A store serves one session, so ``plan`` is one mode, ``serve`` (``servedrop:K`` withholds blocks appended, and
    this store appends none), and the session audits one run: with more than one pair, the later instrumented runs are
    committed and unaudited.  The control arm is never audited.  A verifier that paces the session (``--paced 1:…``: a
    live coin a pass) has each pass's engine step run inside its coin's window, so ``step_ms``, the step's time, goes
    into the calibration's pass times (`StoreBackend.with_step`) and the allowance covers it."""

    def __init__(self, verifier: str, *, exchange: Callable[[], Any] | None = None, plan: str = "serve",
                 kv_pages: int | None = 32,
                 weight_blocks: int | None = None, kv_lib: str | None = None,
                 out: str | Path | None = None, node: str | None = None, codec: rcodec.Codec = rcodec.DEFAULT,
                 step_ms: float = 0.0, filler_gib: float = 0.0, log: Callable[[str], None] = print) -> None:
        serving.address(verifier)
        if not step_ms >= 0 or not filler_gib >= 0:
            raise ValueError("the engine step a paced pass waits on, and the filler, are nonnegative")
        node = f"verity-vllm/{socket.gethostname()}" if node is None else node
        if not 1 <= len(node.encode()) <= audit.NODE_BYTES:
            raise ValueError(f"the session's node id {node!r} is not 1 to {audit.NODE_BYTES} bytes")
        if "," in plan:
            raise ValueError("a store serves one session: --remote-plan is one mode")
        if serving.session_module("prover").parse_plan(plan)[0][0] != "serve":
            raise ValueError(f"the Commit's session is serve, not {plan!r}: its store (weights and filler) appends nothing")
        if kv_pages is not None and kv_pages < 1:
            raise ValueError("the session's KV pool is at least a page (None: the engine's own)")
        self.verifier, self.plan, self.kv_pages, self.weight_blocks = verifier, plan, kv_pages, weight_blocks
        self.kv_lib, self.step_ms, self.filler_gib = kv_lib, step_ms, filler_gib
        self.out, self.node, self.codec, self.log = None if out is None else Path(out), node, codec, log
        self.exchange = exchange
        self.declared: H.Declared | None = None
        self.store = NodeMemory()
        self.binder: Binder | None = None
        self.lockstep: Lockstep | None = None
        self.backend: StoreBackend | None = None
        self.record: dict[str, Any] | None = None
        self._thread: threading.Thread | None = None

    def requests(self) -> list[H.Request]:
        """What `POUB` asks of the run under the session (`poub.scheme.POUB.requests` at `ctx`), which `install`'s
        attach files under the scheme's name."""
        return list(POUB.requests(self.ctx()))

    def ctx(self) -> dict[str, Any]:
        """What the session hands `POUB.attach` beside its settings and the build's key: its store, that it is a
        session in lockstep, its pool's ranges, and the binder's keywords."""
        return {"store": self.store, "session": True, "ranges": self.kv_pages,
                "binder": {"passes": False, "weight_blocks": self.weight_blocks, "kv": False}}

    def target(self, target: Any) -> None:
        """Before the build: the row's target declares synchronous scheduling, so its Program is the one the engine
        runs (a target that leaves it unset is specialized for the scheduler vLLM resolves, async, LAG 1)."""
        got = getattr(target, "async_scheduling", None)
        if got is not False:
            raise SystemExit(f"--remote-verifier: the row's target declares async_scheduling {got!r}, and the session's "
                             "engine schedules synchronously: declare \"async_scheduling\": false in the workload's target")

    def install(self) -> None:
        """`POUB` attached on the exchange of the build about to be made: its binder over the session's store, and
        the session's requests, filed under the scheme's name."""
        if self.exchange is None:
            raise NotImplementedError(f"{GOAL}: the process {PROCESS!r} is handed the exchange of the build it attaches "
                                      "to (exchange=, called here); this adapter passed none")
        ex = self.exchange()
        self.binder = ex.attach(POUB, None, **self.ctx())
        ex.subscribe(POUB.name, "declared", self._on_declared)

    def _on_declared(self, declared: H.Declared) -> None:
        self.declared = declared

    def gate(self) -> H.Gate:
        """The session's pacing as a gate asked before each step (`verity.hooks.Gate.ready`), in place of the
        `Lockstep` the run loop is handed: not built. It waits on the Commit's run loop asking the exchange's gates
        before each engine step, and on an event for an engine step that raised."""
        raise NotImplementedError(f"{GOAL}: the session paces through the Lockstep `begin` returns; a gate waits on "
                                  "the run loop asking Exchange.ready(StepDue) before each step and on an event for "
                                  "a step that raised")

    def before_build(self, ekw: dict[str, Any]) -> None:
        """`target` and `install` before the Commit's engine is built (``target``)."""
        self.target(ekw["target"])
        self.install()

    def check(self, llm: Any, runner: Any) -> dict[str, Any]:
        """After the build: the engine is one the session serves, by what the build declared and by ``runner``, its
        model runner in this process (None where its core runs in another, which a build may also declare as
        ``in_process`` false); then the filler, on ``runner.device``. ``llm`` is not read."""
        if self.declared is None:
            raise RuntimeError("the Commit's session checks its engine once the build has declared it on the exchange "
                               "the session is installed on (`install`, then the build)")
        p, why = self.declared.properties, []
        if runner is None or p.get("in_process") is False:
            why.append("the engine core in another process")
        if p.get("async_steps") is not False:
            why.append("async scheduling not declared off")
        if p.get("prefix_cache"):
            why.append("prefix caching on")
        if p.get("world") != 1:
            why.append(f"tensor parallel size {p.get('world')}")
        if "kv" in {r.kind for r in self.store.layout().regions}:
            why.append("a KV pool registered (served writes stay outside the registration)")
        if why:
            raise SystemExit("--remote-verifier: the engine is not one the session serves: " + "; ".join(why))
        binder = self.binder
        if binder is None:
            raise RuntimeError("the Commit's session checks its engine once its binder is installed (`install`)")
        if self.filler_gib:
            binder.register_filler(int(self.filler_gib * (1 << 30)), runner.device)
        stats = binder.stats()
        self.log(f"[delta] remote session: the store binder on the engine, {stats}")
        return stats

    def begin(self) -> Lockstep:
        if self.lockstep is not None:
            raise RuntimeError("a store serves one session: the Commit's session audits one run")
        self.lockstep = lockstep = Lockstep()
        path = None if self.out is None else self.out / "memory_map.jsonl"
        device = None if self.kv_lib is None else KvDevice(self.kv_lib, self.codec)
        self.backend = backend = StoreBackend(self.store, lockstep, path=path, device=device)
        backend.step_ns = round(self.step_ms * 1e6)
        self._thread = threading.Thread(target=self._session, args=(backend, lockstep), name="remote-session",
                                        daemon=True)
        self._thread.start()
        lockstep.begun()
        return lockstep

    def _session(self, backend: StoreBackend, lockstep: Lockstep) -> None:
        error, result, agree = None, {}, {}
        try:
            (result,) = serving.prove(backend, self.verifier, self.plan, node=self.node.encode(), codec=self.codec, log=self.log)
            agree = store_map.agreement(backend)
        except Exception as e:  # noqa: BLE001  -- recorded: a session that fails is rejected, and the Commit's verdict still written
            error = f"{type(e).__name__}: {e}"
        try:
            trace = backend.trace
            steps = result.get("pass_steps")
            if steps is None:
                steps = []
            self.record = {
                "schema": RECORD_SCHEMA, "verifier": self.verifier, "plan": self.plan,
                "accepted": error is None and bool(result.get("accepted")), "error": error, "prover": result,
                "store": self.binder.stats() if self.binder is not None else None, "layout": backend.layout(),
                "kv_pages": self.kv_pages, "weight_blocks": self.weight_blocks, "step_ms": self.step_ms,
                "filler_gib": self.filler_gib,
                "encode": "kv.cu" if backend.device is not None else "cpu", "gates": backend.gates,
                "memory_map": None if trace is None else {
                    "path": None if backend.path is None else str(backend.path),
                    "registration": backend.registration.digest.hex(), "first_step": trace.first, "steps": trace.steps,
                    "rekeyed": trace.rekeyed, "deferral": trace.deferral(), "pass_steps": steps,
                    "pass_steps_mapped": trace.first is not None and all(
                        trace.first <= x <= trace.step for *_, x in steps),
                    **agree}}
            store_map.end(self.store)
        finally:
            lockstep.close()

    def end(self) -> dict[str, Any]:
        lock, thread = self.lockstep, self._thread
        if lock is None or thread is None:
            raise RuntimeError("the Commit's session ends once it has begun (`begin`)")
        lock.finish()
        thread.join()
        rec: dict[str, Any] = self.record if self.record is not None else {
            "schema": RECORD_SCHEMA, "verifier": self.verifier, "plan": self.plan, "accepted": False,
            "error": "the session's record was not written (its thread raised)", "memory_map": None, "encode": None}
        rec["run"] = {"steps": lock.audited + lock.unaudited, "audited": lock.audited, "unaudited": lock.unaudited}
        self.record = rec
        if self.out is not None:
            (self.out / "remote_session.json").write_text(json.dumps(rec, indent=1, default=str) + "\n")
        return rec

