"""A served Commit window's circuit at its close, before its seal and any coin: its advice committed as `verity-vllm advice`
commits it (`advice.write`), the class circuit compiled from that advice and the window's bodies in the one compile call
(`window_circuit.served_lowering`), and the committer's units in order (`window_plan.Plan`): each template's units, each
port's source and the served rows each reads where they are committed."""
from __future__ import annotations

import json
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from verity_vllm.pipeline import advice as ADV

from . import window_circuit as WC
from .build import Unexpressible
from .close_record import Record
from .window_plan import Plan


class ClosePlan:
    """The plan of each window a serve under ``out`` closes, at class ``cls`` (None: the serve declared none, and each window
    says so). φ is ``serve.json``'s (`phi.phi_of_serve`, with ``hf_config`` where no manifest pins the model), read at the first
    window, else ``registered``; ``compiler`` the advice's (`advice.registered_compiler`'s default); ``flush`` drains the
    advice trace (`StepTrace.drain`) before it is read. Given the binding map the window's close built (`serve_seal.binding_of`:
    ``bmap`` and ``aliases``), each slot's `window_rows.Served` locates its request's committed members, from a Build of its
    Program written beside the window's seal (`naming.write_build`, under `steps.trace_dir`'s name); without it the outcome
    names the circuit's templates and units alone. A window whose advice is pending or refused, or that the class doesn't take,
    has no circuit and says why."""

    def __init__(self, out: Path, cls: Any, *, registered: Any = None, compiler: Any = None, grain: Any = None,
                 flush: Callable[[], Any] | None = None, registered_roots: dict[str, str] | None = None,
                 hf_config: str | None = None, record: Record | None = None) -> None:
        self.out, self.cls, self.registered, self.compiler, self.record = Path(out), cls, registered, compiler, record
        self.grain = grain if grain is not None else WC._GRAIN
        self.flush, self.registered_roots, self.hf_config = flush, registered_roots, hf_config
        self.held: dict[int, tuple[Any, Any, Plan | None]] = {}
        self.unregistered: str | None = None

    def _registered(self) -> Any:
        if self.registered is None:
            from .phi import phi_of_serve, registered_seed
            from .steps import Registered

            serve = json.loads((self.out / "serve.json").read_text())
            self.registered = Registered.of(phi_of_serve(serve, self.hf_config), registered_seed(serve), ADV.CAP_BITS)
        return self.registered

    def __call__(self, window: int, binding: Any = None) -> dict[str, Any]:
        t0 = time.perf_counter()
        if self.flush is not None:
            self.flush()
        if self.compiler is None:
            self.compiler = ADV.registered_compiler()
        if self.record is not None and self.registered_roots is None and self.unregistered is None:
            try:
                self.registered_roots = self.record.roots(self._registered(), self.out)
            except Exception as e:  # noqa: BLE001  (no record without the weights' registration; the plans go on)
                self.unregistered = f"weights registration: {type(e).__name__}: {e}"[:500]
        rep = ADV.write(self.out, self.out / "advice", window=window, compiler=self.compiler,
                        registered_roots=self.registered_roots)
        for key in ("pending", "refused"):
            if str(window) in rep[key]:
                return {"window": window, key: rep[key][str(window)]}
        got = {"window": window, "advice": rep["windows"][str(window)]["root"]}
        if self.cls is None:
            return {**got, "refused": "the serve declares no window class (--window-class S,P,O), so no circuit is compiled"}
        advice = (self.out / "advice" / f"w{window}" / "private" / "advice.bin").read_bytes()
        try:
            compiled, low = WC.served_lowering(self._registered(), self.cls, advice, ADV.bodies(self.out, window), self.grain,
                                               named=binding is not None)
        except (Unexpressible, ValueError) as e:
            return {**got, "refused": f"{type(e).__name__}: {e}"[:500]}
        lowered = time.perf_counter()
        if binding is None:
            self.held[window] = (compiled, low, None)
            return {**got, "templates": len(low.circuit.templates), "units": sum(k for _, k in low.circuit.templates)}
        reader = None
        if self.record is not None:
            from verity_vllm.commit.opened import OpenedReader

            reader = OpenedReader(binding.com, binding.rc)
        served = served_slots(low, self.cls, compiled, binding, self.out / "windows" / f"w{window}", reader)
        plan = Plan(low, served)
        self.held[window] = (compiled, low, plan)
        got = {**got, **plan.describe(), "seconds": {"lower": round(lowered - t0, 3), "plan": round(time.perf_counter() - lowered, 3)}}
        if self.record is not None:
            try:
                if self.unregistered is not None:
                    raise ValueError(self.unregistered)
                got["record"] = self.record.make(self, window, compiled, low, plan, served, binding)
            except Exception as e:  # noqa: BLE001  (a window whose record fails says so; the server goes on)
                got["record"] = {"error": f"{type(e).__name__}: {e}"[:500]}
            sys.stderr.write("WINDOW-RECORD " + json.dumps({"window": window, **got["record"]}) + "\n")
            sys.stderr.flush()
        return got


def served_slots(low: Any, cls: Any, compiled: Any, binding: Any, builds: Path, reader: Any = None) -> list[Any]:
    """Each of ``low``'s Programs' `window_rows.Served`, in Program order: the request whose slot its anchor and lane name (slot
    k the window's k-th request, `window_class.takes`'s placement), read through its Program's index (`ProgramIndex` over the
    Build written under ``builds``) and the window's binding map, from its first step; its values read through ``reader``
    (`commit.opened.OpenedReader`), else located alone."""
    from verity_vllm.check.replay.index import ProgramIndex
    from verity_vllm.check.replay.opening import CommittedStore
    from verity_vllm.query.manifest import format as MF

    from .naming import write_build
    from .steps import trace_dir
    from .window_rows import Served

    units = [u for u in compiled.requests if u is not None]
    slot_of = {(cls.anchor(k), cls.lane(k)): k for k in range(cls.slots)}
    store = CommittedStore.of_record(binding.bmap, reader if reader is not None else lambda *_: None, binding.aliases)
    dirs: dict[str, str] = {}
    out = []
    for P in low.programs:
        u = units[slot_of[(P.anchor, P.lane)]]
        if u.digest not in dirs:
            named = compiled.named[u.digest]
            dirs[u.digest] = write_build(str(builds / trace_dir({**named.shape, "policy": named.shape["sampling"]})),
                                         compiled.programs[u.digest], named)
        pi = ProgramIndex(MF.load_program(dirs[u.digest])["instances"], program_dir=dirs[u.digest])
        out.append(Served.of_record(pi, store, u.id, u.first_step))
    return out
