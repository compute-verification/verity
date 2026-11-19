"""A served Commit window's circuit at its close, before its seal and any coin: its advice committed as `verity-vllm advice`
commits it (`advice.write`), the class circuit compiled from that advice and the window's bodies in the one compile call
(`window_circuit.served_lowering`), and the committer's units in order (`window_rows.Plan`): each template's units, each
port's source and the served rows each reads where they are committed."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Sequence

from verity_vllm.program.predict import window_circuit as WC
from verity_vllm.program.predict import window_rows as WR
from verity_vllm.program.predict.build import Unexpressible

from . import advice as ADV


class ClosePlan:
    """``registered`` (`steps.Registered`) and ``cls`` the window class the circuit is compiled at; ``compiler`` the advice's
    (`advice.registered_compiler`); ``flush`` drains the advice trace (`StepTrace.drain`) before it is read; ``served(lowered,
    window)`` gives each slot's `window_rows.Served` (with its ``locate``), and without it the outcome names the circuit's
    templates and units alone. A window whose advice is pending or refused, or that the class doesn't take, has no circuit and
    says why."""

    def __init__(self, out: Path, registered: Any, cls: Any, *, compiler: Any, grain: Any = None,
                 flush: Callable[[], Any] | None = None,
                 served: Callable[[Any, int], Sequence[Any]] | None = None,
                 registered_roots: dict[str, str] | None = None) -> None:
        self.out, self.registered, self.cls, self.compiler = Path(out), registered, cls, compiler
        self.grain = grain if grain is not None else WC._GRAIN
        self.flush, self.served, self.registered_roots = flush, served, registered_roots
        self.held: dict[int, tuple[Any, Any, WR.Plan | None]] = {}

    def __call__(self, window: int) -> dict[str, Any]:
        if self.flush is not None:
            self.flush()
        rep = ADV.write(self.out, self.out / "advice", window=window, compiler=self.compiler,
                        registered_roots=self.registered_roots)
        for key in ("pending", "refused"):
            if str(window) in rep[key]:
                return {"window": window, key: rep[key][str(window)]}
        got = {"window": window, "advice": rep["windows"][str(window)]["root"]}
        advice = (self.out / "advice" / f"w{window}" / "private" / "advice.bin").read_bytes()
        try:
            compiled, low = WC.served_lowering(self.registered, self.cls, advice, ADV.bodies(self.out, window), self.grain)
        except (Unexpressible, ValueError) as e:
            return {**got, "refused": f"{type(e).__name__}: {e}"[:500]}
        if self.served is None:
            self.held[window] = (compiled, low, None)
            return {**got, "templates": len(low.circuit.templates), "units": sum(k for _, k in low.circuit.templates)}
        plan = WR.Plan(low, self.served(low, window))
        self.held[window] = (compiled, low, plan)
        return {**got, **plan.describe()}
