"""The linear-site interface: what computes y at a linear layer, where its weight comes from, and the one object vLLM's
``quant_method.apply`` calls.  It imports neither vLLM nor torch, so executors and sources are tested on the CPU with
stand-ins (`tests/protocols/linear`), and only the module that finds the sites in a vLLM model touches vLLM.

* A `LinearExecutor` computes each call's y: a PoUW scheme (Pearl-C, ncp-v2), or nothing, in which case vLLM's own
  ``apply`` does.  It declares what it can't compose with (``keeps_weight_state``), which weight formats it reads, which
  controls and schedules it implements, and runs one pass at a time (`ExecutorPass`), fixed by its `PassSpec`.
* A `WeightSource` supplies each use's weight (POUS: decoded from its store) and is told when each forward ends.
* A `Stack` joins them on one rank.  Its `refusals` read the declared attributes, so a composition that can't hold is
  refused before any patch; `activate` switches arms only between passes; `apply` computes inside a pass, and an active
  executor outside one fails closed; `forward` always ends the forward at the source, even when the forward raises.

The design, and the migration of today's executors onto it: research store ``docs/pouw/vllm-integration-api.md``.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Protocol

from verity_vllm.errors import Refused

Tensor = Any


@dataclass(frozen=True)
class Shard:
    """``axis`` ``n`` is column-parallel (qkv, gate_up), ``k`` row-parallel (o, down), None replicated; vLLM adds a
    row-parallel layer's bias on rank 0 only (``bias_here``)."""

    rank: int = 0
    world: int = 1
    axis: str | None = None
    bias_here: bool = True


@dataclass(frozen=True)
class LinearSite:
    """One linear layer on one rank.  ``weight_id`` is global and stable across ranks; ``n`` and ``k`` are this rank's."""

    weight_id: int
    ordinal: int
    qualname: str
    n: int
    k: int
    weight_format: str
    has_bias: bool = False
    shard: Shard = field(default_factory=Shard)


@dataclass(frozen=True)
class Control:
    """A poisoning or negative control: ``poison`` (the byte), ``omit`` (step groups), ``tamper`` (what), ``drop`` (a
    fraction of the store's blocks)."""

    kind: str
    arg: Any = None


@dataclass(frozen=True)
class PassSpec:
    """What a pass runs with, fixed for its life."""

    index: int = 0
    hashing: bool = True
    schedule: str = "serial"
    retain: str | None = None
    controls: tuple[Control, ...] = ()
    run_id: str | None = None


@dataclass(frozen=True)
class Forward:
    index: int
    pass_index: int | None


class WeightReader(Protocol):
    def __call__(self, site: LinearSite, rows: slice | None = None) -> Tensor: ...


class WeightSource(Protocol):
    name: str
    forbids_weight_state: bool
    controls: frozenset[str]

    def commit(self, sites: Sequence[LinearSite], read: WeightReader) -> Mapping[str, Any]: ...
    def install(self, salt: bytes | None) -> Mapping[str, Any]: ...
    def weight(self, site: LinearSite, fwd: Forward | None) -> Tensor: ...
    def forward_end(self, fwd: Forward) -> None: ...
    def plaintext(self) -> Mapping[str, Tensor] | None: ...
    def close(self) -> None: ...


class ExecutorPass(Protocol):
    spec: PassSpec

    def call(self, site: LinearSite, x: Tensor, weight: Tensor, bias: Tensor | None) -> Tensor: ...
    def end(self) -> Any: ...


class LinearExecutor(Protocol):
    name: str
    keeps_weight_state: bool
    accepts: frozenset[str]
    controls: frozenset[str]
    schedules: frozenset[str]

    def refusals(self, sites: Sequence[LinearSite]) -> list[str]: ...
    def commit_weights(self, sites: Sequence[LinearSite], read: WeightReader) -> Mapping[int, bytes]: ...
    def start(self, epoch: Any) -> None: ...
    def begin(self, spec: PassSpec) -> ExecutorPass: ...
    def close(self) -> None: ...


@dataclass(frozen=True)
class Arm:
    """A bench or panel mode.  Stock baselines differ in ``engine`` (vLLM's ``quantization=``), not at the site; ``executor``
    None is vLLM's own ``apply`` with the stack attached, so the wrapper's cost is inside the baseline."""

    name: str
    engine: str
    executor: str | None = None
    source: str | None = None
    spec: PassSpec = field(default_factory=PassSpec)


class _WithWeight:
    """The layer vLLM's own ``apply`` sees under a source: the source's weight, and every other attribute the layer's."""

    def __init__(self, layer: Any, weight: Tensor) -> None:
        object.__setattr__(self, "_layer", layer)
        object.__setattr__(self, "weight", weight)

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_layer"), name)


class Stack:
    """One rank's sites with their executor and source.  ``passes`` holds what each ended pass returned, in order."""

    def __init__(self, sites: Sequence[LinearSite], executor: LinearExecutor | None = None,
                 source: WeightSource | None = None) -> None:
        self.sites, self.executor, self.source = tuple(sites), executor, source
        self.active: Arm | None = None
        self.passes: list[Any] = []
        self._pass: ExecutorPass | None = None
        self._forward: Forward | None = None
        self._forwards = 0

    def refusals(self, spec: PassSpec | None = None) -> list[str]:
        """Why this stack can't run, or run `spec` (empty: it can)."""
        ex, src, out = self.executor, self.source, []
        if ex is not None and src is not None and src.forbids_weight_state and ex.keeps_weight_state:
            out.append(f"{ex.name} keeps weight-derived state across calls, and {src.name} forbids it: its GEMMs would not "
                       f"read the weight {src.name} supplies")
        if ex is not None:
            formats = sorted({s.weight_format for s in self.sites} - set(ex.accepts))
            if formats:
                out.append(f"{ex.name} reads {', '.join(sorted(ex.accepts))} weights, not {', '.join(formats)}")
            out.extend(ex.refusals(self.sites))
        if spec is not None:
            implemented = (ex.controls if ex is not None else frozenset()) | (src.controls if src is not None else frozenset())
            out.extend(f"control {c.kind!r}: nothing in this stack implements it" for c in spec.controls if c.kind not in implemented)
            schedules = ex.schedules if ex is not None else frozenset({"serial"})
            if spec.schedule not in schedules:
                out.append(f"schedule {spec.schedule!r}: {ex.name if ex is not None else 'vLLM own apply'} runs "
                           f"{', '.join(sorted(schedules))}")
        return out

    def activate(self, arm: Arm | None) -> None:
        """Switch to `arm` (None: vLLM's own ``apply``), between passes only."""
        if self._pass is not None:
            raise RuntimeError("a pass is open: arms switch between passes")
        if arm is not None and arm.executor is not None and (self.executor is None or arm.executor != self.executor.name):
            raise ValueError(f"arm {arm.name!r} runs {arm.executor}, and this stack's executor is "
                             f"{self.executor.name if self.executor is not None else 'none'}")
        bad = self.refusals(arm.spec) if arm is not None else []
        if bad:
            raise Refused(bad, "linear stack")
        self.active = arm

    @property
    def executing(self) -> bool:
        return self.active is not None and self.active.executor is not None

    @property
    def open_pass(self) -> ExecutorPass | None:
        """The pass `pass_` holds open now, if any."""
        return self._pass

    @contextmanager
    def pass_(self, spec: PassSpec | None = None) -> Iterator[ExecutorPass | None]:
        """One pass under `spec` (the active arm's by default).  A pass that raises is ended, so its lanes join, and not
        recorded."""
        if self._pass is not None:
            raise RuntimeError("a pass is already open")
        if not self.executing:
            yield None
            return
        spec = self.active.spec if spec is None else spec
        bad = self.refusals(spec)
        if bad:
            raise Refused(bad, "linear stack")
        self._pass = self.executor.begin(spec)
        try:
            yield self._pass
        except BaseException:
            p, self._pass = self._pass, None
            p.end()
            raise
        p, self._pass = self._pass, None
        self.passes.append(p.end())

    @contextmanager
    def forward(self) -> Iterator[Forward]:
        """One model-runner step: the source's `forward_end` runs once it is over, whatever happened inside it."""
        fwd = Forward(self._forwards, None if self._pass is None else self._pass.spec.index)
        self._forwards += 1
        self._forward = fwd
        try:
            yield fwd
        finally:
            self._forward = None
            if self.source is not None:
                self.source.forward_end(fwd)

    def apply(self, site: LinearSite, layer: Any, x: Tensor, bias: Tensor | None, original: Callable[..., Tensor]) -> Tensor:
        """What a patched ``quant_method.apply`` calls: the executor's pass when an arm executes, else vLLM's `original`,
        either one with the source's weight when there is a source."""
        weight = self.source.weight(site, self._forward) if self.source is not None else None
        if not self.executing:
            return original(layer if weight is None else _WithWeight(layer, weight), x, bias)
        if self._pass is None:
            raise RuntimeError(f"{self.executor.name} is active outside a pass: open one (Stack.pass_) before the forward")
        return self._pass.call(site, x, layer.weight if weight is None else weight, bias)


__all__ = ["Arm", "Control", "ExecutorPass", "Forward", "LinearExecutor", "LinearSite", "PassSpec", "Shard", "Stack",
           "Tensor", "WeightReader", "WeightSource"]
