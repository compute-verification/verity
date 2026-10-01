"""The soft-capped logits' call boundaries (`call_boundaries` identities of `logits_processor`: the lm_head GEMM's output, its quotient by the
cap and that quotient's tanh) as a served tap (`served_tap`).  With `--logits-tap 1` the source claims them (`claiming`,
`call_boundary_source.CLAIMS`): the call-boundary source does not evaluate them, and commits in their place the tensors the served forward
computed -- the dividend of the division whose quotient `tanh` reads, the quotient and the tanh -- as `[rows, width]` words under the
identities' names, the same names, shapes, dtypes and place in the step stream its host words have.  The replay's openings of the
producing Calls (the GEMM, the division, tanh) check them against the Program; nothing here evaluates the Program.

A forward pre-hook on `logits_processor` enters a TorchFunctionMode that keeps references to each division's dividend and quotient and to
the one tanh; the forward hook leaves it.  A step whose claimed identities are not one each of the three, a manifest of several ranks or
requests, is refused (exit 3; `taps.attached` leaves the tap out, so the CPU gate names them first).  A forward that ran no tanh of a
quotient, ran twice, wrote a served tensor in place after computing it, or served another shape or dtype is a collector error naming it,
and nothing of the step is committed.  The Build's
call-boundary check and plan (`boundary-plan`, `gate call-boundaries`) and the Commit take the option, VERITY_LOGITS_TAP in the row's
environment by default, and the Commit derives again a plan made under another claim (`pipeline.boundary_plan.use`)."""
from __future__ import annotations

import re
from collections import Counter
from typing import Any, Mapping, Sequence

from verity_vllm.acquire.sources import call_boundary_source as CB
from verity_vllm.acquire.sources import served_tap as ST
from verity_vllm.acquire.sources.call_boundary_plan import FAMILY
from verity_vllm.config import option

ENV = "VERITY_LOGITS_TAP"
MODULE = "logits_processor"
NODE = re.compile(rf"{MODULE}/(triton_launch|div|tanh)(?:_\d+)?")
KINDS = ("triton_launch", "div", "tanh")
DIVS = frozenset({"__truediv__", "div", "true_divide"})
name = ST.name


def option_field() -> Any:
    """The `--logits-tap` option of every command that reads call boundaries (`commit`, `boundary-plan`, `gate call-boundaries`)."""
    return option("--logits-tap", env=ENV, type=int, default=0, metavar="0|1",
                  help="1: the logits tap claims the soft-capped logits' call boundaries and commits the served tensors "
                       "(acquire.sources.logits_tap_source); the row's Build check, its plan and the Commit must agree")


def claim(r: Mapping[str, Any]) -> str | None:
    """This source's `short_name` for a rank-0 call boundary of the logits' GEMM, division or tanh, else None."""
    if r.get("family") != FAMILY or int(r.get("rank") or 0) or int(r.get("invocation") or 0) or r.get("program_op_path"):
        return None
    return LogitsTapSource.short_name if NODE.fullmatch(str(r.get("op_path") or "")) else None


def claiming(on: bool) -> None:
    """Whether the tap claims its identities (`call_boundary_source.CLAIMS`), from a command's `--logits-tap`."""
    if on and claim not in CB.CLAIMS:
        CB.CLAIMS.append(claim)
    elif not on and claim in CB.CLAIMS:
        CB.CLAIMS.remove(claim)


def kind(r: Mapping[str, Any]) -> str:
    m = NODE.fullmatch(str(r["op_path"]))
    assert m is not None
    return m.group(1)


def manifest_sites(manifest: Mapping[str, Any] | None) -> list[Mapping[str, Any]]:
    """The `call_boundaries` identities of a manifest this source commits, while it is claiming: every one its `claim` takes and no
    earlier claim does."""
    if claim not in CB.CLAIMS:
        return []
    return [r for r in (manifest or {}).get("identities") or [] if r.get("family") == FAMILY and claim(r)
            and CB.claimed(r) == LogitsTapSource.short_name]


def refusal(manifest: Mapping[str, Any] | None, ids: Sequence[Mapping[str, Any]]) -> str | None:
    """Why the source cannot commit `ids` of `manifest`, or None."""
    ranks = int((manifest or {}).get("ranks") or 1)
    if ranks > 1:
        return f"a manifest of {ranks} ranks (the tap serves one rank)"
    why = ST.element_refusal(ids)
    if why:
        return why
    by_step: dict[int, Counter] = {}
    for r in ids:
        by_step.setdefault(int(r["engine_step"]), Counter())[kind(r)] += 1
    for e, got in sorted(by_step.items()):
        if got != Counter(KINDS):
            return (f"engine step {e} claims {dict(sorted(got.items()))}; the tap takes one GEMM output, its quotient and its tanh per step "
                    "(one request)")
    return None


class LogitsTapSource(ST.ServedTap):
    short_name = "logits_tap"
    label = "logits tap"

    def __init__(self, committer: Any, model: Any, ids: Sequence[Mapping[str, Any]]) -> None:
        super().__init__(committer, model, ids)
        self._mode: Any = None
        self._step = -1

    def install(self) -> None:
        from torch.overrides import TorchFunctionMode

        mods = dict(self.model.named_modules())
        if MODULE not in mods:
            raise RuntimeError(f"logits tap: {MODULE} is not a module of the model")
        Tensor = self.torch.Tensor

        class Chain(TorchFunctionMode):
            """The forward's divisions (dividend, quotient) and its tanh of one of their quotients, each with the forward's in-place
            writes before it, and every write (`served_tap.note_writes`)."""

            def __init__(self) -> None:
                super().__init__()
                self.divs: list[tuple[tuple[Any, int], tuple[Any, int]]] = []
                self.tanhs = 0
                self.got: tuple[tuple[Any, int], ...] | None = None
                self.writes: list[tuple[int, str]] = []

            def __torch_function__(self, func: Any, types: Any, args: tuple = (), kwargs: dict | None = None) -> Any:
                out = func(*args, **(kwargs or {}))
                ST.note_writes(self.writes, func, args, kwargs, Tensor)
                fn, w = getattr(func, "__name__", ""), len(self.writes)
                if fn in DIVS and args and isinstance(args[0], Tensor) and isinstance(out, Tensor):
                    self.divs.append(((args[0], w), (out, w)))
                elif fn == "tanh" and args:
                    self.tanhs += 1
                    self.got = next(((a, q, (out, w)) for a, q in self.divs if q[0] is args[0]), None)
                return out

        self._Chain = Chain
        m = mods[MODULE]
        self._handles += [m.register_forward_pre_hook(self._pre), m.register_forward_hook(self._post, always_call=True)]
        self.active = True

    def uninstall(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []
        if self._mode is not None:
            self._mode.__exit__(None, None, None)
            self._mode = None
        self.on_step_begin()
        self.active = False

    def _pre(self, _module: Any, args: tuple) -> None:
        job = self.c._cur
        e = int(job.step) if job is not None else None
        if e is None or MODULE not in self.ids_at(e) or self._mode is not None or not self.ran(e, MODULE, args):
            return None
        self._mode = self._Chain()
        self._mode.__enter__()
        self._step = e
        return None

    def _post(self, _module: Any, _args: tuple, _out: Any) -> None:
        mode, self._mode = self._mode, None
        if mode is None:
            return None
        mode.__exit__(None, None, None)
        if mode.got is None or mode.tanhs != 1:
            self.error(f"logits tap: {MODULE} at engine step {self._step} ran {len(mode.divs)} divisions and {mode.tanhs} tanh, not one tanh "
                       "of a quotient")
            return None
        got = dict(zip(KINDS, mode.got))
        self.hold(self._step, MODULE, {name(r): got[kind(r)] for r in self.ids_at(self._step)[MODULE]}, mode.writes)
        return None

    def extra_step_bytes(self, envelope: Mapping[str, Any], model: Any) -> dict[str, int]:
        per = self.step_bytes().values()
        return {"per_token_bytes": 0, "per_seq_bytes": max((b for b, _r, _n in per), default=0), "tensors_per_step": max((n for _b, _r, n in per), default=0)}

    def declaration_classes(self) -> list[dict]:
        return self.declaration(f"logits tap: the served `<fx node>/out` of {len(self._names)} call boundaries of {MODULE} (the lm_head "
                                "GEMM's output as the soft cap's division reads it, the quotient, its tanh)")


def attach(com: Any, manifest: Mapping[str, Any] | None, required_classes: tuple[str, ...], *, explicit_classes: bool = False,
           model: Any = None) -> tuple[tuple[str, ...], LogitsTapSource | None]:
    """Attach the source to `com` when the manifest names identities it claims; one it cannot commit (`refusal`) refuses the Commit (exit
    3).  Returns the required classes with `call_boundaries` added unless they were given explicitly."""
    ids = manifest_sites(manifest)
    if not ids:
        return required_classes, None
    why = refusal(manifest, ids)
    if why:
        print(f"[delta] REFUSE: logits tap: {why}", flush=True)
        raise SystemExit(3)
    src = LogitsTapSource(com, model if model is not None else getattr(com, "model", None), ids)
    com.attach_source(src)
    print(f"[delta] logits tap attached: {len(ids)} call boundaries of {MODULE} in {len(src._at)} engine steps, the served tensors -> "
          f"committer {com.name}", flush=True)
    if not explicit_classes and FAMILY not in required_classes:
        return (*required_classes, FAMILY), src
    return required_classes, src


__all__ = ["ENV", "MODULE", "LogitsTapSource", "option_field", "claim", "claiming", "kind", "name", "manifest_sites", "refusal", "attach"]
