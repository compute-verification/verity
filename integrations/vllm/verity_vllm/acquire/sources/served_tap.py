"""Served taps: acquisition sources that hold the tensors the served forward computed for the `call_boundaries` identities they claim
(`call_boundary_source.CLAIMS`), for the call-boundary source to commit (`ServedTap.words`).  A tap commits nothing itself: the
call-boundary source commits every `call_boundaries` identity, evaluated on the host or served, module by module in the order it commits
host words, so the run root is the same whichever source a word came from, and the binding map records each identity's source
(`call_boundary_source.CallBoundarySource.identity_sources`: `host` or `tap`); its declaration of the class counts both (each tap's
`served`), and a tap declares none of its own.  A served tensor of another dtype, element count or row
width than its identity's (`layout`), or one a torch call of the forward wrote in place after computing it (`note_writes`; an inference
tensor has no version counter), is a collector error naming it, and nothing of that module at that step is committed; the tap keeps a
copy of each served tensor as the forward left it (`hold`), so nothing after the forward changes what is committed.  A run that is never
a record (`placeholder`, the Commit's warm-up instrumented run) commits zero words of each identity's shape and captures nothing.  The replay requires every tapped identity's producing Calls in its draw's
population (`check.replay.coverage.taps_in_draw`): a served value no draw can reach is bound but never checked."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from verity_vllm.acquire.sources import call_boundary_source as CB
from verity_vllm.acquire.sources.call_boundary_plan import WORD_DTYPES

#: the in-place dunders and item assignment, beside the in-place methods `name_`
INPLACE = frozenset({"__iadd__", "__isub__", "__imul__", "__itruediv__", "__ifloordiv__", "__imod__", "__ipow__", "__imatmul__", "__iand__",
                     "__ior__", "__ixor__", "__ilshift__", "__irshift__", "__setitem__"})


def name(r: Mapping[str, Any]) -> str:
    return f"{r['op_path']}/{r['output_member']}"


def module_of(r: Mapping[str, Any]) -> str:
    """The module body an identity's fx node is in: its op_path up to the node."""
    return str(r["op_path"]).rsplit("/", 1)[0]


def layout(torch: Any, r: Mapping[str, Any], t: Any) -> str | None:
    """Why served tensor `t` is not identity `r`'s words in the Program's layout ([rows, width] of its word dtype, row-major), or None."""
    rows, width = int(r["rows"]), int(r["width"])
    if t.dtype != getattr(torch, WORD_DTYPES[str(r["dtype"])][1]) or t.numel() != rows * width or (t.dim() and int(t.shape[-1]) != width):
        return f"is {tuple(t.shape)} {t.dtype} served; the identity is {rows} x {width} {r['dtype']}"
    return None


def note_writes(writes: list[tuple[int, str]], func: Any, args: tuple, kwargs: Mapping[str, Any] | None, Tensor: Any) -> None:
    """Append to `writes` (storage, function name) for each tensor a torch call writes in place: its `out=`, the first argument of an
    in-place method (`add_`, `__iadd__`, `__setitem__`) and each argument its schema marks written (a custom op's mutated operands)."""
    fn = getattr(func, "__name__", "")
    out = (kwargs or {}).get("out")
    got = [*(out if isinstance(out, (tuple, list)) else [out])]
    if args and (fn in INPLACE or (fn.endswith("_") and not fn.startswith("__"))):
        got.append(args[0])
    for i, a in enumerate(getattr(getattr(func, "_schema", None), "arguments", None) or ()):
        if a.alias_info is not None and a.alias_info.is_write:
            got.append(args[i] if i < len(args) else (kwargs or {}).get(a.name))
    writes += [(t.untyped_storage().data_ptr(), fn) for t in got if isinstance(t, Tensor) and t.numel()]


def element_refusal(ids: Sequence[Mapping[str, Any]]) -> str | None:
    """Why identities `ids` are not each a whole [rows, width] Value of a word dtype, or None."""
    for r in ids:
        rows, width = int(r.get("rows") or 0), int(r.get("width") or 0)
        if [int(x) for x in r.get("element_range") or ()] != [0, rows * width] or str(r.get("dtype")) not in WORD_DTYPES:
            return f"{CB.CP._label(r)} is {rows} x {width} {r.get('dtype')} over {r.get('element_range')}"
    return None


class ServedTap:
    """The claimed identities by engine step and module, the modules that ran, the copies of the served tensors a subclass's capture
    holds (`_served[(step, module)]`: name -> tensor, `hold`), and the words the call-boundary source commits from them."""

    short_name = "served_tap"
    label = "served tap"

    def __init__(self, committer: Any, model: Any, ids: Sequence[Mapping[str, Any]]) -> None:
        import torch

        self.torch, self.c, self.model = torch, committer, model
        self._at: dict[int, dict[str, list[Mapping[str, Any]]]] = {}
        for r in ids:
            self._at.setdefault(int(r["engine_step"]), {}).setdefault(module_of(r), []).append(r)
        self._names = frozenset(name(r) for r in ids)
        self.modules = sorted({m for mods in self._at.values() for m in mods})
        self._handles: list[Any] = []
        self._ran: dict[tuple[int, str], Any] = {}                           # (engine step, module) -> the device it ran on
        self._served: dict[tuple[int, str], dict[str, Any]] = {}
        self._steps: set[int] = set()
        self.stats: dict[str, Any] = {"tensors": 0, "bytes": 0, "steps": 0, "errors": 0}
        self.active = False
        self.placeholder = False

    def ids_at(self, engine_step: int) -> dict[str, list[Mapping[str, Any]]]:
        """Module -> its claimed identities at `engine_step`, in the manifest's order."""
        return self._at.get(int(engine_step), {})

    def on_step_begin(self) -> None:
        self._ran.clear()
        self._served.clear()

    def error(self, msg: str) -> None:
        self.stats["errors"] += 1
        self.c.errors.append(msg[:300])

    def ran(self, e: int, module: str, args: Sequence[Any]) -> bool:
        """At `module`'s pre-hook: True when the forward is one to capture (it ran first at a step that claims it), else False."""
        if (e, module) in self._ran:
            self.error(f"{self.label}: {module} ran twice at engine step {e}; the Program has one body")
            return False
        self._ran[(e, module)] = next((a.device for a in args if isinstance(a, self.torch.Tensor)), self.torch.device("cpu"))
        return not self.placeholder

    def hold(self, e: int, module: str, served: Mapping[str, tuple[Any, int]], writes: Sequence[tuple[int, str]]) -> None:
        """At `module`'s forward hook, engine step `e`: keep a contiguous copy of each served tensor (`served`: name -> (tensor, the
        length of `writes` when the forward computed it)); one a later call of the forward wrote in place (`writes`, `note_writes`) is a
        collector error, and nothing of the module is kept."""
        for n, (t, mark) in served.items():
            fn = next((f for p, f in writes[mark:] if p == t.untyped_storage().data_ptr()), None)
            if fn is not None:
                self.error(f"{self.label}: {n} was written in place ({fn}) after the forward computed it: nothing of {module} at engine "
                           f"step {e} is committed")
                return
        self._served[(e, module)] = {n: t.clone(memory_format=self.torch.contiguous_format) for n, (t, _m) in served.items()}

    def words(self, e: int, module: str) -> list[tuple[str, Any]] | None:
        """The claimed identities of `module` at engine step `e` as [rows, width] words, in the manifest's order; None (a collector error)
        when the module did not run, its capture failed, or a served tensor is not its identity's."""
        ids = self.ids_at(e).get(module, [])
        if (e, module) not in self._ran:
            self.error(f"{self.label}: {module} did not run at engine step {e}; the manifest names its call boundaries there")
            return None
        torch, out = self.torch, []
        if self.placeholder:
            out = [(name(r), torch.zeros((int(r["rows"]), int(r["width"])), dtype=getattr(torch, WORD_DTYPES[str(r["dtype"])][1]),
                                         device=self._ran[(e, module)])) for r in ids]
        else:
            got = self._served.get((e, module))
            if got is None:
                return None
            for r in ids:
                t = got.get(name(r))
                why = "is not served by the forward" if t is None else layout(torch, r, t)
                if why:
                    self.error(f"{self.label}: {CB.CP._label(r)} {why}: nothing of {module} at engine step {e} is committed")
                    return None
                out.append((name(r), t.view(int(r["rows"]), int(r["width"]))))
        self.stats["tensors"] += len(out)
        self.stats["bytes"] += sum(int(v.numel()) * int(v.element_size()) for _n, v in out)
        self._steps.add(e)
        self.stats["steps"] = len(self._steps)
        return out

    def targets_at(self, engine_step: int) -> dict[str, int]:
        return {name(r): 1 for rs in self.ids_at(engine_step).values() for r in rs}

    def target_names(self) -> frozenset[str]:
        return self._names

    split_step_names = CB.CallBoundarySource.split_step_names

    def step_bytes(self) -> dict[int, tuple[int, int, int]]:
        """Engine step -> (bytes, token rows, tensors) of its claimed identities (token rows: the most rows any of them has)."""
        out = {}
        for e, mods in self._at.items():
            rs = [r for v in mods.values() for r in v]
            out[e] = (sum(int(r["rows"]) * int(r["width"]) * np.dtype(WORD_DTYPES[str(r["dtype"])][0]).itemsize for r in rs),
                      max(int(r["rows"]) for r in rs), len(rs))
        return out

    def declaration_classes(self) -> list[dict]:
        """None: the call-boundary source commits a tap's words and declares them in its class (`CallBoundarySource.declaration_classes`,
        from each tap's `served`), one declaration per class as the Commit's omitted-producer check (R-14) reads them."""
        return []

    def summary(self, evidence: str) -> dict[str, Any]:
        """This tap's part of the call-boundary class: its counts and what it served (`evidence`)."""
        return {"source": self.short_name, "tensors": self.stats["tensors"], "bytes": self.stats["bytes"], "steps": self.stats["steps"],
                "errors": self.stats["errors"],
                "evidence": f"{evidence}, as the runtime computed them, committed by the call-boundary source in their place in the step "
                            "stream; the replay's openings of the producing Calls, every one in its draw's population, check them against "
                            "the Program"}


__all__ = ["ServedTap", "INPLACE", "name", "module_of", "layout", "note_writes", "element_refusal"]
