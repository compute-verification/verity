"""Served taps: acquisition sources that hold the tensors the served forward computed for the `call_boundaries` identities they claim
(`call_boundary_source.CLAIMS`), for the call-boundary source to commit (`ServedTap.words`).  A tap commits nothing itself: the
call-boundary source commits every `call_boundaries` identity, evaluated on the host or served, module by module in the order it commits
host words, so the run root is the same whichever source a word came from, and the binding map records each identity's source
(`call_boundary_source.CallBoundarySource.identity_sources`: `host` or `tap`).  A served tensor of another dtype, element count or row
width than its identity's (`layout`), or one written in place after the forward computed it, is a collector error naming it, and nothing
of that module at that step is committed.  A run that is never a record (`placeholder`, the Commit's warm-up instrumented run) commits
zero words of each identity's shape and captures nothing.  The replay requires every tapped identity's producing Calls in its draw's
population (`check.replay.coverage.taps_in_draw`): a served value no draw can reach is bound but never checked."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from verity_vllm.acquire.sources import call_boundary_source as CB
from verity_vllm.acquire.sources.call_boundary_plan import FAMILY, WORD_DTYPES


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


def element_refusal(ids: Sequence[Mapping[str, Any]]) -> str | None:
    """Why identities `ids` are not each a whole [rows, width] Value of a word dtype, or None."""
    for r in ids:
        rows, width = int(r.get("rows") or 0), int(r.get("width") or 0)
        if [int(x) for x in r.get("element_range") or ()] != [0, rows * width] or str(r.get("dtype")) not in WORD_DTYPES:
            return f"{CB.CP._label(r)} is {rows} x {width} {r.get('dtype')} over {r.get('element_range')}"
    return None


class ServedTap:
    """The claimed identities by engine step and module, the modules that ran, the served tensors a subclass's capture holds
    (`_served[(step, module)]`: name -> (tensor, its version when the forward computed it)), and the words the call-boundary source
    commits from them."""

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
        self._served: dict[tuple[int, str], dict[str, tuple[Any, int]]] = {}
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
                t, version = got.get(name(r), (None, None))
                why = ("is not served by the forward" if t is None else layout(torch, r, t)
                       or (None if t._version == version else "was written in place after the forward computed it"))
                if why:
                    self.error(f"{self.label}: {CB.CP._label(r)} {why}: nothing of {module} at engine step {e} is committed")
                    return None
                out.append((name(r), t.contiguous().view(int(r["rows"]), int(r["width"]))))
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

    def declaration(self, evidence: str) -> list[dict]:
        return [{"class": FAMILY, "how": "bound-as-leaf", "rule": None, "count": self.stats["tensors"], "bytes": self.stats["bytes"],
                 "tensors": self.stats["tensors"], "steps": self.stats["steps"], "errors": self.stats["errors"], "source": "tap",
                 "evidence": f"{evidence}; committed by the call-boundary source in its place in the step stream, as the runtime computed "
                             "them; the replay's openings of the producing Calls, every one in its draw's population, check them against "
                             "the Program"}]


__all__ = ["ServedTap", "name", "module_of", "layout", "element_refusal"]
