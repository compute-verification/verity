"""The soft-capped logits' call boundaries (`call_boundaries` identities of `logits_processor`: the lm_head GEMM's output, its quotient by the
cap and that quotient's tanh) as an acquisition source that commits the served tensors.  With `--logits-tap 1` the source claims them
(`claiming`, `call_boundary_source.CLAIMS`): the call-boundary source neither plans nor evaluates them, and this one, attached just before it, commits
at each step's end, in the manifest's order, the tensors the served forward computed -- the dividend of the division whose quotient `tanh`
reads, the quotient and the tanh -- as `[rows, width]` words under the identities' names, the same names, shapes, dtypes and place in the
step stream the call-boundary source gives its host words.  The replay's openings of the producing Calls (the GEMM, the division, tanh)
check them against the Program; nothing here evaluates the Program.

A forward pre-hook on `logits_processor` enters a TorchFunctionMode that keeps references to each division's dividend and quotient and to
the one tanh; the forward hook leaves it.  A step whose claimed identities are not one each of the three, a manifest of several ranks or
requests, is refused (exit 3; `taps.attached` leaves the tap out, so the CPU gate names them first).  A forward that ran no tanh of a
quotient, ran twice, or served another shape or dtype is a collector error naming it, and nothing of the step is committed.  The Build's
call-boundary check and plan (`boundary-plan`, `gate call-boundaries`) and the Commit take the option, VERITY_LOGITS_TAP in the row's
environment by default, and the Commit derives again a plan made under another claim (`pipeline.boundary_plan.use`)."""
from __future__ import annotations

import re
import time
from collections import Counter
from typing import Any, Mapping, Sequence

import numpy as np

from verity_vllm.acquire.sources import call_boundary_source as CB
from verity_vllm.acquire.sources.call_boundary_plan import FAMILY, WORD_DTYPES
from verity_vllm.config import option

ENV = "VERITY_LOGITS_TAP"
MODULE = "logits_processor"
NODE = re.compile(rf"{MODULE}/(triton_launch|div|tanh)(?:_\d+)?")
KINDS = ("triton_launch", "div", "tanh")
DIVS = frozenset({"__truediv__", "div", "true_divide"})


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


def name(r: Mapping[str, Any]) -> str:
    return f"{r['op_path']}/{r['output_member']}"


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
    by_step: dict[int, Counter] = {}
    for r in ids:
        by_step.setdefault(int(r["engine_step"]), Counter())[kind(r)] += 1
        rows, width = int(r.get("rows") or 0), int(r.get("width") or 0)
        if [int(x) for x in r.get("element_range") or ()] != [0, rows * width] or str(r.get("dtype")) not in WORD_DTYPES:
            return f"{CB.CP._label(r)} is {rows} x {width} {r.get('dtype')} over {r.get('element_range')}"
    for e, got in sorted(by_step.items()):
        if got != Counter(KINDS):
            return (f"engine step {e} claims {dict(sorted(got.items()))}; the tap takes one GEMM output, its quotient and its tanh per step "
                    "(one request)")
    return None


class LogitsTapSource:
    short_name = "logits_tap"

    def __init__(self, committer: Any, model: Any, ids: Sequence[Mapping[str, Any]]) -> None:
        import torch

        self.torch, self.c, self.model = torch, committer, model
        self._at: dict[int, list[Mapping[str, Any]]] = {}
        for r in ids:
            self._at.setdefault(int(r["engine_step"]), []).append(r)
        self._names = frozenset(name(r) for r in ids)
        self._handles: list[Any] = []
        self._held: list[Any] = []
        self._mode: Any = None
        self._ran: set[int] = set()
        self._chain: dict[int, dict[str, Any]] = {}
        self._step = -1
        self.stats: dict[str, Any] = {"tensors": 0, "bytes": 0, "steps": 0, "errors": 0, "seconds": 0.0}
        self.active = False

    def install(self) -> None:
        from torch.overrides import TorchFunctionMode

        mods = dict(self.model.named_modules())
        if MODULE not in mods:
            raise RuntimeError(f"logits tap: {MODULE} is not a module of the model")
        Tensor = self.torch.Tensor

        class Chain(TorchFunctionMode):
            """The forward's divisions (dividend, quotient) and its tanh of one of their quotients, as references."""

            def __init__(self) -> None:
                super().__init__()
                self.divs: list[tuple[Any, Any]] = []
                self.tanhs = 0
                self.got: tuple[Any, Any, Any] | None = None

            def __torch_function__(self, func: Any, types: Any, args: tuple = (), kwargs: dict | None = None) -> Any:
                out = func(*args, **(kwargs or {}))
                fn = getattr(func, "__name__", "")
                if fn in DIVS and args and isinstance(args[0], Tensor) and isinstance(out, Tensor):
                    self.divs.append((args[0], out))
                elif fn == "tanh" and args:
                    self.tanhs += 1
                    self.got = next(((a, q, out) for a, q in self.divs if q is args[0]), None)
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

    def on_step_begin(self) -> None:
        self._held.clear()
        self._ran.clear()
        self._chain.clear()

    def error(self, msg: str) -> None:
        self.stats["errors"] += 1
        self.c.errors.append(msg[:300])

    def _pre(self, _module: Any, _args: tuple) -> None:
        job = self.c._cur
        e = int(job.step) if job is not None else None
        if e is None or e not in self._at or self._mode is not None:
            return None
        if e in self._ran:
            self.error(f"logits tap: {MODULE} ran twice at engine step {e}; the Program has one body")
            return None
        self._ran.add(e)
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
        self._chain[self._step] = dict(zip(KINDS, mode.got))
        return None

    def on_step_end(self, job: Any) -> None:
        """Commit the step's claimed identities, in the manifest's order, from the served tensors."""
        e = int(job.step)
        ids = self._at.get(e)
        if not ids:
            return
        t0 = time.perf_counter()
        got = self._chain.get(e)
        if got is None:
            if e not in self._ran:
                self.error(f"logits tap: {MODULE} did not run at engine step {e}; the manifest names its call boundaries there")
            return
        torch, words = self.torch, []
        for r in ids:
            t, rows, width = got[kind(r)], int(r["rows"]), int(r["width"])
            dt = getattr(torch, WORD_DTYPES[str(r["dtype"])][1])
            if t.dtype != dt or t.numel() != rows * width or (t.dim() and int(t.shape[-1]) != width):
                self.error(f"logits tap: {CB.CP._label(r)} is {tuple(t.shape)} {t.dtype} served; the identity is {rows} x {width} "
                           f"{r['dtype']}: nothing of the step is committed")
                return
            words.append((name(r), t.contiguous().view(rows, width)))
        n = 0
        for nm, v in words:
            self._held.append(v)
            self.c._acquire(nm, v, job)
            n += int(v.numel()) * int(v.element_size())
        self.stats["tensors"] += len(words)
        self.stats["steps"] += 1
        self.stats["bytes"] += n
        self.stats["seconds"] += time.perf_counter() - t0
        spans = getattr(self.c, "spans", None)
        if spans is not None:
            spans.add("acquisition", time.perf_counter() - t0, step=e, nbytes=n)

    def targets_at(self, engine_step: int) -> dict[str, int]:
        return {name(r): 1 for r in self._at.get(int(engine_step), ())}

    def target_names(self) -> frozenset[str]:
        return self._names

    split_step_names = CB.CallBoundarySource.split_step_names

    def extra_step_bytes(self, envelope: Mapping[str, Any], model: Any) -> dict[str, int]:
        per_seq = max((sum(int(r["rows"]) * int(r["width"]) * np.dtype(WORD_DTYPES[str(r["dtype"])][0]).itemsize for r in rs)
                       for rs in self._at.values()), default=0)
        return {"per_token_bytes": 0, "per_seq_bytes": per_seq, "tensors_per_step": max((len(rs) for rs in self._at.values()), default=0)}

    def declaration_classes(self) -> list[dict]:
        return [{"class": FAMILY, "how": "bound-as-leaf", "rule": None, "count": self.stats["tensors"], "bytes": self.stats["bytes"],
                 "tensors": self.stats["tensors"], "steps": self.stats["steps"], "errors": self.stats["errors"],
                 "evidence": (f"logits tap: the served `<fx node>/out` of {len(self._names)} call boundaries of {MODULE} (the lm_head GEMM's "
                              "output as the soft cap's division reads it, the quotient, its tanh), committed as the runtime computed them; the "
                              "replay's openings of the producing Calls check them against the Program")}]


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
