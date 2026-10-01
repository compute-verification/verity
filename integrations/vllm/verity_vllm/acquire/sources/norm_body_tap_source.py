"""The call boundaries of vLLM's GemmaRMSNorm bodies (`call_boundaries` identities of a `...norm` module: `weight + 1`, the residual sum,
its square, their mean, `+ eps`, its rsqrt, the normalised row, the output) as a served tap (`served_tap`).  With `--norm-body-tap 1` the
source claims them (`claiming`, `call_boundary_source.CLAIMS`): the call-boundary source does not evaluate them, and commits in their place
the tensors the eager forward computed (vLLM's native `rms_norm` / `fused_add_rms_norm`), with the names, shapes, dtypes and place in the
step stream its host words have.  An identity's role is its fx node's kind and the spec of the Program rows producing it (`ROLES`); a node
of another spec is not claimed and stays the host's.  The replay's openings of the producing Calls check the served words against the
Program; nothing here evaluates the Program.

A forward pre-hook on each claimed module enters a TorchFunctionMode that keeps every tensor the forward computes and every storage it
writes in place (`served_tap.note_writes`); the first forward hook leaves it, and the roles are found by dataflow from the forward's one `rsqrt` (`served_roles`): its argument is the sum
of a mean and a number (`variance + eps`), the mean is of a square, the square's base is the normalised input -- the residual sum when it
is an addition of two tensors -- the product of that input and the rsqrt is the normalised row, and its product with the module weight's
`.float() + 1.0` is the output.  A forward of another form (fused, compiled, a second rsqrt), or one that writes a served tensor in place
after computing it, is a collector error naming the module and step, and nothing of that module at that step is committed.  A claimed module the model lacks or without a `weight` parameter, a
manifest of several ranks or requests, and a step naming one role of a module twice are refused (exit 3).  The Build's call-boundary check and plan and the Commit take
the option, VERITY_NORM_BODY_TAP in the row's environment by default."""
from __future__ import annotations

import re
from collections import Counter
from typing import Any, Mapping, Sequence

from verity_vllm.acquire.sources import call_boundary_source as CB
from verity_vllm.acquire.sources import served_tap as ST
from verity_vllm.acquire.sources.call_boundary_plan import FAMILY
from verity_vllm.config import option
from verity_vllm.query.manifest.format import spec_family

ENV = "VERITY_NORM_BODY_TAP"
NODE = re.compile(r"(?P<module>(?:[^/]+\.)?[^./]*norm)/(?P<kind>add|mean|mul|pow|rsqrt)(?:_\d+)*")
#: (fx node kind, spec of the producing rows) -> the role of the value in the served forward
ROLES = {("add", "AddScalarBf16_v1"): "scale", ("add", "AddWidenedBf16_v1"): "resid", ("add", "AddScalarF32_v1"): "var_eps",
         ("pow", "SquareBf16_v1"): "sq", ("pow", "SquareF32_v1"): "sq", ("mean", "MeanTriton_v1"): "var", ("rsqrt", "RsqrtF32_v1"): "rstd",
         ("mul", "ScaleRowBf16_v1"): "normed", ("mul", "ScaleRowF32_v1"): "normed", ("mul", "MulVecF32_v1"): "out"}
ADDS = frozenset({"__add__", "add", "__radd__"})
MULS = frozenset({"__mul__", "mul", "__rmul__"})
POWS = frozenset({"pow", "__pow__"})
name = ST.name


def option_field() -> Any:
    """The `--norm-body-tap` option of every command that reads call boundaries (`commit`, `boundary-plan`, `gate call-boundaries`)."""
    return option("--norm-body-tap", env=ENV, type=int, default=0, metavar="0|1",
                  help="1: the norm body tap claims the call boundaries of the x * (1 + w) norm bodies and commits the served tensors "
                       "(acquire.sources.norm_body_tap_source); the row's Build check, its plan and the Commit must agree")


def role(r: Mapping[str, Any]) -> str | None:
    """The served role of identity `r` of a norm body, or None when its node is not one the tap serves."""
    m = NODE.fullmatch(str(r.get("op_path") or ""))
    return ROLES.get((m.group("kind"), spec_family(str(r.get("spec") or "")))) if m else None


def claim(r: Mapping[str, Any]) -> str | None:
    """This source's `short_name` for a rank-0 call boundary of a norm body's node with a served role, else None."""
    if r.get("family") != FAMILY or int(r.get("rank") or 0) or int(r.get("invocation") or 0) or r.get("program_op_path"):
        return None
    return NormBodyTapSource.short_name if role(r) else None


def claiming(on: bool) -> None:
    """Whether the tap claims its identities (`call_boundary_source.CLAIMS`), from a command's `--norm-body-tap`."""
    if on and claim not in CB.CLAIMS:
        CB.CLAIMS.append(claim)
    elif not on and claim in CB.CLAIMS:
        CB.CLAIMS.remove(claim)


def manifest_sites(manifest: Mapping[str, Any] | None) -> list[Mapping[str, Any]]:
    """The `call_boundaries` identities of a manifest this source commits, while it is claiming: every one its `claim` takes and no
    earlier claim does."""
    if claim not in CB.CLAIMS:
        return []
    return [r for r in (manifest or {}).get("identities") or [] if r.get("family") == FAMILY and claim(r)
            and CB.claimed(r) == NormBodyTapSource.short_name]


def refusal(manifest: Mapping[str, Any] | None, ids: Sequence[Mapping[str, Any]]) -> str | None:
    """Why the source cannot commit `ids` of `manifest`, or None."""
    ranks = int((manifest or {}).get("ranks") or 1)
    if ranks > 1:
        return f"a manifest of {ranks} ranks (the tap serves one rank)"
    requests = sorted({str(r.get("request_id")) for r in ids})
    if len(requests) > 1:
        return f"a manifest of {len(requests)} requests {requests[:4]} (the tap commits one request's rows)"
    why = ST.element_refusal(ids)
    if why:
        return why
    by: dict[tuple[int, str], Counter] = {}
    for r in ids:
        by.setdefault((int(r["engine_step"]), ST.module_of(r)), Counter())[role(r)] += 1
    for (e, m), got in sorted(by.items()):
        twice = sorted(k for k, n in got.items() if n > 1)
        if twice:
            return f"{m} at engine step {e} names the served {twice} {max(got.values())} times; its forward computes each once"
    return None


def model_refusal(model: Any, modules: Sequence[str]) -> str | None:
    """Why `modules` of `model` are not all norm modules with a `weight` parameter, or None; that the forward is vLLM's GemmaRMSNorm chain
    is checked where it runs (`served_roles`)."""
    mods = dict(model.named_modules()) if model is not None else {}
    for m in modules:
        if m not in mods:
            return f"{m} is not a module of the model"
        if "weight" not in dict(mods[m].named_parameters(recurse=False)):
            return f"{m} is a {type(mods[m]).__name__} without a weight parameter"
    return None


def served_roles(ops: Sequence[tuple[str, tuple, Any, int]], weight: Any, Tensor: Any) -> tuple[dict[str, tuple[Any, int]] | None, str | None]:
    """The served tensor of every role of one GemmaRMSNorm forward, with its mark, from its recorded ops [(function name, args, output,
    mark: the forward's in-place writes before it)], or (None, why) for a forward of another form.  A tensor's op is the first that
    output it (a no-op `.to` or an in-place write returns its input; `served_tap.ServedTap.hold` refuses the write)."""
    by: dict[int, tuple] = {}
    for o in ops:
        by.setdefault(id(o[2]), o)

    def made(t: Any, fns: frozenset) -> tuple | None:
        o = by.get(id(t)) if isinstance(t, Tensor) else None
        return o if o is not None and o[0] in fns else None

    rs = [o for o in ops if o[0] == "rsqrt"]
    if len(rs) != 1:
        return None, f"the forward ran {len(rs)} rsqrt, not one"
    rstd = rs[0]
    ve = made(rstd[1][0] if rstd[1] else None, ADDS)
    var = made(ve[1][0], frozenset({"mean"})) if ve is not None and len(ve[1]) == 2 and not isinstance(ve[1][1], Tensor) else None
    sq = made(var[1][0], POWS) if var is not None else None
    if sq is None:
        return None, "the rsqrt does not read the mean of a square plus a number (variance + eps)"
    x = sq[1][0]
    normed = [o for o in ops if o[0] in MULS and len(o[1]) == 2 and {id(a) for a in o[1]} == {id(x), id(rstd[2])}]
    out = [o for o in ops if normed and o[0] in MULS and len(o[1]) == 2 and any(a is normed[0][2] for a in o[1])]
    if len(normed) != 1 or len(out) != 1:
        return None, f"the forward multiplies the input by the rsqrt {len(normed)} times and that product {len(out)} times, not once each"
    sc = made(next(a for a in out[0][1] if a is not normed[0][2]), ADDS)
    w = made(sc[1][0], frozenset({"float", "to"})) if sc is not None and sc[1] else None
    if w is None or w[1][0] is not weight:
        return None, "the output's other factor is not the module weight's .float() + 1"
    roles = {"scale": sc, "sq": sq, "var": var, "var_eps": ve, "rstd": rstd, "normed": normed[0], "out": out[0]}
    resid = made(x, ADDS)
    if resid is not None and len(resid[1]) == 2 and all(isinstance(a, Tensor) for a in resid[1]):
        roles["resid"] = resid
    return {k: (o[2], o[3]) for k, o in roles.items()}, None


class NormBodyTapSource(ST.ServedTap):
    short_name = "norm_body_tap"
    label = "norm body tap"

    def __init__(self, committer: Any, model: Any, ids: Sequence[Mapping[str, Any]]) -> None:
        super().__init__(committer, model, ids)
        self._live: tuple[int, str, Any] | None = None

    def install(self) -> None:
        from torch.overrides import TorchFunctionMode

        mods = dict(self.model.named_modules())
        missing = sorted(set(self.modules) - set(mods))
        if missing:
            raise RuntimeError(f"norm body tap: the manifest's norm modules are not modules of the model: {missing[:4]}")
        Tensor = self.torch.Tensor

        class Capture(TorchFunctionMode):
            """Every tensor the forward computes, (function name, args, output, the in-place writes before it), and every write."""

            def __init__(self) -> None:
                super().__init__()
                self.ops: list[tuple[str, tuple, Any, int]] = []
                self.writes: list[tuple[int, str]] = []

            def __torch_function__(self, func: Any, types: Any, args: tuple = (), kwargs: dict | None = None) -> Any:
                out = func(*args, **(kwargs or {}))
                ST.note_writes(self.writes, func, args, kwargs, Tensor)
                if isinstance(out, Tensor):
                    self.ops.append((getattr(func, "__name__", ""), tuple(args), out, len(self.writes)))
                return out

        self._Capture = Capture
        for m in self.modules:
            mod = mods[m]
            self._handles += [mod.register_forward_pre_hook(self._pre(m)),
                              mod.register_forward_hook(self._post(m, mod), prepend=True, always_call=True)]
        self.active = True

    def uninstall(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []
        if self._live is not None:
            self._live[2].__exit__(None, None, None)
            self._live = None
        self.on_step_begin()
        self.active = False

    def _pre(self, m: str) -> Any:
        def pre(_module: Any, args: tuple) -> None:
            job = self.c._cur
            e = int(job.step) if job is not None else None
            if e is None or m not in self.ids_at(e) or self._live is not None or not self.ran(e, m, args):
                return None
            mode = self._Capture()
            mode.__enter__()
            self._live = (e, m, mode)
            return None

        return pre

    def _post(self, m: str, module: Any) -> Any:
        def post(_module: Any, _args: tuple, _out: Any) -> None:
            if self._live is None or self._live[1] != m:
                return None
            (e, _m, mode), self._live = self._live, None
            mode.__exit__(None, None, None)
            roles, why = served_roles(mode.ops, module.weight, self.torch.Tensor)
            if why:
                self.error(f"norm body tap: {m} at engine step {e}: {why}; nothing of it is committed")
                return None
            self.hold(e, m, {name(r): roles[role(r)] for r in self.ids_at(e)[m] if role(r) in roles}, mode.writes)
            return None

        return post

    def extra_step_bytes(self, envelope: Mapping[str, Any], model: Any) -> dict[str, int]:
        per = self.step_bytes().values()
        return {"per_token_bytes": max((-(-b // max(1, rows)) for b, rows, _n in per), default=0), "per_seq_bytes": 0,
                "tensors_per_step": max((n for _b, _r, n in per), default=0)}

    def served(self) -> dict[str, Any]:
        return self.summary(f"norm body tap: the served values of {len(self._names)} call boundaries of {len(self.modules)} norm bodies "
                            "(weight + 1, the residual sum, its square, their mean, + eps, rsqrt, the normalised row, the output)")


def attach(com: Any, manifest: Mapping[str, Any] | None, required_classes: tuple[str, ...], *, explicit_classes: bool = False,
           model: Any = None) -> tuple[tuple[str, ...], NormBodyTapSource | None]:
    """Attach the source to `com` when the manifest names identities it claims; one it cannot commit (`refusal`, `model_refusal`)
    refuses the Commit (exit 3).  Returns the required classes with `call_boundaries` added unless they were given explicitly."""
    ids = manifest_sites(manifest)
    if not ids:
        return required_classes, None
    model = model if model is not None else getattr(com, "model", None)
    why = refusal(manifest, ids) or model_refusal(model, sorted({ST.module_of(r) for r in ids}))
    if why:
        print(f"[delta] REFUSE: norm body tap: {why}", flush=True)
        raise SystemExit(3)
    src = NormBodyTapSource(com, model, ids)
    com.attach_source(src)
    print(f"[delta] norm body tap attached: {len(ids)} call boundaries of {len(src.modules)} norm bodies in {len(src._at)} engine "
          f"steps, the served tensors -> committer {com.name}", flush=True)
    if not explicit_classes and FAMILY not in required_classes:
        return (*required_classes, FAMILY), src
    return required_classes, src


__all__ = ["ENV", "ROLES", "NormBodyTapSource", "option_field", "role", "claim", "claiming", "manifest_sites", "refusal",
           "model_refusal", "served_roles", "attach"]
