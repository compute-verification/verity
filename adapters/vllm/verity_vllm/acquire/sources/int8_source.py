"""The exact-int8 profile's quantizers (`engine.int8_ops`) as an acquisition source: the values the served linear and attention exchange
inside one module body, committed as the ops computed them.  What it taps is read from the row's capture plan
(`check.capture_plan.of_build`'s classes), never from a request's rows:

- `verity_int8::row_quant` inside a linear module: its (x_q, s_a) are the plan's `call_boundary` classes `<linear>/<op>` of an
  `Int8RowQuant` Definition, members `0` and `1` (read only by the linear's `ScaledMmS8_v1`), claimed here (`claim`) so the
  call-boundary source never evaluates them on the host.  They are committed as `<linear>/<op>/0` and `/1` at every engine step: the
  plan names the op, and the fx node's count over a request (`row_quant_3`) is the request's, which a served step batching several
  requests has none of.  A module has one such op.
- `verity_int8::kv_quant_batch` inside an attention module, or `verity_int8::kv_quant_window_batch` under a credited cache read:
  each request's (K', V', s_v, f32(V'_0)) per KV head, its slices of the batch's results (K'[r, :, :s]
  and so on, s its window), are the plan's `module_output` classes of an `Int8KvQuant` Definition at `<attention>`, its ports 0..3
  (the member's last dotted part), each over every KV head; the requests of a step are committed in batch order, concatenated (a
  request with no window, none).  They are committed as `<attention>/kv_quant.<port>`, labels this source declares as those ports
  (`ports`), so the binder binds them by structure whatever the plan's member spelling (`<hint>.<port>`), and the attention's own
  return binds its one remaining member.

Attached whenever the plan names either (`taps.attach`); no kernel changes: the wrappers run the installed ops unchanged
and acquire their results on their device.
"""
from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from typing import Any

from verity_vllm.records.capture import CALL_BOUNDARY, MODULE_OUTPUT
from verity_vllm.log import init_logger

logger = init_logger(__name__)

SHORT_NAME = "int8"
#: the Definitions, at any version, whose results this source commits
QUANT, KV = "Int8RowQuant", "Int8KvQuant"
#: Int8RowQuant's outputs (x_q, s_a), as the plan's members name them
QUANT_MEMBERS = ("0", "1")
#: Int8KvQuant's ports (K', V', s_v, f32(V'_0)) and the labels they are committed under (never a bare digit: the binder reads
#: `<module>/<k>` as slot k of the module's return)
KV_PORTS = ("0", "1", "2", "3")
KV_LABELS = tuple(f"kv_quant.{p}" for p in KV_PORTS)
KV_CLASS = "int8_kv_quant"
#: the ops whose results hold each request's Int8KvQuant, by the position of their `lens` argument: from the pages
#: (`paged_attention`) or a credited read's windows
KV_OPS = {"kv_quant_batch": 3, "kv_quant_window_batch": 3}


def _of(c: Mapping[str, Any], definition: str) -> bool:
    """Whether plan class ``c``'s rows are ``definition``'s (at any version); a class of it and of another Definition is refused."""
    mine = [str(d).rsplit("_v", 1)[0] == definition for d in c["definitions"]]
    if any(mine) and not all(mine):
        raise ValueError(f"int8: {c['op_path']}/{c['member']} is {definition}'s and another Definition's ({c['definitions']}): one "
                         "tap commits one of them")
    return any(mine)


def claim(c: Mapping[str, Any]) -> str | None:
    """`call_boundary_source.CLAIMS`: a plan class that is an Int8RowQuant call boundary is this source's; a row manifest's
    identity (a TP rank's rows, which carry no capture class) is not."""
    return SHORT_NAME if c.get("capture") == CALL_BOUNDARY and _of(c, QUANT) else None


class Sites(dict):
    """(engine step, linear module) -> the op path `<module>/<op>` its Int8RowQuant outputs are committed under, the same at
    every engine step: read with ``get`` and ``in``; its keys are step 0's."""

    def __init__(self, ops: Mapping[str, str]) -> None:
        super().__init__({(0, m): op for m, op in ops.items()})
        self.ops = dict(ops)

    def get(self, key: Any, default: Any = None) -> Any:
        return self.ops.get(key[1], default)

    def __contains__(self, key: Any) -> bool:
        return key[1] in self.ops


def quant_sites(plan: Sequence[Mapping[str, Any]]) -> Sites:
    """The plan's Int8RowQuant call boundaries, by linear module (`Sites`).  Refused by name: a member that is not one of its
    outputs (`QUANT_MEMBERS`), and a module with two such ops (the wrapper tells a module's calls apart by nothing)."""
    ops: dict[str, str] = {}
    for c in plan:
        if not claim(c):
            continue
        op, member = str(c["op_path"]), str(c["member"])
        if member not in QUANT_MEMBERS:
            raise ValueError(f"int8: {op}/{member}: an Int8RowQuant call boundary's member is its output 0 (the codes) or 1 (the "
                             "scale)")
        module = op.rsplit("/", 1)[0]
        if ops.setdefault(module, op) != op:
            raise ValueError(f"int8: {module} has two Int8RowQuant ops, {ops[module]} and {op}: the source commits one a module")
    return Sites(ops)


def kv_modules(plan: Sequence[Mapping[str, Any]]) -> list[str]:
    """The attention modules whose Int8KvQuant results the plan captures (`module_output` classes of its ports 0..3)."""
    return sorted({str(c["op_path"]) for c in plan
                   if c["capture"] == MODULE_OUTPUT and _of(c, KV) and str(c["member"]).rsplit(".", 1)[-1] in KV_PORTS})


class Int8Source:
    short_name = SHORT_NAME

    def __init__(self, committer: Any, model: Any, quant: Mapping[tuple[int, str], str], kv: list[str]) -> None:
        """``quant``: (engine step, module) -> its op path, read with ``get`` at each step (`Sites`: the same at every step)."""
        import torch

        if not quant and not kv:
            raise ValueError("int8: no quantizer to tap")
        self.torch, self.c, self.model = torch, committer, model
        self.quant, self.kv = quant, list(kv)
        self.modules = sorted({m for _e, m in self.quant} | set(self.kv))
        self._names = frozenset(f"{op}/{m}" for op in self.quant.values() for m in QUANT_MEMBERS)
        self._hooks: list[Any] = []
        self._handles: list[Any] = []
        self._scope: list[str] = []
        self._held: list[Any] = []
        self._seen: set[tuple[int, str]] = set()
        self._kv_out: dict[str, list[tuple]] = {}
        self.stats: dict[str, Any] = {"tensors": 0, "bytes": 0, "quant_tensors": 0, "quant_bytes": 0, "kv_tensors": 0, "kv_bytes": 0,
                                      "quant_sites": len(self.quant), "kv_modules": len(self.kv), "errors": 0}
        self.active = False

    def install(self) -> None:
        from verity_vllm.engine import hooks, int8_ops

        mods = dict(self.model.named_modules())
        missing = sorted(set(self.modules) - set(mods))
        if missing:
            raise RuntimeError(f"int8: the capture plan's modules are not modules of the model: {missing[:4]}")
        for op in self.modules:
            self._handles.append(mods[op].register_forward_pre_hook(self._enter(op)))
            self._handles.append(mods[op].register_forward_hook(self._leave(op), always_call=True))
        if self.quant:
            self._hooks.append(hooks.patch(int8_ops, "row_quant", self._wrap_quant(int8_ops.row_quant)))
        if self.kv:
            for name, lens_at in KV_OPS.items():
                self._hooks.append(hooks.patch(int8_ops, name, self._wrap_kv(getattr(int8_ops, name), lens_at)))
        self.active = True

    def uninstall(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []
        for p in reversed(self._hooks):
            p.uninstall()
        self._hooks = []
        self._scope.clear()
        self.on_step_begin()
        self.active = False

    def on_step_begin(self) -> None:
        self._held.clear()
        self._seen.clear()
        self._kv_out.clear()

    def error(self, msg: str) -> None:
        self.stats["errors"] += 1
        self.c.errors.append(msg[:300])

    def _enter(self, op: str) -> Any:
        def pre(_module: Any, _args: tuple) -> None:
            self._scope.append(op)
            if op in self.kv:
                self._kv_out[op] = []

        return pre

    def _leave(self, op: str) -> Any:
        def post(_module: Any, _args: tuple, _out: Any) -> None:
            if self._scope and self._scope[-1] == op:
                self._scope.pop()
            outs = self._kv_out.pop(op, None)
            job = self.c._cur
            if outs and job is not None:
                self._commit_kv(op, outs, job)

        return post

    def _take(self, name: str, t: Any, job: Any, kind: str) -> None:
        self._held.append(t)
        self.c._acquire(name, t, job)
        n = int(t.numel()) * t.element_size()
        self.stats["tensors"] += 1
        self.stats["bytes"] += n
        self.stats[f"{kind}_tensors"] += 1
        self.stats[f"{kind}_bytes"] += n

    def _wrap_quant(self, orig: Any) -> Any:
        me = self

        def row_quant(x: Any) -> Any:
            xq, s_a = out = orig(x)
            job = me.c._cur
            if job is not None and me._scope:
                key = (int(job.step), me._scope[-1])
                op = me.quant.get(key)
                if op is not None:
                    if key in me._seen:
                        me.error(f"int8: {key[1]} ran row_quant twice at engine step {key[0]}; the Program has one Int8RowQuant site")
                    else:
                        me._seen.add(key)
                        t0 = time.perf_counter()
                        me._take(f"{op}/0", xq.contiguous(), job, "quant")
                        me._take(f"{op}/1", s_a.contiguous(), job, "quant")
                        me._span(job, t0, int(xq.numel()) + 4 * int(s_a.numel()))
            return out

        return row_quant

    def _wrap_kv(self, orig: Any, lens_at: int) -> Any:
        me = self

        def kv_quant(*args: Any) -> Any:
            out = orig(*args)
            if me.c._cur is not None and me._scope and me._scope[-1] in me._kv_out:
                kq, vq, sv, v0 = out
                me._kv_out[me._scope[-1]].extend((kq[r, :, :s], vq[r, :, :s], sv[r], v0[r])
                                                 for r, s in enumerate(args[lens_at]) if s > 0)
            return out

        return kv_quant

    def _commit_kv(self, op: str, outs: list[tuple], job: Any) -> None:
        torch, t0, n = self.torch, time.perf_counter(), 0
        for j, label in enumerate(KV_LABELS):
            parts = [o[j].contiguous().reshape(-1) for o in outs]
            t = parts[0] if len(parts) == 1 else torch.cat(parts)
            self._take(f"{op}/{label}", t, job, "kv")
            n += int(t.numel()) * t.element_size()
        self._span(job, t0, n)

    def _span(self, job: Any, t0: float, nbytes: int) -> None:
        spans = getattr(self.c, "spans", None)
        if spans is not None:
            spans.add("acquisition", time.perf_counter() - t0, step=job.step, nbytes=nbytes)

    def targets_at(self, engine_step: int) -> dict[str, int]:
        """The call-boundary names this source commits at `engine_step`, once each (the KV names are module members, rule (4)'s):
        every site's, at every step."""
        return dict.fromkeys(sorted(self._names), 1)

    def target_names(self) -> frozenset[str]:
        return self._names

    def split_step_names(self, names: Mapping[str, int], engine_step: int, padding_only: bool) -> tuple[dict[str, int], dict[str, tuple[int, int]]]:
        """X-03 for the call-boundary names, as `call_boundary_source.CallBoundarySource.split_step_names`: (the step's other names,
        name -> (committed, targets) where they differ)."""
        mine = self._names
        got = {n: c for n, c in names.items() if n in mine}
        want = self.targets_at(engine_step)
        keys = set(got) ^ set(want) if padding_only else {n for n in set(got) | set(want) if got.get(n, 0) != want.get(n, 0)}
        return {n: c for n, c in names.items() if n not in mine}, {n: (got.get(n, 0), want.get(n, 0)) for n in keys}

    def extra_step_bytes(self, envelope: Mapping[str, Any], model: Any) -> dict[str, int]:
        """Per token: x_q (K bytes) and s_a (4) of every linear; per sequence: K', V' (G x T x D bytes each) and s_v, f32(V'_0) (G x D x 4
        each) of every attention layer, T the envelope's longest window."""
        mods = dict(model.named_modules()) if model is not None else {}
        per_token = 0
        for m in {m for _e, m in self.quant}:
            w = getattr(mods.get(m), "weight", None)
            per_token += (int(w.shape[1]) if w is not None and w.dim() == 2 else 0) + 4
        cfg = getattr(model, "config", None)
        heads = int(getattr(cfg, "num_attention_heads", 0) or 0)
        G = int(getattr(cfg, "num_key_value_heads", heads) or heads)
        D = int(getattr(cfg, "head_dim", 0) or (int(getattr(cfg, "hidden_size", 0) or 0) // heads if heads else 0))
        T = int(envelope.get("max_model_len") or int(envelope.get("max_prompt") or 1) + int(envelope.get("max_new") or 1))
        per_seq = len(self.kv) * (2 * G * T * D + 8 * G * D)
        return {"per_token_bytes": per_token, "per_seq_bytes": per_seq, "tensors_per_step": 2 * len({m for _e, m in self.quant}) + 4 * len(self.kv)}

    def declaration_classes(self) -> list[dict]:
        out = []
        if self.quant:
            out.append({"class": CALL_BOUNDARY, "how": "bound-as-leaf", "rule": None, "count": self.stats["quant_tensors"], "bytes": self.stats["quant_bytes"],
                        "tensors": self.stats["quant_tensors"], "errors": self.stats["errors"],
                        "evidence": (f"int8 source: `<linear>/<op>/0|1` of {len(self.quant)} Int8RowQuant sites, the results of "
                                     "verity_int8::row_quant as the served op computed them (engine.int8_ops; no kernel change)")})
        if self.kv:
            out.append({"class": KV_CLASS, "how": "bound-as-leaf", "rule": None, "count": self.stats["kv_tensors"], "bytes": self.stats["kv_bytes"],
                        "tensors": self.stats["kv_tensors"], "errors": self.stats["errors"],
                        "evidence": (f"int8 source: `<attention>/kv_quant.0..3` of {len(self.kv)} attention modules, each request's slices of "
                                     "verity_int8::kv_quant_batch (or kv_quant_window_batch under a credited read) per KV head as the served op "
                                     "computed them (engine.int8_ops; no kernel change)")})
        return out


def attach(com: Any, enabled: bool, plan: Sequence[Mapping[str, Any]], required_classes: tuple[str, ...], *,
           explicit_classes: bool = False, model: Any = None) -> tuple[tuple[str, ...], Int8Source | None]:
    """Attach the source to `com` when `enabled`; the capture plan's classes (`capture_plan.of_build`) must name an Int8RowQuant
    call boundary or an Int8KvQuant output, else the Commit is refused (exit 3).  Returns the required classes with `call_boundary`
    added (when the plan names quantizer sites) unless they were given explicitly."""
    if not enabled:
        return required_classes, None
    quant, kv = quant_sites(plan), kv_modules(plan)
    if not quant and not kv:
        logger.error("[delta] REFUSE: the int8 source but the capture plan names no Int8RowQuant call boundary and no Int8KvQuant "
                     "output")
        raise SystemExit(3)
    src = Int8Source(com, model if model is not None else getattr(com, "model", None), quant, kv)
    com.attach_source(src)
    logger.info("[delta] int8 source attached: %d row_quant sites, %d kv_quant modules -> committer %s", len(quant), len(kv), com.name)
    if quant and not explicit_classes and CALL_BOUNDARY not in required_classes:
        return (*required_classes, CALL_BOUNDARY), src
    return required_classes, src


def applies(plan: Sequence[Mapping[str, Any]]) -> bool:
    return bool(quant_sites(plan) or kv_modules(plan))


__all__ = ["Int8Source", "SHORT_NAME", "Sites", "claim", "quant_sites", "kv_modules", "applies", "attach"]
