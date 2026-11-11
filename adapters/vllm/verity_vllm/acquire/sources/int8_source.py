"""The exact-int8 profile's quantizers (`engine.int8_ops`) as an acquisition source: the values the served linear and attention exchange
inside one module body, committed as the ops computed them.

- `verity_int8::row_quant` inside a linear module: its (x_q, s_a) are the `call_boundaries` identities `<linear>/<fx node>/0` and `/1`
  (`Int8RowQuant_v1`, read only by the linear's `ScaledMmS8_v1`), claimed here (`claim`) so the call-boundary source never evaluates
  them on the host.  The fx node's name counts on through the run, so the names come from the manifest: per (engine step, module) its
  one Int8RowQuant_v1 site.
- `verity_int8::kv_quant_batch` inside an attention module, or `verity_int8::kv_quant_window_batch` under a credited cache read
  (`engine.int8_window`): each request's (K', V', s_v, f32(V'_0)) per KV head, its slices of the batch's results (K'[r, :, :s]
  and so on, s its window), are the `instance_outputs` identities of `Int8KvQuant_v1`'s ports 0..3 at `<attention>`, each over
  every KV head; the requests of a step are committed in batch order, concatenated (a request with no window, none).  They are committed as `<attention>/kv_quant.<port>`, labels this source declares as those ports (`ports`), so the
  binder binds them by structure whatever the manifest's member spelling (`<hint>.<port>`), and the attention's own return binds its
  one remaining member.

Attached whenever the required manifest names either (`taps.attach`); no kernel changes: the wrappers run the installed ops unchanged
and acquire their results on their device.
"""
from __future__ import annotations

import time
from typing import Any, Mapping

from verity_vllm.acquire.sources.call_boundary_plan import FAMILY
from verity_vllm.query.manifest.bind import Port

SHORT_NAME = "int8"
QUANT_SPEC = "Int8RowQuant_v1"
KV_SPEC = "Int8KvQuant_v1"
#: Int8KvQuant_v1's ports (K', V', s_v, f32(V'_0)) and the labels they are committed under (never a bare digit: the binder reads
#: `<module>/<k>` as slot k of the module's return)
KV_PORTS = ("0", "1", "2", "3")
KV_LABELS = tuple(f"kv_quant.{p}" for p in KV_PORTS)
KV_CLASS = "int8_kv_quant"
#: the ops whose results hold each request's Int8KvQuant_v1, by the position of their `lens` argument: from the pages
#: (`paged_attention`) or a credited read's windows
KV_OPS = {"kv_quant_batch": 3, "kv_quant_window_batch": 3}


def _spec(r: Mapping[str, Any]) -> str:
    return str(r.get("spec") or "").split("{", 1)[0]


def claim(r: Mapping[str, Any]) -> str | None:
    """`call_boundary_source.CLAIMS`: an Int8RowQuant_v1 call boundary is this source's."""
    return SHORT_NAME if r.get("family") == FAMILY and _spec(r) == QUANT_SPEC else None


def quant_sites(manifest: Mapping[str, Any] | None) -> dict[tuple[int, str], str]:
    """(engine step, linear module) -> the op path `<module>/<fx node>` of its Int8RowQuant_v1 call boundaries."""
    out: dict[tuple[int, str], str] = {}
    for r in (manifest or {}).get("identities") or []:
        if claim(r):
            op = str(r["op_path"])
            out[(int(r.get("engine_step", r.get("step", 0))), op.split("/", 1)[0])] = op
    return out


def kv_modules(manifest: Mapping[str, Any] | None) -> list[str]:
    """The attention modules whose Int8KvQuant_v1 results the manifest requires (`instance_outputs` rows of its ports 0..3)."""
    return sorted({str(r["op_path"]) for r in (manifest or {}).get("identities") or []
                   if r.get("family") == "instance_outputs" and _spec(r) == KV_SPEC
                   and str(r.get("port", r.get("output_member"))) in KV_PORTS})


class Int8Source:
    short_name = SHORT_NAME

    def __init__(self, committer: Any, model: Any, quant: Mapping[tuple[int, str], str], kv: list[str]) -> None:
        import torch

        if not quant and not kv:
            raise ValueError("int8: no quantizer to tap")
        self.torch, self.c, self.model = torch, committer, model
        self.quant, self.kv = dict(quant), list(kv)
        self.modules = sorted({m for _e, m in self.quant} | set(self.kv))
        self._at: dict[int, dict[str, int]] = {}
        for (e, _m), op in self.quant.items():
            self._at.setdefault(e, {}).update({f"{op}/0": 1, f"{op}/1": 1})
        self._names = frozenset(n for d in self._at.values() for n in d)
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
            raise RuntimeError(f"int8: the manifest's modules are not modules of the model: {missing[:4]}")
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
                        me.error(f"int8: {key[1]} ran row_quant twice at engine step {key[0]}; the Program has one Int8RowQuant_v1 site")
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
        """The call-boundary names this source commits at `engine_step`, once each (the KV names are module members, rule (4)'s)."""
        return dict(self._at.get(int(engine_step), {}))

    def target_names(self) -> frozenset[str]:
        return self._names

    def ports(self) -> dict[str, Port]:
        """`bind.Binder`'s declared labels: each KV label, its Int8KvQuant_v1 port."""
        return {label: Port(frozenset({KV_SPEC}), port) for label, port in zip(KV_LABELS, KV_PORTS)} if self.kv else {}

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
            out.append({"class": FAMILY, "how": "bound-as-leaf", "rule": None, "count": self.stats["quant_tensors"], "bytes": self.stats["quant_bytes"],
                        "tensors": self.stats["quant_tensors"], "errors": self.stats["errors"],
                        "evidence": (f"int8 source: `<linear>/<fx node>/0|1` of {len(self.quant)} Int8RowQuant_v1 sites, the results of "
                                     "verity_int8::row_quant as the served op computed them (engine.int8_ops; no kernel change)")})
        if self.kv:
            out.append({"class": KV_CLASS, "how": "bound-as-leaf", "rule": None, "count": self.stats["kv_tensors"], "bytes": self.stats["kv_bytes"],
                        "tensors": self.stats["kv_tensors"], "errors": self.stats["errors"],
                        "evidence": (f"int8 source: `<attention>/kv_quant.0..3` of {len(self.kv)} attention modules, each request's slices of "
                                     "verity_int8::kv_quant_batch (or kv_quant_window_batch under a credited read) per KV head as the served op "
                                     "computed them (engine.int8_ops; no kernel change)")})
        return out


def attach(com: Any, enabled: bool, manifest: Mapping[str, Any] | None, required_classes: tuple[str, ...], *,
           explicit_classes: bool = False, model: Any = None) -> tuple[tuple[str, ...], Int8Source | None]:
    """Attach the source to `com` when `enabled`; the required manifest must name an Int8RowQuant_v1 call boundary or an
    Int8KvQuant_v1 output, else the Commit is refused (exit 3).  Returns the required classes with `call_boundaries` added (when the
    manifest names quantizer sites) unless they were given explicitly."""
    if not enabled:
        return required_classes, None
    quant, kv = quant_sites(manifest), kv_modules(manifest)
    if not quant and not kv:
        print("[delta] REFUSE: the int8 source but the required-value manifest names no Int8RowQuant_v1 call boundary and no "
              "Int8KvQuant_v1 output", flush=True)
        raise SystemExit(3)
    src = Int8Source(com, model if model is not None else getattr(com, "model", None), quant, kv)
    com.attach_source(src)
    print(f"[delta] int8 source attached: {len(quant)} row_quant sites, {len(kv)} kv_quant modules -> committer {com.name}", flush=True)
    if quant and not explicit_classes and FAMILY not in required_classes:
        return (*required_classes, FAMILY), src
    return required_classes, src


def applies(manifest: Mapping[str, Any] | None) -> bool:
    return bool(quant_sites(manifest) or kv_modules(manifest))


__all__ = ["Int8Source", "SHORT_NAME", "claim", "quant_sites", "kv_modules", "applies", "attach"]
