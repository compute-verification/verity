"""Production-style acquisition of the naturally-materialized interface tensors of the pinned vLLM Llama model (EXPERIMENTAL,
CMT-1): plain `nn.Module` forward hooks copy each Instance output into a preallocated device buffer (one strided `copy_` per
tensor, same CUDA stream, no host sync), and the per-step stream is committed at the step boundary (the sampler's hook).

NOT the capture observer (no TorchDispatchMode, no Triton hook, no snapshots): the model runs untouched; the hooks see the
same output tensors the next kernel consumes.  In-place consumers (RoPE rotates q/k inside the qkv GEMM output; the fused
norm overwrites its input and the residual) are why every value is copied at its own hook, in stream order, before any
later kernel can mutate it -- the "copy" acquisition mode is the reference; `mode="refs"` (hold references, one gather at
step end) is only valid for tensors nobody mutates and is checked against the copy mode's root.

Manifest classes acquired (ids of `out/gen/r9/cmt-ref/interface_manifest.json`) and the module whose output carries them:

    embedding1/out/gatherbf16x49152_v1/gatherbf16x49152        model.embed_tokens                       [T, 576] bf16
    rmsnormtriton1/out/zip_f2fpbf16_v1_f32mul_v1/f2fpbf16      model.layers.0.input_layernorm (no residual)  [T, 576]
    rmsnormfusedcuda2/out/zip_f2fpbf16_v1_f32mul_v1/f2fpbf16   every other RMSNorm call: normalized output     [T, 576]
    rmsnormfusedcuda2/out/bf16addf2fp_v1/bf16addf2fp           every other RMSNorm call: residual output       [T, 576]
    gemm1/out/gemmcoordinate_v1/f2fpbf16                       qkv_proj [T,960] o_proj [T,576] gate_up_proj [T,3072] down_proj [T,576] lm_head [S,49152]
    rope1/out/{ropeout,ropeoutadd}  (both halves)              self_attn.rotary_emb -> (q [T,576], k [T,192]) rotated
    attention3/out/f2fpbf16_v1/f2fpbf16                        self_attn.attn                            [T, 576]
    silumul1/out/silumulbf16_v1/silumulbf16                    mlp.act_fn                                [T, 1536]
    tokenselect1/out/tokenselect_v1/selecti32                  runner.sampler -> sampled_token_ids (i64 -> i32, exact)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

import torch

from .positions import StepLayout

C_EMB = "embedding1/out/gatherbf16x49152_v1/gatherbf16x49152"
C_NORM_T = "rmsnormtriton1/out/zip_f2fpbf16_v1_f32mul_v1/f2fpbf16"
C_NORM_F = "rmsnormfusedcuda2/out/zip_f2fpbf16_v1_f32mul_v1/f2fpbf16"
C_RESID = "rmsnormfusedcuda2/out/bf16addf2fp_v1/bf16addf2fp"
C_GEMM = "gemm1/out/gemmcoordinate_v1/f2fpbf16"
C_ROPE = "rope1/out/rotated(ropeout+ropeoutadd)"
C_ATTN = "attention3/out/f2fpbf16_v1/f2fpbf16"
C_SILU = "silumul1/out/silumulbf16_v1/silumulbf16"
C_TOK = "tokenselect1/out/tokenselect_v1/selecti32"
#: manifest ids covered by C_ROPE (one rotated tensor = both output primitives, by position)
ROPE_MANIFEST_IDS = ("rope1/out/ropeout_v1/ropeout", "rope1/out/ropeoutadd_v1/ropeoutadd")

DTYPE_NAME = {torch.bfloat16: "bf16", torch.float16: "f16", torch.float32: "f32", torch.int32: "i32", torch.int64: "i64", torch.uint8: "u8"}


def _first_tensor(out: Any) -> torch.Tensor:
    if isinstance(out, torch.Tensor):
        return out
    if isinstance(out, (tuple, list)):
        for x in out:
            if isinstance(x, torch.Tensor):
                return x
    raise TypeError(f"no tensor in module output {type(out)}")


class _SamplerProxy:
    """The pinned runner's `Sampler` (`vllm/v1/worker/gpu/sample/sampler.py`) is a plain class with `__call__`, not an nn.Module:
    wrap it so post-call hooks fire (attribute access is delegated; `runner.sampler.<state>` keeps working)."""

    def __init__(self, inner: Any) -> None:
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_hooks", [])

    def __call__(self, *a, **k):
        out = self._inner(*a, **k)
        for fn in list(self._hooks):
            fn(self._inner, a, out)
        return out

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_inner"), name)

    def __setattr__(self, name, value):
        setattr(self._inner, name, value)


class _ProxyHandle:
    def __init__(self, runner, proxy, fn) -> None:
        self.runner, self.proxy, self.fn = runner, proxy, fn

    def remove(self) -> None:
        hooks = object.__getattribute__(self.proxy, "_hooks")
        if self.fn in hooks:
            hooks.remove(self.fn)
        if not hooks and getattr(self.runner, "sampler", None) is self.proxy:
            self.runner.sampler = object.__getattribute__(self.proxy, "_inner")      # last hook gone: unwrap


def hook_sampler(runner: Any, fn: Callable[[Any, Any, Any], None]):
    """Register `fn(module, inputs, output)` after every sampler call; returns a handle with `.remove()`."""
    sampler = getattr(runner, "sampler", None)
    if sampler is None:
        raise RuntimeError("runner has no .sampler")
    if hasattr(sampler, "register_forward_hook"):
        return sampler.register_forward_hook(fn)
    if not isinstance(sampler, _SamplerProxy):
        sampler = _SamplerProxy(sampler)
        runner.sampler = sampler
    object.__getattribute__(sampler, "_hooks").append(fn)
    return _ProxyHandle(runner, sampler, fn)


@dataclass
class StepRecord:
    step: int
    n_tokens: int
    layout: StepLayout
    copies: int = 0
    bytes: int = 0
    t_first_hook: float = 0.0
    t_last_hook: float = 0.0
    commit: dict = field(default_factory=dict)


class Acquirer:
    """Install on (model, runner); every engine step produces a StepRecord and (optionally) a committed step."""

    def __init__(self, model: torch.nn.Module, runner: Any, *, on_step: Callable[[StepRecord, torch.Tensor], None] | None = None,
                 buffer_bytes: int = 160 << 20, device: torch.device | str = "cuda", mode: str = "copy", commit_inline: bool = True) -> None:
        if mode not in ("copy", "refs"):
            raise ValueError(mode)
        self.model, self.runner, self.on_step, self.mode, self.commit_inline = model, runner, on_step, mode, commit_inline
        self.buf = torch.zeros(buffer_bytes, dtype=torch.uint8, device=device)
        self.handles: list[Any] = []
        self.records: list[StepRecord] = []
        self.cur: StepRecord | None = None
        self.refs: list[tuple[int, torch.Tensor]] = []
        self.errors: list[str] = []
        self.step_index = 0

    # -- per tensor --------------------------------------------------------------------------------------------------------
    def _put(self, class_id: str, instance: str, t: torch.Tensor) -> None:
        rec = self.cur
        if rec is None:
            self.errors.append(f"tensor before step start: {instance}")
            return
        t = t.detach()
        if t.dtype == torch.int64:
            t = t.to(torch.int32)          # token ids: exact for ids < 2^31 (vocab 49152)
        dt = DTYPE_NAME[t.dtype]
        seg = rec.layout.add(class_id, instance, dt, tuple(t.shape))
        if seg.offset + seg.nbytes > self.buf.numel():
            raise RuntimeError("acquisition buffer too small")
        dst = self.buf[seg.offset: seg.offset + seg.nbytes].view(t.dtype).view(t.shape)
        if self.mode == "copy":
            dst.copy_(t)                   # one strided D2D copy kernel on the current stream; no sync
        else:
            self.refs.append((seg.offset, t))
        end = seg.offset + seg.nbytes
        if rec.layout.cursor > end:        # alignment gap (nbytes % 4 != 0): keep it zero so both backends hash the same bytes
            self.buf[end: rec.layout.cursor].zero_()
        rec.copies += 1
        rec.bytes += seg.nbytes
        rec.t_last_hook = time.perf_counter()

    def _start_step(self, n_tokens: int) -> None:
        self.cur = StepRecord(self.step_index, n_tokens, StepLayout(self.step_index, n_tokens))
        self.cur.t_first_hook = time.perf_counter()
        self.refs = []
        self._shared_calls = {}

    def _end_step(self) -> None:
        rec = self.cur
        if rec is None:
            return
        if self.mode == "refs":
            for off, t in self.refs:
                self.buf[off: off + t.numel() * t.element_size()].view(t.dtype).view(t.shape).copy_(t)
            self.refs = []
        rec.layout.freeze()
        if rec.layout.padded_bytes > rec.layout.stream_bytes:       # the last leaf's zero padding (the buffer is reused across steps)
            self.buf[rec.layout.stream_bytes: rec.layout.padded_bytes].zero_()
        stream = self.buf[: rec.layout.padded_bytes]
        if self.on_step is not None:
            try:
                self.on_step(rec, stream)
            except Exception as e:  # noqa: BLE001
                self.errors.append(f"on_step {rec.step}: {e!r}")
                raise
        self.records.append(rec)
        self.cur = None
        self.step_index += 1

    # -- hooks -------------------------------------------------------------------------------------------------------------
    def _hook(self, class_id: str, instance: str, pick: Callable[[Any], list[tuple[str, str, torch.Tensor]]] | None = None):
        def fn(_m, _inp, out):
            if pick is not None:
                for cid, inst, t in pick(out):
                    self._put(cid, inst, t)
            else:
                self._put(class_id, instance, _first_tensor(out))
        return fn

    def install(self) -> dict:
        m = self.model
        core = getattr(m, "model", m)
        names: dict[str, str] = {}

        def emb_hook(_m, _inp, out):
            t = _first_tensor(out)
            self._start_step(int(t.shape[0]))
            self._put(C_EMB, "model.embed_tokens", t)
        self.handles.append(core.embed_tokens.register_forward_hook(emb_hook))
        names["model.embed_tokens"] = type(core.embed_tokens).__name__

        def norm_pick(inst):
            def pick(out):
                if isinstance(out, tuple) and len(out) == 2 and isinstance(out[1], torch.Tensor):
                    return [(C_NORM_F, inst, out[0]), (C_RESID, inst + ".residual", out[1])]
                return [(C_NORM_T, inst, _first_tensor(out))]
            return pick

        def rope_pick(inst):
            def pick(out):
                q, k = out[0], out[1]
                return [(C_ROPE, inst + ".q", q), (C_ROPE, inst + ".k", k)]
            return pick

        # modules shared between layers (vLLM's `get_rope` returns one cached RotaryEmbedding for every layer): register one hook per
        # module object and name the instance by the call index within the step (execution order == layer order)
        shared: dict[int, list[str]] = {}
        self._shared_calls: dict[int, int] = {}

        def shared_hook(mod, kind: str):
            key = id(mod)

            def fn(_m, _inp, out):
                n = self._shared_calls.get(key, 0)
                self._shared_calls[key] = n + 1
                inst = shared[key][n % len(shared[key])]
                if kind == "rope":
                    for cid, i2, t in rope_pick(inst)(out):
                        self._put(cid, i2, t)
                elif kind == "norm":
                    for cid, i2, t in norm_pick(inst)(out):
                        self._put(cid, i2, t)
                else:
                    self._put(kind, inst, _first_tensor(out))
            return fn

        seen: dict[int, tuple[Any, str]] = {}
        for i, layer in enumerate(core.layers):
            p = f"model.layers.{i}"
            sa, mlp = layer.self_attn, layer.mlp
            for mod, kind, inst in ((layer.input_layernorm, "norm", f"{p}.input_layernorm"),
                                    (sa.qkv_proj, C_GEMM, f"{p}.self_attn.qkv_proj"),
                                    (sa.rotary_emb, "rope", f"{p}.self_attn.rotary_emb"),
                                    (sa.attn, C_ATTN, f"{p}.self_attn.attn"),
                                    (sa.o_proj, C_GEMM, f"{p}.self_attn.o_proj"),
                                    (layer.post_attention_layernorm, "norm", f"{p}.post_attention_layernorm"),
                                    (mlp.gate_up_proj, C_GEMM, f"{p}.mlp.gate_up_proj"),
                                    (mlp.act_fn, C_SILU, f"{p}.mlp.act_fn"),
                                    (mlp.down_proj, C_GEMM, f"{p}.mlp.down_proj")):
                names[inst] = type(mod).__name__
                shared.setdefault(id(mod), []).append(inst)
                if id(mod) not in seen:
                    seen[id(mod)] = (mod, kind)
        for key, (mod, kind) in seen.items():
            self.handles.append(mod.register_forward_hook(shared_hook(mod, kind)))
        self.shared_modules = {shared[k][0]: len(v) for k, v in shared.items() if len(v) > 1}
        self.handles.append(core.norm.register_forward_hook(self._hook(None, "model.norm", norm_pick("model.norm"))))
        names["model.norm"] = type(core.norm).__name__
        lp = getattr(m, "logits_processor", None)
        if lp is not None:
            self.handles.append(lp.register_forward_hook(self._hook(C_GEMM, "lm_head(logits_processor)")))
            names["lm_head(logits_processor)"] = type(lp).__name__
        def sampler_hook(_m, _inp, out):
            ids = getattr(out, "sampled_token_ids", None)
            if ids is None:
                ids = _first_tensor(out)
            self._put(C_TOK, "sampler", ids)
            self._end_step()
        self.handles.append(hook_sampler(self.runner, sampler_hook))
        names["sampler"] = type(getattr(self.runner.sampler, "_inner", self.runner.sampler)).__name__
        return names

    def remove(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles = []
