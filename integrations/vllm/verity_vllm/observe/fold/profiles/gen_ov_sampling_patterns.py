"""Recognizer for vLLM d9105ea80's Gumbel-max sampler at temperature 1 (no top-k / top-p): the served
token of every sampled row becomes `Compute(GumbelArgmaxBf16_v1{V, SEED, POS}, rows=1, (PerRow(logits[r]),), (PerRow(out[r]),))`
with SEED the request record's seed (children of an n>1 request: parent seed + index, the fork fact the adapter records)
and POS the row's StepStart position; rows of a greedy request (per-request temperature 0, `mixed` workloads) become the
registered `TokenSelect_v1{V}`.  Same three-stage shape as `patterns.TokenSelectGumbelTwoStage` (launch -> aten.argmax over
the block maxima -> aten.gather), which stays first in line for greedy headers (this pattern returns None for them).

What is deliberately Unsupported here (each is a contract finding):
  * a sampled row whose request record has no seed (`seed: null`): the kernel consumed a host draw (`np.random` in
    vLLM's `SamplingStates`) that no trusted fact explains -> "unseeded sampled row";
  * float32 logits (the `_temperature_kernel` quotient / a top-k or top-p masked row): those need TemperatureDiv / TopKMask /
    TopPMask definitions that are not in the registry -> "float32 logits";
  * APPLY_TEMPERATURE / IS_DRAFTING / PER_TOKEN_COL / USE_FP64 launches, logits caches, non-pinned BLOCK_SIZE.

Note on provenance: SEED is a *static* of the specialization, taken from the run header's request record.  Nothing in the
fold checks that the device buffer `seeds_ptr[expanded_idx_mapping_ptr[r]]` the kernel read holds that value -- the only
downstream check is Tier A (the served token must equal the definition's output, which fails for a wrong seed with
probability ~1).  A recorder-level `Root(role="sampling_seed")` per request with a validator against the request record is
the PROPOSED closing of that gap (REPORT.md, "RNG/root/state contract").
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from verity.ml.circuits.defs import bind
from verity.ml.definitions.registry import b1
import numpy as np

from verity.ml.definitions.registry.sampling import GumbelArgmaxBf16, GumbelArgmaxTempBf16

from verity_vllm.observe.events import ViewDesc
from verity_vllm.observe.fold.patterns import SCRATCH_TOKEN_SELECT, ComputeGroup, _fmt, bound_args, first, hint_of, is_int, sub_rows, triton_name, unsupported, widened_source
from verity_vllm.observe.fold.resolver import Compute, PerRow, Resolution, StepContext, Transparent
from verity_vllm.observe.fold.tree import TreeNode

SCRATCH_OV = "ov_sampling_token_sample"


def header_sampling(ctx: StepContext) -> dict:
    return dict((ctx.header.workload or {}).get("sampling") or (ctx.header.profile or {}).get("sampling") or {})


def request_seeds(ctx: StepContext) -> dict[str, int | None]:
    """request id -> seed of the run header's request record (children: the recorded fork fact parent seed + index)."""
    wl = ctx.header.workload or {}
    seeds: dict[str, int | None] = {r["request_id"]: (None if r.get("seed") is None else int(r["seed"])) for r in wl.get("requests", [])}
    for c in wl.get("children", []) or []:
        seeds[c["request_id"]] = None if c.get("seed") is None else int(c["seed"])
    return seeds


def request_temperatures(ctx: StepContext) -> tuple[float, dict[str, float]]:
    s = header_sampling(ctx)
    per = {p["request_id"]: float(p["temperature"]) for p in s.get("per_request", []) or []}
    return float(s.get("temperature", 0.0)), per


def sampling_filters_off(s: dict) -> str | None:
    if s.get("top_k") not in (None, 0, -1) or s.get("top_p") not in (None, 1, 1.0):
        return f"top_k/top_p ({s.get('top_k')!r}/{s.get('top_p')!r})"
    for p in s.get("per_request", []) or []:
        if p.get("top_k") not in (None, 0, -1) or p.get("top_p") not in (None, 1, 1.0):
            return f"per-request top_k/top_p ({p.get('top_k')!r}/{p.get('top_p')!r}) for {p.get('request_id')}"
    return None


@dataclass
class TokenSampleGumbel:
    vocab: int | None = None
    block_size: int = 1024
    name: str = "ov_sampling.gumbel_token_sample"

    def match(self, node: TreeNode, ctx: StepContext) -> Resolution | None:
        if node.is_triton:
            k = triton_name(node)
            if k == "_gumbel_sample_kernel":
                return self._stage1(node, ctx)
            if k == "_temperature_kernel":
                return self._temperature(node, ctx)
            return None
        op = node.op.split(".")
        b = ".".join(op[:2]) if len(op) >= 2 else node.op
        if b == "aten.argmax":
            return self._stage2(node, ctx)
        if b == "aten.gather":
            return self._stage3(node, ctx)
        if b == "aten.index" and node.phase == "sample":
            return self._state_gather(node, ctx)
        return None

    def _state_gather(self, node: TreeNode, ctx: StepContext) -> Resolution | None:
        """vLLM's top-p/top-k path gathers the per-request sampling state (`top_p[req_indices]`, `top_k[...]`: small 1-D
        float32/int32 buffers of max_num_seqs entries) with `aten.index.Tensor` in the sample phase.  These are request facts
        of the header (the workload's sampling block), not activations: a metadata noop.  Anything else is left to the pinned
        patterns (the hidden-state `logits_gather`)."""
        a = bound_args(node)
        src = first(a.get("self"), a.get("arg0"))
        if not isinstance(src, ViewDesc) or src.rank != 1 or src.dtype not in ("float32", "int32", "int64") or src.numel > 64:
            return None
        return Transparent("noop", (node.id,), hint=hint_of(node, f"per-request sampling-state gather ({src.dtype}[{src.numel}]): a request fact"),
                           pattern=self.name)

    def _state(self, ctx: StepContext) -> dict:
        return ctx.scratch.setdefault(SCRATCH_OV, {"stage1": {}, "stage2": {}, "temperature_written": set()})

    def _temperature(self, node: TreeNode, ctx: StepContext) -> Resolution:
        """`_temperature_kernel(logits f32[R,V], expanded_idx_mapping, temperature[·], ...)`: in place `logits[r] /= T[r]`
        (Triton fp32 `/` = div.full.f32, NOT IEEE division) for rows with T not in {0, 1}; rows at T in {0, 1} return early.
        A noop that claims its kernel: the quotient is *defined* by `GumbelArgmaxTempBf16_v1{.., T}` at the sampler launch
        (the row's T is a request fact of the header; a wrong T fails Tier A on the quotient).  The written allocation is
        remembered so the sampler stage knows its float32 row is the quotient, not the plain widening of the bf16 logits."""
        a = bound_args(node)
        logits = a.get("logits_ptr")
        if not isinstance(logits, ViewDesc) or logits.dtype != "float32" or logits.rank != 2:
            return unsupported(node, "_temperature_kernel logits are not a float32 [R,V] view", logits=_fmt(logits))
        self._state(ctx)["temperature_written"].add(logits.alloc)
        return Transparent("noop", (node.id,), hint=hint_of(node, "temperature: logits[r] /= T[r] in place (div.full.f32; defined by "
                                                                  "GumbelArgmaxTempBf16_v1 at the sampler launch)"),
                           kernels=(triton_name(node),), pattern=self.name)

    def _stage1(self, node: TreeNode, ctx: StepContext) -> Resolution | None:
        s = header_sampling(ctx)
        if s.get("greedy") is True:
            return None                                        # the registered greedy recognizer's case
        a = bound_args(node)
        consts = dict(node.enter.constexprs)
        flags = {k: first(consts.get(k), a.get(k)) for k in ("IS_DRAFTING", "APPLY_TEMPERATURE", "PER_TOKEN_COL", "USE_FP64")}
        if any(bool(v) for v in flags.values()):
            return unsupported(node, "_gumbel_sample_kernel with drafting / in-kernel temperature / per-token-column / fp64 flags", **flags)
        if a.get("logits_cache_ptr") is not None:
            return unsupported(node, "_gumbel_sample_kernel writing a logits cache", logits_cache=_fmt(a.get("logits_cache_ptr")))
        why = sampling_filters_off(s)
        if why is not None:
            return unsupported(node, f"sampled rows under {why}: the kernel's logits row is the -inf-masked output of "
                                     "`_topk_topp_kernel` / `_topp_sb_*` for which no TopKMask / TopPMask definition is registered")
        block = first(consts.get("BLOCK_SIZE"), a.get("BLOCK_SIZE"))
        logits = a.get("logits_ptr")
        local_argmax = a.get("local_argmax_ptr")
        local_max = a.get("local_max_ptr")
        if not all(isinstance(v, ViewDesc) for v in (logits, local_argmax, local_max)):
            return unsupported(node, "_gumbel_sample_kernel pointers not bound as expected", args={k: _fmt(v) for k, v in a.items()})
        if logits.rank != 2 or logits.strides[1] != 1:
            return unsupported(node, "logits are not a row-contiguous [R,V] view", logits=repr(logits))
        tempered = logits.dtype == "float32" and logits.alloc in self._state(ctx)["temperature_written"]
        if logits.dtype == "float32":
            narrow = widened_source(logits, ctx)
            if narrow is None:
                return unsupported(node, "float32 logits that are not an exact widening of a bf16 view: the `_temperature_kernel` quotient "
                                         "(Triton div.full.f32) / a masked row needs a TemperatureDiv / TopKMask / TopPMask definition "
                                         "that is not in the registry", logits=repr(logits))
            logits = narrow
        if logits.dtype != "bfloat16":
            return unsupported(node, "GumbelArgmaxBf16_v1 is defined over bf16 logits", dtype=logits.dtype)
        R, V = logits.shape
        if self.vocab is not None and V != self.vocab:
            return unsupported(node, "logits width is not the vocabulary", V=V, vocab=self.vocab)
        if not isinstance(block, int) or block != self.block_size:
            return unsupported(node, "BLOCK_SIZE differs from the pinned sampler block", BLOCK_SIZE=block, pinned=self.block_size)
        nb = -(-V // block)
        if local_argmax.shape != (R, nb) or local_max.shape != (R, nb) or not is_int(local_argmax) or local_max.dtype != "float32":
            return unsupported(node, "local argmax/max buffers are not int64/f32 [R, ceil(V/BLOCK)]", local_argmax=repr(local_argmax),
                               local_max=repr(local_max), R=R, NB=nb)
        rows = list(ctx.start.logits_rows)
        if len(rows) != R:
            return unsupported(node, "logits rows do not match the step's sampled rows", R=R, logits_rows=len(rows))
        seeds = request_seeds(ctx)
        T0, per_T = request_temperatures(ctx)
        specs: list[tuple[Any, str, int]] = []
        for rid, pos in rows:
            T = per_T.get(rid, T0)
            if T == 0.0:
                specs.append((bind(b1.TokenSelect, V=V), rid, int(pos)))
                continue
            if T != 1.0 and not tempered:
                return unsupported(node, "bf16 logits with a non-unit temperature: the kernel would have to divide in-kernel", request=rid, temperature=T)
            seed = seeds.get(rid)
            if seed is None:
                return unsupported(node, "unseeded sampled row: the kernel consumed a seed drawn on the host by vLLM's SamplingStates "
                                         "(`seed: null` in the request record); no trusted fact explains it -- PROPOSED root class "
                                         "`sampling-seed` (REPORT.md)", request=rid, position=int(pos))
            if T != 1.0:
                # the row is the `_temperature_kernel` quotient of the bf16 logits: T (a request fact) is a static of the definition
                t_bits = int(np.float32(T).view(np.uint32))
                specs.append((bind(GumbelArgmaxTempBf16, V=V, SEED=int(seed), POS=int(pos), T=t_bits), rid, int(pos)))
            else:
                specs.append((bind(GumbelArgmaxBf16, V=V, SEED=int(seed), POS=int(pos)), rid, int(pos)))
        st = self._state(ctx)
        st["stage1"][local_max.alloc] = {"logits": logits, "local_argmax": local_argmax, "local_max": local_max, "specs": specs,
                                          "kernels": (triton_name(node),)}
        return Transparent("noop", (node.id,), hint=hint_of(node, "gumbel_sample stage 1 (noised blockwise max/argmax)"),
                           kernels=(triton_name(node),), pattern=self.name)

    def _stage2(self, node: TreeNode, ctx: StepContext) -> Resolution | None:
        a = bound_args(node)
        x = first(a.get("self"), a.get("arg0"))
        if not isinstance(x, ViewDesc):
            return None
        st = ctx.scratch.get(SCRATCH_OV) or {}
        s1 = (st.get("stage1") or {}).get(x.alloc)
        if s1 is None or x.shape != s1["local_max"].shape:
            return None
        dim = a.get("dim", a.get("arg1"))
        outs = node.output_views()
        if dim not in (1, -1) or not outs:
            return unsupported(node, "argmax over local_max not along the block axis", dim=dim)
        out = outs[0]
        R = x.shape[0]
        if out.numel != R or not is_int(out):
            return unsupported(node, "argmax over local_max does not produce int [R]", out=repr(out))
        st["stage2"][out.alloc] = {"idx": out, **s1}
        return Transparent("noop", (node.id,), hint=hint_of(node, "gumbel_sample stage 2 (winning block)"), kernels=("ArgMaxOps",),
                           pattern=self.name)

    def _stage3(self, node: TreeNode, ctx: StepContext) -> Resolution | None:
        a = bound_args(node)
        src = first(a.get("self"), a.get("arg0"))
        idx = first(a.get("index"), a.get("arg2"))
        if not isinstance(src, ViewDesc) or not isinstance(idx, ViewDesc):
            return None
        st = ctx.scratch.get(SCRATCH_OV) or {}
        s2 = (st.get("stage2") or {}).get(idx.alloc)
        if s2 is None or src.alloc != s2["local_argmax"].alloc:
            return None
        dim = a.get("dim", a.get("arg1"))
        outs = node.output_views()
        if dim not in (1, -1) or not outs:
            return unsupported(node, "gather over local_argmax not along the block axis", dim=dim)
        logits: ViewDesc = s2["logits"]
        R, V = logits.shape
        out = outs[0]
        if out.numel != R or not is_int(out):
            return unsupported(node, "sampled ids are not an integer [R] view", out=repr(out))
        if out.rank == 1:
            out = ViewDesc(out.alloc, out.offset, (R, 1), (out.strides[0], 1), out.dtype, out.device)
        kernels = (*s2["kernels"], "ArgMaxOps", "gather_kernel")
        parts = []
        for r, (spec, rid, pos) in enumerate(s2["specs"]):
            parts.append(Compute(spec, 1, (PerRow(sub_rows(logits, r, r + 1)),), (PerRow(sub_rows(out, r, r + 1)),), (node.id,),
                                 hint=hint_of(node, f"gumbel_sample stage 3 row {r}: {spec.id}"), kernels=kernels if r == 0 else (),
                                 attribution=((rid, pos),), pattern=self.name))
        primary = parts[0]
        return ComputeGroup(primary.spec, primary.rows, primary.operands, primary.outputs, primary.consumed, hint=primary.hint,
                            kernels=primary.kernels, attribution=primary.attribution, pattern=self.name, also=tuple(parts[1:]))


def ov_sampling_patterns(C: dict) -> list:
    return [TokenSampleGumbel(vocab=int(C["V"]))]
