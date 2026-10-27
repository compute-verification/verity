"""Sampling patterns: the gather of the logits rows and the token selects (greedy argmax, the greedy two-stage Gumbel sampler,
the seeded temperature + top-p sampler)."""

from __future__ import annotations

import collections
import math
from dataclasses import dataclass, replace
from typing import Any

from verity.ml.circuits.defs import bind
from verity.ml.definitions.registry import b1

from verity_vllm.observe.events import ViewDesc
from verity_vllm.observe.fold.patterns.geometry import is_int, sub_rows
from verity_vllm.observe.fold.patterns.ops import base_op, bound_args, first, triton_name
from verity_vllm.observe.fold.patterns.resolutions import ComputeGroup, _fmt, hint_of, unsupported
from verity_vllm.observe.fold.patterns.target import target_of
from verity_vllm.observe.fold.patterns.transparent import widened_source
from verity_vllm.observe.fold.resolver import Compute, Const, Gather, Param, PerRow, Resolution, StepContext, Supplied, Transparent, Unsupported
from verity_vllm.observe.fold.tree import TreeNode

SCRATCH_TOKEN_SELECT = "token_select"  # ctx.scratch key: two-stage greedy sampler state (TokenSelectGumbelTwoStage)


def greedy_sampling(ctx: StepContext) -> str | None:
    """None when the run header records greedy sampling (temperature 0) for the whole workload, else the reason."""
    s = (ctx.header.workload or {}).get("sampling") or (ctx.header.profile or {}).get("sampling") or {}
    if s.get("greedy") is not True:
        return "the run header does not record greedy sampling"
    if s.get("temperature") not in (0, 0.0):
        return f"sampling temperature is {s.get('temperature')!r}, not 0"
    if s.get("top_k") not in (None, 0, -1) or s.get("top_p") not in (None, 1, 1.0):
        return f"top_k/top_p sampling ({s.get('top_k')!r}/{s.get('top_p')!r}) is not the registered TokenSelect"
    return None


@dataclass
class TokenSelectGumbelTwoStage:
    """vLLM V2 sampler (`vllm/v1/worker/gpu/sample/gumbel.py`), greedy path.  `gumbel_sample(logits[R,V])` is a two-stage
    argmax:

        _gumbel_sample_kernel[(R, NB)](local_argmax int64[R,NB], local_max f32[R,NB], logits bf16[R,V], seeds, pos, temp, ...)
            # per (row, block): local_max[row,b] = max(logits[row, b*BLOCK:(b+1)*BLOCK]), local_argmax[row,b] = its index
        idx     = aten.argmax(local_max, dim=-1, keepdim=True)     # int64 [R,1]: the winning block
        sampled = aten.gather(local_argmax, -1, idx)               # int64 [R,1]: the token id

    With temperature 0 the kernel adds no Gumbel noise (`temp == 0` branch) and the composite is the first global maximum
    of the row = the registered `TokenSelect_v1{V}` (tl.argmax and torch.argmax both take the first maximum).  Required:
    `APPLY_TEMPERATURE=False`, `IS_DRAFTING=False`, `PER_TOKEN_COL=False`, no logits cache, and a run header recording
    greedy sampling (the kernel branches on the device-side temperature buffer, whose value is the public workload
    sampling parameter; anything else is Unsupported).  Stage 1 (the launch) and stage 2 (the argmax over `local_max`)
    resolve to `Transparent(noop)` and leave their views in `ctx.scratch["token_select"]`; stage 3 (the gather) emits
    `Compute(TokenSelect_v1{V}, rows=R, (PerRow(logits),), (PerRow(sampled),))`."""

    vocab: int | None = None
    block_size: int = 1024
    name: str = "sample.gumbel_two_stage"

    def match(self, node: TreeNode, ctx: StepContext) -> Resolution | None:
        if node.is_triton:
            return self._stage1(node, ctx) if triton_name(node) == "_gumbel_sample_kernel" else None
        b = base_op(node.op)
        if b == "aten.argmax":
            return self._stage2(node, ctx)
        if b == "aten.gather":
            return self._stage3(node, ctx)
        return None

    def _state(self, ctx: StepContext) -> dict:
        return ctx.scratch.setdefault(SCRATCH_TOKEN_SELECT, {"stage1": {}, "stage2": {}})

    def _stage1(self, node: TreeNode, ctx: StepContext) -> Resolution | None:
        why = greedy_sampling(ctx)
        if why is not None:
            return unsupported(node, f"_gumbel_sample_kernel outside greedy sampling: {why}")
        a = bound_args(node)
        consts = dict(node.enter.constexprs)
        flags = {k: first(consts.get(k), a.get(k)) for k in ("IS_DRAFTING", "APPLY_TEMPERATURE", "PER_TOKEN_COL")}
        if any(bool(v) for v in flags.values()):
            return unsupported(node, "_gumbel_sample_kernel with drafting / temperature / per-token-column flags", **flags)
        if a.get("logits_cache_ptr") is not None:
            return unsupported(node, "_gumbel_sample_kernel writing a logits cache", logits_cache=_fmt(a.get("logits_cache_ptr")))
        block = first(consts.get("BLOCK_SIZE"), a.get("BLOCK_SIZE"))
        logits = a.get("logits_ptr")
        local_argmax = a.get("local_argmax_ptr")
        local_max = a.get("local_max_ptr")
        if not all(isinstance(v, ViewDesc) for v in (logits, local_argmax, local_max)):
            return unsupported(node, "_gumbel_sample_kernel pointers not bound as expected", args={k: _fmt(v) for k, v in a.items()})
        if logits.rank != 2 or logits.strides[1] != 1 or logits.dtype != "bfloat16":
            return unsupported(node, "TokenSelect_v1 is defined over row-contiguous bf16 [R,V] logits", logits=repr(logits))
        R, V = logits.shape
        if self.vocab is not None and V != self.vocab:
            return unsupported(node, "logits width is not the vocabulary", V=V, vocab=self.vocab)
        vs = a.get("vocab_size")
        if vs is not None and int(vs) != V:
            return unsupported(node, "vocab_size argument differs from the logits width", vocab_size=vs, V=V)
        if not isinstance(block, int) or block <= 0:
            return unsupported(node, "BLOCK_SIZE constexpr missing", BLOCK_SIZE=block)
        if block != self.block_size:
            return unsupported(node, "BLOCK_SIZE differs from the pinned sampler block", BLOCK_SIZE=block, pinned=self.block_size)
        nb = -(-V // block)
        if local_argmax.shape != (R, nb) or local_max.shape != (R, nb) or not is_int(local_argmax) or local_max.dtype != "float32":
            return unsupported(node, "local argmax/max buffers are not int64/f32 [R, ceil(V/BLOCK)]", local_argmax=repr(local_argmax),
                               local_max=repr(local_max), R=R, NB=nb)
        if a.get("logits_stride") not in (None, logits.strides[0]):
            return unsupported(node, "logits_stride differs from the logits view", logits_stride=a.get("logits_stride"), view=repr(logits))
        st = self._state(ctx)
        st["stage1"][local_max.alloc] = {"logits": logits, "local_argmax": local_argmax, "local_max": local_max, "kernels": (triton_name(node),)}
        return Transparent("noop", (node.id,), hint=hint_of(node, "gumbel_sample stage 1 (blockwise max/argmax)"), kernels=(triton_name(node),),
                           pattern=self.name)

    def _stage2(self, node: TreeNode, ctx: StepContext) -> Resolution | None:
        a = bound_args(node)
        x = first(a.get("self"), a.get("arg0"))
        if not isinstance(x, ViewDesc):
            return None
        st = ctx.scratch.get(SCRATCH_TOKEN_SELECT) or {}
        s1 = (st.get("stage1") or {}).get(x.alloc)
        if s1 is None or x.shape != s1["local_max"].shape:
            return None  # not the two-stage sampler's argmax; TokenSelectArgmax may claim it
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
        st = ctx.scratch.get(SCRATCH_TOKEN_SELECT) or {}
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
        rows = tuple(ctx.start.logits_rows) if R == len(ctx.start.logits_rows) else ()
        return Compute(bind(b1.TokenSelect, V=V), R, (PerRow(logits),), (PerRow(out),), (node.id,),
                       hint=hint_of(node, "gumbel_sample stage 3 (token id) = argmax(logits)"), kernels=(*s2["kernels"], "ArgMaxOps", "gather_kernel"),
                       attribution=rows, pattern=self.name)


@dataclass
class TokenSelectArgmax:
    """`aten.argmax(logits[R,V], dim=-1)` (phase sample, greedy) → `Compute(TokenSelect_v1{V}, rows=R, (PerRow(logits),),
    (PerRow(out),))`.  The registered `TokenSelect_v1` is the argmax over *bf16* logits; float32 logits are Unsupported
    (reported, not silently accepted).  An argmax over an integer tensor is not over logits: the integer rule's
    (`classify_transparent`), as vLLM's `split_decodes_and_prefills` finding a mixed step's first prefill request."""

    vocab: int | None = None
    name: str = "sample.argmax_token_select"

    def match(self, node: TreeNode, ctx: StepContext) -> Resolution | None:
        if node.is_triton or base_op(node.op) != "aten.argmax":
            return None
        a = bound_args(node)
        x = first(a.get("self"), a.get("arg0"))
        dim = a.get("dim", a.get("arg1"))
        outs = node.output_views()
        if not isinstance(x, ViewDesc) or not outs:
            return unsupported(node, "argmax arguments not bound as expected", args={k: _fmt(v) for k, v in a.items()})
        if is_int(x):
            return None
        if x.rank == 1:
            logits = ViewDesc(x.alloc, x.offset, (1, x.shape[0]), (x.shape[0] * x.strides[0], x.strides[0]), x.dtype, x.device)
            if dim not in (None, 0, -1):
                return unsupported(node, "argmax over a 1-D tensor with an unexpected dim", dim=dim)
        elif x.rank == 2 and dim in (1, -1):
            logits = x
        else:
            return unsupported(node, "argmax is not over the last axis of [R,V] logits", view=repr(x), dim=dim)
        R, V = logits.shape
        if logits.strides[1] != 1:
            return unsupported(node, "logits rows are not contiguous", view=repr(logits))
        if self.vocab is not None and V != self.vocab:
            return unsupported(node, "argmax width is not the vocabulary", V=V, vocab=self.vocab)
        why = greedy_sampling(ctx)
        if why is not None:
            return unsupported(node, f"argmax over the logits outside greedy sampling: {why}")
        if logits.dtype == "float32":
            # the sampler's `logits.to(torch.float32)`: argmax over the exact widening of the bf16 logits is the argmax
            # over the bf16 logits (the registered `TokenSelect_v1` body); the operand is the bf16 source view
            narrow = widened_source(logits, ctx)
            if narrow is None:
                return unsupported(node, "TokenSelect_v1 is defined over bf16 logits; these float32 logits are not an exact "
                                         "widening of a bf16 view seen this step", view=repr(logits))
            logits = narrow
        if logits.dtype != "bfloat16":
            return unsupported(node, "TokenSelect_v1 is defined over bf16 logits", dtype=logits.dtype)
        out = outs[0]
        if out.numel != R or not is_int(out):
            return unsupported(node, "argmax output is not an integer [R] view", out=repr(out))
        return Compute(bind(b1.TokenSelect, V=V), R, (PerRow(logits),), (PerRow(out),), (node.id,), hint=hint_of(node, "argmax"),
                       kernels=("ArgMaxOps",), attribution=tuple(ctx.start.logits_rows) if R == len(ctx.start.logits_rows) else (),
                       pattern=self.name)


SCRATCH_TOPP_SELECT = "topp_token_select"  # ctx.scratch key: the seeded temperature + top-p sampler chain (TokenSelectGumbelTopP)
TOPP_STATS_BLOCK, TOPP_STEP_BLOCK, TOPP_FANOUT, TOPP_ROUNDS = 8192, 2048, 8, 5   # vLLM `_apply_topp_split` pins (registry/topp_split.py)
TEMPERATURE_BLOCK = 8192                                                          # `apply_temperature` BLOCK_SIZE


def header_sampling(ctx: StepContext) -> dict:
    return dict((ctx.header.workload or {}).get("sampling") or (ctx.header.profile or {}).get("sampling") or {})


def request_sampling(ctx: StepContext) -> dict[str, dict]:
    """request id -> {seed, temperature, top_p, top_k} of the run header's request record (children of an n>1 request: the
    recorded fork fact parent seed + index); the workload's sampling block supplies what a record does not pin."""
    wl = ctx.header.workload or {}
    s = header_sampling(ctx)
    per = {p["request_id"]: p for p in s.get("per_request", []) or []}
    out: dict[str, dict] = {}
    for r in wl.get("requests", []) or []:
        p = per.get(r["request_id"], {})
        out[r["request_id"]] = {"seed": None if r.get("seed") is None else int(r["seed"]),
                                "temperature": float(first(r.get("temperature"), p.get("temperature"), s.get("temperature", 0.0))),
                                "top_p": float(first(r.get("top_p"), p.get("top_p"), s.get("top_p", 1.0))),
                                "top_k": int(first(r.get("top_k"), p.get("top_k"), s.get("top_k", 0)) or 0)}
    for c in wl.get("children", []) or []:
        base = dict(out.get(c["parent"], {}))
        base["seed"] = None if c.get("seed") is None else int(c["seed"])
        out[c["request_id"]] = base
    return out


def _f32_bits(x: float) -> int:
    import struct
    return struct.unpack("<I", struct.pack("<f", float(x)))[0]


@dataclass
class TokenSelectGumbelTopP:
    """vLLM V2 sampler (d9105ea80 `Sampler.sample`), the SEEDED temperature + top-p path (an explicit `SamplingParams.seed`
    disables FlashInfer sampling; `top_k` off).  The launch chain of one sampling step, on `logits32 = f32(logits)` (the exact widening
    `classify_transparent` records):

        _temperature_kernel[(R, ceil(V/8192))](logits32, stride, expanded_idx_mapping, temperature[·], V)      logits32[r] /= T[r] in place
        aten.index(top_p[·], expanded_idx_mapping) -> p f32[R]                                                  request fact (header)
        _topp_sb_stats_kernel[(R*S,)] / _topp_sb_step_kernel x5 / _topp_sb_mask_kernel   (HAS_K=False, S, F=8, NUM_ROUNDS=5)  -inf mask in place
        _gumbel_sample_kernel[(R, ceil(V/1024))](local_argmax, local_max, logits32, idx_mapping, seeds[·], pos[·], temperature[·], V;
                                                 APPLY_TEMPERATURE=False, IS_DRAFTING=False, USE_FP64=False, PER_TOKEN_COL=False)
        aten.argmax(local_max) -> winning block ; aten.gather(local_argmax) -> the token id

    Every launch but the last resolves to `Transparent(noop)` that claims its kernel and records the chain in `ctx.scratch["topp_token_select"]`
    (keyed by the fp32 logits allocation); the gather emits, per logits row r of request `rid` at position `pos`,

        Compute(GumbelTopPTokenSelect_v1{V}, 1, (PerRow(logits_bf16[r]), Const32[top_p], Param(seed of rid), Const64[pos], Const32[T], Const32[S]),
                (PerRow(sampled[r]),))

    -- the registry composite (`verity/ml/definitions/registry/sampling.py`: TemperatureScale -> TopPMask{V} (topp_split model) -> GumbelSelectF32),
    the same instance the derived Program's `GumbelTopPTokenSelect[vllm]` rule emits: temperature / top_p / pos are Program constants
    (the fold emits their `Const` instances in operand order right before the select), the seed is the request's `seed` PARAMETER.  S
    (the 6th operand, `Const32[S]` "splits") is read from the split kernels' constexpr (the launch geometry:
    live batch x SM count), must agree across the three kernels, and is bound at compare to the request Program's `splits` Input entry
    for the event (or held equal to a single-request Program's own `Const32[S]`); the relation S = SplitsFor_v1(|live(t)|, num_SMs) is
    checked per engine step by GM-01 G6.  Values (which seed / T / p the kernels consumed) are NOT read from device memory here: they
    are the header's request record; the value check
    (`bench.stoch_recompute`) recomputes the served token from the snapshotted logits under the DECLARED seed and fails on a wrong one.

    Unsupported (fail closed, each a named reason): a greedy header (the registered TokenSelect patterns' case, returns None); top-k
    (`_topk_topp_kernel`); min_p / logit bias / penalties / bad words (a write into logits32 before the temperature kernel makes it not the
    exact widening); an unseeded sampled row; a row at T in {0, 1} or p == 1 (other Program families); missing / duplicated / unpinned
    split kernels; the fp64 Gumbel path; a logits cache."""

    vocab: int | None = None
    block_size: int = 1024
    #: the construction the Program declares (`pipeline.match --program-dir`: `sampling_event.program_sampler_construction`); None = the
    #: record's declared target's.  A single-request Program's select, specialised on its constant S, has no shared-greedy form.
    construction: str | None = None
    name: str = "sample.gumbel_topp_select"

    def match(self, node: TreeNode, ctx: StepContext) -> Resolution | None:
        if header_sampling(ctx).get("greedy") is True:
            return None
        if node.is_triton:
            k = triton_name(node)
            if k == "_temperature_kernel":
                return self._temperature(node, ctx)
            if k in ("_topp_sb_stats_kernel", "_topp_sb_step_kernel", "_topp_sb_mask_kernel"):
                return self._topp(node, ctx, k)
            if k == "_gumbel_sample_kernel":
                return self._stage1(node, ctx)
            if k in ("_topk_topp_kernel", "_min_p_kernel", "_logit_bias_kernel", "_penalties_kernel", "_bad_words_kernel"):
                return unsupported(node, f"{k}: the top-k / min_p / logit-bias / penalties sampler paths have no registered Definition "
                                         "(only the seeded temperature + p-only split pipeline is bound: GumbelTopPTokenSelect_v1)")
            return None
        b = base_op(node.op)
        if b == "aten.argmax":
            return self._stage2(node, ctx)
        if b == "aten.gather":
            return self._stage3(node, ctx)
        if b == "aten.index" and node.phase == "sample":
            return self._state_gather(node, ctx)
        return None

    def _state(self, ctx: StepContext) -> dict:
        return ctx.scratch.setdefault(SCRATCH_TOPP_SELECT, {"rows": {}, "gathers": {}, "stage1": {}, "stage2": {}})

    def _state_gather(self, node: TreeNode, ctx: StepContext) -> Resolution | None:
        """`top_p.gpu[expanded_idx_mapping]` (and `top_k.gpu[...]`): a 1-D per-request sampling-state buffer (max_num_reqs entries)
        gathered by the step's request indices -- a request fact of the header, not an activation.  Remembered so the split kernels'
        `p_ptr` can be checked to be exactly this gather."""
        a = bound_args(node)
        src = first(a.get("self"), a.get("arg0"))
        outs = node.output_views()
        if not isinstance(src, ViewDesc) or src.rank != 1 or src.dtype not in ("float32", "int32", "int64") or not outs:
            return None
        idxs = a.get("indices", a.get("arg1"))
        if not isinstance(idxs, (list, tuple)) or len(idxs) != 1 or not isinstance(idxs[0], ViewDesc) or not is_int(idxs[0]):
            return None
        out = outs[0]
        self._state(ctx)["gathers"][out.alloc] = {"src": src, "out": out, "dtype": src.dtype}
        return Transparent("noop", (node.id,), hint=hint_of(node, f"per-request sampling-state gather ({src.dtype}[{src.numel}]): a request fact"),
                           pattern=self.name)

    def _row_state(self, node: TreeNode, ctx: StepContext, logits: Any) -> dict | Unsupported:
        """The chain record of the fp32 logits allocation `logits` writes / reads, created at first sight from the exact widening."""
        if not isinstance(logits, ViewDesc) or logits.dtype != "float32" or logits.rank != 2 or logits.strides[1] != 1:
            return unsupported(node, "sampler logits are not a row-contiguous float32 [R,V] view", logits=_fmt(logits))
        st = self._state(ctx)["rows"]
        rec = st.get(logits.alloc)
        if rec is None:
            narrow = widened_source(logits, ctx)
            if narrow is None or narrow.dtype != "bfloat16" or narrow.shape != logits.shape:
                return unsupported(node, "the float32 logits are not the exact widening of a bf16 [R,V] view seen this step (a logit-bias / "
                                         "penalties / bad-words / min_p write before the sampler chain?)", logits=repr(logits), narrow=_fmt(narrow))
            rec = {"bf16": narrow, "f32": logits, "temperature": False, "topp": collections.Counter(), "S": None, "kernels": []}
            st[logits.alloc] = rec
        elif logits.shape != rec["f32"].shape or logits.offset != rec["f32"].offset:
            return unsupported(node, "a sampler kernel on a different view of the logits allocation", logits=repr(logits), chain=repr(rec["f32"]))
        return rec

    def _temperature(self, node: TreeNode, ctx: StepContext) -> Resolution:
        a = bound_args(node)
        consts = dict(node.enter.constexprs)
        rec = self._row_state(node, ctx, a.get("logits_ptr"))
        if isinstance(rec, Unsupported):
            return rec
        if rec["temperature"] or rec["topp"]:
            return unsupported(node, "_temperature_kernel after another sampler kernel on the same logits (chain order)", chain=rec["kernels"])
        block = first(consts.get("BLOCK_SIZE"), a.get("BLOCK_SIZE"))
        if block != TEMPERATURE_BLOCK:
            return unsupported(node, "_temperature_kernel BLOCK_SIZE differs from the pinned block", BLOCK_SIZE=block, pinned=TEMPERATURE_BLOCK)
        vs = a.get("vocab_size")
        if vs is not None and int(vs) != rec["f32"].shape[1]:
            return unsupported(node, "vocab_size argument differs from the logits width", vocab_size=vs, V=rec["f32"].shape[1])
        rec["temperature"] = True
        rec["kernels"].append(triton_name(node))
        return Transparent("noop", (node.id,), hint=hint_of(node, "temperature: logits32[r] /= T[r] in place (div.full.f32; TemperatureScale_v1)"),
                           kernels=(triton_name(node),), pattern=self.name)

    def _topp(self, node: TreeNode, ctx: StepContext, k: str) -> Resolution:
        a = bound_args(node)
        consts = dict(node.enter.constexprs)
        rec = self._row_state(node, ctx, first(a.get("LOGITS"), a.get("logits_ptr")))     # the split kernels' signature: LOGITS, STATS, PARTS, K, P, ROUND
        if isinstance(rec, Unsupported):
            return rec
        R, V = rec["f32"].shape
        if bool(first(consts.get("HAS_K"), a.get("HAS_K"))):
            return unsupported(node, f"{k} with HAS_K: the top-k + top-p split path has no registered Definition")
        vs = first(consts.get("VOCAB_SIZE"), a.get("VOCAB_SIZE"))
        if vs is not None and int(vs) != V:
            return unsupported(node, "VOCAB_SIZE constexpr differs from the logits width", VOCAB_SIZE=vs, V=V)
        S = first(consts.get("S"), a.get("S"))
        if not isinstance(S, int) or S <= 0 or S & (S - 1) or S > 32:
            return unsupported(node, "S (splits) is not a power of two <= 32", S=S)
        if rec["S"] is not None and rec["S"] != S:
            return unsupported(node, "S differs between the split kernels of one step", S=S, chain_S=rec["S"])
        pins = {"_topp_sb_stats_kernel": {"BLOCK": TOPP_STATS_BLOCK},
                "_topp_sb_step_kernel": {"BLOCK": TOPP_STEP_BLOCK, "F": TOPP_FANOUT, "NUM_ROUNDS": TOPP_ROUNDS},
                "_topp_sb_mask_kernel": {"BLOCK": TOPP_STATS_BLOCK, "F": TOPP_FANOUT, "NUM_ROUNDS": TOPP_ROUNDS}}[k]
        for key, want in pins.items():
            got = first(consts.get(key), a.get(key))
            if got != want:
                return unsupported(node, f"{k} {key} differs from the pinned split pipeline", **{key: got, "pinned": want})
        if k == "_topp_sb_mask_kernel":
            mv = first(consts.get("MASK_VALUE"), a.get("MASK_VALUE"))
            if mv is not None and not (isinstance(mv, float) and math.isinf(mv) and mv < 0):
                return unsupported(node, "_topp_sb_mask_kernel MASK_VALUE is not -inf", MASK_VALUE=mv)
        p = first(a.get("P"), a.get("p_ptr"))
        g = self._state(ctx)["gathers"].get(getattr(p, "alloc", None))
        if not isinstance(p, ViewDesc) or g is None or g["dtype"] != "float32" or p.numel != R:
            return unsupported(node, f"{k} P is not the step's float32 [R] gather of the per-request top_p state", p=_fmt(p), R=R)
        # HAS_K=False launches pass the LOGITS pointer itself as the dummy `K` (`k_ptr = k if k is not None else logits`): any other buffer
        # there is a top-k buffer of a path that is not bound
        kk = first(a.get("K"), a.get("k_ptr"))
        if isinstance(kk, ViewDesc) and kk.alloc != rec["f32"].alloc:
            return unsupported(node, f"{k} with a top-k buffer bound (HAS_K path)", k=_fmt(kk))
        order = {"_topp_sb_stats_kernel": 0, "_topp_sb_step_kernel": 1, "_topp_sb_mask_kernel": 2}[k]
        seen = rec["topp"]
        if order == 0 and seen:
            return unsupported(node, "_topp_sb_stats_kernel after another split kernel", chain=rec["kernels"])
        if order == 1 and (seen["_topp_sb_stats_kernel"] != 1 or seen["_topp_sb_mask_kernel"]):
            return unsupported(node, "_topp_sb_step_kernel outside stats -> steps -> mask", chain=rec["kernels"])
        if order == 1:
            ri = first(a.get("ROUND"), a.get("round_i"))
            if isinstance(ri, int) and ri != seen["_topp_sb_step_kernel"]:
                return unsupported(node, "_topp_sb_step_kernel round out of order", ROUND=ri, expected=seen["_topp_sb_step_kernel"])
        if order == 2 and (seen["_topp_sb_stats_kernel"] != 1 or seen["_topp_sb_step_kernel"] != TOPP_ROUNDS or seen["_topp_sb_mask_kernel"]):
            return unsupported(node, f"_topp_sb_mask_kernel without exactly one stats + {TOPP_ROUNDS} step launches before it", chain=rec["kernels"])
        seen[k] += 1
        rec["S"] = S
        rec["kernels"].append(triton_name(node))
        return Transparent("noop", (node.id,), hint=hint_of(node, f"top-p split pipeline S={S}: {k} (TopPMask_v1{{V}}, splits operand)"),
                           kernels=(triton_name(node),), pattern=self.name)

    def _stage1(self, node: TreeNode, ctx: StepContext) -> Resolution | None:
        a = bound_args(node)
        consts = dict(node.enter.constexprs)
        flags = {k: first(consts.get(k), a.get(k)) for k in ("IS_DRAFTING", "APPLY_TEMPERATURE", "PER_TOKEN_COL", "USE_FP64")}
        if any(bool(v) for v in flags.values()):
            return unsupported(node, "_gumbel_sample_kernel with drafting / in-kernel temperature / per-token-column / fp64 flags", **flags)
        if a.get("logits_cache_ptr") is not None:
            return unsupported(node, "_gumbel_sample_kernel writing a logits cache", logits_cache=_fmt(a.get("logits_cache_ptr")))
        logits = a.get("logits_ptr")
        local_argmax = a.get("local_argmax_ptr")
        local_max = a.get("local_max_ptr")
        if not all(isinstance(v, ViewDesc) for v in (logits, local_argmax, local_max)):
            return unsupported(node, "_gumbel_sample_kernel pointers not bound as expected", args={k: _fmt(v) for k, v in a.items()})
        rec = self._row_state(node, ctx, logits)
        if isinstance(rec, Unsupported):
            return rec
        R, V = rec["f32"].shape
        if self.vocab is not None and V != self.vocab:
            return unsupported(node, "logits width is not the vocabulary", V=V, vocab=self.vocab)
        vs = a.get("vocab_size")
        if vs is not None and int(vs) != V:
            return unsupported(node, "vocab_size argument differs from the logits width", vocab_size=vs, V=V)
        block = first(consts.get("BLOCK_SIZE"), a.get("BLOCK_SIZE"))
        if not isinstance(block, int) or block != self.block_size:
            return unsupported(node, "BLOCK_SIZE differs from the pinned sampler block", BLOCK_SIZE=block, pinned=self.block_size)
        nb = -(-V // block)
        if local_argmax.shape != (R, nb) or local_max.shape != (R, nb) or not is_int(local_argmax) or local_max.dtype != "float32":
            return unsupported(node, "local argmax/max buffers are not int64/f32 [R, ceil(V/BLOCK)]", local_argmax=repr(local_argmax),
                               local_max=repr(local_max), R=R, NB=nb)
        seen = rec["topp"]
        chain_ok = seen["_topp_sb_stats_kernel"] == 1 and seen["_topp_sb_step_kernel"] == TOPP_ROUNDS and seen["_topp_sb_mask_kernel"] == 1
        if not chain_ok:
            return unsupported(node, "_gumbel_sample_kernel without the complete split pipeline (stats, 5 steps, mask) on its logits: rows at "
                                     "top_p == 1 are GumbelTokenSelect_v1 (not bound in this profile)", chain=rec["kernels"])
        rows = list(ctx.start.logits_rows)
        if len(rows) != R:
            return unsupported(node, "logits rows do not match the step's sampled rows", R=R, logits_rows=len(rows))
        rs = request_sampling(ctx)
        specs: list[tuple[str, int, int, float, float]] = []
        for rid, pos in rows:
            q = rs.get(rid)
            if q is None:
                return unsupported(node, "sampled row of a request the run header does not record", request=rid)
            T, p, seed, top_k = q["temperature"], q["top_p"], q["seed"], q["top_k"]
            if top_k not in (0, -1):
                return unsupported(node, "top_k row in a p-only split batch (the header contradicts the launch)", request=rid, top_k=top_k)
            if T in (0.0, 1.0):
                return unsupported(node, "row at temperature 0 / 1 in a stochastic batch: TokenSelect_v1 / an unscaled row (not bound "
                                         "together with GumbelTopPTokenSelect_v1 in this profile)", request=rid, temperature=T)
            if not rec["temperature"]:
                return unsupported(node, "header temperature != 1 but no _temperature_kernel launch on the logits", request=rid, temperature=T)
            if p >= 1.0 or p <= 0.0:
                return unsupported(node, "row at top_p outside (0, 1) in a split-pipeline batch", request=rid, top_p=p)
            if seed is None:
                return unsupported(node, "unseeded sampled row: vLLM's SamplingStates drew the seed on the host (`seed: null` in the request "
                                         "record) -- the RNG stream is not a declared input", request=rid, position=int(pos))
            specs.append((rid, int(pos), int(seed), T, p))
        st = self._state(ctx)
        st["stage1"][local_max.alloc] = {"bf16": rec["bf16"], "local_argmax": local_argmax, "local_max": local_max, "S": int(rec["S"]),
                                          "specs": specs, "kernels": tuple(rec["kernels"]) + (triton_name(node),)}
        return Transparent("noop", (node.id,), hint=hint_of(node, "gumbel_sample stage 1 (noised blockwise max/argmax over the masked row)"),
                           kernels=(triton_name(node),), pattern=self.name)

    def _stage2(self, node: TreeNode, ctx: StepContext) -> Resolution | None:
        a = bound_args(node)
        x = first(a.get("self"), a.get("arg0"))
        if not isinstance(x, ViewDesc):
            return None
        st = ctx.scratch.get(SCRATCH_TOPP_SELECT) or {}
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
        st = ctx.scratch.get(SCRATCH_TOPP_SELECT) or {}
        s2 = (st.get("stage2") or {}).get(idx.alloc)
        if s2 is None or src.alloc != s2["local_argmax"].alloc:
            return None
        dim = a.get("dim", a.get("arg1"))
        outs = node.output_views()
        if dim not in (1, -1) or not outs:
            return unsupported(node, "gather over local_argmax not along the block axis", dim=dim)
        logits: ViewDesc = s2["bf16"]
        R, V = logits.shape
        out = outs[0]
        if out.numel != R or not is_int(out):
            return unsupported(node, "sampled ids are not an integer [R] view", out=repr(out))
        if out.rank == 1:
            out = ViewDesc(out.alloc, out.offset, (R, 1), (out.strides[0], 1), out.dtype, out.device)
        from verity.ml.definitions.registry import sampling as _sampling
        from verity.ml.definitions.registry import sampling_vocab
        construction = self.construction or target_of(ctx)[0].effective_sampler_construction()
        spec = bind(sampling_vocab.select_for(_sampling.GumbelTopPTokenSelect, construction), V=V)
        S = int(s2["S"])        # the split kernels' constexpr: the launch's S -> the 6th operand (bound to the Program's `splits` Input)
        kernels = (*s2["kernels"], "ArgMaxOps", "gather_kernel")
        parts = []
        run = getattr(ctx.profile, "registered_values", False)
        for r, (rid, pos, seed, T, p) in enumerate(s2["specs"]):
            pos_bits = int(pos) & 0xFFFFFFFFFFFFFFFF
            if run:     # the request's sampling values are inputs of its run, the Philox position one per sampled position
                top_p, phi, temp = (Supplied("run", "top_p", 32, _f32_bits(p), rid), Supplied("run", f"pos_{int(pos)}", 64, pos_bits, rid),
                                    Supplied("run", "temperature", 32, _f32_bits(T), rid))
            else:
                top_p, phi, temp = Const(32, _f32_bits(p), "top_p"), Const(64, pos_bits, "pos"), Const(32, _f32_bits(T), "temperature")
            operands = (PerRow(sub_rows(logits, r, r + 1)), top_p, Param("seed", (rid,)), phi, temp, Const(32, S, "splits"))
            parts.append(Compute(spec, 1, operands, (PerRow(sub_rows(out, r, r + 1)),), (node.id,),
                                 hint=hint_of(node, f"gumbel_topp_token_select row {r}"), kernels=kernels if r == 0 else (),
                                 attribution=((rid, pos),), pattern=self.name))
        primary = parts[0]
        return ComputeGroup(primary.spec, primary.rows, primary.operands, primary.outputs, primary.consumed, hint=primary.hint,
                            kernels=primary.kernels, attribution=primary.attribution, pattern=self.name, also=tuple(parts[1:]))


@dataclass
class LogitsGather:
    """`aten.index.Tensor(hidden[M,H], [logits_indices])` / `aten.index_select(hidden, 0, logits_indices)` with the
    step's public `logits_indices` → `Transparent(move, rows=R, src=Gather(hidden, logits_indices), dst=PerRow(out))`.
    Integer sources are left to the integer-metadata rule."""

    name: str = "sample.logits_gather"

    def match(self, node: TreeNode, ctx: StepContext) -> Resolution | None:
        if node.is_triton or base_op(node.op) not in ("aten.index", "aten.index_select"):
            return None
        a = bound_args(node)
        src = first(a.get("self"), a.get("arg0"))
        if not isinstance(src, ViewDesc) or is_int(src):
            return None
        if base_op(node.op) == "aten.index":
            idxs = a.get("indices", a.get("arg1"))
            if not isinstance(idxs, (list, tuple)):
                return unsupported(node, "aten.index without an index list")
            tensors = [i for i in idxs if isinstance(i, ViewDesc)]
            if len(tensors) != 1 or idxs[0] is None and len(idxs) > 1:
                return unsupported(node, "aten.index with an index pattern other than one leading index tensor", indices=_fmt(idxs))
            idx = tensors[0]
        else:
            dim = a.get("dim", a.get("arg1"))
            idx = a.get("index", a.get("arg2"))
            if dim not in (0,) or not isinstance(idx, ViewDesc):
                return unsupported(node, "index_select not along the leading axis", dim=dim)
        outs = node.output_views()
        if not outs or not is_int(idx) or idx.rank != 1:
            return unsupported(node, "gather index is not an integer [R] view", index=repr(idx))
        R = idx.shape[0]
        li = ctx.start.logits_indices
        if R != len(li):
            return unsupported(node, "gather rows differ from the step's logits_indices", R=R, logits_indices=len(li))
        if src.rank < 1 or any(i < 0 or i >= src.shape[0] for i in li):
            return unsupported(node, "logits_indices outside the gathered source", src=repr(src), logits_indices=list(li)[:8])
        M = len(ctx.start.rows)
        if src.shape[0] not in (M, ctx.start.num_tokens_padded):
            return unsupported(node, "gathered source is not the step's [M, ...] token rows", src=repr(src), M=M)
        out = outs[0]
        if out.shape != (R,) + src.shape[1:]:
            return unsupported(node, "gather output shape does not match", out=repr(out), src=repr(src))
        return Transparent("move", (node.id,), src=Gather(src, tuple(int(i) for i in li)), dst=PerRow(out), rows=R,
                           hint=hint_of(node, "hidden[logits_indices]"), kernels=("index_elementwise_kernel", "indexSelectLargeIndex", "indexSelectSmallIndex"),
                           pattern=self.name)


def with_sampler_construction(profile, construction: str | None):
    """`profile` with its top-p select pattern bound under `construction`, the one the Program declares (`pipeline.match --program-dir`);
    `profile` itself when None.  A construction `TargetProfile` does not name is refused by name."""
    if construction is None:
        return profile
    from verity_vllm.program.frontend.target_profile import TargetProfile
    if construction not in TargetProfile.SAMPLER_CONSTRUCTIONS:
        raise SystemExit(f"sampler construction {construction!r} is not one of {list(TargetProfile.SAMPLER_CONSTRUCTIONS)}")
    return replace(profile, patterns=[replace(p, construction=construction) if isinstance(p, TokenSelectGumbelTopP) else p for p in profile.patterns])
