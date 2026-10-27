"""GPT-NeoX (Pythia) patterns: what vLLM d9105ea80's `GPTNeoXForCausalLM` runs that the base κ pattern list
(`vllm_d9105ea80_sm89_eager.make_patterns`) does not explain, eager, `VLLM_BATCH_INVARIANT=1`, bf16.

    nn.LayerNorm(H, eps) with bias          aten.layer_norm(x[M,H], [H], w, b, eps, cudnn) {vectorized_layer_norm_kernel}
                                              -> LayerNormAten{N=H, EPS} rows=M  (PerRow(x), Shared(w), Shared(b)) -> (PerRow(y),)
    nn.GELU() (erf, `hidden_act="gelu"`)     aten.gelu(x[M,I], approximate='none')  -> GeluErf{N=I} rows=M
    parallel residual `mlp + attn + x`      aten.add.Tensor(a[M,H] bf16, b[M,H] bf16), neither a weight  -> ResidualAdd_v1{H} (b1)
    partial rotary (rotary_pct 0.25)        _C.rotary_embedding(pos, q, k, head_size=64, cos_sin_cache[MAX_POS, 16], is_neox=True)
                                              -> RoPE_v1{NHEADS, D=16} on the `[M, NH, 16]` sub-view of each head (the kernel
                                                 rotates dims 0..rot_dim-1 of every head and leaves 16..63 untouched: those words keep
                                                 their previous writer, the fused-QKV BiasAdd)

Everything else is the base kernel set: the batch-invariant Triton GEMM (`Gemm_v1`), the separate `aten.add` of every dense bias
(`BiasAdd_v1`: qkv, dense, dense_h_to_4h, dense_4h_to_h), FA2 attention at head dim 64 (`Attention_v3{..., BN=128}`), the KV cache
update, the embedding gather (`embed_in`), the untied `embed_out` logits GEMM, the greedy sampler.

Every pattern checks dtypes / layouts / scalars and resolves ONLY the shapes above; anything else falls through (`None`) or is
reported with a reason.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from verity.ml.circuits.defs import bind
from verity.ml.definitions.registry import b1
from verity.ml.definitions.registry import act, layer_norm

from verity_vllm.observe.events import ViewDesc
from verity_vllm.observe.fold.patterns import (BiasAdd, GeometryError, RotaryEmbedding, _fmt, _group_compute, _positions_ok, as_float,
                                     attribution_for, base_op, bound_args, hint_of, is_row_major, op_name, split_heads, unsupported)
from verity_vllm.observe.fold.resolver import Compute, Gather, PerRow, Resolution, Shared, StepContext
from verity_vllm.observe.fold.tree import TreeNode

KERNELS_ELEMENTWISE = ("vectorized_elementwise_kernel", "elementwise_kernel", "unrolled_elementwise_kernel")
KERNELS_LAYER_NORM = ("vectorized_layer_norm_kernel",)


def schema_default(schema: str | None, param: str) -> str | None:
    """The literal default of `param` in a dispatcher schema string (`'... float eps=1.0000000000000001e-05, bool cudnn_enable=True) ->
    Tensor'` -> `'1.0000000000000001e-05'`), None when the parameter has none or the schema does not parse.  vLLM's `nn.LayerNorm`
    calls `F.layer_norm(x, shape, w, b, eps)` and the recorded dispatch carries only the arguments that differ from the schema
    defaults, so an `eps` equal to torch's default 1e-5 (Pythia's `layer_norm_eps`) arrives as *absent* and must be read here."""
    if not schema or "(" not in schema:
        return None
    body = schema[schema.index("(") + 1:schema.rfind(") ->") if ") ->" in schema else schema.rfind(")")]
    depth = 0
    cur: list[str] = []
    parts: list[str] = []
    for c in body:
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
        if c == "," and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(c)
    if cur:
        parts.append("".join(cur))
    for p in parts:
        lhs, sep, rhs = p.strip().partition("=")
        toks = lhs.split()
        if sep and len(toks) >= 2 and toks[-1] == param:
            return rhs.strip()
    return None


def _compute(node: TreeNode, ctx: StepContext, spec, rows: int, operands: tuple, outputs: tuple, name: str, kernels, *hint_extra: str,
             consumed: tuple[int, ...] | None = None) -> Compute:
    return Compute(spec, rows, operands, outputs, consumed if consumed is not None else (node.id,), hint=hint_of(node, *hint_extra),
                   kernels=kernels, attribution=attribution_for(ctx, rows, node.phase), pattern=name)


@dataclass
class LayerNormAtenPattern:
    """`aten.layer_norm(input[M,N] bf16, normalized_shape=[N], weight[N], bias[N], eps, cudnn_enable)` (or the `aten.native_layer_norm`
    it decomposes to) with bf16 weight AND bias Roots -> `LayerNormAten{N, EPS}` rows=M.  The subtree (the native op, its output
    allocations, the f32 mean/rstd it also writes — read by nothing) is consumed whole; `eps` must equal the profile's EPS."""

    eps: float
    name: str = "layernorm.aten_layer_norm"

    def match(self, node: TreeNode, ctx: StepContext) -> Resolution | None:
        if node.is_triton or base_op(node.op) not in ("aten.layer_norm", "aten.native_layer_norm"):
            return None
        a = bound_args(node)
        x = a.get("input", a.get("arg0"))
        shape = a.get("normalized_shape", a.get("arg1"))
        w = a.get("weight", a.get("arg2"))
        b = a.get("bias", a.get("arg3"))
        eps_raw = a.get("eps", a.get("arg4"))
        if eps_raw is None:                                          # not recorded: the schema default applies (see schema_default)
            lit = schema_default(getattr(node.enter, "schema", None), "eps")
            try:
                eps_raw = float(lit) if lit is not None else None
            except ValueError:
                eps_raw = None
        eps = as_float(eps_raw)
        outs = node.output_views()
        if not isinstance(x, ViewDesc) or not outs:
            return unsupported(node, "layer_norm arguments not bound as expected", args={k: _fmt(v) for k, v in a.items()})
        if not isinstance(w, ViewDesc) or not isinstance(b, ViewDesc):
            return unsupported(node, "LayerNormAten needs both weight and bias tensors (elementwise_affine with bias)", weight=_fmt(w), bias=_fmt(b))
        if x.rank != 2 or not is_row_major(x):
            return unsupported(node, "layer_norm input is not a row-major [M,N] view", x=repr(x))
        M, N = x.shape
        if list(shape or []) != [N]:
            return unsupported(node, "normalized_shape is not the last axis [N]", normalized_shape=shape, N=N)
        y = outs[0]
        if y.shape != (M, N) or not is_row_major(y):
            return unsupported(node, "layer_norm output is not a row-major [M,N] view", y=repr(y))
        if w.shape != (N,) or w.strides != (1,) or b.shape != (N,) or b.strides != (1,):
            return unsupported(node, "weight / bias are not contiguous [N] views", w=repr(w), b=repr(b))
        if not (x.dtype == y.dtype == w.dtype == b.dtype == "bfloat16"):
            return unsupported(node, "LayerNormAten is bf16 (x, y, weight, bias)", dtypes=[x.dtype, y.dtype, w.dtype, b.dtype])
        if eps is None or eps != self.eps:
            return unsupported(node, "eps differs from the profile's EPS", eps=eps, pinned=self.eps)
        if N % layer_norm.VEC:
            return unsupported(node, "N is not a multiple of vec_size=4: torch takes the non-vectorized LayerNorm path (not modelled)", N=N)
        if not any(r.role == "weight" for r in ctx.roots_of(w)) or not any(r.role == "weight" for r in ctx.roots_of(b)):
            return unsupported(node, "layer_norm weight / bias are not weight Roots", w=repr(w), b=repr(b))
        roots = ctx.roots_of(w)
        return _compute(node, ctx, bind(layer_norm.LayerNormAten, N=N, EPS=self.eps), M, (PerRow(x), Shared(w), Shared(b)), (PerRow(y),),
                        self.name, KERNELS_LAYER_NORM, roots[0].name if roots else "layer_norm", consumed=tuple(sorted(node.ids())))


@dataclass
class GeluErfPattern:
    """`aten.gelu(x[M,N] bf16, approximate='none')` -> `GeluErf{N}` rows=M."""

    name: str = "act.aten_gelu_erf"

    def match(self, node: TreeNode, ctx: StepContext) -> Resolution | None:
        if node.is_triton or base_op(node.op) != "aten.gelu":
            return None
        a = bound_args(node)
        x = a.get("self", a.get("arg0"))
        approx = a.get("approximate", "none")
        outs = node.output_views()
        if not isinstance(x, ViewDesc) or not outs:
            return unsupported(node, "gelu arguments not bound as expected", args={k: _fmt(v) for k, v in a.items()})
        if approx not in (None, "none"):
            return unsupported(node, "gelu with the tanh approximation is not this Definition (see quarantine.dense GeluTanh*)", approximate=approx)
        y = outs[0]
        if x.rank != 2 or not is_row_major(x) or y.shape != x.shape or not is_row_major(y):
            return unsupported(node, "gelu over views that are not row-major [M,N]", x=repr(x), y=repr(y))
        if not (x.dtype == y.dtype == "bfloat16"):
            return unsupported(node, "GeluErf is bf16", dtypes=[x.dtype, y.dtype])
        M, N = x.shape
        return _compute(node, ctx, bind(act.GeluErf, N=N), M, (PerRow(x),), (PerRow(y),), self.name, KERNELS_ELEMENTWISE, "gelu")


@dataclass
class ResidualAddPattern:
    """`aten.add.Tensor(a[M,N] bf16, b[M,N] bf16)` with neither operand a weight Root (the parallel residual `mlp + attn`, then
    `+ hidden_states`) -> `ResidualAdd_v1{N}` rows=M (b1: `Bf16Add`, one IEEE f32 add, RNE).  `[M,N] + [N]` is `BiasAdd`'s."""

    name: str = "residual_add.aten_add"

    def match(self, node: TreeNode, ctx: StepContext) -> Resolution | None:
        if node.is_triton or base_op(node.op) != "aten.add":
            return None
        args = node.args
        if len(args) < 2 or not isinstance(args[0], ViewDesc) or not isinstance(args[1], ViewDesc):
            return None
        x, r = args[0], args[1]
        outs = node.output_views()
        if x.rank != 2 or r.rank != 2 or x.shape != r.shape or not outs:
            return None
        if any(rt.role == "weight" for rt in ctx.roots_of(x)) or any(rt.role == "weight" for rt in ctx.roots_of(r)):
            return None
        alpha = bound_args(node).get("alpha", 1)
        if alpha not in (1, 1.0):
            return unsupported(node, "aten.add with alpha != 1", alpha=alpha)
        y = outs[0]
        if not (x.dtype == r.dtype == y.dtype == "bfloat16") or not is_row_major(x) or not is_row_major(r) or y.shape != x.shape or not is_row_major(y):
            return unsupported(node, "ResidualAdd_v1 is bf16 over row-major [M,N] views of one shape", x=repr(x), r=repr(r), y=repr(y))
        M, N = x.shape
        return _compute(node, ctx, bind(b1.ResidualAdd, N=N), M, (PerRow(x), PerRow(r)), (PerRow(y),), self.name, KERNELS_ELEMENTWISE, "residual")


@dataclass
class RotaryEmbeddingPartial:
    """`_C.rotary_embedding(positions, query!, key!?, head_size=D, cos_sin_cache[MAX_POS, ROT], is_neox=True)` with ROT <= D (GPT-NeoX
    `rotary_pct`) -> `ComputeGroup(RoPE_v1{NHEADS, D=ROT}, rows=M, (PerRow(q[:, :, :ROT] as [M,NH,ROT]), Gather(cos_sin_cache,
    positions)), (PerRow(the same sub-view),))`, the k rotation in `.also`.  vLLM's `rotary_embedding_kernel` (NeoX) applies
    `x_i*cos - x_{i+R}*sin` / `x_{i+R}*cos + x_i*sin` to dims 0..ROT-1 of each head (R = ROT/2) and never touches dims ROT..D-1, so
    the compute writes exactly the rotated words.  ROT == D is the base `RoPE` pattern's case and is left to it."""

    rot: int
    name: str = "rope.rotary_embedding_partial"

    def match(self, node: TreeNode, ctx: StepContext) -> Resolution | None:
        if node.is_triton or op_name(node.op) != "rotary_embedding":
            return None
        a = bound_args(node)
        pos = a.get("positions", a.get("arg0"))
        q = a.get("query", a.get("arg1"))
        k = a.get("key", a.get("arg2"))
        D = a.get("head_size", a.get("arg3"))
        cs = a.get("cos_sin_cache", a.get("arg4"))
        neox = a.get("is_neox", a.get("arg5"))
        if not isinstance(pos, ViewDesc) or not isinstance(q, ViewDesc) or not isinstance(cs, ViewDesc) or not isinstance(D, int):
            return unsupported(node, "rotary_embedding arguments not bound as expected", args={k_: _fmt(v) for k_, v in a.items()})
        if cs.rank != 2 or cs.shape[1] >= D:
            return None                                          # full rotary: the base pattern's case
        ROT = cs.shape[1]
        if ROT != self.rot:
            return unsupported(node, "cos_sin_cache width differs from the profile's ROT", width=ROT, pinned=self.rot, D=D)
        if neox is not True:
            return unsupported(node, "RoPE_v1 is the NeoX rotation (is_neox must be True)", is_neox=neox)
        if not is_row_major(cs) or cs.dtype != "bfloat16" or ROT % 2:
            return unsupported(node, "cos_sin_cache is not a row-major bf16 [MAX_POS, ROT] table with even ROT", cs=repr(cs), ROT=ROT)
        M = q.shape[0]
        reason = _positions_ok(pos, M, ctx)
        if reason:
            return unsupported(node, reason)
        positions = tuple(int(p) for p in ctx.start.positions[:M])
        if any(p < 0 or p >= cs.shape[0] for p in positions):
            return unsupported(node, "a position is outside the cos/sin table", positions=positions[:8], rows=cs.shape[0])
        computes: list[Compute] = []
        for which, view in (("q", q), ("k", k)):
            if view is None:
                continue
            if not isinstance(view, ViewDesc):
                return unsupported(node, f"{which} is not a tensor view")
            if view.dtype != "bfloat16":
                return unsupported(node, "RoPE_v1 is bf16", dtype=view.dtype)
            width = view.shape[-1] if view.rank == 2 else (view.shape[1] * view.shape[2] if view.rank == 3 else 0)
            if width % D or width == 0 or view.shape[0] != M:
                return unsupported(node, f"{which} view is not [M, NHEADS*D]", view=repr(view), D=D, M=M)
            nheads = width // D
            try:
                v3 = split_heads(view, nheads, D)
            except GeometryError as e:
                return unsupported(node, str(e))
            sub = ViewDesc(v3.alloc, v3.offset, (M, nheads, ROT), v3.strides, v3.dtype, v3.device)   # dims 0..ROT-1 of every head
            computes.append(Compute(bind(b1.RoPE, NHEADS=nheads, D=ROT), M, (PerRow(sub), Gather(cs, positions)), (PerRow(sub),),
                                    (node.id,) if not computes else (), hint=hint_of(node, f"rope.{which}[:{ROT}]"),
                                    kernels=("rotary_embedding_kernel",), attribution=attribution_for(ctx, M, node.phase), pattern=self.name))
        if not computes:
            return unsupported(node, "rotary_embedding without a query tensor")
        return _group_compute(computes[0], also=tuple(computes[1:]))


def ln_patterns(C: dict[str, Any]) -> list:
    return [LayerNormAtenPattern(eps=float(C["EPS"])), GeluErfPattern(), ResidualAddPattern()]


def with_ln_patterns(patterns: list, C: dict[str, Any]) -> list:
    """The base pattern list with `RotaryEmbeddingPartial` ahead of the base `RotaryEmbedding` (which reports partial rotary as
    Unsupported) and the LayerNorm / GELU / residual patterns ahead of the base `BiasAdd` / `TransparentAten` tail."""
    out: list = []
    inserted = False
    rot = int(C.get("ROT") or C["D"])
    for p in patterns:
        if type(p) is RotaryEmbedding and rot != C["D"]:
            out.append(RotaryEmbeddingPartial(rot=rot))
        if not inserted and type(p) is BiasAdd:
            out.extend(ln_patterns(C))
            inserted = True
        out.append(p)
    if not inserted:
        out.extend(ln_patterns(C))
    return out
