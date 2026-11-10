"""Multi-node pattern rules: a decomposed computation whose *whole* dataflow is recognised and bound to one registered
kind per row. The pattern is checked structurally on the exported graph (producers, single users, static mask
values, the scale constant); anything that deviates is left to the single-op rules (which then report the honest
`Unsupported`). No node of the pattern may be read from outside it.

Eager (decomposed) causal attention — HF `eager_attention_forward` / GPT-2 `_attn`:

    s = matmul(q[NH,T,D], k^T[NH,D,T]) ; s = s / sqrt(D) (or s * D^-0.5) ; s = masked_fill(s, ~tril, -inf) (or s + mask)
    p = softmax(s (as f32), -1) (.to(dt)) ; y = matmul(p, v[NH,T,D])

-> `EagerAttentionRef_v1{T=i+1,NH,KVH,D,DT,SCALE_OP}` per query row i (NUMERIC: omitting the masked columns is exact
because exp(-inf) = 0 and adding 0.0 / finfo.min to a visible score does not change the softmax).
"""

from __future__ import annotations

import math

import numpy as np
import torch

from verity.ml.circuits.refs import Coll, array_of
from verity.ml.circuits.types import Array, Value
from verity_vllm.program.frontend.derive import StaticIndex, _LiftedConstant
from verity_vllm.program.frontend.strided_views import _permute, _tensor_type
from verity_vllm.program.frontend.value_types import shape_of
from verity_vllm.program.frontend.rules.base import OpRule, RuleInfo
from verity_vllm.program.frontend.rules.common import shape

aten = torch.ops.aten
_DT = {torch.float32: "f32", torch.bfloat16: "bf16"}
_SOFTMAX = {aten._softmax.default, aten.softmax.int}
# dtype casts, including the device-carrying spellings `.to(device=…, dtype=…)` / `.to(dtype=…, layout=…, device=…)` (the
# device is not semantics; every use below checks the dtypes)
_CASTS = {aten._to_copy.default, aten.to.dtype, aten.to.device, aten.to.dtype_layout}
_MATMUL = {aten.matmul.default, aten.bmm.default}


class Consumed:
    """The value of a graph node that a pattern rule absorbed: it exists only inside the pattern."""

    def __init__(self, rule_id: str, anchor) -> None:
        self.rule_id, self.anchor = rule_id, anchor

    def __repr__(self) -> str:
        return f"Consumed(by {self.rule_id} at {self.anchor.name})"


def _constant_tensor(ctx, c: _LiftedConstant):
    """The real tensor of a lifted constant, or None when only a meta tensor is known."""
    t = getattr(ctx.ep, "constants", {}).get(c.name)
    if t is None or getattr(t, "is_meta", False) or t.device.type == "meta":
        return None
    return t


def _ensure_static(ctx, node) -> bool:
    """Translate `node` (a mask built in forward, later in graph order than the scores) now, if its whole producer
    chain is static structure (constant makers / integer & bool arithmetic). Returns False if a value flows in."""
    if node in ctx.env:
        return not isinstance(ctx.env[node], Coll)
    if node.op != "call_function":
        return False
    from verity_vllm.program.frontend.rules.views import _STATIC_ARITH, _STATIC_MAKERS, StaticIndexRule
    if node.target not in _STATIC_MAKERS and node.target not in _STATIC_ARITH and node.target not in StaticIndexRule().targets():
        return False
    for a in node.args:
        for x in (a if isinstance(a, (list, tuple)) else [a]):
            if isinstance(x, torch.fx.Node) and not _ensure_static(ctx, x):
                return False
    from verity_vllm.program.frontend.derive import _translate_node
    saved = ctx.current
    ctx.current = node
    try:
        _translate_node(ctx, node)
    finally:
        ctx.current = saved
    return isinstance(ctx.env.get(node), StaticIndex)


_ASSERTS = {aten._assert_tensor_metadata.default}


def _single_user(node):
    users = [u for u in node.users if u.target not in _ASSERTS]  # metadata assertions are not reads
    return users[0] if len(users) == 1 else None


def _scalar(node, i):
    a = node.args[i] if len(node.args) > i else None
    return a if isinstance(a, (int, float)) else None


class EagerAttentionPatternRule(OpRule):
    modes = frozenset({"reference"})
    info = RuleInfo("EagerAttention[reference]", 1, "op-binding",
                    applies_to="matmul(q, k^T) -> div|mul by the head scale -> masked_fill(-inf | finfo.min) or + additive causal mask -> "
                               "[to f32] -> softmax(-1) -> [to dt] -> matmul(p, v), on f32 or bf16 values",
                    preconditions=("q [NH,T,D] (or [1,NH,T,D]), k^T [NH,D,T], v [NH,T,D] are value collections of one dtype in {f32, bf16}",
                                   "every intermediate has exactly one user and is read nowhere else",
                                   "the scale is a Python float equal (as f32) to sqrt(D) (div) or D^-0.5 (mul)",
                                   "the mask is static (StaticIndex bool / lifted constant) and equals the causal pattern: masked iff key > query",
                                   "softmax is over the last axis; casts (if any) are bf16<->f32 around it",
                                   "one EagerAttentionRef{T=i+1,...} call per query row: keys 0..i only (exact, see NUMERIC's note)"),
                    evidence_domain="numerics are EagerAttentionRef_v1's (DotRef chains, one DT rounding per stated point, SoftmaxRef) — reference-defined")

    def targets(self):
        return _MATMUL

    # -- pattern ------------------------------------------------------------------------------------------------

    def _match(self, node, ctx):
        """Return the matched chain (dict) or a decline string."""
        qn, ktn = node.args[0], node.args[1]
        for nm, x in (("q", qn), ("k^T", ktn)):
            if not isinstance(x, torch.fx.Node) or not isinstance(ctx.val(x), Coll):
                return f"{nm} is not a value collection"
        DT = _DT.get(ctx.dtype_of(qn))
        if DT is None or _DT.get(ctx.dtype_of(ktn)) != DT:
            return f"dtypes {ctx.dtype_of(qn)} / {ctx.dtype_of(ktn)}"
        sq, sk = shape(ctx, qn), shape(ctx, ktn)
        if sq is None or sk is None or len(sq) != len(sk) or len(sq) not in (3, 4) or (len(sq) == 4 and sq[0] != 1):
            return f"q {sq} @ k^T {sk} is not [NH,T,D] @ [NH,D,T]"
        NH, T, D = sq[-3], sq[-2], sq[-1]
        if tuple(sk[-3:]) != (NH, D, T):
            return f"k^T {sk} is not [NH,D,T] for q {sq}"
        scale = _single_user(node)
        if scale is None or scale.target not in (aten.div.Tensor, aten.mul.Tensor, aten.div.Scalar, aten.mul.Scalar):
            return "matmul(q,k^T) is not followed by a single scale op"
        c = _scalar(scale, 1)
        if c is None:
            return "the scale is not a Python scalar"
        op = "div" if scale.target in (aten.div.Tensor, aten.div.Scalar) else "mul"
        want = float(D) ** 0.5 if op == "div" else float(D) ** -0.5
        if np.float32(c) != np.float32(want):
            return f"scale {c} is not {'sqrt' if op == 'div' else '1/sqrt'}(D={D}) as f32"
        mask_node = _single_user(scale)
        if mask_node is None:
            return "the scaled scores have several users"
        masked_kind = None
        if mask_node.target is aten.masked_fill.Scalar:
            fill = _scalar(mask_node, 2)
            if isinstance(mask_node.args[1], torch.fx.Node) and not _ensure_static(ctx, mask_node.args[1]):
                return "the mask is computed from values (not static structure)"
            mval = ctx.val(mask_node.args[1]) if isinstance(mask_node.args[1], torch.fx.Node) else None
            if fill is None or not (math.isinf(fill) and fill < 0 or fill <= float(torch.finfo(ctx.dtype_of(qn)).min)):
                return f"masked_fill value {fill} is not -inf / finfo.min"
            if not isinstance(mval, (StaticIndex, _LiftedConstant)):
                return "the mask is not static structure"
            mt = mval.tensor if isinstance(mval, StaticIndex) else _constant_tensor(ctx, mval)
            if mt is None:
                return "the mask is a lifted constant whose values are not available (meta export)"
            if mt.dtype is not torch.bool:
                return f"mask dtype {mt.dtype} is not bool"
            masked = mt.detach().cpu()
            masked_kind = "masked_fill"
        elif mask_node.target is aten.add.Tensor:
            other = mask_node.args[1] if mask_node.args[0] is scale else mask_node.args[0]
            mval = ctx.val(other) if isinstance(other, torch.fx.Node) else None
            if not isinstance(mval, _LiftedConstant):
                return "the additive mask is not a lifted constant"
            mt = _constant_tensor(ctx, mval)
            if mt is None:
                return "the additive mask is a lifted constant whose values are not available (meta export)"
            mt = mt.detach().cpu().float()
            lo = float(torch.finfo(ctx.dtype_of(qn)).min)
            if not bool(((mt == 0) | (mt <= lo)).all()):
                return "the additive mask has entries other than 0 / finfo.min"
            masked = mt != 0
            masked_kind = "additive"
        else:
            return f"{mask_node.target} is not a causal mask op"
        # the mask must be exactly causal on [T, T] (leading broadcast dims of size 1 allowed)
        m2 = masked.reshape(-1, *masked.shape[-2:]) if masked.dim() >= 2 else None
        if m2 is None or tuple(masked.shape[-2:]) != (T, T):
            return f"mask shape {tuple(masked.shape)} is not [.., T={T}, T]"
        causal = torch.ones(T, T, dtype=torch.bool).triu(1)
        if not all(bool(torch.equal(m2[b], causal)) for b in range(m2.shape[0])):
            return "the mask is not the causal pattern (masked iff key > query)"
        cur = _single_user(mask_node)
        if cur is None:
            return "the masked scores have several users"
        casts = []
        if cur.target in _CASTS:
            if ctx.dtype_of(cur) is not torch.float32:
                return "cast before softmax is not to f32"
            casts.append(cur)
            cur = _single_user(cur)
            if cur is None:
                return "the f32 scores have several users"
        if cur.target not in _SOFTMAX:
            return f"{cur.target} is not softmax"
        dim = cur.args[1] if len(cur.args) > 1 else cur.kwargs.get("dim", -1)
        if dim not in (-1, 2, 3):
            return f"softmax dim {dim} is not the last axis"
        if cur.target is aten._softmax.default and len(cur.args) > 2 and cur.args[2] and ctx.dtype_of(qn) is not torch.bfloat16:
            return "half_to_float on a non-bf16 input"
        sm = cur
        prev = sm
        cur = _single_user(sm)
        if cur is None:
            return "softmax has several users"
        if cur.target in _CASTS:
            if ctx.dtype_of(cur) is not ctx.dtype_of(qn):
                return "cast after softmax is not back to the input dtype"
            casts.append(cur)
            prev = cur
            cur = _single_user(cur)
            if cur is None:
                return "the probabilities have several users"
        if cur.target not in _MATMUL or cur.args[0] is not prev:
            return "the probabilities are not the left operand of a matmul"
        vn = cur.args[1]
        if not isinstance(vn, torch.fx.Node) or not isinstance(ctx.val(vn), Coll) or _DT.get(ctx.dtype_of(vn)) != DT:
            return "v is not a value collection of the input dtype"
        sv = shape(ctx, vn)
        if sv is None or tuple(sv[-3:]) != (NH, T, D) or len(sv) != len(sq):
            return f"v {sv} is not [NH,T,D]"
        return {"q": qn, "kT": ktn, "v": vn, "DT": DT, "NH": NH, "T": T, "D": D, "scale_op": op, "scale": c, "mask": masked_kind,
                "chain": [node, scale, mask_node, *casts[:1], sm, *casts[1:], cur], "anchor": cur, "batched": len(sq) == 4}

    def applies(self, node, ctx):
        m = self._match(node, ctx)
        return True if isinstance(m, dict) else f"not the eager-attention pattern ({m})"

    def emit(self, node, ctx):
        m = self._match(node, ctx)
        assert isinstance(m, dict)
        q, kT, v = ctx.coll(m["q"], "q"), ctx.coll(m["kT"], "k^T"), ctx.coll(m["v"], "v")
        NH, T, D, DT = m["NH"], m["T"], m["D"], m["DT"]
        w = 32 if DT == "f32" else 16
        if m["batched"]:
            q = q.reshape(Array(NH, Array(T, Array(D, Value(w)))))
            kT = kT.reshape(Array(NH, Array(D, Array(T, Value(w)))))
            v = v.reshape(Array(NH, Array(T, Array(D, Value(w)))))
        k = _permute(kT, (0, 2, 1), "k")                      # [NH][T][D]
        q_t = _permute(q, (1, 0, 2), "q rows")                # [T][NH][D]
        kc = _permute(k, (1, 0, 2), "k cache")                # [T][NH][D]
        vc = _permute(v, (1, 0, 2), "v cache")
        outs = []
        for i in range(T):
            fn = ctx.kind("eager_attention", node, T=i + 1, NH=NH, KVH=NH, D=D, DT=DT, SCALE_OP=m["scale_op"])
            outs.append(ctx.call(fn, q_t[i], kc[0:i + 1], vc[0:i + 1], node=m["anchor"], rule=self, name=f"{ctx.hint_name(node)}[t={i}]"))
        y = _permute(array_of(outs), (1, 0, 2), "attention out")  # [NH][T][D]
        if m["batched"]:
            y = y.reshape(Array(1, Array(NH, Array(T, Array(D, Value(w))))))
        for n in m["chain"][:-1]:
            ctx.env[n] = Consumed(self.rule_id, m["anchor"])
            for u in n.users:
                if u.target in _ASSERTS:
                    ctx.env[u] = None
        ctx.define(m["anchor"], y)
        ctx.record(self, node, {"dtype": DT, "NH": NH, "T": T, "D": D, "scale_op": m["scale_op"], "scale": m["scale"], "mask": m["mask"],
                                "causal_mask_verified": True, "chain": [n.name for n in m["chain"]]},
                   emitted=f"EagerAttentionRef_v1{{T=1..{T},NH={NH},KVH={NH},D={D},DT={DT},SCALE_OP={m['scale_op']}}} x{T}")


def _users(node):
    return [u for u in node.users if u.target not in _ASSERTS]


class HFRMSNormPatternRule(OpRule):
    """HF `LlamaRMSNorm` / `Qwen2RMSNorm` as the exported graph spells it (the whole chain, single-use intermediates):

        xf = x.to(f32) ; var = mean(pow(xf, 2), [-1], keepdim) ; rs = rsqrt(var + eps) ; h = xf * rs ; y = w * h.to(dt)

    -> `RMSNormRef_v1{N,EPS,DT}` per row — NUMERIC's reference kind is exactly this decomposition (weight multiply AFTER the
    DT cast; vLLM's fused norms multiply before it and are a different function). Fires at the `to(f32)` node; the chain's
    nodes become `Consumed`. Any deviation (other dims, a different eps form, an extra reader) leaves the single-op rules to
    report the honest `Unsupported`."""
    modes = frozenset({"reference"})
    info = RuleInfo("HFRMSNorm[reference]", 1, "op-binding",
                    applies_to="aten.to.dtype(x, f32) -> pow(2) -> mean([-1], keepdim) -> add(eps) -> rsqrt -> mul(xf, .) -> to(dt) -> mul(w, .)",
                    preconditions=("x is a value of bf16 or f32 leaves with a last axis N; w is a value [N]",
                                   "every intermediate has exactly one reader (xf: the pow and the mul); eps is a Python float",
                                   "mean over the last axis only, keepdim; the final cast returns to x's dtype"),
                    evidence_domain="structural; numerics: RMSNormRef_v1 (reference-defined: HF's decomposition, f32 sequential sum)")

    def targets(self):
        # also at `pow`: an f32 input's `to(f32)` is an identity view that the view rules translate first
        return {aten.to.dtype, aten._to_copy.default, aten.pow.Tensor_Scalar}

    def _match(self, node, ctx):
        if node.target is aten.pow.Tensor_Scalar:
            node = node.args[0] if isinstance(node.args[0], torch.fx.Node) else None
            if node is None or node.target not in _CASTS:
                return "pow of a non-cast"
        x = node.args[0]
        if not isinstance(x, torch.fx.Node) or not isinstance(ctx.val(x), Coll):
            return "input is not a value"
        dt_in = ctx.dtype_of(x)
        if dt_in not in _DT:
            return f"input dtype {dt_in}"
        if ctx.dtype_of(node) != torch.float32:
            return "not a cast to f32"
        us = _users(node)
        identity = dt_in == torch.float32  # `x.to(f32)` of an f32 x is an alias of x: other readers of it read x
        if len(us) != 2 and not identity:
            return f"the f32 cast has {len(us)} readers (pattern: pow and mul)"
        pw = next((u for u in us if u.target is aten.pow.Tensor_Scalar), None)
        ml = next((u for u in us if u.target is aten.mul.Tensor and any(a.target is aten.rsqrt.default for a in u.args if isinstance(a, torch.fx.Node))), None)
        if pw is None or ml is None or _scalar(pw, 1) != 2:
            return "readers are not pow(.,2) and mul"
        mean = _single_user(pw)
        if mean is None or mean.target is not aten.mean.dim:
            return "pow is not read by mean.dim only"
        dims = mean.args[1] if len(mean.args) > 1 else None
        keep = mean.args[2] if len(mean.args) > 2 else False
        rank = len(shape(ctx, x))
        if not (isinstance(dims, (list, tuple)) and len(dims) == 1 and dims[0] in (-1, rank - 1) and keep is True):
            return f"mean over {dims} keepdim={keep} (pattern: last axis, keepdim)"
        add = _single_user(mean)
        if add is None or add.target is not aten.add.Tensor or not isinstance(_scalar(add, 1), float):
            return "mean is not read by add(., eps: float) only"
        eps = float(add.args[1])
        rs = _single_user(add)
        if rs is None or rs.target is not aten.rsqrt.default:
            return "add is not read by rsqrt only"
        if _single_user(rs) is not ml or set(a for a in ml.args if isinstance(a, torch.fx.Node)) != {node, rs}:
            return "rsqrt is not multiplied with the f32 input"
        cast = _single_user(ml)
        if cast is None or cast.target not in _CASTS or ctx.dtype_of(cast) != dt_in:
            return "the normalised rows are not cast back to the input dtype"
        out = _single_user(cast)
        if out is None or out.target is not aten.mul.Tensor:
            return "the cast is not read by the weight multiply only"
        w = next((a for a in out.args if isinstance(a, torch.fx.Node) and a is not cast), None)
        if w is None or not isinstance(ctx.val(w), Coll):
            return "no weight value in the final multiply"
        N = shape(ctx, x)[-1]
        if tuple(shape(ctx, w)) != (N,) or ctx.dtype_of(w) != dt_in:
            return f"weight shape {shape(ctx, w)} / dtype {ctx.dtype_of(w)} (pattern: [N] of the input dtype)"
        chain = [pw, mean, add, rs, ml, cast, out] if identity else [node, pw, mean, add, rs, ml, cast, out]
        return {"x": x, "w": w, "eps": eps, "N": N, "DT": _DT[dt_in], "chain": chain, "anchor": out}

    def applies(self, node, ctx):
        m = self._match(node, ctx)
        return True if isinstance(m, dict) else f"not the HF RMSNorm pattern ({m})"

    def emit(self, node, ctx):
        from verity_vllm.program.frontend.rules.common import from_rows, rows_of
        m = self._match(node, ctx)
        assert isinstance(m, dict)
        x, w = ctx.coll(m["x"], "x"), ctx.coll(m["w"], "w")
        rows, lead = rows_of(x)
        fn = ctx.kind("rmsnorm", node, N=m["N"], EPS=m["eps"], DT=m["DT"])
        outs = [ctx.call(fn, r, w, node=m["anchor"], rule=self, name=f"{ctx.hint_name(node)}[{i}]") for i, r in enumerate(rows)]
        for n in m["chain"][:-1]:
            ctx.env[n] = Consumed(self.rule_id, m["anchor"])
            for u in n.users:
                if u.target in _ASSERTS:
                    ctx.env[u] = None
        ctx.define(m["anchor"], from_rows(outs, lead))
        ctx.record(self, node, {"N": m["N"], "EPS": m["eps"], "dtype": m["DT"], "rows": len(rows), "chain": [n.name for n in m["chain"]]},
                   emitted=f"RMSNormRef_v1{{N={m['N']},EPS={m['eps']},DT={m['DT']}}} x{len(rows)}")


# ---------------------------------------------------------------------------------------------------------
# HF rotary embedding: the cos/sin TABLE (a derived Input) and `apply_rotary_pos_emb` (RoPERef per head-row)
# ---------------------------------------------------------------------------------------------------------

_SHAPE_VIEWS = {aten.unsqueeze.default, aten.squeeze.dim, aten.view.default, aten.reshape.default, aten._unsafe_view.default,
                aten.expand.default, aten.alias.default, aten.clone.default}
_STATIC_MAKERS = {aten.arange.default, aten.arange.start, aten.arange.start_step}
_INT_MAX = 9223372036854775807


def _through_views(node):
    """Follow shape-only views and same-dtype casts back to their source node."""
    seen = 0
    while isinstance(node, torch.fx.Node) and seen < 32:
        seen += 1
        if node.target in _SHAPE_VIEWS:
            node = node.args[0]
            continue
        if node.target in _CASTS:
            src = node.args[0]
            v, sv = node.meta.get("val"), getattr(src, "meta", {}).get("val")
            if hasattr(v, "dtype") and hasattr(sv, "dtype") and v.dtype == sv.dtype:
                node = src
                continue
        break
    return node


_POS_OFFSET = {aten.add.Tensor, aten.sub.Tensor, aten.add.Scalar, aten.sub.Scalar}


def _static_positions_source(node):
    """The `arange` a positions tensor is made from through shape views, same-dtype casts and `+` / `-` of a Python int
    (transformers 5.x: `position_ids = arange(T) + past_seen_tokens`); None if anything else feeds it. This is the
    graph-level pre-scan only (it runs before the Builder exists, to declare the table Inputs); the positions actually used are
    the StaticIndex values the view rules compute, which `applies` requires and `emit` records."""
    seen = 0
    while isinstance(node, torch.fx.Node) and seen < 64:
        seen += 1
        if node.target in _STATIC_MAKERS:
            return node
        if node.target in _SHAPE_VIEWS:
            node = node.args[0]
            continue
        if node.target in _CASTS:
            src = node.args[0]
            v, sv = node.meta.get("val"), getattr(src, "meta", {}).get("val")
            if hasattr(v, "dtype") and hasattr(sv, "dtype") and v.dtype == sv.dtype:
                node = src
                continue
            return None
        if node.target in _POS_OFFSET:
            ts = [a for a in node.args[:2] if isinstance(a, torch.fx.Node)]
            ks = [a for a in node.args[:2] if isinstance(a, int) and not isinstance(a, bool)]
            alpha = node.args[2] if len(node.args) > 2 else node.kwargs.get("alpha", 1)
            if len(ts) == 1 and len(ks) == 1 and alpha == 1:
                node = ts[0]
                continue
            return None
        return None
    return None


def _nodes_between(start, end):
    """All view/cast nodes on the single path start -> ... -> end (start excluded, end included)."""
    out, n = [], end
    while n is not start and isinstance(n, torch.fx.Node):
        out.append(n)
        n = n.args[0]
    return out[::-1]


_TABLE_BUFFER_DTYPES = {torch.float32, torch.bfloat16}


def _buffer_source(node):
    """The source of the rotary table's left operand through shape views, same-dtype casts and AT MOST ONE widening cast
    bf16 -> f32 (HF casts the `inv_freq` buffer to bf16 with the model — `model.to(bfloat16)` casts buffers too — so the
    forward's `.float()` is a real cast). Returns `(source node, cast node | None)`; `(None, None)` if anything else feeds it
    (a second dtype change, a cast from any other dtype, an arithmetic op)."""
    seen, cast = 0, None
    while isinstance(node, torch.fx.Node) and seen < 32:
        seen += 1
        if node.target in _SHAPE_VIEWS:
            node = node.args[0]
            continue
        if node.target in _CASTS:
            src = node.args[0]
            v, sv = node.meta.get("val"), getattr(src, "meta", {}).get("val")
            if not (hasattr(v, "dtype") and hasattr(sv, "dtype")):
                return None, None
            if v.dtype == sv.dtype:
                node = src
                continue
            if cast is None and sv.dtype == torch.bfloat16 and v.dtype == torch.float32:
                cast, node = node, src
                continue
            return None, None
        break
    return node, cast


def _stack_path(node) -> str:
    stack = node.meta.get("nn_module_stack") or {}
    return list(stack.values())[-1][0] if stack else ""


class HFRotaryTableRule(OpRule):
    """HF `LlamaRotaryEmbedding.forward`: `freqs = inv_freq[None,:,None] @ positions[:,None,:].float()`; `emb = cat(freqs, freqs)`;
    `cos = emb.cos() * scaling` ; `sin = emb.sin() * scaling` ; `.to(dtype)`. The table is a function of a buffer (`inv_freq`) and
    STATIC positions only; there is no reference kind for cos/sin (a transcendental table is not reproducible bit-exactly from the
    formula), so — exactly as vLLM's `cos_sin_cache` — the two tables become prescribed Inputs (`<rope>.cos_table[T,D]`,
    `<rope>.sin_table[T,D]`, fields of the weights record) with the derivation recorded: source buffer, positions, scaling, dtype.
    `inv_freq` stays a (now unread) field so the record's identity is the module's. Fires at the positions' `to(f32)`; the whole
    chain up to the two final casts becomes `Consumed`; the final casts are the table Inputs.

    v2 (transformers 5.x spellings): the buffer views may include `.to(device=…, dtype=f32)` of the f32 buffer (a same-dtype cast);
    the positions pre-scan looks through `+` / `-` of a Python int (`arange(T) + past_seen_tokens`) — the positions
    bound are still the StaticIndex values, recorded per application.

    v3 (bf16 models): the `inv_freq` buffer may be **bf16** — `model.to(torch.bfloat16)` casts buffers with the parameters, so
    on a real bf16 Llama / Qwen2 / Qwen3 the forward's `inv_freq[None, :, None].float()` is a real widening cast — and then exactly one
    bf16 -> f32 cast (`Bf16ToF32`, exact) is allowed on the buffer's path to the matmul. The table is then `cos|sin(pos * f32(bf16
    inv_freq))`: computed from the **bf16-rounded** inv_freq. That is the computation HF runs for that model (the fixed computation
    this program preserves), not a deviation from it; the derived-Input declaration records the buffer dtype and the cast
    (`source_dtype`, `source_cast`), so a verifier supplying the table computes it from the same bf16 buffer. The cast is part of
    the consumed chain (no `CastF32Bf16` gates are emitted for it): the rule fires at whichever of its two entry casts — the
    buffer's `.float()` or the positions' `.float()` — comes first in graph order, and translates the static positions on demand.
    Any other dtype change on the buffer path (f16, f64, a second cast) is not the pattern.

    v4 (Olmo): the terminal `.to(x.dtype)` may be ABSENT — `OlmoRotaryEmbedding.forward` returns the f32 `cos` / `sin` as
    computed. Then the table is the f32 value the chain produced (cos|sin, or its scalar multiply), that node is the table Input
    (not consumed; the layers read it), and the application records `terminal_cast: none (...)` vs `cast to model dtype`. Both
    tables must agree. A terminal cast to a dtype outside {f32, bf16} is still not the pattern."""
    modes = frozenset({"reference"})
    info = RuleInfo("HFRotaryTable[reference]", 4, "op-binding",
                    applies_to="matmul(views([to(f32)](inv_freq buffer f32|bf16[R])), to(f32)(static positions[T])) -> transpose -> cat(x,x) -> {cos,sin} -> [mul(scalar)] -> [to(dt)]",
                    preconditions=("the left operand is a 1-D f32 or bf16 buffer of the module through shape views, same-dtype casts and (bf16 only) "
                                   "exactly one bf16 -> f32 cast; the buffer dtype and the cast are recorded in the table's derivation",
                                   "the positions are a StaticIndex made from an arange through shape views, same-dtype casts and +/- of a Python int",
                                   "cos and sin are read exactly by an optional scalar multiply and then EITHER one cast each (to f32|bf16; the cast is the "
                                   "table) OR no cast (the f32 cos|sin is the table); which one is recorded as `terminal_cast`, the same for both",
                                   "the resulting tables are [1, T, D] with D = 2R"),
                    evidence_domain="structural; the tables are prescribed Inputs (derived tables), their derivation recorded in the weights record")

    def targets(self):
        return set(_CASTS)

    # -- graph scan (no ctx: also used before the Builder exists, to declare the table inputs) --------------
    def _scan(self, ep) -> list[dict]:
        out = []
        buffers = set(ep.graph_signature.inputs_to_buffers) | set(ep.graph_signature.inputs_to_parameters)
        for mm in ep.graph.nodes:
            if mm.op != "call_function" or mm.target not in {aten.matmul.default, aten.bmm.default}:
                continue
            lhs, rhs = mm.args[0], mm.args[1]
            src, src_cast = _buffer_source(lhs)
            if not (isinstance(src, torch.fx.Node) and src.op == "placeholder" and src.name in buffers):
                continue
            sv = src.meta.get("val")
            if not (hasattr(sv, "dtype") and sv.dtype in _TABLE_BUFFER_DTYPES and sv.ndim == 1):
                continue
            # a bf16 buffer reaches the f32 matmul through its one widening cast; an f32 buffer has none (by construction of the walk)
            if (sv.dtype == torch.bfloat16) != (src_cast is not None):
                continue
            # rhs: to(f32) of an integer static chain
            cast = _through_views(rhs)
            if not (isinstance(cast, torch.fx.Node) and cast.target in _CASTS):
                continue
            pos = cast.args[0]
            pv = getattr(pos, "meta", {}).get("val") if isinstance(pos, torch.fx.Node) else None
            if not (hasattr(pv, "dtype") and not pv.dtype.is_floating_point and _static_positions_source(pos) is not None):
                continue
            tr = _single_user(mm)
            if tr is None or tr.target is not aten.transpose.int:
                continue
            cat = _single_user(tr)
            if cat is None or cat.target is not aten.cat.default or list(cat.args[0]) != [tr, tr]:
                continue
            us = _users(cat)
            cos = next((u for u in us if u.target is aten.cos.default), None)
            sin = next((u for u in us if u.target is aten.sin.default), None)
            if len(us) != 2 or cos is None or sin is None:
                continue
            ends, chain, scaling, terminal = {}, [cast, *_nodes_between(cast, rhs), *_nodes_between(src, lhs), mm, tr, cat], None, None
            ok = True
            for nm, trig in (("cos", cos), ("sin", sin)):
                n, sc, last = _single_user(trig), 1.0, trig
                chain.append(trig)
                if n is not None and n.target is aten.mul.Tensor and isinstance(_scalar(n, 1), (int, float)):
                    sc = float(n.args[1])
                    chain.append(n)
                    last, n = n, _single_user(n)
                if n is not None and n.target in _CASTS:
                    # terminal `.to(x.dtype)` (Llama and most HF classes): the cast IS the table; its dtype the table's
                    if getattr(n.meta.get("val"), "dtype", None) not in _DT:
                        ok = False
                        break
                    term, end = "cast to model dtype", n
                else:
                    # v4: NO terminal cast — the module returns cos/sin as computed (Olmo). The table is then the f32 the
                    # matmul chain produced; `last` (cos|sin or its scalar multiply) is the value the layers read, so it is the table
                    # Input and NOT part of the consumed chain. Structural: the terminal's dtype must be f32 (it is, by the chain).
                    if getattr(last.meta.get("val"), "dtype", None) is not torch.float32:
                        ok = False
                        break
                    term, end = "none (f32 table returned as computed; no cast in the source)", last
                    chain.pop()
                if (scaling is not None and sc != scaling) or (terminal is not None and term != terminal):
                    ok = False
                    break
                scaling, terminal = sc, term
                ends[nm] = end
            if not ok:
                continue
            shp = tuple(int(x) for x in ends["cos"].meta["val"].shape)
            R = int(sv.shape[0])
            if len(shp) != 3 or shp[0] != 1 or shp[2] != 2 * R:
                continue
            out.append({"anchor": cast, "pos": pos, "src": src, "src_cast": src_cast, "src_dtype": sv.dtype, "mm": mm, "ends": ends,
                        "chain": chain, "scaling": scaling, "T": shp[1], "D": 2 * R, "dtype": ends["cos"].meta["val"].dtype,
                        "terminal": terminal, "path": _stack_path(cos)})
        return out

    @staticmethod
    def _source_cast_record(m) -> str | None:
        if m["src_cast"] is None:
            return None
        return (f"{m['src_cast'].name}: {m['src_cast'].target} bf16 -> f32 (Bf16ToF32, exact widening) of the bf16-rounded inv_freq buffer; "
                f"the table is cos|sin(f32(pos) * f32(bf16(inv_freq))) — the computation the source runs for a bf16 model")

    def _field_names(self, m, ep) -> tuple[str, str]:
        base = m["path"] or "rotary"
        same = [x for x in self._scan(ep) if x["path"] == m["path"]]
        suffix = "" if len(same) == 1 else f"[{same.index(m)}]"
        return f"{base}.cos_table{suffix}", f"{base}.sin_table{suffix}"

    def declare_inputs(self, ep, module, mode):
        out = []
        for m in self._scan(ep):
            w = 32 if m["dtype"] == torch.float32 else 16
            t = Array(m["T"], Array(m["D"], Value(w)))
            fqn = ep.graph_signature.inputs_to_buffers.get(m["src"].name) or ep.graph_signature.inputs_to_parameters.get(m["src"].name)
            der = {"formula": "table[t, d] = DT(f32(cos|sin)(f32(pos[t]) * f32(inv_freq[d mod R])) * scaling)", "source_buffer": fqn,
                   "source_dtype": _DT[m["src_dtype"]], "source_cast": self._source_cast_record(m),
                   "scaling": m["scaling"], "dtype": _DT[m["dtype"]], "terminal_cast": m["terminal"],
                   "positions": "recorded at derivation (provenance application)",
                   "qualification": "derived table: the same class as vLLM's cos_sin_cache (values not reproducible bit-exactly from the formula)"}
            cn, sn = self._field_names(m, ep)
            out.append((cn, t, {**der, "table": "cos"}))
            out.append((sn, t, {**der, "table": "sin"}))
        return out

    def _match(self, node, ctx):
        """The scanned pattern whose entry cast is `node`: the positions' `to(f32)` (anchor) or, for a bf16 buffer, the buffer's
        `to(f32)` — whichever torch.export placed first is where the rule fires; the other is then already Consumed."""
        return next((x for x in self._scan(ctx.ep) if x["anchor"] is node or (x["src_cast"] is not None and x["src_cast"] is node)), None)

    def applies(self, node, ctx):
        m = self._match(node, ctx)
        if m is None:
            src = node.args[0] if node.args and isinstance(node.args[0], torch.fx.Node) else None
            sdt = ctx.dtype_of(src) if src is not None else None
            if sdt is not None and not sdt.is_floating_point and ctx.dtype_of(node) == torch.float32 and isinstance(ctx.val(src), Coll):
                return ("positions are a value (an input or computed), not static structure — a rotary table is only derivable "
                        "for statically known positions (arange in forward); pass positions as static structure or derive per allowed binding")
            return "not the HF rotary table pattern"
        # firing at the buffer's cast (bf16 path) may precede the positions' views in graph order: fold the static chain now
        if m["pos"] not in ctx.env and not _ensure_static(ctx, m["pos"]):
            return "positions are not statically known (a value flows into them)"
        if not isinstance(ctx.val(m["pos"]), StaticIndex):
            return "positions are not statically known (a value flows into them)"
        return True

    def emit(self, node, ctx):
        m = self._match(node, ctx)
        assert m is not None
        pos = ctx.val(m["pos"]).flat()
        if len(pos) != m["T"]:
            from verity_vllm.program.frontend.derive import refuse
            refuse("shape", node, ctx.module_path(node), f"{len(pos)} positions for a {m['T']}-row table")
        cn, sn = self._field_names(m, ctx.ep)
        fields = {w.name: w for w in ctx.weights}
        for n in m["chain"]:
            ctx.env[n] = Consumed(self.rule_id, node)
            for u in n.users:
                if u.target in _ASSERTS:
                    ctx.env[u] = None
        for nm, fname in (("cos", cn), ("sin", sn)):
            end = m["ends"][nm]
            tbl = ctx.weights_coll[fields[fname].index]
            ctx.define(end, tbl.reshape(_tensor_type(tuple(int(x) for x in end.meta["val"].shape), 32 if m["dtype"] == torch.float32 else 16)))
            ctx.weight_roles.setdefault(fname, {"rule": self.rule_id, "role": f"rope_{nm}_table", "statics": {"T": m["T"], "D": m["D"]}})
        ctx.record(self, node, {"T": m["T"], "D": m["D"], "dtype": _DT[m["dtype"]], "scaling": m["scaling"], "terminal_cast": m["terminal"], "positions": pos,
                                "source_buffer": fields[cn].derivation["source_buffer"], "source_dtype": _DT[m["src_dtype"]],
                                "source_cast": self._source_cast_record(m), "fired_at": "buffer cast" if node is m["src_cast"] else "positions cast",
                                "fields": [cn, sn]},
                   emitted=f"Inputs {cn}, {sn}: [{m['T']}][{m['D']}] {_DT[m['dtype']]} (derived tables; no gates)")


class HFRoPEPatternRule(OpRule):
    """HF `apply_rotary_pos_emb`: `x * cos + rotate_half(x) * sin` with `rotate_half(x) = cat(-x[..., R:], x[..., :R])`, `cos`/`sin`
    `[1, 1, T, D]` (the table rows, unsqueezed over heads) against `x [B, NH, T, D]` -> `RoPEHeadsRef_v1{NHEADS=NH, D, DT}` per (batch,
    position) — NUMERIC's kind is this decomposition (three DT roundings per output; `-x2 * s` == `-(x2 * s)` exactly). Fires at the
    first multiply; the seven nodes (two muls, two slices, neg, cat, add) become `Consumed`."""
    modes = frozenset({"reference"})
    info = RuleInfo("HFRoPE[reference]", 1, "op-binding",
                    applies_to="add(mul(x, cos), mul(cat(neg(x[..., R:]), x[..., :R]), sin)) with cos, sin [.., 1, T, D] broadcast over heads",
                    preconditions=("x [B, NH, T, D], cos and sin [B|1, 1, T, D] values of one dtype (bf16 / f32); both halves read the same x",
                                   "the slices are exactly [:R] and [R:] with R = D/2 on the last axis; every intermediate has one reader"),
                    evidence_domain="structural; numerics: RoPERef_v1 / RoPEHeadsRef_v1 (reference-defined: HF's three-rounding decomposition)")

    def targets(self):
        return {aten.mul.Tensor}

    def _match(self, node, ctx):
        from verity_vllm.program.frontend.derive import _runs
        args = [a for a in node.args if isinstance(a, torch.fx.Node)]
        if len(args) != 2:
            return "not a tensor-tensor multiply"
        add = _single_user(node)
        if add is None or add.target is not aten.add.Tensor or (len(add.args) > 2 and add.args[2] != 1):
            return "not read by one add"
        other = next((a for a in add.args if isinstance(a, torch.fx.Node) and a is not node), None)
        if other is None or other.target is not aten.mul.Tensor or _single_user(other) is not add:
            return "the add's other operand is not a single-use multiply"
        oargs = [a for a in other.args if isinstance(a, torch.fx.Node)]
        cat = next((a for a in oargs if a.target is aten.cat.default), None)
        if cat is None or len(oargs) != 2 or _single_user(cat) is not other:
            return "no rotate_half cat feeding the second multiply"
        sin = next(a for a in oargs if a is not cat)
        parts = list(cat.args[0])
        if len(parts) != 2 or parts[0].target is not aten.neg.default or parts[1].target is not aten.slice.Tensor:
            return "cat is not (neg(x2), x1)"
        neg, x1 = parts
        x2 = neg.args[0]
        if not (isinstance(x2, torch.fx.Node) and x2.target is aten.slice.Tensor) or _single_user(neg) is not cat or _single_user(x2) is not neg or _single_user(x1) is not cat:
            return "rotate_half slices are not single-use"
        xs = x2.args[0]
        if x1.args[0] is not xs:
            return "the two halves are sliced from different tensors"
        xshape = shape(ctx, xs)
        rank = len(xshape)
        D = xshape[-1]
        R = D // 2
        if D % 2 or x1.args[1] not in (-1, rank - 1) or x2.args[1] not in (-1, rank - 1):
            return "slices are not on the last axis"
        if (x1.args[2] or 0, x1.args[3]) != (0, R) or (x2.args[2], x2.args[3] if x2.args[3] < _INT_MAX else D) != (R, D):
            return f"slices are not [:R] and [R:] (R={R})"
        # which operand of the first multiply is x: same value (references) as xs
        x = cos = None
        for a in args:
            va = ctx.val(a)
            if isinstance(va, Coll) and isinstance(ctx.val(xs), Coll) and _runs(va.refs) == _runs(ctx.val(xs).refs) and shape(ctx, a) == xshape:
                x = a
            else:
                cos = a
        if x is None or cos is None:
            return "the first multiply does not read the rotated tensor"
        for nm, t in (("cos", cos), ("sin", sin), ("x", x)):
            if not isinstance(ctx.val(t), Coll):
                return f"{nm} is not a value"
        dts = {ctx.dtype_of(x), ctx.dtype_of(cos), ctx.dtype_of(sin)}
        if len(dts) != 1 or next(iter(dts)) not in _DT:
            return f"dtypes {dts}"
        if rank != 4:
            return f"x has rank {rank} (pattern: [B, NH, T, D])"
        B_, NH, T, _ = xshape
        cshape, sshape = shape(ctx, cos), shape(ctx, sin)
        if cshape != sshape or len(cshape) != 4 or cshape[1] != 1 or cshape[2:] != (T, D) or cshape[0] not in (1, B_):
            return f"cos {cshape} / sin {sshape} do not broadcast over heads of x {xshape}"
        return {"x": x, "xs": xs, "cos": cos, "sin": sin, "B": B_, "NH": NH, "T": T, "D": D, "DT": _DT[next(iter(dts))], "anchor": add,
                "chain": [node, x1, x2, neg, cat, other, add]}

    def applies(self, node, ctx):
        m = self._match(node, ctx)
        return True if isinstance(m, dict) else f"not the HF rotary pattern ({m})"

    def emit(self, node, ctx):
        m = self._match(node, ctx)
        assert isinstance(m, dict)
        x, cos, sin = ctx.coll(m["x"], "x"), ctx.coll(m["cos"], "cos"), ctx.coll(m["sin"], "sin")
        w = 32 if m["DT"] == "f32" else 16
        B_, NH, T, D = m["B"], m["NH"], m["T"], m["D"]
        cb = cos.reshape(Array(shape(ctx, m["cos"])[0], Array(T, Array(D, Value(w)))))
        sb = sin.reshape(Array(shape(ctx, m["sin"])[0], Array(T, Array(D, Value(w)))))
        outs = []
        for b in range(B_):
            xb = _permute(x[b], (1, 0, 2), "rope rows")  # [T][NH][D]
            cbb, sbb = cb[b if cb.type.n > 1 else 0], sb[b if sb.type.n > 1 else 0]
            rows = []
            for t in range(T):
                fn = ctx.kind("rope_heads", node, NHEADS=NH, D=D, DT=m["DT"])
                rows.append(ctx.call(fn, xb[t], cbb[t], sbb[t], node=m["anchor"], rule=self, name=f"{ctx.hint_name(node)}[t={t}]"))
            outs.append(_permute(array_of(rows), (1, 0, 2), "rope out"))  # [NH][T][D]
        for n in m["chain"][:-1]:
            ctx.env[n] = Consumed(self.rule_id, m["anchor"])
            for u in n.users:
                if u.target in _ASSERTS:
                    ctx.env[u] = None
        ctx.define(m["anchor"], array_of(outs).reshape(_tensor_type((B_, NH, T, D), w)))
        ctx.record(self, node, {"B": B_, "NH": NH, "T": T, "D": D, "dtype": m["DT"], "chain": [n.name for n in m["chain"]]},
                   emitted=f"RoPEHeadsRef_v1{{NHEADS={NH},D={D},DT={m['DT']}}} x{B_ * T}")


PATTERN_RULES = [EagerAttentionPatternRule(), HFRMSNormPatternRule(), HFRotaryTableRule(), HFRoPEPatternRule()]

__all__ = ["PATTERN_RULES", "EagerAttentionPatternRule", "HFRMSNormPatternRule", "HFRotaryTableRule", "HFRoPEPatternRule", "Consumed"]
