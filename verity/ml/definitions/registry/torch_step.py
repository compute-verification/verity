"""A decoder language model's training step as one Program root: `TrainStep_v1{L, B, S}`, L layers over B sequences of S tokens
(M = B·S), as torch 2.14 eager computes it on CUDA with bf16 parameters and an f32 master copy under AdamW: HF transformers'
Llama decoder with a tied embedding, eager attention and the causal-LM cross-entropy, at smollm2-135m's shape (`CONFIG`: hidden
576, 9 query heads on 3 KV heads of 64, MLP 1536, vocabulary 49152), then the gradient clip and the optimizer step. Every node
of its body is one call or batch of a training Definition on bits.

    LinearTorch_v1{M, K, N, ROUTE}     (x: Array<M, Array<K, BF16>>, w: Array<N, Array<K, BF16>>) -> Array<M, Array<N, BF16>>
    LinearTorchDx_v1{M, K, N, ROUTE}   (dy: Array<M, Array<N, BF16>>, wt: Array<K, Array<N, BF16>>) -> Array<M, Array<K, BF16>>
    LinearTorchDw_v1{M, K, N, ROUTE}   (dyt: Array<N, Array<M, BF16>>, xt: Array<K, Array<M, BF16>>) -> Array<N, Array<K, BF16>>
    TrainStep_v1{L, B, S}              (ids, labels: Array<M, U32>, n: Array<S, U16> of kind position, cs: Array<S, Array<D, BF16>>,
                                        each weight W.w (bf16), then each weight's W.master, W.exp_avg, W.exp_avg_sq (f32, its
                                        shape), step: F32, lr, beta1, beta2, eps, wd, max_norm: F64)
                                       -> Tuple<loss: F32, total_norm: BF16, step: F32, ok: BIT, then each weight's new
                                                (W.w, W.master, W.exp_avg, W.exp_avg_sq)>

The weights W are `weights(L)`: embed, layer i's l{i}.NAME (`LAYER`) and norm, in HF's ``named_parameters`` order. A weight's
parameter is its node's name with ``.w``, so a walk's values name both apart (`program_walk`). step is the optimizer's step
count before this step, the same for every tensor; the hyperparameters are private values.

`LinearTorch`, `LinearTorchDx` and `LinearTorchDw` are F.linear's three GEMMs at x (M, K), W (N, K): the forward, dX = dY·W and
dW = dYᵀ·X, each ROUTE's row GEMM (`cublas_bf16.route_gemm`) batched over its first operand's rows with the second shared; the
root binds each at `cublas_bf16.train_route`'s route for its shape, so its body is `cublas_bf16.train_linear`'s. A
transposed operand (Wᵀ, dYᵀ, Xᵀ) comes in as its `Transpose_v1` copy, a node of the root, so one copy serves every GEMM that
reads it: a layer's normed input feeds the dW of q, k and v.

**The forward**, layer i's nodes named ``l{i}.NAME``:

    embed               EmbeddingTorch_v1{M, V, H} of the ids
    ln1, ln2            RmsNormTorch_v1 on each token: (y, hb, r)
    q, k, v, o          LinearTorch_v1 on ln1's y (o on attention's output)
    attn                EagerAttentionTorch_v1 on each sequence: (o, qe, ke, y, p)
    h2                  Bf16AddF2fp_v2 of the layer's input and o, the residual add
    gate, up, down      LinearTorch_v1 (gate and up on ln2's y, down on mul)
    act, mul            SiluTorch_v1 on gate, ACT_N words a call; Bf16MulBf16_v2 of act and up
    h3                  Bf16AddF2fp_v2 of h2 and down, the next layer's input

then ``norm`` (RmsNormTorch_v1) and ``logits`` (LinearTorch_v1 against the embedding), and the loss: ``constants``
(TrainStepConstants_v1, the words -100 and 1.0, so that the body holds no gate of its own), HF's labels shifted one token left
within each sequence with -100 (0xFFFFFF9C, outside [0, V)) at its end; ``loss.rows`` (LogSoftmaxRow_v1 on each row), ``loss``
(NllLossTorch_v1, the mean), ``loss.dx`` (NllLossTorchBackward_v1 of the seed 1.0) and ``logits.dx``
(LogSoftmaxNllBackwardRow_v1 on each row).

**The backward**, in autograd's order and with its operands' order wherever two gradients meet: ``lm_head.dx`` and
``lm_head.dw``; ``norm.dx`` (RmsNormTorchBackward_v1) and ``norm.dw`` (RmsNormTorchWeightGrad_v1 on each column); then from
the last layer down ``down.dx``/``.dw``, ``mul.da`` (the gradient into act, its dY times up) and ``mul.db`` (into up, dY times
act), ``up.dx``/``.dw``, ``act.dx`` (SiluTorchBackward_v1), ``gate.dx``/``.dw``, ``mlp.dx`` (up's dX plus gate's),
``ln2.dx``/``.dw``, ``h2.grad`` (the layer output's gradient plus ln2's dx), ``o.dx``/``.dw``, ``attn.dx``
(EagerAttentionTorchBackward_v1: dq, dk, dv), ``v.dx``/``.dw``, ``k.dx``/``.dw``, ``q.dx``/``.dw``, ``kv.dx`` (v's dX plus
k's), ``qkv.dx`` (that plus q's), ``ln1.dx``/``.dw`` and ``x.grad`` (h2's gradient plus ln1's dx), the gradient of the layer's
input; and ``embed.dw``, EmbeddingBackwardTied_v2 of layer 0's input gradient and lm_head's dW.

**The optimizer**, the recipe's ``clip_grad_norm_(params, max_norm, foreach=True)`` on the bf16 gradients, then
``master.grad = p.grad.float()``, ``AdamW(masters, foreach=True).step()`` and ``p.copy_(master)``: ``clip``
(ClipGradNorm_v1 over every gradient in the parameters' order: the total norm and the clipped gradients), ``adamw.scalars``
(AdamWScalars_v1, once a step: the kernels' seven f32 scalars, the incremented step and ok, the conjunction of what would make
torch raise), and per weight W ``W.grad`` (Bf16ToF32_v2 on each element), ``W.adamw`` (AdamWForeach_v1 on each run of ADAM_N
elements, LERP_FMA 1 as the sm_120 SASS has it) and ``W.copy`` (F2fpBf16_v2, cvt.rn.bf16.f32, on each element).

The rotary table cs (each position's cos and sin halves, bf16) and n (row i sees n[i] = i + 1 keys, the causal mask) are
inputs: the rotary embedding's f32 products and cos/sin have no Definition yet.
"""
from __future__ import annotations

from typing import Any

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.ops.boolean import cast as CA
from verity.ml.circuits.ops.boolean import elementwise as EW
from verity.ml.circuits.ops.boolean import gemm as BG
from verity.ml.circuits.ops.boolean import mufu as BMU
from verity.ml.circuits.ports import In, Ports
from verity.ml.circuits.refs import Coll, Explicit, concat, strided_view, tuple_of
from verity.ml.circuits.types import Array, Tuple

from . import cublas_bf16 as CB
from . import torch_attention as TA
from . import torch_clip as TC
from . import torch_embedding as TE
from . import torch_loss as TL
from . import torch_optim as TO
from . import torch_train as TT

__all__ = ["ACT_N", "ADAM_N", "CONFIG", "EPS", "HYPER", "LAYER", "LinearTorch", "LinearTorchDw", "LinearTorchDx", "STATE",
           "TrainStep", "TrainStepConstants", "params", "weights"]

BF16, F32, F64, U16, U32 = (Array(w, BL.BIT) for w in (16, 32, 64, 16, 32))
#: the model's shape
CONFIG = {"H": 576, "NH": 9, "KVH": 3, "D": 64, "I": 1536, "V": 49152}
#: rms_norm_eps, 1e-5, as its f32 word
EPS = 0x3727C5AC
#: words of one SiluTorch_v1 / SiluTorchBackward_v1 call, a third of a token's MLP row
ACT_N = 512
#: a layer's weights in HF's ``named_parameters`` order, the order clip_grad_norm_'s ``_foreach_norm`` reads the gradients in:
#: name -> (rows, columns)
LAYER = {"q": ("NH·D", "H"), "k": ("KVH·D", "H"), "v": ("KVH·D", "H"), "o": ("H", "NH·D"), "gate": ("I", "H"),
         "up": ("I", "H"), "down": ("H", "I"), "ln1": (None, "H"), "ln2": (None, "H")}
#: AdamW's f32 state of a weight, as its parameters' suffixes: the master copy, exp_avg and exp_avg_sq
STATE = ("master", "exp_avg", "exp_avg_sq")
#: the private hyperparameters, f64 each: AdamW's (lr, beta1, beta2, eps, weight_decay), then the clip's max_norm
HYPER = ("lr", "beta1", "beta2", "eps", "wd", "max_norm")
#: elements of one AdamWForeach_v1 proof block, a run of one tensor
ADAM_N = 64
#: -100, HF's ignore index, as the low 32 bits of its int64 label
IGNORE = 0xFFFFFF9C
#: the loss's gradient seed, 1.0
ONE = 0x3F800000
_POSITION = Ports(params={"n": In(kind="position")})


def _dims() -> dict[str, int]:
    c = CONFIG
    return {"H": c["H"], "I": c["I"], "NH·D": c["NH"] * c["D"], "KVH·D": c["KVH"] * c["D"]}


def _row_gemm(role: str, route: Any, M: int, K: int, N: int) -> Any:
    if not isinstance(route, dict):
        raise TypeError(f"ROUTE = {route!r} is not a route record")
    return CB.route_gemm(dict(route), *CB.TRAIN_LINEAR_ROLES[role][2](M, K, N))


def _route_refusal(role: str):
    def refuse(b: dict) -> str | None:
        if not all(k in b for k in ("M", "K", "N", "ROUTE")):
            return None
        try:
            _row_gemm(role, b["ROUTE"], int(b["M"]), int(b["K"]), int(b["N"]))
        except (TypeError, ValueError, KeyError) as e:
            return f"{role}: {e}"
        return None
    return refuse


def _rows(B, role: str, S: Any, a: Coll, b: Coll) -> Coll:
    return B.batch(_row_gemm(role, S.ROUTE, S.M, S.K, S.N), a, b, axes=(0, None))


def _mat(r: int, c: int) -> Array:
    return Array(r, Array(c, BF16))


_CONFORMANCE = ("exact-model-tested; the batch over the rows of ROUTE's row GEMM (cublas_bf16.route_gemm); the root binds "
                "cublas_bf16.train_route's route, the cuBLAS capture's (cublas_bf16.TRAIN_CAPTURE)")
LinearTorch = CompositeDefinition(
    "LinearTorch", 1, ("M", "K", "N", "ROUTE"),
    lambda S: ((("x", _mat(S.M, S.K)), ("w", _mat(S.N, S.K))), _mat(S.M, S.N)),
    lambda B, S, x, w: _rows(B, "fwd", S, x, w),
    doc="F.linear's forward at x (M, K), W (N, K): ROUTE's row GEMM on each of x's rows against W's rows",
    conformance=_CONFORMANCE, refuse=_route_refusal("fwd"))
LinearTorchDx = CompositeDefinition(
    "LinearTorchDx", 1, ("M", "K", "N", "ROUTE"),
    lambda S: ((("dy", _mat(S.M, S.N)), ("wt", _mat(S.K, S.N))), _mat(S.M, S.K)),
    lambda B, S, dy, wt: _rows(B, "dX", S, dy, wt),
    doc="F.linear's input gradient dX = dY·W at x (M, K), W (N, K): ROUTE's dX row GEMM on each of dY's rows against the "
        "rows of Wᵀ, its Transpose_v1 copy",
    conformance=_CONFORMANCE, refuse=_route_refusal("dX"))
LinearTorchDw = CompositeDefinition(
    "LinearTorchDw", 1, ("M", "K", "N", "ROUTE"),
    lambda S: ((("dyt", _mat(S.N, S.M)), ("xt", _mat(S.K, S.M))), _mat(S.N, S.K)),
    lambda B, S, dyt, xt: _rows(B, "dW", S, dyt, xt),
    doc="F.linear's weight gradient dW = dYᵀ·X at x (M, K), W (N, K): ROUTE's dW row GEMM on each row of dYᵀ against the "
        "rows of Xᵀ, both Transpose_v1 copies",
    conformance=_CONFORMANCE, refuse=_route_refusal("dW"))


def _linear(F: CompositeDefinition, role: str, M: int, K: int, N: int) -> Any:
    """F at (M, K, N) on the route sm_120 takes there (`cublas_bf16.train_route`), which refuses any shape the capture does not
    cover."""
    return bind(F, M=M, K=K, N=N, ROUTE=CB.train_route(role, M, K, N))


def weights(L: int) -> tuple[tuple[str, int | None, int], ...]:
    """The weights in HF's ``named_parameters`` order, each (node name, rows or None for a vector, columns)."""
    c, d = CONFIG, _dims()
    out = [("embed", c["V"], c["H"])]
    for i in range(L):
        out += [(f"l{i}.{k}", None if rows is None else d[rows], d[cols]) for k, (rows, cols) in LAYER.items()]
    return (*out, ("norm", None, c["H"]))


def _typed(rows: int | None, cols: int, elem: Any) -> Array:
    return Array(cols, elem) if rows is None else Array(rows, Array(cols, elem))


def params(L: int, B: int, S: int) -> tuple[tuple[str, Any], ...]:
    """`TrainStep_v1{L, B, S}`'s parameters, in order."""
    c, M, ws = CONFIG, B * S, weights(L)
    return (("ids", Array(M, U32)), ("labels", Array(M, U32)), ("n", Array(S, U16)), ("cs", _mat(S, c["D"])),
            *((f"{w}.w", _typed(r, k, BF16)) for w, r, k in ws),
            *((f"{w}.{s}", _typed(r, k, F32)) for w, r, k in ws for s in STATE),
            ("step", F32), *((h, F64) for h in HYPER))


def _ret(L: int) -> Tuple:
    return Tuple(F32, BF16, F32, BL.BIT, *(t for _, r, k in weights(L)
                                          for t in (_typed(r, k, BF16), *(_typed(r, k, F32),) * len(STATE))))


def _shape(S: Any) -> str | None:
    for k in ("L", "B", "S"):
        v = getattr(S, k, None)
        if v is not None and (not isinstance(v, int) or isinstance(v, bool) or v < 1):
            return f"TrainStep: {k} = {v!r} is not a positive int"
    if getattr(S, "B", None) is not None and getattr(S, "S", None) is not None:
        if S.B * S.S not in CB.TRAIN_M:
            return f"TrainStep: M = B·S = {S.B * S.S} is not a captured M ({CB.TRAIN_M})"
        if (S.S, CONFIG["D"]) not in CB.TRAIN_BMM_SHAPES:
            return f"TrainStep: S = {S.S} is not a captured attention shape ({CB.TRAIN_BMM_SHAPES})"
    return None


def _refuse_step(b: dict) -> str | None:
    return _shape(type("S", (), b))


def _signature(S: Any):
    why = _shape(S)
    if why:
        raise ValueError(why)
    return params(S.L, S.B, S.S), _ret(S.L)


def _field(c: Coll, j: int) -> Coll:
    """Field j of each member of c, an Array of Tuples."""
    t = c.type.elem
    ft = t.fields[j]
    return Coll(Array(c.type.n, ft), strided_view(c, t.field_offset(j), ft.leaves, t.leaves, 1, c.type.n * ft.leaves))


def _flat(c: Coll, elem: Any = BF16) -> Coll:
    return c.reshape(Array(c.type.leaves // elem.leaves, elem))


def _transpose(B, x: Coll, R: int, C: int) -> Coll:
    """A token-major R x C matrix of bf16 words as its C x R transpose, by `Transpose_v1` tiles of at most TILE x TILE, one
    batch a tile shape (`torch_attention.transposes`' tiling), each tile cut from x's rows whatever x's references are."""
    L, T = BF16.leaves, TA.TILE
    rows = [(r, min(T, R - r)) for r in range(0, R, T)]
    cols = [(c, min(T, C - c)) for c in range(0, C, T)]
    out = {}
    for rn, cn in sorted({(rn, cn) for _, rn in rows for _, cn in cols}):
        at = [(r, c) for r, n in rows for c, k in cols if (n, k) == (rn, cn)]
        tiles = Coll(Array(len(at), _mat(rn, cn)), concat(
            [x.refs.slice(L * ((r + i) * C + c), L * ((r + i) * C + c + cn)) for r, c in at for i in range(rn)]))
        y = B.batch(bind(TT.Transpose, R=rn, C=cn), tiles, axes=(0,))
        out.update({k: y[i] for i, k in enumerate(at)})
    return Coll(_mat(C, R), concat([out[(r, j - j % T)][j % T].refs for j in range(C) for r, _ in rows]))


def _constants(B, S: Any) -> Coll:
    bit = (B.call(BL.constant(0)).refs[0], B.call(BL.constant(1)).refs[0])
    return Coll(Tuple(U32, F32), Explicit(tuple(bit[(w >> i) & 1] for w in (IGNORE, ONE) for i in range(32))))


TrainStepConstants = CompositeDefinition(
    "TrainStepConstants", 1, (), lambda S: ((), Tuple(U32, F32)), _constants,
    doc="the training step's two constant words: IGNORE, HF's ignore index -100 as the low 32 bits of its int64 label, and "
        "ONE, the loss's gradient seed 1.0",
    conformance="exact-model-tested; its words are IGNORE and ONE")


def _shift(labels: Coll, nb: int, s: int, ignore: Coll) -> Coll:
    """HF's ``pad(labels, (0, 1), value=-100)[..., 1:]`` on each of nb sequences of s labels, ``ignore`` being the word -100."""
    L, ig = U32.leaves, ignore.refs
    return Coll(labels.type, concat([p for b in range(nb) for p in (labels.refs.slice(L * (b * s + 1), L * (b + 1) * s), ig)]))


class _Step:
    """One build of TrainStep_v1's body: the builder, the statics and the parameters by name."""

    def __init__(self, B, S: Any, args: tuple) -> None:
        self.B, self.L, self.nb, self.s = B, S.L, S.B, S.S
        self.M = S.B * S.S
        self.p = {name: a for (name, _), a in zip(params(S.L, S.B, S.S), args, strict=True)}
        c = CONFIG
        self.H, self.I, self.V, self.D = c["H"], c["I"], c["V"], c["D"]
        self.att = {"N": S.S, "NH": c["NH"], "KVH": c["KVH"], "D": c["D"]}
        self.lanes = TT.reduce_lanes(self.H, self.M)

    def norm(self, name: str, x: Coll, w: Coll) -> tuple[Coll, Coll, Coll]:
        f = bind(TT.RmsNormTorch, H=self.H, LANES=self.lanes, EPS=EPS, RSQRT=bind(BMU.RsqrtApprox))
        o = self.B.batch(f, x, w, axes=(0, None), name=name)
        return _field(o, 0), _field(o, 1), _field(o, 2)

    def norm_backward(self, name: str, dy: Coll, x: Coll, w: Coll, r: Coll, hb: Coll) -> tuple[Coll, Coll]:
        B, M, H = self.B, self.M, self.H
        dx = B.batch(bind(TT.RmsNormTorchBackward, H=H, LANES=self.lanes), dy, x, w, r, axes=(0, 0, None, 0), name=f"{name}.dx")
        dw = B.batch(bind(TT.RmsNormTorchWeightGrad, M=M, H=H), _transpose(B, dy, M, H), _transpose(B, hb, M, H),
                     axes=(0, 0), name=f"{name}.dw")
        return dx, dw

    def linear(self, name: str, x: Coll, w: Coll) -> Coll:
        (N, K), M = (w.type.n, w.type.elem.n), self.M
        return self.B.call(_linear(LinearTorch, "fwd", M, K, N), x, w, name=name)

    def linear_backward(self, name: str, dy: Coll, w: Coll, xt: Coll) -> tuple[Coll, Coll]:
        """dX and dW of the linear ``name`` from its output gradient dy, its weight and its input's transpose."""
        B, M = self.B, self.M
        N, K = w.type.n, w.type.elem.n
        dx = B.call(_linear(LinearTorchDx, "dX", M, K, N), dy, _transpose(B, w, N, K), name=f"{name}.dx")
        dw = B.call(_linear(LinearTorchDw, "dW", M, K, N), _transpose(B, dy, M, N), xt, name=f"{name}.dw")
        return dx, dw

    def add(self, name: str, a: Coll, b: Coll) -> Coll:
        return self.B.batch(EW.Bf16AddF2fp, _flat(a), _flat(b), axes=(0, 0), name=name).reshape(a.type)

    def mul(self, name: str, a: Coll, b: Coll) -> Coll:
        return self.B.batch(EW.Bf16MulBf16, _flat(a), _flat(b), axes=(0, 0), name=name).reshape(a.type)

    def seqs(self, x: Coll) -> Coll:
        return x.reshape(Array(self.nb, Array(self.s, x.type.elem)))

    def tokens(self, x: Coll) -> Coll:
        return x.reshape(Array(self.M, x.type.elem.elem))

    def act_rows(self, x: Coll) -> Coll:
        return x.reshape(Array(x.type.leaves // (ACT_N * BF16.leaves), Array(ACT_N, BF16)))

    def build(self) -> Coll:
        B, p, M, H, V = self.B, self.p, self.M, self.H, self.V
        x = B.call(bind(TE.EmbeddingTorch, M=M, V=V, H=H), p["ids"], p["embed.w"], name="embed")
        fw = []
        for i in range(self.L):
            w = {k: p[f"l{i}.{k}.w"] for k in LAYER}
            n = f"l{i}."
            y1, hb1, r1 = self.norm(n + "ln1", x, w["ln1"])
            q, k, v = (self.linear(n + j, y1, w[j]) for j in ("q", "k", "v"))
            att = B.batch(bind(TA.EagerAttentionTorch, **self.att), self.seqs(q), self.seqs(k), self.seqs(v), p["cs"], p["n"],
                          axes=(0, 0, 0, None, None), name=n + "attn")
            a = [_field(att, j) for j in range(5)]
            o_in = self.tokens(a[0])
            o = self.linear(n + "o", o_in, w["o"])
            h2 = self.add(n + "h2", x, o)
            y2, hb2, r2 = self.norm(n + "ln2", h2, w["ln2"])
            gate, up = self.linear(n + "gate", y2, w["gate"]), self.linear(n + "up", y2, w["up"])
            f = bind(TT.SiluTorch, N=ACT_N, EX2=bind(BMU.MufuEx2Ftz))
            act = B.batch(f, self.act_rows(gate), axes=(0,), name=n + "act").reshape(gate.type)
            mul = self.mul(n + "mul", act, up)
            down = self.linear(n + "down", mul, w["down"])
            fw.append({"w": w, "x": x, "y1": y1, "hb1": hb1, "r1": r1, "v": v, "a": a, "o_in": o_in, "h2": h2, "y2": y2,
                       "hb2": hb2, "r2": r2, "gate": gate, "up": up, "act": act, "mul": mul})
            x = self.add(n + "h3", h2, down)
        yn, hbn, rn = self.norm("norm", x, p["norm.w"])
        logits = self.linear("logits", yn, p["embed.w"])

        k = B.call(TrainStepConstants, name="constants")
        t = _shift(p["labels"], self.nb, self.s, k[0])
        rows = B.batch(bind(TL.LogSoftmaxRow, V=V), logits, t, axes=(0, 0), name="loss.rows").reshape(Array(M, Array(3, F32)))
        loss = B.call(bind(TL.NllLossTorch, M=M, V=V, MEAN=1), rows.take(1, 2), t, name="loss")
        g = B.call(bind(TL.NllLossTorchBackward, MEAN=1), k[1], loss[1], name="loss.dx")
        dlogits = B.batch(bind(TL.LogSoftmaxNllBackwardRow, V=V), logits, t, rows.take(1, 0), rows.take(1, 1), g,
                          axes=(0, 0, 0, 0, None), name="logits.dx")

        dlm_x, dlm_w = self.linear_backward("lm_head", dlogits, p["embed.w"], _transpose(B, yn, M, H))
        dh, dnw = self.norm_backward("norm", dlm_x, x, p["norm.w"], rn, hbn)
        grads: dict[str, Coll] = {"norm.w": dnw}
        for i in reversed(range(self.L)):
            f, w, n = fw[i], fw[i]["w"], f"l{i}."
            d = {}
            ddown, d["down"] = self.linear_backward(n + "down", dh, w["down"], _transpose(B, f["mul"], M, self.I))
            dact, dup = self.mul(n + "mul.da", ddown, f["up"]), self.mul(n + "mul.db", ddown, f["act"])
            y2t = _transpose(B, f["y2"], M, H)
            dup_x, d["up"] = self.linear_backward(n + "up", dup, w["up"], y2t)
            fb = bind(TT.SiluTorchBackward, N=ACT_N, EX2=bind(BMU.MufuEx2Ftz), CONTRACT=1)
            dgate = B.batch(fb, self.act_rows(dact), self.act_rows(f["gate"]), axes=(0, 0), name=n + "act.dx").reshape(
                dact.type)
            dgate_x, d["gate"] = self.linear_backward(n + "gate", dgate, w["gate"], y2t)
            dy2 = self.add(n + "mlp.dx", dup_x, dgate_x)
            dln2, d["ln2"] = self.norm_backward(n + "ln2", dy2, f["h2"], w["ln2"], f["r2"], f["hb2"])
            dh2 = self.add(n + "h2.grad", dh, dln2)
            do, d["o"] = self.linear_backward(n + "o", dh2, w["o"], _transpose(B, f["o_in"], M, w["o"].type.elem.n))
            a = f["a"]
            ab = B.batch(bind(TA.EagerAttentionTorchBackward, **self.att), self.seqs(do), a[1], a[2], self.seqs(f["v"]), a[3],
                         a[4], p["cs"], axes=(0, 0, 0, 0, 0, 0, None), name=n + "attn.dx")
            dq, dk, dv = (self.tokens(_field(ab, j)) for j in range(3))
            y1t = _transpose(B, f["y1"], M, H)
            dv_x, d["v"] = self.linear_backward(n + "v", dv, w["v"], y1t)
            dk_x, d["k"] = self.linear_backward(n + "k", dk, w["k"], y1t)
            dq_x, d["q"] = self.linear_backward(n + "q", dq, w["q"], y1t)
            dy1 = self.add(n + "qkv.dx", self.add(n + "kv.dx", dv_x, dk_x), dq_x)
            dln1, d["ln1"] = self.norm_backward(n + "ln1", dy1, f["x"], w["ln1"], f["r1"], f["hb1"])
            dh = self.add(n + "x.grad", dh2, dln1)
            grads |= {f"l{i}.{k}.w": d[k] for k in LAYER}
        grads["embed.w"] = B.call(bind(TE.EmbeddingBackwardTiedV2, M=M, V=V, H=H), p["ids"], dh, dlm_w, name="embed.dw")

        names = [w for w, _, _ in weights(self.L)]
        g = [_flat(grads[f"{w}.w"]) for w in names]
        clip = B.call(bind(TC.ClipGradNorm, SIZES=tuple(x.type.n for x in g)), tuple_of(*g), p["max_norm"], name="clip")
        sc = B.call(TO.AdamWScalars, *(p[h] for h in HYPER[:5]), p["step"], name="adamw.scalars")
        new = [c for j, w in enumerate(names) for c in self.adamw(w, clip[1][j], sc[0])]
        return tuple_of(loss[0], clip[0], sc[1], sc[2], *new)

    def adamw(self, w: str, g: Coll, k: Coll) -> list[Coll]:
        """The weight w's optimizer step from its clipped gradient g and the step's scalars k: the gradient widened to f32
        (``master.grad = p.grad.float()``), AdamW on the master copy, and the master's bf16 copy (``p.copy_(master)``) ->
        (the bf16 weight, master, exp_avg, exp_avg_sq)."""
        B, p = self.B, self.p
        t, n = p[f"{w}.master"].type, g.type.n
        if n % ADAM_N:
            raise ValueError(f"TrainStep: {w} has {n} elements, not a multiple of ADAM_N = {ADAM_N}")
        runs = Array(n // ADAM_N, Array(ADAM_N, F32))
        g32 = B.batch(CA.Bf16ToF32, g, axes=(0,), name=f"{w}.grad")
        o = B.batch(bind(TO.AdamWForeach, N=ADAM_N, LERP_FMA=1, BF16=0), p[f"{w}.master"].reshape(runs), g32.reshape(runs),
                    p[f"{w}.exp_avg"].reshape(runs), p[f"{w}.exp_avg_sq"].reshape(runs), k, axes=(0, 0, 0, 0, None),
                    name=f"{w}.adamw")
        master, m, v = (_field(o, j).reshape(t) for j in range(len(STATE)))
        copy = B.batch(BG.F2fpBf16, _flat(master, F32), axes=(0,), name=f"{w}.copy").reshape(p[f"{w}.w"].type)
        return [copy, master, m, v]


TrainStep = CompositeDefinition(
    "TrainStep", 1, ("L", "B", "S"), _signature, lambda B, S, *a: _Step(B, S, a).build(),
    doc="a decoder's training step on B sequences of S tokens through L layers, torch 2.14 eager with bf16 parameters: the "
        "embedding, each layer's norms, attention and MLP, the final norm, the tied head and the cross-entropy, autograd's "
        "backward to every parameter's gradient, the gradient clip, and AdamW on the f32 master copy with its bf16 copy-back",
    conformance="exact-model-tested in its blocks: each node's Definition is held to the GPU's words in its own module's tests; "
                "the root's call-by-call walk against a step's GPU dump is program_walk's",
    refuse=_refuse_step, ports=_POSITION)
