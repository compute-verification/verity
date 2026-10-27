"""The toy's served core `Program`: one slot's decode step on circuits' FAMILIES map (`plan.Map`), every unit a Boolean
Definition on main, one batch node a family.

Its root parameters are the weight tensors by their names in `verity_torch_loop.weights.tensors(shape)` (registered, and
the model's weights), each typed as the tensor whole (an s8 linear ``Array<N, Array<K, s8 bits>>``, a scale ``Array<N, f32
bits>``, a gain ``Array<H, bf16 bits>``, ``tok`` ``Array<H, Array<V, bf16 bits>>``), then the run inputs:

* ``token``: the slot's input token, a u16 word;
* ``n``: the slot's key count, position + 1, a u16 word of kind position (`QkSoftmaxRowPad_v1`'s ``n`` and `PvRowPad_v2`'s);
* ``p``: the slot's position, n - 1, an i32 word of kind position: the index each coordinate's ``MoeSelectBf16x{context}_v2``
  takes into its ``pos`` rows (a root constant is data, so the Program can't compute it from ``n``);
* ``keys``, ``values``: each layer's 32 K and V rows of the slot (head_dim bf16 words each), the rows committed at its
  positions 0 .. n - 1, this step's own at n - 1, the rest absent.

It returns the next token (`Argmax_v2`'s u16 index) and each layer's new K and V rows.

The map's families, in the step's order, with the Boolean Definition each calls (the map's keys name some by their word
view; the Program calls the Boolean Definition whose `word` that is):

| map key | Program's Definition |
|---|---|
| ``EmbeddingCoordinate_v1{V}`` | the same, its ``pos`` ``MoeSelectBf16x{context}_v2`` (``p``) over the coordinate's ``pos`` rows |
| ``RmsSumSqChunks_v1{C, K}``, ``RmsNormOutSums_v1{C, S, G, N, EPS}`` | the same |
| ``RowAbsMax_v1{N, RELU}`` | ``RowAbsMax_v2`` |
| ``QuantChunkS8_v1{N, RELU}`` | ``QuantChunkS8_v2`` (chunk 0's scale is the linear's ``sx``) |
| ``ScaledMmS8CoordinateRows_v1{K, R, BlackwellS8ImmaDot32_v1}`` | ``ScaledMmS8CoordinateRows_v2{K, R, BlackwellS8ImmaDot32_v2}`` |
| ``QkSoftmaxRowPad_v1{T, D}`` | ``QkSoftmaxRowPad_v1{T, D, HopperBF16WgmmaDot16_v2}`` |
| ``PvRow_v1{T, K, D, HopperBF16WgmmaDot16_v1}`` | ``PvRowPad_v2{T, K, D, HopperBF16WgmmaDot16_v2}``: keys at n and past read as 0 |
| ``Bf16AddRow_v2{N}``, ``Argmax_v2{V}`` | the same |
| ``AllGather_v1{WORLD, N}`` | none: a gather is the references its parts' outputs are (`GATHER`) |

W's row parts: the coordinate's ``wrow{r}`` is codes ``r K / R .. (r + 1) K / R`` of W's row, a strided view of the
parameter, so a registered row of W is ``K / R`` codes.
"""
from __future__ import annotations

from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.ops.boolean import gemm as BG
from verity.ml.circuits.ops.boolean import moe as BM
from verity.ml.circuits.ops.boolean import quant as BQ
from verity.ml.circuits.ops.boolean import rmsnorm as RN
from verity.ml.circuits.ops.boolean import rows as BR
from verity.ml.circuits.ops.boolean import rowwise as RW
from verity.ml.circuits.ops.boolean import softmax_row as SR
from verity.ml.circuits.ports import In, Ports
from verity.ml.circuits.program import Program
from verity.ml.circuits.refs import Coll, array_of, concat, strided_view, tuple_of
from verity.ml.circuits.types import Array, Tuple, Value

from verity_torch_loop import weights as SW

from .plan import EPS_KEY
from .shape import PV_K, Shape, check, forms

BIT = Value(1)
BF16 = U16 = Array(16, BIT)
F32 = I32 = Array(32, BIT)
CODE = BQ.CODE
S8_DOT = BG.GEMM_SM120_S8["DOT"]
BF16_DOT = BG.GEMM_HOPPER["DOT"]
NAME = "E2eH200Step"
#: gathers as references (no unit): no Boolean gather Definition is on main
GATHER = "references"


def _weight_type(t: SW.Tensor) -> Array:
    elem = {"s8": CODE, "f32": F32, "bf16": BF16}[t.dtype]
    return Array(t.cols, elem) if t.rows == 1 else Array(t.rows, Array(t.cols, elem))


def program(shape: Shape) -> Program:
    """The served step's Program at ``shape`` (module docstring)."""
    sh, f = check(shape), forms(shape)
    w, d, h, c, L = sh.width, sh.head_dim, sh.heads, f.chunk, sh.layers
    ts = SW.tensors(sh)
    kv = Array(L, Array(PV_K, Array(d, BF16)))
    params = tuple((t.name, _weight_type(t)) for t in ts) + (
        ("token", U16), ("n", U16), ("p", I32), ("keys", kv), ("values", kv))
    ret_t = Tuple(U16, Array(L, Array(d, BF16)), Array(L, Array(d, BF16)))

    def body(B, S, *args):
        a = dict(zip((p for p, _ in params), args))
        tok, n = a["token"], a["n"]

        def norm(op: str, x: Coll, gain: Coll) -> Coll:
            N, sc = x.type.n, f.sums_c(x.type.n)
            rows = N // c
            sums = B.batch(bind(RN.RmsSumSqChunks, C=sc, K=c), x.reshape(Array(N // sc, Array(sc, BF16))), axes=(0,),
                           name=op + "_sums")
            out = B.batch(bind(RN.RmsNormOutSums, C=c, S=rows, G=f.groups(N), N=N, EPS=EPS_KEY),
                          x.reshape(Array(rows, Array(c, BF16))), gain.reshape(Array(rows, Array(c, BF16))),
                          sums.reshape(Array(rows, F32)), axes=(0, 0, None), name=op + "_norm")
            return out.reshape(Array(N, BF16))

        def quant(op: str, x: Coll, relu: bool) -> Coll:
            N = x.type.n
            parts, pw = f.parts(N)
            if parts > 1:
                xp = x.reshape(Array(parts, Array(pw, BF16)))
                partial = B.batch(bind(BQ.RowAbsMax, N=pw, RELU=relu), xp[0:parts - 1], axes=(0,), name=op + "_amax.part")
                last = Coll(Array(pw + parts - 1, BF16), concat([xp[parts - 1].refs, partial.refs]))
                amax = B.call(bind(BQ.RowAbsMax, N=pw + parts - 1, RELU=relu), last, name=op + "_amax")
            else:
                amax = B.call(bind(BQ.RowAbsMax, N=N, RELU=relu), x, name=op + "_amax")
            q = f.qchunk(N)
            return B.batch(bind(BQ.QuantChunkS8, N=q, RELU=relu), x.reshape(Array(N // q, Array(q, BF16))), amax,
                           axes=(0, None), name=op + "_quant")

        def linear(op: str, qx: Coll) -> Coll:
            W, sw = a[op], a[op + ".scale"]
            N, K = W.type.n, W.type.elem.n
            q = f.qchunk(K)
            R = K // q
            xs = [qx[r][0] for r in range(R)]
            wrows = [Coll(Array(N, Array(q, CODE)), strided_view(W, r * q * 8, q * 8, K * 8, 1, N * q * 8)) for r in range(R)]
            return B.batch(bind(BG.ScaledMmS8CoordinateRows, K=K, R=R, DOT=S8_DOT), *xs, qx[0][1], *wrows, sw,
                           axes=(None,) * (R + 1) + (0,) * (R + 1), name=op)

        def residual(op: str, x: Coll, y: Coll) -> Coll:
            rows = Array(w // c, Array(c, BF16))
            return B.batch(bind(BR.Bf16AddRow, N=c), x.reshape(rows), y.reshape(rows), axes=(0, 0),
                           name=op).reshape(Array(w, BF16))

        pos = B.batch(BM.select_bf16(sh.context), a["p"], a["pos"], axes=(None, 0), name="pos")
        x = B.batch(bind(RW.EmbeddingCoordinate, V=sh.vocab), tok, a["tok"], pos, axes=(None, 0, 0), name="embed")
        ks, vs = [], []
        for i in range(L):
            p = f"l{i}."
            qa = quant(p + "attn", norm(p + "attn", x, a[p + "attn_norm"]), False)
            qh = linear(p + "wq", qa).reshape(Array(h, Array(d, BF16)))
            ks.append(linear(p + "wk", qa))
            vs.append(linear(p + "wv", qa))
            keys, values = a["keys"][i], a["values"][i]
            probs = B.batch(bind(SR.QkSoftmaxRowPad, T=PV_K, D=d, DOT=BF16_DOT), qh, keys, n, axes=(0, None, None),
                            name=p + "softmax")
            vrows = [values[t] for t in range(PV_K)] + [n]
            att = B.batch(bind(BG.PvRowPad, T=PV_K, K=PV_K, D=d, DOT=BF16_DOT), probs, *vrows, axes=(0,) + (None,) * len(vrows),
                          name=p + "pv")
            o = linear(p + "wo", quant(p + "o", att.reshape(Array(h * d, BF16)), False))
            x = residual(p + "attn_residual", x, o)
            up = linear(p + "wup", quant(p + "up", norm(p + "mlp", x, a[p + "mlp_norm"]), False))
            x = residual(p + "mlp_residual", x, linear(p + "wdown", quant(p + "down", up, True)))
        logits = linear("head", quant("head", norm("final", x, a["final_norm"]), False))
        nxt = B.call(bind(BR.Argmax, V=sh.vocab), logits, name="argmax")
        return tuple_of(nxt, array_of(ks), array_of(vs))

    fn = CompositeDefinition(NAME, 1, (), lambda S: (params, ret_t), body, register=False,
                             ports=Ports(params={"n": In(kind="position"), "p": In(kind="position")}),
                             doc=f"one slot's decode step of {sh.name} on circuits' FAMILIES map")
    names = tuple(t.name for t in ts)
    return Program(bind(fn), registered=names, theta=names)

