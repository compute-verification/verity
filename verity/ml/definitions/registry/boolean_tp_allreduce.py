"""`registry.tp_allreduce`'s all-reduces as Boolean Definitions: the next version of each word Definition, the same function on
bits (bf16 = Array<16, Value<1>>).

    AllReduceMnnvlCoordinate_v2{WORLD,SHOT}    (col: Array<WORLD, BF16>) -> BF16          word view: AllReduceMnnvlCoordinate_v1
    AllReduceMnnvl_v2{WORLD,N,SHOT}            (parts: Array<WORLD, Array<N, BF16>>) -> Array<N, BF16>
                                                                                          word view: AllReduceMnnvl_v1
    AllReduceCustomCoordinate_v2{WORLD,START}  (col: Array<WORLD, BF16>) -> BF16          word view: AllReduceCustomCoordinate_v1
    AllReduceCustom_v2{WORLD,SIZE,STAGE}       (parts: Array<WORLD, Array<SIZE, BF16>>) -> Array<SIZE, BF16>
                                                                                          word view: AllReduceCustom_v1
    AllReduceMultimemCoordinate_v2{WORLD,TREE,FTZ}
                                               (col: Array<WORLD, BF16>) -> BF16          word view: AllReduceMultimemCoordinate_v1
    AllReduceMultimemVector_v2{WORLD,TREE,FTZ} (parts: Array<WORLD, Array<8, BF16>>) -> Array<8, BF16>
                                                                                          word view: AllReduceMultimemVector_v1
    AllReduceMultimem_v2{WORLD,SIZE,TREES,FTZ} (parts: Array<WORLD, Array<SIZE, BF16>>) -> Array<SIZE, BF16>
                                                                                          word view: AllReduceMultimem_v1
    AllReduceNcclTreeCoordinate_v2{WORLD,CHAIN}
                                               (col: Array<WORLD, BF16>) -> BF16          word view: AllReduceNcclTreeCoordinate_v1
    AllReduceNcclTree_v2{WORLD,SIZE,CHAIN}     (parts: Array<WORLD, Array<SIZE, BF16>>) -> Array<SIZE, BF16>
                                                                                          word view: AllReduceNcclTree_v1

Each coordinate is traced for its statics (`elementwise.specialized`) through `verity.ml.circuits.silicon.fp`, so a payload's 16
constant-zero low bits fold into its adds. The MNNVL sum is written as what each FADD.FTZ does to the words it can meet:

* a payload is read with -0 written as +0 and a subnormal flushed to its signed zero (its mantissa ANDed with "exponent
  nonzero");
* the first add, ``+0 + x0``, is x0 so flushed with its sign kept only when its exponent is nonzero (a flushed x0 that is zero
  becomes +0), which the sanitize already did for 0x8000;
* the running sum is an FTZ result, never subnormal, so later adds read it unflushed;
* two-shot's ``+0 + sum`` is the sum with its sign kept only when its exponent is nonzero.

The adds make the GPU's NaN (`F32AddFtz_v2`'s 0x7FFFFFFF) and F2FP ends every NaN in 0x7FFF; ``+0 + x`` keeping a NaN's payload
changes no output. The custom all-reduce's IEEE adds are `F32Add_v2`'s, as are the multimem tree's; its FTZ adds read a payload
with a subnormal flushed to its signed zero and an add's FTZ result unflushed. NCCL's tree adds are `Bf16AddF2fp_v2`'s builder
(`elementwise.bf16_add_f2fp`), each sum rounded to bf16 before the next add reads it. The rows are the word bodies on bits.
"""
from __future__ import annotations

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.boolean.forms import OR, or_reduce
from verity.ml.circuits.defs import bind
from verity.ml.circuits.ops import fp32 as W
from verity.ml.circuits.ops.boolean import elementwise as EW
from verity.ml.circuits.ops.boolean.softmax import ftz
from verity.ml.circuits.refs import array_of, coll_concat
from verity.ml.circuits.silicon import fp
from verity.ml.circuits.types import Array
from verity.ml.definitions.registry import tp_allreduce as TA

__all__ = ["AllReduceCustom", "AllReduceCustomCoordinate", "AllReduceMnnvl", "AllReduceMnnvlCoordinate", "AllReduceMultimem",
           "AllReduceMultimemCoordinate", "AllReduceMultimemVector", "AllReduceNcclTree", "AllReduceNcclTreeCoordinate",
           "custom_sum", "mnnvl_sum", "multimem_sum", "nccl_tree_sum"]

BF16 = Array(16, BL.BIT)


def _exp_nz(C, x):
    return or_reduce(C, x[7:15])


def _first(C, x):
    """``+0 + x`` on a bf16 payload widened, under FTZ: mantissa flushed when the exponent is 0, the sign kept only when it is
    not."""
    e = _exp_nz(C, x)
    return fp.bf16_to_f32([C.AND(e, m) for m in x[:7]] + list(x[7:15]) + [C.AND(x[15], e)])


def _payload(C, x):
    """A payload as FADD.FTZ reads it after `sanitizeLamportPayload`: 0x8000 written as 0x0000, then a subnormal flushed to its
    signed zero, widened."""
    e = _exp_nz(C, x)
    sign = C.AND(x[15], OR(C, e, or_reduce(C, x[:7])))
    return fp.bf16_to_f32([C.AND(e, m) for m in x[:7]] + list(x[7:15]) + [sign])


def mnnvl_sum(S):
    """`AllReduceMnnvlCoordinate_v1{WORLD,SHOT}`'s builder."""
    shot = TA.strategy(S.SHOT)

    def build(C, *col):
        acc = _first(C, col[0])
        for x in col[1:]:
            acc = ftz(C, fp.f32_add(C, acc, _payload(C, x), nan_word=W.F32_NAN_PTX))
        if shot == 2:
            acc = list(acc[:31]) + [C.AND(acc[31], or_reduce(C, acc[23:31]))]
        return fp.f32_to_bf16(C, acc, W.BF16_NAN_F2FP)
    return build


def custom_sum(S):
    """`AllReduceCustomCoordinate_v1{WORLD,START}`'s builder."""
    order = TA.custom_order(TA.ranks(S.WORLD), int(S.START))

    def build(C, *col):
        acc = fp.bf16_to_f32(col[order[0]])
        for r in order[1:]:
            acc = fp.f32_add(C, acc, fp.bf16_to_f32(col[r]), nan_word=W.F32_NAN_PTX)
        return fp.f32_to_bf16(C, acc, W.BF16_NAN_F2FP)
    return build


def _flushed(C, x):
    """A bf16 payload widened as FADD.FTZ reads it: a subnormal flushed to its signed zero."""
    e = _exp_nz(C, x)
    return fp.bf16_to_f32([C.AND(e, m) for m in x[:7]] + list(x[7:16]))


def multimem_sum(S):
    """`AllReduceMultimemCoordinate_v1{WORLD,TREE,FTZ}`'s builder."""
    tree, flush = TA.multimem_tree(S.TREE, TA.ranks(S.WORLD)), bool(S.FTZ)

    def build(C, *col):
        def value(t):
            if isinstance(t, int):
                return _flushed(C, col[t]) if flush else fp.bf16_to_f32(col[t])
            s = fp.f32_add(C, value(t[0]), value(t[1]), nan_word=W.F32_NAN_PTX)
            return ftz(C, s) if flush else s

        return fp.f32_to_bf16(C, value(tree), W.BF16_NAN_F2FP)
    return build


def _col(S):
    return ((("col", Array(int(S.WORLD), BF16)),), BF16)


AllReduceMnnvlCoordinate = EW.specialized(
    "AllReduceMnnvlCoordinate", 2, ["WORLD", "SHOT"], mnnvl_sum, _col,
    doc="FlashInfer's MNNVL all-reduce of one element on bits: AllReduceMnnvlCoordinate_v1, the FTZ f32 sum from +0 of the "
        "sanitized payloads in rank order (two-shot once more onto +0), F2FP",
    word=lambda b: bind(TA.AllReduceMnnvlCoordinate, WORLD=b["WORLD"], SHOT=b["SHOT"]))

AllReduceCustomCoordinate = EW.specialized(
    "AllReduceCustomCoordinate", 2, ["WORLD", "START"], custom_sum, _col,
    doc="vLLM's custom all-reduce of one element on bits: AllReduceCustomCoordinate_v1, the IEEE f32 sum in rank order from "
        "START, F2FP",
    word=lambda b: bind(TA.AllReduceCustomCoordinate, WORLD=b["WORLD"], START=b["START"]))


def _mnnvl(B, S, parts):
    """`AllReduceMnnvl_v1`'s body on bits: one `AllReduceMnnvlCoordinate_v2` per element."""
    return B.batch(bind(AllReduceMnnvlCoordinate, WORLD=S.WORLD, SHOT=S.SHOT), parts, axes=(1,))


AllReduceMnnvl = EW.on_bits(
    "AllReduceMnnvl", 2, ("WORLD", "N", "SHOT"),
    lambda S: ((("parts", Array(int(S.WORLD), Array(S.N, BF16))),), Array(S.N, BF16)), _mnnvl, TA.AllReduceMnnvl)


def _custom(B, S, parts):
    """`AllReduceCustom_v1`'s body on bits: each run of `tp_allreduce.custom_owners` one `AllReduceCustomCoordinate_v2{START=owner}`
    per element."""
    world = TA.ranks(S.WORLD)
    outs = []
    for lo, hi, owner in TA.custom_owners(world, int(S.SIZE), int(S.STAGE)):
        cols = array_of([parts[r][lo:hi] for r in range(world)])
        outs.append(B.batch(bind(AllReduceCustomCoordinate, WORLD=world, START=owner), cols, axes=(1,)))
    return outs[0] if len(outs) == 1 else coll_concat(Array(int(S.SIZE), BF16), outs)


AllReduceCustom = EW.on_bits(
    "AllReduceCustom", 2, ("WORLD", "SIZE", "STAGE"),
    lambda S: ((("parts", Array(int(S.WORLD), Array(S.SIZE, BF16))),), Array(S.SIZE, BF16)), _custom, TA.AllReduceCustom)

AllReduceMultimemCoordinate = EW.specialized(
    "AllReduceMultimemCoordinate", 2, ["WORLD", "TREE", "FTZ"], multimem_sum, _col,
    doc="torch's multimem all-reduce of one element on bits: AllReduceMultimemCoordinate_v1, the f32 adds of TREE (FTZ or IEEE), "
        "F2FP",
    word=lambda b: bind(TA.AllReduceMultimemCoordinate, WORLD=b["WORLD"], TREE=b["TREE"], FTZ=b["FTZ"]))


def _multimem_vector(B, S, parts):
    """`AllReduceMultimemVector_v1`'s body on bits: one `AllReduceMultimemCoordinate_v2` per element."""
    return B.batch(bind(AllReduceMultimemCoordinate, WORLD=S.WORLD, TREE=S.TREE, FTZ=bool(S.FTZ)), parts, axes=(1,))


AllReduceMultimemVector = EW.on_bits(
    "AllReduceMultimemVector", 2, ("WORLD", "TREE", "FTZ"),
    lambda S: ((("parts", Array(int(S.WORLD), Array(TA.MULTIMEM_VECTOR, BF16))),), Array(TA.MULTIMEM_VECTOR, BF16)),
    _multimem_vector, TA.AllReduceMultimemVector)


def _multimem(B, S, parts):
    """`AllReduceMultimem_v1`'s body on bits: each run of `tp_allreduce.multimem_runs` one `AllReduceMultimemVector_v2` per 16-byte
    vector."""
    world, V = TA.ranks(S.WORLD), TA.MULTIMEM_VECTOR
    outs = []
    for lo, hi, tree in TA.multimem_runs(world, int(S.SIZE), S.TREES):
        vectors = Array((hi - lo) // V, Array(V, BF16))
        cols = array_of([parts[r][lo:hi].reshape(vectors) for r in range(world)])
        outs.append(B.batch(bind(AllReduceMultimemVector, WORLD=world, TREE=tree, FTZ=bool(S.FTZ)), cols, axes=(1,)))
    return coll_concat(Array(int(S.SIZE), BF16), outs)


AllReduceMultimem = EW.on_bits(
    "AllReduceMultimem", 2, ("WORLD", "SIZE", "TREES", "FTZ"),
    lambda S: ((("parts", Array(int(S.WORLD), Array(S.SIZE, BF16))),), Array(S.SIZE, BF16)), _multimem, TA.AllReduceMultimem)


def nccl_tree_sum(S):
    """`AllReduceNcclTreeCoordinate_v1{WORLD,CHAIN}`'s builder."""
    chain = TA.nccl_chain(S.CHAIN, TA.ranks(S.WORLD))

    def build(C, *col):
        acc = list(col[chain[-1]])
        for r in reversed(chain[:-1]):
            acc = EW.bf16_add_f2fp(C, col[r], acc)
        return acc
    return build


AllReduceNcclTreeCoordinate = EW.specialized(
    "AllReduceNcclTreeCoordinate", 2, ["WORLD", "CHAIN"], nccl_tree_sum, _col,
    doc="NCCL's tree all-reduce of one element on bits: AllReduceNcclTreeCoordinate_v1, the leaf's word and then each rank up "
        "CHAIN added to it, one bf16 rounding per add (NaN 0x7FFF)",
    word=lambda b: bind(TA.AllReduceNcclTreeCoordinate, WORLD=b["WORLD"], CHAIN=b["CHAIN"]))


def _nccl_tree(B, S, parts):
    """`AllReduceNcclTree_v1`'s body on bits: one `AllReduceNcclTreeCoordinate_v2` per element."""
    world = TA.ranks(S.WORLD)
    return B.batch(bind(AllReduceNcclTreeCoordinate, WORLD=world, CHAIN=TA.nccl_chain(S.CHAIN, world)), parts, axes=(1,))


AllReduceNcclTree = EW.on_bits(
    "AllReduceNcclTree", 2, ("WORLD", "SIZE", "CHAIN"),
    lambda S: ((("parts", Array(int(S.WORLD), Array(S.SIZE, BF16))),), Array(S.SIZE, BF16)), _nccl_tree, TA.AllReduceNcclTree)
