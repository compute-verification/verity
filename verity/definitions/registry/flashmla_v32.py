"""[GLM-5.3] The sparse MLA attention over an FP8 cache as the H200 runs it: GLM-5.3 at TP8 on sm_90, vLLM d9105ea80's
`FlashMLASparseImpl._forward_fp8_kv_mixed_batch` (vllm/v1/attention/backends/mla/flashmla_sparse.py:826) over the FlashMLA it
vendors (6bc49418): the `sparse_fp8` decode kernel at its V32 instance, `flash_fwd_splitkv_mla_fp8_sparse_kernel<
KernelTemplate<V32, 64>>` (csrc/sm90/decode/sparse_fp8/instantiations/v32_persistent_h64.cu), with the planner and the combine
`flashmla` describes.  Line numbers are at those commits.

The call.  With fewer than 32 heads a rank (`MIN_HEADS_FOR_BF16_PREFILL`, :63) and an `fp8_ds_mla` cache, every token of a
step that the layer hands the sparse path goes through one call (`forward_mqa`, :984-1031; :870-897, :899-937).  Those are the
step's decode tokens, and its prefill tokens too once its longest prefill is past `index_topk` rows; a step whose prefills are
all within them sends its prefill tokens to the dense prefill instead (vllm/model_executor/layers/attention/mla_attention.py:
890-898, :1113-1136; sparse_mla_attention.py:265-310; the masked prefill is refused below sm_100 and on a quantized cache,
:80-112).  A decode there is wider than a request generating one token: the step's requests come ordered, and its decodes are
the leading ones whose query is as long as the first's, that being at most the builder's threshold, 256 tokens at 8 heads on
sm_90 or one more than the draft tokens if that is larger (flashmla_sparse.py:265, :279-291; vllm/v1/attention/backend.py:
627-657; vllm/v1/attention/backends/utils.py:768-839).  q (1, T, 64, 576) bf16, the rank's 8 real heads padded to 64 with
zeros (`_compute_fp8_decode_padded_heads`, :587-590); the cache (blocks, 64, 1, 656) bytes; indices (1, T, 2048), each token's
`index_topk` rows as cache slots, -1 where it has none (`triton_convert_req_index_to_global_index`); head_dim_v 512;
softmax_scale qk_head_dim^-0.5 = 1/16; no sink, no lengths, no second cache.  The 56 pad heads' outputs are sliced away.

What this instance does that `flashmla`'s (MODEL1) does not:

1. The row (656 bytes; splitkv_mla.cuh:543-598, dequant.h:21-34): 512 E4M3 NoPE bytes, then four f32 scales, one per 128 NoPE
   elements, then 64 bf16 RoPE elements.  A scale is rounded to bf16 before it multiplies (`(bf16)scales_float[i]`), and a NoPE
   element is `__float22bfloat162_rn(float(e4m3)) * scale`, one bf16 multiply: `FlashMlaV32KvRow_v1`.
2. Validity: a row is valid iff its index is not -1; there is no length (`topk_length` is null at this instance), so
   `FlashMlaRowValid_v1`'s second term is gone and the row's Definition takes the index itself.  An invalid row's NoPE bytes
   and scales are zeroed, so its NoPE elements are +0; its RoPE elements are whatever row the index -1 addresses and reach no
   output, the score they enter being replaced by -inf and the value being the NoPE elements alone.
3. The key is the 576-element row [NoPE 512 | RoPE 64] and the value its first 512 elements: the score chain is 36 k16
   steps, O and the output 512 words a head.  `FlashMlaBlock_v1` has one width for both, so at D = 576 it carries 64 columns of
   O the kernel never forms; `FlashMlaV32Block_v1` is its arithmetic (`flashmla.block`) with the value DV wide.
4. No sink.  The decode kernel runs its sink term on -inf (`rAttn_sink = -inf` when the pointer is null, :199-206), which
   `FlashMlaV32SplitEnd_v1` does on a constant; the combine skips its own (combine.cu:101-112), and so does
   `FlashMlaV32CombineWeights_v1`.
5. The schedule.  The call is one request of s_q = T query rows, so num_sm_parts = max(132 / T, 1) and every token of the
   step has the same SPLITS over its 32 blocks, a function of T alone (`flashmla_v32_twin.step_splits`): 32 splits of one
   block at T <= 3 down to one split, written by the decode kernel with no combine, from T = 67.

Registered functions (all `Name_v1`):
  FlashMlaV32KvRow{DN,DR,QT}                      (nope: DN e4m3, scale: DN/QT f32, rope: DR bf16, idx: i32) -> (DN+DR bf16, valid);
                                                  a staged row
  FlashMlaV32Block{DK,DV,BN,FIRST,DOT,SCALE}      (q: DK bf16, k: BN rows, nm, l[4], o[DV]) -> (nm, l[4], o[DV]); a split's
                                                  first block (FIRST) takes q and k only
  FlashMlaV32SplitEnd{DV,PARTIAL}                 (nm, l[4], o[DV]) -> DV bf16, or (PARTIAL) (oacc: DV f32, lse)
  FlashMlaV32CombineWeights{NS}                   (lse: NS f32) -> NS f32
  FlashMlaV32CombineValues{NS,G}                  (w: NS f32, oacc: NS x G f32) -> G bf16
  FlashMlaV32DecodeHead{SPLITS,DK,DV,BN,DOT,SCALE}  (q: DK bf16, k: N rows) -> DV bf16
  FlashMlaV32Decode{NH,SPLITS,DN,DR,QT,BN,DOT,SCALE}  (q: NH x (DN+DR) bf16, nope, scale, rope, idx: the N rows' cache bytes and
                                                  indices) -> NH x DN bf16

A row is a staged row (DK bf16, valid bit), N = SPLITS[-1] * BN the token's rows in index order.  `FLASHMLA_GLM53` holds
GLM-5.3's statics.

A token is cut exactly into units that each fit a Flock block (`decode_cut`, ``CUTS``): its N rows, and for each head the
blocks of each split, each carrying the split's state to the next, the split's end, and where there is more than one split the
weights and then the weighing, `COMBINE_G` columns a unit.  The bodies of `FlashMlaV32DecodeHead` and `FlashMlaV32Decode` hold
nothing but those calls, so the cut is the body.
"""
from __future__ import annotations

from types import SimpleNamespace

from verity.circuits.defs import bind, composite
from verity.circuits.refs import Coll, array_of, concat, tuple_of
from verity.circuits.types import Array, Tuple
from verity.definitions.numerics import flashmla_v32_twin
from verity.definitions.ops.gemm import DotBf16 as DotBf16V2
from verity.definitions.ops.prims import HopperBF16WgmmaDot16
from verity.definitions.registry import flashmla as F

_CONF = ("model-untested: read from FlashMLA 6bc49418's Hopper sparse_fp8 decode at its V32 instance (the module docstring's "
         "files); tests/properties/glm53_flashmla_fp8_capture_gpu.py is its hardware check")

#: GLM-5.3 at TP8 on the H200: 8 of the 64 padded heads per rank, NoPE 512 + RoPE 64, one f32 scale per 128 NoPE elements,
#: 64-row blocks, softmax_scale 256^-0.5
FLASHMLA_GLM53 = {"NH": 8, "DN": 512, "DR": 64, "QT": 128, "BN": 64, "DOT": HopperBF16WgmmaDot16, "SCALE": 0.0625}
#: index_topk: the rows a token names, 32 blocks
GLM53_TOPK = 2048
#: the most columns of the output one `FlashMlaV32CombineValues` weighs: at 512 columns the 32 splits of a step of at most
#: three tokens are past a Flock block, at 256 they fit
COMBINE_G = 256


# ---- the row ----------------------------------------------------------------------------------------------------------------

def kv_row(o: F.Ops, S, nope, scale, rope, idx):
    """`FlashMlaV32KvRow_v1`'s body: `flashmla.kv_row` with each scale an f32 rounded to bf16 and the validity read off the
    index."""
    DN, DR, QT = S.DN, S.DR, S.QT
    F.check_row_statics(DN, DR, QT, "FlashMlaV32KvRow")
    valid = o.bnot(o.ieq(idx, o.word(F.M32)))
    z = o.half(0)
    sc = [o.cvt(scale[j]) for j in range(DN // QT)]
    x = [o.bsel(valid, o.bmul(o.cvt(o.e4m3(nope[d])), sc[d // QT]), z) for d in range(DN)]
    return tuple_of(array_of(x + [o.bsel(valid, rope[d], z) for d in range(DR)]), valid)


def _row_type(S, T: F.Types = F.WORD_T):
    return ((("nope", Array(S.DN, T.e4m3)), ("scale", Array(S.DN // S.QT, T.f32)), ("rope", Array(S.DR, T.bf16)), ("idx", T.i32)),
            F._staged(S.DN + S.DR, T))


@composite("FlashMlaV32KvRow", 1, ["DN", "DR", "QT"], _row_type, conformance=_CONF)
def FlashMlaV32KvRow(B, S, nope, scale, rope, idx):
    """One 656-byte cache row as the V32 decode stages it (splitkv_mla.cuh:543-598), with its validity: the row is valid iff its
    index is not -1 (`is_index_valid`, :630-636, no length at this instance); NoPE element d
    `Bf16MulBf16(F2fpBf16(E4m3ToF32(nope[d])), F2fpBf16(scale[d / QT]))` (dequant.h:21-34), then the RoPE elements as stored;
    every element +0 when the row is invalid."""
    return kv_row(F.word_ops(B), S, nope, scale, rope, idx)


# ---- one block and a split's end --------------------------------------------------------------------------------------------

def check_value_statics(DK: int, DV: int, name: str) -> None:
    if not 1 <= DV <= DK:
        raise ValueError(f"{name}: the value is the key's first DV={DV} of DK={DK} elements")


def _state(DV: int, T: F.Types):
    return Tuple(T.f32, Array(4, T.f32), Array(DV, T.f32))


def _block_type(S, T: F.Types = F.WORD_T):
    state = () if S.FIRST else (("nm_prev", T.f32), ("l_prev", Array(4, T.f32)), ("o_prev", Array(S.DV, T.f32)))
    return ((("q", Array(S.DK, T.bf16)), ("k", Array(S.BN, F._staged(S.DK, T)))) + state, _state(S.DV, T))


@composite("FlashMlaV32Block", 1, ["DK", "DV", "BN", "FIRST", "DOT", "SCALE"], _block_type, conformance=_CONF)
def FlashMlaV32Block(B, S, q, k, nm_prev=None, l_prev=None, o_prev=None):
    """One BN-row block of the decode's online softmax for one head, `FlashMlaBlock_v1`'s arithmetic (`flashmla.block`) with the
    key DK wide and the value its first DV elements: the scores over all DK, P V over the rows' first DV columns, O of DV
    words.  A split's first block (FIRST) starts from the constant state and takes none."""
    check_value_statics(S.DK, S.DV, "FlashMlaV32Block")
    as_block = SimpleNamespace(D=S.DK, BN=S.BN, FIRST=S.FIRST, DOT=S.DOT, SCALE=S.SCALE)
    return F.block(F.word_ops(B), as_block, bind(DotBf16V2, K=S.DK, DOT=S.DOT), q, k, nm_prev, l_prev, o_prev, DV=S.DV)


def _end_type(S, T: F.Types = F.WORD_T):
    args = (("nm", T.f32), ("l", Array(4, T.f32)), ("o", Array(S.DV, T.f32)))
    return args, Tuple(Array(S.DV, T.f32), T.f32) if S.PARTIAL else Array(S.DV, T.bf16)


@composite("FlashMlaV32SplitEnd", 1, ["DV", "PARTIAL"], _end_type, conformance=_CONF)
def FlashMlaV32SplitEnd(B, S, nm, l, acc):
    """A split's end from its last block's state, `FlashMlaSplit_v1`'s (`flashmla.split_end`): the head's DV bf16 words where
    the split is the token's only one, its sink term on the constant -inf (splitkv_mla.cuh:199-206, :305-306); else (PARTIAL)
    the f32 partial and its lse."""
    o = F.word_ops(B)
    return F.split_end(o, S.PARTIAL, nm, l, acc, None if S.PARTIAL else o.word(F.F_NINF))


# ---- the combine ------------------------------------------------------------------------------------------------------------

@composite("FlashMlaV32CombineWeights", 1, ["NS"], lambda S: ((("lse", Array(S.NS, F.F32)),), Array(S.NS, F.F32)), conformance=_CONF)
def FlashMlaV32CombineWeights(B, S, lse):
    """The weights of a head's NS partials: split s weighs exp2(lse[s] - g), g as `flash_fwd_mla_combine_kernel` has it for a
    call with no sink (combine.cu:71-116; `flashmla.combine_scale`)."""
    o = F.word_ops(B)
    ng = F.combine_scale(o, S.NS, lse)
    return array_of([o.exp2(o.add(lse[s], ng)) for s in range(S.NS)])


def check_values_statics(NS: int, G: int) -> None:
    if NS < 2 or G < 1:
        raise ValueError(f"FlashMlaV32CombineValues: NS={NS} splits (at least two) over G={G} columns (at least one)")


def _values_type(S):
    return ((("w", Array(S.NS, F.F32)), ("oacc", Array(S.NS, Array(S.G, F.F32)))), Array(S.G, F.BF16))


@composite("FlashMlaV32CombineValues", 1, ["NS", "G"], _values_type, conformance=_CONF)
def FlashMlaV32CombineValues(B, S, w, oacc):
    """G columns of a head's output from its NS partials' and their weights: from 0, one FFMA a split in split order, then the
    bf16 (combine.cu:120-157).  A column reads its own column of each partial and nothing else, so the units side by side are
    the head's output."""
    check_values_statics(S.NS, S.G)
    o = F.word_ops(B)
    res = o.word(0)
    for s in range(S.NS):
        res = B.batch(o._defs["fma"], w[s], oacc[s], res, axes=(None, 0, None if s == 0 else 0))
    return B.batch(o._defs["cvt"], res, axes=(0,))


def combine_columns(DV: int) -> int:
    """The columns one `FlashMlaV32CombineValues` of a DV-wide head weighs: all of them up to `COMBINE_G`."""
    if DV > COMBINE_G and DV % COMBINE_G:
        raise ValueError(f"FlashMlaV32DecodeHead: past {COMBINE_G} columns DV={DV} must be a multiple of {COMBINE_G}")
    return min(DV, COMBINE_G)


# ---- a head and a token -----------------------------------------------------------------------------------------------------

def _head_units(S) -> dict:
    blk = dict(DK=S.DK, DV=S.DV, BN=S.BN, DOT=S.DOT, SCALE=S.SCALE)
    return {"block": lambda first: bind(FlashMlaV32Block, FIRST=first, **blk),
            "end": lambda partial: bind(FlashMlaV32SplitEnd, DV=S.DV, PARTIAL=partial),
            "weights": lambda ns: bind(FlashMlaV32CombineWeights, NS=ns),
            "values": lambda ns, g: bind(FlashMlaV32CombineValues, NS=ns, G=g)}


def decode_head(B, S, units: dict, q, k):
    """`FlashMlaV32DecodeHead_v1`'s body over ``units`` (`_head_units`): calls and nothing else."""
    BN, DV, sp = S.BN, S.DV, tuple(S.SPLITS)
    F.check_splits(sp)
    check_value_statics(S.DK, DV, "FlashMlaV32DecodeHead")

    def split(a: int, b: int, partial: bool):
        nm = l = acc = None
        for blk in range(a, b):
            kb = k[blk * BN:(blk + 1) * BN]
            res = B.call(units["block"](True), q, kb) if blk == a else B.call(units["block"](False), q, kb, nm, l, acc)
            nm, l, acc = res[0], res[1], res[2]
        return B.call(units["end"](partial), nm, l, acc)

    if len(sp) == 2:
        return split(0, sp[1], False)
    parts = [split(a, b, True) for a, b in zip(sp, sp[1:])]
    NS, G = len(parts), combine_columns(DV)
    w = B.call(units["weights"](NS), array_of([p[1] for p in parts]))
    cols = [B.call(units["values"](NS, G), w, Coll(Array(NS, Array(G, F.F32)), concat([p[0][c:c + G].refs for p in parts])))
            for c in range(0, DV, G)]
    return Coll(Array(DV, F.BF16), concat([c.refs for c in cols]))


def head_cut(head) -> list[tuple[object, int]]:
    """The units a `FlashMlaV32DecodeHead_v1` binding is cut into, each Definition bound and how many of it: each split's first
    block, each later block, each split's end, and past one split the weights and the `combine_columns` blocks of columns."""
    if getattr(head, "defn", None) is not FlashMlaV32DecodeHead:
        raise ValueError(f"head_cut cuts a FlashMlaV32DecodeHead_v1 binding, not {head!r}")
    S = SimpleNamespace(**head.bindings)
    sp, units = tuple(S.SPLITS), _head_units(S)
    NS, NB = len(sp) - 1, sp[-1]
    out = [(units["block"](True), NS)]
    if NB > NS:
        out.append((units["block"](False), NB - NS))
    out.append((units["end"](NS > 1), NS))
    if NS > 1:
        G = combine_columns(S.DV)
        out += [(units["weights"](NS), 1), (units["values"](NS, G), S.DV // G)]
    return out


def _head_type(S, T: F.Types = F.WORD_T):
    N = F.check_splits(S.SPLITS) * S.BN
    return ((("q", Array(S.DK, T.bf16)), ("k", Array(N, F._staged(S.DK, T)))), Array(S.DV, T.bf16))


@composite("FlashMlaV32DecodeHead", 1, ["SPLITS", "DK", "DV", "BN", "DOT", "SCALE"], _head_type, conformance=_CONF)
def FlashMlaV32DecodeHead(B, S, q, k):
    """One head of one token over its N staged rows.  Each span of SPLITS is a split: `FlashMlaV32Block_v1` over its blocks in
    ascending order, the first from the constant state, then `FlashMlaV32SplitEnd_v1`.  One split's end is the head's output;
    several leave partials, which `FlashMlaV32CombineWeights_v1` weighs by their lse and `FlashMlaV32CombineValues_v1` sums,
    `combine_columns` columns a unit."""
    return decode_head(B, S, _head_units(S), q, k)


def _decode_units(S) -> tuple:
    return (bind(FlashMlaV32KvRow, DN=S.DN, DR=S.DR, QT=S.QT),
            bind(FlashMlaV32DecodeHead, SPLITS=tuple(S.SPLITS), DK=S.DN + S.DR, DV=S.DN, BN=S.BN, DOT=S.DOT, SCALE=S.SCALE))


def decode_cut(dec) -> list[tuple[object, int]]:
    """The units a `FlashMlaV32Decode_v1` binding is cut into: its N rows, staged once, and `head_cut` for each of its NH heads.
    What serving stages a token as, and what circuit-check holds to the block in its place."""
    if getattr(dec, "defn", None) is not FlashMlaV32Decode:
        raise ValueError(f"decode_cut cuts a FlashMlaV32Decode_v1 binding, not {dec!r}")
    S = SimpleNamespace(**dec.bindings)
    row, head = _decode_units(S)
    return [(row, F.check_splits(S.SPLITS) * S.BN)] + [(u, n * S.NH) for u, n in head_cut(head)]


def _decode_type(S, T: F.Types = F.WORD_T):
    N, DK = F.check_splits(S.SPLITS) * S.BN, S.DN + S.DR
    return ((("q", Array(S.NH, Array(DK, T.bf16))), ("nope", Array(N, Array(S.DN, T.e4m3))),
             ("scale", Array(N, Array(S.DN // S.QT, T.f32))), ("rope", Array(N, Array(S.DR, T.bf16))), ("idx", Array(N, T.i32))),
            Array(S.NH, Array(S.DN, T.bf16)))


@composite("FlashMlaV32Decode", 1, ["NH", "SPLITS", "DN", "DR", "QT", "BN", "DOT", "SCALE"], _decode_type, conformance=_CONF)
def FlashMlaV32Decode(B, S, q, nope, scale, rope, idx):
    """One token's NH heads over its N cache rows in index order, given each row's cache bytes and index: `FlashMlaV32KvRow_v1`
    of each row once, then head h `FlashMlaV32DecodeHead_v1` over q[h] and the staged rows.  A step's tokens share SPLITS; a
    request of several positions is as many tokens."""
    row, head = _decode_units(S)
    return B.batch(head, q, B.batch(row, nope, scale, rope, idx, axes=(0, 0, 0, 0)), axes=(0, None))


CUTS = {"FlashMlaV32DecodeHead_v1": head_cut, "FlashMlaV32Decode_v1": decode_cut}

FLASHMLA_V32_COMPOSITES = [FlashMlaV32KvRow, FlashMlaV32Block, FlashMlaV32SplitEnd, FlashMlaV32CombineWeights,
                           FlashMlaV32CombineValues, FlashMlaV32DecodeHead, FlashMlaV32Decode]

flashmla_v32_twin.register_twin({d.id: d for d in FLASHMLA_V32_COMPOSITES})
