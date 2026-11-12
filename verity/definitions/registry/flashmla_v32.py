"""[GLM-5.3] The sparse MLA attention over an FP8 cache as the H200 runs it: GLM-5.3 at TP8 on sm_90, vLLM d9105ea80's
`FlashMLASparseImpl._forward_fp8_kv_mixed_batch` (vllm/v1/attention/backends/mla/flashmla_sparse.py:826) over the FlashMLA it
vendors (6bc49418): the `sparse_fp8` decode kernel at its V32 instance, `flash_fwd_splitkv_mla_fp8_sparse_kernel<
KernelTemplate<V32, 64>>` (csrc/sm90/decode/sparse_fp8/instantiations/v32_persistent_h64.cu), with the planner and the combine
`flashmla` describes.  Line numbers are at those commits.

The call.  With fewer than 32 heads a rank (`MIN_HEADS_FOR_BF16_PREFILL`) and an `fp8_ds_mla` cache, every token of a step,
decode and prefill alike, goes through one call (:870-897, :899-937): q (1, T, 64, 576) bf16, the rank's 8 real heads padded to
64 with zeros (`_compute_fp8_decode_padded_heads`, :587-590); the cache (blocks, 64, 1, 656) bytes; indices (1, T, 2048), each
token's `index_topk` rows as cache slots, -1 where it has none (`triton_convert_req_index_to_global_index`); head_dim_v 512;
softmax_scale qk_head_dim^-0.5 = 1/16; no sink, no lengths, no second cache.  The 56 pad heads' outputs are sliced away.

What this instance does that `flashmla`'s (MODEL1) does not:

1. The row (656 bytes; splitkv_mla.cuh:543-598, dequant.h:21-34): 512 E4M3 NoPE bytes, then four f32 scales, one per 128 NoPE
   elements, then 64 bf16 RoPE elements.  A scale is rounded to bf16 before it multiplies (`(bf16)scales_float[i]`), and a NoPE
   element is `__float22bfloat162_rn(float(e4m3)) * scale`, one bf16 multiply: `FlashMlaV32KvRow_v1`.
2. Validity: a row is valid iff its index is not -1; there is no length (`topk_length` is null at this instance).  An invalid
   row's NoPE bytes and scales are zeroed, so its NoPE elements are +0; its RoPE elements are whatever row the index -1
   addresses and reach no output, the score they enter being replaced by -inf and the value being the NoPE elements alone.
3. The key is the 576-element row [NoPE 512 | RoPE 64] and the value its first 512 elements, so the score chain is 36 k16 steps
   and the output 512 bf16 a head.  `FlashMlaSplit_v1` at D = 576 computes those 512 words and 64 more the kernel never
   forms (the same wgmma columns over the RoPE elements), which `FlashMlaV32DecodeHead_v1` drops.
4. No sink: the kernel runs its sink arm on -inf (`rAttn_sink = -inf` when the pointer is null) and the combine skips its
   own, which is `FlashMlaSplit_v1` and `FlashMlaCombine_v1` on a sink of -inf.
5. The schedule.  The call is one request of s_q = T query rows, so num_sm_parts = max(132 / T, 1) and every token of the
   step has the same SPLITS over its 32 blocks, a function of T alone (`flashmla_twin.sched_meta([2048], topk=2048,
   parts=num_sm_parts(64, s_q=T))`): 32 splits of one block at T <= 3 down to one split, written by the decode kernel with no
   combine, from T = 67.

Registered functions (all `Name_v1`):
  FlashMlaV32KvRow{DN,DR,QT}                      (nope: DN e4m3, scale: DN/QT f32, rope: DR bf16, valid: bit) -> DN+DR bf16
  FlashMlaV32DecodeHead{SPLITS,DK,DV,BN,DOT,SCALE}  (q: DK bf16, k: N rows) -> DV bf16
  FlashMlaV32Decode{NH,SPLITS,DN,DR,QT,BN,DOT,SCALE}  (q: NH x (DN+DR) bf16, nope, scale, rope, idx: the N rows' cache bytes and
                                                  indices) -> NH x DN bf16

A row is a staged row (DK bf16, valid bit), N = SPLITS[-1] * BN the token's rows in index order.  `FLASHMLA_GLM53` holds
GLM-5.3's statics.
"""
from __future__ import annotations

from verity.circuits.defs import bind, composite
from verity.circuits.refs import Coll, array_of, concat
from verity.circuits.types import Array
from verity.definitions.numerics import flashmla_v32_twin
from verity.definitions.ops.prims import HopperBF16WgmmaDot16
from verity.definitions.registry import flashmla as F

_CONF = ("model-untested: read from FlashMLA 6bc49418's Hopper sparse_fp8 decode at its V32 instance (the module docstring's "
         "files); tests/properties/glm53_flashmla_fp8_capture_gpu.py is its hardware check")

#: GLM-5.3 at TP8 on the H200: 8 of the 64 padded heads per rank, NoPE 512 + RoPE 64, one f32 scale per 128 NoPE elements,
#: 64-row blocks, softmax_scale 256^-0.5
FLASHMLA_GLM53 = {"NH": 8, "DN": 512, "DR": 64, "QT": 128, "BN": 64, "DOT": HopperBF16WgmmaDot16, "SCALE": 0.0625}
#: index_topk: the rows a token names, 32 blocks
GLM53_TOPK = 2048


def kv_row(o: F.Ops, S, nope, scale, rope, valid):
    """`FlashMlaV32KvRow_v1`'s body: `flashmla.kv_row` with each scale an f32 rounded to bf16."""
    DN, DR, QT = S.DN, S.DR, S.QT
    F.check_row_statics(DN, DR, QT, "FlashMlaV32KvRow")
    z = o.half(0)
    sc = [o.cvt(scale[j]) for j in range(DN // QT)]
    x = [o.bsel(valid, o.bmul(o.cvt(o.e4m3(nope[d])), sc[d // QT]), z) for d in range(DN)]
    return array_of(x + [o.bsel(valid, rope[d], z) for d in range(DR)])


def _row_type(S, T: F.Types = F.WORD_T):
    return ((("nope", Array(S.DN, T.e4m3)), ("scale", Array(S.DN // S.QT, T.f32)), ("rope", Array(S.DR, T.bf16)),
             ("valid", T.bit)), Array(S.DN + S.DR, T.bf16))


@composite("FlashMlaV32KvRow", 1, ["DN", "DR", "QT"], _row_type, conformance=_CONF)
def FlashMlaV32KvRow(B, S, nope, scale, rope, valid):
    """One 656-byte cache row as the V32 decode stages it (splitkv_mla.cuh:543-598): NoPE element d
    `Bf16MulBf16(F2fpBf16(E4m3ToF32(nope[d])), F2fpBf16(scale[d / QT]))` (dequant.h:21-34), then the RoPE elements as stored;
    every element +0 when the row is invalid."""
    return kv_row(F.word_ops(B), S, nope, scale, rope, valid)


def check_head_statics(DK: int, DV: int) -> None:
    if not 1 <= DV <= DK:
        raise ValueError(f"FlashMlaV32DecodeHead: the value is the key's first DV={DV} of DK={DK} elements")


def decode_head(o: F.Ops, S, split_of, combine_of, q, k):
    """`FlashMlaV32DecodeHead_v1`'s body, ``split_of(nb, partial)`` the bound split over the DK-wide rows, ``combine_of(ns)``
    the bound combine over DV words."""
    B, BN, DV, sp = o.B, S.BN, S.DV, tuple(S.SPLITS)
    F.check_splits(sp)
    check_head_statics(S.DK, DV)
    ninf = o.word(F.F_NINF)
    if len(sp) == 2:
        return B.call(split_of(sp[1], False), q, k, ninf)[:DV]
    parts = [B.call(split_of(b - a, True), q, k[a * BN:b * BN]) for a, b in zip(sp, sp[1:])]
    oacc = Coll(Array(len(parts), Array(DV, o.T.f32)), concat([p[0][:DV].refs for p in parts]))
    return B.call(combine_of(len(parts)), array_of([p[1] for p in parts]), oacc, ninf)


def _head_type(S, T: F.Types = F.WORD_T):
    N = F.check_splits(S.SPLITS) * S.BN
    return ((("q", Array(S.DK, T.bf16)), ("k", Array(N, F._staged(S.DK, T)))), Array(S.DV, T.bf16))


@composite("FlashMlaV32DecodeHead", 1, ["SPLITS", "DK", "DV", "BN", "DOT", "SCALE"], _head_type, conformance=_CONF)
def FlashMlaV32DecodeHead(B, S, q, k):
    """One head of one token over its N staged rows: the first DV words of one `FlashMlaSplit_v1` at D = DK on a sink of -inf, or
    one partial split per span of SPLITS and `FlashMlaCombine_v1` over the first DV words of each, on a sink of -inf."""
    def split_of(nb, partial):
        return bind(F.FlashMlaSplit, NB=nb, D=S.DK, BN=S.BN, PARTIAL=partial, DOT=S.DOT, SCALE=S.SCALE)
    return decode_head(F.word_ops(B), S, split_of, lambda ns: bind(F.FlashMlaCombine, NS=ns, D=S.DV), q, k)


def decode(o: F.Ops, S, row, head, q, nope, scale, rope, idx):
    """`FlashMlaV32Decode_v1`'s body: each row valid iff its index is not -1, staged once for every head."""
    B, T = o.B, o.T
    N = F.check_splits(S.SPLITS) * S.BN
    valid = [o.bnot(o.ieq(idx[r], o.word(F.M32))) for r in range(N)]
    rows = B.batch(row, nope, scale, rope, Coll(Array(N, T.bit), concat([v.refs for v in valid])), axes=(0, 0, 0, 0))
    staged = Coll(Array(N, F._staged(S.DN + S.DR, T)), concat([x for r in range(N) for x in (rows[r].refs, valid[r].refs)]))
    return B.batch(head, q, staged, axes=(0, None))


def _decode_type(S, T: F.Types = F.WORD_T):
    N, DK = F.check_splits(S.SPLITS) * S.BN, S.DN + S.DR
    return ((("q", Array(S.NH, Array(DK, T.bf16))), ("nope", Array(N, Array(S.DN, T.e4m3))),
             ("scale", Array(N, Array(S.DN // S.QT, T.f32))), ("rope", Array(N, Array(S.DR, T.bf16))), ("idx", Array(N, T.i32))),
            Array(S.NH, Array(S.DN, T.bf16)))


@composite("FlashMlaV32Decode", 1, ["NH", "SPLITS", "DN", "DR", "QT", "BN", "DOT", "SCALE"], _decode_type, conformance=_CONF)
def FlashMlaV32Decode(B, S, q, nope, scale, rope, idx):
    """One token's NH heads over its N cache rows in index order, given each row's cache bytes and index: each row valid iff its
    index is not -1 and `FlashMlaV32KvRow_v1` once, then head h `FlashMlaV32DecodeHead_v1` over q[h]."""
    row = bind(FlashMlaV32KvRow, DN=S.DN, DR=S.DR, QT=S.QT)
    head = bind(FlashMlaV32DecodeHead, SPLITS=tuple(S.SPLITS), DK=S.DN + S.DR, DV=S.DN, BN=S.BN, DOT=S.DOT, SCALE=S.SCALE)
    return decode(F.word_ops(B), S, row, head, q, nope, scale, rope, idx)


FLASHMLA_V32_COMPOSITES = [FlashMlaV32KvRow, FlashMlaV32DecodeHead, FlashMlaV32Decode]

flashmla_v32_twin.register_twin({d.id: d for d in FLASHMLA_V32_COMPOSITES})
