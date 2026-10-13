"""[DSV4] DeepSeek-V4-Pro's sparse-attention index builder as an H200 (sm_90, TP8) runs it: vLLM d9105ea80, Triton 3.7.1.  Line
numbers are at that commit under `vllm/models/deepseek_v4/`; "the PTX" is Triton 3.7.1's sm_90a output at the served constexprs
(TOP_K 1024 at ratio 4, a step's `active_topk_width` at ratio 128, the window 128, no image width).

A prefill chunk's sparse attention (nvidia/flashmla.py:266-389) reads one gathered workspace: request b's block of M rows from
row b*M, its compressed rows from 0 and its last `gather_len` window rows from N.  `combine_topk_swa_indices`
(common/ops/cache_utils.py:528-582, kernel 654-739) writes each query token's row of workspace indices and the row's length:
`Dsv4CombineTopkSwa_v1{TOPK, RATIO, WINDOW, WIDTH}`.  Its top-k row is, at ratio 4, the indexer's selection
(`topk_indices_buffer`, `dsv4_indexer`'s module docstring), and at ratio 128 `_build_c128a_topk_metadata_kernel`'s prefill row
(sparse_mla.py:361-427): slot i is i when i < min((position + 1) // 128, K), else -1, over the int64 position, which is
`dsv4_indexer.Dsv4ShortContextTopK_v1{K, 128}` (K the step's `active_topk_width`, sparse_mla.py:246-253; no slot below K sees
the min).  TOPK is the row's width (`top_k = topk_indices.shape[-1]`, flashmla.py:304), so the kernel reads no slot past it, and
WIDTH is `round_up(TOPK + WINDOW, 128)` (316): 1024 and 1152 at ratio 4, K and K + 128 at ratio 128.  The model's window-only
layer (ratio 1, top-k 0) is the MTP layer, outside `num_hidden_layers` (attention.py:223-226); it is not stated here.

The PTX, per token (cache_utils.py line numbers; every add wraps at 32 bits):

    t        = add.s32(pos, 1);  q = shr.s32(add.s32(t, shr.u32(shr.s32(t, 31), 32 - k)), k)   (t // 2^k toward 0, k = log2 RATIO)
    topk_len = min.s32(q, TOPK)                                                                  698
    swa_start = max.s32(pos - (WINDOW - 1), 0);  swa_len = pos - swa_start + 1                   709-711
    slot j < PADDED_TOP_K, stored when q > j:  topk[j] + mul.lo.s32(M, b)                        713-723
    slot topk_len + o, o < WINDOW (the address `mad.wide.s32`, so it does not wrap), stored when
        setp.lt.s32(o, swa_len):  M*b + N + swa_start + o - gather_start                         728-736
    len      = topk_len + swa_len                                                                738-739

over a row first filled with -1 (564).  swa_len <= WINDOW at every pos: pos - (WINDOW - 1) wraps only below INT32_MIN + WINDOW
- 1, where swa_len wraps back to WINDOW.  The Definition takes pos and gather_start (seq_len - query_len + the token's index,
and seq_len - gather_len, from `seq_lens`, `query_start_loc` and `gather_lens`), mb = mul.lo.s32(M, b) and N: the schedule's
and the workspace's integers.  Slot j of its row is

    j < TOPK and j < topk_len                        topk[j] + mb
    else j >= topk_len and 0 < swa_len and j < len   c + j,  c = mb + N + swa_start - gather_start - topk_len
    else                                             -1

the window's test j - topk_len < swa_len written as j < len: with 0 < swa_len <= WINDOW and -2^30 <= topk_len <= TOPK (RATIO >=
2), and j < WIDTH < 2^30, neither side wraps.  The kernel's own slots past its row are not stated: a token whose topk_len is
negative while swa_len > 0 (pos = INT32_MAX, or pos below INT32_MIN + WINDOW - 1) stores its window at negative offsets, memory
outside its row; the Definition gives that row's own slots (all -1) and its length.  A served position is below
`max_model_len`.

The decode's indices are not Definitions here.  The window's (`_compute_swa_indices_and_lens_kernel`,
v1/attention/backends/mla/sparse_swa.py:931-1008) are slot ids of the paged window cache: for a valid token, slot o is
`block_table[req][(start + o) // block_size] * block_size + (start + o) % block_size` for o < its window length (the
arithmetic of the window above) and -1 past it, and an invalid token's row is all -1 with length 0.  That is a function of the
block table, the position and the window, all schedule metadata, the class of `_compute_slot_mappings_kernel` (the
public-metadata kernels `kernel_allowlist` names): FlashMLA's decode takes the rows they select and their lengths, and which
rows is wiring the verifier derives from the schedule, as the capture resolves each slot to its position through the step's
block tables.  Ratio 128's decode row is the same class (`_build_c128a_topk_metadata_kernel`'s decode branch, sparse_mla.py:
388-416: the block-table slots of positions below min((position + 1) // 128, K), the length 0 for an invalid token).  Still
open: ratio 4's decode row, `compute_global_topk_indices_and_lens` (cache_utils.py:435-517), the block-table lookup of the
indexer's selection, a function of a committed value.

Registered functions:
  Dsv4CombineTopkSwa{TOPK,RATIO,WINDOW,WIDTH}  (topk: TOPK i32, pos, mb, n, gs: i32) -> (WIDTH i32, len: i32)
`boolean_dsv4_indices` has its Boolean `_v2`; `kernels.dsv4_indices_twin` is its numpy twin, the kernel's stores.
"""
from __future__ import annotations

from types import SimpleNamespace

from verity.ml.circuits.defs import composite
from verity.ml.circuits.ops.prims import const
from verity.ml.circuits.ops.scalar import BitAnd, BitNot, F32Neg, I32Add, I32Le, SelectI32
from verity.ml.circuits.refs import Coll, concat
from verity.ml.circuits.types import Array, Tuple
from verity_vllm.program.kernels import dsv4_indices_twin
from verity_vllm.program.registry import dsa as D
from verity_vllm.program.registry.prims import I32
from verity_vllm.program.registry.sampling import bit_at

_CONF = ("model-untested: read from the Triton kernel and Triton 3.7.1's PTX at the commits the module docstring pins; "
         "tests/properties/dsv4_indices_capture_gpu.py is its hardware check")

M32 = D.M32

#: the served statics: ratio 4's top-k of 1024 in a row of 1152, ratio 128's at its widest (`active_topk_width` 8192), window 128
DSV4_INDICES = {"c4": {"TOPK": 1024, "RATIO": 4, "WINDOW": 128, "WIDTH": 1152},
                "c128": {"TOPK": 8192, "RATIO": 128, "WINDOW": 128, "WIDTH": 8320}}

#: the element types of a signature: these words, and their bits in `boolean_dsv4_indices`
WORDS = SimpleNamespace(I32=I32)

WORD = dict(iadd=I32Add, isub=D.I32Sub, ile=I32Le, isel=SelectI32, band=BitAnd, bnot=BitNot)


class Ops(D.Ops):
    """`dsa.Ops` with ``bit(x, i)``, bit i of an i32 word (`BitAtx32_v1`)."""

    def bit(self, x, i: int):
        return self.B.call(bit_at(32), x, self.word(i))


def word_ops(B) -> Ops:
    return Ops(B, WORD, lambda bits: B.call(const(32, bits)), lambda x: B.call(F32Neg, x))


def check_combine_statics(TOPK: int, RATIO: int, WINDOW: int, WIDTH: int) -> int:
    """log2 RATIO, for a power-of-two RATIO >= 2 and 1 <= TOPK, 1 <= WINDOW with TOPK + WINDOW <= WIDTH < 2^30."""
    if not (RATIO >= 2 and RATIO & (RATIO - 1) == 0 and TOPK >= 1 and WINDOW >= 1 and TOPK + WINDOW <= WIDTH < 1 << 30):
        raise ValueError(f"Dsv4CombineTopkSwa: RATIO a power of two >= 2, TOPK, WINDOW >= 1, TOPK + WINDOW <= WIDTH < 2^30 "
                         f"(TOPK={TOPK}, RATIO={RATIO}, WINDOW={WINDOW}, WIDTH={WIDTH})")
    return RATIO.bit_length() - 1


def combine_sig(T):
    return lambda S: ((("topk", Array(S.TOPK, T.I32)), ("pos", T.I32), ("mb", T.I32), ("n", T.I32), ("gs", T.I32)),
                      Tuple(Array(S.WIDTH, T.I32), T.I32))


def combine(o, S, topk, pos, mb, n, gs) -> list:
    """`Dsv4CombineTopkSwa_v1`'s body: q from u = t + (t < 0 ? RATIO - 1 : 0), its bit i + k as bit i below 31 - k and its sign
    as the top k + 1 bits, then the row of the module docstring."""
    k = check_combine_statics(S.TOPK, S.RATIO, S.WINDOW, S.WIDTH)
    (p,), (mb,), (n,), (gs,) = pos, mb, n, gs
    zero, minus1 = o.word(0), o.word(M32)
    t = o.iadd(p, o.word(1))
    u = o.iadd(t, o.isel(o.bit(t, 31), o.word(S.RATIO - 1), zero))
    q = o.isel(o.bit(u, 31), o.word((M32 << (31 - k)) & M32), zero)
    for i in range(31 - k):
        q = o.iadd(q, o.isel(o.bit(u, i + k), o.word(1 << i), zero))
    tl = o.isel(o.ile(o.word(S.TOPK), q), o.word(S.TOPK), q)
    s0 = o.isub(p, o.word(S.WINDOW - 1))
    start = o.isel(o.ile(zero, s0), s0, zero)
    sl = o.isub(t, start)
    ln = o.iadd(tl, sl)
    c = o.isub(o.isub(o.iadd(o.iadd(mb, n), start), gs), tl)
    live = o.ile(o.word(1), sl)
    row = []
    for j in range(S.WIDTH):
        win = o.band(live, o.ile(o.word(j + 1), ln))
        if j < S.TOPK:
            lt = o.ile(o.word(j + 1), tl)
            row.append(o.isel(lt, o.iadd(topk[j], mb), o.isel(win, o.iadd(c, o.word(j)), minus1)))
        else:
            row.append(o.isel(win, o.iadd(c, o.word(j)), minus1))
    return row + [ln]


@composite("Dsv4CombineTopkSwa", 1, ["TOPK", "RATIO", "WINDOW", "WIDTH"], combine_sig(WORDS), conformance=_CONF)
def Dsv4CombineTopkSwa(B, S, topk, pos, mb, n, gs):
    """One query token of `combine_topk_swa_indices` (common/ops/cache_utils.py:654-739): its row of WIDTH workspace indices, the
    top-k's below topk_len = min((pos + 1) // RATIO, TOPK) offset by mb, then its window's swa_len rows from N + swa_start -
    gs, -1 elsewhere, and the row's length topk_len + swa_len (the module docstring)."""
    _, ret = combine_sig(WORDS)(S)
    return Coll(ret, concat([e.refs for e in combine(word_ops(B), S, [topk[i] for i in range(S.TOPK)], [pos], [mb], [n], [gs])]))


DSV4_INDICES_COMPOSITES = [Dsv4CombineTopkSwa]

dsv4_indices_twin.register_twin({d.id: d for d in DSV4_INDICES_COMPOSITES})
