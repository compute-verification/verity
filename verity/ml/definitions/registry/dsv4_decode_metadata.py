"""[DSV4] DeepSeek-V4-Pro's window (SWA) and ratio-128 (HCA) decode index rows as an H200 (sm_90, TP8) runs them: vLLM d9105ea80,
Triton 3.7.1.  Line numbers are at that commit under `vllm/`; "the PTX" is Triton 3.7.1's sm_90a output at the served arguments
(four warps).

Every layer's decode attention (models/deepseek_v4/nvidia/flashmla.py:174-264) reads its window's rows of the SWA cache at
`decode_swa_indices`, `decode_swa_lens` of them (209-210, 256-257), and an HCA layer also its compressed rows at
`c128a_global_decode_topk_indices`, `c128a_decode_topk_lens` of them (204-207, 261-262); a CSA layer's compressed rows are
`dsv4_decode_indices`'.  Both rows here are metadata builders', one Triton program per decode token.

The window.  `_compute_swa_indices_and_lens_kernel` (v1/attention/backends/mla/sparse_swa.py:924-1008), launched by the SWA
builder (593-609) at window_size = index_width = the config's sliding_window, 128 (424, 451-457), block_size 64 (the SWA cache's,
92, 401), HAS_IMAGE false and token_offset 0, reads per decode token t: its `is_valid_token` byte (torch.bool, slot_mapping >= 0,
555-556), req = `token_to_req_indices[t]`, qs = `query_start_loc[req]`, qe = `query_start_loc[req + 1]`, seq = `seq_lens[req]`
and the request's row of the SWA group's block table (535-539).  Per token (sparse_swa.py lines; int32, every add wrapping, `//`
and `%` C's):

    valid  = the byte != 0 (`ld.global.b8`, `and.b16` 255, `setp.ne.b16`)                                        952-953
    invalid: the length 0 and the row -1                                                                          953-963
    pos    = (seq - (qe - qs)) + t - qs  (the PTX's seq + t - qe, the same word)                                  965-974
    start  = max(pos - (SW - 1), 0)  (left = right = 0, so left_add = max(0 - (SW - 1), 0) = 0)                  982-985
    end    = pos + 1;  len = end - start                                                                          986-989
    lane j < SW: p = start + j;  bn = blocks[p // BS], loaded where p < end (0 elsewhere);
                 slot = bn * BS + p % BS;  j < len ? slot : -1                                                    991-1008

The PTX does p // BS as `div.s32` by the block size, p % BS as p - (p // BS) * BS and the address as `mad.wide.s32`, so it does not
wrap.  start >= 0, so on a lane whose p = start + j does not wrap, p // BS = start // BS + (r0 + j) // BS and p % BS = (r0 + j) %
BS, r0 = start % BS: the row reads only the K = (BS - 1 + SW - 1) // BS + 1 blocks from start // BS (3 at the served 128 and 64).
The Definition looks each of them up once (`GatherI32x{NB}_v1`, as the CSA row does) and each lane selects among the K
(`GatherI32x{K}_v1`).  An invalid token's row is the row of length 0, so the Definition gives the length 0 there and keeps the
rest.  It differs from the kernel only where a live lane (j < len) wraps (pos below INT32_MIN + SW - 1, where len wraps to SW),
is masked (p >= end) or names a block past its row: the twin declines those, and a served position is below max_model_len.

The ratio-128 row.  `_build_c128a_topk_metadata_kernel`'s decode branch (models/deepseek_v4/sparse_mla.py:361-416), launched once
per step by `build_c128a_topk_metadata` (295-358) from the HCA builder (222-278) at max_compressed_tokens W = the step's
`active_topk_width` (246-252: min(max(next_power_of_2(max_seq_len // 128), 128), cdiv(cdiv(max_model_len, 128), 128) * 128); sm_90
writes and FlashMLA reads that width, 328-332), compress_ratio R = 128 and block_size BS = 256 // 128 = 2 (255), reads per decode
token t: its int64 position, its int64 `slot_mapping[t]` (the token's, 263) and the request's row of the HCA layers' KV-cache
group's block table (`cm.block_table_tensor`, clamped in place to >= 0 by `build`, 194), at block 256 as a CSA layer's is.  Per
token (sparse_mla.py lines):

    q    = (position + 1) // R  (int64, truncating: `add.s64`, then `div.s64`, or `div.u32` when both fit 32 bits);
    nc   = min(q, W)  (`min.s64`)                                                                                 383-385
    lane j < W: live = j < nc (`setp.gt.s64`);  bn = blocks[j // BS], loaded where live;
                slot = bn * BS + j % BS;  live ? slot : -1                                                        393-410
    len  = slot_mapping[t] >= 0 ? (the count of live j) : 0                                                       390, 411-416

For j < W, j < min(q, W) is j < q: lane j is live iff `dsv4_indexer.Dsv4ShortContextTopK_v1{W, R}` gives it j (else -1), which is
also this kernel's prefill row and the top-k fill's (`dsv4_indices`' docstring), and on a live lane j % BS = x % BS for that word x.
Lane j's block index j // BS is a constant, so the row reads blocks[j // BS] without a lookup, one Call per block of BS lanes (W is
a multiple of BS, as the served widths, multiples of 128, are of 2), which computes the block's bn * BS once for its lanes; a lane
past the row (j // BS >= NB) reads -1 there, `GatherI32x{NB}_v1`'s fill (the kernel reads the next request's row: the twin declines
it).  A served live lane is below max_model_len / 128 <= BS * NB.

Each row's NB words are its block table's row stride (`get_block_table_width`, v1/worker/block_table.py:29-49, of
`max_num_blocks_per_req` = cdiv(max_model_len, block size), v1/kv_cache_interface.py:440-443, aligned to 128 tokens): the SWA
group's at block 64, 16384 at max_model_len 1,048,576 and 64 at 4096; the HCA group's at block 256, 4096 and 16.

Registered functions (`_v1`):
  Dsv4SwaWindow{SW,BS,NB}          (t, qs, qe, seq: i32, blocks: NB i32, valid: u8) -> (r0, len: i32, base: K i32): r0 = start % BS,
                                   the stored length, base[i] = blocks[start // BS + i] * BS
  Dsv4SwaLanes{SW,BS}              (r0, len: i32, base: K i32) -> SW i32: lane j, q = r0 % BS + j, j < len ? base[q // BS] + q % BS : -1
  Dsv4SwaIndices{SW,BS,NB}         (t, qs, qe, seq, blocks, valid) -> (SW i32, len: i32): one token's window row and length
  Dsv4C128aSlot{BS}                (x: BS i32, bn: i32) -> (BS i32, live: a bit a lane): lane i, x[i] >= 0 ? bn * BS + x[i] % BS :
                                   -1, and x[i] >= 0
  Dsv4C128aPastSlot{BS}            (x: i32) -> (i32, live: bit): a lane past the row, bn = -1: x >= 0 ? x % BS - BS : -1, and x >= 0
  Dsv4C128aLen{W}                  (live: W bits, slot: i64) -> i32: the count of live, 0 where slot < 0
  Dsv4C128aGlobalTopk{W,R,BS,NB}   (position: i64, slot: i64, blocks: NB i32) -> (W i32, len: i32): one token's row and length
`boolean_dsv4_decode_metadata` has their Boolean `_v2`s; `kernels.dsv4_decode_metadata_twin` is the two rows' numpy twin.
"""
from __future__ import annotations

from types import SimpleNamespace

from verity.ml.circuits.defs import bind, composite
from verity.ml.circuits.ops.prims import const
from verity.ml.circuits.ops.scalar import BIT, BitNot, BitOr, I32Add, I32Le, SelectI32
from verity.ml.circuits.refs import Coll, array_of, concat, tuple_of
from verity.ml.circuits.types import Array, Tuple, Value
from verity.ml.definitions.numerics import dsv4_decode_metadata_twin
from verity.ml.definitions.registry import dsa as D
from verity.ml.definitions.registry import dsv4_indexer as IX
from verity.ml.definitions.registry import marlin_moe as MM
from verity.ml.definitions.registry.dsv4_decode_indices import FILL, U8, gather_i32
from verity.ml.definitions.registry.prims import I32
from verity.ml.definitions.registry.sampling import bit_at

_CONF = ("model-untested: read from the Triton kernels and Triton 3.7.1's PTX at the commits the module docstring pins; "
         "tests/properties/dsv4_decode_metadata_capture_gpu.py is its hardware check")

M32 = D.M32
I64 = Value(64)

#: the served statics: the window of 128 at block 64 over the SWA group's row at max_model_len 1,048,576 and at the inventory's
#: 4096; ratio 128 at its widest step (`active_topk_width` 8192) and at 4096's only one (128), block 2, the HCA group's row
DSV4_DECODE_METADATA = {"swa": {"SW": 128, "BS": 64, "NB": 16384}, "swa-4096": {"SW": 128, "BS": 64, "NB": 64},
                        "c128": {"W": 8192, "R": 128, "BS": 2, "NB": 4096}, "c128-4096": {"W": 128, "R": 128, "BS": 2, "NB": 16}}

#: the element types of a signature: these words, and their bits in `boolean_dsv4_decode_metadata`
WORDS = SimpleNamespace(I32=I32, I64=I64, U8=U8, BIT=BIT, LIVE=BIT)

WORD = dict(iadd=I32Add, isub=D.I32Sub, iand=D.U32And, ile=I32Le, isel=SelectI32, bnot=BitNot, idiv=MM.I32Div, irem=MM.I32Rem,
            imul=MM.I32Mul)


class Ops(D.Ops):
    """`dsa.Ops` over `WORD`, with the lookup ``gather(idx, words)`` (`GatherI32x{n}_v1`, n the count of ``words``), ``nonzero(byte)``
    (any of a u8's 8 bits, as `ld.global.b8` then a test against 0) and the C division, remainder and product by a static power of
    two n (``div``, ``rem``, ``mul``: `I32Div_v1`, `I32Rem_v1`, `I32Mul_v1` by the constant)."""

    def gather(self, idx, words):
        return self.B.call(gather_i32(words.type.n), idx, words)

    def nonzero(self, byte):
        at = bit_at(8)
        nz = self.B.call(at, byte, self.word(0))
        for i in range(1, 8):
            nz = self.B.call(BitOr, nz, self.B.call(at, byte, self.word(i)))
        return nz

    def div(self, x, n: int):
        return self.idiv(x, self.word(n))

    def rem(self, x, n: int):
        return self.irem(x, self.word(n))

    def mul(self, x, n: int):
        return self.imul(x, self.word(n))


def word_ops(B) -> Ops:
    return Ops(B, WORD, lambda bits: B.call(const(32, bits)), None)


# --- the window ------------------------------------------------------------------------------------------------------------------

def check_swa_statics(SW: int, BS: int, NB: int = 1, name: str = "Dsv4SwaIndices") -> int:
    """K, the blocks a window reads, for 1 <= SW < 2^30, a power-of-two BS below 2^30 and 1 <= NB <= 2^(31 - log2 BS) (every
    block a start can name)."""
    k = BS.bit_length() - 1
    if not (1 <= SW < 1 << 30 and BS >= 1 and BS & (BS - 1) == 0 and k < 30 and 1 <= NB <= 1 << (31 - k)):
        raise ValueError(f"{name}: 1 <= SW < 2^30, BS a power of two below 2^30, 1 <= NB <= 2^(31 - log2 BS) "
                         f"(SW={SW}, BS={BS}, NB={NB})")
    return (BS - 1 + SW - 1) // BS + 1


def window_sig(T):
    return lambda S: ((("t", T.I32), ("qs", T.I32), ("qe", T.I32), ("seq", T.I32), ("blocks", Array(S.NB, T.I32)), ("valid", T.U8)),
                      Tuple(T.I32, T.I32, Array(check_swa_statics(S.SW, S.BS, S.NB, "Dsv4SwaWindow"), T.I32)))


def lanes_sig(T):
    return lambda S: ((("r0", T.I32), ("len", T.I32), ("base", Array(check_swa_statics(S.SW, S.BS, 1, "Dsv4SwaLanes"), T.I32))),
                      Array(S.SW, T.I32))


def swa_sig(T):
    return lambda S: (window_sig(T)(S)[0], Tuple(Array(S.SW, T.I32), T.I32))


def window(o, S, t, qs, qe, seq, blocks, valid) -> tuple:
    """`Dsv4SwaWindow_v1`'s body over the steps ``o``: the token's position, its window's start, end and stored length (0 where the
    byte is 0), r0 = start % BS and the K blocks from start // BS, each times BS: (r0, len, [base])."""
    K = check_swa_statics(S.SW, S.BS, S.NB, "Dsv4SwaWindow")
    zero = o.word(0)
    pos = o.isub(o.iadd(o.isub(seq, o.isub(qe, qs)), t), qs)
    s0 = o.isub(pos, o.word(S.SW - 1))
    start = o.isel(o.ile(zero, s0), s0, zero)
    end = o.iadd(pos, o.word(1))
    length = o.isel(o.nonzero(valid), o.isub(end, start), zero)
    b0 = o.div(start, S.BS)
    base = [o.mul(o.gather(o.iadd(b0, o.word(i)), blocks), S.BS) for i in range(K)]
    return o.rem(start, S.BS), length, base


def lanes(o, S, r0, length, base) -> list:
    """`Dsv4SwaLanes_v1`'s body over the steps ``o``: lane j < SW is q = (r0 & (BS - 1)) + j, then j < len ? base[q / BS] + q % BS
    : -1 (`GatherI32x{K}_v1`; q < BS + SW - 1, so q / BS < K)."""
    check_swa_statics(S.SW, S.BS, 1, "Dsv4SwaLanes")
    r = o.iand(r0, o.word(S.BS - 1))
    minus1 = o.word(M32)
    out = []
    for j in range(S.SW):
        q = o.iadd(r, o.word(j))
        slot = o.iadd(o.gather(o.div(q, S.BS), base), o.rem(q, S.BS))
        out.append(o.isel(o.bnot(o.ile(length, o.word(j))), slot, minus1))
    return out


@composite("Dsv4SwaWindow", 1, ["SW", "BS", "NB"], window_sig(WORDS), conformance=_CONF)
def Dsv4SwaWindow(B, S, t, qs, qe, seq, blocks, valid):
    """One decode token's window in `_compute_swa_indices_and_lens_kernel` (v1/attention/backends/mla/sparse_swa.py:952-989): its
    position, start = max(pos - (SW - 1), 0), the stored length (end - start, 0 where the byte is 0), r0 = start % BS and the K
    blocks of the request's block-table row from start // BS, each times BS (`GatherI32x{NB}_v1`, -1 past the row)."""
    r0, length, base = window(word_ops(B), S, t, qs, qe, seq, blocks, valid)
    return tuple_of(r0, length, array_of(base))


@composite("Dsv4SwaLanes", 1, ["SW", "BS"], lanes_sig(WORDS), conformance=_CONF)
def Dsv4SwaLanes(B, S, r0, length, base):
    """The window's SW slots (v1/attention/backends/mla/sparse_swa.py:991-1008) over `Dsv4SwaWindow_v1`'s (r0, len, base): lane j
    is base[q / BS] + q % BS, q = r0 % BS + j, where j < len, else -1 (the module docstring)."""
    return array_of(lanes(word_ops(B), S, r0, length, base))


def swa_row(B, S, T, window_fn, lanes_fn, t, qs, qe, seq, blocks, valid) -> Coll:
    """`Dsv4SwaIndices`' body over its pieces' Definitions (of element types ``T``): the window's Call, then its lanes'."""
    _, ret = swa_sig(T)(S)
    w = B.call(bind(window_fn, SW=S.SW, BS=S.BS, NB=S.NB), t, qs, qe, seq, blocks, valid)
    row = B.call(bind(lanes_fn, SW=S.SW, BS=S.BS), w[0], w[1], w[2])
    return Coll(ret, concat([row.refs, w[1].refs]))


@composite("Dsv4SwaIndices", 1, ["SW", "BS", "NB"], swa_sig(WORDS), conformance=_CONF)
def Dsv4SwaIndices(B, S, t, qs, qe, seq, blocks, valid):
    """One decode token of `_compute_swa_indices_and_lens_kernel` (v1/attention/backends/mla/sparse_swa.py:924-1008): its row of
    SW slot ids of the paged window cache and its length, `Dsv4SwaWindow_v1{SW,BS,NB}` then `Dsv4SwaLanes_v1{SW,BS}` (the module
    docstring)."""
    return swa_row(B, S, WORDS, Dsv4SwaWindow, Dsv4SwaLanes, t, qs, qe, seq, blocks, valid)


# --- ratio 128 ---------------------------------------------------------------------------------------------------------------------

def check_c128a_slot_statics(BS: int) -> int:
    """log2 BS, for a power-of-two BS below 2^31."""
    k = BS.bit_length() - 1
    if not (BS >= 1 and BS & (BS - 1) == 0 and k < 31):
        raise ValueError(f"Dsv4C128aSlot: BS a power of two below 2^31 (BS={BS})")
    return k


def check_c128a_len_statics(W: int) -> None:
    if not 1 <= W < 1 << 31:
        raise ValueError(f"Dsv4C128aLen: 1 <= W < 2^31 (W={W})")


def check_c128a_statics(W: int, R: int, BS: int, NB: int) -> None:
    IX.check_topk_statics(W, R)
    check_c128a_slot_statics(BS)
    if NB < 1 or W % BS:
        raise ValueError(f"Dsv4C128aGlobalTopk: NB >= 1 and W a multiple of BS (W={W}, BS={BS}, NB={NB})")


def c128a_slot_sig(T):
    return lambda S: ((("x", Array(S.BS, T.I32)), ("bn", T.I32)), Tuple(Array(S.BS, T.I32), Tuple(*[T.BIT] * S.BS)))


def c128a_past_slot_sig(T):
    return lambda S: ((("x", T.I32),), Tuple(T.I32, T.BIT))


def c128a_len_sig(T):
    return lambda S: ((("live", Array(S.W, T.LIVE)), ("slot", T.I64)), T.I32)


def c128a_sig(T):
    return lambda S: ((("position", T.I64), ("slot", T.I64), ("blocks", Array(S.NB, T.I32))), Tuple(Array(S.W, T.I32), T.I32))


def c128a_slot(o, S, xs, bn) -> tuple[list, list]:
    """The slots and lives of one block's lanes (models/deepseek_v4/sparse_mla.py:396-405), given each lane's word x of
    `Dsv4ShortContextTopK_v1` (its lane j where live, else -1) and the block's number bn: x >= 0 ? bn * BS + x % BS : -1 in int32,
    and x >= 0, which the length counts.  The block's product bn * BS is computed once for its lanes."""
    base = o.mul(bn, S.BS)
    lives = [o.ile(o.word(0), x) for x in xs]
    return [o.isel(live, o.iadd(base, o.rem(x, S.BS)), o.word(M32)) for x, live in zip(xs, lives)], lives


@composite("Dsv4C128aSlot", 1, ["BS"], c128a_slot_sig(WORDS), conformance=_CONF)
def Dsv4C128aSlot(B, S, x, bn):
    """One block's BS lanes of `_build_c128a_topk_metadata_kernel`'s decode row (models/deepseek_v4/sparse_mla.py:396-405), lane
    j = b * BS + i given its word x[i] and the block's number bn = blocks[b]: `c128a_slot`."""
    check_c128a_slot_statics(S.BS)
    rows, lives = c128a_slot(word_ops(B), S, [x[i] for i in range(S.BS)], bn)
    return tuple_of(array_of(rows), tuple_of(*lives))


@composite("Dsv4C128aPastSlot", 1, ["BS"], c128a_past_slot_sig(WORDS), conformance=_CONF)
def Dsv4C128aPastSlot(B, S, x):
    """A lane past the block table's row (j / BS >= NB), its block number read as -1, `GatherI32x{NB}_v1`'s fill: `c128a_slot` of
    the one lane over bn = -1, x >= 0 ? x % BS - BS : -1 in int32, and x >= 0."""
    check_c128a_slot_statics(S.BS)
    o = word_ops(B)
    (row,), (live,) = c128a_slot(o, S, [x], o.word(FILL))
    return tuple_of(row, live)


@composite("Dsv4C128aLen", 1, ["W"], c128a_len_sig(WORDS), conformance=_CONF)
def Dsv4C128aLen(B, S, live, slot):
    """The decode row's length (models/deepseek_v4/sparse_mla.py:390, 411-416): the count of j with live[j], summed in j's order,
    where the token's int64 slot is >= 0 (`setp` on its sign), else 0."""
    check_c128a_len_statics(S.W)
    o = word_ops(B)
    zero, one = o.word(0), o.word(1)
    count = zero
    for j in range(S.W):
        count = o.iadd(count, o.isel(live[j], one, zero))
    return o.isel(o.bnot(B.call(bit_at(64), slot, o.word(63))), count, zero)


def c128a_row(B, S, T, local, slot, past_slot, length, position, slot_mapping, blocks) -> Coll:
    """`Dsv4C128aGlobalTopk`'s body over its pieces' Definitions (of element types ``T``): the lanes' words ``local`` at
    {TOPK = W, RATIO = R}, a ``slot`` per block of BS lanes over blocks[b], a ``past_slot`` per lane past the row and the length of
    their lives."""
    check_c128a_statics(S.W, S.R, S.BS, S.NB)
    _, ret = c128a_sig(T)(S)
    xs = B.call(bind(local, TOPK=S.W, RATIO=S.R), position)
    nb = min(S.W // S.BS, S.NB)
    inside = Coll(Array(nb, Array(S.BS, T.I32)), concat([xs[j].refs for j in range(nb * S.BS)]))
    bn = Coll(Array(nb, T.I32), concat([blocks[b].refs for b in range(nb)]))
    outs = list(B.batch(bind(slot, BS=S.BS), inside, bn, axes=(0, 0)))
    if S.W > nb * S.BS:
        rest = Coll(Array(S.W - nb * S.BS, T.I32), concat([xs[j].refs for j in range(nb * S.BS, S.W)]))
        outs += list(B.batch(bind(past_slot, BS=S.BS), rest, axes=(0,)))
    n = B.call(bind(length, W=S.W), Coll(Array(S.W, T.LIVE), concat([s[1].refs for s in outs])), slot_mapping)
    return Coll(ret, concat([s[0].refs for s in outs] + [n.refs]))


@composite("Dsv4C128aGlobalTopk", 1, ["W", "R", "BS", "NB"], c128a_sig(WORDS), conformance=_CONF)
def Dsv4C128aGlobalTopk(B, S, position, slot, blocks):
    """One decode token of `_build_c128a_topk_metadata_kernel` (models/deepseek_v4/sparse_mla.py:361-416): its row of W slot ids of
    the paged compressed cache, `Dsv4C128aSlot_v1{BS}` of each block's BS lanes' `Dsv4ShortContextTopK_v1{W, R}` words over
    blocks[b] (`Dsv4C128aPastSlot_v1{BS}` past the row), and its length `Dsv4C128aLen_v1{W}` (the module docstring)."""
    return c128a_row(B, S, WORDS, IX.Dsv4ShortContextTopK, Dsv4C128aSlot, Dsv4C128aPastSlot, Dsv4C128aLen, position, slot, blocks)


DSV4_DECODE_METADATA_COMPOSITES = [Dsv4SwaWindow, Dsv4SwaLanes, Dsv4SwaIndices, Dsv4C128aSlot, Dsv4C128aPastSlot, Dsv4C128aLen,
                                   Dsv4C128aGlobalTopk]

dsv4_decode_metadata_twin.register_twin({d.id: d for d in DSV4_DECODE_METADATA_COMPOSITES})
