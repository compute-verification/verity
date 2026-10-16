"""[DSV4] DeepSeek-V4-Pro's ratio-4 (CSA) decode index row as an H200 (sm_90, TP8) runs it: vLLM d9105ea80, Triton 3.7.1.  Line
numbers are at that commit under `vllm/models/deepseek_v4/`; "the PTX" is Triton 3.7.1's sm_90a output at the served constexprs
(top-k 1024, block size 64, TRITON_BLOCK_SIZE 1024, four warps).

A CSA layer's decode attention (nvidia/flashmla.py:174-264) reads the compressed cache at the indexer's selection through the
block table: `compute_global_topk_indices_and_lens` (common/ops/cache_utils.py:435-466), a Triton kernel (469-517) launched once
per decode token, maps the token's row of local top-k indices (`topk_indices_buffer[:MD]`, the indexer's selection) to slot ids of
the paged compressed cache and counts them (flashmla.py:193-203), and `flash_mla_with_kvcache` reads both as
`extra_indices_in_kvcache` and `extra_topk_length` (248-264).  Its operands per token: the row, the request's row of the block
table (`attn_metadata.block_table[:num_decodes][req]`, req = `token_to_req_indices[token]`), the compressed block size
BS = `attn_metadata.block_size // 4` (64 at the kernel block size 256, sparse_mla.py:54, 213) and the token's `is_valid_token`
byte (torch.bool, `slot_mapping >= 0`, v1/attention/backends/mla/sparse_swa.py:555-556: a padding token's is 0).  The kernel
reads no position.

The kernel, per token (cache_utils.py line numbers; its int32 `//`, `%` and `*` are C's, here `I32Div`, `I32Rem`, `I32Mul`):

    x        = topk[j] for j < TOPK (the load's other -1 past it is never stored)                      489-495
    valid    = x >= 0                                                                                    496
    b        = x // BS;  bn = blocks[b], loaded only where valid                                         498-502
    slot[j]  = valid ? bn * BS + x % BS : -1                                                             503-506
    len      = is_valid_token ? (the count of valid j) : 0                                               513-517

The PTX does `//` and `%` by the constant BS as shifts (shr.s32 / shr.u32 / add.s32 / shr.s32, and.b32 / sub.s32: truncating, so
C's on every x), the product as `shl.b32 bn, k` and the count as `popc.b32`, `redux.sync.add.s32` and a shared-memory sum; the
block's address is `mad.wide.s32` of b over the row's base, so it does not wrap; `is_valid_token` is `ld.global.b8`, true iff the
byte is nonzero.  BS is a power of two, so on a valid x, b = x >> k and x % BS = x & (BS - 1) (k = log2 BS), and only bn's low
32 - k bits reach the slot.

The row's NB words are the block table's row stride (`block_table.stride(0)`, the cache group's `max_num_blocks_per_req`:
cdiv(max_model_len, 256), 4096 at the model's 1,048,576 and 16 at 4096).  A served x is below the request's compressed rows,
(position + 1) // 4 <= max_model_len / 4, so b < NB: the kernel reads its own row.  Past it (b >= NB) the kernel reads the next
request's row; the Definition's lookup selects -1 there instead (`GatherI32x{NB}_v1`'s fill, a total extension, and the slot then
negative), the twin declines such a row and the capture holds every served b below NB.

Registered functions (`_v1`):
  GatherI32x{NB}                    (b: i32, blocks: NB i32) -> blocks[b] for 0 <= b < NB, else -1 (`fp8_moe_experts`' gather)
  Dsv4GlobalTopkSlot{BS,NB}         (x: i32, blocks: NB i32) -> (i32, live: bit): one slot of the row, and x >= 0
  Dsv4GlobalTopkLen{TOPK}           (live: TOPK bits, valid: u8) -> i32: the row's length
  Dsv4GlobalTopk{TOPK,BS,NB}        (topk, blocks, valid) -> (TOPK i32, len: i32): one token's row and length
`boolean_dsv4_decode_indices` has their Boolean `_v2`s; `kernels.dsv4_decode_indices_twin` is the row's numpy twin.
"""
from __future__ import annotations

from types import SimpleNamespace

from verity.ml.circuits.codec import register_lazy_family
from verity.ml.circuits.defs import bind, composite
from verity.ml.circuits.ops.prims import const
from verity.ml.circuits.ops.scalar import BIT, BitOr, I32Add, I32Le, SelectI32
from verity.ml.circuits.refs import Coll, concat, tuple_of
from verity.ml.circuits.types import Array, Tuple, Value
from verity_vllm.program.kernels import dsv4_decode_indices_twin
from verity_vllm.program.registry import fp8_moe_experts as FX
from verity_vllm.program.registry import marlin_moe as MM
from verity_vllm.program.registry.prims import I32
from verity_vllm.program.registry.sampling import bit_at

_CONF = ("model-untested: read from the Triton kernel and Triton 3.7.1's PTX at the commits the module docstring pins; "
         "tests/properties/dsv4_decode_index_row_capture_gpu.py is its hardware check")

M32 = 0xFFFFFFFF
#: the block number a lookup past the row selects (`GatherI32x{NB}_v1`)
FILL = M32
U8 = Value(8)

#: the served statics: the indexer's top-k of 1024, 64 compressed rows per block, the block table's row at max_model_len 1,048,576
#: and at the inventory's 4096
DSV4_DECODE_INDICES = {"c4": {"TOPK": 1024, "BS": 64, "NB": 4096}, "c4-4096": {"TOPK": 1024, "BS": 64, "NB": 16}}

#: the element types of a signature: these words, and their bits in `boolean_dsv4_decode_indices`
WORDS = SimpleNamespace(I32=I32, U8=U8, BIT=BIT, LIVE=BIT)


def gather_i32(nb: int):
    """`GatherI32x{NB}_v1`: blocks[b] of NB i32 words for 0 <= b < NB, else -1."""
    return FX._gather("GatherI32", nb, I32, FILL)


register_lazy_family(r"^GatherI32x(\d+)_v1$", lambda m: gather_i32(int(m[1])))   # decodable where the program was not built


def check_slot_statics(BS: int, NB: int) -> int:
    """log2 BS, for a power-of-two BS < 2^31 and 1 <= NB <= 2^(31 - log2 BS) (every block a valid x can name)."""
    k = BS.bit_length() - 1
    if not (BS >= 1 and BS & (BS - 1) == 0 and k < 31 and 1 <= NB <= 1 << (31 - k)):
        raise ValueError(f"Dsv4GlobalTopkSlot: BS a power of two below 2^31, 1 <= NB <= 2^(31 - log2 BS) (BS={BS}, NB={NB})")
    return k


def check_len_statics(TOPK: int) -> None:
    if not 1 <= TOPK < 1 << 31:
        raise ValueError(f"Dsv4GlobalTopkLen: 1 <= TOPK < 2^31 (TOPK={TOPK})")


def slot_sig(T):
    return lambda S: ((("x", T.I32), ("blocks", Array(S.NB, T.I32))), Tuple(T.I32, T.BIT))


def len_sig(T):
    return lambda S: ((("live", Array(S.TOPK, T.LIVE)), ("valid", T.U8)), T.I32)


def row_sig(T):
    return lambda S: ((("topk", Array(S.TOPK, T.I32)), ("blocks", Array(S.NB, T.I32)), ("valid", T.U8)),
                      Tuple(Array(S.TOPK, T.I32), T.I32))


@composite("Dsv4GlobalTopkSlot", 1, ["BS", "NB"], slot_sig(WORDS), conformance=_CONF)
def Dsv4GlobalTopkSlot(B, S, x, blocks):
    """One slot of `_compute_global_topk_indices_and_lens_kernel` (common/ops/cache_utils.py:496-506): its validity
    live = x >= 0 and live ? blocks[x / BS] * BS + x % BS : -1, C's int32 / % * + (`I32Div`, `I32Rem`, `I32Mul`, `I32Add`), the
    lookup `GatherI32x{NB}_v1`; (the slot, live), the length's count reading the same live (496, 513)."""
    check_slot_statics(S.BS, S.NB)
    bs = B.call(const(32, S.BS))
    bn = B.call(gather_i32(S.NB), B.call(MM.I32Div, x, bs), blocks)
    slot = B.call(I32Add, B.call(MM.I32Mul, bn, bs), B.call(MM.I32Rem, x, bs))
    live = B.call(I32Le, B.call(const(32, 0)), x)
    return tuple_of(B.call(SelectI32, live, slot, B.call(const(32, M32))), live)


@composite("Dsv4GlobalTopkLen", 1, ["TOPK"], len_sig(WORDS), conformance=_CONF)
def Dsv4GlobalTopkLen(B, S, live, valid):
    """The kernel's length (common/ops/cache_utils.py:496, 513-517): the count of j with live[j] (topk[j] >= 0, its slot's), summed
    in j's order, when the token's byte `valid` is nonzero (`ld.global.b8`), else 0."""
    check_len_statics(S.TOPK)
    zero, one = B.call(const(32, 0)), B.call(const(32, 1))
    inc = B.batch(SelectI32, live, one, zero, axes=(0, None, None))
    count = zero
    for j in range(S.TOPK):
        count = B.call(I32Add, count, inc[j])
    nz = B.call(bit_at(8), valid, zero)
    for i in range(1, 8):
        nz = B.call(BitOr, nz, B.call(bit_at(8), valid, B.call(const(32, i))))
    return B.call(SelectI32, nz, count, zero)


def row(B, S, T, slot, length, topk, blocks, valid) -> Coll:
    """`Dsv4GlobalTopk`'s body over its pieces' Definitions ``slot`` and ``length`` (of element types ``T``): TOPK slots, one Call
    each, and the length of their lives."""
    _, ret = row_sig(T)(S)
    slots = B.batch(bind(slot, BS=S.BS, NB=S.NB), topk, blocks, axes=(0, None))
    live = Coll(Array(S.TOPK, T.LIVE), concat([s[1].refs for s in slots]))
    length = B.call(bind(length, TOPK=S.TOPK), live, valid)
    return Coll(ret, concat([s[0].refs for s in slots] + [length.refs]))


@composite("Dsv4GlobalTopk", 1, ["TOPK", "BS", "NB"], row_sig(WORDS), conformance=_CONF)
def Dsv4GlobalTopk(B, S, topk, blocks, valid):
    """One decode token of `compute_global_topk_indices_and_lens` (common/ops/cache_utils.py:435-517): its row of TOPK slot ids,
    `Dsv4GlobalTopkSlot_v1{BS,NB}` of each top-k index over the request's block-table row, and its length
    `Dsv4GlobalTopkLen_v1{TOPK}` (the module docstring)."""
    return row(B, S, WORDS, Dsv4GlobalTopkSlot, Dsv4GlobalTopkLen, topk, blocks, valid)


DSV4_DECODE_INDICES_COMPOSITES = [Dsv4GlobalTopkSlot, Dsv4GlobalTopkLen, Dsv4GlobalTopk]

dsv4_decode_indices_twin.register_twin({d.id: d for d in DSV4_DECODE_INDICES_COMPOSITES})
