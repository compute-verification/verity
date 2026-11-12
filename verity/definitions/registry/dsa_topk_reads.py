"""DSA's long row by table reads: the TOPK history rows a row's counted mask selects, each read at its position from a table of
the request's history for the layer, so that no unit routes the row's every history row (`DsaButterflyRows_v1`'s form).

    DsaTopKReadRows_v1{N,TOPK,BLOCK,READ}
        (scores: Array<N, F32>, n: I32, root, domain, row: Array<TOPK, Row>, salt: Array<TOPK, Salt>, path: Array<TOPK, Path>)
            -> Tuple<Array<TOPK, Row>, Array<TOPK, Value<1>>>                                                       (kv, ok)

READ is the read, `commitments.gates.embedding`'s `EmbeddingRead_v1{V, H}` where served (V the table's capacity, at least N,
and Row its H u16 words): a static, as `boolean_fp8_moe`'s is, so this module imports no commitment code.  Root, domain, Salt
and Path are READ's; `READ_PORTS` its parameters.  The body is three stages, each a unit or several (`cut`, `CUTS`):

* the mask, `DsaTopKMaskCounted_v2{N,TOPK}` (scores, n), cut past `dsa_topk_mask_cut.WHOLE_MAX` keys and up to it a unit,
  `DsaTopKMaskWhole_v1{N,TOPK}`;
* the index, over a block a mask unit (`index_block`): the row's blocks of BLOCK keys (`dsa_topk_mask_cut.blocks`) where the
  mask is cut, and the whole row where it is one unit.  Per block,
  ``DsaMaskPositions_v1{N,AT,K}`` compacts its members (its mask in u16 words, `dsa_topk_mask_cut.words`) to their positions,
  ascending, and counts them; then
  ``DsaPositionsMerge_v1{M,K}`` writes the M blocks' lists one after another, each at the count of those before it, and 0 past
  them.  So slot j holds the row's j-th member by position, the rows network's order (`dsa_topk_mask.rows`), whenever the mask
  has at most TOPK members, as a counted mask always has.  Each block's list and count is a port of the merge's own
  (``pos_b``, ``count_b``), one whole output of one writer.  Its slots are one output, Array<K, I32>: K outputs of one I32
  each would be K committed rows, which put the served merge's block (8 blocks, 2,048 slots) at 2^31 bits, past Flock's
  proved range.  So a read takes its position in two steps, each index an input so that one template serves every
  instance: ``DsaPositionsChunk_v1{K,W}`` (slots, c) takes chunk c, the merge's slots cW .. cW+W-1 (W = `CHUNK`), and
  ``DsaPositionsSlot_v1{K=W,N}`` (slots, j) slot j mod W of chunk j // W, 0 where its j >= W or the slot is N or more, so a
  read's key is below N whatever its inputs;
* TOPK reads, read j at slot j's position with opening (row[j], salt[j], path[j]) under the table's root and domain id: kv[j]
  and its verdict ok[j].  The cut stages each chunk, slot and read as its own unit, each reading the whole output of the one
  before it; the body reads the merge's slots directly, the same function, as every slot is below N.

So where every ok is 1, kv[j] is the row the table holds at the mask's j-th member: today's long row's kv when the table holds
the history (`DsaButterflyRowsPad_v1` over the counted mask's settings), which the definitions' tests hold it to.  `values` is
the index units' twin.

`DsaPositionsMerge_v1` is one traced circuit, so its description is every gate of it.  ``DsaPositionsMerge_v2{M,K}`` is the
same function written as Calls, with `_v1`'s signature and its ANDs: a block's kept list is a batch of `DsaPositionsKeep_v1`
over its slots, on the bits of `DsaRowValid_v2{K}`; each stage of its shift is a batch of `DsaPositionsShift_v1` over the
slots that have one the stage's distance below and one of `DsaPositionsShiftIn_v1` over the rest; the offsets are
`DsaPositionsFirstOffset_v1{K}` and `DsaPositionsOffset_v1{K}`; and the lists are XORed a gate a bit.  Its description is those
rows and the pieces: with K it grows by a row for each bit of K (a stage of the shift, a level of `DsaRowValid_v2{K}`'s trie).

`DsaMaskPositions_v1` is Calls, but its tags (`DsaButterflyTagBits_v1`) and its count (`DsaTopKPopCount_v1`) are traced and
its positions are a constant a bit.  ``DsaMaskPositions_v2{N,AT,K}`` is the same function with `_v1`'s signature: the tags
and the count are one tree of counts written as Calls (`boolean_dsa_topk_mask.DsaButterflyTagsCount_v1`), and the positions
AT .. AT+N-1 are runs of 2^L that start at a multiple of 2^L, each `DsaPositionWord_v1{L,HI}` over the indices of
`DsaIndexBits_v1{L}`, which are those of L - 1 bits twice with a bit after each (`DsaBitAppend_v1`).  The cut names the
`_v1` units; a `_v2` serves only a caller that binds it.
"""
from __future__ import annotations

from collections import Counter
from types import SimpleNamespace
from typing import Any, Sequence

from verity.circuits import boolean as BL
from verity.circuits.boolean.forms import ZERO, add, const_bits, mux, sub
from verity.circuits.defs import CompositeDefinition, bind
from verity.definitions.ops.boolean import elementwise as EW
from verity.circuits.refs import Coll, concat, strided_view, tuple_of
from verity.circuits.types import Array, Tuple
from verity.definitions.registry import boolean_dsa_topk_mask as BM
from verity.definitions.registry import dsa_topk_mask as M
from verity.definitions.registry import dsa_topk_mask_cut as X
from verity.definitions.registry.boolean_dsa import own_gates
from verity.definitions.registry.dsa_topk import check_statics

BIT = BL.BIT
F32 = I32 = Array(32, BIT)
BF16 = Array(16, BIT)
BLOCK = X.BLOCK
#: the slots of a chunk (`DsaPositionsChunk_v1`'s W), which a slot unit of the cut selects among: 64 chunks of 2,048 slots.
#: Flock proves a unit only if one of its rows is past `interface.LAST_BLOCK_BYTES` (111) bytes, and the slot unit's
#: longest row is its chunk, 4W bytes: 32 is the least power of two past it
CHUNK = 32
#: a read's ports: its hidden key (the position), the table's root and domain id, and the opening (the row, its salt, its path)
READ_PORTS = ("tok", "root", "domain", "row", "salt", "path")


# --- the index ---------------------------------------------------------------------------------------------------------------

def _positions(B, S, mask):
    """`DsaMaskPositions_v1{N,AT,K}`: `DsaButterflyRows_v2{N,K,1,32}` at the block's `DsaButterflySettings_v2{N,K}` over the
    constant positions AT..AT+N-1, and `DsaTopKPopCount_v1{N}` of the mask (its words' first N bits)."""
    mask = X.low(mask, S.N)
    sw = B.call(bind(BM.DsaButterflySettingsV2, N=S.N, K=S.K), mask)
    zero, one = B.call(BL.constant(0)), B.call(BL.constant(1))
    at = concat([(one if (S.AT + j) >> i & 1 else zero).refs for j in range(S.N) for i in range(32)])
    pos = B.call(bind(BM.DsaButterflyRowsV2, N=S.N, K=S.K, W=1, E=32), sw, Coll(Array(S.N, Array(1, I32)), at))
    return tuple_of(Coll(Array(S.K, I32), pos.refs), B.call(bind(X.DsaTopKPopCount, N=S.N), mask))


def _refuse_positions(b: dict) -> str | None:
    if b.get("AT") is not None and not 0 <= b["AT"] < 1 << 31:
        return f"a block's first key is a u31 (AT={b['AT']})"
    if b.get("N") is not None and b.get("AT") is not None and b["AT"] + b["N"] > 1 << 31:
        return f"a block's keys are u31 positions (AT={b['AT']}, N={b['N']})"
    if b.get("N") is not None and b.get("K") is not None:
        try:
            M.check_net(b["N"], b["K"])
        except ValueError as e:
            return str(e)
    return None


DsaMaskPositions = CompositeDefinition(
    "DsaMaskPositions", 1, ("N", "AT", "K"),
    lambda S: ((("mask", X.words(S.N)),), Tuple(Array(S.K, I32), I32)), _positions,
    conformance="exact-model-tested: the rows network's order (DsaTopKRow_v2's construction over a block's positions), held to "
                "dsa_topk_mask.butterfly_apply by the definitions' test_dsa_topk_reads.py",
    refuse=_refuse_positions,
    doc="(pos, count) of a block of N keys AT..AT+N-1, its mask in u16 words (`dsa_topk_mask_cut.words`): count its mask's "
        "members (u32), pos[s] the s-th of them by position for s < count, ascending, and what the rows network leaves past "
        "count")


def _bit_append(S):
    """`DsaBitAppend_v1{W}`: a row of W bits and one more after them (no gate)."""
    return lambda C, x, b: [[*x, b[0]]]


def _position_word(S):
    """`DsaPositionWord_v1{L,HI}`: the position HI 2^L + i (u32) of an index i of L bits (no gate)."""
    return lambda C, i: [[*i, *const_bits(S.HI, 32 - S.L)]]


def _index_bits(B, S, zero, one):
    """`DsaIndexBits_v1{L}`: the indices 0 .. 2^L - 1, L bits each LSB first, from the two constant bits: those of L - 1
    bits twice, a 0 after each and then a 1 (`DsaBitAppend_v1`, a batch each)."""
    if S.L == 1:
        return Coll(Array(2, Array(1, BIT)), concat([zero.refs, one.refs]))
    low = B.call(bind(DsaIndexBits, L=S.L - 1), zero, one)
    return Coll(Array(1 << S.L, Array(S.L, BIT)),
                concat([B.batch(bind(DsaBitAppend, W=S.L - 1), low, b, axes=(0, None)).refs for b in (zero, one)]))


def _positions_of(B, N: int, AT: int) -> Coll:
    """The positions AT .. AT+N-1 (u32 each) from the constant bits: the block in runs of 2^L positions that start at a
    multiple of 2^L, each a batch of `DsaPositionWord_v1` over `DsaIndexBits_v1{L}`, and a run of one position its bits."""
    zero, one = B.call(BL.constant(0)), B.call(BL.constant(1))
    parts, a, end = [], AT, AT + N
    while a < end:
        L = min((a & -a).bit_length() - 1 if a else 31, (end - a).bit_length() - 1)
        if L:
            parts.append(B.batch(bind(DsaPositionWord, L=L, HI=a >> L), B.call(bind(DsaIndexBits, L=L), zero, one), axes=(0,)).refs)
        else:
            parts.append(concat([(one if a >> i & 1 else zero).refs for i in range(32)]))
        a += 1 << L
    return Coll(Array(N, Array(1, I32)), concat(parts))


def _positions_calls(B, S, mask):
    """`DsaMaskPositions_v2{N,AT,K}`: `DsaMaskPositions_v1` with each traced part written as Calls: the tags of
    `DsaButterflySettings_v2{N,K}` and the count are one tree of counts (`DsaButterflyTagsCount_v1{N,K}`), the switches are
    set from those tags as that Definition sets them (`dsa_topk_mask.settings` of `DsaButterflyCtl_v1`), and the positions
    are `_positions_of`'s runs and not a constant a bit."""
    mask, kb = X.low(mask, S.N), M.rank_bits(S.K)
    tags = B.call(bind(BM.DsaButterflyTagsCount, N=S.N, K=S.K), mask)
    n = S.N * (1 + kb)
    sw = M.settings(B, S.N, S.K, BM._tag(S.K), Coll(Array(S.N, BM._tag(S.K)), tags.refs.slice(0, n)), B.call(BL.constant(0)),
                    lambda t, two: bind(BM.DsaButterflyCtl, T=t, KB=kb, TWO=int(two)))
    pos = B.call(bind(BM.DsaButterflyRowsV2, N=S.N, K=S.K, W=1, E=32), sw, _positions_of(B, S.N, S.AT))
    return tuple_of(Coll(Array(S.K, I32), pos.refs), Coll(I32, tags.refs.slice(n, n + 32)))


def _clamp(C, x, K: int) -> list:
    """min(x, K) in K.bit_length() bits, x an unsigned word."""
    RB = K.bit_length()
    below = sub(C, x, const_bits(K, len(x)), len(x))[1]
    return mux(C, below, (list(x) + [ZERO] * RB)[:RB], const_bits(K, RB))


def _merge(S):
    """`DsaPositionsMerge_v1{M,K}` (pos_0..pos_{M-1}, count_0..count_{M-1}) -> Array<K, I32>: block b's list pos_b,
    its slots s < min(count_b, K) kept (`DsaRowValid_v1{K}`'s trie over count_b) and the rest 0, shifted up by
    off_b = min(sum of min(count_c, K) over c < b, K) (a barrel shifter over off_b's bits), the blocks' lists XORed, which
    places each where no other is."""
    RB = S.K.bit_length()
    kept = BM._row_valid(SimpleNamespace(N=S.K))

    def clamp(C, x):
        return _clamp(C, x, S.K)

    def build(C, *els):
        pos, counts = els[:S.M * S.K], els[S.M * S.K:]
        out, off = None, [ZERO] * RB
        for b in range(S.M):
            keep = kept(C, counts[b])[0]
            x = [[C.AND(keep[s], w) for w in pos[b * S.K + s]] for s in range(S.K)]
            if out is None:
                out = x
            else:
                for i in range(RB):
                    d = 1 << i
                    x = [mux(C, off[i], x[s - d] if s >= d else [ZERO] * 32, x[s]) for s in range(S.K)]
                out = [[a ^ y for a, y in zip(o, w)] for o, w in zip(out, x)]
            if b + 1 < S.M:
                off = clamp(C, add(C, off, clamp(C, counts[b]), width=RB + 1))
        return out
    return build


def _refuse_merge(b: dict) -> str | None:
    if b.get("M") is not None and b["M"] < 1:
        return f"a merge of M >= 1 lists (M={b['M']})"
    if b.get("K") is not None and not 1 <= b["K"] <= 1 << M.RANK_BITS:
        return f"lists of 1 <= K <= 2^{M.RANK_BITS} positions (K={b['K']})"
    return None


def _merge_sig(S) -> tuple[tuple[tuple[str, Any], ...], Any]:
    return ((*((f"pos_{b}", Array(S.K, I32)) for b in range(S.M)), *((f"count_{b}", I32) for b in range(S.M))),
            Array(S.K, I32))


DsaPositionsMerge = EW.specialized(
    "DsaPositionsMerge", 1, ["M", "K"], _merge, _merge_sig, doc=_merge.__doc__, refuse=_refuse_merge,
    conformance="exact-model-tested: held to merge, the blocks' lists one after another, by the definitions' "
                "test_dsa_topk_reads.py")


# --- the merge as Calls ------------------------------------------------------------------------------------------------------

def _piece(name: str, statics: Sequence[str], build, signature, doc: str) -> CompositeDefinition:
    """``name_v1{statics}``, a piece of the merge written as Calls: ``build`` traced per binding (`EW.specialized`); Flock lowers
    an activation by its own gates (`boolean_dsa.own_gates`)."""
    d = EW.specialized(name, 1, statics, build, signature, doc=doc)
    d.build = own_gates(d)  # type: ignore[attr-defined]
    return d


def _offsets(S) -> Array:
    return Array(S.K.bit_length(), BIT)


def _next_offset(C, off, count, K: int) -> list:
    """min(off + min(count, K), K), the merge's offset past one more block."""
    return [_clamp(C, add(C, off, _clamp(C, count, K), width=K.bit_length() + 1), K)]


DsaPositionsKeep = _piece(
    "DsaPositionsKeep", (), lambda S: lambda C, k, w: [[C.AND(k[0], b) for b in w]], lambda S: ((("k", BIT), ("w", I32)), I32),
    "a slot of a block's list where its keep bit k is 1, else 0: one bit of DsaRowValid_v2{K} over the block's count")
DsaPositionsShift = _piece(
    "DsaPositionsShift", (), lambda S: lambda C, s, a, w: [mux(C, s[0], a, w)],
    lambda S: ((("s", BIT), ("a", I32), ("w", I32)), I32),
    "a slot past one stage of the merge's barrel shifter: a, the slot the stage's distance below, where the offset's bit s is "
    "1, else w, the slot itself")
DsaPositionsShiftIn = _piece(
    "DsaPositionsShiftIn", (), lambda S: lambda C, s, w: [mux(C, s[0], [ZERO] * 32, w)], lambda S: ((("s", BIT), ("w", I32)), I32),
    "DsaPositionsShift_v1 at a slot with none the stage's distance below: 0 where the offset's bit s is 1, else w")
DsaPositionsFirstOffset = _piece(
    "DsaPositionsFirstOffset", ("K",), lambda S: lambda C, count: _next_offset(C, [ZERO] * S.K.bit_length(), count, S.K),
    lambda S: ((("count", I32),), _offsets(S)), "the merge's offset past its first block: min(count, K)")
DsaPositionsOffset = _piece(
    "DsaPositionsOffset", ("K",), lambda S: lambda C, off, count: _next_offset(C, off, count, S.K),
    lambda S: ((("off", _offsets(S)), ("count", I32)), _offsets(S)),
    "the merge's offset past one more block: min(off + min(count, K), K)")


DsaBitAppend = _piece("DsaBitAppend", ["W"], _bit_append, lambda S: ((("x", Array(S.W, BIT)), ("b", BIT)), Array(S.W + 1, BIT)),
                      _bit_append.__doc__)
DsaPositionWord = _piece("DsaPositionWord", ["L", "HI"], _position_word, lambda S: ((("i", Array(S.L, BIT)),), I32),
                         _position_word.__doc__)
DsaIndexBits = CompositeDefinition("DsaIndexBits", 1, ("L",),
                                   lambda S: ((("zero", BIT), ("one", BIT)), Array(1 << S.L, Array(S.L, BIT))), _index_bits,
                                   doc=_index_bits.__doc__, conformance="protocol-defined")
DsaMaskPositionsV2 = CompositeDefinition(
    "DsaMaskPositions", 2, ("N", "AT", "K"),
    lambda S: ((("mask", X.words(S.N)),), Tuple(Array(S.K, I32), I32)), _positions_calls,
    conformance="exact-model-tested: held to DsaMaskPositions_v1 and to the index units' twin by "
                "adapters/vllm/tests/program/test_dsa_long_row_calls.py",
    refuse=_refuse_positions, doc=_positions_calls.__doc__)


def _merge_calls(B, S, *ports):
    """`DsaPositionsMerge_v1{M,K}`'s function as Calls, a row of them a step of `_merge`: a block's keep bits are
    `DsaRowValid_v2{K}` of its count and its kept list a batch of `DsaPositionsKeep_v1` over the K slots; stage i of its
    shift is a batch of `DsaPositionsShift_v1` over the slots 2^i and up, each on the slot 2^i below it, and one of
    `DsaPositionsShiftIn_v1` over the slots below 2^i, both on bit i of the offset; the lists are XORed a bit a gate; and the
    offset past each block is `DsaPositionsFirstOffset_v1{K}` and then `DsaPositionsOffset_v1{K}`."""
    pos, counts = ports[:S.M], ports[S.M:]
    K = S.K

    def words(refs, n: int = K) -> Coll:
        return Coll(Array(n, I32), refs)
    out = off = None
    for b in range(S.M):
        x = B.batch(DsaPositionsKeep, B.call(bind(BM.DsaRowValidV2, N=K), counts[b]), pos[b], axes=(0, 0))
        if out is None:
            out = x
        else:
            for i in range(K.bit_length()):
                d, s = min(1 << i, K), Coll(BIT, off.refs.slice(i, i + 1))
                parts = [B.batch(DsaPositionsShiftIn, s, words(x.refs.slice(0, 32 * d), d), axes=(None, 0)).refs]
                if d < K:
                    parts.append(B.batch(DsaPositionsShift, s, words(x.refs.slice(0, 32 * (K - d)), K - d),
                                         words(x.refs.slice(32 * d, 32 * K), K - d), axes=(None, 0, 0)).refs)
                x = words(concat(parts))
            out = words(B.batch(BL.XOR, Coll(Array(2, Array(32 * K, BIT)), concat([out.refs, x.refs])), axes=(1,)).refs)
        if b + 1 < S.M:
            off = (B.call(bind(DsaPositionsFirstOffset, K=K), counts[b]) if off is None
                   else B.call(bind(DsaPositionsOffset, K=K), off, counts[b]))
    return out


DsaPositionsMergeV2 = CompositeDefinition(
    "DsaPositionsMerge", 2, ("M", "K"), _merge_sig, _merge_calls, refuse=_refuse_merge, doc=_merge.__doc__,
    conformance="exact-model-tested: held to merge and to DsaPositionsMerge_v1, whose ANDs it has count for count, by "
                "adapters/vllm/tests/program/test_dsa_long_row_calls.py")


def _select(C, rows: list[list], idx: list) -> list:
    """rows[idx] for idx < len(rows), and the zero row for idx below the next power of two: a tree of muxes over idx's low
    bits.  Constant idx bits make every mux wiring, so an index bound as a static selects at no AND."""
    J = (len(rows) - 1).bit_length()
    rows = rows + [[ZERO] * len(rows[0])] * ((1 << J) - len(rows))
    for i in range(J):
        rows = [mux(C, idx[i], rows[t + 1], rows[t]) for t in range(0, len(rows), 2)]
    return rows[0]


def _chunk(S):
    """`DsaPositionsChunk_v1{K,W}` (slots: Array<K, I32>, c: I32) -> Array<W, I32>: slots cW .. cW+W-1, 0 past K, and every
    slot 0 where c is ceil(K/W) or more; the chunk by `_select` over c's low bits."""
    n = -(-S.K // S.W)

    def build(C, *els):
        slots, c = [list(e) for e in els[:S.K]], list(els[S.K])
        slots += [[ZERO] * 32] * (n * S.W - S.K)
        x = _select(C, [[b for s in slots[k * S.W:(k + 1) * S.W] for b in s] for k in range(n)], c)
        ok = sub(C, c, const_bits(n, 32), 32)[1]
        return [[C.AND(ok, b) for b in x[w * 32:(w + 1) * 32]] for w in range(S.W)]
    return build


def _refuse_chunk(b: dict) -> str | None:
    if b.get("K") is not None and not 1 <= b["K"] <= 1 << M.RANK_BITS:
        return f"a list of 1 <= K <= 2^{M.RANK_BITS} positions (K={b['K']})"
    if b.get("W") is not None and not 1 <= b["W"] <= 1 << M.RANK_BITS:
        return f"chunks of 1 <= W <= 2^{M.RANK_BITS} slots (W={b['W']})"
    return None


DsaPositionsChunk = EW.specialized(
    "DsaPositionsChunk", 1, ["K", "W"], _chunk, lambda S: ((("slots", Array(S.K, I32)), ("c", I32)), Array(S.W, I32)),
    doc=_chunk.__doc__, refuse=_refuse_chunk,
    conformance="exact-model-tested: held to the merge's slots cW .. cW+W-1, and to 0 past K and at c >= ceil(K/W), by the "
                "definitions' test_dsa_topk_reads.py")


def _slot(S):
    """`DsaPositionsSlot_v1{K,N}` (slots: Array<K, I32>, j: I32) -> I32: slots[j] where j < K and slots[j] < N, else 0, so
    its output is below N whatever its inputs; slots[j] by `_select` over j's low bits.  Served over a chunk: K = W."""
    def build(C, *els):
        j = list(els[S.K])
        x = _select(C, [list(e) for e in els[:S.K]], j)
        ok = C.AND(sub(C, j, const_bits(S.K, 32), 32)[1], sub(C, x, const_bits(S.N, 32), 32)[1])
        return [C.AND(ok, b) for b in x]
    return build


def _refuse_slot(b: dict) -> str | None:
    if b.get("K") is not None and not 1 <= b["K"] <= 1 << M.RANK_BITS:
        return f"a list of 1 <= K <= 2^{M.RANK_BITS} positions (K={b['K']})"
    if b.get("N") is not None and not 1 <= b["N"] <= 1 << 31:
        return f"a bound of 1 <= N <= 2^31 (N={b['N']})"
    return None


DsaPositionsSlot = EW.specialized(
    "DsaPositionsSlot", 1, ["K", "N"], _slot, lambda S: ((("slots", Array(S.K, I32)), ("j", I32)), I32), doc=_slot.__doc__,
    refuse=_refuse_slot,
    conformance="exact-model-tested: held to the merge's slot j through its chunk, and to 0 at j >= K or a slot of N or "
                "more, by the definitions' test_dsa_topk_reads.py")


def index_block(N: int, TOPK: int, block: int) -> int:
    """The index's block at a row of N keys: ``block`` past `dsa_topk_mask_cut.WHOLE_MAX`, where the mask is cut a block a
    unit, and N up to it, where the mask is one unit, so that each `DsaMaskPositions_v1` reads one mask unit's whole output.
    The whole mask with a block an output instead would have 5 rows at 40,960 keys, so 8 hm96 slots for its 3 rows' 4, and
    be past Flock's 2^27 bits."""
    return block if N > X.WHOLE_MAX else N


def index(B, N: int, K: int, block: int, mask) -> Coll:
    """The K positions of a row's mask (Array<N, BIT>, a Coll of ``B``): a `DsaMaskPositions_v1` per block of ``block`` keys,
    on the block's mask in u16 words, then the merge, its ``pos_b`` and ``count_b`` block b's two outputs."""
    lists, counts = [], []
    for lo, hi in X.blocks(N, K, block):
        out = B.call(bind(DsaMaskPositions, N=hi - lo, AT=lo, K=K),
                     X.to_words(B, Coll(Array(hi - lo, BIT), mask.refs.slice(lo, hi)), hi - lo))
        lists.append(out[0])
        counts.append(out[1])
    return B.call(bind(DsaPositionsMerge, M=len(lists), K=K), *lists, *counts)


# --- the long row ------------------------------------------------------------------------------------------------------------

def _read(S) -> tuple[tuple, Any]:
    """READ's parameters and its row type, refused (ValueError) unless it is a read of u16 words at a 32-bit position of at
    least N rows: `READ_PORTS` in, (row, ok) out."""
    params, ret = S.READ.signature()
    fields = getattr(ret, "fields", ())
    row = fields[0] if len(fields) == 2 else None
    if (tuple(n for n, _ in params) != READ_PORTS or params[0][1] != I32 or fields[1:] != (BIT,)
            or not isinstance(row, Array) or row.elem != BF16):
        raise ValueError(f"READ = {S.READ.id} is no read of u16 rows: it takes {[n for n, _ in params]} and returns {ret!r}; "
                         f"bind EmbeddingRead_v1{{V, H}}")
    V = getattr(S.READ, "bindings", {}).get("V")
    if V is not None and V < S.N:
        raise ValueError(f"READ = {S.READ.id} reads V = {V} rows, fewer than the row's N = {S.N} keys: bind V >= N")
    return tuple(params), row


def _rows_sig(S):
    params, row = _read(S)
    root, domain, _, salt, path = (t for _, t in params[1:])
    return ((("scores", Array(S.N, F32)), ("n", I32), ("root", root), ("domain", domain), ("row", Array(S.TOPK, row)),
             ("salt", Array(S.TOPK, salt)), ("path", Array(S.TOPK, path))), Tuple(Array(S.TOPK, row), Array(S.TOPK, BIT)))


def _rows(B, S, scores, n, root, domain, row, salt, path):
    _, row_t = _read(S)
    mask = B.call(bind(BM.DsaTopKMaskCountedV2, N=S.N, TOPK=S.TOPK), scores, n)
    tok = index(B, S.N, S.TOPK, index_block(S.N, S.TOPK, S.BLOCK), mask)
    out = B.batch(S.READ, tok, root, domain, row, salt, path, axes=(0, None, None, 0, 0, 0))
    w = row_t.leaves
    return tuple_of(Coll(Array(S.TOPK, row_t), strided_view(out, 0, w, w + 1, 1, S.TOPK * w)),
                    Coll(Array(S.TOPK, BIT), strided_view(out, w, 1, w + 1, 1, S.TOPK)))


def _refuse_rows(b: dict) -> str | None:
    N, TOPK, block = b.get("N"), b.get("TOPK"), b.get("BLOCK")
    if N is not None and TOPK is not None:
        try:
            check_statics(N, TOPK)
        except ValueError as e:
            return str(e)
    if TOPK is not None and block is not None and block < TOPK:
        return f"a block of BLOCK >= TOPK keys (BLOCK={block}, TOPK={TOPK})"
    if (N is not None and TOPK is not None and block is not None
            and any(e - a < 2 for a, e in X.blocks(N, TOPK, index_block(N, TOPK, block)))):
        return f"blocks of at least 2 keys, as a block's positions take (N={N}, TOPK={TOPK}, BLOCK={block})"
    return None


DsaTopKReadRows = CompositeDefinition(
    "DsaTopKReadRows", 1, ("N", "TOPK", "BLOCK", "READ"), _rows_sig, _rows,
    conformance="model-untested: the counted mask's selection (DsaTopKMaskCounted_v1's tie rule) at the rows network's order, "
                "each row READ's; held to DsaButterflyRowsPad_v1 over the counted mask's settings and the history by the "
                "definitions' test_dsa_topk_reads.py",
    refuse=_refuse_rows,
    doc="(kv, ok): the TOPK rows READ opens at the positions of the counted mask's members (DsaTopKMaskCounted_v2{N,TOPK}), "
        "ascending, from the table of root and domain (DsaMaskPositions_v1 a block of index_block(N, TOPK, BLOCK) keys, "
        "DsaPositionsMerge_v1), ok[j] read j's verdict")

DSA_TOPK_READS = [DsaMaskPositions, DsaPositionsMerge, DsaPositionsChunk, DsaPositionsSlot, DsaTopKReadRows, DsaPositionsKeep,
                  DsaPositionsShift, DsaPositionsShiftIn, DsaPositionsFirstOffset, DsaPositionsOffset, DsaPositionsMergeV2,
                  DsaBitAppend, DsaPositionWord, DsaIndexBits, DsaMaskPositionsV2]


def cut(t) -> list[tuple[Any, int]]:
    """The units a binding of `DsaTopKReadRows_v1` is cut into, each Definition bound and how many of it: the mask's own (its
    `dsa_topk_mask_cut.cut`, else the mask whole, `DsaTopKMaskWhole_v1{N,TOPK}`), a `DsaMaskPositions_v1` a block of
    `index_block` keys, block b's reading the mask unit of block b, the merge, ceil(TOPK/CHUNK) of
    `DsaPositionsChunk_v1{K=TOPK,W=CHUNK}`, chunk c's c its input, TOPK of `DsaPositionsSlot_v1{K=CHUNK,N}`, slot j's
    slots chunk j // CHUNK's output and its j the input j mod CHUNK, and TOPK of READ, read j at slot j's output.  The body
    computes the same function with the reads over the merge's slots directly: a slot's output is the merge's slot j
    wherever that is below N, as every one is."""
    N, TOPK, block, READ = int(t.bindings["N"]), int(t.bindings["TOPK"]), int(t.bindings["BLOCK"]), t.bindings["READ"]
    mask = bind(BM.DsaTopKMaskCountedV2, N=N, TOPK=TOPK)
    bl = X.blocks(N, TOPK, index_block(N, TOPK, block))
    units: Counter = Counter(dict(X.cut(mask) or [(bind(X.DsaTopKMaskWhole, N=N, TOPK=TOPK), 1)]))
    units.update(bind(DsaMaskPositions, N=hi - lo, AT=lo, K=TOPK) for lo, hi in bl)
    units[bind(DsaPositionsMerge, M=len(bl), K=TOPK)] += 1
    units[bind(DsaPositionsChunk, K=TOPK, W=CHUNK)] += -(-TOPK // CHUNK)
    units[bind(DsaPositionsSlot, K=CHUNK, N=N)] += TOPK
    units[READ] += TOPK
    return list(units.items())


#: the long row's units (`cut`), by family
CUTS = {"DsaTopKReadRows_v1": cut}


# --- the twin ----------------------------------------------------------------------------------------------------------------

def merge(lists: Sequence[Sequence[int]], counts: Sequence[int], K: int) -> list[int]:
    """`DsaPositionsMerge_v1{M,K}`'s reference: each list's first min(count, K) positions after those of the lists before it,
    up to K, then 0."""
    out: list[int] = []
    for pos, c in zip(lists, counts):
        out += list(pos[:min(int(c), K)])
    out = out[:K]
    return out + [0] * (K - len(out))


def chunk(slots: Sequence[int], c: int, W: int) -> list[int]:
    """`DsaPositionsChunk_v1{K,W}`'s reference, c and the slots as u32 words: slots cW .. cW+W-1, 0 past K, all 0 where c is
    ceil(K/W) or more."""
    if c >= -(-len(slots) // W):
        return [0] * W
    return [slots[s] if s < len(slots) else 0 for s in range(c * W, (c + 1) * W)]


def slot(slots: Sequence[int], j: int, N: int) -> int:
    """`DsaPositionsSlot_v1{K,N}`'s reference, j and the slots as u32 words: slots[j] where j < K and slots[j] < N, else 0."""
    return slots[j] if j < len(slots) and slots[j] < N else 0


def values(mask: Sequence[int], K: int, block: int = BLOCK) -> dict[tuple, tuple]:
    """Each index unit's outputs on a row's mask (N bits) in blocks of ``block`` keys (the long row's: `index_block`), by key:
    ("positions", b) block b's (K positions, count), ("merge",) the K positions, ("chunk", c) chunk unit c's CHUNK
    positions, ("slot", j) slot unit j's position (its chunk's slot j mod CHUNK)."""
    mask = [int(m) & 1 for m in mask]
    out: dict[tuple, tuple] = {}
    lists, counts = [], []
    for b, (lo, hi) in enumerate(X.blocks(len(mask), K, block)):
        m = mask[lo:hi]
        pos = M.butterfly_apply(M.butterfly_settings(m, K), list(range(lo, hi)), K, 0)
        out[("positions", b)] = (tuple(pos), sum(m))
        lists.append(pos)
        counts.append(sum(m))
    merged = merge(lists, counts, K)
    out[("merge",)] = (tuple(merged),)
    out.update((("chunk", c), (tuple(chunk(merged, c, CHUNK)),)) for c in range(-(-K // CHUNK)))
    out.update((("slot", j), (slot(out[("chunk", j // CHUNK)][0], j % CHUNK, len(mask)),)) for j in range(K))
    return out
