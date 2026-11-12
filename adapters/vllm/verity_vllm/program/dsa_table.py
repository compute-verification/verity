"""[GLM-5.3] DSA's long row by table reads (`Registered.dsa_table_reads`), the two word Definitions its Program calls past
INDEX_TOPK keys in place of the rows network (`DsaButterflySettings_v1`, `DsaButterflyRows(Pad)_v1`):

    DsaTopKIndex_v1{N,TOPK}        (mask: Array<N, Value<1>>) -> Array<TOPK, Value<32>>
        `DsaButterflyRows_v1{N,K=TOPK,W=1,E=32}` over the positions 0..N-1 at `DsaButterflySettings_v1{N,K=TOPK}` of the mask
        (`dsa_topk_mask.DsaTopKRow`'s construction): at a mask of TOPK members, as a counted mask past TOPK keys always is, slot j
        is the j-th member's position, the slot the rows network gives that member's history row
    DsaHistoryReads_v1{V,TOPK,H}   (tok: Array<TOPK, Value<32>>, hist: Array<V, Array<H, Value<16>>>) -> Array<TOPK, Array<H, Value<16>>>
        slot j is the history row at tok[j], `b1.Embedding_v1{V,H}` per slot (0x7FFF words at a tok outside [0, V), which no index
        of a row of at most V keys holds)

So `DsaHistoryReads_v1` at `DsaTopKIndex_v1` of a counted mask of TOPK members is the row `DsaButterflyRowsPad_v1` routes at that
mask's settings (tests/program/test_dsa_table.py).  An indexer layer computes the index once per long row and an IndexShare layer
reads its own history at it, so the index and the reads are two Calls.  Serving cuts a reads Call into TOPK reads of the
(request, layer) history table, `EmbeddingRead_v1{V=2^17,H}` each, read j's ``tok`` wired to index output j.

These are stand-ins: the Boolean index units and table read (`commitments.gates.embedding`) are on bits only, with no word view a
Program can call.  Once `verity.definitions.registry` holds word Definitions of these two names with their Boolean versions, the
Program calls those and this module goes.
"""
from __future__ import annotations

from verity.circuits.defs import bind, composite
from verity.definitions.ops.prims import const
from verity.circuits.refs import Coll, concat
from verity.circuits.types import Array
from verity.definitions.registry import b1
from verity.definitions.registry import dsa_topk_mask as M
from verity.definitions.registry.prims import BF16, BIT, I32

_INDEX_CONF = ("reference-defined: a mask's members' positions in ascending order, held to the rows network (DsaButterflyRows_v1 at "
               "DsaButterflySettings_v1) on counted masks of TOPK members (adapters/vllm/tests/program/test_dsa_table.py)")
_READS_CONF = ("reference-defined: an index into its operand, held to the rows network's read at DsaTopKIndex_v1 of counted masks "
               "(adapters/vllm/tests/program/test_dsa_table.py)")


def _refuse_index(b: dict) -> str | None:
    N, K = b.get("N"), b.get("TOPK")
    if N is not None and K is not None:
        try:
            M.check_net(N, K)
        except ValueError as e:
            return str(e)
    return None


def _refuse_reads(b: dict) -> str | None:
    bad = [k for k in ("V", "TOPK", "H") if b.get(k) is not None and b[k] < 1]
    return f"V, TOPK and H at least 1 ({', '.join(f'{k}={b[k]}' for k in bad)})" if bad else None


@composite("DsaTopKIndex", 1, ["N", "TOPK"], lambda S: ((("mask", Array(S.N, BIT)),), Array(S.TOPK, I32)), conformance=_INDEX_CONF,
           refuse=_refuse_index)
def DsaTopKIndex(B, S, mask):
    """The mask's members' positions, ascending: the concentrator at the mask's settings over the positions."""
    sw = B.call(bind(M.DsaButterflySettings, N=S.N, K=S.TOPK), mask)
    at = Coll(Array(S.N, Array(1, I32)), concat([B.call(const(32, j)).refs for j in range(S.N)]))
    out = B.call(bind(M.DsaButterflyRows, N=S.N, K=S.TOPK, W=1, E=32), sw, at)
    return Coll(Array(S.TOPK, I32), out.refs)


@composite("DsaHistoryReads", 1, ["V", "TOPK", "H"],
           lambda S: ((("tok", Array(S.TOPK, I32)), ("hist", Array(S.V, Array(S.H, BF16)))), Array(S.TOPK, Array(S.H, BF16))),
           conformance=_READS_CONF, refuse=_refuse_reads)
def DsaHistoryReads(B, S, tok, hist):
    """TOPK reads of the history: slot j is ``hist[tok[j]]``."""
    return B.batch(bind(b1.Embedding, V=S.V, H=S.H), tok, hist, axes=(0, None))


__all__ = ["DsaHistoryReads", "DsaTopKIndex"]
