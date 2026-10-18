"""[GLM-5.3] The DSA indexer's top-k of a row past TOPK keys, under Daniel's pattern for data-dependent structure (20:56Z; circuits'
sign-off for decode past 2048 keys, 18 Oct): a static superset of wires plus selection values computed in the circuit.

* The selected-key order is a value: the served kernel's `topk_indices` row, whose order (the above-threshold keys' places come
  from shared-memory atomics) no circuit computes.  The Program takes it from its `run` parameter (`derive.RunInputs`).
* `DsaTopKOrder` checks in gates that the order is a permutation of the top-TOPK set the circuit computes from the row's logits,
  under the tie rule below; its bit is one of the Program's checks, which the verifier requires to be 1.
* Attention reads the selected rows through `DsaGatherRows`, a mux over the row's whole history: the wiring is static.

So a wrong unit can only emit an order the relation accepts, a permutation of the right set, which changes only the order sparse
MLA accumulates the rows in; this is not a witness exemption, since the relation is checked in the circuit.

The served kernel (vLLM d9105ea80): decode takes `cooperative_topk` at most 64 rows a call (sparse_attn_indexer.py:615-642),
whose rows of 2049..16384 keys run `histogram_4096_topk` (topk_histogram_4096.cuh:380-548): a 4096-bin histogram of each logit's
fp16 top 12 bits (`extract_coarse_bin_N`, monotone in the logit, -0 below +0), the keys above the threshold bin written in
`atomicAdd` order, and the threshold bin's resolved by a warp ballot on `(score >, then idx <)` at most 64 ties, else by a 4-round
radix on `convert_to_uint32_v2` (cuh:71-74, also persistent_topk.cuh:50-53).

The tie rule (`topk_set`): key(s) = `convert_to_uint32_v2`(s) = ~bits for a negative word, bits | 2^31 otherwise; j precedes j'
when key_j > key_j', or key_j = key_j' and j < j'.  The top-TOPK set is the TOPK first indices.  Within one coarse bin the
ballot's float order is this key order (±0 never share a bin), so the kernel selects this set but where its row has
  - exactly equal logits straddling the boundary in a threshold bin of more than 64 keys (the radix places equal keys by atomics),
  - more than TOPK keys in the threshold bin (its tie buffer keeps the first TOPK by atomics), or
  - a NaN or an infinity in the threshold bin (the ballot's float comparison is no order there);
where it does not, the relation refuses the served order: a visible failure, never a wrong acceptance.  `persistent_topk` (more
than 64 rows) and the prefill's `top_k_per_row_prefill` key by `convert_to_uint32_v2` too, but their tie placement is unread.

Registered functions (`_v1`; their Boolean versions in `boolean_dsa_topk` are the definitions, these the accelerators tests hold
equal to them):
  DsaTopKOrder{N,TOPK}        (scores: N f32, order: TOPK i32) -> bit: order is a permutation of `topk_set`(scores, TOPK)
  DsaTopKOrderCheck{N}x{K}    the same function as one primitive (`order_check`), DsaTopKOrder's body
  DsaGatherRows{N,TOPK,W}     (order: TOPK i32, rows: N x W bf16) -> TOPK x W bf16: slot s is rows[order[s]] (`b1.Embedding_v1`,
                              0x7FFF past N)
"""
from __future__ import annotations

from verity.ml.circuits.defs import PrimitiveDefinition, bind, composite
from verity.ml.circuits.types import Array
from verity_vllm.program.registry import b1
from verity_vllm.program.registry.prims import BF16, BIT, F32, I32

M32 = 0xFFFFFFFF

_CONF = ("model-untested: the tie rule read from vLLM d9105ea80 topk_histogram_4096.cuh:69-84, 380-548 (cooperative_topk's rows "
         "of 2049..16384 keys); a captured decode past TOPK keys is its hardware check")
_ORDERS: dict[tuple[int, int], PrimitiveDefinition] = {}


def topk_key(bits: int) -> int:
    """`convert_to_uint32_v2` on an f32 word: an unsigned key whose order is the logits' (a larger float a larger key, -0 below
    +0, a positive NaN above +inf and a negative one below -inf)."""
    bits &= M32
    return ~bits & M32 if bits & 0x80000000 else bits | 0x80000000


def topk_set(scores, TOPK: int) -> set[int]:
    """The TOPK indices first under the tie rule: key descending, then index ascending."""
    return set(sorted(range(len(scores)), key=lambda j: (-topk_key(scores[j]), j))[:TOPK])


def order_ok(scores, order, TOPK: int) -> int:
    """1 when the TOPK words ``order`` (u32) are distinct and are `topk_set`(scores, TOPK), else 0."""
    got = [int(w) & M32 for w in order]
    if len(got) != TOPK:
        raise ValueError(f"an order of {len(got)} slots for TOPK {TOPK}")
    return int(len(set(got)) == TOPK and set(got) == topk_set(scores, TOPK))


def check_statics(N: int, TOPK: int) -> None:
    if not 1 <= TOPK <= N < 1 << 31:
        raise ValueError(f"DsaTopKOrder: a row of N keys selects 1 <= TOPK <= N (N={N}, TOPK={TOPK})")


def order_check(N: int, TOPK: int) -> PrimitiveDefinition:
    """`DsaTopKOrderCheck{N}x{TOPK}_v1`: `order_ok` over a row of N f32 scores and a TOPK-slot order."""
    check_statics(N, TOPK)
    p = _ORDERS.get((N, TOPK))
    if p is None:
        def ev(*w: int, _N=N, _K=TOPK) -> int:
            return order_ok(w[:_N], w[_N:], _K)

        p = PrimitiveDefinition(f"DsaTopKOrderCheck{N}x{TOPK}", 1, (("scores", Array(N, F32)), ("order", Array(TOPK, I32))), BIT,
                                ev, doc=order_check.__doc__ or "", conformance=_CONF)
        _ORDERS[(N, TOPK)] = p
    return p


@composite("DsaTopKOrder", 1, ["N", "TOPK"], lambda S: ((("scores", Array(S.N, F32)), ("order", Array(S.TOPK, I32))), BIT),
           conformance=_CONF)
def DsaTopKOrder(B, S, scores, order):
    """The selection relation of a row of N keys: 1 when ``order``'s TOPK slots are distinct indices in [0, N) forming the
    top-TOPK set of ``scores`` under the module docstring's tie rule, in any order; else 0.  `DsaTopKOrder_v2` is it on bits."""
    return B.call(order_check(S.N, S.TOPK), scores, order)


@composite("DsaGatherRows", 1, ["N", "TOPK", "W"],
           lambda S: ((("order", Array(S.TOPK, I32)), ("rows", Array(S.N, Array(S.W, BF16)))), Array(S.TOPK, Array(S.W, BF16))),
           conformance="exact-model-tested")
def DsaGatherRows(B, S, order, rows):
    """The rows sparse MLA reads at a row's top-k indices: slot s is `Embedding_v1{V=N, H=W}`(order[s], rows), a mux over all N
    rows by the slot's index (0x7FFF words past N or below 0)."""
    return B.batch(bind(b1.Embedding, V=S.N, H=S.W), order, rows, axes=(0, None))


DSA_TOPK_COMPOSITES = [DsaTopKOrder, DsaGatherRows]
