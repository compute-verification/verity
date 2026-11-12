"""verity_vllm.program.dsa_table: a DSA row past INDEX_TOPK keys read by table.  `DsaTopKIndex_v1{N,TOPK}` gives a mask's TOPK
members' positions in ascending order, `DsaHistoryReads_v1{V,TOPK,H}` the history rows at given positions, and the two over a
row's counted mask (`DsaTopKMaskCounted_v1`) read the rows the rows network routes for it: `DsaButterflyRowsPad_v1{T,N}` over the
row's T history rows and its pad (`DsaButterflyRows_v1{N}` at T = N) at `DsaButterflySettings_v1{N}` of that mask.  The stand-ins
are held to that until the registry holds word Definitions of the two names."""

from __future__ import annotations

import random

import pytest

from verity.circuits import evaluation as E
from verity.circuits.defs import bind
from verity.definitions.registry import dsa_topk_mask as M
from verity_vllm.program import dsa_table as DT


def _members(rng: random.Random, T: int, N: int, TOPK: int) -> tuple[list[int], list[int]]:
    """A mask of N positions with TOPK members among its first T, and those members."""
    on = sorted(rng.sample(range(T), TOPK))
    return [int(j in on) for j in range(N)], on


@pytest.mark.parametrize("N, TOPK", [(8, 4), (9, 8), (12, 4), (16, 5)])
def test_the_index_lists_a_masks_members_in_ascending_order(N, TOPK):
    rng, fn = random.Random(31 * N + TOPK), bind(DT.DsaTopKIndex, N=N, TOPK=TOPK)
    assert fn.id == f"DsaTopKIndex_v1{{N={N},TOPK={TOPK}}}"
    for _ in range(6):
        mask, on = _members(rng, N, N, TOPK)
        assert list(E.evaluate(fn, mask)) == on


def test_the_reads_are_the_history_rows_at_their_positions():
    """Each output row is the history row its position names, a position read more than once or not at all included."""
    V, TOPK, H, rng = 7, 4, 3, random.Random(5)
    fn = bind(DT.DsaHistoryReads, V=V, TOPK=TOPK, H=H)
    hist = [[rng.getrandbits(16) for _ in range(H)] for _ in range(V)]
    for tok in ([0, 1, 2, 3], [6, 6, 0, 5], [3, 1, 4, 1]):
        assert list(E.evaluate(fn, tok, [w for r in hist for w in r])) == [w for j in tok for w in hist[j]]


@pytest.mark.parametrize("T, N, TOPK", [(5, 8, 4), (7, 8, 4), (8, 8, 4), (9, 16, 8), (12, 12, 4)])
def test_reads_at_a_counted_masks_index_are_the_rows_the_rows_network_routes(T, N, TOPK):
    """A row of T keys at its class N: the history read at the index of its counted mask (its own top-TOPK of T scores, PAD_SCORE
    past them) is the rows network's read of the same mask over its T history rows and N - T zero rows, row for row."""
    from verity_vllm.program.frontend.rules.vllm_dsa import PAD_SCORE

    W, rng = 3, random.Random(T * N + TOPK)
    counted, index = bind(M.DsaTopKMaskCounted, N=N, TOPK=TOPK), bind(DT.DsaTopKIndex, N=N, TOPK=TOPK)
    reads, settings = bind(DT.DsaHistoryReads, V=T, TOPK=TOPK, H=W), bind(M.DsaButterflySettings, N=N, K=TOPK)
    network = (bind(M.DsaButterflyRowsPad, T=T, N=N, K=TOPK, W=W, E=16) if T < N else bind(M.DsaButterflyRows, N=N, K=TOPK, W=W, E=16))
    for _ in range(4):
        scores = [rng.choice([0x3F800000, 0x40000000, rng.getrandbits(31)]) for _ in range(T)] + [PAD_SCORE] * (N - T)
        mask = list(E.evaluate(counted, scores, T))
        assert sum(mask[:T]) == TOPK and not any(mask[T:])
        hist = [rng.getrandbits(16) for _ in range(T * W)]
        sw = list(E.evaluate(settings, mask))
        want = E.evaluate(network, sw, hist, [0] * ((N - T) * W)) if T < N else E.evaluate(network, sw, hist)
        assert E.evaluate(reads, list(E.evaluate(index, mask)), hist) == want


@pytest.mark.parametrize("statics", [dict(N=4, TOPK=5), dict(N=1, TOPK=1), dict(N=8, TOPK=0)])
def test_an_index_past_the_concentrators_statics_is_refused(statics):
    with pytest.raises(ValueError, match="concentrator"):
        bind(DT.DsaTopKIndex, **statics).signature()
