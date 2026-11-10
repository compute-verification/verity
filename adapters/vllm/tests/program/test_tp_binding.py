"""`program.tp_binding`, a tensor-parallel request's binding derived from its rank Programs, and `tp_workload.TpWorkload`, the
composition of the rank workloads with it.

The dense toy compiles at TP1 and TP2 only, so the toy at TP2 (`test_steps`' TINY, `_mixed`'s requests) is a compiled request,
and the rank Programs of three and four ranks are built here as `predict` builds a rank's collectives (`common._Peers`): a
Gemm partial per row all-reduced per row, and with ``whole`` a second one all-reduced over both rows at once."""
from __future__ import annotations

import pytest

from tests.program.predict.test_steps import _mixed, _registered
from verity_vllm.program import tp_binding as TB
from verity_vllm.program.predict.common import _Peers
from verity_vllm.program.predict.steps import compile_steps
from verity_vllm.program.tp_workload import TP_WORKLOAD_TAG, TpWorkload

from verity.ml.circuits.defs import Builder, bind
from verity.ml.circuits.ops import gemm as G
from verity.ml.circuits.refs import Affine, Coll
from verity.ml.circuits.types import Array
from verity.ml.definitions.registry import b1_tp2 as TP
from verity.ml.definitions.registry.prims import BF16

N, ROWS = 16, 2
GEMM = bind(G.Gemm, K=N, N=N, **G.GEMM_AMPERE)


def rank_body(world: int, rank: int, whole: bool = False, peers: list[int] | None = None, combine=None, chain: bool = False):
    """Rank ``rank``'s request body of a ``world``-rank request: a Gemm partial per row all-reduced per row (`_Peers.rows`), and
    with ``whole`` a second Gemm per row all-reduced over both rows at once (`_Peers.whole`), with ``chain`` a second Gemm
    per row reading that row's all-reduce, all-reduced per row; ``peers`` the order `_Peers` reads its peer Inputs in,
    ``combine`` the per-row all-reduce's Definition."""
    params = (("x", Array(ROWS * N, BF16)), ("w", Array(N, Array(N, BF16))),
              *((f"peers_{m}", Array(ROWS * N, BF16)) for m in range((1 + whole + chain) * (world - 1))))
    row = combine or bind(TP.AllReduce, WORLD=world, N=N)
    wide = bind(TP.AllReduce, WORLD=world, N=ROWS * N)
    B = Builder(params, (wide if whole else row).signature()[1])

    def partials() -> list:
        w = Coll(params[1][1], Affine("p", 1, 0, N * N))
        return [B.call(GEMM, Coll(Array(N, BF16), Affine("p", 0, i * N, N)), w).refs for i in range(ROWS)]

    inputs = [Coll(t, Affine("p", i, 0, t.leaves)) for i, (_, t) in enumerate(params[2:], start=2)]
    p = _Peers(inputs if peers is None else [inputs[i] for i in peers], world, rank)
    outs = [B.call(row, Coll(row.signature()[0][0][1], parts)) for parts in p.rows(partials())]
    if chain:
        w = Coll(params[1][1], Affine("p", 1, 0, N * N))
        outs = [B.call(row, Coll(row.signature()[0][0][1], parts))
                for parts in p.rows([B.call(GEMM, Coll(Array(N, BF16), o.refs), w).refs for o in outs])]
    out = outs[-1]
    if whole:
        out = B.call(wide, Coll(wide.signature()[0][0][1], p.whole(partials())))
    return B.finish(out)


@pytest.fixture(scope="module")
def toy() -> list[TB.RequestBinding]:
    """The toy's three requests at TP2, each one's binding of its two rank Programs."""
    advice, ingress = _mixed()
    cs = [compile_steps(_registered(tp={"world": 2, "rank": r}), advice, ingress) for r in (0, 1)]
    ds = [[u.digest for u in c.requests] for c in cs]
    return [TB.derive([cs[r].programs[ds[r][i]].fn.body for r in (0, 1)], [ds[r][i] for r in (0, 1)]) for i in range(3)]


def test_a_compiled_requests_rank_programs_bind_every_peer_slice_to_the_other_ranks_partial(toy):
    """At TP2 each collective of the toy's request Programs is a site per row, live at both ranks and combined there with one
    Definition, each rank's own segment one node's whole output: the same node at both ranks, which are one structure, the
    embedding's shard its rank's. Every peer slice a rank reads is in one site, two requests of one shape bind alike, and the
    binding's digest is its record's."""
    assert [len(b.sites) for b in toy] == [45, 30, 45] and toy[0] == toy[2] and toy[0].digest == toy[2].digest != toy[1].digest
    for b in toy:
        assert b.world == 2 and len(set(b.ranks)) == 2 and len({(s.k, s.lo) for s in b.sites}) == len(b.sites)
        for s in b.sites:
            (a,), (c,) = s.own
            assert a == c and a[2] - a[1] == s.hi - s.lo and s.combine[0] == s.combine[1] and s.combine[0][1] == "parts"
            producers = [b.fns[r][a[0]].id for r in (0, 1)]
            assert producers[0] == producers[1] or producers == [f"EmbeddingShard_v1{{VS=128,H=64,START={r * 128}}}" for r in (0, 1)]
        assert {b.fns[0][s.combine[0][0]].id for s in b.sites} == {"AllReduce2_v1{N=64}", "AllGather2_v1{N=128}"}
    assert toy[0].digest == TB.RequestBinding(2, toy[0].ranks, toy[0].sites).digest


@pytest.mark.parametrize("world", [3, 4])
def test_each_ranks_segment_of_a_collective_is_the_peer_input_that_rank_sends(world):
    """At three and four ranks each rank reads rank j's partial of collective k from its Input k·(W−1)+j', j' rank j's place
    among the others: every site pairs, rank r's segment p being rank p's own Gemm row, a collective over both rows (`whole`)
    too, its own segment two producers' rows. A rank that reads its peers in another order is refused by name."""
    b = TB.derive([rank_body(world, r, whole=True) for r in range(world)])
    assert [(s.k, s.lo, s.hi) for s in b.sites] == [(0, 0, N), (0, N, 2 * N), (1, 0, 2 * N)]
    assert [s.own for s in b.sites] == [(o,) * world for o in [((0, 0, N),), ((1, 0, N),), ((4, 0, N), (5, 0, N))]]
    assert [s.combine for s in b.sites] == [((n, "parts"),) * world for n in (2, 3, 6)]
    swapped = list(range(world - 1))
    swapped[0], swapped[1] = 1, 0
    with pytest.raises(ValueError, match=r"rank 1's Program .*: node 2 \(AllReduce_v2\{WORLD=\d,N=16\}\): its segment 0 reads peers_1, "
                                         "another rank's Input than rank 0's"):
        TB.derive([rank_body(world, r, peers=swapped if r == 1 else None) for r in range(world)])


def test_what_does_not_pair_across_ranks_is_refused_by_name():
    """A collective live at some ranks only, or combined with different Definitions at two ranks, pairs with nothing."""
    with pytest.raises(ValueError, match=r"collective 1's \[0, 32\) is read at ranks \[0, 2\] only: a collective live at some ranks"):
        TB.derive([rank_body(3, r, whole=r != 1) for r in range(3)])
    gather = bind(TP.AllGather, WORLD=3, N=N)
    with pytest.raises(ValueError, match=r"collective 0's \[0, 16\): rank 0 combines it with AllReduce_v2\{WORLD=3,N=16\}, rank 2 with "
                                         r"AllGather_v1\{WORLD=3,N=16\}"):
        TB.derive([rank_body(3, r, combine=gather if r == 2 else None) for r in range(3)])
    with pytest.raises(ValueError, match="not each rank of one world once"):
        TB.pair([TB.rank_sites(rank_body(3, r), 3, r) for r in (0, 2, 1)])


def test_the_tp_workload_digest_commits_to_the_rank_workloads_in_order_and_each_requests_binding(toy):
    """The composition's digest is SHA-512 of its record (W, the rank workloads' digests in rank order and each request's
    binding's digest, under its tag): another rank order or another binding is another digest, and a binding of another world,
    one rank or a rank twice is refused."""
    ranks = ["a" * 128, "b" * 128]
    w = TpWorkload(ranks, [toy[0], None, toy[2]])
    assert w.record == {"tag": TP_WORKLOAD_TAG, "world": 2, "ranks": ranks, "bindings": [toy[0].digest, None, toy[0].digest]}
    assert w.digest == TpWorkload(list(ranks), [toy[0], None, toy[0]]).digest
    assert len({w.digest, TpWorkload(ranks[::-1], [toy[0], None, toy[2]]).digest, TpWorkload(ranks, [toy[1], None, toy[2]]).digest}) == 3
    assert w.artifact()["requests"][0] == {"ranks": list(toy[0].ranks), "binding": toy[0].digest, "sites": 45}
    three = TB.derive([rank_body(3, r) for r in range(3)])
    for args, why in [((ranks, [three]), "request 0's binding pairs 3 ranks' Programs, not the workload's 2"),
                      ((ranks[:1], []), "two or more ranks' distinct workloads, not 1"),
                      ((ranks[:1] * 2, []), r"not 2 \(1 distinct\)")]:
        with pytest.raises(ValueError, match=why):
            TpWorkload(*args)
