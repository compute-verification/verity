"""At WORLD > 2 a collective reads W-1 peer partials.  The rank Program lowers one `peers_<k>` Input per peer (one field each); the fold
keeps ONE homogeneous peers Record per request and reads the collective's peers as one run over adjacent fields.  The compare splits
such a run at the field boundaries, so both sides canonicalise to the same ("PEER", k, ..) per peer, in record form, in compact form and
through GM-01's per-request Projection, and a peer read out of order or missing still fails.  (Qwen3-235B TP8 replay: before the split,
every decode all-reduce and the logits all-gather of every rank differed, 5,891 operands per rank.)"""

from __future__ import annotations

import pytest

from verity_vllm.check.match import batch_decomp as BD
from verity_vllm.check.match import program_compare as C
from verity_vllm.program import compact as K

H, W, K0 = 8, 8, 1323          # leaves per peer partial, world, the request's first collective ordinal
TOK, WTS, PEERS = 0, 1, 2
WNAME = "model.layers.0.mlp.down_proj.weight"


def _prog(label: str, inputs: dict, peer_fields: dict, ar_operand: list[tuple], *, compact: bool = False) -> C.RootView:
    """One step: a Gemm partial, its all-reduce with the peers, a Gemm reading the reduction."""
    P = C.RootView.__new__(C.RootView)
    P.label, P.digest = label, f"digest-{label}"
    P.inputs = inputs
    P.tok_node, P.seed_nodes, P.seed_index = TOK, set(), {}
    P.weight_fields = {WTS: WNAME}
    P.param_names, P.oracle_rows, P.n_nodes, P.nodes = [], None, 0, None
    P.supplied_fields, P.supplied_owner = {}, {}
    P.peer_fields, P.peer_index = peer_fields, {n: 0 for n in peer_fields}
    k = max(inputs) + 1
    gemm, ar = k, k + 1
    operand = [run if run[1] != "own" else ("n", gemm, 0, H) for run in ar_operand]
    inst = [{"k": gemm, "spec": f"Gemm_v1{{M=1,N={H},K=4}}", "form": "call", "args": [[("n", TOK, 0, 4)], [("n", WTS, 0, 4 * H)]], "out": f"bf16[{H}]"},
            {"k": ar, "spec": f"AllReduce_v2{{WORLD={W},N={H}}}", "form": "call", "args": [operand], "out": f"bf16[{H}]"},
            {"k": ar + 1, "spec": f"Gemm_v1{{M=1,N={H},K={H}}}", "form": "call", "args": [[("n", ar, 0, H)], [("n", WTS, 0, H * H)]], "out": f"bf16[{H}]"}]
    P.compact = compact
    P.inst = [dict(r, args=[K.from_runs(a) if compact else list(a) for a in r["args"]]) for r in inst]
    P.k2i = {r["k"]: i for i, r in enumerate(P.inst)}
    P._bind_splits()
    P._segment()
    return P


OWN = ("n", "own", 0, H)


def rank_program(rank: int, order: list[int] | None = None, **kw) -> C.RootView:
    """The rank Program: tok, weights, then `peers_<K0+i>` Inputs; the all-reduce reads the parts in rank order (its own at `rank`)."""
    order = list(range(W - 1)) if order is None else order
    inputs = {TOK: ("Input32_v1", 4), WTS: ("Input16_v1", H * H)} | {PEERS + i: ("Input16_v1", H) for i in range(W - 1)}
    peer_fields = {PEERS + i: [(0, K0 + i)] for i in range(W - 1)}
    reads = [("n", PEERS + i, 0, H) for i in order]
    return _prog(f"build{rank}", inputs, peer_fields, reads[:rank] + [OWN] + reads[rank:], **kw)


def rank_fold(rank: int, peers_runs=None, **kw) -> C.RootView:
    """The fold: ONE peers Record of W-1 fields at increasing offsets, read as one run on each side of the rank's own partial."""
    inputs = {TOK: ("Input32_v1", 4), WTS: ("Input16_v1", H * H), PEERS: ("Input16_v1", (W - 1) * H)}
    peer_fields = {PEERS: [(i * H, K0 + i) for i in range(W - 1)]}
    if peers_runs is None:
        before, after = [("n", PEERS, 0, rank * H)] if rank else [], [("n", PEERS, rank * H, (W - 1) * H)] if rank < W - 1 else []
        operand = before + [OWN] + after
    else:
        operand = peers_runs
    F = _prog(f"fold{rank}", inputs, peer_fields, operand, **kw)
    F.weight_fields, F.oracle_wmap, F._wlos, F.oracle_wmap_overlaps = {}, [(0, H * H, WNAME)], [0], []   # named by the oracle's field map
    return F


def _ar(P: C.RootView) -> int:
    return next(i for i, r in enumerate(P.inst) if r["spec"].startswith("AllReduce_v2"))


@pytest.mark.parametrize("rank", [0, 3, W - 1])
@pytest.mark.parametrize("compact", [False, True])
def test_a_peers_run_over_adjacent_fields_compares_as_one_peer_read_per_field(rank, compact):
    A, F = rank_program(rank, compact=compact), rank_fold(rank, compact=compact)
    want = [("PEER", K0 + i, 0, H) for i in range(W - 1)]
    want.insert(rank, ("N", 0, 0, 0, H))
    for P in (A, F):
        got = P.canon(_ar(P))[0]
        assert (K.expand_mixed(got) if compact else got) == want, (P.label, got)
    res = C.compare_steps(A, F)
    assert res["totals"]["operands_different"] == 0 and res["canonical_sha256"]["equal"], res["steps"][0].get("wiring")


@pytest.mark.parametrize("compact", [False, True])
def test_the_request_projection_splits_its_own_peers_run_alike(compact):
    """GM-01 compares each request's Projection of the fold with the request's Program: the same split through `Projection._src(P)`."""
    A, F = rank_program(3, compact=compact), rank_fold(3, compact=compact)
    proj = BD.Projection(F, "r0", list(range(len(F.inst))), TOK, WTS, peer_node=PEERS)
    assert proj.canon(_ar(proj)) == F.canon(_ar(F))
    res = C.compare_steps(A, proj)
    assert res["totals"]["operands_different"] == 0 and res["canonical_sha256"]["equal"]
    assert not proj.xreq


def test_a_run_ending_inside_a_field_splits_into_the_partial_reads():
    F = rank_fold(0, peers_runs=[OWN, ("n", PEERS, H // 2, 2 * H + 3)])
    assert F.canon(_ar(F))[0] == [("N", 0, 0, 0, H), ("PEER", K0, H // 2, H), ("PEER", K0 + 1, 0, H), ("PEER", K0 + 2, 0, 3)]
    assert F._split_peer(("n", PEERS, H, 2 * H)) == [("n", PEERS, H, 2 * H)]       # inside one field: unchanged
    assert F._split_peer(("n", WTS, 0, 3 * H)) == [("n", WTS, 0, 3 * H)]           # not a peers node: unchanged


@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("name, build_kw, fold_runs", [
    ("peers_out_of_order", {"order": [1, 0, 2, 3, 4, 5, 6]}, None),     # the Program reduces peer k1 before k0
    ("peer_missing", {}, [OWN, ("n", PEERS, 0, (W - 2) * H)]),          # the fold's run stops one field short
    ("peer_offset", {}, [OWN, ("n", PEERS, 1, (W - 1) * H)]),           # shifted by one leaf: every field read off its start
])
def test_negatives_still_fail(name, build_kw, fold_runs, compact):
    A, F = rank_program(0, compact=compact, **build_kw), rank_fold(0, peers_runs=fold_runs, compact=compact)
    res = C.compare_steps(A, F)
    assert res["totals"].get("operands_different", 0) > 0 or not res["steps"][0]["spec_seq_equal"], name
    assert not res["canonical_sha256"]["equal"], name
