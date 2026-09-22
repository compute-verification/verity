"""CUDA-graphed logUp prover (:mod:`logup_graph`) == the Python reference, message for message; both segmentations
(one graph per level with pre-filled coins; one graph per round, Fiat-Shamir style); replay twice (a second proof on
the same shape reuses the graphs)."""
from __future__ import annotations

import random

import pytest
import torch
from packed import logup_reference as lref
from packed import reference as ref

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _case(Nw: int, Nt: int, seed: int):
    rng = random.Random(seed)
    table = list(range(Nt))
    queries = [rng.randrange(Nt) for _ in range(Nw)]
    z = ref.random_ext(rng)
    return queries, table, z


def _check(tr, claim, tr_ref, claim_ref, p, q, n, seed):
    assert tr.q_root == tr_ref.q_root
    assert len(tr.levels) == len(tr_ref.levels)
    for lv, (a, b) in enumerate(zip(tr.levels, tr_ref.levels)):
        assert len(a["rounds"]) == len(b["rounds"])
        for rd, (x, y) in enumerate(zip(a["rounds"], b["rounds"])):
            assert x == y, f"level {lv} round {rd}"
        assert a["vals"] == b["vals"], f"level {lv} vals"
    assert claim.rho == claim_ref.rho and claim.lam == claim_ref.lam and claim.value == claim_ref.value
    c2 = lref.verify_table(n, tr, lref.Coins(seed))
    assert lref.check_leaves(p, q, c2)


def _run(Nw: int, Nt: int, k: int, rpg, seeds=(1, 2), k1_log2: int = 7):
    from packed import logup_graph as lg

    prover = lg.GraphLogUp("cuda", k=k, k1_log2=k1_log2, rounds_per_graph=rpg)
    for seed in seeds:
        queries, table, z = _case(Nw, Nt, seed)
        mult = lref.multiplicities(queries, table)
        p, q, n = lref.leaves(queries, table, mult, z)
        tr_ref, claim_ref = lref.prove_table(p, q, n, lref.Coins(seed))
        w = torch.tensor(queries, dtype=torch.int32, device="cuda")
        t = torch.tensor(table, dtype=torch.int32, device="cuda")
        tr, claim, tm = prover.prove(w, t, z, lref.Coins(seed))
        _check(tr, claim, tr_ref, claim_ref, p, q, n, seed)
        assert tm.graphs >= n


@cuda
@pytest.mark.parametrize("k", [0, 3])
@pytest.mark.parametrize("rpg", [None, 1])
def test_graph_agreement_packed(k, rpg):
    _run(12288, 2048, k, rpg)


@cuda
def test_graph_tiny_generic():
    _run(64, 64, 0, None)


@cuda
def test_graph_no_pad_per_round():
    _run(14336, 2048, 3, 1, seeds=(3,))


@cuda
def test_device_consts_match_host():
    from packed import kernels_graph as kg

    rng = random.Random(9)
    n = 6
    rho = [ref.random_ext(rng) for _ in range(n)]
    coins = [ref.random_ext(rng) for _ in range(n)]
    dev = "cuda"
    rho_t = torch.tensor(rho, dtype=torch.int32, device=dev)
    coins_t = torch.tensor(coins, dtype=torch.int32, device=dev)
    for start, nv in ((0, 0), (0, 3), (2, 4), (1, 1)):
        got = [tuple(r) for r in kg.eq_table_vars(rho_t, start, nv).tolist()]
        assert got == ref.eq_table(rho[start:start + nv])
    for i in range(1, n + 1):
        out = torch.empty(36, dtype=torch.int32, device=dev)
        kg.round_consts(rho_t, coins_t, i, out)
        c = out.tolist()
        assert tuple(c[0:6]) == ref.eq_points(rho[:i - 1], coins[:i - 1])
        gm = rho[i - 1]
        for x in range(4):
            assert tuple(c[6 + 6 * x:12 + 6 * x]) == ref.eval_at(ref.sub(ref.ONE, gm), gm, x)
        assert tuple(c[30:36]) == tuple(v * pow(2, 32, ref.P) % ref.P for v in coins[i - 1])
