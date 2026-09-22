"""GPU logUp prover == the Python reference, message for message (CUDA only)."""
from __future__ import annotations

import random

import pytest
import torch

from .. import logup_reference as lref
from .. import reference as ref

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _case(Nw: int, Nt: int, seed: int):
    rng = random.Random(seed)
    table = list(range(Nt))
    queries = [rng.randrange(Nt) for _ in range(Nw)]
    z = ref.random_ext(rng)
    return queries, table, z


def _run(Nw: int, Nt: int, k: int, seed: int = 1, k1_log2: int = 7):
    from .. import logup_gpu as lg

    queries, table, z = _case(Nw, Nt, seed)
    mult = lref.multiplicities(queries, table)
    p, q, n = lref.leaves(queries, table, mult, z)
    tr_ref, claim_ref = lref.prove_table(p, q, n, lref.Coins(seed))
    prover = lg.LogUpProver("cuda", k=k, k1_log2=k1_log2)
    w = torch.tensor(queries, dtype=torch.int32, device="cuda")
    t = torch.tensor(table, dtype=torch.int32, device="cuda")
    tr, claim, _tm = prover.prove(w, t, z, lref.Coins(seed))
    assert tr.q_root == tr_ref.q_root
    assert len(tr.levels) == len(tr_ref.levels)
    for lv, (a, b) in enumerate(zip(tr.levels, tr_ref.levels)):
        assert a["vals"] == b["vals"], f"level {lv} vals"
        assert len(a["rounds"]) == len(b["rounds"])
        for rd, (x, y) in enumerate(zip(a["rounds"], b["rounds"])):
            assert x == y, f"level {lv} round {rd}"
    assert claim.rho == claim_ref.rho and claim.lam == claim_ref.lam and claim.value == claim_ref.value
    # and the verifier accepts the GPU transcript
    c2 = lref.verify_table(n, tr, lref.Coins(seed))
    assert lref.check_leaves(p, q, c2)


@cuda
@pytest.mark.parametrize("k", [0, 3])
def test_agreement_packed(k):
    # Nw = 12 K, Nt = 2 K, pad 2 K (n = 14): every region's b-range is a multiple of 128 through round 3
    _run(12288, 2048, k)


@cuda
def test_agreement_tiny_generic():
    _run(64, 64, 0)


@cuda
def test_agreement_no_pad():
    _run(14336, 2048, 3)
