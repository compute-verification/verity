"""The published test vectors (``kernels.test_vectors``): the reference digests are pinned here, and on CUDA every GPU entry
point of ``kernels.py`` -- host-loop, sync-free device pipeline, graphed logUp in both segmentations -- reproduces them."""
import hashlib
import json
import os
import random

import pytest

torch = pytest.importorskip("torch")

from packed import kernels as K
from packed import logup_reference as lref
from packed import reference as ref

SUMCHECK_DIGEST = "a6a12f3b"      # prefix of sha256(rounds, rs, final) for sumcheck_t4096_k3_seed7 (README)
LOGUP_DIGEST = "2a8fe70f"         # prefix of sha256(q_root, levels, claim) for logup_Nw12288_Nt2048_seed1


def _digest(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def test_reference_digests_pinned():
    v = K.test_vectors()
    assert v["sumcheck_t4096_k3_seed7"]["sha256(rounds, rs, final)"].startswith(SUMCHECK_DIGEST)
    assert v["logup_Nw12288_Nt2048_seed1"]["sha256(q_root, levels, claim)"].startswith(LOGUP_DIGEST)


cuda = pytest.mark.skipif(not (os.environ.get("PACKED_DEVICE") == "cuda" and torch.cuda.is_available()), reason="PACKED_DEVICE=cuda")


@cuda
def test_sumcheck_vector_on_gpu_both_paths():
    rng = random.Random(7)
    n, k = 12, 3
    gammas = [ref.random_ext(rng) for _ in range(n)]
    g = [rng.randrange(K.P) for _ in range(1 << n)]
    h = [rng.randrange(K.P) for _ in range(1 << n)]
    coins = [ref.random_ext(rng) for _ in range(n)]
    kern = K.make_kernel("cuda", k1_log2=9)       # K1 = 512 >= 128: the U pass is exercised at t = 2^12
    gt = torch.tensor(g, dtype=torch.int32, device="cuda")
    ht = torch.tensor(h, dtype=torch.int32, device="cuda")
    rounds, rs, final = K.sumcheck_layer(kern, gammas, gt, ht, k, lambda i, S: coins[i - 1])
    assert _digest([rounds, rs, list(final)]).startswith(SUMCHECK_DIGEST)
    msgs = K.sumcheck_layer_device(kern, gammas, coins, gt, ht, k)
    assert [K._sp.S_to_ext(S) for S in msgs] == rounds


@cuda
@pytest.mark.parametrize("rpg", [None, 1])
def test_logup_vector_on_graphs(rpg):
    Nw, Nt = 12288, 2048
    rng = random.Random(1)
    queries = [rng.randrange(Nt) for _ in range(Nw)]
    z = ref.random_ext(rng)
    w = torch.tensor(queries, dtype=torch.int32, device="cuda")
    table = torch.arange(Nt, dtype=torch.int32, device="cuda")
    prover = K.GraphLogUp("cuda", k=3, k1_log2=7, rounds_per_graph=rpg)
    tr, claim, _tm = K.logup_prove_graphed(prover, w, table, z, lref.Coins(1))
    assert _digest([tr.q_root, tr.levels, [claim.rho, claim.lam, claim.value]]).startswith(LOGUP_DIGEST)
