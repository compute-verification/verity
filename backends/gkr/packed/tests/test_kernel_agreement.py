"""The torch kernel (CPU, and CUDA when ``PACKED_DEVICE=cuda``) emits the reference transcript bit for bit."""
import os
import random

import pytest

torch = pytest.importorskip("torch")

from packed import reference as ref
from packed import sumcheck_packed as sp


def _devices():
    """``(device, mode)`` pairs: CPU eager always; ``PACKED_DEVICE=cuda`` adds CUDA eager and (if triton imports) triton."""
    out = [("cpu", "eager")]
    dev = os.environ.get("PACKED_DEVICE")
    if dev and torch.cuda.is_available():
        out.append((dev, "eager"))
        try:
            import triton  # noqa: F401

            out.append((dev, "triton"))
        except ImportError:
            pass
    return out


def _instance(n: int, seed: int):
    rng = random.Random(seed)
    gammas = [ref.random_ext(rng) for _ in range(n)]
    g = [rng.randrange(ref.P) for _ in range(1 << n)]
    h = [rng.randrange(ref.P) for _ in range(1 << n)]
    return gammas, g, h


@pytest.mark.parametrize("device,mode", _devices())
@pytest.mark.parametrize("n,k,k1_log2", [(6, 0, 3), (6, 1, 3), (6, 2, 2), (7, 3, 2), (8, 4, 3), (8, 2, 13), (9, 3, 4), (10, 4, 5)])
def test_kernel_matches_reference(device, mode, n, k, k1_log2):
    gammas, g, h = _instance(n, 1000 + 10 * n + k)
    ker = sp.Kernel(device, k1_log2=k1_log2, mode=mode)
    gt = torch.tensor(g, dtype=torch.int32, device=device)
    ht = torch.tensor(h, dtype=torch.int32, device=device)
    stats = sp.Stats()
    rounds, rs, final = sp.prove(ker, gammas, gt, ht, k, ref.seeded_challenger(5), stats)
    rounds_ref, rs_ref, final_ref = ref.packed_prover(gammas, g, h, k, ref.seeded_challenger(5))
    assert rs == rs_ref
    assert rounds == rounds_ref
    assert final == final_ref
    assert stats.macs_performed > 0


def test_scalar_matrix_is_the_convolution():
    rng = random.Random(3)
    c = ref.random_ext(rng)
    v = ref.random_ext(rng)
    C = torch.tensor(sp.scalar_matrix(c, ref.DEGREE))
    L = torch.tensor([ref.limbs_balanced(x) for x in v], dtype=torch.int8).reshape(1, -1)
    Z, _ = sp.mm_int8(L, C)
    out = sp.recompose(Z, ref.DEGREE, torch.tensor(sp.POW2, dtype=torch.int64))
    assert tuple(int(x) for x in out[0].tolist()) == ref.mul(c, v)
