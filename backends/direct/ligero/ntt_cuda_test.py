"""``ntt_cuda`` == ``field.ntt`` (torch radix-2 chain), both directions, every supported size, random rows; skipped
without CUDA + cupy.  Run: ``python -m backends.direct.ligero.ntt_cuda_test``."""
import sys

import torch

from . import field
from .field import P


def check(device="cuda", seed=1) -> bool:
    from . import ntt_cuda
    g = torch.Generator(device="cpu").manual_seed(seed)
    ok = True
    for n in (8192, 16384, 32768, 65536):
        for B in (1, 6, 36):
            x = torch.randint(0, P, (B, n), generator=g, dtype=torch.int64).to(device)
            for inverse in (False, True):
                field.CUDA_NTT = False
                ref = field.ntt(x, inverse)
                field.CUDA_NTT = True
                got = ntt_cuda.ntt(x, inverse)
                same = torch.equal(ref, got)
                ok &= same
                print(f"n={n:6d} B={B:3d} inverse={inverse!s:5s} {'OK' if same else 'MISMATCH'}")
    # non-canonical input (>= p and negative) reduces the same way
    x = torch.randint(-3 * P, 3 * P, (4, 65536), generator=g, dtype=torch.int64).to(device)
    field.CUDA_NTT = False
    ref = field.ntt(x)
    field.CUDA_NTT = True
    same = torch.equal(ref, ntt_cuda.ntt(x))
    ok &= same
    print(f"non-canonical input: {'OK' if same else 'MISMATCH'}")
    # round trip
    x = torch.randint(0, P, (3, 65536), generator=g, dtype=torch.int64).to(device)
    same = torch.equal(ntt_cuda.ntt(ntt_cuda.ntt(x), True), x)
    ok &= same
    print(f"round trip: {'OK' if same else 'MISMATCH'}")
    return ok


def test_ntt_cuda():
    import pytest
    if not torch.cuda.is_available():
        pytest.skip("CUDA")
    try:
        import cupy  # noqa: F401
    except ImportError:
        pytest.skip("cupy")
    assert check()


if __name__ == "__main__":
    ok = check()
    if len(sys.argv) > 1 and sys.argv[1] == "--time":
        from . import ntt_cuda
        for shape in ((6, 65536), (36, 65536), (36, 16384)):
            x = torch.randint(0, P, shape, dtype=torch.int64, device="cuda")
            for name, fn in (("torch", lambda: (setattr(field, "CUDA_NTT", False), field.ntt(x, True))[1]),
                             ("cuda", lambda: ntt_cuda.ntt(x, True))):
                for _ in range(3):
                    fn()
                torch.cuda.synchronize()
                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                e0.record()
                for _ in range(20):
                    fn()
                e1.record(); e1.synchronize()
                print(f"{name:6s} intt {shape}: {e0.elapsed_time(e1) / 20:.3f} ms")
        field.CUDA_NTT = True
    sys.exit(0 if ok else 1)
