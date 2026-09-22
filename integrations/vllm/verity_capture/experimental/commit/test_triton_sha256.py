"""Triton SHA-256 launch == hashlib (CUDA only).  EXPERIMENTAL, CMT-1."""
from __future__ import annotations

import hashlib

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

from . import torch_sha256 as TS  # noqa: E402
from .triton_sha256 import sha256_messages_triton  # noqa: E402


@pytest.mark.parametrize("L", [0, 55, 56, 64, 90, 103, 120, 511, 4096, 65536])   # 1..1025-block messages, one kernel
@pytest.mark.parametrize("n", [1, 129, 3000])
def test_triton_vs_hashlib(L, n):
    g = torch.Generator().manual_seed(L * 1000 + n)
    m = torch.randint(0, 256, (n, L), dtype=torch.uint8, generator=g).cuda()
    d = sha256_messages_triton(m).cpu()
    mc = m.cpu()
    for i in range(0, n, max(1, n // 50)):
        assert bytes(d[i].numpy().tobytes()) == hashlib.sha256(bytes(mc[i].numpy().tobytes())).digest()


def test_frames_and_fold_match_torch_backend():
    x = torch.randint(0, 256, (3001, 64), dtype=torch.uint8).cuda()
    a = TS.fold_levels(TS.pos_leaf_digests(x, compiled="triton"), compiled="triton")
    b = TS.fold_levels(TS.pos_leaf_digests(x, compiled=False), compiled=False)
    assert len(a) == len(b) and all(torch.equal(p, q) for p, q in zip(a, b, strict=True))


def test_direct_frame_kernels_match_buffer_path_and_hashlib():
    """Register-built frames (leaf/node/lift) == frame-buffer path == hashlib, including an odd level (lift)."""
    from .triton_sha256 import fold_level_direct, leaf_digests_direct
    stream = torch.randint(0, 256, (64 * 1001,), dtype=torch.uint8).cuda()
    d = leaf_digests_direct(stream, 64)
    buf = TS._frames(1001, 26 + 64, ((0, TS.POS_LEAF_TAG + (64).to_bytes(8, "big")),), stream.device)
    buf[:, 26:90] = stream.view(1001, 64)
    assert torch.equal(d, TS.sha256_padded(buf, compiled="triton"))
    sc = stream.cpu().numpy().tobytes()
    for i in (0, 500, 1000):
        assert bytes(d[i].cpu().numpy().tobytes()) == hashlib.sha256(TS.POS_LEAF_TAG + (64).to_bytes(8, "big") + sc[64 * i: 64 * i + 64]).digest()
    lv = fold_level_direct(d)                      # 1001 -> 500 pairs + 1 lift
    assert lv.shape == (501, 32)
    assert torch.equal(lv, TS._fold_level_triton(d))
    dc = d.cpu().numpy().tobytes()
    assert bytes(lv[0].cpu().numpy().tobytes()) == hashlib.sha256(TS.NODE_PREFIX + dc[:32] + TS._U64_32 + dc[32:64]).digest()
    assert bytes(lv[500].cpu().numpy().tobytes()) == hashlib.sha256(TS.LIFT_PREFIX + dc[32000:32032]).digest()
