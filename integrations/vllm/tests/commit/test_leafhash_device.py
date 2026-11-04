"""The device weights registration produces the SAME per-tensor roots as the host path -- byte for byte -- over sizes with
and without a shorter tail leaf, single-leaf tensors, odd leaf counts (lift rule), both chunk sizes the committer uses (256 GPU-tree, 4096),
and every dtype view the model carries.  CUDA + a compiler required (the extension is JIT-built); skipped elsewhere."""
from __future__ import annotations

import hashlib

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA required for the device leaf hash", allow_module_level=True)

from verity_vllm.commit.committer import leafhash as LH  # noqa: E402
from verity_vllm.commit import scheme  # noqa: E402


@pytest.fixture(scope="module")
def ext():
    try:
        return LH.ext()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"leafhash extension did not build here: {e!r}"[:300])


SIZES = [1, 7, 255, 256, 257, 511, 512, 513, 4096, 4096 + 13, 3 * 256 * 1000 + 7, (1 << 20) + 100]


@pytest.mark.parametrize("nbytes", SIZES)
@pytest.mark.parametrize("chunk", [256, 4096])
def test_device_root_equals_host_root_bytes(ext, nbytes, chunk):
    g = torch.Generator().manual_seed(nbytes * 7919 + chunk)
    t = torch.randint(0, 256, (nbytes,), dtype=torch.uint8, generator=g).cuda()
    assert LH.tensor_root_device(t, chunk) == LH.tensor_root_host(t, chunk)


@pytest.mark.parametrize("shape,dtype", [((129,), torch.bfloat16), ((2048, 3), torch.float16), ((1000,), torch.float32), ((), torch.float32),
                                         ((17, 33), torch.int8), ((4, 128, 5), torch.bfloat16)])
def test_dtype_views_hash_like_the_host_path(ext, shape, dtype):
    t = (torch.randn(shape) * 3).to(dtype).cuda() if dtype.is_floating_point else torch.randint(-100, 100, shape, dtype=dtype).cuda()
    assert LH.tensor_root_device(t, 256) == LH.tensor_root_host(t, 256)


@pytest.mark.parametrize("n", [1, 2, 3, 5, 8, 9, 1001])
def test_tree_root_is_the_scheme_fold(ext, n):
    leaves = [hashlib.sha256(bytes([i % 251]) * 40).digest() for i in range(n)]
    dev = ext.tree_root(torch.tensor(list(b"".join(leaves)), dtype=torch.uint8).reshape(n, 32).cuda())
    assert bytes(dev.cpu().numpy().tobytes()) == scheme.fold(list(leaves))


def test_pos_leaves_are_the_semantic_pos_leaf(ext):
    t = torch.randint(0, 256, (256 * 5 + 100,), dtype=torch.uint8).cuda()
    dev = ext.pos_leaves(t, 256).cpu().numpy().tobytes()
    mv = memoryview(t.cpu().numpy()).cast("B")
    host = b"".join(scheme.pos_leaf(mv[o:o + 256]) for o in range(0, len(mv), 256))
    assert dev == host and len(dev) == 6 * 32


def test_empty_and_non_cuda_inputs_are_refused(ext):
    with pytest.raises(RuntimeError):
        ext.pos_leaves(torch.empty(0, dtype=torch.uint8).cuda(), 256)
    with pytest.raises(RuntimeError):
        ext.pos_leaves(torch.zeros(10, dtype=torch.uint8), 256)
