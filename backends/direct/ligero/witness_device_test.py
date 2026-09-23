"""The fused witness kernel (witness_device.py) equals the torch program (witness._run_program) value for value, and the
CUDA-graph capture of the torch hint generator succeeds (hints_torch._Graph) -- lane hostphase.  CUDA only."""
import numpy as np
import pytest
import torch

from . import witness as witness_mod
from .relations import relation
from .relchain import RelationChainRunner, instances, vu_units

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA")


def _sub_batch(rel_name: str, n_vus: int = 6):
    rel = relation(rel_name)
    R = RelationChainRunner(rel, "cuda", -128.0, 2, n_proofs=1, zk=False, mode="fiat-shamir")
    data = instances(rel, n_vus)
    lay = R.layout(n_vus)
    units = []
    for a, b, accs, y in data:
        units += vu_units(rel, a, b, accs)
    while len(units) < lay.l:
        units.append(R._pad_unit())
    a = np.asarray([u[1] for u in units], dtype=np.int64)
    b = np.asarray([u[2] for u in units], dtype=np.int64)
    c_in = np.asarray([u[0] for u in units], dtype=np.int64)
    y_out = np.asarray([u[3] for u in units], dtype=np.int64)
    pub = rel.public_vectors(R.p, None, a, b, None, device="cuda")
    hints = rel.hints(R.p, R.sys, c_in, a, b, y_out, "cuda")
    return R, pub, hints


@pytest.mark.parametrize("rel_name", ["bf16-hopper", "fp8-hopper"])
def test_fused_witness_equals_torch_program(rel_name):
    from . import witness_device
    R, pub, hints = _sub_batch(rel_name)
    f = witness_device.fused_for(R.sys, "cuda")
    assert f is not None, "NVRTC / cupy unavailable"
    old = witness_mod.IMPL
    try:
        witness_mod.IMPL = "legacy"
        W_ref = witness_mod.build_witness(R.sys, pub, hints, "cuda")
        witness_mod.IMPL = "device"
        W_dev = witness_mod.build_witness(R.sys, pub, hints, "cuda")
    finally:
        witness_mod.IMPL = old
    assert W_dev.dtype == torch.int32 and W_dev.shape == W_ref.shape
    assert torch.equal(W_dev.to(torch.int64), W_ref)


def test_hints_graph_captures():
    from . import hints_torch
    if not hints_torch.USE_CUDA_GRAPH:
        pytest.skip("graphs off")
    R, _pub, _hints = _sub_batch("bf16-hopper")
    key = next((k for k, g in hints_torch._GRAPHS.items() if k[0] == id(R.sys)), None)
    assert key is not None, "the hint generator did not try to capture"
    assert hints_torch._GRAPHS[key], "the hint generator's CUDA-graph capture failed (fell back to eager)"
