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


@pytest.mark.parametrize("rel_name", ["bf16-hopper", "fp8-ada"])
def test_interpreter_kernel_equals_torch_program(rel_name):
    """The program-independent interpreter kernel (witness_device.INTERP_OPS; lane blake3-leaf, for the 38k-op BLAKE3
    leaf systems) forced on a small system: value for value the torch program, and the same W as the codegen kernel."""
    from . import witness_device
    R, pub, hints = _sub_batch(rel_name)
    old_ops, old_impl = witness_device.INTERP_OPS, witness_mod.IMPL
    try:
        witness_device.INTERP_OPS = 0
        f = witness_device.FusedWitness(R.sys, "cuda")
        assert f.interp
        pub_rows = torch.stack([pub[name].to("cuda") for _, name in R.sys.pins]) % witness_mod.P
        W_int = f.run(pub_rows, hints)
    finally:
        witness_device.INTERP_OPS = old_ops
    try:
        witness_mod.IMPL = "legacy"
        W_ref = witness_mod.build_witness(R.sys, pub, hints, "cuda")
    finally:
        witness_mod.IMPL = old_impl
    assert W_int.shape == W_ref.shape
    assert torch.equal(W_int.to(torch.int64), W_ref)


@pytest.mark.parametrize("rel_name", ["bf16-hopper", "fp8-ada"])
def test_level_interpreter_equals_sequential_interpreter(rel_name):
    """The level-scheduled interpreter (witness_device.INTERP_LEVELS; lane blake3-leaf-3) produces the sequential
    interpreter's W, which the test above ties to the torch program."""
    from . import witness_device
    R, pub, hints = _sub_batch(rel_name)
    old = witness_device.INTERP_OPS, witness_device.INTERP_LEVELS
    Ws = []
    try:
        witness_device.INTERP_OPS = 0
        pub_rows = torch.stack([pub[name].to("cuda") for _, name in R.sys.pins]) % witness_mod.P
        for levels in (False, True):
            witness_device.INTERP_LEVELS = levels
            f = witness_device.FusedWitness(R.sys, "cuda")
            assert f.interp and bool(f.levels) == levels
            Ws.append(f.run(pub_rows, hints))
    finally:
        witness_device.INTERP_OPS, witness_device.INTERP_LEVELS = old
    assert torch.equal(Ws[0], Ws[1])


def test_fused_witness_equals_torch_program_fp4():
    """The FP4 relation's program (fp4/relation.py: 1583 rows/unit, chain-mode system) through the same fused kernel --
    the FP4 chain runner takes the shared ``witness.build_witness`` path, so ``--impl device`` covers its witness too."""
    from . import witness_device
    from .fp4.chain import _public_vectors_torch
    from .fp4.relation import NVFP4, compile_fp4_unit
    from .fp4.witness import FAMILIES, hints_fp4, random_units
    s = compile_fp4_unit(NVFP4, chain=True)
    rng = np.random.default_rng(11)
    cs, as_, bs = zip(*(random_units(rng, 8, fam) for fam in FAMILIES))
    c, a, b = np.concatenate(cs), np.concatenate(as_), np.concatenate(bs)
    pub = _public_vectors_torch(NVFP4, None, a, b, None, device="cuda")
    hints = torch.as_tensor(hints_fp4(s, c, a, b), dtype=torch.int64, device="cuda")
    f = witness_device.fused_for(s, "cuda")
    assert f is not None, "NVRTC / cupy unavailable"
    old = witness_mod.IMPL
    try:
        witness_mod.IMPL = "legacy"
        W_ref = witness_mod.build_witness(s, pub, hints, "cuda")
        witness_mod.IMPL = "device"
        W_dev = witness_mod.build_witness(s, pub, hints, "cuda")
    finally:
        witness_mod.IMPL = old
    assert W_dev.shape == W_ref.shape
    assert torch.equal(W_dev.to(torch.int64), W_ref)


@pytest.mark.parametrize("which", ["fp4-nvf4", "fp4-nvf4+poseidon2"])
def test_ptx_route_equals_native_build(which):
    """The compute_89 PTX route (``witness_device.ptx_arch``: the default on compute capability >= 12.0) computes the native
    build's W bit for bit.  W is a function of (pins, hints) alone, so arbitrary residues exercise every op."""
    import os

    import cupy as cp

    from . import witness_device
    if which == "fp4-nvf4":
        from .fp4.relation import NVFP4, compile_fp4_unit
        s = compile_fp4_unit(NVFP4, chain=True)
    else:
        if int(cp.cuda.Device().compute_capability) >= 120 and not os.environ.get("LIGERO_TEST_NATIVE_CC12"):
            pytest.skip("the native sm_120 build of the hashed kernel takes ~7 min cold (LIGERO_TEST_NATIVE_CC12=1 runs it)")
        from . import hashchain
        from .fp4.hashed import FP4_HASHED
        s = hashchain.compose(FP4_HASHED).sys
    g = torch.Generator(device="cuda").manual_seed(5)
    l = 512
    pub_rows = torch.randint(0, witness_device.P, (len(s.pins), l), generator=g, device="cuda", dtype=torch.int64)
    hints = torch.randint(0, witness_device.P, (len(s.hints), l), generator=g, device="cuda", dtype=torch.int64)
    native = witness_device.FusedWitness(s, "cuda", arch=None)
    ptx = witness_device.FusedWitness(s, "cuda", arch=witness_device.PTX_ARCH_CC12)
    W_native, W_ptx = native.run(pub_rows, hints), ptx.run(pub_rows, hints)
    assert W_native.shape == (s.m, l) and torch.equal(W_native, W_ptx)


def test_hints_graph_captures():
    from . import hints_torch
    if not hints_torch.USE_CUDA_GRAPH:
        pytest.skip("graphs off")
    R, _pub, _hints = _sub_batch("bf16-hopper")
    key = next((k for k, g in hints_torch._GRAPHS.items() if k[0] == id(R.sys)), None)
    assert key is not None, "the hint generator did not try to capture"
    assert hints_torch._GRAPHS[key], "the hint generator's CUDA-graph capture failed (fell back to eager)"
