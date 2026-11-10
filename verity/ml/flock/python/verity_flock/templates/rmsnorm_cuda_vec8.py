"""Flock lowering of ``rmsnorm-cuda-vec8`` (RMSNormCudaVector8_v1, vLLM's served width-8 ``rms_norm_kernel``; GLM-5.3's
hidden 6144): ``rmsnorm-fused-cuda-vec8``'s cut and units (``rmsnorm_fused_cuda_vec8``) on the kernel without the residual. A
thread's partial is one FMA chain over its elements from ``x0 * x0``, a warp reads x and w at its elements, and the launch's
BLOCK is at most N / 8 (GLM-5.3: 24 warps of one vector at BLOCK 768, three a unit; 8 warps of three vectors at BLOCK 256).
The unit netlist is pinned per (N, EPS, BLOCK) by :data:`PINS`; like the fused kernel's, it is proved under
``verity/flock-circuit`` only."""
from verity_flock.templates import rmsnorm_fused_cuda_vec8 as _fused

TEMPLATE = "rmsnorm-cuda-vec8"
FAMILY = "RMSNormCudaVector8_v1"
UNIT = "rmsnorm-cuda-vec8/bf16/warps"
FUSED = False
PINS: dict[tuple[int, float, int], str] = {
    (6144, 1e-05, 768): "99450d152d1f7b873818ab34274db51bfcc6de20e936c31abc51c0bb745c235b",
    (6144, 1e-05, 256): "a643807263670eceb928209cec9fe7d7dab7f8e197cfc5a572b71e72883f467e",
}
SCHEMES = _fused.SCHEMES
warps = _fused.warps


def cut(n: int, block: int) -> tuple[int, int] | None:
    return _fused.cut(n, block, FUSED)


def definition(subcircuit):
    return subcircuit.definition()


def lowering(subcircuit):
    return _fused.lowering(subcircuit, FUSED, UNIT)


def circuit_lowering(subcircuit):
    return lowering(subcircuit)


def supports(subcircuit, scheme):
    return _fused.supports(subcircuit, scheme, FAMILY, UNIT)


def relation(subcircuit, scheme):
    return _fused.relation(subcircuit, scheme)
