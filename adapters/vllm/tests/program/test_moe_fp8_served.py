"""GLM-5.3's MoE on one TP8 rank of the H200 binds, at every launch from 1 to 8,192 tokens, only Definitions the sidecars bind at
those statics (``served-``): the expert GEMMs at the tile `devices.fused_moe_config` takes, SWAP from `enable_swap_ab` (BLOCK_SIZE_M
< 64 and BLOCK_SIZE_N >= 64) and DOT from `fp8_moe.sm90_dot`, and the router on the kernel `router_kernel` picks.  The match gate
can't tell the SWAP variants apart on a one-word block, so the tile rule is what binds them; a tile, SWAP or DOT the compiler
reaches without its binding, or a binding it never reaches, fails here by id."""
from __future__ import annotations

import importlib.util
from pathlib import Path

from verity.definitions.registry import fp8_moe_experts, moe
from verity_vllm.engine.profiles.devices import fused_moe_config
from verity_vllm.program.frontend import targets as TG
from verity_vllm.program.frontend.target_profile import TargetProfile
from verity_vllm.program.predict.common import kind

H200 = TargetProfile(compute_capability=(9, 0), num_sms=132, batch_invariant=False, moe_backend="triton")
#: one TP8 rank of GLM-5.3-FP8: experts, hidden, a rank's expert width, the FP8 block, top-k and the routed scaling factor
E, H, I, G, TOPK, SCALE = 256, 6144, 256, 128, 8, 2.5
LAUNCHES = range(1, 8193)


def _served(module) -> set[str]:
    path = Path(module.__file__).with_name(f"{Path(module.__file__).stem}.circuit-check.py")
    spec = importlib.util.spec_from_file_location(f"{path.stem}_sidecar", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return {make().id for label, make in mod._model_roots() if "/served-" in label}


def _compiled(M: int) -> tuple[str, str, str, tuple]:
    """The ids `predict.dsa`'s MoE block binds at an M-token launch (its statics as it writes them), and the tile."""
    cfg = fused_moe_config(E, I, "NVIDIA H200", "fp8_w8a8", (G, G), M)
    BM, BN, KB, warps = cfg["BLOCK_SIZE_M"], cfg["BLOCK_SIZE_N"], min(cfg["BLOCK_SIZE_K"], G), cfg["num_warps"]
    swap = BM < 64 and BN >= 64
    assert TG.moe_fp8_tuned_tile_refusal(H200, BM, BN, KB, warps, swap) is None
    st = {"E": E, "G": G, "KB": KB, "SWAP": swap, "DOT": TG.moe_expert_fp8_launch_dot(H200, BM, BN, warps, swap)}
    router = kind("moe_router_sigmoid_bias", E=E, TOPK=TOPK, RSF=SCALE, KERNEL=moe.router_kernel(E, M, TOPK))
    return (kind("moe_expert_gemm_dot", K=H, N=2 * I, **st).id, kind("moe_expert_gemm_w_dot", K=I, N=H, **st).id, router.id,
            (BM, BN, swap, st["DOT"].id))


def test_every_served_glm53_moe_binding_is_bound_at_its_statics():
    got = [_compiled(M) for M in LAUNCHES]
    gemms, routers = {i for c in got for i in c[:2]}, {c[2] for c in got}
    for module, want in ((fp8_moe_experts, gemms), (moe, routers)):
        bound = _served(module)
        assert bound == want, (f"bind each of these in {module.__name__}'s sidecar _model_roots, labelled '<family>/served-<what>', "
                               f"and drop the rest: missing {sorted(want - bound)}, never served {sorted(bound - want)}")


def test_the_served_tiles_take_both_swaps_on_both_steps():
    """vLLM's rule, not the tuned file, sets SWAP_AB: the 8x64 decode tile (4 to 6 tokens, as at a 4-token step) runs SWAP=True on
    MMAv2 and the 16x64 and 16x128 tiles (7 to 768) SWAP=True on wgmma; only 8x32 (1 to 3) and 64x128 (past 768) run unswapped."""
    tiles = {M: _compiled(M)[3] for M in (1, 3, 4, 6, 7, 12, 13, 768, 769, 8192)}
    mma, qgmma = "HopperE4m3MmaV2Dot32_v1", "HopperE4m3QgmmaDot32_v1"
    assert tiles == {1: (8, 32, False, mma), 3: (8, 32, False, mma), 4: (8, 64, True, mma), 6: (8, 64, True, mma),
                     7: (16, 64, True, qgmma), 12: (16, 64, True, qgmma), 13: (16, 128, True, qgmma), 768: (16, 128, True, qgmma),
                     769: (64, 128, False, qgmma), 8192: (64, 128, False, qgmma)}
