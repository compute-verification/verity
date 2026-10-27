"""[M6] κ profile for Qwen2.5-1.5B (`b1.B1_CONFIG`, checkpoint role B1 @ `8faed761d45a263340a0528343f099c05c9a4323`) under
the SAME pins as `vllm_d9105ea80_sm89_eager` (vLLM d9105ea80, torch 2.13.0+cu129, Triton 3.7.1, RTX 4090, eager,
`VLLM_BATCH_INVARIANT=1`, bf16).  Nothing model-level is authored here: the profile is `build_profile(B1_CONFIG)`.

What changes with the configuration (all derived from `C`): H=1536 NL=28 NH=12 KVH=2 D=128 I=8960 V=151936, QKV=2048 GU=17920,
EPS=1e-6, ROPE_THETA=1e6, MAX_POS=131072, `BIAS=True` (Qwen2's fused QKV projection carries a bias -> the batch-invariant
linear's separate `aten.add` -> `BiasAdd_v1{N=2048}`, the `qkv_b` weight field), and the FA2 static `BN = fa2_kblock_n(128) = 64`
(the head-dim-128 split-KV trait `Flash_fwd_kernel_traits<128, 64, 64, 4>`; `b1.fa2_kblock_n`).
"""

from __future__ import annotations

from verity.ml.definitions.registry import b1

from verity_vllm.observe.fold.profiles.vllm_d9105ea80_sm89_eager import (build_profile, make_check_header, make_root_class, make_weight_field,
                                        with_geometry)

PROFILE_ID = "vllm-d9105ea80-sm89-eager-bf16-qwen2.5-1.5b"
CASE = "B1"

C = with_geometry(b1.B1_CONFIG)      # BN = 64
assert C["BN"] == 64 and C["BIAS"] is True and C["D"] == 128, C

weight_field = make_weight_field(C)
root_class = make_root_class(weight_field)
check_header = make_check_header(C)

PROFILE = build_profile(C, profile_id=PROFILE_ID, case=CASE,
                        model_note="Qwen2.5-1.5B @ 8faed761 (b1.B1_CONFIG): H=1536 NL=28 NH=12 KVH=2 D=128 I=8960 V=151936, "
                                   "QKV bias (BiasAdd_v1{2048}), FA2 kBlockN 64")
