"""difftest adapter for `dense/AttentionSoftcap@v1`: production = vLLM's FA2 `flash_attn_varlen_func(..., softcap=CAP, fa_version=2,
num_splits=1)` on a paged bf16 KV cache (block 16), one query token (`seqlen_q = 1`, the decode launch) over T keys, causal — the
launch Gemma-2 makes for every decode step (`flash_fwd_splitkv_kernel<..., Is_softcap=true>`).

`difftest check` evaluates the Verity body with `MufuTanh` reading the in-tree measured table (quarantine/dense/tables/mufu_tanh_sm89,
sm_89 / driver 580.178.04), so both halves run on the worker (`produce` in the vLLM venv, `check` in venv-py312).  Shapes: Gemma-2-2B
geometry (NH 8, KVH 4, D 256, CAP 50) at T = 5 (one partial 64-key block), 65 (two blocks, second partial), 64 and 130, plus a head-dim-128
/ CAP-30 case so the CAP static is exercised; the order is such that `--n 5` covers T = 5 with every edge kind, `--n 15` adds the CAP-30
and the two-block cases (the reference interpreter is O(NH·T·D) Python nodes per case, so T is kept small here; model-scale runs go
through the capture patterns, not this harness)."""
from __future__ import annotations

from verity_vllm.registry.quarantine.dense._difftest_common import KINDS

SEMANTIC_ID = "dense/AttentionSoftcap@v1"
GEOMETRIES = (  # (T, NH, KVH, D, CAP)
    (5, 8, 4, 256, 50.0), (17, 4, 2, 128, 30.0), (65, 8, 4, 256, 50.0), (64, 8, 4, 256, 50.0), (130, 8, 4, 256, 50.0),
)
BLOCK = 16  # paged KV cache block size (vLLM default; irrelevant to the arithmetic, `num_splits = 1`)


def cases(rng, i: int) -> dict:
    T, NH, KVH, D, CAP = GEOMETRIES[(i // len(KINDS)) % len(GEOMETRIES)]
    kind = KINDS[i % len(KINDS)]
    return {"statics": {"T": T, "NH": NH, "KVH": KVH, "D": D, "CAP": CAP},
            "inputs": [{"shape": [NH, D], "dtype": "bfloat16", "kind": kind},
                       {"shape": [T, KVH, D], "dtype": "bfloat16", "kind": kind},
                       {"shape": [T, KVH, D], "dtype": "bfloat16", "kind": kind}],
            "edge": kind != "random"}


def production(statics: dict, inputs: list) -> list:
    import torch
    from vllm.vllm_flash_attn import flash_attn_varlen_func

    q, kc, vc = inputs  # q [NH, D], kc / vc [T, KVH, D] bf16 on cuda
    T, KVH, D, CAP = int(statics["T"]), int(statics["KVH"]), int(statics["D"]), float(statics["CAP"])
    nblocks = (T + BLOCK - 1) // BLOCK
    kpad = torch.zeros((nblocks * BLOCK, KVH, D), dtype=torch.bfloat16, device=q.device)
    vpad = torch.zeros_like(kpad)
    kpad[:T] = kc
    vpad[:T] = vc
    qt = q.reshape(1, q.shape[0], D).contiguous()
    out = torch.empty_like(qt)
    flash_attn_varlen_func(
        q=qt, k=kpad.view(nblocks, BLOCK, KVH, D), v=vpad.view(nblocks, BLOCK, KVH, D), out=out,
        cu_seqlens_q=torch.tensor([0, 1], dtype=torch.int32, device=q.device), max_seqlen_q=1,
        seqused_k=torch.tensor([T], dtype=torch.int32, device=q.device), max_seqlen_k=T,
        softmax_scale=float(D ** -0.5), causal=True, alibi_slopes=None, window_size=None,
        block_table=torch.arange(nblocks, dtype=torch.int32, device=q.device).reshape(1, -1),
        softcap=CAP, scheduler_metadata=None, fa_version=2, q_descale=None, k_descale=None, v_descale=None, num_splits=1,
    )
    torch.cuda.synchronize()
    return [out.reshape(q.shape[0], D)]


def spec(statics: dict):
    from verity.ir.defs import bind
    from verity_vllm.registry.b1 import fa2_kblock_n
    from verity_vllm.registry.quarantine.dense.softcap import AttentionSoftcap
    D = int(statics["D"])
    return bind(AttentionSoftcap, T=int(statics["T"]), NH=int(statics["NH"]), KVH=int(statics["KVH"]), D=D, BN=fa2_kblock_n(D),
                CAP=float(statics["CAP"]))
