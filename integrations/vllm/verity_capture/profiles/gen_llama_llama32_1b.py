"""[GEN llama] κ profile for Llama-3.2-1B (`unsloth/Llama-3.2-1B` @ `9535bd9b1d1dea6acafbdc4813b728796aeb28da`, the ungated mirror
of meta-llama/Llama-3.2-1B; checkpoint role LLAMA32_1B) under the SAME pins as `vllm_d9105ea80_sm89_eager` (vLLM d9105ea80,
torch 2.13.0+cu129, Triton 3.7.1, RTX 4090, eager, `VLLM_BATCH_INVARIANT=1`, bf16).

Configuration-only in the model dimension: `C` is `analytic.config_of(<the checkpoint's config.json>)` and the profile is the
base module's `build_profile(with_geometry(C))` — the Llama-architecture weight map, pattern list and κ pins are the shared ones
(SmolLM2-135M is the same `LlamaForCausalLM` vocabulary).  ONE specialization was forced by the capture (finding, see
`gen_llama_patterns.py`): vLLM's batch-invariant GEMM launches the o_proj `[M,2048] x [2048,2048]` with the tiling of its tuned
sm_89 table for the key (N,K)=(2048,2048) — `BLOCK_SIZE_M 16/32, num_stages 5` — instead of the pinned default, so
`GemmLaunchTuned` (pinned to that table row, same `Gemm_v1{K,N}` body) precedes the base `GemmLaunch` in this profile's pattern
list.  `gen_llama_LLAMA32_1B.config.json` beside this module
is a byte-for-byte copy of the hub's `config.json` at the pinned revision (sha256 `8f028e2c…` = the manifest entry's
`config_sha256`; `tests/test_gen_llama.py` checks the bytes and the dict against `data/hf_configs/LLAMA32_1B.config.json`).
It lives in the package because `record_v5/ship.sh` ships `verity_capture/` to the pod but not `out/gen/`.

What the configuration derives (every value read from config.json, provenance in `PROVENANCE`; nothing hand-written but the
repo id and the pinned revision): H=2048 NL=16 NH=32 KVH=8 D=64 (`head_dim` is explicit in this config) I=8192 V=128256,
QKV=3072 GU=16384, EPS=1e-5, ROPE_THETA=5e5, MAX_POS=131072, BIAS=False (`attention_bias`, the llama family rule),
`tie_word_embeddings=True` (the runtime's alias of `lm_head.weight` onto the embedding is proven by the root policy, not
assumed here), `rope_scaling` of type llama3 (factor 32, low/high freq factors 1/4, original max position 8192 — the
`analytic.cos_sin_table` rule the root policy checks the captured `cos_sin_cache.npy` against), and the FA2 static
`BN = fa2_kblock_n(64) = 128` (the head-dim-64 split-KV trait, as SmolLM2).
"""

from __future__ import annotations

import json
from pathlib import Path

from .. import analytic
from .gen_llama_patterns import ADA_TUNED_GEMM, with_tuned_gemm
from .vllm_d9105ea80_sm89_eager import (build_profile, make_check_header, make_root_class, make_weight_field,  # noqa: F401
                                        with_geometry)

CASE = "LLAMA32_1B"
REPO_ID = "unsloth/Llama-3.2-1B"
REVISION = "9535bd9b1d1dea6acafbdc4813b728796aeb28da"
PROFILE_ID = "vllm-d9105ea80-sm89-eager-bf16-llama3.2-1b"
#: verbatim copy of the hub's config.json at REVISION (sha256 == checkpoints.json `config_sha256` of role LLAMA32_1B)
CONFIG_JSON = Path(__file__).with_name("gen_llama_LLAMA32_1B.config.json")
CONFIG_SHA256 = "8f028e2cd88148fad38c7dece460947681adf6178e46e11ece9742aa49b378dd"

HF_CONFIG: dict = json.loads(CONFIG_JSON.read_text())
C, PROVENANCE = analytic.config_of(HF_CONFIG, model=REPO_ID, revision=REVISION)
C = with_geometry(C)                  # BN = fa2_kblock_n(D)

# family facts this profile relies on, checked against the derived configuration (never the other way round)
assert C["family"] == "llama" and C["BIAS"] is False and C["tie_word_embeddings"] is True, C
assert isinstance(C["rope_scaling"], dict) and C["rope_scaling"].get("rope_type") == "llama3", C["rope_scaling"]
assert C["D"] == 64 and C["BN"] == 128, (C["D"], C["BN"])
assert C["H"] % 64 == 0 and C["I"] % 64 == 0, "the pinned Gemm tiling (BLOCK_SIZE_K 64) needs K a multiple of 64"

weight_field = make_weight_field(C)
root_class = make_root_class(weight_field)
check_header = make_check_header(C)

PROFILE = build_profile(C, profile_id=PROFILE_ID, case=CASE,
                        model_note=f"Llama-3.2-1B @ {REVISION[:8]} (analytic.config_of(config.json)): H={C['H']} NL={C['NL']} NH={C['NH']} "
                                   f"KVH={C['KVH']} D={C['D']} I={C['I']} V={C['V']}, no QKV bias, tied embeddings, rope_scaling llama3 "
                                   f"(factor {C['rope_scaling']['factor']}), FA2 kBlockN {C['BN']}; o_proj GEMM (N,K)=(2048,2048) launched "
                                   f"with vLLM's tuned sm_89 tiling -> gen_llama_patterns.GemmLaunchTuned")
# the tuned-table specialization precedes the base GemmLaunch (first-match resolver); same Gemm_v1 body, table-pinned tiling
PROFILE.patterns = with_tuned_gemm(PROFILE.patterns)
PROFILE.notes["pins"]["gemm_tuned_ada"] = {f"N={n},K={k}": row for (n, k), row in ADA_TUNED_GEMM.items()}
PROFILE.notes["specializations"] = ["gen_llama_patterns.GemmLaunchTuned: matmul_kernel_persistent launches whose (N,K) is a key of vLLM "
                                    "batch_invariant_configs' sm_89 table take that row's M-bucket tiling (BLOCK_SIZE_M/N, num_warps, "
                                    "num_stages) -> the same Gemm_v1{K,N}; here the o_proj (2048,2048)"]
