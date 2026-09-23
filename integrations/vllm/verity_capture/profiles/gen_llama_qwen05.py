"""[GEN llama] κ profile for Qwen2.5-0.5B (`Qwen/Qwen2.5-0.5B` @ `060db6499f32faf8b98477b0a26969ef7d8b9987`, checkpoint role QWEN05)
under the SAME pins as `vllm_d9105ea80_sm89_eager` (vLLM d9105ea80, torch 2.13.0+cu129, Triton 3.7.1, RTX 4090, eager,
`VLLM_BATCH_INVARIANT=1`, bf16).

Configuration-only: the Qwen2.5-1.5B profile (`vllm_d9105ea80_sm89_eager_qwen15`, M6) re-parameterized from the checkpoint's
config.json — `C = analytic.config_of(...)`, `PROFILE = build_profile(with_geometry(C))`.  `gen_llama_QWEN05.config.json`
beside this module is a byte-for-byte copy of the hub's `config.json` at the pinned revision (sha256 `479dcf0c…` = the manifest
entry's `config_sha256`; checked by `tests/test_gen_llama.py`).  It lives in the package because `record_v5/ship.sh` ships
`verity_capture/` to the pod but not `out/gen/`.

What the configuration derives (provenance in `PROVENANCE`; hand-written: repo id and pinned revision only): H=896 NL=24
NH=14 KVH=2 D=64 (`head_dim` is not in this config: H/NH, "derived") I=4864 V=151936, QKV=1152 GU=9728, EPS=1e-6,
ROPE_THETA=1e6, MAX_POS=32768, BIAS=True (the qwen2 family rule: vLLM's Qwen2Attention passes bias=True unconditionally, so
the fused QKV projection carries a bias -> the batch-invariant linear's separate `aten.add` -> `BiasAdd_v1{N=1152}` and the
`qkv_b` weight field), `tie_word_embeddings=True`, no rope scaling, and the FA2 static `BN = fa2_kblock_n(64) = 128` (head
dim 64: the SmolLM2 trait, not Qwen2.5-1.5B's 64 at head dim 128).  `verity_vllm.registry.b1.QWEN05_CONFIG` is the same
configuration written by hand on 2026-09-12; the test asserts the two agree on every static key.
"""

from __future__ import annotations

import json
from pathlib import Path

from .. import analytic
from .vllm_d9105ea80_sm89_eager import (build_profile, make_check_header, make_root_class, make_weight_field,  # noqa: F401
                                        with_geometry)

CASE = "QWEN05"
REPO_ID = "Qwen/Qwen2.5-0.5B"
REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"
PROFILE_ID = "vllm-d9105ea80-sm89-eager-bf16-qwen2.5-0.5b"
#: verbatim copy of the hub's config.json at REVISION (sha256 == checkpoints.json `config_sha256` of role QWEN05)
CONFIG_JSON = Path(__file__).with_name("gen_llama_QWEN05.config.json")
CONFIG_SHA256 = "479dcf0c5286339e41ad3992cd08ae88a467c4187587936248e2b7c96283484b"

HF_CONFIG: dict = json.loads(CONFIG_JSON.read_text())
C, PROVENANCE = analytic.config_of(HF_CONFIG, model=REPO_ID, revision=REVISION)
C = with_geometry(C)                  # BN = fa2_kblock_n(D)

# family facts this profile relies on, checked against the derived configuration (never the other way round)
assert C["family"] == "qwen2" and C["BIAS"] is True and C["tie_word_embeddings"] is True, C
assert C["rope_scaling"] is None, C["rope_scaling"]
assert C["D"] == 64 and C["BN"] == 128, (C["D"], C["BN"])
assert C["H"] % 64 == 0 and C["I"] % 64 == 0, "the pinned Gemm tiling (BLOCK_SIZE_K 64) needs K a multiple of 64"

weight_field = make_weight_field(C)
root_class = make_root_class(weight_field)
check_header = make_check_header(C)

PROFILE = build_profile(C, profile_id=PROFILE_ID, case=CASE,
                        model_note=f"Qwen2.5-0.5B @ {REVISION[:8]} (analytic.config_of(config.json)): H={C['H']} NL={C['NL']} NH={C['NH']} "
                                   f"KVH={C['KVH']} D={C['D']} I={C['I']} V={C['V']}, QKV bias (BiasAdd_v1{{{C['QKV']}}}), tied embeddings, "
                                   f"FA2 kBlockN {C['BN']}")
