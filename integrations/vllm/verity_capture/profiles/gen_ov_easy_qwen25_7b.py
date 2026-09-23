"""[GEN ov-easy] κ profile for Qwen2.5-7B (`Qwen/Qwen2.5-7B` @ `d149729398750b98c0af14eb82c78cfe92750796`; checkpoint case `B7`,
the contract's 7B dense checkpoint pinned in `manifests/checkpoints.json` since 2026-09-07 — `verity_vllm_adapter.profile.CASES`
maps the case to that entry, so no manifest change) under the SAME pins as `vllm_d9105ea80_sm89_eager` (vLLM d9105ea80, torch
2.13.0+cu129, Triton 3.7.1, RTX 4090, eager, `VLLM_BATCH_INVARIANT=1`, bf16).

Configuration-only in the model dimension: `C = analytic.config_of(<the checkpoint's config.json>)`, `PROFILE = build_profile(
with_geometry(C))` — the Qwen2.5-1.5B (M6) profile re-parameterized.  `gen_ov_easy_B7.config.json` beside this module is a
byte-for-byte copy of the hub's `config.json` at the pinned revision (sha256 `267ce685…` = the manifest entry's `config_sha256`;
checked by `tests/test_gen_ov_easy.py`).  It lives in the package because `record_v5/ship.sh` ships `verity_capture/` to the pod
but not `out/gen/`.

What the configuration derives (provenance in `PROVENANCE`; hand-written: repo id and pinned revision only): H=3584 NL=28 NH=28
KVH=4 D=128 (`head_dim` is not in this config: H/NH, "derived") I=18944 V=152064, QKV=4608 GU=37888, EPS=1e-6, ROPE_THETA=1e6,
MAX_POS=131072, BIAS=True (the qwen2 family rule -> `BiasAdd_v1{N=4608}` and the `qkv_b` weight field), no rope scaling, and the
FA2 static `BN = fa2_kblock_n(128) = 64` (head dim 128: Qwen2.5-1.5B's trait, not SmolLM2's 128).

What is NEW relative to every accepted profile so far: **`tie_word_embeddings = False`**.  The output projection is a weight root
of its own — `analytic.weights_type(C)` carries the `lm_head` field `[V][H]`, the base `make_weight_field(C)` maps
`lm_head.weight` onto it (not onto `embed`) when `C` says untied, and `root_policy` proves the claim from the runtime (a separate
`lm_head.weight` parameter with its own storage; a tied checkpoint has none).  The base module's `make_check_header` was written
for the tied B0/B1 and flags EVERY untied header ("this profile maps lm_head.weight onto the embed field") regardless of `C`; this
module's `check_header` (below) replaces that message with the exact check the base omits — the recorded HF config's
`tie_word_embeddings` must equal `C["tie_word_embeddings"]` — and keeps every other base check.

GEMM tiling: the (N, K) of this model's five GEMMs — qkv (4608, 3584), o_proj (3584, 3584), gate_up (37888, 3584), down
(3584, 18944), lm_head (152064, 3584) — are not keys of vLLM's tuned sm_89 table (`batch_invariant_configs`, keys
(12288|4096|151936|2048, 2048) and (2048, 6144)), so every launch takes the default tiling the base `GemmLaunch` pins (asserted in
the tests against `gen_llama_patterns.tuned_launch_config` over a verbatim copy of the table's key set).  If the capture shows
otherwise it is a finding, not a matcher to widen.

Memory: 7.6 B bf16 parameters = 15.2 GB on a 24 GB RTX 4090; the run passes `--engine-arg gpu_memory_utilization=0.95` (recorded in
`run.json` / `versions.json` / the profile manifest) with the workload's 4096 `max_model_len`.

Analytic reference (diagnostic): `compare.py`'s structure section matches the derived body against the registered TIED family
`Serve_v4` (its `Final` reads `weights["embed"]`), which cannot be isomorphic to an untied program -- the B7 structure check fails
at the first lm_head Gemm while tokens / boundary / subcircuits and every gate pass.  `gen_ov_easy_family.ServeUntied_v1` is the
untied reference; `SERVE_FAMILY` below is the hook a SHARED-CODE change to `compare.main` would call (`SERVE_FAMILY(C, LP=, STEPS=,
BN=)` -> the bound family); the fork point ignores it.
"""

from __future__ import annotations

import json
from pathlib import Path

from .. import analytic
from .gen_ov_easy_family import serve_family
from .vllm_d9105ea80_sm89_eager import (
                                        build_profile,
                                        make_check_header,
                                        make_root_class,
                                        make_weight_field,
                                        with_geometry,
)

CASE = "B7"
REPO_ID = "Qwen/Qwen2.5-7B"
REVISION = "d149729398750b98c0af14eb82c78cfe92750796"
PROFILE_ID = "vllm-d9105ea80-sm89-eager-bf16-qwen2.5-7b"
#: verbatim copy of the hub's config.json at REVISION (sha256 == checkpoints.json `config_sha256` of the Qwen/Qwen2.5-7B entry)
CONFIG_JSON = Path(__file__).with_name("gen_ov_easy_B7.config.json")
CONFIG_SHA256 = "267ce68584c5f24c3b267d934db2de68dd21d1ca677fb78ed809eb60067f7642"
#: the engine argument the 24 GB card needs for 15.2 GB of weights (the base engine default is 0.80 = 19.6 GB, too tight with the
#: profiling run's activations); every GPU stage of `run_config.py` receives it through `--engine-arg`
ENGINE_ARGS = {"gpu_memory_utilization": 0.95}

HF_CONFIG: dict = json.loads(CONFIG_JSON.read_text())
C, PROVENANCE = analytic.config_of(HF_CONFIG, model=REPO_ID, revision=REVISION)
C = with_geometry(C)                  # BN = fa2_kblock_n(D)

# family facts this profile relies on, checked against the derived configuration (never the other way round)
assert C["family"] == "qwen2" and C["BIAS"] is True and C["tie_word_embeddings"] is False, C
assert C["rope_scaling"] is None, C["rope_scaling"]
assert C["D"] == 128 and C["BN"] == 64, (C["D"], C["BN"])
assert C["H"] % 64 == 0 and C["I"] % 64 == 0, "the pinned Gemm tiling (BLOCK_SIZE_K 64) needs K a multiple of 64"
assert [f for f, _ in analytic.weights_type(C).fields] == ["embed", "layers", "norm", "cos_sin", "lm_head"], "untied: lm_head is a field"

weight_field = make_weight_field(C)
root_class = make_root_class(weight_field)
#: [X-07] the base `make_check_header` now compares the recorded `tie_word_embeddings` to `C` (it used to reject every untied
#: config with a tied-only message this module filtered out and replaced); the exact check lives in one place.
check_header = make_check_header(C)


PROFILE = build_profile(C, profile_id=PROFILE_ID, case=CASE,
                        model_note=f"Qwen2.5-7B @ {REVISION[:8]} (analytic.config_of(config.json)): H={C['H']} NL={C['NL']} NH={C['NH']} "
                                   f"KVH={C['KVH']} D={C['D']} I={C['I']} V={C['V']}, QKV bias (BiasAdd_v1{{{C['QKV']}}}), UNTIED lm_head "
                                   f"(weight root of its own), FA2 kBlockN {C['BN']}; engine args {ENGINE_ARGS}")
PROFILE.notes["engine_args"] = dict(ENGINE_ARGS)
PROFILE.notes["untied_lm_head"] = ("C[tie_word_embeddings]=False from config.json: lm_head.weight -> weights.lm_head (analytic.weights_type); "
                                   "root_policy proves the runtime kept a separate parameter; check_header requires the recorded HF config to agree")
PROFILE.notes["serve_family"] = ("diagnostic: the registered Serve_v4 is tied (Final reads weights.embed) so compare's structure section fails for "
                                 "this model at the first lm_head Gemm; gen_ov_easy_family.ServeUntied_v1 is the untied reference (SERVE_FAMILY hook)")
#: the analytic family `compare.compare_structure` should bind for this configuration (`ServeUntied_v1{C, LP, STEPS, BN}`); read by
#: nothing at the fork point -- the SHARED-CODE proposal in gen_ov_easy_family's docstring makes `compare.main` call it
SERVE_FAMILY = serve_family
