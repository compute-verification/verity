"""[X-07, M-NOPROFILE] Configuration-derived κ profile for ANY dense Llama-architecture checkpoint role — no per-model module.

Why (R12 finding M-NOPROFILE, lane dC): Mistral-7B Build passed but Match said "no profile module declares case 'MISTRAL7B'".
The κ machinery was already configuration-only in the model dimension (`build_profile(analytic.config_of(config.json))`, D72) —
but a hand-authored MODULE per role was still required (`profiles.PROFILE_BY_CASE` / the `PROFILE.notes["case"]` scan), so the
profile was in effect keyed to the model NAME.  Daniel's rule: applicability comes from the selected kernels and the target facts,
not from an allowlist of names.  This module removes the name: `verity_capture.profiles.dense_<ROLE>` is materialised on import
(a `sys.meta_path` finder in `profiles/__init__.py`) from the role's HF `config.json` — every consumer that does
`importlib.import_module(f"verity_capture.profiles.{name}")` (resolve_log, compare, gates, run_config, fold, cb_a) works
unchanged, in a fresh process, with `--profile dense_MISTRAL7B`.

What decides applicability at Match time (unchanged, and the point):
  * the PATTERNS — first-match over the launched kernels with every constexpr / launch-geometry field pinned (`GemmLaunch`,
    `GemmLaunchTuned` for vLLM's tuned (N,K) table, `RmsNorm*`, `FlashFwdSplitKV{BN}`, …); a kernel the pins do not cover is
    `Unsupported`, never silently folded;
  * `check_header(C)` — the run's recorded HF config vs the derived `C` (H, NL, NH, KVH, D, I, V, EPS, theta, MAX_POS, bias, tied);
  * `weight_field(C)` — the Llama-architecture vLLM parameter names (`qkv_proj`, `o_proj`, `gate_up_proj`, `down_proj`,
    `input_layernorm`, `post_attention_layernorm`; `qkv_b` iff `C["BIAS"]`, `lm_head` iff untied) → `analytic.weights_type(C)`.
A checkpoint whose architecture is NOT this vocabulary (MoE routing, different param names, sliding window, softcap, …) is not
made to fit: its roots resolve as unknown / its kernels as Unsupported, and `supported_family(cfg)` refuses the derivation
fail-closed for the `model_type`s whose vLLM implementation is known to add ops outside the dense vocabulary.

Where the config comes from (`find_hf_config(role)`, first hit wins; the source is recorded in `PROFILE.notes["hf_config"]`):
  1. `profiles/hf_configs/<ROLE>.config.json` — a byte copy shipped with the package (`record_v5/ship.sh` ships `verity_capture/`,
     not `out/gen/`); its sha256 must equal the manifest entry's `config_sha256` when the manifest is present;
  2. the manifest entry's `local_path/config.json` (the downloaded snapshot on the pod);
  3. `$HF_HOME/hub/models--<repo>/snapshots/<revision>/config.json` (the hub cache);
  4. `data/hf_configs/<ROLE>.config.json` (laptop-side copies).
"""
from __future__ import annotations

import hashlib
import json
import os
import types
from pathlib import Path
from typing import Any

from .. import analytic
from ..resolver import Profile
from .gen_llama_patterns import ADA_TUNED_GEMM, with_tuned_gemm
from .vllm_d9105ea80_sm89_eager import (
    build_profile,
    make_check_header,
    make_root_class,
    with_geometry,
)

MODULE_PREFIX = "dense_"                    # verity_capture.profiles.dense_<ROLE>
PACKAGE_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = PACKAGE_ROOT / "manifests" / "checkpoints.json"
HF_CONFIG_DIR = Path(__file__).with_name("hf_configs")
OUT_HF_CONFIG_DIR = PACKAGE_ROOT / "data" / "hf_configs"

#: `model_type`s whose vLLM d9105ea80 implementation is the dense Llama-architecture vocabulary the shared patterns / weight map
#: cover (LlamaForCausalLM, Qwen2ForCausalLM, MistralForCausalLM — same kernels, same parameter names; Qwen2 adds the fused QKV
#: bias, a `C["BIAS"]` fact).  Anything else refuses fail-closed here (Gemma-2: embedding scale / softcap / sliding window;
#: Phi-3: fused `qkv_proj`/`gate_up_proj` under other names + LongRoPE; MoE families: routing) — those need their own mechanism
#: record, not a silently mis-fitted profile.
#: [R15 fp8] `qwen3` (Qwen3ForCausalLM): the same dense block plus the per-head q/k RMSNorm before RoPE -- `analytic.FAMILY_RULES["qwen3"]`
#: records it as `C["QK_NORM"] = "head"`, `dense_layer_weights_type` adds the `q_norm`/`k_norm` `[D]` fields, and the existing
#: `RMSNormTritonLaunch` pattern owns the launches (vLLM flattens `q.view(M, NH, D)` to `[M*NH, D]` rows for `_rms_norm_kernel`).
DENSE_FAMILIES: tuple[str, ...] = ("llama", "qwen2", "mistral", "qwen3")


def module_name(role: str) -> str:
    return f"{MODULE_PREFIX}{role}"


def role_of_module(name: str) -> str | None:
    """`dense_<ROLE>` -> ROLE (the suffix is a checkpoint role: upper-case letters / digits / underscores), else None."""
    short = name.rsplit(".", 1)[-1]
    if not short.startswith(MODULE_PREFIX):
        return None
    role = short[len(MODULE_PREFIX):]
    return role if role and all(c.isupper() or c.isdigit() or c == "_" for c in role) else None


def checkpoint_entry(role: str) -> dict[str, Any] | None:
    if not MANIFEST.exists():
        return None
    m = json.loads(MANIFEST.read_text())
    for e in m.get("checkpoints", []):
        if e.get("role") == role:
            return e
    return None


def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def find_hf_config(role: str) -> tuple[Path, str, dict[str, Any] | None] | None:
    """`(config_path, source, manifest_entry)` for a role, or None when no source has it (docstring order)."""
    entry = checkpoint_entry(role)
    shipped = HF_CONFIG_DIR / f"{role}.config.json"
    if shipped.exists():
        if entry and entry.get("config_sha256") and _sha256(shipped) != entry["config_sha256"]:
            raise ValueError(f"{shipped} sha256 {_sha256(shipped)[:12]} != manifest config_sha256 {entry['config_sha256'][:12]} for {role}")
        return shipped, "package:hf_configs", entry
    for legacy in sorted(HF_CONFIG_DIR.parent.glob(f"*_{role}.config.json")):      # gen_llama_<ROLE>.config.json etc. (earlier lanes' copies)
        return legacy, f"package:{legacy.name}", entry
    if entry:
        lp = Path(entry.get("local_path") or "") / "config.json"
        if entry.get("local_path") and lp.exists():
            return lp, "manifest:local_path", entry
        hf_home = os.environ.get("HF_HOME") or str(Path.home() / ".cache" / "huggingface")
        repo, rev = entry.get("repo"), entry.get("revision")
        if repo and rev:
            snap = Path(hf_home) / "hub" / f"models--{repo.replace('/', '--')}" / "snapshots" / rev / "config.json"
            if snap.exists():
                return snap, "hf_home:snapshot", entry
    out = OUT_HF_CONFIG_DIR / f"{role}.config.json"
    if out.exists():
        return out, "data/hf_configs", entry
    return None


def supported_family(cfg: dict[str, Any]) -> str | None:
    """None when the config's `model_type` is in the dense vocabulary, else the refusal reason."""
    fam = str(cfg.get("model_type", "")).lower()
    if fam not in DENSE_FAMILIES:
        return f"model_type {fam!r} is not in the dense Llama-architecture vocabulary {DENSE_FAMILIES} (needs its own mechanism record)"
    if cfg.get("sliding_window"):
        return f"sliding_window={cfg['sliding_window']!r}: windowed attention is outside the pinned FA2 splitkv patterns"
    if fam == "qwen3" and not cfg.get("head_dim"):
        return "qwen3 without head_dim in config.json: the per-head q/k norm width is not derivable"
    # [R15 fp8] a quantized checkpoint is served only under the ONE modelled scheme (block-scaled dynamic e4m3 FP8); anything else refuses
    why = analytic.quant_refusal(cfg.get("quantization_config"))
    if why:
        return why
    return None


def derive_config(cfg: dict[str, Any], *, repo: str | None, revision: str | None) -> tuple[dict[str, Any], dict[str, str]]:
    """`analytic.config_of` + FA2 geometry, with the dense-vocabulary structural checks (fail-closed)."""
    why = supported_family(cfg)
    if why:
        raise ValueError(why)
    C, prov = analytic.config_of(cfg, model=repo, revision=revision)
    C = with_geometry(C)
    if C["H"] % 64 or C["I"] % 64:
        raise ValueError(f"H={C['H']} I={C['I']}: the pinned Gemm tiling (BLOCK_SIZE_K 64) needs K a multiple of 64")
    return C, prov


def build_dense_profile(role: str, cfg: dict[str, Any], *, repo: str | None = None, revision: str | None = None,
                        source: str = "caller") -> Profile:
    """The κ profile of a dense role from its HF config: shared pins / patterns / weight map; per-model `C` only."""
    C, prov = derive_config(cfg, repo=repo, revision=revision)
    rev = (revision or "")[:8]
    # [R15 fp8] the datatype is part of the profile identity: a block-scaled FP8 checkpoint runs other kernels (per-token-group quant +
    # blockwise CUTLASS scaled GEMM instead of the bf16 persistent GEMM) and its record has other fields (e4m3 weights, f32 block scales)
    dtype_tag = f"fp8-block{C['FP8_BLOCK']}" if C.get("FP8_BLOCK") else "bf16"
    prof = build_profile(C, profile_id=f"vllm-d9105ea80-sm89-eager-{dtype_tag}-{role.lower()}", case=role,
                         model_note=f"{repo or role} @ {rev or '?'} (dense_generic: analytic.config_of(config.json)): H={C['H']} NL={C['NL']} "
                                    f"NH={C['NH']} KVH={C['KVH']} D={C['D']} I={C['I']} V={C['V']} family={C['family']} "
                                    f"{'QKV bias' if C['BIAS'] else 'no QKV bias'}, {'tied' if C['tie_word_embeddings'] else 'untied'} "
                                    f"embeddings, FA2 kBlockN {C['BN']}"
                                    + (f", per-head q/k RMSNorm (D={C['D']})" if C.get("QK_NORM") == "head" else "")
                                    + (f", block-scaled FP8 linears (e4m3 weights + f32 [N/{C['FP8_BLOCK']}][K/{C['FP8_BLOCK']}] scales, "
                                       f"dynamic per-token-group activations)" if C.get("FP8_BLOCK") else ""))
    if C.get("FP8_BLOCK"):
        prof.notes["fp8"] = {"scheme": "block-scaled dynamic e4m3 FP8 (vLLM Fp8LinearMethod -> CutlassFp8BlockScaledMMKernel)",
                             "weight_block_size": [C["FP8_BLOCK"], C["FP8_BLOCK"]], "activation_group": C["FP8_BLOCK"],
                             "kernels": ["_C.per_token_group_fp8_quant (per_token_group_quant_8bit_kernel)",
                                         "_C.cutlass_scaled_mm blockwise (cutlass_3x_gemm_fp8_blockwise<bf16>, sm90)"],
                             "env": ["VLLM_USE_DEEP_GEMM=0 (the DeepGEMM JIT path is not modelled)", "VLLM_USE_FLASHINFER_SAMPLER=0"],
                             "families": ["Fp8GroupQuant_v1{K,G}", "ScaledMmFp8Block_v1{K,N,G}"],
                             "quantization_config": C.get("quantization_config")}
    # the tuned-table specialization is a TARGET fact (vLLM's sm_89 batch-invariant GEMM table), not a model fact: it precedes the
    # base GemmLaunch for every dense role and owns only launches whose (N,K) is a table key (None otherwise)
    prof.patterns = with_tuned_gemm(prof.patterns)
    prof.notes["pins"]["gemm_tuned_ada"] = {f"N={n},K={k}": row for (n, k), row in ADA_TUNED_GEMM.items()}
    prof.notes["derived_by"] = "profiles.dense_generic [X-07]: no per-model module; applicability = patterns (kernels) + check_header (facts)"
    prof.notes["provenance"] = dict(prov)
    prof.notes["hf_config"] = {"source": source, "revision": revision,
                               "sha256": hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest() if source == "caller" else None}
    return prof


def make_module(fullname: str) -> types.ModuleType:
    """The module object behind `verity_capture.profiles.dense_<ROLE>` (PROFILE / C / PROVENANCE / check_header / root_class …)."""
    role = role_of_module(fullname)
    if role is None:
        raise ImportError(f"{fullname}: not a dense_<ROLE> profile name")
    hit = find_hf_config(role)
    if hit is None:
        raise ImportError(f"{fullname}: no HF config.json for role {role!r} (profiles/hf_configs/{role}.config.json, the manifest entry's "
                          f"local_path, $HF_HOME snapshot, or data/hf_configs) — ship the byte copy with the checkpoints.json entry")
    path, source, entry = hit
    cfg = json.loads(path.read_text())
    repo = (entry or {}).get("repo")
    revision = (entry or {}).get("revision")
    try:
        prof = build_dense_profile(role, cfg, repo=repo, revision=revision, source=source)
    except ValueError as e:
        raise ImportError(f"{fullname}: {e}") from e
    prof.notes["hf_config"].update({"path": str(path), "sha256": _sha256(path)})
    mod = types.ModuleType(fullname)
    mod.__doc__ = f"[X-07] configuration-derived dense κ profile for role {role} ({source}: {path})"
    mod.CASE, mod.ROLE, mod.REPO_ID, mod.REVISION = role, role, repo, revision
    mod.CONFIG_JSON, mod.CONFIG_SHA256, mod.HF_CONFIG = path, prof.notes["hf_config"]["sha256"], cfg
    mod.C, mod.PROVENANCE = prof.config, prof.notes["provenance"]
    mod.PROFILE_ID, mod.PROFILE = prof.id, prof
    mod.weight_field = prof.weight_field
    mod.root_class = make_root_class(prof.weight_field)
    mod.check_header = make_check_header(prof.config)
    mod.__file__ = __file__
    return mod
