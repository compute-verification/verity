"""[X-07, M-NOPROFILE] Configuration-derived κ profile for ANY dense Llama-architecture checkpoint role — no per-model module.

[P6, 2026-09-23] The constructor moved to `profiles.generic` (every architecture class, `world > 1`, the family facts table); this module keeps
the HF config.json lookup (`find_hf_config`, `checkpoint_entry`), the dense vocabulary (`DENSE_FAMILIES`, `supported_family`) and the X-07
module name `dense_<ROLE>` (`role_of_module`), and delegates `build_dense_profile` / `make_module` to `generic`.

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
from .vllm_d9105ea80_sm89_eager import with_geometry

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
    """The manifest entry of a role: `role == case`, else (the contract's B0 / B1 / B7 / SMOL360 cases, whose manifest `role` is prose) the
    entry of the repo `verity_vllm_adapter.profile.CASES[case]` names."""
    if not MANIFEST.exists():
        return None
    m = json.loads(MANIFEST.read_text())
    for e in m.get("checkpoints", []):
        if e.get("role") == role:
            return e
    try:
        from verity_vllm_adapter.profile import CASES
    except Exception:                                   # pragma: no cover - the adapter is optional off-pod
        return None
    repo = (CASES.get(role) or {}).get("repo")
    if repo:
        for e in m.get("checkpoints", []):
            if e.get("repo") == repo:
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
    """[P6] `generic.build_profile(role, hf_config=cfg)` after the dense-vocabulary structural checks (fail-closed)."""
    from . import generic
    derive_config(cfg, repo=repo, revision=revision)
    return generic.build_profile(role, 1, hf_config=cfg, repo=repo, revision=revision)


def make_module(fullname: str) -> types.ModuleType:
    """[P6] `verity_capture.profiles.dense_<ROLE>` -> `generic.make_module` (the X-07 name, dense-architecture roles only)."""
    from . import generic
    return generic.make_module(fullname)
