"""A checkpoint role's HF config.json, and the module name `dense_<ROLE>` of its configuration-derived κ profile — no per-model module.

The constructor is `profiles.generic` (every architecture class, `world > 1`, the family facts table); this module keeps the HF config.json
lookup (`find_hf_config`, `checkpoint_entry`, `hf_snapshot`) and the module name `dense_<ROLE>` (`role_of_module`), which the
`sys.meta_path` finder in `profiles/__init__.py` materialises through `generic.make_module`.

Why: a Mistral-7B Build passed but Match said "no profile module declares case 'MISTRAL7B'".
The κ machinery was already configuration-only in the model dimension (`build_profile(analytic.config_of(config.json))`, D72) —
but a hand-authored MODULE per role was still required (`profiles.PROFILE_BY_CASE` / the `PROFILE.notes["case"]` scan), so the
profile was in effect keyed to the model NAME.  The rule: applicability comes from the selected kernels and the target facts,
not from an allowlist of names.  This module removes the name: `verity_vllm.observe.fold.profiles.dense_<ROLE>` is materialised on import
(a `sys.meta_path` finder in `profiles/__init__.py`) from the role's HF `config.json` — every consumer that does
`importlib.import_module(f"verity_vllm.observe.fold.profiles.{name}")` (resolve_log, compare, gates, run_config, fold, cb_a) works
unchanged, in a fresh process, with `--profile dense_MISTRAL7B`.

What decides applicability at Match time (unchanged, and the point):
  * the PATTERNS — first-match over the launched kernels with every constexpr / launch-geometry field pinned (`GemmLaunch`,
    `GemmLaunchTuned` for vLLM's tuned (N,K) table, `RmsNorm*`, `FlashFwdSplitKV{BN}`, …); a kernel the pins do not cover is
    `Unsupported`, never silently folded;
  * `check_header(C)` — the run's recorded HF config vs the derived `C` (H, NL, NH, KVH, D, I, V, EPS, theta, MAX_POS, bias, tied);
  * `weight_field(C)` — the Llama-architecture vLLM parameter names (`qkv_proj`, `o_proj`, `gate_up_proj`, `down_proj`,
    `input_layernorm`, `post_attention_layernorm`; `qkv_b` iff `C["BIAS"]`, `lm_head` iff untied) → `analytic.weights_type(C)`.
A checkpoint whose architecture is NOT this vocabulary (MoE routing, different param names, sliding window, softcap, …) is not
made to fit: its roots resolve as unknown / its kernels as Unsupported, and `generic.structural_refusal(cfg)` refuses the derivation
fail-closed for the `model_type`s whose vLLM implementation is outside the family facts table.

Where the config comes from (`find_hf_config(role)`, first hit wins; the source is recorded in `PROFILE.notes["hf_config"]`):
  1. `profiles/hf_configs/<ROLE>.config.json` — a byte copy shipped with the package (`record_v5/ship.sh` ships `verity_vllm/observe/`,
     not `out/gen/`); its sha256 must equal the manifest entry's `config_sha256` when the manifest is present;
  2. the manifest entry's `local_path/config.json` (the downloaded snapshot on the pod);
  3. `$HF_HOME/hub/models--<repo>/snapshots/<revision>/config.json` (the hub cache).
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from verity_vllm import settings
from verity_vllm.config import CHECKPOINTS as MANIFEST, checkpoint_entries

MODULE_PREFIX = "dense_"                    # verity_vllm.observe.fold.profiles.dense_<ROLE>
HF_CONFIG_DIR = Path(__file__).with_name("hf_configs")

#: the served vocabulary is `family_facts.FAMILY_FACTS` (every family's vLLM hard-codes with provenance); a configuration outside it, or
#: with a structure the table does not describe, refuses fail-closed in `generic.structural_refusal` -- not a silently mis-fitted profile.


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
    entry of the repo `verity_vllm.engine.profiles.cases.CASES[case]` names."""
    if not MANIFEST.exists():
        return None
    entries = checkpoint_entries(MANIFEST)
    for e in entries:
        if e.get("role") == role:
            return e
    from verity_vllm.engine.profiles.cases import CASES
    repo = (CASES.get(role) or {}).get("repo")
    if repo:
        for e in entries:
            if e.get("repo") == repo:
                return e
    return None


def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def hf_snapshot(repo: str, revision: str) -> Path:
    """The hub cache's snapshot of `repo` at `revision`: `$HF_HOME/hub/models--<repo>/snapshots/<revision>`, where vLLM's config
    load reads it (HF_HUB_OFFLINE)."""
    hf_home = settings.HF_HOME or str(Path.home() / ".cache" / "huggingface")
    return Path(hf_home) / "hub" / f"models--{repo.replace('/', '--')}" / "snapshots" / revision


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
        repo, rev = entry.get("repo"), entry.get("revision")
        if repo and rev:
            snap = hf_snapshot(repo, rev) / "config.json"
            if snap.exists():
                return snap, "hf_home:snapshot", entry
    return None


def pinned_hf_config(repo: str, revision: str) -> Path | None:
    """The config.json of record of the checkpoint the manifests pin at ``repo``@``revision``: `find_hf_config` of its role, its
    sha256 held to the pin's ``config_sha256``, so a prover and a verifier read the same bytes wherever they run. None when no
    manifest pins it; a ValueError when it is pinned and no source has its config, or one has another config."""
    if not MANIFEST.exists():
        return None
    entry = next((e for e in checkpoint_entries(MANIFEST) if e.get("repo") == repo and str(e.get("revision")) == str(revision)), None)
    if entry is None:
        return None
    found = find_hf_config(str(entry.get("role")))
    if found is None:
        raise ValueError(f"{repo}@{revision} (role {entry.get('role')}) is pinned, and no source has its config.json "
                         f"(profiles/hf_configs/{entry.get('role')}.config.json, its local_path, the hub cache)")
    if entry.get("config_sha256") and _sha256(found[0]) != entry["config_sha256"]:
        raise ValueError(f"{found[0]} sha256 {_sha256(found[0])[:12]} != {repo}@{revision}'s pinned config_sha256 "
                         f"{entry['config_sha256'][:12]}")
    return found[0]
