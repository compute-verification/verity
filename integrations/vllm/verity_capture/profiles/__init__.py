"""Pinned capture profiles: pattern registrations and accepted constexpr/geometry values per (vLLM, torch, Triton, GPU) tuple."""

#: checkpoint role -> profile module (the κ pins are shared; the model enters as `build_profile(C)`, D72).  Lanes register
#: new roles here (one line) together with their `checkpoints.json` entry.
PROFILE_BY_CASE: dict[str, str] = {
    "B0": "vllm_d9105ea80_sm89_eager",           # SmolLM2-135M
    "B1": "vllm_d9105ea80_sm89_eager_qwen15",    # Qwen2.5-1.5B
    # OLMoE-1B-7B: two modules declare the role (wave 1's `gen_moe_olmoe` resolves the MoE block as a recognised Unsupported;
    # `gen_ov_moe_olmoe` folds it as pair-major Computes).  The case scan would return wave 1's (module order); the coverage
    # runner (`cov_pod.sh run OLMOE …`) must get the constructive profile.
    "OLMOE": "gen_ov_moe_olmoe",
    # [R19 qwen3, row #75] Qwen3-30B-A3B: the constructive MoE profile (renormalising router, E=128 pins); its `PROFILE` is the TP2 RANK
    # profile of the row of record (`tp2_fold_match --profile gen_ov_moe_qwen3`); `build_profile(world=1)` is the TP1 one.
    "QWEN3_30B_A3B": "gen_ov_moe_qwen3",
}


class _DenseProfileFinder:
    """[X-07] `verity_capture.profiles.dense_<ROLE>`: a κ profile derived on import from the role's HF config.json
    (`dense_generic.make_module`) — no hand-authored module per model.  Installed on `sys.meta_path` when this package is
    imported, so every `importlib.import_module(f"verity_capture.profiles.{name}")` consumer (resolve_log, compare, gates,
    run_config, cb_a) accepts `--profile dense_MISTRAL7B` in a fresh process."""

    def find_spec(self, fullname, path=None, target=None):
        import importlib.util
        if not fullname.startswith(__name__ + ".dense_"):
            return None
        from . import dense_generic
        if dense_generic.role_of_module(fullname) is None:
            return None
        return importlib.util.spec_from_loader(fullname, self)

    def create_module(self, spec):
        from . import dense_generic
        return dense_generic.make_module(spec.name)

    def exec_module(self, module):
        return None


def _install_finder() -> None:
    import sys
    if not any(isinstance(f, _DenseProfileFinder) for f in sys.meta_path):
        sys.meta_path.append(_DenseProfileFinder())


_install_finder()


def profile_module_for_case(case: str) -> str:
    """The profile MODULE name of a checkpoint role: `PROFILE_BY_CASE`, else the module whose PROFILE.notes["case"] is the
    role, else [X-07] the configuration-derived `dense_<ROLE>` when the role's HF config.json is available and its
    `model_type` is in the dense Llama-architecture vocabulary (`dense_generic.DENSE_FAMILIES`)."""
    import importlib
    import pkgutil

    name = PROFILE_BY_CASE.get(case)
    if name is not None:
        return name
    for m in pkgutil.iter_modules(__path__):
        if m.name.startswith("workload") or m.name.startswith("_"):
            continue
        try:
            mod = importlib.import_module(f"verity_capture.profiles.{m.name}")
        except Exception:
            continue
        prof = getattr(mod, "PROFILE", None)
        if prof is not None and (prof.notes or {}).get("case") == case:
            return m.name
    from . import dense_generic
    hit = dense_generic.find_hf_config(case)
    if hit is not None:
        import json
        why = dense_generic.supported_family(json.loads(hit[0].read_text()))
        if why is None:
            return dense_generic.module_name(case)
        raise KeyError(f"no profile module declares case {case!r} and dense_generic refuses it: {why}")
    raise KeyError(f"no profile module declares case {case!r} (PROFILE.notes['case']) and no HF config.json is available for a "
                   f"configuration-derived dense profile (profiles/hf_configs/{case}.config.json)")


def profile_for_case(case: str):
    """The `Profile` of a checkpoint role (torch-free import).  Roles not in `PROFILE_BY_CASE` are found by scanning the
    package for a module whose `PROFILE.notes["case"]` is the role — a lane adds a profile module, never edits this file."""
    import importlib

    return importlib.import_module(f"verity_capture.profiles.{profile_module_for_case(case)}").PROFILE
