"""Pinned capture profiles: pattern registrations and accepted constexpr/geometry values per (vLLM, torch, Triton, GPU) tuple.

There is no per-model profile module and no role -> module table.  A checkpoint role's κ profile is
`profiles.generic.build_profile(role, world)` -- the κ pins of `vllm_d9105ea80_sm89_eager`, the architecture's pattern libraries
(`gen_*_patterns`) and the role's HF `config.json` plus the declared family facts, which stay with the model facts in `engine.profiles`
(`engine.profiles.family_facts.FAMILY_FACTS[model_type]`, the one place a family name appears).
`verity_vllm.observe.fold.profiles.derived_<ROLE>[_tp<W>]` (and the legacy name `dense_<ROLE>`) is materialised on import by the
finder below, so every `importlib.import_module(f"verity_vllm.observe.fold.profiles.{name}")` consumer (resolve_log, compare, gates, run_config,
tp2_fold_match, cb_a) works unchanged in a fresh process.  The two protected base modules (`vllm_d9105ea80_sm89_eager`, `..._qwen15`) stay
as the κ pin record and the golden corpus's profiles; `derived_B0` is the base module's profile derived from config.json, behind the
target's tuned GEMM (`tests/engine/test_profiles_generic.py` pins the agreement)."""


class _DerivedProfileFinder:
    """`sys.meta_path` hook: `verity_vllm.observe.fold.profiles.derived_<ROLE>[_tp<W>]` / `dense_<ROLE>` -> `generic.make_module`."""

    def find_spec(self, fullname, path=None, target=None):
        import importlib.util
        if not fullname.startswith(__name__ + "."):
            return None
        from verity_vllm.observe.fold.profiles import generic
        if generic.parse_module(fullname) is None:
            return None
        return importlib.util.spec_from_loader(fullname, self)

    def create_module(self, spec):
        from verity_vllm.observe.fold.profiles import generic
        return generic.make_module(spec.name)

    def exec_module(self, module):
        return None


def _install_finder() -> None:
    import sys
    if not any(isinstance(f, _DerivedProfileFinder) for f in sys.meta_path):
        sys.meta_path.append(_DerivedProfileFinder())


_install_finder()


def profile_module_for_case(case: str, world: int = 1) -> str:
    """The profile MODULE name of a checkpoint role: `derived_<ROLE>[_tp<W>]` when the role's HF config.json is available and its
    `model_type` is in the family facts table; KeyError (with the reason) otherwise -- never a name looked up in a table."""
    import json

    from verity_vllm.engine.profiles import dense_generic
    from verity_vllm.observe.fold.profiles import generic
    hit = dense_generic.find_hf_config(case)
    if hit is None:
        raise KeyError(f"no HF config.json is available for role {case!r} (profiles/hf_configs/{case}.config.json, the manifest entry's local_path, "
                       f"or the $HF_HOME snapshot): a configuration-derived profile needs the checkpoint's configuration")
    why = generic.structural_refusal(json.loads(hit[0].read_text()))
    if why:
        raise KeyError(f"role {case!r}: the configuration-derived profile refuses it: {why}")
    return generic.module_name(case, world)


def profile_for_case(case: str, world: int = 1):
    """The `Profile` of a checkpoint role (torch-free import)."""
    import importlib

    return importlib.import_module(f"verity_vllm.observe.fold.profiles.{profile_module_for_case(case, world)}").PROFILE
