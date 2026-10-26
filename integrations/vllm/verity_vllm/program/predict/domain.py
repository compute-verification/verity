"""φ's domain: what of φ a predicted Program can't be the prediction of, refused before `build.build_program` writes one (a
sampler the vocabulary lacks, a world, profile, execution or checkpoint dtype outside the decoder's, and an engine the target
doesn't apply)."""
from __future__ import annotations

from .common import Unexpressible
from .phi import Phi

#: the engine environment a Program reads only as `tp_allreduce.settings_of` (above two ranks; two ranks' AllReduce2 is route-free)
#: or not at all (the transport: `pipeline.build` records NCCL_P2P_DISABLE, never consults it); the target's own are `TargetProfile.env()`
_ENV_MODELLED = ("VLLM_ALLREDUCE_USE_SYMM_MEM", "VLLM_ALLREDUCE_USE_FLASHINFER", "NCCL_P2P_DISABLE")
#: the engine args a component Program does not read but through `settings_of` (the scheduler's: a component is one request's)
_ARGS_MODELLED = ("disable_custom_all_reduce", "max_model_len", "max_num_batched_tokens", "max_num_seqs", "enable_prefix_caching",
                  "enable_chunked_prefill")


def check_phi(phi: Phi, domain: tuple[tuple[int, ...], str, tuple[str, ...]]) -> None:
    """Refuse φ outside a decoder's domain (worlds, execution, checkpoint dtypes) or the samplers the vocabulary has."""
    worlds, execution, dtypes = domain
    facts = []
    s = phi.sampling or {}
    if s.get("kind", "greedy") == "stochastic":
        if int(s.get("top_k", 0) or 0) != 0 or not s.get("seeded", False):
            facts.append(f"stochastic sampling with top_k {s.get('top_k')!r}, seeded {s.get('seeded')!r} (seeded top-p only)")
        if s.get("splits") != "derived" and not isinstance(s.get("splits"), int):
            facts.append(f"stochastic sampling without a split count (splits {s.get('splits')!r})")
    elif s.get("kind", "greedy") != "greedy":
        facts.append(f"sampling {s.get('kind')!r} (greedy or seeded top-p)")
    world = int((phi.tp or {}).get("world", 1))
    if world not in worlds:
        facts.append(f"tensor parallel world {world} ({' and '.join(f'TP{w}' for w in worlds)}, the worlds a Build has traced)")
    if phi.profile != "b1-eager-v3":
        facts.append(f"numerics profile {phi.profile!r}")
    if phi.execution != execution:
        facts.append(f"execution {phi.execution!r}")
    if (phi.model or {}).get("dtype", "bfloat16") not in dtypes:
        facts.append(f"checkpoint dtype {phi.model.get('dtype')!r} ({' or '.join(dtypes)} only)")
    facts += engine_facts(phi, world)
    if facts:
        raise Unexpressible("vocabulary cannot express: " + "; ".join(facts))


def engine_facts(phi: Phi, world: int) -> list[str]:
    """What of φ's engine the Program would not be the prediction of: a target whose execution label is not the row's, an env
    switch the target sets to another value or none models (`_ENV_MODELLED`), an engine arg the target applies at another value, the
    eager execution contradicted, another world, or an arg none models (`_ARGS_MODELLED`)."""
    tp, eng = phi.target_profile(), dict(phi.engine or {})
    facts = [] if tp.execution_label == phi.execution else [
        f"a target labelled {tp.execution_label!r} (batch invariant {tp.batch_invariant}) for the {phi.execution!r} row"]
    target_env = tp.env()
    for k, v in sorted((eng.pop("env", None) or {}).items()):
        if k in target_env and str(v) != target_env[k]:
            facts.append(f"env {k}={v} under a target that sets {target_env[k]!r}")
        elif k not in target_env and k not in _ENV_MODELLED:
            facts.append(f"env {k}={v} (the switches modelled: the target's {sorted(target_env)} and {list(_ENV_MODELLED)})")
    want = {**tp.engine_args(), "enforce_eager": tp.compilation is None, "tensor_parallel_size": world}
    for k, v in sorted(eng.items()):
        if k in want and v != want[k]:
            facts.append(f"engine arg {k}={v!r} where the {phi.execution!r} target applies {want[k]!r}")
        elif k not in want and k not in _ARGS_MODELLED:
            facts.append(f"engine arg {k}={v!r} (not modelled)")
    return facts
