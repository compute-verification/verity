"""The workload's declared `TargetProfile` in a tensor-parallel driver (`tp-capture`, `tp-commit`), applied and checked as
`engine.build.engine_kwargs_for` and `build_engine` apply and check it for their in-process engine: its environment knobs before vLLM
reads them, its engine arguments under the workload's and the CLI's (`apply`), and every rank's built engine against it
(`require_ranks`; the driver holds no model).  The driver passes the same object to each rank's `tp2_install`, whose header attests
it (`profile.target`, and `backends.moe_experts_class` under a declared `moe_backend`)."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from verity_vllm.engine import env as engine_env, served


def apply(target: Any, execution: str | None) -> tuple[dict[str, str], dict[str, Any]]:
    """The environment the row's ``execution`` label (`engine.served.apply`) and then ``target`` declare, applied, and the engine
    arguments ``target`` declares (none without a declared target).  Its int8 parts are registered here and exported to the ranks
    (`engine.int8_ranks`)."""
    env = served.apply(execution)
    if target is None:
        return env, {}
    from verity_vllm.engine.build import register_int8_parts
    env = {**env, **engine_env.apply_target(target.env())}
    register_int8_parts(target)
    return env, target.engine_args()


def extension(target: Any, want: str | None) -> str | None:
    """The engine's `worker_extension_cls`: `want`, else `engine.int8_ranks.EXTENSION` when ``target`` serves int8 parts (its ranks
    import it before they build the model), else None."""
    if want is None and getattr(target, "int8_parts", ()):
        from verity_vllm.engine.int8_ranks import EXTENSION
        return EXTENSION
    return want


def as_json(target: Any) -> dict | None:
    return target.to_json() if target is not None else None


def rank_mismatches(worker: Any, target: Any) -> dict[str, Any]:
    """On a rank (`LLM.collective_rpc` with a callable): `vllm_adapter.target_mismatches` of its device and model against
    ``target``."""
    from vllm.distributed import get_tensor_model_parallel_rank

    from verity_vllm.engine import vllm_adapter as va

    llm = SimpleNamespace(llm_engine=SimpleNamespace(vllm_config=worker.vllm_config), collective_rpc=lambda fn, *a, **k: [fn(worker)])
    eng = SimpleNamespace(target=target, llm=llm, model=worker.get_model())
    return {"rank": int(get_tensor_model_parallel_rank()), "mismatches": va.target_mismatches(eng)}


def require_ranks(llm: Any, target: Any, timeout: float) -> list[dict[str, Any]]:
    """Every rank's engine realises ``target`` (`rank_mismatches`), else SystemExit (fail closed) naming each rank's mismatch; no
    check without a declared target."""
    if target is None:
        return []
    answers = llm.collective_rpc(rank_mismatches, timeout=timeout, kwargs={"target": target})
    bad = [f"rank {a['rank']}: {m}" for a in answers for m in a["mismatches"]]
    if bad:
        raise SystemExit("fail-closed: the engine does not realise the declared TargetProfile: " + "; ".join(bad))
    return answers


__all__ = ["apply", "as_json", "extension", "rank_mismatches", "require_ranks"]
