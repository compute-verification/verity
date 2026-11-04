"""B, a served run's registered compiler: `compile_steps` (`predict.steps`, I2) capped at ℓ (`advice_windows.Capped`), the one
compiler each of its windows' advice is bound to. serve-openai registers it before its first window (`serve_advice`, through
``verity-vllm predict compiler``) and the verifier recomputes it from the run's ``serve.json`` (``prove_path.py compiler``), by
this one formula.

Its inner digest (`inner_digest`) names the compiler's name, its code (`served_identity`: `compile_steps` by module path,
its digest the verifier's `interface.compiler_digest` over the integration's import closure), what the compile reads of
``serve.json`` (φ of the served engine, `phi_of_serve`: the pinned model, revision and config, the target and the execution;
the default seed, `registered_seed`), ℓ, the served model name where it names one, and the transfer gates its model's
compiler of record lists (`served_gates`). So a ``serve.json`` other than the one B was registered under names another B.
"""
from __future__ import annotations

import functools
import hashlib
import importlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from verity.ml.circuits.codec import canonical_json
from verity_vllm.engine.profiles import dense_generic as DG
from verity_vllm.pipeline.advice_windows import CAP_BITS, Capped


def phi_of_serve(serve: Mapping[str, Any], hf_config: str | None = None, target: str | None = None) -> Any:
    """φ of a served engine as B compiles under it: the pinned model (``serve.json``'s ``model`` and ``revision``, and the
    checkpoint's ``config.json``: ``hf_config``, else the pin's config of record, as `advice_compile` reads it
    (`dense_generic.pinned_hf_config`), else for a model no manifest pins unsloth/Llama-3.2-1B's), the serving device's target
    profile (``serve.json``'s ``target``, the int8 scope its layers ran included, else the file ``target``) and the engine's
    execution label (``execution``, else ``engine.execution``); no workload."""
    from .phi import Phi

    prof = json.loads(Path(target).read_text()) if target else serve.get("target")
    if not isinstance(prof, Mapping):
        raise ValueError("serve.json records no target profile (the serving device's TargetProfile.to_json()), so φ can't be read: "
                         "--target F gives it")
    engine = serve.get("engine")
    execution = serve.get("execution") or (engine.get("execution") if isinstance(engine, Mapping) else None)
    if not isinstance(execution, str) or not execution:
        raise ValueError("serve.json records no execution label (execution, engine.execution), so φ can't be read")
    hf = json.loads(Path(hf_config or DG.pinned_hf_config(serve["model"], serve["revision"]) or DG.UNPINNED_HF_CONFIG).read_text())
    return Phi(model={"repo": serve["model"], "revision": serve["revision"], "hf_config": hf}, target=dict(prof), execution=execution)


def registered_seed(serve: Mapping[str, Any]) -> int:
    """The default seed B fixes, `engine.served_seed.DEFAULT_SEED`; a ValueError when ``serve.json`` says the server filled in
    another (or says none), since a request that sent no seed would be compiled under a seed its sample didn't use."""
    from verity_vllm.engine.served_seed import DEFAULT_SEED

    seed = serve.get("default_seed")
    if type(seed) is not int or seed != DEFAULT_SEED:
        raise ValueError(f"serve.json: the run's server filled in default seed {seed!r}, and the registered one is "
                         f"engine.served_seed.DEFAULT_SEED ({DEFAULT_SEED}): serve it under that rule")
    return DEFAULT_SEED


def registered_of_serve(serve: Mapping[str, Any], hf_config: str | None, target: str | None, cap_bits: int) -> Any:
    from .steps import Registered

    return Registered.of(phi_of_serve(serve, hf_config, target), registered_seed(serve), cap_bits)


def served_gates(serve: Mapping[str, Any], given: Sequence[str] = ()) -> tuple[str, ...]:
    """The transfer gates B lists: its model's compiler of record's (`dense_generic.GATES_OF_RECORD`, none for a model it has
    no line for).
    ``given`` (``--transfer-gate``) may only restate them: any other list is refused, so neither side builds a B listing
    another."""
    model, pinned = str(serve.get("model")), DG.GATES_OF_RECORD.get(str(serve.get("model")), ())
    if given and sorted(set(map(str, given))) != sorted(pinned):
        raise ValueError(f"--transfer-gate {sorted(set(map(str, given)))}: {model}'s compiler of record lists {list(pinned)} "
                         "(dense_generic.GATES_OF_RECORD), and a served window's compiler lists those and no others")
    return pinned


@functools.lru_cache(maxsize=1)
def served_identity() -> tuple[str, str]:
    """The code B runs, as `verity-vllm advice` names it (``pipeline.advice``'s ``registered_compiler``): compile_steps by its
    module path, its digest the verifier's `interface.compiler_digest` over the integration's import closure
    (``integrations/vllm`` and ``verity``), so the audit's tree holds the server's."""
    ADV = importlib.import_module("verity_vllm.pipeline.advice")
    inner = getattr(ADV.registered_compiler(), "inner", None)
    if inner is None:
        raise ValueError("verity_vllm.pipeline.advice registers no capped compiler (its registered_compiler wraps none)")
    return str(inner.name), str(inner.digest)


def inner_digest(name: str, code: str, registered: Any, transfer_gates: Sequence[str]) -> str:
    """The digest of I2 named ``name`` running ``code`` under ``registered`` (`predict.steps.Registered`: φ, the default seed,
    ℓ and the served model name, where it names one), listing ``transfer_gates``."""
    served_model = getattr(registered, "served_model", None)
    return hashlib.sha512(canonical_json({"name": name, "code": code, "phi": registered.phi.to_json(),
                                          "default_seed": registered.default_seed, "cap_bits": registered.cap_bits,
                                          **({"served_model": served_model} if served_model is not None else {}),
                                          "transfer_gates": sorted(transfer_gates)})).hexdigest()


def compiler_doc(serve: Mapping[str, Any], hf_config: str | None = None, target: str | None = None, cap_bits: int = CAP_BITS,
                 transfer_gates: Sequence[str] = ()) -> dict[str, Any]:
    """B of the run whose ``serve.json`` is ``serve``: the name and digest its windows' bindings must carry, ℓ and the compiler
    it wraps (``inner``, what `verity-vllm advice --compiler-name --compiler-digest` takes), the server's code (``served``), the
    transfer gates it lists, and the φ and default seed I2 compiles each window's Programs under."""
    name, code = served_identity()
    reg = registered_of_serve(serve, hf_config, target, cap_bits)
    gates = served_gates(serve, transfer_gates)
    inner = {"name": name, "digest": inner_digest(name, code, reg, gates)}
    c = Capped(SimpleNamespace(**inner), cap_bits)
    return {"name": c.name, "digest": c.digest, "cap_bits": c.cap_bits, "inner": inner, "served": {"name": name, "digest": code},
            "transfer_gates": list(gates), "phi": reg.phi.to_json(), "default_seed": reg.default_seed}
