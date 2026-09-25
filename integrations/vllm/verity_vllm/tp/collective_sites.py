"""[R19 moe F-r19-moe-1, row #70] MID-module TP collective sites of a vLLM model -- the ONE finder the GPU-side observers share (the
committer's `TPPartialSource`, the Match recorder `tp2_worker.tp2_coll_install`).

vLLM issues most TP collectives from its layer modules (`RowParallelLinear`, `VocabParallelEmbedding`, `LogitsProcessor`, the fused-MoE
runner): the observers wrap the module-level names `tensor_model_parallel_all_reduce` / `_all_gather` of THOSE python modules and attribute
a call to the innermost hooked module (`SITES` / `MOE_COLLECTIVE_MODULES`).  A MODEL class may import the same names into its own module
and call them from inside its forward -- `vllm/model_executor/models/olmoe.py`'s `OlmoeAttention._apply_qk_norm` all-gathers q and k
(`tensor_model_parallel_all_gather(q.contiguous())`, twice per layer under TP) -- and such a call is invisible to the wrappers above: the
name is bound in the model's module.  R17 row #70 (OLMoE TP2 B8): the recorder saw 34 collectives per step against the Build's 66 links
(structure FAIL), and nothing committed the q/k shards or the gathered rows (6,944 identities per rank unbound).

A mid-module site is (module path, class) of a model module whose CLASS source contains a call of one of the collective names (static:
`inspect.getsource`), when the module runs under TP (`tp_size > 1` where the module declares one, else the world).  The observers hook
these modules for attribution and patch the collective names of the classes' python modules.  The per-step count of calls is the static
call count (OlmoeAttention: 2 all_gather); the committer's identity_check requires exactly that many partials per site per step, the
manifest requires one identity per site in Program order (`required_manifest.collective_site` / `collective_site_aliases`).  A class
whose source cannot be read is NOT a site (fail closed by omission: the collective would then be recorded unattributed / missing, both
named verdicts); a count the run contradicts is named by identity_check, never relabelled.
"""
from __future__ import annotations

import importlib
import inspect
import re
from typing import Any

#: (module-level name a model class may import, collective kind)
COLLECTIVE_NAMES = (("tensor_model_parallel_all_reduce", "all_reduce"), ("tensor_model_parallel_all_gather", "all_gather"))
#: the python-module prefix of vLLM's model implementations (the classes whose direct collective calls are sites)
MODEL_MODULE_PREFIX = "vllm.model_executor.models."
_CALL_RE = re.compile(r"\btensor_model_parallel_(all_reduce|all_gather)\s*\(")
#: the module classes that issue the fused-MoE block's final all-reduce from inside their forward; both observers hook them, and a
#: wrapper holding a runner submodule is the site (the nested runner is not hooked)
MOE_COLLECTIVE_CLASSES = ("MoERunner", "FusedMoE", "SharedFusedMoE")
#: the vLLM modules whose OWN `tensor_model_parallel_all_reduce` name those forwards call (the call moved from fused_moe/layer.py to
#: fused_moe/runner/moe_runner.py in 0.28); every one that imports is patched, missing ones are skipped
MOE_COLLECTIVE_MODULES = ("vllm.model_executor.layers.fused_moe.runner.moe_runner", "vllm.model_executor.layers.fused_moe.layer",
                          "vllm.model_executor.layers.fused_moe.shared_fused_moe")


def class_collective_calls(cls: type, *, model_prefix: str | None = None) -> dict[str, int]:
    """{kind: static call count} of the collective names called in `cls`'s own source; {} for a class outside the model modules or one
    whose source cannot be read.  Inherited methods are not searched (a vLLM layer base class is not a model class)."""
    mod = getattr(cls, "__module__", "") or ""
    if model_prefix is None:
        model_prefix = MODEL_MODULE_PREFIX
    if model_prefix and not mod.startswith(model_prefix):
        return {}
    try:
        src = inspect.getsource(cls)
    except (OSError, TypeError):
        return {}
    out: dict[str, int] = {}
    for m in _CALL_RE.finditer(src):
        out[m.group(1)] = out.get(m.group(1), 0) + 1
    return out


def _under_tp(m: Any, world: int) -> bool:
    tp = getattr(m, "tp_size", None)
    return int(tp if tp is not None else world) > 1


def mid_module_sites(model: Any, world: int, *, model_prefix: str | None = None) -> list[dict[str, Any]]:
    """[{"name", "cls", "py_module", "calls": {kind: n}, "per_step": n_total}] for every module of `model` whose class calls a TP collective
    directly, in `named_modules()` order.  Empty for a dense vLLM model (its collectives are the layer modules')."""
    if int(world) <= 1:
        return []
    out: list[dict[str, Any]] = []
    cache: dict[type, dict[str, int]] = {}
    for name, m in model.named_modules():
        cls = type(m)
        if cls not in cache:
            cache[cls] = class_collective_calls(cls, model_prefix=model_prefix)
        calls = cache[cls]
        if not calls or not _under_tp(m, world):
            continue
        out.append({"name": name, "cls": cls.__name__, "py_module": cls.__module__, "calls": dict(calls), "per_step": sum(calls.values())})
    return out


def patch_targets(sites: list[dict[str, Any]]) -> list[tuple[Any, str, str]]:
    """The (python module, attribute, kind) triples the observers wrap for `sites`: each site class's module x each collective name it
    binds (deduplicated, in site order).  A module that does not bind a name is skipped (the class called it through another path)."""
    seen: set[tuple[str, str]] = set()
    out: list[tuple[Any, str, str]] = []
    for s in sites:
        try:
            mod = importlib.import_module(s["py_module"])
        except Exception:  # noqa: BLE001 -- the class's module always imports (it is loaded); defensive
            continue
        for attr, kind in COLLECTIVE_NAMES:
            if (s["py_module"], attr) in seen or not hasattr(mod, attr):
                continue
            seen.add((s["py_module"], attr))
            out.append((mod, attr, kind))
    return out
