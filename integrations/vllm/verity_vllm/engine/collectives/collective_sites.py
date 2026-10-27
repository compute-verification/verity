"""MID-module TP collective sites of a vLLM model -- the ONE finder the GPU-side observers share (the
committer's `TPPartialSource`, the Match recorder `tp2_worker.tp2_coll_install`).

vLLM issues most TP collectives from its layer modules (`RowParallelLinear`, `VocabParallelEmbedding`, `LogitsProcessor`, the fused-MoE
runner): the observers wrap the module-level names `tensor_model_parallel_all_reduce` / `_all_gather` of THOSE python modules and attribute
a call to the innermost hooked module (`SITES` / `MOE_COLLECTIVE_MODULES`).  A MODEL class may import the same names into its own module
and call them from inside its forward -- `vllm/model_executor/models/olmoe.py`'s `OlmoeAttention._apply_qk_norm` all-gathers q and k
(`tensor_model_parallel_all_gather(q.contiguous())`, twice per layer under TP) -- and such a call is invisible to the wrappers above: the
name is bound in the model's module.  On OLMoE at TP2 (B8) the recorder saw 34 collectives per step against the Build's 66 links
(structure FAIL), and nothing committed the q/k shards or the gathered rows (6,944 identities per rank unbound).

A mid-module site is (module path, class) of a model module whose CLASS source contains a call of one of the collective names (static:
`inspect.getsource`), when the module runs under TP (`tp_size > 1` where the module declares one, else the world).  The observers hook
these modules for attribution and patch the collective names of the classes' python modules.  The per-step count of calls is the static
call count (OlmoeAttention: 2 all_gather), less the calls under an `if self.<flag>:` / `if not self.<flag>:` branch (or a flag conjunct
of an `and`) that the module instance's flag rules out (Qwen3MoeSparseMoeBlock all-gathers only `if self.is_sequence_parallel:`, off
under plain TP; any other condition, such as OLMoE's `if self.tp_size > 1:`, is not evaluated and its calls count).  The committer's
identity_check requires exactly that many partials per site per step, the manifest one identity per site in Program order
(`required_manifest.collective_site` / `collective_site_aliases`).  A class whose source cannot be read is NOT a site (fail closed by
omission: the collective would then be recorded unattributed / missing, both named verdicts); a count the run contradicts is named by
identity_check, never relabelled.  Each ruled-out call is listed with its flag, the truth its branch wants and the instance's value
(`mid_module_site_census` "ruled_out": the Match recorder's `coll_meta`, the Commit's identity report), and the observers keep its
python module patched, so a ruled-out call that runs anyway is recorded unattributed (or over its site's count) and named.
"""
from __future__ import annotations

import ast
import importlib
import inspect
import textwrap
from typing import Any

#: (module-level name a model class may import, collective kind)
COLLECTIVE_NAMES = (("tensor_model_parallel_all_reduce", "all_reduce"), ("tensor_model_parallel_all_gather", "all_gather"))
_KIND_OF = dict(COLLECTIVE_NAMES)
#: the python-module prefix of vLLM's model implementations (the classes whose direct collective calls are sites)
MODEL_MODULE_PREFIX = "vllm.model_executor.models."
#: the module classes that issue the fused-MoE block's final all-reduce from inside their forward; both observers hook them, and a
#: wrapper holding a runner submodule is the site (the nested runner is not hooked)
MOE_COLLECTIVE_CLASSES = ("MoERunner", "FusedMoE", "SharedFusedMoE")
#: the vLLM modules whose OWN `tensor_model_parallel_all_reduce` name those forwards call (the call moved from fused_moe/layer.py to
#: fused_moe/runner/moe_runner.py in 0.28); every one that imports is patched, missing ones are skipped
MOE_COLLECTIVE_MODULES = ("vllm.model_executor.layers.fused_moe.runner.moe_runner", "vllm.model_executor.layers.fused_moe.layer",
                          "vllm.model_executor.layers.fused_moe.shared_fused_moe")


def _flag_guard(test: ast.expr) -> tuple[str, bool] | None:
    """`self.X` -> ("X", True), `not self.X` -> ("X", False); None for any other condition (not evaluated)."""
    want = True
    while isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
        test, want = test.operand, not want
    if isinstance(test, ast.Attribute) and isinstance(test.value, ast.Name) and test.value.id == "self":
        return test.attr, want
    return None


def _branch_guards(test: ast.expr) -> tuple[tuple[tuple[str, bool], ...], tuple[tuple[str, bool], ...]]:
    """(guards of the `if` body, guards of its `else`): a bare flag guards both ways; every flag conjunct of an `and` guards the body
    (`if self.is_sequence_parallel and not already_sequence_parallel:`), and nothing is concluded for that `else`."""
    g = _flag_guard(test)
    if g is not None:
        return (g,), ((g[0], not g[1]),)
    if isinstance(test, ast.BoolOp) and isinstance(test.op, ast.And):
        return tuple(x for x in map(_flag_guard, test.values) if x is not None), ()
    return (), ()


def _guarded_calls(src: str) -> list[tuple[str, tuple[tuple[str, bool], ...]]]:
    """[(kind, guards)] for every call of a collective name in `src`, in source order; `guards` are the (flag, required truth) pairs the
    enclosing `if` branches require (`_branch_guards`)."""
    out: list[tuple[str, tuple[tuple[str, bool], ...]]] = []

    def visit(node: ast.AST, guards: tuple[tuple[str, bool], ...]) -> None:
        if isinstance(node, ast.If):
            body, orelse = _branch_guards(node.test)
            visit(node.test, guards)
            for n in node.body:
                visit(n, guards + body)
            for n in node.orelse:
                visit(n, guards + orelse)
            return
        if isinstance(node, ast.Call):
            f = node.func
            kind = _KIND_OF.get(f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else "")
            if kind is not None:
                out.append((kind, guards))
        for child in ast.iter_child_nodes(node):
            visit(child, guards)

    visit(ast.parse(textwrap.dedent(src)), ())
    return out


def _ruling(instance: Any, guards: tuple[tuple[str, bool], ...]) -> tuple[str, bool, Any] | None:
    """(flag, required truth, the instance's value) of the first guard flag (a bool, int or None attribute) whose value contradicts the
    branch; None when none does.  A flag the instance lacks, or holds as any other type, does not rule a call out."""
    for flag, want in guards:
        v = getattr(instance, flag, ...)
        if (v is None or isinstance(v, (bool, int))) and bool(v) != want:
            return flag, want, v
    return None


def _class_calls(cls: type, model_prefix: str | None) -> list[tuple[str, tuple[tuple[str, bool], ...]]]:
    mod = getattr(cls, "__module__", "") or ""
    if model_prefix is None:
        model_prefix = MODEL_MODULE_PREFIX
    if model_prefix and not mod.startswith(model_prefix):
        return []
    try:
        return _guarded_calls(inspect.getsource(cls))
    except (OSError, TypeError, SyntaxError):
        return []


def _count(calls: list[tuple[str, tuple[tuple[str, bool], ...]]], instance: Any) -> tuple[dict[str, int], list[dict[str, Any]]]:
    """({kind: n} of the calls the instance's flags allow, [{"kind", "flag", "want", "value"}] of those they rule out)."""
    out: dict[str, int] = {}
    ruled: list[dict[str, Any]] = []
    for kind, guards in calls:
        r = None if instance is None else _ruling(instance, guards)
        if r is None:
            out[kind] = out.get(kind, 0) + 1
        else:
            ruled.append({"kind": kind, "flag": r[0], "want": r[1], "value": r[2]})
    return out, ruled


def class_collective_calls(cls: type, *, model_prefix: str | None = None, instance: Any = None) -> dict[str, int]:
    """{kind: call count} of the collective names called in `cls`'s own source; {} for a class outside the model modules or one
    whose source cannot be read.  Inherited methods are not searched (a vLLM layer base class is not a model class).  With `instance`,
    a call under an `if self.<flag>:` branch the instance's flag rules out is not counted."""
    return _count(_class_calls(cls, model_prefix), instance)[0]


def _under_tp(m: Any, world: int) -> bool:
    tp = getattr(m, "tp_size", None)
    return int(tp if tp is not None else world) > 1


def mid_module_site_census(model: Any, world: int, *, model_prefix: str | None = None) -> dict[str, list[dict[str, Any]]]:
    """{"sites": [{"name", "cls", "py_module", "calls": {kind: n}, "per_step": n_total}], "ruled_out": [{"name", "cls", "py_module",
    "site_dropped", "calls": [{"kind", "flag", "want", "value"}]}]}: every module of `model` whose class calls a TP collective directly,
    in `named_modules()` order, with the calls its instance's flags allow; a module with a call they rule out is listed in "ruled_out"
    ("site_dropped": no call is left, so the module is no site)."""
    if int(world) <= 1:
        return {"sites": [], "ruled_out": []}
    sites: list[dict[str, Any]] = []
    ruled_out: list[dict[str, Any]] = []
    cache: dict[type, list[tuple[str, tuple[tuple[str, bool], ...]]]] = {}
    for name, m in model.named_modules():
        cls = type(m)
        if cls not in cache:
            cache[cls] = _class_calls(cls, model_prefix)
        if not cache[cls] or not _under_tp(m, world):
            continue
        calls, ruled = _count(cache[cls], m)
        where = {"name": name, "cls": cls.__name__, "py_module": cls.__module__}
        if ruled:
            ruled_out.append({**where, "site_dropped": not calls, "calls": ruled})
        if calls:
            sites.append({**where, "calls": dict(calls), "per_step": sum(calls.values())})
    return {"sites": sites, "ruled_out": ruled_out}


def mid_module_sites(model: Any, world: int, *, model_prefix: str | None = None) -> list[dict[str, Any]]:
    """The "sites" of `mid_module_site_census`.  Empty for a dense vLLM model (its collectives are the layer modules')."""
    return mid_module_site_census(model, world, model_prefix=model_prefix)["sites"]


def patch_targets(sites: list[dict[str, Any]]) -> list[tuple[Any, str, str]]:
    """The (python module, attribute, kind) triples the observers wrap for `sites`: each site class's module x each collective name it
    binds (deduplicated, in site order).  A module that does not bind a name is skipped (the class called it through another path).
    The observers pass the census's "ruled_out" entries too, so a ruled-out call that runs anyway is still seen."""
    seen: set[tuple[str, str]] = set()
    out: list[tuple[Any, str, str]] = []
    for s in sites:
        mod = importlib.import_module(s["py_module"])
        for attr, kind in COLLECTIVE_NAMES:
            if (s["py_module"], attr) in seen or not hasattr(mod, attr):
                continue
            seen.add((s["py_module"], attr))
            out.append((mod, attr, kind))
    return out
