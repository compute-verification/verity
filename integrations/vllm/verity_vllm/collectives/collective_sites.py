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

A call the INSTANCE can never execute is not expected (`mid_module_site_census`): a call in the body of a bare `if self.<attr>:`, in
an undecorated method of the class whose first parameter is `self`, whose attribute is in the live module's own `__dict__` and is
False, None or 0.  vLLM's `Qwen3MoeSparseMoeBlock.forward` all-gathers only `if self.is_sequence_parallel:` (sequence-parallel MoE, off
by default); counting that call made every `model.layers.<L>.mlp` an expected site no step executes (`missing_modules` at TP8).  Every
other condition (`not`, `and` / `or`, comparisons, calls, the `else` branch of the `if`) keeps the call, and the class keeps all its
calls when its source does not parse, its parsed calls disagree with the static count, it uses `setattr` / `delattr` / `__dict__` /
`__setattr__`, or the attribute is written anywhere but directly in `__init__`.  A skipped call is listed (module path, attribute,
value, line) and its python module stays patched: if it runs anyway it is recorded unattributed (or over its site's count) and named,
never silent.
"""
from __future__ import annotations

import ast
import importlib
import inspect
import re
import textwrap
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


_KINDS = dict(COLLECTIVE_NAMES)
#: nested scopes: a call inside one may run later or elsewhere, so no enclosing guard reaches into it
_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)
#: names through which a class can write an attribute the AST does not show as `self.<attr> = ...`
_REFLECTION = frozenset({"setattr", "delattr", "__dict__", "__setattr__", "__delattr__"})


def _model_class_source(cls: type, model_prefix: str | None) -> tuple[str, int] | None:
    """(source, first line) of a class under the model modules; None outside them or when its source cannot be read."""
    mod = getattr(cls, "__module__", "") or ""
    if model_prefix is None:
        model_prefix = MODEL_MODULE_PREFIX
    if model_prefix and not mod.startswith(model_prefix):
        return None
    try:
        lines, first = inspect.getsourcelines(cls)
    except (OSError, TypeError):
        return None
    return "".join(lines), first


def _static_counts(src: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for m in _CALL_RE.finditer(src):
        out[m.group(1)] = out.get(m.group(1), 0) + 1
    return out


def class_collective_calls(cls: type, *, model_prefix: str | None = None) -> dict[str, int]:
    """{kind: static call count} of the collective names called in `cls`'s own source; {} for a class outside the model modules or one
    whose source cannot be read.  Inherited methods are not searched (a vLLM layer base class is not a model class)."""
    src = _model_class_source(cls, model_prefix)
    return _static_counts(src[0]) if src else {}


def _self_name(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> str | None:
    """The name `self` for an undecorated method whose first parameter is `self` and that never rebinds it, else None (no guard read)."""
    params = fn.args.posonlyargs + fn.args.args
    if fn.decorator_list or not params or params[0].arg != "self":
        return None
    if any(isinstance(n, ast.Name) and n.id == "self" and not isinstance(n.ctx, ast.Load) for n in ast.walk(fn)):
        return None
    return "self"


def _rebound_attrs(cdef: ast.ClassDef) -> set[str]:
    """Attribute names the class writes or deletes anywhere except as `self.<attr> = ...` directly in `__init__`."""
    out: set[str] = set()
    for stmt in cdef.body:
        direct: set[int] = set()
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)) and stmt.name == "__init__" and _self_name(stmt):
            nested = {id(x) for d in ast.walk(stmt) if d is not stmt and isinstance(d, _SCOPES) for x in ast.walk(d)}
            direct = {id(n) for n in ast.walk(stmt) if id(n) not in nested}
        for n in ast.walk(stmt):
            if isinstance(n, ast.Attribute) and not isinstance(n.ctx, ast.Load):
                if not (id(n) in direct and isinstance(n.value, ast.Name) and n.value.id == "self"):
                    out.add(n.attr)
    return out


def _guarded_calls(src: str, first: int) -> list[tuple[str, tuple[str, ...], int]] | None:
    """[(kind, the bare `if self.<attr>:` guards whose body encloses the call, source line)] per collective call in the class source;
    None when the source does not parse to one class or the class can write attributes reflectively."""
    try:
        tree = ast.parse(textwrap.dedent(src))
    except SyntaxError:
        return None
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.ClassDef):
        return None
    cdef = tree.body[0]
    if any(isinstance(n, ast.Name) and n.id in _REFLECTION or isinstance(n, ast.Attribute) and n.attr in _REFLECTION
           for n in ast.walk(cdef)):
        return None
    rebound = _rebound_attrs(cdef)
    out: list[tuple[str, tuple[str, ...], int]] = []

    def visit(node: ast.AST, guards: tuple[str, ...], self_name: str | None) -> None:
        if isinstance(node, ast.If):
            visit(node.test, guards, self_name)
            t = node.test
            bare = (self_name is not None and isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) and t.value.id == self_name
                    and t.attr not in rebound)
            for b in node.body:
                visit(b, guards + (t.attr,) if bare else guards, self_name)
            for b in node.orelse:
                visit(b, guards, self_name)
            return
        if isinstance(node, _SCOPES):
            guards, self_name = (), None
        if isinstance(node, ast.Call):
            f = node.func
            kind = _KINDS.get(f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else "")
            if kind is not None:
                out.append((kind, guards, first + node.lineno - 1))
        for ch in ast.iter_child_nodes(node):
            visit(ch, guards, self_name)

    for stmt in cdef.body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            body, self_name = {id(b) for b in stmt.body}, _self_name(stmt)
            for ch in ast.iter_child_nodes(stmt):
                visit(ch, (), self_name if id(ch) in body else None)
        else:
            visit(stmt, (), None)
    return out


def _class_calls(src: tuple[str, int] | None) -> tuple[dict[str, int], list[tuple[str, tuple[str, ...], int]] | None]:
    """(static counts, guarded calls or None).  The guards are trusted only when the parsed calls ARE the static count, kind by kind."""
    if src is None:
        return {}, None
    static = _static_counts(src[0])
    guarded = _guarded_calls(*src) if static else None
    if guarded is not None:
        parsed: dict[str, int] = {}
        for kind, _g, _line in guarded:
            parsed[kind] = parsed.get(kind, 0) + 1
        if parsed != static:
            guarded = None
    return static, guarded


def _falsy_on_instance(m: Any, attr: str) -> tuple[bool, Any]:
    """(True, value) when `attr` is in the instance's own `__dict__`, no class descriptor intercepts it, and it is False, None or 0."""
    d = getattr(m, "__dict__", None)
    if not isinstance(d, dict) or attr not in d:
        return False, None
    if hasattr(type(inspect.getattr_static(type(m), attr, None)), "__get__"):
        return False, None
    v = d[attr]
    return (v is None or v is False or (type(v) is int and v == 0)), v


def _under_tp(m: Any, world: int) -> bool:
    tp = getattr(m, "tp_size", None)
    return int(tp if tp is not None else world) > 1


def mid_module_site_census(model: Any, world: int, *, model_prefix: str | None = None) -> dict[str, list[dict[str, Any]]]:
    """{"sites": [{"name", "cls", "py_module", "calls": {kind: n}, "per_step": n_total}], "skipped": [{"name", "cls", "py_module", "kind",
    "attr", "value", "line", "site_dropped"}]}: every module of `model` whose class calls a TP collective directly, in `named_modules()`
    order, with the calls the instance can execute; each call a falsy bare `if self.<attr>:` rules out is listed in "skipped" (the
    module's python module stays a patch target), and a module none of whose calls is left is no site ("site_dropped")."""
    if int(world) <= 1:
        return {"sites": [], "skipped": []}
    sites: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    cache: dict[type, tuple[dict[str, int], list[tuple[str, tuple[str, ...], int]] | None]] = {}
    for name, m in model.named_modules():
        cls = type(m)
        if cls not in cache:
            cache[cls] = _class_calls(_model_class_source(cls, model_prefix))
        static, guarded = cache[cls]
        if not static or not _under_tp(m, world):
            continue
        calls, dropped = dict(static), []
        for kind, guards, line in guarded or ():
            for attr in guards:
                falsy, v = _falsy_on_instance(m, attr)
                if falsy:
                    calls[kind] -= 1
                    dropped.append({"kind": kind, "attr": attr, "value": repr(v), "line": line})
                    break
        calls = {k: n for k, n in calls.items() if n}
        where = {"name": name, "cls": cls.__name__, "py_module": cls.__module__}
        skipped += [{**where, **d, "site_dropped": not calls} for d in dropped]
        if calls:
            sites.append({**where, "calls": calls, "per_step": sum(calls.values())})
    return {"sites": sites, "skipped": skipped}


def mid_module_sites(model: Any, world: int, *, model_prefix: str | None = None) -> list[dict[str, Any]]:
    """The "sites" of `mid_module_site_census`.  Empty for a dense vLLM model (its collectives are the layer modules')."""
    return mid_module_site_census(model, world, model_prefix=model_prefix)["sites"]


def patch_targets(sites: list[dict[str, Any]]) -> list[tuple[Any, str, str]]:
    """The (python module, attribute, kind) triples the observers wrap for `sites`: each site class's module x each collective name it
    binds (deduplicated, in site order).  A module that does not bind a name is skipped (the class called it through another path).
    The observers pass the census's "skipped" entries too, so a skipped call that runs anyway is still seen."""
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
