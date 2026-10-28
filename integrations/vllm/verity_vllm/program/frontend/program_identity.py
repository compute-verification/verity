"""§8.1 over a whole program: a fully qualified specialisation (`Name_vN{statics}`) resolves to exactly one body, within the
program and against the registry."""

from __future__ import annotations

from typing import Any

from verity.ml.circuits.codec import descriptor_id
from verity.ml.circuits.defs import REGISTRY, CompositeDefinition, PrimitiveDefinition, SpecializedDefinition
from verity.ml.circuits.defs import bind as ir_bind
from verity.ml.circuits.program import Program
from verity_vllm.program.frontend.correspond import compare_bodies
from verity_vllm.program.frontend.errors import FrontendError


def _visit_static(v, visit) -> None:
    if isinstance(v, (PrimitiveDefinition, SpecializedDefinition)):
        visit(v)
    elif isinstance(v, (list, tuple)):
        for x in v:
            _visit_static(x, visit)
    elif isinstance(v, dict):
        for x in v.values():
            _visit_static(x, visit)


def check_identity(program: Program, *, registry=REGISTRY, check_registry: bool = True) -> dict[str, SpecializedDefinition | PrimitiveDefinition]:
    """§8.1: within a program a fully qualified specialisation (`Name_vN{statics}`) resolves to exactly one
    body. Two reachable definition objects with the same id must encode the same body; with `check_registry`
    an authored definition whose id a *registered* definition also carries must agree with it as well —
    conflicts are `E_IDENTITY`, never first-wins. Returns the reachable definitions by id."""
    seen: dict[str, Any] = {}

    def visit(fn) -> None:
        fid = descriptor_id(fn)
        prev = seen.get(fid)
        if prev is not None:
            if prev is fn or isinstance(fn, PrimitiveDefinition):
                if isinstance(fn, PrimitiveDefinition) and prev is not fn and (prev.params != fn.params or prev.ret != fn.ret):
                    raise FrontendError("E_IDENTITY", f"two primitives claim {fid} with different signatures")
                return
            v = compare_bodies(prev.body, fn.body, fid=fid)
            if v.verdict not in ("identical", "same-references"):
                raise FrontendError("E_IDENTITY", f"two different bodies claim {fid} within one program "
                                                  f"({prev.defn!r} and {fn.defn!r}; {v.verdict}: {v.detail}); give one of them a new version")
            return
        seen[fid] = fn
        if isinstance(fn, PrimitiveDefinition):
            return
        for n in fn.body.nodes:
            visit(n.fn)
        for v in fn.bindings.values():
            _visit_static(v, visit)

    visit(program.root)
    visit(program.fn)  # the program function itself is inlined into the root; check its identity too
    if check_registry:
        for fid, fn in list(seen.items()):
            if isinstance(fn, PrimitiveDefinition):
                reg = registry.defs.get(fn.id)
                if reg is not None and reg is not fn and (reg.params != fn.params or reg.ret != fn.ret):
                    raise FrontendError("E_IDENTITY", f"primitive {fid} differs from the registered primitive of that id")
                continue
            reg = registry.defs.get(fn.base_id)
            if reg is None or reg is fn.defn or not isinstance(reg, CompositeDefinition):
                continue
            try:
                rsd = ir_bind(reg, **fn.bindings)
                v = compare_bodies(rsd.body, fn.body, fid=fid)
            except FrontendError:
                raise
            except Exception as e:  # noqa: BLE001  (incompatible statics / signature)
                raise FrontendError("E_IDENTITY", f"{fid} is registered with an incompatible definition: {type(e).__name__}: {e}") from None
            if v.verdict not in ("identical", "same-references"):
                raise FrontendError("E_IDENTITY", f"the authored body of {fid} differs from the registered definition's body "
                                                  f"({v.verdict}: {v.detail}); the id denotes one body — give the authored function a new version")
    return seen
