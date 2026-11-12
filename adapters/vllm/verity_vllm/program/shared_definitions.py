"""The definitions one compose shares: each decoded request Program's definitions re-pointed at one canonical object per id
(`canonicalize`), so the workload Program (`workload.WorkloadProgram`) encodes each definition once."""

from __future__ import annotations

from typing import Any

from verity.circuits.codec import definition_entry, descriptor_id
from verity.circuits.defs import (
    IdentityConflict,
    PrimitiveDefinition,
    SpecializedDefinition,
)

_CANON: dict[str, tuple[Any, dict]] = {}  # specialization id -> (the canonical decoded object, its v0 encoding); the DEFAULT canon
# the canon is the set of definitions ONE compose shares; a process composing several workloads (the
# reviewer's fixture runner, a test module) must not compare a later compose against an earlier one's definitions -> callers pass their
# own `canon` dict per compose (global_program.build does); the module-level dict serves callers that pass none (single-compose processes).


def canonicalize(dp, canon: dict[str, tuple[Any, dict]] | None = None) -> None:
    """Share definitions across components. Each descriptor decodes into FRESH objects, so a definition used by two request
    Programs (every layer's Gemm/RmsNorm/... specialization) would be two objects under one id: `encode_program` then proves
    them equal by re-encoding both for every pair (O(components²) encodes — 20+ min at B=8). Here each decoded definition is
    encoded ONCE (v0, self-contained), compared with the canonical one under its id (fail-closed: a different body under the
    same id is `IdentityConflict`, never first-wins), and every reference to it (node callees, function-valued statics) is
    re-pointed to the canonical object. Primitives come from the registry and are already shared."""
    if canon is None:
        canon = _CANON
    subst: dict[str, Any] = {}
    # pass 1: an encoding names callees by id, so it does not depend on object identity — compare first, re-point after
    for fid, fn in dp.defs.items():
        if isinstance(fn, PrimitiveDefinition):
            continue
        if fn is dp.root:
            # the request Program's ROOT is never shared: its body becomes the component's
            # (`WorkloadRequest{program_digest, program_fn}`, identity = the whole descriptor's digest) and nothing calls it. Its id is
            # derive's `Root[Derived[vLLMRequest[...]]{SRC=...}]` whose SRC static names class/mode/profile/INPUT SHAPES/config -- not T --
            # so two requests of one manifest with the same prompt length and different token limits (LP11: T24/T36) carry ONE root id over two
            # bodies, and comparing roots under that id would refuse a valid workload (E_IDENTITY) although their digests differ. Callees stay
            # under the fail-closed rule below; the root's identity is the component digest.
            continue
        enc = definition_entry(fn)
        c = canon.get(fid)
        if c is None:
            canon[fid] = (fn, enc)
        elif c[1] != enc:
            raise IdentityConflict(f"E_IDENTITY: request Programs disagree on the definition {fid!r}")
        elif c[0] is not fn:
            subst[fid] = c[0]
    if not subst:
        return
    # pass 2: every node callee / function-valued static of this decode now points at the canonical object
    for fn in list(dp.defs.values()) + [dp.root]:
        if not isinstance(fn, PrimitiveDefinition):
            repoint(fn, subst)


def repoint(fn, subst: dict[str, Any]) -> None:
    """Re-point `fn`'s node callees and function-valued statics at their canonical objects (`subst`, by `descriptor_id`), in
    place. Each substitute must have the id it replaces (`_canonical`): a complete specialization's id is formatted once
    from its bindings, so a static swapped for another id would leave it stale."""
    if fn._body is None:
        return
    for n in fn.body.nodes:
        c = _canonical(n.fn, subst)
        if c is not n.fn:
            n.fn = c
    for k, v in list(fn.bindings.items()):
        fn.bindings[k] = _subst_static(v, subst)


def _canonical(v, subst):
    """`v`'s substitute in `subst`, or `v` itself; refused (IdentityConflict) before it is applied when its id is not `v`'s."""
    key = descriptor_id(v)
    c = subst.get(key, v)
    if c is not v and descriptor_id(c) != key:
        raise IdentityConflict(f"E_IDENTITY: {key!r} would be re-pointed to a definition of another id, {descriptor_id(c)!r}")
    return c


def _subst_static(v, subst):
    if isinstance(v, SpecializedDefinition):
        return _canonical(v, subst)
    if isinstance(v, PrimitiveDefinition):
        return v
    if isinstance(v, tuple):
        return tuple(_subst_static(x, subst) for x in v)
    if isinstance(v, list):
        return [_subst_static(x, subst) for x in v]
    if isinstance(v, dict):
        return {k: _subst_static(x, subst) for k, x in v.items()}
    return v
