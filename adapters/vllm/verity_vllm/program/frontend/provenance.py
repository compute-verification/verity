"""The registry's version, which every derivation's provenance holds.

    Gate -> exactly one Value<w>.
    Composite interfaces -> ordered collections of value references.
    Commitment positions -> physical authentication locations for those values,
                            not semantic components of gate outputs.

* `registry_version()` — the fixed interpretation of the primitive vocabulary a program refers to. The program
  digest records primitives by id and signature and the codec retrieves their evaluators from the registry at
  decode time, so the digest alone does not pin them (§8.1). This record does: the ids, signatures and
  conformance statuses of every registered primitive, the ids and static parameters of every registered
  composite, and the SHA-256 of the registry sources that define the evaluators. It is part of every
  derivation's provenance; conformance evidence should be keyed by (program digest, specialisation, this version).
"""

from __future__ import annotations

import hashlib
import inspect

from verity.circuits.codec import canonical_json
from verity.circuits.defs import CompositeDefinition, PrimitiveDefinition


def registry_version() -> dict:
    """The primitive-vocabulary version: ids + signatures + conformance of every primitive, ids + statics of
    every composite, and the SHA-256 of the registry sources; `digest` is the SHA-256 of that record."""
    from verity.definitions.registry import b1, prims  # registration
    # The record is computed over the DECLARED vocabulary (module-level definitions of `registry.prims` / `registry.b1`),
    # not over `registry.defs` at call time: after a program is built the registry also holds shape-instantiated
    # primitives (`GatherBf16x49152_v1`, `Const32[…]`, `Input16_v1`) and the digest would depend on process state
    # (R-CAPCORR-1).
    prim_rows = []
    comp_rows = []
    seen: set[str] = set()
    for mod, cls in ((prims, PrimitiveDefinition), (b1, CompositeDefinition)):
        for _n, d in inspect.getmembers(mod):
            if not isinstance(d, cls) or d.id in seen:
                continue
            seen.add(d.id)
            if isinstance(d, PrimitiveDefinition):
                if getattr(d, "family", None) in ("Const", "Input"):
                    continue  # families with an unbounded id space; their meaning is fixed by the constructor
                prim_rows.append([d.id, [[n, t.to_json()] for n, t in d.params], d.ret.to_json(), d.conformance])
            else:
                comp_rows.append([d.id, list(d.statics), d.conformance])
    prim_rows.sort(key=lambda r: r[0])
    comp_rows.sort(key=lambda r: r[0])
    sources = {}
    for mod in (prims, b1):
        src = inspect.getsource(mod).encode()
        sources[mod.__name__] = hashlib.sha256(src).hexdigest()
    rec = {"schema": "verity-ir/registry-version/v0", "primitives": prim_rows, "composites": comp_rows, "sources": sources}
    digest = hashlib.sha256(canonical_json(rec)).hexdigest()
    return {"schema": rec["schema"], "digest": digest, "primitives": len(prim_rows), "composites": len(comp_rows), "sources": sources}


__all__ = ["registry_version"]
