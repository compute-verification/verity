"""Units known by their registered body: a unit is of a registered family only by that family's name and a registered body
of it at its statics (`family`), and a token select (`egress`) by either."""
from __future__ import annotations

import functools
from typing import Any

from verity.circuits.codec import definition_digest, definition_entry
from verity.circuits.defs import CompositeDefinition, SpecializedDefinition, bind

from .build import Unexpressible

#: the token-select families, whose registered composites `egress` holds a unit's name or body to: its sampled ids are a request
#: Program's egress (`Compiled.transfer_gates`)
EGRESS = ("TokenSelect", "GumbelTopPTokenSelect")


@functools.cache
def registered_composites(families: tuple[str, ...]) -> tuple[CompositeDefinition, ...]:
    """The registry's composites of ``families``."""
    from verity_vllm.definitions.registry.catalog import load

    reg, errors = load()
    if errors:
        raise ValueError(f"the registry's {', '.join(families)} can't be listed: {errors[0]}")
    return tuple(d for d in reg.defs.values() if isinstance(d, CompositeDefinition) and d.name in families)


def egress(fn: Any) -> bool:
    """Whether ``fn`` is a token select: named as an `EGRESS` family, or with a registered select's body under another name.
    A unit so named whose body is no registered select's at its statics is refused (ValueError), neither listed nor sampled."""
    family = str(getattr(fn, "id", "")).split("{")[0].rsplit("_v", 1)[0]
    if registered_body(fn, EGRESS):
        return True
    if family in EGRESS:
        raise ValueError(f"{fn.id}: a unit named as the token select {family} whose body is no registered select's at its "
                         "statics is refused: it is neither a transfer gate nor sampled")
    return False


def registered_body(fn: Any, families: tuple[str, ...]) -> bool:
    """Some registered composite of ``families`` bound at ``fn``'s statics, of ``fn``'s signature, has the definition digest
    (`codec.definition_digest`) of ``fn``'s entry under that composite's id."""
    if not isinstance(fn, SpecializedDefinition):
        return False
    if any(sel is fn.defn for sel in registered_composites(families)):
        return True
    mine = None
    for sel in registered_composites(families):
        if set(sel.statics) != set(fn.bindings):
            continue
        try:
            want = bind(sel, **fn.bindings)
        except (TypeError, ValueError):
            continue
        if want is fn:
            return True
        if want.signature() != fn.signature():
            continue
        mine = mine or definition_entry(fn)
        theirs = definition_entry(want)
        if definition_digest({**mine, "id": theirs["id"]}) == definition_digest(theirs):
            return True
    return False


@functools.cache
def _found(fn: SpecializedDefinition, families: tuple[str, ...]) -> tuple[str | None, str | None]:
    """``fn``'s family among ``families`` by its name, and by its registered body."""
    name = fn.name if fn.name in families else None
    if name is not None and registered_body(fn, (name,)):
        return name, name
    return name, next((f for f in families if registered_body(fn, (f,))), None)


def family(fn: Any, families: tuple[str, ...], why: str) -> str | None:
    """``fn``'s family among ``families``: its name, where a registered composite of that name has its body at its statics
    (`registered_body`); None where it has neither the name nor such a body. A name over another body or a registered body
    under another name is refused by name, since ``why``."""
    if isinstance(fn, SpecializedDefinition):
        named, body = _found(fn, families)
    else:
        named, body = (name if (name := getattr(fn, "name", None)) in families else None), None
    if named == body:
        return named
    raise Unexpressible(f"{fn.id}: " + (f"a unit named as {named} whose body is no registered {named}'s at its statics"
                                        if named is not None else f"a registered {body}'s body under another name") +
                        f" is refused: {why}")
