"""Which Definition is a census type's Boolean version, as the tree's registry says: the one pairing `verity bench prove`
names a row's form by and `spot_figures`' figure of record prices a census type's row by. It imports none of the
harness's other modules, so a reader of a table imports it alone."""

from __future__ import annotations


def boolean_version(block: str, definition: str) -> bool:
    """Whether the tree registers `definition` (a Definition's id) as the Boolean version of the census type `block`
    (its census name): the registry binds the type's Definition, and `verity.definitions.boolean.boolean_version` of
    it, the newest registered Boolean Definition that names it as its word and no caller has to opt in to, has that
    id. A type this tree's registry doesn't bind, or registers no Boolean version of, has none."""

    from verity.circuits.defs import CompositeDefinition, bind

    # registers `EmbeddingRead_v1`, a `READ` static's Definition, which `boolean.load` doesn't
    from verity.commitments.gates import embedding  # noqa: F401
    from verity.definitions import boolean
    from verity.definitions.call_scope import spec_base, spec_statics, specialization

    boolean.load()
    try:
        own = specialization(spec_base(block), spec_statics(block))
        return boolean.boolean_version(bind(own) if isinstance(own, CompositeDefinition) else own).id == definition
    except (LookupError, TypeError, ValueError):
        return False
