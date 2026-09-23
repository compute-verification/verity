"""The leaf-scheme registry: ``get(name)`` -> a FRESH :class:`LeafScheme` instance (one per compose: the gadget may record
row indices for its witness hook), ``by_schema(schema)`` for the statement reader, ``register(factory)`` for new schemes.

Names are additive.  ``"hash"`` is an alias of ``"poseidon2"`` (``--auth included-hash`` and the Rust pin name
``<rel>+hash`` predate the registry); :data:`DEFAULT` is the scheme a v5 statement names when its relation string carries
no ``+<leaf>`` suffix.
"""
from __future__ import annotations

from collections.abc import Callable

from .base import LeafScheme, check_scheme

__all__ = ["ALIASES", "DEFAULT", "by_schema", "canonical", "get", "names", "register", "relation_name", "split_relation_name"]

DEFAULT = "poseidon2"
ALIASES: dict[str, str] = {"hash": "poseidon2"}
_FACTORIES: dict[str, Callable[[], LeafScheme]] = {}
_PROTOTYPES: dict[str, LeafScheme] = {}


def register(factory: Callable[[], LeafScheme], *, test_only: bool = False) -> None:
    """Register a scheme by its factory; the prototype instance's ``name`` is the key.  ``test_only`` schemes (the dummy
    of ``leaf/dummy.py``) are registered like any other but flagged so a runner refuses to bench them."""
    proto = factory()
    check_scheme(proto)
    if proto.name in ALIASES:
        raise ValueError(f"leaf scheme name {proto.name!r} is an alias")
    if proto.name in _FACTORIES and _FACTORIES[proto.name] is not factory:
        raise ValueError(f"leaf scheme {proto.name!r} registered twice")
    proto.test_only = bool(test_only)   # type: ignore[attr-defined]
    _FACTORIES[proto.name] = factory
    _PROTOTYPES[proto.name] = proto


def canonical(name: str | None) -> str:
    """The registry name of ``name`` (aliases resolved; ``None`` / "" = :data:`DEFAULT`)."""
    n = name or DEFAULT
    return ALIASES.get(n, n)


def _ensure_builtin() -> None:
    if DEFAULT not in _FACTORIES:
        from . import poseidon2  # noqa: F401  (registers itself on import)


def get(name: str | None = None) -> LeafScheme:
    """A fresh instance of the scheme ``name`` (alias-resolved)."""
    _ensure_builtin()
    n = canonical(name)
    try:
        s = _FACTORIES[n]()
    except KeyError:
        raise KeyError(f"unknown leaf scheme {name!r}; known: {names()}") from None
    s.test_only = _PROTOTYPES[n].test_only   # type: ignore[attr-defined]
    return s


def names() -> list[str]:
    _ensure_builtin()
    return sorted(_FACTORIES)


def by_schema(schema: str) -> LeafScheme:
    """The scheme whose leaf ``schema`` string this is (the statement reader's check that the block names the scheme the
    relation suffix names)."""
    _ensure_builtin()
    for n, proto in _PROTOTYPES.items():
        if proto.schema == schema:
            return get(n)
    raise KeyError(f"no leaf scheme has schema {schema!r}")


def split_relation_name(name: str) -> tuple[str, str]:
    """``"fp8-ada+blake3"`` -> ``("fp8-ada", "blake3")``; ``"fp8-ada"`` / ``"fp8-ada+hash"`` -> ``("fp8-ada", "poseidon2")``.
    ``ValueError`` on a malformed name, ``KeyError`` on a suffix no registered scheme answers to."""
    base, _, leaf = name.partition("+")
    if not base or "+" in leaf:
        raise ValueError(f"relation name {name!r}: expected <relation>[+<leaf>]")
    _ensure_builtin()
    leaf = canonical(leaf)
    if leaf not in _FACTORIES:
        raise KeyError(f"relation {name!r} names leaf scheme {leaf!r}; known: {names()}")
    return base, leaf


def relation_name(base: str, leaf: str | None) -> str:
    """The statement's relation string: ``base`` for the default scheme (statement v5 compatibility), else ``base+leaf``."""
    n = canonical(leaf)
    return base if n == DEFAULT else f"{base}+{n}"
