"""The leaf-scheme registry: ``get(name)`` -> a FRESH :class:`LeafScheme` instance (one per compose: the gadget may record
row indices for its witness hook), ``by_schema(schema)`` for the statement reader, ``register(factory)`` for new schemes.

Names are additive.  ``"hash"`` is an alias of ``"poseidon2"`` (``--auth included-hash`` and the Rust pin name
``<rel>+hash`` predate the registry); :data:`DEFAULT` is the scheme a v5 statement names when its relation string carries
no ``+<leaf>`` suffix.
"""
from __future__ import annotations

from collections.abc import Callable

from .base import LeafScheme, check_scheme

__all__ = ["ALIASES", "BUILTIN", "DEFAULT", "by_schema", "canonical", "get", "names", "register", "register_family", "relation_name",
           "split_relation_name", "suffix_of", "families"]

DEFAULT = "poseidon2"
ALIASES: dict[str, str] = {"hash": "poseidon2"}
_FACTORIES: dict[str, Callable[[], LeafScheme]] = {}
_PROTOTYPES: dict[str, LeafScheme] = {}
_FAMILIES: dict[str, tuple[Callable[[str], LeafScheme | None], bool]] = {}


def register(factory: Callable[[], LeafScheme], *, test_only: bool = False) -> None:
    """Register a scheme by its factory; the prototype instance's ``name`` is the key.  ``test_only`` schemes (the dummy
    of ``leaf/dummy.py``) are registered like any other but flagged so a runner refuses to bench them."""
    proto = factory()
    check_scheme(proto)
    if proto.name in ALIASES or proto.name in _FAMILIES:
        raise ValueError(f"leaf scheme name {proto.name!r} is an alias or a family prefix")
    if proto.name in _FACTORIES and _FACTORIES[proto.name] is not factory:
        raise ValueError(f"leaf scheme {proto.name!r} registered twice")
    proto.test_only = bool(test_only)   # type: ignore[attr-defined]
    _FACTORIES[proto.name] = factory
    _PROTOTYPES[proto.name] = proto


def canonical(name: str | None) -> str:
    """The registry name of ``name`` (aliases resolved; ``None`` / "" = :data:`DEFAULT`)."""
    n = name or DEFAULT
    return ALIASES.get(n, n)


#: the in-tree scheme modules (each registers itself on import); a scheme whose optional dependency is missing is skipped.
#: The test-only ``dummy`` is NOT here: a test imports ``leaf.dummy`` explicitly.
BUILTIN = ("poseidon2", "blake3", "blake3_xob", "sha256", "ajtai")


def _ensure_builtin() -> None:
    import importlib
    for n in BUILTIN:
        if n not in _FACTORIES:
            try:
                importlib.import_module(f".{n}", __package__)
            except ImportError:
                if n == DEFAULT:
                    raise


def register_family(prefix: str, resolver: Callable[[str], LeafScheme | None], *, test_only: bool = False) -> None:
    """A PARAMETRIC scheme family: ``resolver(name)`` returns a fresh instance for ``prefix`` or ``prefix-<params>``
    (``"ajtai-n64"``), or ``None`` when the parameters are not admissible.  The instance's :func:`suffix_of` is the relation
    suffix it travels under (its ``tag`` attribute when it has one, else its ``name``)."""
    if prefix in ALIASES or prefix in _FACTORIES:
        raise ValueError(f"leaf family prefix {prefix!r} is taken")
    _FAMILIES[prefix] = (resolver, bool(test_only))


def _resolve(n: str) -> LeafScheme | None:
    if n in _FACTORIES:
        s = _FACTORIES[n]()
        s.test_only = _PROTOTYPES[n].test_only   # type: ignore[attr-defined]
        return s
    for prefix, (resolver, test_only) in _FAMILIES.items():
        if n == prefix or n.startswith(prefix + "-"):
            s = resolver(n)
            if s is not None:
                check_scheme(s)
                s.test_only = test_only   # type: ignore[attr-defined]
                return s
    return None


def get(name: str | None = None) -> LeafScheme:
    """A fresh instance of the scheme ``name`` (alias-resolved; a registered name or a family member such as ``ajtai-n64``)."""
    _ensure_builtin()
    n = canonical(name)
    s = _resolve(n)
    if s is None:
        raise KeyError(f"unknown leaf scheme {name!r}; known: {_known()}")
    return s


def names() -> list[str]:
    """The registered (non-parametric) scheme names; see :func:`families` for the parametric ones."""
    _ensure_builtin()
    return sorted(_FACTORIES)


def families() -> list[str]:
    """The registered family prefixes (a member is ``<prefix>-<params>``, e.g. ``ajtai-n64``)."""
    _ensure_builtin()
    return sorted(_FAMILIES)


def _known() -> str:
    return ", ".join(names() + [f"{p}-<params>" for p in families()])


def by_schema(schema: str) -> LeafScheme:
    """A scheme whose leaf ``schema`` string this is, the first registered (``blake3`` and ``blake3-xob`` share one: the
    same leaf, two circuits).  Registered schemes only (a family member is found through its relation suffix)."""
    _ensure_builtin()
    for n, proto in _PROTOTYPES.items():
        if proto.schema == schema:
            return get(n)
    raise KeyError(f"no leaf scheme has schema {schema!r}")


def suffix_of(scheme: LeafScheme) -> str:
    """The relation suffix a scheme instance travels under: its ``tag`` (parametric families: ``"ajtai-n64"``) or ``name``."""
    return getattr(scheme, "tag", None) or scheme.name


def split_relation_name(name: str) -> tuple[str, str]:
    """``"fp8-ada+blake3"`` -> ``("fp8-ada", "blake3")``; ``"fp8-ada"`` / ``"fp8-ada+hash"`` -> ``("fp8-ada", "poseidon2")``;
    ``"fp8-ada+ajtai-n64"`` -> ``("fp8-ada", "ajtai-n64")``.  ``ValueError`` on a malformed name, ``KeyError`` on a suffix
    no registered scheme or family answers to."""
    base, _, leaf = name.partition("+")
    if not base or "+" in leaf:
        raise ValueError(f"relation name {name!r}: expected <relation>[+<leaf>]")
    _ensure_builtin()
    leaf = canonical(leaf)
    if leaf not in _FACTORIES and _resolve(leaf) is None:
        raise KeyError(f"relation {name!r} names leaf scheme {leaf!r}; known: {_known()}")
    return base, leaf


def relation_name(base: str, leaf: str | LeafScheme | None) -> str:
    """The statement's relation string: ``base`` for the default scheme (statement v5 compatibility), else ``base+<suffix>``
    (``leaf``: a name, a scheme instance, or ``None`` = the default)."""
    n = canonical(leaf if leaf is None or isinstance(leaf, str) else suffix_of(leaf))
    return base if n == DEFAULT else f"{base}+{n}"
