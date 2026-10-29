"""The operations: the composite Definitions whose own definition is gates, and the guard that a Program holds no data.

An operation's own definition is gates (the floating-point formats' constants, the round
constants of SHA-512 and Keccak-f, and every measured MUFU table in `verity.ml.circuits.library`: exp2, rcp, sqrt, rsq, tanh and lg2), and
everything a program or a request brings is a value, committed under a hiding leaf and always hidden (`verity.ml.circuits.constants`).
`verity.ml.circuits` names no operations, so the list is here and the guard passes it in:

    program_data(root) = verity.ml.circuits.constants.program_data(root, is_operation, is_library_table)

lists every constant held outside an operation and every table a primitive holds that is not a library table. No statement
or table claims that a Program's values are always hidden until the guard returns {} for it, and
`Program(fn, strict=program_data)` refuses a Program that holds data.

An entry is a composite Definition, by its id ``Name_vN``, whose every constant is the operation's own bits. A specialization
of one is an operation when it is the registry's own Definition, so a decoded Program (whose composites the descriptor
rebuilds) lists their constants, and when every function it is bound to holds no data itself. A primitive is never on the
list: it has no body, and the tables it reads are its own declaration (`PrimitiveDefinition.tables`).

A statement may make a Program's input public (a public input port, `verity_flock.circuit.compose`'s ``public_inputs``) only
when its value is the verifier's own recomputation from public data: the port names its seed and derivation, the verifier
derives the words itself and refuses any other. The Program declares nothing new:
the port is one of its run inputs. `public_data` is the guard's rule for such ports: a port whose derivation is not on
`DERIVATIONS` is data, so a statement with public ports claims its values hidden only when `program_data(root)` and
`public_data(public_inputs)` both return {}.

The lists are fixed and versioned. Adding or removing an entry is a new `VERSION`, and so a new `digest()`.
"""
from __future__ import annotations

import hashlib
from typing import Any, Iterator

from verity.ml.circuits import constants as K
from verity.ml.circuits.codec import canonical_json
from verity.ml.circuits.defs import REGISTRY, PrimitiveDefinition, SpecializedDefinition

from . import library

TAG = "verity/operations"
VERSION = 2

OPERATIONS: frozenset[str] = frozenset({
    # the floating-point formats' Boolean lowerings (`verity.ml.circuits.ops.boolean`), MUFU.EX2's table among them
    "Bf16ToF32_v2", "F32Fabs_v2", "MufuEx2Ftz_v2",
    # MUFU.LG2's table on bits, the library's `lg2` (the vLLM integration's `registry.boolean_dsv4_moe`)
    "MufuLg2Mantissa_v2",
    # the tensor-core dot products and GEMM coordinates (`verity.ml.circuits.ops.gemm`, `verity.ml.circuits.ops.boolean.gemm`): the accumulator's +0
    "DotBf16_v2", "DotBf16_v3", "GemmCoordinate_v3", "GemmCoordinateE4m3_v1", "GemmCoordinateE4m3_v2",
    "GemmCoordinateMxf4_v1", "GemmCoordinateMxf4_v2", "GemmCoordinateNvf4_v1", "GemmCoordinateNvf4_v2",
    # SHA-512's compression function and the Keccak-f[1600] permutation, with their round constants (PoUW's circuit,
    # `verity.core.protocols.pouw.circuit.hashes`)
    "PouwKeccakF1600_v1", "PouwSha512Compress_v1",
})

#: the recomputations a verifier runs from public data, by derivation id, which a public input port may carry. None is listed
#: yet, so every public port is data; the first entry enters `spec()`, and so `digest()`.
DERIVATIONS: frozenset[str] = frozenset()

_LIBRARY_TABLES = frozenset(t.name for t in library.TABLES)


def is_library_table(name: str) -> bool:
    """A table on `verity.ml.circuits.library`'s list, by its name."""
    return name in _LIBRARY_TABLES


def _functions(v: Any) -> Iterator[Any]:
    if isinstance(v, (PrimitiveDefinition, SpecializedDefinition)):
        yield v
    elif isinstance(v, (tuple, list)):
        for x in v:
            yield from _functions(x)
    elif isinstance(v, dict):
        for x in v.values():
            yield from _functions(x)


class _Guard:
    """One guard pass, with what it found about each function a specialization is bound to."""

    def __init__(self) -> None:
        self._holds_none: dict[str, tuple[Any, bool]] = {}

    def operation(self, fn: Any) -> bool:
        base, defn = getattr(fn, "base_id", None), getattr(fn, "defn", None)
        if base not in OPERATIONS or defn is None or REGISTRY.defs.get(base) is not defn:
            return False
        return all(self.holds_none(g) for g in _functions(tuple(fn.bindings.values())))

    def holds_none(self, fn: Any) -> bool:
        if isinstance(fn, PrimitiveDefinition):
            return not K.held_tables(fn, is_library_table)
        hit = self._holds_none.get(fn.id)
        if hit is None or hit[0] is not fn:
            hit = self._holds_none[fn.id] = (fn, self.operation(fn) or not self.data(fn))
        return hit[1]

    def data(self, root: Any) -> dict[str, list[tuple[int, str]]]:
        return K.program_data(root, self.operation, is_library_table)


def is_operation(fn: Any) -> bool:
    """Whether `fn`, a Definition a body calls, is an operation."""
    return _Guard().operation(fn)


def program_data(root: Any) -> dict[str, list[tuple[int, str]]]:
    """The guard: the data `root` (a Program's root Definition) still holds, as `verity.ml.circuits.constants.program_data` lists it
    under this list and the library's tables. {} exactly when the Program holds no data."""
    return _Guard().data(root)


def public_data(public: dict[str, Any]) -> dict[str, list[tuple[int, str]]]:
    """The guard's rule for a statement's public input ports (``{port: {"seed", "derivation"}}``): the data they hold, keyed
    ``public input <port>``, [(0, the derivation it names)]. A port is no data only when it names a non-empty seed and a
    derivation on `DERIVATIONS`. {} exactly when every public port is the verifier's own recomputation."""
    out = {}
    for port, r in public.items():
        seed, derivation = (r.get("seed"), r.get("derivation")) if isinstance(r, dict) else (None, None)
        if not (isinstance(seed, str) and seed and derivation in DERIVATIONS):
            out[f"public input {port}"] = [(0, derivation if isinstance(derivation, str) else "")]
    return out


def spec() -> dict:
    """The lists as canonical data: what `digest` hashes (`DERIVATIONS` once it lists one)."""
    return {"format": TAG, "version": VERSION, "operations": sorted(OPERATIONS),
            **({"derivations": sorted(DERIVATIONS)} if DERIVATIONS else {})}


def digest() -> str:
    """``SHA-512(TAG || "/v" || version || 0 || canon(spec))``: the list version a statement can pin."""
    return hashlib.sha512(f"{TAG}/v{VERSION}".encode() + b"\0" + canonical_json(spec())).hexdigest()


__all__ = ["DERIVATIONS", "OPERATIONS", "TAG", "VERSION", "digest", "is_library_table", "is_operation", "program_data",
           "public_data", "spec"]
