"""Every table a primitive reads by index is declared (`PrimitiveDefinition.tables`): the guard
(`verity.ml.circuits.operations.program_data`) sees a table only through its primitive's declaration, so an undeclared one passes it.

A primitive reads a table when its evaluator reaches an indexable constant (one it closes over, or a global or module
attribute of the code it calls) or a memoized loader that returns anything but a word, or when Flock's lowering of it reads
one (`Circuit.lookup` by name, or `fp.lookup`'s table laid into the linear forms). A table read by name must be declared under
that name; one without a name needs the primitive to declare at least one table. A lifted primitive runs its base primitive's
evaluator, so it reads, and declares, the base's tables."""
from __future__ import annotations

import array
import dis
import functools
import importlib
import itertools
import pkgutil
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from circuit_check import sidecars as SC
from circuit_check import targets as TG
from verity.ml.circuits.defs import REGISTRY, PrimitiveDefinition
from verity.ml.circuits.types import Array
from verity.ml.circuits.ops.prims import F32
from verity_flock import fp as FP
from verity_flock import gf2
from verity_flock import ir_lower as IL
from verity.ml.definitions.registry import lifted as L
from verity.ml.definitions import registry as R

#: the fewest entries an indexable constant has to be a table
MIN_ENTRIES = 16
#: the words a loader-reaching evaluator runs on: every region of an f32 rule (zero, subnormal, the tanh table's range, one,
#: large, inf, negative)
WORDS = (0x00000000, 0x00400000, 0x3A83126F, 0x3F400000, 0x3F800000, 0x40400000, 0x7E800000, 0x7F800000, 0xBF000000)
_LAID_IN = object()


def _is_table(v) -> bool:
    if isinstance(v, (bytes, bytearray, memoryview, array.array, np.ndarray)):
        return len(v) >= MIN_ENTRIES
    if isinstance(v, (list, tuple)):
        return len(v) >= MIN_ENTRIES and all(isinstance(x, (int, np.integer, list, tuple)) for x in v[:MIN_ENTRIES])
    if isinstance(v, dict):
        return len(v) >= MIN_ENTRIES and all(isinstance(x, (int, np.integer)) for x in list(v.values())[:MIN_ENTRIES])
    return False


def _codes(code: types.CodeType):
    yield code
    for k in code.co_consts:
        if isinstance(k, types.CodeType):
            yield from _codes(k)


def _referenced(f: types.FunctionType) -> list:
    """What `f`'s code can read: its closure, defaults, the globals it loads, and those of their module attributes it names."""
    out = [c.cell_contents for c in (f.__closure__ or ()) if c.cell_contents is not None]
    out += [*(f.__defaults__ or ()), *(f.__kwdefaults__ or {}).values()]
    mods, attrs = [], set()
    for code in _codes(f.__code__):
        for ins in dis.get_instructions(code):
            if ins.opname in ("LOAD_GLOBAL", "LOAD_NAME") and ins.argval in f.__globals__:
                v = f.__globals__[ins.argval]
                (mods if isinstance(v, types.ModuleType) else out).append(v)
            elif ins.opname == "LOAD_ATTR":
                attrs.add(ins.argval)
    return out + [getattr(m, a) for m in mods if m.__name__.startswith("verity") for a in sorted(attrs) if hasattr(m, a)]


def _reach(evaluate) -> tuple[int, list]:
    """How many indexable constants `evaluate` reaches through Verity's own code, and the memoized loaders it reaches."""
    seen: set[int] = set()
    tables, loaders, stack = 0, [], [evaluate]
    while stack:
        f = stack.pop()
        if id(f) in seen:
            continue
        seen.add(id(f))
        if _is_table(f):
            tables += 1
        elif isinstance(f, PrimitiveDefinition):
            stack.append(f.evaluate)
        elif isinstance(f, functools.partial):
            stack += [f.func, *f.args, *f.keywords.values()]
        elif hasattr(f, "cache_info") and hasattr(f, "__wrapped__"):
            loaders.append(f)
            stack.append(f.__wrapped__)
        elif isinstance(f, types.MethodType):
            stack.append(f.__func__)
        elif isinstance(f, types.FunctionType) and (f is evaluate or (f.__module__ or "").startswith("verity")):
            stack += _referenced(f)
    return tables, loaders


def _loaded(p: PrimitiveDefinition, loaders: list) -> set[tuple[int, str | None]]:
    """The tables `loaders` give `p`'s evaluator on WORDS, as (loader, name): a call that returns anything but a word, or
    raises, is a table read, named by its string argument (None without one)."""
    if not loaders:
        return set()
    assert not any(isinstance(t, Array) for _n, t in p.params), f"{p.id}: an array parameter (WORDS covers words only)"
    reads: set[tuple[int, str | None]] = set()

    def recorder(w):
        def call(*a, **k):
            read = (id(w), next((x for x in a if isinstance(x, str)), None))
            try:
                v = w(*a, **k)
            except Exception:
                reads.add(read)
                raise
            if not isinstance(v, (int, np.integer, float, bool, type(None))):
                reads.add(read)
            return v
        return call

    mods = [m for n, m in list(sys.modules.items()) if n.startswith("verity") or n == p.evaluate.__module__]
    with pytest.MonkeyPatch.context() as mp:
        for w in loaders:
            rec = recorder(w)
            for m in mods:
                for k, v in list(vars(m).items()):
                    if v is w:
                        mp.setattr(m, k, rec)
        masks = [(1 << t.leaf_widths()[0]) - 1 for _n, t in p.params]
        for ws in itertools.product(WORDS, repeat=len(masks)):
            try:
                p.evaluate(*(x & m for x, m in zip(ws, masks)))
            except Exception:
                pass
    return reads


def _lowered(p: PrimitiveDefinition) -> tuple[set[str], int]:
    """The tables Flock's lowering of `p` reads: the names `Circuit.lookup` reads, and how many tables `fp.lookup` lays in."""
    if not p.params or p.id not in IL.PIECES:
        return set(), 0
    laid: list[int] = []

    def lookup(C, idx, table, out_bits, lo_bits=None):
        laid.append(len(table))
        return C.lookup(_LAID_IN, idx, out_bits)

    C = gf2.Circuit()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(FP, "lookup", lookup)
        IL.lower(C, p, [C.input(f"in{j}", w) for j, w in enumerate(w for _n, t in p.params for w in t.leaf_widths())])
    return {n for n, _i, _w in C.lookups if n is not _LAID_IN}, sum(n >= MIN_ENTRIES for n in laid)


def _base(p: PrimitiveDefinition) -> PrimitiveDefinition:
    """The primitive whose evaluator `p` runs: a lift's (or its ordinary-operand form's) base, behind a wrapper that holds only
    operand widths and encodings, and that runs the base only under an enable no word of WORDS gives it."""
    while (q := getattr(p, "ordinary_of", None) or getattr(p, "lifted_from", None)) is not None and q is not p:
        p = q
    return p


def undeclared(p: PrimitiveDefinition) -> list[str]:
    """What `p` reads by index and does not declare."""
    tables, loaders = _reach(_base(p).evaluate)
    loads = _loaded(_base(p), loaders)
    named, laid = _lowered(p)
    named |= {n for _w, n in loads if n is not None}
    unnamed = tables + laid + sum(n is None for _w, n in loads)
    declared = {n for n, _kind in p.tables}
    out = [f"{p.id} reads the table {n!r} and does not declare it" for n in sorted(named - declared)]
    if unnamed and not declared:
        out.append(f"{p.id} reads {unnamed} unnamed table(s) and declares none")
    return out


def _primitives() -> list[PrimitiveDefinition]:
    """Every primitive Verity defines, the same in any session: the registries, every module of the vLLM registry (its quarantine
    too: a quarantined primitive is still registered and callable), and both lifts of each (made on demand, so a session holds
    only those its tests asked for); never one a test module registered."""
    TG.load_registries()
    root = Path(R.__path__[0])
    loaded = sorted(importlib.import_module(m.name).__name__ for m in pkgutil.walk_packages(R.__path__, f"{R.__name__}."))
    on_disk = {".".join((R.__name__, *f.relative_to(root).with_suffix("").parts)).removesuffix(".__init__")
               for f in root.rglob("*.py") if SC.SUFFIX not in f.name}
    assert loaded == sorted(on_disk - {R.__name__}), "a module of the vLLM registry was skipped"
    for p in [d for d in REGISTRY.defs.values() if isinstance(d, PrimitiveDefinition) and TG._authored(d)]:
        if not p.id.startswith("Lifted[") and not hasattr(p, "ordinary_of"):
            L.lifted_prim(p)
    return [d for d in REGISTRY.defs.values() if isinstance(d, PrimitiveDefinition) and TG._authored(d)]


@pytest.mark.slow
def test_every_registered_primitive_declares_the_tables_it_reads():
    """Lowers every primitive Flock has a piece for (about 30 s, most of it the Gumbel pieces): a table only the lowering
    reads, as SiluMulBf16's is, never shows in its evaluator."""
    prims = _primitives()
    assert len(prims) > 300
    assert [x for p in prims for x in undeclared(p)] == []


@functools.lru_cache(maxsize=None)
def _planted_table(name: str) -> tuple[int, ...]:
    return tuple((i * 0x9E3779B1) & 0xFFFFFFFF for i in range(256))


def test_a_planted_primitive_that_reads_an_undeclared_table_is_found(monkeypatch):
    """Each way of reading a table, planted in a primitive without the declaration, is found; with it, nothing is."""
    closed = tuple(range(256))
    monkeypatch.setitem(IL.PIECES, "KLooksUp_v1", lambda C, a: C.lookup("k_table", list(a[:8]), 32))
    monkeypatch.setitem(IL.PIECES, "KLaysIn_v1", lambda C, a: FP.lookup(C, list(a[:8]), closed, 32))
    planted = {"KClosesOver": (lambda x: closed[x & 0xFF], None),
               "KLoads": (lambda x: _planted_table("k_table")[x & 0xFF], "k_table"),
               "KLooksUp": (lambda x: x, "k_table"),
               "KLaysIn": (lambda x: x, None)}
    for name, (evaluate, table) in planted.items():
        def prim(tables=()):
            return PrimitiveDefinition(name, 1, (("x", F32),), F32, evaluate, register=False, tables=tables)
        assert undeclared(prim()) == [f"{name}_v1 reads the table {table!r} and does not declare it" if table else
                                      f"{name}_v1 reads 1 unnamed table(s) and declares none"], name
        assert undeclared(prim(((table or "k_table", "registered"),))) == [], name
