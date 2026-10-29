"""`targets._authored`: a Definition is this tree's by the file of the module that defines it, so a package that moves to a new
directory and import name stays in `REGISTERED`, and one whose module can't be judged fails by name.  `targets.root_functions`:
every root function a binding file defines runs, the listed ones first.  `targets.definition`: a Definition-valued static in an
id parses to the Definition it names, bound, recursively, and a record static to the record a catalog root binds, without the
catalog.  `targets.word_call_families`: a vocabulary kind that doesn't bind is refused by name."""
from __future__ import annotations

import sys
import types

import pytest

from circuit_check import targets as TG


def _definition(module: str, id: str = "Probe_v1"):
    def evaluate(*args):
        return args

    evaluate.__module__ = module
    return types.SimpleNamespace(id=id, evaluate=evaluate)


def _module(monkeypatch, name: str, file) -> None:
    mod = types.ModuleType(name)
    mod.__file__ = None if file is None else str(file)
    monkeypatch.setitem(sys.modules, name, mod)


def test_a_definition_is_the_trees_by_its_modules_file(monkeypatch):
    _module(monkeypatch, "catalog.silicon", TG.REPO / "catalog" / "src" / "catalog" / "silicon.py")
    assert TG._authored(_definition("catalog.silicon"))
    _module(monkeypatch, "tests.test_probe", TG.REPO / "catalog" / "tests" / "test_probe.py")
    assert not TG._authored(_definition("tests.test_probe"))
    _module(monkeypatch, "conftest", TG.REPO / "catalog" / "conftest.py")
    assert not TG._authored(_definition("conftest"))
    _module(monkeypatch, "elsewhere", TG.REPO.parent / "other-checkout" / "verity" / "x.py")
    assert not TG._authored(_definition("elsewhere"))


def test_a_definition_whose_module_has_no_file_is_refused_by_id(monkeypatch):
    _module(monkeypatch, "fileless", None)
    with pytest.raises(LookupError, match="'Fileless_v1'"):
        TG._authored(_definition("fileless", "Fileless_v1"))
    with pytest.raises(LookupError, match="'Unloaded_v1'"):
        TG._authored(_definition("never.imported.anywhere", "Unloaded_v1"))
    with pytest.raises(LookupError, match="'Bare_v1'"):
        TG._authored(types.SimpleNamespace(id="Bare_v1"))


def _binding_files(monkeypatch, **sidecars: dict) -> None:
    """Stand-in sidecars, each a binding file of these names (`sidecars.binding_modules`), and no registries."""
    from circuit_check import sidecars as SC

    mods = []
    for name, ns in sidecars.items():
        m = types.ModuleType(f"{name}.circuit-check")
        m.__file__ = f"{name}.circuit-check.py"
        vars(m).update(ns)
        mods.append((name, m))
    monkeypatch.setattr(TG, "load_registries", lambda: None)
    monkeypatch.setattr(SC, "binding_modules", lambda: mods)


def test_a_sidecars_own_root_function_runs_after_the_listed_ones_by_name(monkeypatch):
    """A root function need not be listed in `ROOT_FUNCTIONS` (one file per binding, no shared list to edit): any ``_X_roots``
    a binding file defines runs, after the listed ones, by name."""
    roots = lambda: []  # noqa: E731
    _binding_files(monkeypatch, a={"_zeta_roots": roots, "_core_roots": roots}, b={"_alpha_roots": roots, "_alpha_bindings": roots})
    assert TG.root_functions() == (*TG.ROOT_FUNCTIONS, "_alpha_roots", "_zeta_roots")
    assert [units for units, _make in TG._root_bindings("_alpha_roots")] == [("b.circuit-check#_alpha_roots",
                                                                            "b.circuit-check#_alpha_bindings")]


def test_a_sidecars_bindings_without_their_root_function_is_refused(monkeypatch):
    _binding_files(monkeypatch, a={"_lone_bindings": lambda: []})
    with pytest.raises(ValueError, match="_lone_bindings is no binding circuit-check runs"):
        TG.binding_files()


def _no_catalog(monkeypatch) -> None:
    monkeypatch.setattr(TG, "suite", lambda: pytest.fail("the catalog was built"))


def _registered(monkeypatch, name: str, statics: tuple[str, ...]):
    """A composite of these statics, registered for this test only; its body is never built."""
    from verity.ml.circuits.defs import REGISTRY, CompositeDefinition

    d = CompositeDefinition(name, 1, statics, lambda S: pytest.fail("signature built"), lambda B, S: pytest.fail("body built"),
                            register=False)
    monkeypatch.setitem(REGISTRY.defs, d.id, d)
    return d


def test_a_definition_valued_static_parses_to_the_definition_it_names_recursively(monkeypatch):
    """``EXACT=GemmAccumulatorS8_v2{...}`` in an id is not bound as its text (which fails with ``'str' object has no
    attribute 'signature'``).  A Definition-valued static with statics of its own parses to that Definition bound to them, a
    Definition among those the same way, in a tuple too, beside a bare id, a literal and a tuple of literals; without the
    catalog."""
    from verity.ml.circuits.defs import REGISTRY, bind
    from verity.ml.circuits.ops.boolean import gemm as BG

    TG.load_registries()
    _no_catalog(monkeypatch)
    exact = bind(BG.GemmAccumulatorS8, K=64, **BG.GEMM_SM120_S8)
    mid = bind(_registered(monkeypatch, "CircuitCheckMid", ("IN", "N")), IN=exact, N=2)
    outer = bind(_registered(monkeypatch, "CircuitCheckOuter", ("EXACT", "BOTH", "PAIRS", "DOT", "NAME")),
                 EXACT=mid, BOTH=(REGISTRY.defs["F32Add_v1"], bind(BG.GemmAccumulatorS8, K=32, **BG.GEMM_SM120_S8), 3), PAIRS=((0, 1), (1, 0)),
                 DOT=BG.BlackwellS8ImmaDot32, NAME="tanh")
    assert outer.id == ("CircuitCheckOuter_v1{EXACT=CircuitCheckMid_v1{IN=GemmAccumulatorS8_v2{K=64,DOT=BlackwellS8ImmaDot32_v2},"
                        "N=2},BOTH=(F32Add_v1,GemmAccumulatorS8_v2{K=32,DOT=BlackwellS8ImmaDot32_v2},3),PAIRS=((0,1),(1,0)),"
                        "DOT=BlackwellS8ImmaDot32_v2,NAME='tanh'}")
    got = TG.definition(outer.id)
    assert got is outer
    assert got.bindings["EXACT"] is mid and mid.bindings["IN"] is exact
    assert TG.definition(mid.id) is mid and TG.definition(exact.id) is exact


def test_the_pouw_h1_tile_roots_parse_from_their_ids(monkeypatch):
    """`PouwH1Tile`'s roots (its sidecar's bindings of each version: ``EXACT=GemmAccumulatorS8_v2{K=64,DOT=...}`` and
    the tile's cells) can be targeted by id: each parses to the very Definition the sidecar binds, without building it
    or the catalog. The sidecar's last root is `TemplateInstances_v1` of the v1 tile, which is not a tile."""
    TG.load_registries()
    _no_catalog(monkeypatch)
    (make,) = [make for units, make in TG._root_bindings("_pouw_roots") if units[0].startswith(
        "verity.core.protocols.pouw.circuit.h1tile.circuit-check#")]
    roots = make()
    if not roots:
        pytest.skip("PouwH1Tile is not registered on this tree (no circuit.segment)")
    assert len({root.id for root in roots}) == len(roots)
    *roots, population = roots
    assert population.id.startswith("TemplateInstances_v1{template='PouwH1Tile_v1{") and population.id.endswith(",N=2}")
    for root in roots:
        assert root.id.startswith("PouwH1Tile_v") and "{EXACT=GemmAccumulatorS8_v2{K=64," in root.id
        assert TG.definition(root.id) is root


def test_a_record_static_a_catalog_root_binds_parses_without_the_catalog(monkeypatch):
    """An id with a record static (``S={45bd63c5a436}``, a digest) does not look the record up among the suite's Definitions,
    which would build the whole catalog for one target.  Every catalog root whose id prints a record parses back to that root
    from the roots alone (`targets.binding_roots`); only a digest no root binds is looked up in the suite."""
    import re

    TG.load_registries()
    _no_catalog(monkeypatch)
    monkeypatch.setattr(TG, "_RECORDS", {})
    monkeypatch.setattr(TG, "_LOOKED", set())
    roots = {fn.id: fn for fn in TG.binding_roots() if re.search(r"=\{[0-9a-f]{12}\}", getattr(fn, "id", ""))}
    assert {"GfResiduals_v1", "Serve_v1", "StepBody_v1"} <= {fid.split("{")[0] for fid in roots}
    for fid, fn in roots.items():
        assert TG.definition(fid) is fn
    with pytest.raises(pytest.fail.Exception, match="the catalog was built"):
        TG.definition("GfResiduals_v1{S={000000000000}}")


def test_a_vocabulary_kind_that_does_not_bind_leaves_the_call_families_refused_by_name(monkeypatch):
    """A kind left out of `word_call_families` would make every check of its family judge the partition as no Call's, and
    `test_every_call_the_suite_checks_is_partitioned` pass without it."""
    monkeypatch.setattr(TG, "_SHORT", {})
    monkeypatch.setattr(TG, "_BOUND", {})
    monkeypatch.setattr(TG, "load_registries", lambda: None)
    monkeypatch.setattr(TG, "templates", lambda: [])
    monkeypatch.setattr(TG, "_vocabulary_bindings", lambda: [("b1-eager-v3/dsv4_indexer_compress", "KeyError: 'ROPE'")])
    with pytest.raises(RuntimeError, match=r"b1-eager-v3/dsv4_indexer_compress: KeyError: 'ROPE'"):
        TG.word_call_families()
    assert "words" not in TG._SHORT
