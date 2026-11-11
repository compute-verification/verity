"""The primitives, composites and serving family's split into Definitions, twins and vocabulary moved no Definition: at every
statics circuit-check binds, the family's targets are the same tasks in the same order, with the same descriptor digests, the
same registered ids and the same pins as on the main it was cut from (`fixtures/split/prims_serve.json`).  A digest pins the
whole descriptor, so a Definition whose body, statics or callees changed in the move fails here; the import-boundary test
(adapters/vllm `tests/program/test_prims_serve_boundaries.py`) cannot see that.  A change that means to add or move one
rewrites the fixture from the tree with this module's ``--write`` (`fields`, which the tests compare, over the catalog
conftest.py builds), which `REWRITE`'s command runs."""
from __future__ import annotations

import ast
import importlib
import json
import sys
from pathlib import Path

import pytest

from circuit_check import checks as CK
from circuit_check import cli
from circuit_check import targets as TG

FIXTURE = Path(__file__).parent / "fixtures" / "split" / "prims_serve.json"
REWRITE = ("a change meant to add or move a family Definition rewrites the fixture with "
           "`uv run python tools/verity/check/pins.py --write tools/verity/circuit_check/tests/fixtures/split/prims_serve.json`")
#: the fixture's ``family``, which no test reads
FAMILY = "primitives, composites and serving"
REG = "verity.definitions.registry"
#: the family's Definition modules (`FAMILY` of adapters/vllm `tests/program/test_prims_serve_boundaries.py`)
DEFINITION_MODULES = ("__init__", "catalog", "prims", "ref_prims", "pad_prims", "lifted", "tables", "value_inputs", "operand_widths",
                      "b1", "b1_tp2", "serve3", "spec", "pouw_rows")


def _digest(fn) -> str | None:
    """The descriptor digest of a circuit-check target: a specialization's Program digest; a primitive's descriptor entry inside
    a one-Call wrapper (which is not registered)."""
    from verity.circuits import codec
    from verity.circuits.defs import CompositeDefinition, PrimitiveDefinition, SpecializedDefinition, bind
    from verity.circuits.program import Program

    if isinstance(fn, SpecializedDefinition):
        return codec.program_digest(codec.encode_program(Program(fn)))
    if isinstance(fn, PrimitiveDefinition):
        params, ret = fn.signature()
        wrap = CompositeDefinition("SplitDigestWrap", 1, (), lambda S: (params, ret), lambda B, S, *a: B.call(fn, *a), register=False)
        desc = codec.encode_program(Program(bind(wrap)))
        return codec.definition_digest(desc["definitions"][fn.id], desc)
    return None


def _family_ids() -> dict[str, str]:
    """Each Definition family (`Name_vN`) a family Definition module defines (not one it imports), by that module."""
    from verity.circuits.defs import CompositeDefinition, PrimitiveDefinition

    out: dict[str, str] = {}
    for name in DEFINITION_MODULES:
        mod = importlib.import_module(REG if name == "__init__" else f"{REG}.{name}")
        tree = ast.parse(Path(mod.__file__).read_text(encoding="utf-8"))
        imported = {a.asname or a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom)) for a in n.names}
        for k, v in vars(mod).items():
            if k not in imported and isinstance(v, (CompositeDefinition, PrimitiveDefinition)):
                out.setdefault(v.id, name)
    return dict(sorted(out.items()))


def fields(suite: dict) -> dict:
    """The fixture's fields on this tree, from circuit-check's catalog ``suite`` (which `targets.suite` returns): every
    circuit-check task whose target is of a family Definition, in task order, with its descriptor digest; the family's
    registered ids; and the pins of those targets."""
    TG.load_registries()
    tasks = cli._tasks([], CK.Options(), True, None)
    family_ids = _family_ids()
    fam = set(family_ids)
    families = {fid: TG.family(fn) for fid, fn in suite.items()}
    tasks = [[t[0], t[1]] for t in tasks if t[0] == "definition" and families.get(t[1]) in fam]
    labels = {label for _, label in tasks}
    pins = CK.load_pins()
    return {"family_ids": family_ids,
            "tasks": tasks,
            "digests": {label: _digest(suite[label]) for _, label in tasks},
            "registered": sorted(i for i in TG.REGISTERED if i in fam),
            "pins": {sec: {k: v for k, v in (pins.get(sec) or {}).items() if k.split(" @ ")[0] in labels or k in labels}
                     for sec in sorted(pins)}}


@pytest.fixture(scope="module")
def now(suite) -> dict:
    return fields(suite)


@pytest.fixture(scope="module")
def pinned() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


@pytest.mark.slow
def test_family_definitions_are_the_same(now, pinned):
    assert now["family_ids"] == pinned["family_ids"], REWRITE
    assert now["registered"] == pinned["registered"], REWRITE


@pytest.mark.slow
def test_task_list_is_the_same(now, pinned):
    assert now["tasks"] == pinned["tasks"], REWRITE
    assert now["pins"] == pinned["pins"], REWRITE


@pytest.mark.slow
def test_descriptor_digests_are_the_same(now, pinned):
    changed = sorted(label for label in pinned["digests"] if now["digests"].get(label) != pinned["digests"][label])
    assert not changed, f"descriptor digests that changed: {changed[:20]}; {REWRITE}"


def write() -> None:
    """``--write``: the fixture from `fields` over the catalog conftest.py builds for the suite (from `check`'s circuit-check
    records where they count), with `targets.suite` returning it, as the suite has them."""
    import conftest

    built = conftest.build()
    TG.call_families(built.suite)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(TG, "suite", lambda: built.suite)
        doc = {**fields(built.suite), "family": FAMILY}
    FIXTURE.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"{FIXTURE}: {len(doc['family_ids'])} family Definitions, {len(doc['tasks'])} tasks")


if __name__ == "__main__":
    if sys.argv[1:] != ["--write"]:
        sys.exit("usage: test_split_prims_serve.py --write")
    write()
