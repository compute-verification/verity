"""The sampling family's catalog split (`tools/migrate/split_sampling.py`): its Definitions are what they were, and its Definition
modules hold only Definitions.

* the Definition ids the family's modules define, and the canonical descriptor digest of each at every statics circuit-check's
  task list, its pins and the served bindings reach, equal the record taken before the split (`data/split_sampling_digests.json`);
* the family's Definition modules import only `verity`, the standard library, numpy and other Definition modules, except the
  kernel models `topp_split` still reads (`RESIDUE`, which only shrinks);
* the twin, the admission adapter and the vocabulary are no longer in the registry."""
from __future__ import annotations

import ast
import importlib
import json
import sys
from pathlib import Path

import pytest

from verity.ir.annotations import definition_digest
from verity.ir.codec import encode_program
from verity.ir.defs import CompositeDefinition, PrimitiveDefinition, bind
from verity.ir.program import Program

REG = "verity_vllm.program.registry"
FAMILY = ("sampling", "sampling_boolean", "gumbel_boolean", "topp_keep_builder", "topp_split", "topp_word_gates", "topp_words",
          "sampling_topp_difftest")
#: the other families' registry modules the family's Definitions call (Definition modules, or a Definition half still to split)
OTHER_DEFINITIONS = frozenset({"b1", "moe", "prims", "ref_prims"})
#: what a family Definition module imports that is not a Definition: `topp_split`'s exact-fp32 model of vLLM's split top-p (the
#: evaluator of `TopPMaskWordx{V}` and `SplitsForSMS`) reads the jitted MUFU.EX2 and div.full models and their tables from
#: `program.kernels`.  They leave when those models move to core or the catalog's device models; this set only shrinks.
RESIDUE = frozenset({
    ("topp_split", "verity_vllm.program.kernels.fa2_model"),
    ("topp_split", "verity_vllm.program.kernels.fa2_relation"),
    ("topp_split", "verity_vllm.program.kernels.rms_relation"),
    ("topp_split", "verity_vllm.program.kernels.rms_triton_model"),
})
#: what a Definition module may never import, even lazily
FORBIDDEN = ("torch", "vllm", "triton")
DATA = json.loads((Path(__file__).parent / "data" / "split_sampling_digests.json").read_text())
#: the lazy families `registry/__init__.py` makes on demand, by id prefix -> their maker in `registry.sampling`
LAZY = {"TopPMaskWordx": "topp_mask_word", "BitAtx": "bit_at"}
DEFS = (PrimitiveDefinition, CompositeDefinition)


def _module(m: str):
    return importlib.import_module(f"{REG}.{m}")


def _imports(path: Path) -> set[str]:
    """Every module a file imports, at module level or in a function; `from a import b` counts as `a.b`."""
    out = set()
    for n in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(n, ast.ImportFrom) and n.level == 0 and n.module and n.module != "__future__":
            out.update(f"{n.module}.{a.name}" for a in n.names)
        elif isinstance(n, ast.Import):
            out.update(a.name for a in n.names)
    return out


def _definition(i: str):
    for prefix, maker in LAZY.items():
        n = i[len(prefix):-len("_v1")]
        if i.startswith(prefix) and i.endswith("_v1") and n.isdigit():
            return getattr(_module("sampling"), maker)(int(n))
    for m in FAMILY:
        for v in vars(_module(m)).values():
            if isinstance(v, DEFS) and v.id == i:
                return v
    raise KeyError(i)


def _wrap(fn) -> CompositeDefinition:
    """A static-free composite calling `fn`: a Program's root is inlined into its own entry, so `fn`'s entry is a callee's."""
    return CompositeDefinition(f"SplitSamplingWrap[{fn.id}]", 1, (), lambda S: fn.signature(),
                               lambda B, S, *a: B.call(fn, *a), conformance="derived: split-sampling probe", register=False)


def test_the_family_defines_the_same_definitions():
    """A Definition is the family's when a family module binds it and no module outside the family it imports from does."""
    family = {f"{REG}.{m}" for m in FAMILY}
    owned = set()
    for name in sorted(family):
        mod = importlib.import_module(name)
        imports = _imports(Path(mod.__file__))
        direct = imports | {d.rsplit(".", 1)[0] for d in imports}
        foreign = [sys.modules[d] for d in direct if d in sys.modules and d not in family]
        owned |= {v.id for v in vars(mod).values()
                  if isinstance(v, DEFS) and not any(any(x is v for x in vars(f).values()) for f in foreign)}
    assert sorted(owned) == DATA["owned"]


_BY_ID: dict[str, list[str]] = {}
for _fid, _e in DATA["entries"].items():
    _BY_ID.setdefault(_e["id"], []).append(_fid)


@pytest.mark.slow
@pytest.mark.parametrize("did", sorted(_BY_ID))
def test_descriptor_digests_unchanged(did: str):
    """About 100 s of CPU over the 67 ids (the largest V-wide bodies dominate).  A digest covers a Definition's whole entry at
    its statics, its body, signature and every callee's entry, so it catches a move that changed what a Definition computes
    or calls, which neither the ids nor circuit-check's task list can show."""
    d = _definition(did)
    for fid in _BY_ID[did]:
        e = DATA["entries"][fid]
        desc = encode_program(Program(_wrap(d if isinstance(d, PrimitiveDefinition) else bind(d, **e["bindings"]))))
        assert definition_digest(desc, fid) == e["digest"], fid


def test_definition_modules_import_only_definitions():
    found, forbidden = set(), []
    for m in FAMILY:
        for imp in _imports(Path(_module(m).__file__)):
            top = imp.split(".")[0]
            if top in FORBIDDEN:
                forbidden.append((m, imp))
            elif top in sys.stdlib_module_names or top in ("numpy", "verity"):
                continue
            elif imp.startswith(f"{REG}.") and imp.split(".")[3] in (*FAMILY, *OTHER_DEFINITIONS):
                continue
            else:
                found.add((m, imp))
    assert not forbidden, forbidden
    assert not found - RESIDUE, f"a Definition module imports what is not a Definition: {sorted(found - RESIDUE)}"
    assert not RESIDUE - found, f"RESIDUE only shrinks: drop {sorted(RESIDUE - found)}"


def test_the_twin_harness_and_vocabulary_left_the_registry():
    reg = Path(importlib.import_module(REG).__file__).parent
    assert not (reg / "sampling_rows.py").exists()
    rows = importlib.import_module("verity_vllm.program.kernels.sampling_rows")
    assert callable(rows.gumbel_topp_token_select_row)

    S = _module("sampling")
    assert not [n for n in ("SHARED_GREEDY", "select_for") if hasattr(S, n)]
    from verity_vllm.program import sampling_vocab
    assert sampling_vocab.select_for(S.GumbelTopPTokenSelect, "shared-greedy") is S.GumbelTopPTokenSelectSharedGreedy
    assert sampling_vocab.select_for(S.GumbelTopPTokenSelect, "per-row") is S.GumbelTopPTokenSelect

    D = _module("sampling_topp_difftest")
    assert not [n for n in ("SEMANTIC_ID", "cases", "production") if hasattr(D, n)]
    A = importlib.import_module("verity_vllm.properties.sampling_topp_admission")
    assert A.spec is D.spec and callable(A.cases) and callable(A.production) and A.SEMANTIC_ID
