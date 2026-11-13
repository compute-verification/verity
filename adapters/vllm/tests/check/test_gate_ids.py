"""The gates' ids: `verdict_names` states them and imports nothing, and `gates` hands the same objects on."""
from __future__ import annotations

import ast
from pathlib import Path

from verity_vllm.check import gates as G
from verity_vllm.check import verdict_names as N

IDS = ("GATES", "GATE_NAMES", "HOLDOUT_GATES", "STRUCTURAL_GATES", "INTEGRITY_GATES",
       "OBSERVATION", "SEMANTIC_CLOSURE", "ROOT_CLOSURE", "PRIMITIVE", "COMPOSITION", "CANONICALITY", "HOLDOUT", "VU_PARTITION")


def test_the_gates_hand_on_the_ids_that_verdict_names_states():
    assert tuple(N.__all__) == IDS and set(IDS) <= set(G.__all__)
    assert [name for name in IDS if getattr(G, name) is not getattr(N, name)] == []


def test_each_gate_has_a_name_by_what_it_checks():
    by_what = (N.OBSERVATION, N.SEMANTIC_CLOSURE, N.ROOT_CLOSURE, N.PRIMITIVE, N.COMPOSITION, N.CANONICALITY, N.HOLDOUT, N.VU_PARTITION)
    assert by_what == ("G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8") == N.GATES + N.STRUCTURAL_GATES
    holdout, integrity, every = N.HOLDOUT_GATES, N.INTEGRITY_GATES, G.ALL_GATES
    assert holdout == by_what[:6] and integrity == ("R1",) and every == (*by_what, "R1")
    assert list(N.GATE_NAMES) == list(every)
    assert N.GATE_NAMES == {
        "G1": "observation fidelity", "G2": "semantic closure", "G3": "root closure", "G4": "primitive exactness",
        "G5": "composition exactness", "G6": "canonicality + regression", "G7": "holdout generalization", "G8": "VU partition",
        "R1": "reference integrity (fail-closed)"}


def test_the_module_that_states_the_ids_imports_only_typing():
    tree = ast.parse(Path(N.__file__).read_text())
    imported = [node.module if isinstance(node, ast.ImportFrom) else alias.name
                for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))
                for alias in node.names]
    assert imported == ["__future__", "typing"]
