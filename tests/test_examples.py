"""Every example states its rung on the ladder and carries the evidence for it (`examples/README.md`).

Each `examples/<name>/` has an `example.toml` that parses, with a known protocol, workload, placement and level, and the
evidence that level and every one below it need, in the form that can be checked offline: run-id format, a guarantee listed in
a `lean-audit.json`, lemma and proof names declared in the tracked `.lean` files (`verity_cli.examples`). The failure says
what to add. The negatives hold the checker to refusing each fault by what it names.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from verity_cli import examples as E

ROOT = Path(__file__).resolve().parents[1]
ALL = E.examples(ROOT)
LISTED = "NetTiming.Guarantees.SampledAuditAccepted"
PROOF = "NetTiming.SecurityProofs.SampledAuditAccepted"


def test_there_are_examples():
    assert ALL, "examples/ holds no example"


@pytest.mark.parametrize("ex", ALL, ids=lambda e: e.name)
def test_example_has_its_level(ex):
    assert not (p := E.problems(ex, ROOT)), f"examples/{ex.name}/example.toml:\n  " + "\n  ".join(p)


def _ex(level: int, **evidence) -> E.Example:
    placement = evidence.pop("placement", {"prover": "gpu-node", "verifier": "auditor-host"})
    return E.Example("probe", {"protocol": "pous", "workload": "none", "level": level, "placement": placement,
                               "evidence": evidence})


def _says(ex: E.Example, *words: str) -> None:
    p = "\n".join(E.problems(ex, ROOT))
    assert all(w in p for w in words), p


def test_a_document_fault_is_named():
    ex = E.Example("toy-onsite", {"protocol": ["pous"], "workload": "torch", "level": 6, "placement": {"warden": "laptop"},
                                  "evidence": {"run": []}, "extra": 1})
    _says(ex, "rename the directory", "unknown keys ['extra']", "a list of two or more", "workload 'torch'", "level 6",
          "'warden'", "warden = 'laptop'", "unknown evidence ['run']")
    _says(_ex(0, runs=["r2026-1"]), "'r2026-1' is neither a run id")


def test_each_level_needs_its_evidence():
    assert not E.problems(_ex(0), ROOT)
    _says(_ex(1), "level 1 (runs): add a run of record")
    _says(_ex(1, tests=["tests/test_examples.py::nope"]), "names no test")
    assert not E.problems(_ex(1, tests=["tests/test_examples.py::test_there_are_examples"]), ROOT)
    _says(_ex(2, runs=["r20261024-055630-5377"], placement={"prover": "cpu-node", "verifier": "cpu-node"}),
          "put verifier (now cpu-node) on auditor-host")
    _says(_ex(3, runs=["r20261024-055630-5377"], guarantee="Nci.Guarantees.Nope"), "listed in no lean-audit.json",
          "add [evidence] honest", "add [evidence] nontrivial")
    _says(_ex(3, runs=["r20261024-055630-5377"], guarantee=LISTED, honest="Nope.honestLemma", nontrivial=PROOF),
          "honest lemma 'Nope.honestLemma' is declared in no tracked .lean file")
    full = dict(runs=["r20261024-055630-5377"], guarantee=LISTED, honest=PROOF, nontrivial=PROOF)
    assert not E.problems(_ex(3, **full), ROOT)
    _says(_ex(4, **full, proof="NetTiming.Guarantees.SampledAuditAccepted"), "<Area>.SecurityProofs.G")
    assert not E.problems(_ex(4, **full, proof=PROOF), ROOT)
    _says(_ex(5, **full, proof=PROOF), "add [evidence.premises]")
    _says(_ex(5, **full, proof=PROOF, premises={"Nope.Premise": "soon"}), "'Nope.Premise' is an assumption no",
          "'soon' is neither")


def test_declared_reads_the_namespace():
    assert E.declared(ROOT, PROOF)
    assert not E.declared(ROOT, "Pous.SecurityProofs.SampledAuditAccepted")
