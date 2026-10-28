"""Every example states its doneness on both axes and carries the evidence for each (`examples/README.md`).

Each `examples/<name>/` has an `example.toml` that parses, with a known protocol, workload and placement, a `run` (moved, runs,
placed) and a `proof` (none, stated, proved, measured), and the evidence every step up to each needs on its own axis, in the
form that can be checked offline: run-id format, a guarantee listed in a `lean-audit.json`, lemma and proof names declared in
the tracked `.lean` files (`verity_cli.examples`). The failure says what to add. The negatives hold the checker to refusing
each fault by what it names, and to reading neither axis's evidence for the other.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from verity_cli import examples as E

ROOT = Path(__file__).resolve().parents[1]
ALL = E.examples(ROOT)
RUN = "r20261024-055630-5377"
LISTED = "NetTiming.Guarantees.SampledAuditAccepted"
PROOF = "NetTiming.SecurityProofs.SampledAuditAccepted"
STATED = dict(guarantee=LISTED, honest=PROOF, nontrivial=PROOF)
LOCAL = {"prover": "local", "verifier": "local"}


def test_there_are_examples():
    assert ALL, "examples/ holds no example"


@pytest.mark.parametrize("ex", ALL, ids=lambda e: e.name)
def test_example_has_what_it_claims(ex):
    assert not (p := E.problems(ex, ROOT)), f"examples/{ex.name}/example.toml:\n  " + "\n  ".join(p)


def _ex(run: str = "moved", proof: str = "none", /, **evidence) -> E.Example:
    placement = evidence.pop("placement", {"prover": "gpu-node", "verifier": "auditor-host"})
    return E.Example("probe", {"protocol": "pous", "workload": "none", "run": run, "proof": proof, "placement": placement,
                               "evidence": evidence})


def _says(ex: E.Example, *words: str) -> None:
    p = "\n".join(E.problems(ex, ROOT))
    assert p and all(w in p for w in words), p


def test_a_document_fault_is_named():
    ex = E.Example("toy-onsite", {"protocol": ["pous"], "workload": "torch", "run": "served", "proof": 4,
                                  "placement": {"relay": "laptop"}, "evidence": {"run": []}, "extra": 1})
    _says(ex, "rename the directory", "unknown keys ['extra']", "a list of two or more", "workload 'torch'", "run 'served'",
          "proof 4", "'relay'", "relay = 'laptop'", "unknown evidence ['run']")
    _says(_ex(runs=["r2026-1"]), "'r2026-1' is neither a run id")
    old = E.Example("probe", {"protocol": "pous", "workload": "none", "level": 1, "placement": LOCAL})
    _says(old, "level is retired", "add run, one of moved", "add proof, one of none")


def test_a_file_with_the_retired_level_reads_as_its_two_axes(tmp_path):
    d = tmp_path / "examples" / "toy"
    d.mkdir(parents=True)
    (d / "example.toml").write_text('protocol = "pous"\nworkload = "none"\nlevel = 1\n')
    (ex,) = E.examples(tmp_path)
    assert (ex.data["run"], ex.data["proof"], "level" in ex.data) == ("runs", "none", False)


def test_each_run_step_needs_its_evidence():
    assert not E.problems(_ex(), ROOT)
    _says(_ex("runs"), "run runs: add a run of record")
    _says(_ex("runs", tests=["tests/test_examples.py::nope"]), "names no test")
    assert not E.problems(_ex("runs", tests=["tests/test_examples.py::test_there_are_examples"]), ROOT)
    _says(_ex("placed", runs=[RUN], placement={"prover": "cpu-node", "verifier": "cpu-node"}),
          "put verifier (now cpu-node) on auditor-host")
    _says(_ex("placed", runs=[RUN], placement={"prover": "auditor-host", "verifier": "auditor-host"}),
          "the prover's roles ['prover'] are on auditor-host")
    _says(_ex("placed", placement={"prover": "gpu-node", "verifier": "auditor-host"}), "run placed: add a run of record")
    assert not E.problems(_ex("placed", runs=[RUN]), ROOT)


def test_each_proof_step_needs_its_evidence():
    _says(_ex("moved", "stated", guarantee="Nci.Guarantees.Nope"), "proof stated: guarantee 'Nci.Guarantees.Nope' is listed in no",
          "add [evidence] honest", "add [evidence] nontrivial")
    _says(_ex("moved", "stated", guarantee=LISTED, honest="Nope.honestLemma", nontrivial=PROOF),
          "honest lemma 'Nope.honestLemma' is declared in no tracked .lean file")
    assert not E.problems(_ex("moved", "stated", **STATED), ROOT)
    _says(_ex("moved", "proved", **STATED, proof=LISTED), "<Area>.SecurityProofs.G")
    _says(_ex("moved", "proved", proof=PROOF), "proof stated: add [evidence] guarantee")
    assert not E.problems(_ex("moved", "proved", **STATED, proof=PROOF), ROOT)
    _says(_ex("moved", "measured", **STATED, proof=PROOF), "add [evidence.premises]")
    _says(_ex("moved", "measured", **STATED, proof=PROOF, premises={"Nope.Premise": "soon"}), "'Nope.Premise' is an assumption no",
          "'soon' is neither")


def test_the_axes_are_independent():
    assert not E.problems(_ex("moved", "proved", **STATED, proof=PROOF, placement=LOCAL), ROOT), "a local toy, fully proved"
    assert not E.problems(_ex("placed", "none", runs=[RUN]), ROOT), "a placed demo, its guarantee unstated"
    ex = _ex("runs", "none", runs=[RUN], placement=LOCAL)
    assert E.next_missing(ex, "run", ROOT).startswith("run placed: the auditor's roles run off the prover's node")
    assert E.next_missing(ex, "proof", ROOT).startswith("proof stated: add [evidence] guarantee")


def test_declared_reads_the_namespace():
    assert E.declared(ROOT, PROOF)
    assert not E.declared(ROOT, "Pous.SecurityProofs.SampledAuditAccepted")
