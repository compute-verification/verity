"""The status export: each lock guarantee proved, resting on named assumptions, or unknown, and the results of record."""
from __future__ import annotations

import gzip
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location("status_export", Path(__file__).parents[1] / "console" / "status_export.py")
se = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(se)

ART = "art:" + "a" * 64


def _repo(root: Path, files: dict[str, object]) -> Path:
    for rel, body in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body if isinstance(body, str) else json.dumps(body))
    git = ["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false"]
    subprocess.run([*git, "init", "-q"], check=True)
    subprocess.run([*git, "add", "-A"], check=True)
    subprocess.run([*git, "commit", "-q", "-m", "t"], check=True)
    return root


def _g(sig: str, owner: str = "@lean", assumptions: list[str] | None = None) -> dict:
    return {"owner": owner, "signature": [sig], "assumptions": assumptions or [], "type_hash": "0" * 64}


LOCK = {
    "proved_in": ["Proofs"],
    "assumptions": {"A.Assumptions.One": {"owner": "@lean", "digest": "1" * 64},
                    "A.Assumptions.Two": {"owner": "@proofs", "digest": "2" * 64}},
    "guarantees": {
        "A.SecurityProofs.Hyp": _g("A.SecurityProofs.Hyp (h : A.Assumptions.One) : A.G", assumptions=["A.Assumptions.One"]),
        "A.SecurityProofs.Premise": _g("A.SecurityProofs.Premise : A.Guarantees.P"),
        "A.SecurityProofs.Alias": _g("A.SecurityProofs.Alias : A.Guarantees.Q"),
        "A.SecurityProofs.Clean": _g("A.SecurityProofs.Clean : A.Guarantees.C"),
        "A.SecurityProofs.Unread": _g("A.SecurityProofs.Unread : A.Guarantees.U"),
        "A.SecurityProofs.Moved": _g("A.SecurityProofs.Moved : A.Guarantees.M"),
    },
}

FACTS = {
    "own_modules": ["A.Definitions"],
    "pins": {
        "A.SecurityProofs.Premise": {"kind": "theorem", "signature": "A.SecurityProofs.Premise : A.Guarantees.P",
                                     "premises": [{"head": "A.Assumptions.Two", "module": "A.Assumptions"}]},
        "A.SecurityProofs.Alias": {"kind": "theorem", "signature": "A.SecurityProofs.Alias : A.Guarantees.Q",
                                   "premises": [{"head": "A.Definitions.Both", "module": "A.Definitions",
                                                 "unfolds": {"A.Definitions.Both": [{"head": "A.Assumptions.One"},
                                                                                    {"head": "A.Assumptions.Two"}]}}]},
        "A.SecurityProofs.Clean": {"kind": "theorem", "signature": "A.SecurityProofs.Clean : A.Guarantees.C", "premises": []},
        "A.SecurityProofs.Moved": {"kind": "theorem", "signature": "A.SecurityProofs.Moved : A.Guarantees.Other", "premises": []},
    },
}


def _status(doc: dict) -> dict[str, tuple[str, list[str], bool]]:
    return {g["name"]: (g["status"], g["assumptions"], g["premises_read"]) for g in doc["guarantees"]}


@pytest.mark.parametrize("name", ["facts.json.gz", "facts.json"])
def test_a_guarantee_is_proved_only_when_its_premises_were_read_and_name_no_assumption(tmp_path, name):
    """The audit keeps its facts gzipped (`facts.json.gz`); an older run's plain `facts.json` reads the same."""
    root = _repo(tmp_path / "repo", {"Sec/lean-audit.json": LOCK, "x/tests/lean-audit.json": LOCK})
    facts = tmp_path / "facts"
    (facts / "audit-Sec" / "proved-in-Proofs").mkdir(parents=True)
    data = json.dumps(FACTS).encode()
    (facts / "audit-Sec" / "proved-in-Proofs" / name).write_bytes(gzip.compress(data) if name.endswith(".gz") else data)
    doc, _ = se.export(root, facts=facts)
    assert _status(doc) == {
        "A.SecurityProofs.Hyp": ("assumes", ["A.Assumptions.One"], False),
        "A.SecurityProofs.Premise": ("assumes", ["A.Assumptions.Two"], True),
        "A.SecurityProofs.Alias": ("assumes", ["A.Assumptions.One", "A.Assumptions.Two"], True),
        "A.SecurityProofs.Clean": ("proved", [], True),
        "A.SecurityProofs.Unread": ("unknown", [], False),
        "A.SecurityProofs.Moved": ("unknown", [], False),
    }
    assert doc["packages"] == [{"path": "Sec", "guarantees": 6, "assumptions": 2, "open": 0, "domain": 0}]
    assert [g["name"] for g in doc["guarantees"]] == sorted(LOCK["guarantees"])
    assert [(a["name"], a["owner"]) for a in doc["assumptions"]] == [("A.Assumptions.One", "@lean"), ("A.Assumptions.Two", "@proofs")]


def test_a_domain_hypothesis_is_part_of_what_a_guarantee_says_and_not_an_assumption_it_rests_on(tmp_path):
    scope = "A.Definitions.Scope"
    lock = LOCK | {"domain": {scope: {"owner": "@lean", "digest": "3" * 64, "why": "the inputs it covers"}},
                   "guarantees": LOCK["guarantees"] | {
                       "A.SecurityProofs.Scoped": _g(f"A.SecurityProofs.Scoped (h : {scope}) : A.G", assumptions=[scope])}}
    pin = {"kind": "theorem", "signature": f"A.SecurityProofs.Scoped (h : {scope}) : A.G", "premises": []}
    facts = tmp_path / "facts"
    (facts / "audit-Sec" / "proved-in-Proofs").mkdir(parents=True)
    (facts / "audit-Sec" / "proved-in-Proofs" / "facts.json").write_text(json.dumps(
        FACTS | {"pins": FACTS["pins"] | {"A.SecurityProofs.Scoped": pin}}))
    doc, _ = se.export(_repo(tmp_path / "repo", {"Sec/lean-audit.json": lock}), facts=facts)
    scoped = next(g for g in doc["guarantees"] if g["name"] == "A.SecurityProofs.Scoped")
    assert (scoped["status"], scoped["assumptions"], scoped["domain"]) == ("proved", [], [scope])
    assert doc["packages"][0]["domain"] == 1
    assert [(a["kind"], a["why"]) for a in doc["assumptions"] if a["name"] == scope] == [("domain", "the inputs it covers")]


def test_without_facts_no_guarantee_is_proved(tmp_path):
    doc, _ = se.export(_repo(tmp_path, {"Sec/lean-audit.json": LOCK}))
    assert {s for s, _, read in _status(doc).values()} == {"assumes", "unknown"}
    assert not any(read for _, _, read in _status(doc).values())


def test_results_of_record_come_from_a_demos_evidence_and_run_ids_are_only_counted(tmp_path):
    demo = f'[evidence]\nruns = ["{ART}", "r20260101-000000-abcd"]\n[evidence.premises]\nbandwidth = "art:{"b" * 64}"\n'
    doc, runs = se.export(_repo(tmp_path, {"Sec/lean-audit.json": LOCK, "demos/toy/demo.toml": demo}))
    assert doc["results"] == [{"art": ART, "kind": None, "what": "the toy demo's run of record", "table": None},
                              {"art": "art:" + "b" * 64, "kind": None, "what": "bandwidth, measured for the toy demo", "table": None}]
    assert runs == 1


def test_the_document_is_written_as_canonical_json_of_the_commit_it_read(tmp_path):
    root = _repo(tmp_path / "repo", {"Sec/lean-audit.json": LOCK})
    out = tmp_path / "status.json"
    assert se.main(["--root", str(root), "--out", str(out)]) == 0
    data = out.read_bytes()
    doc = json.loads(data)
    assert data == se.canonical(doc) and not data.endswith(b"\n")
    assert doc["schema"] == "verity/status/v0"
    assert doc["commit"] == subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
