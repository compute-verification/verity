"""The status export: each lock guarantee proved, resting on named assumptions, or unknown, and the results of record."""
from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

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


def test_a_guarantee_is_proved_only_when_its_premises_were_read_and_name_no_assumption(tmp_path):
    root = _repo(tmp_path / "repo", {"Sec/lean-audit.json": LOCK, "x/tests/lean-audit.json": LOCK})
    facts = tmp_path / "facts"
    (facts / "audit-Sec" / "proved-in-Proofs").mkdir(parents=True)
    (facts / "audit-Sec" / "proved-in-Proofs" / "facts.json").write_text(json.dumps(FACTS))
    doc, _ = se.export(root, facts=facts)
    assert _status(doc) == {
        "A.SecurityProofs.Hyp": ("assumes", ["A.Assumptions.One"], False),
        "A.SecurityProofs.Premise": ("assumes", ["A.Assumptions.Two"], True),
        "A.SecurityProofs.Alias": ("assumes", ["A.Assumptions.One", "A.Assumptions.Two"], True),
        "A.SecurityProofs.Clean": ("proved", [], True),
        "A.SecurityProofs.Unread": ("unknown", [], False),
        "A.SecurityProofs.Moved": ("unknown", [], False),
    }
    assert doc["packages"] == [{"path": "Sec", "guarantees": 6, "assumptions": 2}]
    assert [g["name"] for g in doc["guarantees"]] == sorted(LOCK["guarantees"])
    assert [(a["name"], a["owner"]) for a in doc["assumptions"]] == [("A.Assumptions.One", "@lean"), ("A.Assumptions.Two", "@proofs")]


def test_without_facts_no_guarantee_is_proved(tmp_path):
    doc, _ = se.export(_repo(tmp_path, {"Sec/lean-audit.json": LOCK}))
    assert {s for s, _, read in _status(doc).values()} == {"assumes", "unknown"}
    assert not any(read for _, _, read in _status(doc).values())


def test_results_of_record_come_from_an_examples_evidence_and_run_ids_are_only_counted(tmp_path):
    example = f'[evidence]\nruns = ["{ART}", "r20260101-000000-abcd"]\n[evidence.premises]\nbandwidth = "art:{"b" * 64}"\n'
    doc, runs = se.export(_repo(tmp_path, {"Sec/lean-audit.json": LOCK, "demos/demo/example.toml": example}))
    assert doc["results"] == [{"art": ART, "kind": None, "what": "the demo example's run of record", "table": None},
                              {"art": "art:" + "b" * 64, "kind": None, "what": "bandwidth, measured for the demo example", "table": None}]
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
