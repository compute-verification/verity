"""The PoUW window's audit (`verity_torch_loop.pouw_audit`) proves in scratch: under ``--out`` it leaves the audit's
records, and no member's circuit, session archive or prover's file, and it answers from what it kept."""
from __future__ import annotations

import json
from pathlib import Path

from verity_torch_loop import pouw_audit as PA

#: an audit's tree as `records_audit.py` leaves it (r20261026-004315-feca's), each file with whether `pouw_audit` keeps it
TREE = {"summary.json": True, "window/audit.json": True, "window/registration.json": True, "window/receipt.json": True,
        "window/program.json": True, "window/case.json": True, "window/m3/pub-8.bin": True,
        "window/m3/circuit.txt": False, "window/m3/inst-8.bin": False, "window/m3/prover/m3-pub-8.bin": False,
        "window/m3/prover/m3-inst-8.bin": False, "window/m3/rows-private.bin": False,
        "window/session-m3/archive/sha512/ab12": False, "window/session-m3/prove.log": True,
        "window/session-m3/sessions/l0000-1/session.json": True, "window/session-m3/sessions/l0000-1/circuit.rep0.bin": True,
        "window/session-m3/sessions/l0000-1/verifier.json": True}
KEPT = sorted(r for r, kept in TREE.items() if kept)


def tree(root: Path) -> None:
    for rel in TREE:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text("{}" if rel.endswith(".json") else rel)


def files(root: Path) -> list[str]:
    return sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file())


def test_the_audit_keeps_its_records_and_no_circuit_archive_or_prover_file(tmp_path):
    tree(tmp_path / "work")
    PA.keep(tmp_path / "work", tmp_path / "kept")
    assert files(tmp_path / "kept") == KEPT


def test_the_audit_proves_in_scratch_and_answers_from_what_it_kept(tmp_path, monkeypatch):
    rec = {"law": {"law": "subset", "k": 1}, "window": {"context": "ab", "kind": "pouw", "public": {
        "length": {"max_tokens": [2]}, "sampling": {"eps": "1/2", "delta": "1/1000000", "law": {"k": 1}},
        "work": {"credited": 0, "credited_tiles": 0, "n": 1}}}}
    win = tmp_path / "win"
    win.mkdir()
    (win / "registration.json").write_text(json.dumps(rec))
    (win / "provenance.json").write_text(json.dumps({"run": "ab"}))
    seen = []

    def records_audit(argv):
        work = Path(argv[argv.index("--out") + 1])
        seen.append(work)
        tree(work)
        (work / "summary.json").write_text(json.dumps({"refused": {"records": "a test"}}))
        return 1

    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setenv("TMPDIR", str(scratch))
    monkeypatch.setattr(PA.tempfile, "tempdir", None)
    monkeypatch.setattr(PA, "_records_audit", records_audit)
    monkeypatch.setattr(PA.RA, "main", PA.RA.main)
    out = tmp_path / "out"
    PA.main(["--window", str(win), "--circuit", "00", "--run", "ab", "--m0-pythonpath", "", "--bin", "b", "--lean", "l",
             "--firewall", "f", "--out", str(out)])
    assert seen[0].parent == scratch and not seen[0].exists()
    assert files(out / "audit") == KEPT
    got = json.loads((out / "pouw.json").read_text())
    assert got["verdict"] == "reject" and got["audit_rc"] == 1
    assert json.loads((win / "answer.json").read_text())["reason"].startswith("refused:")
