"""Stand-ins for the Lean answers the memory verdicts don't have yet (`coins.UNWRITTEN`): the auditor line's ledger
read, *complete*, commit on main and served session read, and the served session's LMS verify.
Each agrees with the Python it is held to, so a test of what follows an accept runs; `unwritten` takes the real
interfaces back, under which each of those is undecided."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str((Path(__file__).resolve().parents[3] / "benchmarks" / "poub_session")))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import auditor_line as al
from verity.protocols.resource_accounting.memory.poub.verifier.poub_session import verifier_process as VP

REAL = {(m, name): getattr(m, name) for m, name in (
    (al, "periods_answer"), (al, "complete_answer"), (al, "on_main_answer"), (al, "served_session_answer"),
    (VP, "lms_answer"))}


def agree(monkeypatch) -> None:
    """Each interface stood in through ``monkeypatch`` by an answer equal to the Python's."""
    monkeypatch.setattr(al, "periods_answer", lambda lines, key, python: python)
    monkeypatch.setattr(al, "complete_answer", lambda lines, R, period, session, key, python: python)
    monkeypatch.setattr(al, "on_main_answer", lambda commit, main, python: python)
    monkeypatch.setattr(al, "served_session_answer", lambda key, files, python: python)
    monkeypatch.setattr(VP, "lms_answer", lambda key, message, signature: True)


def unwritten(monkeypatch) -> None:
    """The real interfaces back, through ``monkeypatch``: each raises `NotImplementedError` with the goal's id."""
    for (m, name), f in REAL.items():
        monkeypatch.setattr(m, name, f)
