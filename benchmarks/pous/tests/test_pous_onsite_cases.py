"""`onsite_cases` (onsite_pieces.sh), the composite's step for onsite's own two cases, on a CPU: sourced and called as
`served_composite.sh` sources and calls it, over a serve dir's private directory as PoUS's piece and the wipe's leave
it. It names each piece's case beside its Lean's answer and makes neither: no card is asked for and no Lean is built
(`gpu-lease`, `nvidia-smi`, `lake` and served_composite.sh's `exe` are stand-ins that record a call), no encoder or
script is run (the encoder's build and the old case step are stand-ins too, and its Python runs only its own lines),
and nothing under the private directory changes but its own log. A piece that left no case or no answer is
``NOT_WRITTEN`` by its protocol and the composite goes on; a case served_composite.sh would compose that is not its
piece's with its Lean's answer stops the step before any product is composed."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
ONSITE_PIECES = ROOT / "benchmarks" / "pous" / "onsite_pieces.sh"
POUS, WIPE, BANDWIDTH = "verity/pous/p2/v4/run", "verity/wipe/v0", "verity/remote-bandwidth/p2/v0"
LEAN = {POUS: "pous-run-protocol", WIPE: "wipe-protocol"}
WHERE = {POUS: "pous/w3", WIPE: "wipe"}
SUMMARY = {POUS: "pous/pous.json", WIPE: "wipe/summary.json"}
NONE_MADE = "COMPOSITE CASES: PoUS's and the wipe's as their pieces wrote them, none made here"
#: served_composite.sh up to its composing: its settings, `say` and `exe`, this file sourced, the call, then the products
STEP = """set -uo pipefail
say() { echo "$*" | tee -a "$OUT/verdicts.txt"; }
exe() { echo "exe $*" >> "$CALLS"; return 1; }
. "$ONSITE_PIECES"
onsite_cases
echo composed
"""


def _exe(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(0o755)
    return path


def _write(path: Path, doc: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, sort_keys=True))


def _files(root: Path) -> dict[Path, bytes]:
    return {f: f.read_bytes() for f in sorted(root.rglob("*")) if f.is_file()}


@pytest.fixture
def served(tmp_path) -> tuple[Path, Path]:
    """A serve dir and a run's public pieces as window 3's pieces leave them: PoUS's case and the wipe's, each beside
    its Lean's answer, with each piece's summary naming that Lean; beside them the bandwidth case PoUS's piece writes
    and an earlier composite's own output, which served_composite.sh never composes."""
    d, out = (tmp_path / "serve").resolve(), (tmp_path / "out").resolve()
    private = d / "private"
    for protocol in (POUS, WIPE):
        _write(private / WHERE[protocol] / "case.json", {"commitment": "c0", "evidence": "beef", "params": {}})
        _write(private / WHERE[protocol] / "answer.json", {"commitment": "d1", "protocol": protocol, "verdict": "accept"})
        _write(out / "pieces" / SUMMARY[protocol], {"decided_by": {"host": "node", "program": LEAN[protocol]},
                                                     "verdict": "accept", "window": 3})
    _write(private / "bandwidth" / "case.json", {"commitment": "c0", "evidence": "beef", "params": {}})
    _write(private / "bandwidth" / "answer.json", {"commitment": "d1", "protocol": BANDWIDTH, "verdict": "accept"})
    _write(private / "composite" / "onsite" / "private" / "case.json", {"evidence": "beef"})
    _write(private / "composite" / "onsite" / "private" / "answer.json", {"protocol": POUS, "verdict": "reject"})
    return d, out


def _step(tmp_path: Path, d: Path, out: Path, python: str | None = None) -> tuple[subprocess.CompletedProcess, list[str],
                                                                                 list[str], list[tuple[str, dict]]]:
    """`STEP` on a node whose card, Lean builder and encoder are stand-ins: how it ended, what it called of them, the
    first argument of each Python it ran, and its private log's lines."""
    bin_, tree, calls, argv = tmp_path / "bin", tmp_path / "tree", tmp_path / "calls", tmp_path / "python.argv"
    for name in ("gpu-lease", "nvidia-smi", "lake"):
        _exe(bin_ / name, f'#!/bin/sh\necho "{name} $*" >> "$CALLS"\nexit 1\n')
    for script in ("benchmarks/pous/served_case.sh", "benchmarks/torch_gpu/p2_stage/build.sh"):
        _exe(tree / script, f'#!/bin/sh\necho "{script} $*" >> "$CALLS"\n')
    py = _exe(tmp_path / "python", python or f'#!/bin/sh\nprintf "%s\\n" "$1" >> "{argv}"\nexec {sys.executable} "$@"\n')
    env = {"PATH": f"{bin_}:{os.defpath}", "HOME": str(tmp_path / "home"), "D": str(d), "OUT": str(out),
           "P": str(d / "private" / "composite"), "PY": str(py), "ONSITE_PIECES": str(ONSITE_PIECES), "CALLS": str(calls)}
    r = subprocess.run(["bash", "-c", STEP], env=env, cwd=tree, check=False, capture_output=True, text=True)
    log = d / "private" / "composite" / "onsite_cases.log"
    lines = [(head, json.loads(body)) for head, _, body in (line.partition(" ") for line in log.read_text().splitlines())]
    return (r, calls.read_text().splitlines() if calls.exists() else [],
            argv.read_text().splitlines() if argv.exists() else [], lines)


def _said(out: Path) -> list[str]:
    return (out / "verdicts.txt").read_text().splitlines()


def _stopped(protocol: str) -> str:
    return (f"COMPOSITE: not run, the serve dir's private holds a case of {protocol} that is not its piece's with its "
            "Lean's answer, and every case there is composed (onsite_cases.log on the node)")


def test_each_pieces_case_is_taken_as_written_with_no_card_asked_for_and_no_encoder_run(served, tmp_path):
    """Given each piece's case and its Lean's answer, the step says each case's protocol and verdict and goes on to the
    products. It asks for no card, builds no Lean, runs no encoder and no script, and writes only its log, which names
    the one directory of each protocol that served_composite.sh then composes."""
    d, out = served
    before = _files(d / "private")
    r, calls, pythons, lines = _step(tmp_path, d, out)
    assert r.returncode == 0 and r.stdout.splitlines()[-1] == "composed", r.stdout + r.stderr
    assert _said(out) == [f"COMPOSITE CASE {POUS} accept", f"COMPOSITE CASE {WIPE} accept", NONE_MADE]
    assert calls == [] and pythons == ["-"]
    assert lines == [("CASE", {"piece": str(d / "private" / WHERE[p]), "protocol": p, "verdict": "accept"})
                     for p in (POUS, WIPE)]
    log = d / "private" / "composite" / "onsite_cases.log"
    assert _files(d / "private") == before | {log: log.read_bytes()}


@pytest.mark.parametrize("protocol", [POUS, WIPE])
@pytest.mark.parametrize("missing", [("answer.json",), ("case.json",), ("case.json", "answer.json")])
def test_a_piece_that_left_no_answer_or_no_case_is_not_written_and_none_is_made(served, tmp_path, protocol, missing):
    """A missing answer, a missing case, or neither there: ``NOT_WRITTEN`` by the protocol, the other piece's case as
    written, and the composite goes on without it. No card, no Lean and no encoder make the missing file, and why
    stays in the private log: the public lines hold no path and no reason."""
    d, out = served
    for name in missing:
        (d / "private" / WHERE[protocol] / name).unlink()
    before = _files(d / "private")
    r, calls, pythons, lines = _step(tmp_path, d, out)
    assert r.returncode == 0 and r.stdout.splitlines()[-1] == "composed", r.stdout + r.stderr
    want = {p: f"COMPOSITE NOT_WRITTEN {p}" if p == protocol else f"COMPOSITE CASE {p} accept" for p in (POUS, WIPE)}
    assert _said(out) == [want[POUS], want[WIPE], NONE_MADE]
    assert calls == [] and pythons == ["-"]
    log = d / "private" / "composite" / "onsite_cases.log"
    assert _files(d / "private") == before | {log: log.read_bytes()}
    (reason,) = [doc["reason"] for head, doc in lines if head == "NOT_WRITTEN" and doc["protocol"] == protocol]
    assert ("no case" if "case.json" in missing else "no answer") in reason
    assert str(d) not in (out / "verdicts.txt").read_text() and reason not in (out / "verdicts.txt").read_text()


def _python_decided(d: Path, out: Path) -> None:
    summary = out / "pieces" / SUMMARY[POUS]
    _write(summary, json.loads(summary.read_text()) | {"decided_by": {"host": "node", "program": "python"}})


def _another_verdict(d: Path, out: Path) -> None:
    _write(d / "private" / WHERE[POUS] / "answer.json", {"commitment": "d1", "protocol": POUS, "verdict": "reject"})


def _another_window(d: Path, out: Path) -> None:
    _write(d / "private" / "pous" / "w4" / "case.json", {"commitment": "c0", "evidence": "beef", "params": {}})
    _write(d / "private" / "pous" / "w4" / "answer.json", {"commitment": "d1", "protocol": POUS, "verdict": "accept"})


def _no_summary(d: Path, out: Path) -> None:
    (out / "pieces" / SUMMARY[POUS]).unlink()


@pytest.mark.parametrize("change, written, others", [
    (_python_decided, False, ["pous/w3"]),
    (_another_verdict, False, ["pous/w3"]),
    (_no_summary, False, ["pous/w3"]),
    (_another_window, True, ["pous/w4"]),
])
def test_a_case_that_is_not_its_pieces_with_its_leans_answer_stops_the_composite(served, tmp_path, change, written,
                                                                                 others):
    """served_composite.sh composes every case under the private directory, so a PoUS case there that the step can't
    name as its piece's with its Lean's answer (answered by the Python alone, of another verdict than the piece
    published, of a piece with no summary, or a second window's beside the piece's own) ends the step with exit 1
    before any product is composed. The public line names the protocol; the directories stay in the private log."""
    d, out = served
    change(d, out)
    before = _files(d / "private")
    r, calls, pythons, lines = _step(tmp_path, d, out)
    assert r.returncode == 1 and "composed" not in r.stdout.splitlines(), r.stdout + r.stderr
    assert _said(out) == [f"COMPOSITE CASE {POUS} accept" if written else f"COMPOSITE NOT_WRITTEN {POUS}",
                          f"COMPOSITE CASE {WIPE} accept", _stopped(POUS)]
    assert calls == [] and pythons == ["-"]
    assert lines[-1] == ("REFUSED", {"pieces": [str(d / "private" / o) for o in others], "protocol": POUS})
    log = d / "private" / "composite" / "onsite_cases.log"
    assert _files(d / "private") == before | {log: log.read_bytes()}
    assert str(d) not in (out / "verdicts.txt").read_text()


def test_cases_that_could_not_be_read_stop_the_composite(served, tmp_path):
    """A Python that fails leaves the two cases unnamed: the step says so and ends with exit 1, composing nothing."""
    d, out = served
    r, calls, _, lines = _step(tmp_path, d, out, python="#!/bin/sh\nexit 2\n")
    assert r.returncode == 1 and "composed" not in r.stdout.splitlines(), r.stdout + r.stderr
    assert _said(out) == ["COMPOSITE: not run, onsite's own cases weren't read (exit 2, onsite_cases.log on the node)"]
    assert calls == [] and lines == []
