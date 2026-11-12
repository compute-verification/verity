"""`onsite_cases` (onsite_pieces.sh), the composite's step for onsite's own two cases, on a CPU: sourced and called as
`served_composite.sh` sources and calls it, over a serve dir's private directory as PoUS's piece and the wipe's leave
it. It names each piece's case beside its Lean's answer and makes neither: no card is asked for and no Lean is built
(`gpu-lease`, `nvidia-smi`, `lake` and served_composite.sh's `exe` are stand-ins that record a call), no encoder or
script is run (the encoder's build and the old case step are stand-ins too, and its Python runs only its own lines),
and nothing under the private directory changes but its own log and its list of what it left out. A piece that left no
case or no answer is ``NOT_WRITTEN`` by its protocol; a case served_composite.sh would compose that is not its piece's
with its Lean's answer is ``LEFT_OUT`` and its directory goes into ``ONSITE_SKIP``, named as that script's scan names
it; either way the step goes on to the products. Only its own failure (a Python that fails) stops it."""

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
#: a directory that is not the piece's: PoUS's of another window, the wipe's of an earlier run (its name holds a space)
ELSEWHERE = {POUS: "pous/w4", WIPE: "earlier run/wipe"}
SUMMARY = {POUS: "pous/pous.json", WIPE: "wipe/summary.json"}
NONE_MADE = "COMPOSITE CASES: PoUS's and the wipe's as their pieces wrote them, none made here"
#: served_composite.sh up to its composing: its settings, `say` and `exe`, this file sourced, the call, then the products,
#: which are handed no directory of ONSITE_SKIP (its `skips`)
STEP = """set -uo pipefail
say() { echo "$*" | tee -a "$OUT/verdicts.txt"; }
exe() { echo "exe $*" >> "$CALLS"; return 1; }
. "$ONSITE_PIECES"
onsite_cases
echo composed
for s in ${ONSITE_SKIP[@]+"${ONSITE_SKIP[@]}"}; do echo "skip $s"; done
"""


def _exe(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(0o755)
    return path


def _write(path: Path, doc: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, sort_keys=True))


def _case(directory: Path, protocol: str, verdict: str = "accept") -> None:
    _write(directory / "case.json", {"commitment": "c0", "evidence": "beef", "params": {}})
    _write(directory / "answer.json", {"commitment": "d1", "protocol": protocol, "verdict": verdict})


def _files(root: Path) -> dict[Path, bytes]:
    return {f: f.read_bytes() for f in sorted(root.rglob("*")) if f.is_file()}


def _own(d: Path) -> dict[Path, bytes]:
    """The step's own two files under the private directory, the only ones it writes."""
    return {f: f.read_bytes() for f in (d / "private" / "composite" / "onsite_cases.log",
                                        d / "private" / "composite" / "onsite_cases.skip")}


@pytest.fixture
def served(tmp_path) -> tuple[Path, Path]:
    """A serve dir and a run's public pieces as window 3's pieces leave them: PoUS's case and the wipe's, each beside
    its Lean's answer, with each piece's summary naming that Lean; beside them the bandwidth case PoUS's piece writes
    and an earlier composite's own output, which served_composite.sh never composes."""
    d, out = (tmp_path / "serve").resolve(), (tmp_path / "out").resolve()
    private = d / "private"
    for protocol in (POUS, WIPE):
        _case(private / WHERE[protocol], protocol)
        _write(out / "pieces" / SUMMARY[protocol], {"decided_by": {"host": "node", "program": LEAN[protocol]},
                                                     "verdict": "accept", "window": 3})
    _case(private / "bandwidth", BANDWIDTH)
    _case(private / "composite" / "onsite" / "private", POUS, "reject")
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


def _went_on(r: subprocess.CompletedProcess) -> list[str]:
    """The step returned to the script, which went on to the products: the directories of its ONSITE_SKIP."""
    said = r.stdout.splitlines()
    assert r.returncode == 0 and "composed" in said, r.stdout + r.stderr
    after = said[said.index("composed") + 1:]
    assert all(line.startswith("skip ") for line in after), after
    return [line.removeprefix("skip ") for line in after]


def _said(out: Path) -> list[str]:
    return (out / "verdicts.txt").read_text().splitlines()


def _left_out(protocol: str) -> str:
    return (f"COMPOSITE LEFT_OUT {protocol}: a case under the serve dir's private that is not its piece's with its "
            "Lean's answer (onsite_cases.log on the node)")


def test_each_pieces_case_is_taken_as_written_with_no_card_asked_for_and_no_encoder_run(served, tmp_path):
    """Given each piece's case and its Lean's answer, the step says each case's protocol and verdict and goes on to the
    products, leaving none out. It asks for no card, builds no Lean, runs no encoder and no script, and writes only
    its log, which names the one directory of each protocol that served_composite.sh then composes, and its empty
    list."""
    d, out = served
    before = _files(d / "private")
    r, calls, pythons, lines = _step(tmp_path, d, out)
    assert _went_on(r) == []
    assert _said(out) == [f"COMPOSITE CASE {POUS} accept", f"COMPOSITE CASE {WIPE} accept", NONE_MADE]
    assert calls == [] and pythons == ["-"]
    assert lines == [("CASE", {"piece": str(d / "private" / WHERE[p]), "protocol": p, "verdict": "accept"})
                     for p in (POUS, WIPE)]
    assert _files(d / "private") == before | _own(d)


@pytest.mark.parametrize("protocol", [POUS, WIPE])
@pytest.mark.parametrize("missing", [("answer.json",), ("case.json",), ("case.json", "answer.json")])
def test_a_piece_that_left_no_answer_or_no_case_is_not_written_and_none_is_made(served, tmp_path, protocol, missing):
    """A missing answer, a missing case, or neither there: ``NOT_WRITTEN`` by the protocol, the other piece's case as
    written, and the composite goes on without it, with nothing to leave out: served_composite.sh composes no
    directory that lacks either file. No card, no Lean and no encoder make the missing file, and why stays in the
    private log: the public lines hold no path and no reason."""
    d, out = served
    for name in missing:
        (d / "private" / WHERE[protocol] / name).unlink()
    before = _files(d / "private")
    r, calls, pythons, lines = _step(tmp_path, d, out)
    assert _went_on(r) == []
    want = {p: f"COMPOSITE NOT_WRITTEN {p}" if p == protocol else f"COMPOSITE CASE {p} accept" for p in (POUS, WIPE)}
    assert _said(out) == [want[POUS], want[WIPE], NONE_MADE]
    assert calls == [] and pythons == ["-"]
    assert _files(d / "private") == before | _own(d)
    (reason,) = [doc["reason"] for head, doc in lines if head == "NOT_WRITTEN" and doc["protocol"] == protocol]
    assert ("no case" if "case.json" in missing else "no answer") in reason
    assert str(d) not in (out / "verdicts.txt").read_text() and reason not in (out / "verdicts.txt").read_text()


def _killed(d: Path, out: Path, protocol: str) -> None:
    """The piece wrote its case and its answer and was killed before it published its summary."""
    (out / "pieces" / SUMMARY[protocol]).unlink()


def _python_decided(d: Path, out: Path, protocol: str) -> None:
    summary = out / "pieces" / SUMMARY[protocol]
    _write(summary, json.loads(summary.read_text()) | {"decided_by": {"host": "node", "program": "python"}})


def _another_verdict(d: Path, out: Path, protocol: str) -> None:
    _case(d / "private" / WHERE[protocol], protocol, "reject")


def _elsewhere(d: Path, out: Path, protocol: str) -> None:
    _case(d / "private" / ELSEWHERE[protocol], protocol)


def _killed_with_another(d: Path, out: Path, protocol: str) -> None:
    _killed(d, out, protocol)
    _elsewhere(d, out, protocol)


@pytest.mark.parametrize("protocol", [POUS, WIPE])
@pytest.mark.parametrize("change, written, where, why", [
    (_killed, False, [WHERE], "published no piece"),
    (_python_decided, False, [WHERE], "not its Lean's"),
    (_another_verdict, False, [WHERE], "its piece published accept"),
    (_elsewhere, True, [ELSEWHERE], "not in its piece's directory"),
    (_killed_with_another, False, [ELSEWHERE, WHERE], "published no piece"),
])
def test_a_case_that_is_not_its_pieces_with_its_leans_answer_is_left_out_and_the_composite_goes_on(
        served, tmp_path, protocol, change, written, where, why):
    """served_composite.sh composes each case under the private directory, so one of PoUS's or the wipe's that the
    step can't name as its piece's with its Lean's answer (of a piece killed before it published its summary, answered
    by the Python alone, of another verdict than the piece published, or in another directory beside the piece's own:
    a second window's, one an earlier run left) goes into ONSITE_SKIP, which that script hands no product, and the
    step goes on: the other protocol's case is taken as written. The piece's own case is ``NOT_WRITTEN`` where it is
    the one left out. One public line for each says the protocol; the directory and why stay in the private log."""
    d, out = served
    change(d, out, protocol)
    before = _files(d / "private")
    r, calls, pythons, lines = _step(tmp_path, d, out)
    skipped = [str(d / "private" / w[protocol]) for w in where]
    assert _went_on(r) == sorted(skipped)
    want = {p: [f"COMPOSITE CASE {p} accept"] if p != protocol or written else [f"COMPOSITE NOT_WRITTEN {p}"]
            for p in (POUS, WIPE)}
    want[protocol] += [_left_out(protocol)] * len(skipped)
    assert _said(out) == [*want[POUS], *want[WIPE], NONE_MADE]
    assert calls == [] and pythons == ["-"]
    left = [doc for head, doc in lines if head == "LEFT_OUT"]
    assert [(doc["piece"], doc["protocol"]) for doc in left] == [(s, protocol) for s in sorted(skipped)]
    assert all(why in doc["reason"] for doc in left), left
    assert _files(d / "private") == before | _own(d)
    public = (out / "verdicts.txt").read_text()
    assert str(d) not in public and not any(doc["reason"] in public for doc in left)


def test_a_directory_left_out_is_named_as_the_script_was_given_the_serve_dir_not_as_it_resolves(served, tmp_path):
    """served_composite.sh leaves a directory out where its own name for it, under the SERVE_DIR it was given, is one
    of its skips: through a link to the serve dir, ONSITE_SKIP names the directory through the link."""
    d, out = served
    _elsewhere(d, out, POUS)
    link = tmp_path / "link"
    link.symlink_to(d, target_is_directory=True)
    r, _, _, lines = _step(tmp_path, link, out)
    assert _went_on(r) == [f"{link}/private/{ELSEWHERE[POUS]}"]
    assert lines[0] == ("CASE", {"piece": f"{link}/private/{WHERE[POUS]}", "protocol": POUS, "verdict": "accept"})


def test_cases_that_could_not_be_read_stop_the_composite(served, tmp_path):
    """A Python that fails leaves the two cases unnamed: the step says so and ends with exit 1, composing nothing."""
    d, out = served
    r, calls, _, lines = _step(tmp_path, d, out, python="#!/bin/sh\nexit 2\n")
    assert r.returncode == 1 and "composed" not in r.stdout.splitlines(), r.stdout + r.stderr
    assert _said(out) == ["COMPOSITE: not run, onsite's own cases weren't read (exit 2, onsite_cases.log on the node)"]
    assert calls == [] and lines == []
