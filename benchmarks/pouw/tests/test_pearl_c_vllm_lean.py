"""``served_passes.py lean``, the served Pearl+ flow's Lean step, against a stand-in ``pouw-pearlplus`` (a shell script that
records its command line and prints what a test gives it) over a served run ``verify`` judged: two honest passes of one
epoch (`test_pearl_c_vllm.proved_pass`) in a serve dir, verified at 3 draws on the node's draw, its coins fixed.

The step runs the executable on the passes that run's document names, the epoch's coins, its draws and its coin record
(PR #7221's command line), and writes the Lean's answer, W at ``tie``, only where the Lean's printout is the Python's on
every field #7221's difftest holds (`LEAN_HELD`), its ``drawn`` the draws and its W, where it names one, the Python's.
Anything else (a field that differs, an exit code that isn't its verdict's, a printout that isn't one, no coin record,
passes other than those the verdict read, no executable) writes none and removes one there, and both verdicts go to the
check document either way. The real executable's run is `benchmarks/one_stage`'s ``test_served_pearlplus_lean``."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import stat
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")

HERE = Path(__file__).resolve().parent


def _module(name: str, path: Path):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        sys.modules[name] = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sys.modules[name])
    return sys.modules[name]


T = _module("test_pearl_c_vllm", HERE / "test_pearl_c_vllm.py")
SP = T.load("pearl_c_vllm_served_passes", T.KERNEL.parent / "pearl_c_vllm" / "served_passes.py")
H100 = "pearl-c-h100-v1-h3"
SOURCE = b"verifier" * 4
W = "d7" * 64
pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def run(tmp_path_factory) -> tuple[Path, Path, dict]:
    """The serve dir, ``verify``'s document's path and the document, accepted."""
    from verity.core.protocols.pouw import verifier as PV

    tmp = tmp_path_factory.mktemp("lean")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(PV, "coins", lambda n: SOURCE[:n])
        for k in ("PEARLC_CHALLENGER_GATE", "PEARLC_CHALLENGER_PEER"):
            mp.delenv(k, raising=False)
        a, b = (T.proved_pass(tmp / name, H100, ix)[0] for name, ix in (("a", (0, 1)), ("b", (2, 3))))
        d = T._serve_dir(tmp, [a, b])
        rc, doc, _answer = T.served_verify(d, tmp / "piece" / "verify.json")
    assert rc == 0 and doc["answer"]["verdict"] == "accept" and doc["source"] == SOURCE.hex(), doc["line"]
    return d, tmp / "piece" / "verify.json", doc


def stub(tmp_path: Path, said: dict | str, rc: int) -> Path:
    """A ``pouw-pearlplus`` that writes its arguments to ``argv`` beside it, prints `said` and exits `rc`."""
    (tmp_path / "said").write_text(said if isinstance(said, str) else json.dumps(said) + "\n")
    exe = tmp_path / "pouw-pearlplus"
    exe.write_text('#!/bin/sh\nd=$(dirname "$0")\nprintf \'%s\\n\' "$@" > "$d/argv"\ncat "$d/said"\nexit ' + str(rc) + "\n")
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR)
    return exe


def printout(doc: dict, **kw) -> dict:
    """The Lean's printout where it is the Python's verdict `doc`, with `kw` over it."""
    return {**SP.python_answer(doc), "drawn": doc["draws"], **kw}


def lean(tmp_path: Path, verify: Path, d: Path, exe: Path, *more: str) -> tuple[int, dict | None, dict]:
    out = tmp_path / "piece" / "pearlplus-lean.json"
    rc = SP.main(["lean", "--passes", str(d), "--verify", str(verify), "--exe", str(exe), "--out", str(out), *more])
    check = json.loads((out.with_name(SP.LEAN_CHECK)).read_text())
    return rc, json.loads(out.read_text()) if out.exists() else None, check


def test_an_agreeing_lean_verdict_is_written_where_the_compose_reads_it_with_both_verdicts_and_its_command_line(run, tmp_path):
    """The step runs the executable on the run's passes in order, the epoch's coins, the draws and the coin record verify
    wrote, and writes the Lean's printout as its answer; the check document holds both verdicts, the command line and
    the executable's SHA-256."""
    d, verify, doc = run
    got = printout(doc)
    exe = stub(tmp_path, got, 0)
    rc, answer, check = lean(tmp_path, verify, d, exe)
    passes = [Path(p["manifest"]) for p in json.loads((d / "served.json").read_text())["passes"]]
    argv = [str(exe), "--coins", str(d / "coins.json"), "--draws", "3", "--record", str(d / SP.COIN), *map(str, passes)]
    assert rc == 0 and answer == {**got, "tie": None}
    assert (tmp_path / "argv").read_text().splitlines() == argv[1:] and check["argv"] == argv
    assert (check["reason"], check["differ"], check["lean"], check["python"]) == (None, [], got, SP.python_answer(doc))
    assert check["exe_sha256"] == hashlib.sha256(exe.read_bytes()).hexdigest() and check["rc"] == 0
    assert answer["work"] == doc["answer"]["public"]["work"] and answer["source"] == SOURCE.hex()


def test_a_lean_verdict_that_differs_on_any_held_field_writes_no_answer_and_removes_one_there(run, tmp_path):
    """Each field the difftest holds, ``drawn`` and the W the Lean names: one differing leaves no answer (a stale one
    removed), exit 2, and the check document names the fields beside both verdicts."""
    d, verify, doc = run
    work = doc["answer"]["public"]["work"]
    cases = [({"work": {**work, "credited": work["credited"] + 1}}, ["work"]),
             ({"source": "00" * 32}, ["source"]),
             ({"commitment": "00" * 64, "verdicts": []}, ["commitment", "verdicts"]),
             ({"drawn": 4}, ["drawn"]),
             ({"public": {"tie": "ab" * 64}}, ["tie"])]
    out = tmp_path / "piece" / "pearlplus-lean.json"
    for kw, differ in cases:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("{}")
        rc, answer, check = lean(tmp_path, verify, d, stub(tmp_path, printout(doc, **kw), 0))
        assert (rc, answer, check["differ"], check["answer"]) == (2, None, differ, None), kw
        assert check["lean"] == printout(doc, **kw) and check["python"] == SP.python_answer(doc)
        assert check["reason"] == f"{SP.LEAN_DISAGREES}: they differ in {', '.join(differ)}"


def test_the_lean_verdicts_w_is_the_answers_tie_and_must_be_the_pythons(run, tmp_path):
    """A Lean that names W (as ``public.tie`` or ``tie``, a digest or ``{"record": digest}``) agrees only on the W the
    Python's verdict held the passes to, and its answer carries it at ``tie.record``, where the compose reads it."""
    d, verify, doc = run
    tied = {**doc, "tie": {"record": W, "tables": 8}}
    (v := tmp_path / "verify.json").write_text(json.dumps(tied))
    for kw in ({"public": {"tie": W}}, {"tie": W}, {"tie": {"record": W}}):
        rc, answer, _ = lean(tmp_path, v, d, stub(tmp_path, printout(doc, **kw), 0))
        assert rc == 0 and answer["tie"]["record"] == W, kw
    rc, answer, check = lean(tmp_path, verify, d, stub(tmp_path, printout(doc, tie=W), 0))
    assert (rc, answer, check["differ"]) == (2, None, ["tie"])


def test_an_agreeing_rejection_is_written_and_the_lean_exit_code_must_be_its_verdicts(run, tmp_path):
    """A rejection both give is written (exit 1), for the compose to name; a printout whose exit code isn't its verdict's,
    or that isn't one JSON object, is no verdict."""
    d, verify, doc = run
    no = {**doc, "answer": {**doc["answer"], "verdict": "reject", "reason": "pass 0 tile 3: B opening", "public": None}}
    (v := tmp_path / "verify.json").write_text(json.dumps(no))
    rc, answer, _ = lean(tmp_path, v, d, stub(tmp_path, printout(no), 1))
    assert rc == 1 and (answer["verdict"], answer["reason"], answer["work"]) == ("reject", "pass 0 tile 3: B opening", None)
    for said, code in ((printout(no), 0), (printout(doc), 1), ("lake: no such target\n", 1), ("[1, 2]\n", 0), ("", 0)):
        rc, answer, check = lean(tmp_path, verify, d, stub(tmp_path, said, code))
        assert (rc, answer, check["lean"]) == (2, None, None) and check["reason"].startswith(f"the Lean exited {code}")


def test_the_step_refuses_before_the_lean_without_a_coin_record_the_verdicts_passes_or_an_executable(run, tmp_path):
    """No coin record (nothing pins the Lean's source to the Python's), passes other than those the verdict read (a pass's
    manifest changed since), or no executable: no answer, and the executable never runs."""
    d, verify, doc = run
    exe = stub(tmp_path, printout(doc), 0)
    (v := tmp_path / "verify.json").write_text(json.dumps({**doc, "coin_record": None}))
    rc, answer, check = lean(tmp_path, v, d, exe)
    assert (rc, answer, check["argv"]) == (2, None, None) and "recorded no coin" in check["reason"]
    (v := tmp_path / "verify.json").write_text(json.dumps({**doc, "passes": doc["passes"][:1]}))
    rc, answer, check = lean(tmp_path, v, d, exe)
    assert (rc, answer) == (2, None) and "not the ones the Python verdict read" in check["reason"]
    rc, answer, check = lean(tmp_path, verify, d, tmp_path / "none")
    assert (rc, answer) == (2, None) and check["reason"] == f"no Lean executable at {tmp_path / 'none'}"
    assert not (tmp_path / "argv").exists()
