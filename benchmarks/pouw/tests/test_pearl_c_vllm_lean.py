"""``served_passes.py lean``, the served Pearl+ flow's Lean step, against a stand-in ``pouw-pearlplus`` (a shell script that
records its command line and prints what a test gives it) over a served run ``verify`` judged: two honest passes of one
epoch (`test_pearl_c_vllm.proved_pass`) in a serve dir, verified at 3 draws on the node's draw, its coins fixed.

The step runs the executable on the passes that run's document names, the epoch's coins, its draws, its coin record, the W
``verify --tie`` held the passes to and, on a live session's coins, ``--of-record`` (`lean_argv`), and writes the Lean's
answer, W at ``tie``, only where the Lean's printout is the Python's on every field #7221's difftest holds (`LEAN_HELD`),
its ``drawn`` the draws and its W the Python's (or none on both).
Anything else (a field that differs, an exit code that isn't its verdict's, a printout that isn't one, no coin record,
passes other than those the verdict read, no executable) writes none and removes one there, and both verdicts go to the
check document either way. The real executable's run is `benchmarks/one_stage`'s ``test_served_pearlplus_lean``.

``served_passes.py case`` on the Lean answer written there writes the run's case's parts for remote's composite of record
(`verity/pouw/pearl-plus/v1`): its params, the retained manifests, the evidence of each file the Lean's opened tiles read
and those files' bytes once each, private; it writes none without a Lean answer or a coin record, on passes the verdict
didn't read, or on a Lean answer over another source."""
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
    from verity.protocols.pouw.verifier import verifier as PV

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
    argv = [str(exe), "--coins", str(d / "coins.json"), "--draws", "3", "--record", str(d / SP.COIN), "--over-cap",
            doc["over_cap"], "--test-instance", *map(str, passes)]
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
    rc, answer, check = lean(tmp_path, v, d, stub(tmp_path, printout(doc), 0))
    assert (rc, answer, check["differ"]) == (2, None, ["tie"])


def test_the_lean_is_given_w_and_on_live_coins_is_asked_for_a_verdict_of_record(run, tmp_path):
    """The W ``verify --tie`` named goes to the Lean as ``--tie W``, and a run on a live session's coins (``coins.json`` of
    `window.LIVE_SCHEME`) adds ``--of-record``; both after the over-cap rule and test instance and before the passes,
    which stay last."""
    import shutil
    from verity.protocols.pouw.verifier import window as Wd
    from verity.protocols.spot.device import challenger as Ch

    d, verify, doc = run
    (wfile := tmp_path / "w.json").write_text("{}")
    (v := tmp_path / "verify.json").write_text(json.dumps({**doc, "tie_record": str(wfile), "tie": {"record": W}}))
    exe = stub(tmp_path, printout(doc, tie={"record": W, "tables": 8}), 0)
    passes = [str(Path(p["manifest"])) for p in json.loads((d / "served.json").read_text())["passes"]]
    rc, _answer, check = lean(tmp_path, v, d, exe)
    head = [str(exe), "--coins", str(d / "coins.json"), "--draws", "3", "--record", str(d / SP.COIN), "--over-cap",
            doc["over_cap"], "--test-instance"]
    assert rc == 0 and check["argv"] == [*head, "--tie", str(wfile), *passes]
    (live := tmp_path / "live").mkdir()
    shutil.copytree(d, live, dirs_exist_ok=True)
    coins = Ch.LiveChallenger(Wd.LIVE_SCHEME, 0, Wd.SALT_FAMILIES, lambda n: b"\x01" * n).commitment
    (live / "coins.json").write_text(json.dumps(coins.to_json()))
    rc, _answer, check = lean(tmp_path, v, live, exe)
    head = [str(exe), "--coins", str(live / "coins.json"), "--draws", "3", "--record", str(d / SP.COIN), "--over-cap",
            doc["over_cap"], "--test-instance"]
    assert check["argv"] == [*head, "--tie", str(wfile), "--of-record", *passes]


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


def case(d: Path, verify: Path, answer: Path, out: Path) -> int:
    return SP.main(["case", "--passes", str(d), "--verify", str(verify), "--lean", str(answer), "--out", str(out)])


def test_case_writes_the_runs_case_parts_and_each_file_its_opened_tiles_read_privately(run, tmp_path, capsys):
    """The params are the epoch's coins, the draws and the coin record's run and source; the manifests the retained
    passes' in order; the evidence, per pass, each file the Lean's opened tiles read (`tile_files`, drawn and exclusion
    tiles alike) under the SHA-512 of its bytes, and the blobs exactly those files, each once, 0700 with files 0600."""
    d, verify, doc = run
    assert lean(tmp_path, verify, d, stub(tmp_path, printout(doc), 0))[0] == 0
    answer, out = tmp_path / "piece" / "pearlplus-lean.json", tmp_path / "case"
    capsys.readouterr()
    assert case(d, verify, answer, out) == 0
    assert capsys.readouterr().out.startswith("case accept: 2 passes, ")
    parts, coin = json.loads((out / SP.CASE_PARTS).read_text()), json.loads((d / SP.COIN).read_text())
    paths = [Path(p["manifest"]) for p in json.loads((d / "served.json").read_text())["passes"]]
    assert (parts["schema"], parts["delta"]) == (SP.CASE_SCHEMA, doc["delta"])
    assert parts["params"] == {"coins": json.loads((d / "coins.json").read_text()), "draws": 3,
                               "record": {"commitment": coin["commitment"], "source": SOURCE.hex()}}
    assert parts["manifests"] == [json.loads(p.read_text()) for p in paths]
    assert parts["lean"] == {"verdict": "accept", "reason": None, "drawn": 3, "work": doc["answer"]["public"]["work"]}
    coins = T.VR.epoch_coins(d / "coins.json")
    passes, want = [T.VR.load(p, coins) for p in paths], [{} for _ in paths]
    for v in doc["verdicts"]:
        for spec in SP.tile_files(passes[v["pass"]], v["tile"]):
            data = (paths[v["pass"]].parent / spec["path"]).read_bytes()
            want[v["pass"]][spec["path"]] = hashlib.sha512(data).hexdigest()
    assert parts["evidence"] == {"files": want} and all(want)
    blobs = out / SP.CASE_BLOBS
    assert sorted(f.name for f in blobs.iterdir()) == sorted({h for f in want for h in f.values()})
    assert all(hashlib.sha512(f.read_bytes()).hexdigest() == f.name for f in blobs.iterdir())
    assert stat.S_IMODE(out.stat().st_mode) == stat.S_IMODE(blobs.stat().st_mode) == 0o700
    assert all(stat.S_IMODE(f.stat().st_mode) == 0o600 for f in out.rglob("*") if f.is_file())


def test_case_writes_nothing_without_the_lean_answer_its_coin_record_or_its_passes_or_on_another_source(run, tmp_path):
    d, verify, doc = run
    assert lean(tmp_path, verify, d, stub(tmp_path, printout(doc), 0))[0] == 0
    answer = tmp_path / "piece" / "pearlplus-lean.json"
    (other := tmp_path / "other.json").write_text(json.dumps({**json.loads(answer.read_text()), "source": "00" * 32}))
    (no_coin := tmp_path / "no-coin.json").write_text(json.dumps({**doc, "coin_record": None}))
    (fewer := tmp_path / "fewer.json").write_text(json.dumps({**doc, "passes": doc["passes"][:1]}))
    for n, (v, a) in enumerate(((verify, tmp_path / "none.json"), (no_coin, answer), (fewer, answer), (verify, other))):
        assert case(d, v, a, tmp_path / f"case{n}") == 2, n
        assert not (tmp_path / f"case{n}").exists(), n
