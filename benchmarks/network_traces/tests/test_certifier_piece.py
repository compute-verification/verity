"""certifier_piece: the front sends one group of the window circuit's class through the gate's scan rehearsal and keeps
the gate's fetch, witness and all, 0700 under the serve dir's private/certifier, publishing only the bundle less witness/
and its outcome. The piece, on a serve dir whose window was the network certifier's synthetic run's certified requests
(the run `test_warden_chain.gate_run` publishes in the gate's layout), runs as `served_verdicts.py pieces` runs it and writes
`network-period`'s answer only when the chain accepted and the window's requests are certified ones in the scan's
link-window: the period mirror's certifier/v0 accept (`period_case`), its profile with the placement (the certifier on
the kept gate's host, the checks on the prover's node), then goes through `served_verdicts.py record` into the window
report and the block record's certifier leg, which `block_record.check` holds and its schema admits, and its case.json
is the three files `network-period` read, as `composite.py` reads a piece's case, each field of DIR's of a public kind
or a certificate (`kind_scan`); the chain opens no endpoint records, and its report stays in the kept run. A window
request the warden didn't certify, and network certifier records that `network-rows` refuses, are rejected before any
chain runs (`network-rows` itself accepts the synthetic run's records and refuses them with cells re-cut, re-clocked or
reordered, or re-signed under another key than the auditor's record's); a chain that refused, requests outside the
scan's link-window and a period the check rejects are rejects; a rehearsal the gate's check failed (each request late)
and a step that can't run (no kept run, no auditor's record of the run's deployment or a fetch that isn't the record's,
no report, no build, the time limit) refuse, which leaves the certifier missing, and no process of a step's group outlives
the piece, the time limit's SIGTERM included, whenever it comes. Of the chain, DIR gets only `PUBLIC`'s key paths,
which hold none of the window's counts or advice bits; the leak guard reads the piece's DIR against the run's witness
and moves out any file DIR doesn't declare, and the case is written only from the period files it left. The front's
warm-ups go straight to the endpoint, one at a time, each recorded in a closed window of its own before the gate is
asked, and the pick passes over those windows for the group's. At a tick of 4000 ms the front's rehearsal reaches this
tree's gate with the tick in its request and the minutes counted at it, and the warden runs the scan's schedule there
(`certifier_piece.gate_params`: link-windows of 32 buckets, each a window of the window circuit's class 8,32,4,4,16);
a tick the gate refuses (past its range, or on a request that isn't the scan rehearsal) refuses the front before the
gate is asked. An attack's front puts the prover's relay between the gate and the endpoint, and ends it with the
front."""
from __future__ import annotations

import argparse
import hashlib
import http.server
import importlib.util
import json
import os
import re
import shlex
import shutil
import signal
import socket
import socketserver
import subprocess
import sys
import tarfile
import threading
from dataclasses import replace
from pathlib import Path

import jsonschema
import pytest

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(Path(__file__).resolve().parent), str(ROOT / "benchmarks" / "one_stage")]

import attacks as AT  # noqa: E402
import certifier_piece as CP  # noqa: E402
import chain_record as CR  # noqa: E402
import kind_scan as KS  # noqa: E402
import scan_period as SP  # noqa: E402
import served_verdicts as S  # noqa: E402
import warden_serve as WS  # noqa: E402
from test_scan_zk import published, sz  # noqa: E402
from test_served_session import RUN, STUB  # noqa: E402
from test_warden_chain import WC, gate_run, warden_network  # noqa: E402
from verity.core.protocols.certifier import warden as W  # noqa: E402
from verity.core.protocols.sampled_proofs.one_stage import registration as RG  # noqa: E402
from verity.ml.circuits.codec import canonical_json  # noqa: E402
from verity_e2e_demo import block_record as BR  # noqa: E402

PERIOD_CASE = ROOT / "verity" / "core" / "protocols" / "network" / "tests" / "period_case.py"
SCHEMA = json.loads((Path(BR.__file__).parent / "schemas" / "block-record.v1.schema.json").read_text())
K = 3
#: the gate's end of a rehearsal its check passed, and of one whose every request was late past the network certifier's
#: D·τ
PASSED = {"run": RUN, "rc": 0, "ended": "the rehearsal ended", "verdict": {"accepted": True, "problems": 0}}
LATE = {"run": RUN, "rc": 1, "ended": "the rehearsal exited 1",
        "verdict": {"accepted": False, "problems": 8, "requests": 4, "ended": 0, "failed": ["fail"]}}
GATE_HOST = "gate-host"
#: the window's private values, as numbers no other field of the chain's files takes: its request and output counts
REQUESTS, OUTPUTS = 7919, 7933
PRIVATE = {REQUESTS, OUTPUTS}


@pytest.fixture(scope="module")
def period():
    spec = importlib.util.spec_from_file_location("period_case", PERIOD_CASE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["period_case"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def warden(tmp_path_factory):
    """The synthetic network certifier run, its fetch (``late``: the same run, its rehearsal failed late; ``resigned``:
    the run re-signed under another network certifier key; ``rewritten``: the run with its out/ copy of
    deployment.json the re-signed run's), the auditor's record of its deployment (``key``, `warden_chain.own_record`'s,
    the file ``CERTIFIER_KEY`` names), and the execution phase holding the most whole certified requests."""
    d = tmp_path_factory.mktemp("warden")
    run = gate_run(d)
    io, T = WC.io(run), WC.deployment(run)[1].window
    by: dict[int, list[dict]] = {}
    for r in io["records"]:
        ws = {b // T for b in [r["release"], *(b for _, b in r["frames"])]}
        if len(ws) == 1:
            by.setdefault(ws.pop(), []).append(r)
    w = max(by, key=lambda x: (len(by[x]), -x))
    other = gate_run(d / "other", seed=b"another network certifier")
    shutil.copytree(run, rewritten := d / "rewritten")
    shutil.copyfile(other / "out" / "deployment.json", rewritten / "out" / "deployment.json")
    fetches = {}
    for name, end, x in (("late", LATE, run), ("fetch", PASSED, run), ("resigned", PASSED, other),
                         ("rewritten", PASSED, rewritten)):
        (x / "end.json").write_text(json.dumps(end))
        fetches[name] = d / f"{name}.tar.gz"
        with tarfile.open(fetches[name], "w:gz") as tar:
            tar.add(x, arcname=RUN)
    (key := d / "certifier-key.json").write_text(json.dumps(WC.own_record(run)))
    dep = WC.deployment(run)[0]
    return {**fetches, "key": key, "w": w, "records": by[w][:4],
            "others": [r for x, rs in by.items() if x != w for r in rs], "windows": dep["windows"],
            "all": io["records"], "run_id": dep["run_id"]}


@pytest.fixture(autouse=True)
def _certifier_key(request, monkeypatch):
    """``CERTIFIER_KEY``, the auditor's record of the kept run's deployment as the follow-on exports it, for each test
    on the synthetic run."""
    if "warden" in request.fixturenames:
        monkeypatch.setenv(CP.CERTIFIER_KEY, str(request.getfixturevalue("warden")["key"]))


def _auditors(warden: dict) -> dict:
    """The auditor's record of the synthetic run's deployment."""
    return json.loads(warden["key"].read_text())


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _serve_dir(d: Path, warden: dict, records: list[dict] | None = None, kept: bool = True,
               fetch: str = "fetch") -> Path:
    """A serve dir whose Commit period K served ``records`` (the network certifier's, by default those of its
    execution phase) and Commit period K + 1 the network certifier's other certified requests, each released in its
    run, and whose gate fetch (``warden[fetch]``) the front kept."""
    records = warden["records"] if records is None else records
    d.mkdir(parents=True)
    rest = [r for r in warden["all"] if r["user"] not in {x["user"] for x in records}]
    reqs = [{"engine_id": f"e{i}", "user": r["user"], "prompt_ids": r["prompt"], "streamed_ids": r["output"],
             "windows": [K if i < len(records) else K + 1], "release": [warden["run_id"], r.get("stream", i)],
             "check": {"passed": True}} for i, r in enumerate([*records, *rest])]
    _write(d / "requests.jsonl", reqs)
    reqs = reqs[:len(records)]
    _write(d / "seals.jsonl", [{"window": K, "requests": [r["engine_id"] for r in reqs]}])
    _write(d / "windows.jsonl", [{"window": K, "passed": True, "commit": {"steps": 9}}])
    _write(d / "advice_trace.jsonl", [{"step": 0}])
    (d / "serve.json").write_text("{}")
    if kept:
        (d / CP.PRIVATE / RUN).mkdir(parents=True)
        (d / CP.PRIVATE / RUN / "fetch.tar.gz").write_bytes(warden[fetch].read_bytes())
        (d / CP.PRIVATE / RUN / CP.SESSION).write_text(json.dumps({"run": RUN, "gate": f"research@{GATE_HOST}"}))
    return d


def _chain_out(d: Path, verdict: str, w: int) -> Path:
    """What the chain leaves in its run dir's out/: its report and the scan's one link-window."""
    out = d / "out"
    (out / "report").mkdir(parents=True)
    (out / "scan" / "joint").mkdir(parents=True)
    (out / "report" / "report.json").write_text(json.dumps({"verdict": verdict, "why": None if verdict == "accepted"
                                                            else {"n2": "a window refused"}}))
    (out / "scan" / "answers.json").write_text(json.dumps({"registration": {"program": "ab" * 64}, "windows": [
        {"window": w, "answer": {"verdict": "accept"}, "registrations": []}]}))
    (out / "scan" / "joint" / "program.json").write_text("{}")
    return d


def _stub_lean(tmp_path: Path, answer: dict) -> Path:
    """A stand-in `network-period` that keeps the three files it reads in TMP/seen and answers ``answer``."""
    lean, seen = tmp_path / "network-period", shlex.quote(str(tmp_path / "seen"))
    (tmp_path / "lean-answer.json").write_text(json.dumps(answer, sort_keys=True))
    lean.write_text(f"#!/bin/sh\nmkdir -p {seen} && cp \"$2\" \"$3\" \"$4\" {seen}/\n"
                    f"cat {shlex.quote(str(tmp_path / 'lean-answer.json'))}\n")
    lean.chmod(0o755)
    return lean


def _stub_rows(tmp_path: Path, answer: dict | None = None) -> Path:
    """A stand-in `network-rows` that answers ``answer``, by default its accept."""
    rows, said = tmp_path / "network-rows", tmp_path / "rows-answer.json"
    said.write_text(json.dumps(answer or {"protocol": CP.ROWS, "verdict": "accept", "reason": None}))
    rows.write_text(f"#!/bin/sh\ncat {shlex.quote(str(said))}\n")
    rows.chmod(0o755)
    return rows


def _piece_cmd(tmp_path: Path, chain: Path, answer: dict, cmd: str | None = None, lean: Path | None = None,
               rows: Path | None = None) -> str:
    """The piece's command, its auditor's record the follow-on's ``CERTIFIER_KEY``, its `network-period`
    `_stub_lean`'s answering ``answer`` and its `network-rows` `_stub_rows`' accept (or ``rows``)."""
    lean, rows = lean or _stub_lean(tmp_path, answer), rows or _stub_rows(tmp_path)
    cmd = cmd or f'mkdir -p "$RESEARCH_RUN_DIR" && cp -r {shlex.quote(str(chain / "out"))} "$RESEARCH_RUN_DIR/"'
    return (f"{shlex.quote(sys.executable)} {shlex.quote(str(HERE / 'certifier_piece.py'))} piece \"$@\" "
            f"--certifier-key \"${{CERTIFIER_KEY:-}}\" --lean {shlex.quote(str(lean))} --rows {shlex.quote(str(rows))} "
            f"--chain-cmd {shlex.quote(cmd)}")


def _integrity() -> dict:
    a = {"protocol": "verity/sampled-proofs/one-stage/v1", "verdict": "accept", "reason": None, "commitment": "ab" * 64,
         "public": {"bound": 2, "drawn": [0, 3], "per_template": None, "proved": [0, 3]},
         "profile": {"levels": [{"name": "unit", "size": 4}], "drawn": {"unit": 2}, "delta": "1/100"}}
    return S.row_of(S.INTEGRITY_OWNER, a, Path("/nonexistent"))


def _record(monkeypatch, capsys, serve: Path, pieces: Path) -> tuple[dict | None, dict]:
    """`served_verdicts.py record` over the pieces `pieces` ran, sampled proofs' row an accept: the block record (None
    where none was written) and the ``RECORD`` line."""
    monkeypatch.setattr(S, "sampled_proofs_row", lambda d, run=None, registration=None: _integrity())
    monkeypatch.setenv("RESEARCH_RUN_ID", "r1")
    out = pieces.parent / "record"
    capsys.readouterr()
    S.main(["record", "--serve-run", str(serve), "--window", str(K), "--sampled-proofs", str(out / "sampled-proofs"),
            "--registration", str(out / "registration.json"), "--pieces", str(pieces), "--out", str(out)])
    (line,) = [x.removeprefix("RECORD ") for x in capsys.readouterr().out.splitlines() if x.startswith("RECORD ")]
    rec = out / "block-record.json"
    return (json.loads(rec.read_text()) if rec.exists() else None), json.loads(line)


def test_the_pieces_certifier_row_is_the_period_checks_accept_and_composes_into_the_block_record(tmp_path, warden,
                                                                                                period, capsys,
                                                                                                monkeypatch):
    accept = period.reference(period.case("accept", 1, "clock"))
    assert accept["verdict"] == "accept" and accept["protocol"] == CP.PROTOCOL
    d = _serve_dir(tmp_path / "serve", warden)
    env = dict(os.environ, CERTIFIER_CMD=_piece_cmd(tmp_path, _chain_out(tmp_path / "chain", "accepted", warden["w"]),
                                                    accept))
    got = S.pieces(d, K, tmp_path / "pieces", env=env)
    piece = tmp_path / "pieces" / "certifier"
    assert (got["certifier"]["verdict"], got["certifier"]["rc"]) == ("accept", 0), (piece / "piece.log").read_text()
    assert "MISSING the network certifier" not in capsys.readouterr().out
    got = json.loads((piece / "answer.json").read_text())
    where = got["profile"].pop("placement")
    assert got == accept
    node = socket.gethostname()
    assert (where["kind"], where["parties"]) == ("not-placed", {"network_certifier": GATE_HOST, "prover": node,
                                                                "verifier": node})
    assert where[CP.PREMISE].startswith("assumed, the auditor's devices not placed")
    assert [(x["name"], x["counted"]) for x in where["links"]] == [("certified", True), ("gate", False)]
    assert json.loads((piece / "chain" / "report" / "report.json").read_text())["verdict"] == "accepted"
    assert {p.name for p in piece.iterdir()} - {"piece.json"} == {"answer.json", CP.CASE, "chain", "piece.log"}
    assert [(f.file, f.path) for f in KS.scan(piece)[0]] == [("piece.log", "")]  # served_verdicts.sh's CERTIFIER_PRIVATE
    seen = {f.name: f.read_bytes() for f in (tmp_path / "seen").iterdir()}
    raw = (piece / CP.CASE).read_bytes()
    case = json.loads(raw)
    assert sorted(case) == ["commitment", "evidence", "params"] and raw == canonical_json(case) + b"\n"
    assert (bytes.fromhex(case["commitment"]), bytes.fromhex(case["evidence"]), canonical_json(case["params"])) == (
        seen["commitment.bin"], seen["evidence.json"], seen["params.json"])
    log = (piece / "piece.log").read_text()
    assert "leak guard (exit 0)" in log and f"case: {CP.CASE}, {len(raw)} bytes" in log
    keep = d / CP.PRIVATE / RUN
    assert sorted(p.name for p in keep.iterdir()) == ["chain", "fetch.tar.gz", CP.SESSION]
    assert (keep / "chain" / "out" / "report" / "report.json").is_file()
    row = S.piece_row("communication", tmp_path / "pieces")
    assert (row["verdict"], row["quantity"], row["owner"]) == ("accept", accept["public"], "network-accounting")
    rec, line = _record(monkeypatch, capsys, d, tmp_path / "pieces")
    verdicts = json.loads((tmp_path / "record" / "verdicts.json").read_text())
    assert verdicts["protocols"]["communication"]["answer"]["profile"]["placement"] == where
    jsonschema.Draft202012Validator(SCHEMA).validate(rec)
    assert BR.check(rec) == {} and not line["check"]
    assert "certifier" not in line["record_missing"] and line["verdicts"]["the network certifier"] == "accept"
    leg = rec["legs"]["certifier"]
    assert leg["quantity"]["symbol"] == "K" and leg["quantity"]["value"] == int(accept["public"]["K"])


def test_a_window_request_the_warden_didnt_certify_is_rejected_before_any_chain(tmp_path, warden, capsys, monkeypatch):
    stranger = {"user": "not-the-wardens", "prompt": [1, 2], "output": [3]}
    d = _serve_dir(tmp_path / "serve", warden, [*warden["records"][:1], stranger])
    env = dict(os.environ, CERTIFIER_CMD=_piece_cmd(tmp_path, tmp_path / "none", {}))
    got = S.pieces(d, K, tmp_path / "pieces", env=env)
    assert got["certifier"]["verdict"] == "reject" and "is no certified request" in got["certifier"]["reason"]
    assert not (d / CP.PRIVATE / RUN / "chain").exists()
    rec, line = _record(monkeypatch, capsys, d, tmp_path / "pieces")
    assert rec is None and (line["composed"], line["by"]) == ("reject", ["the network certifier"])
    changed = dict(warden["records"][0], output=[*warden["records"][0]["output"], 7])
    d = _serve_dir(tmp_path / "changed", warden, [changed])
    a, _ = CP.decide(d, K, tmp_path / "out", record=_auditors(warden), chain=lambda *x: pytest.fail("no chain"),
                     check=lambda *x: {}, rows=lambda run: None)
    assert a["verdict"] == "reject" and "streamed_ids aren't the certified output" in a["reason"]


def test_records_the_rows_audit_refuses_are_the_pieces_reject_before_any_chain(tmp_path, warden, capsys, monkeypatch):
    """``network-rows`` refusing the network certifier's records (a stand-in's reject here) is the piece's reject,
    its reason's numbers as N, before any chain runs, and the composite rejects with it."""
    rows = _stub_rows(tmp_path, {"protocol": CP.ROWS, "verdict": "reject", "reason": "egress bucket 27 isn't the "
                                 "schedule's"})
    d = _serve_dir(tmp_path / "serve", warden)
    env = dict(os.environ, CERTIFIER_CMD=_piece_cmd(tmp_path, tmp_path / "none", {}, rows=rows))
    got = S.pieces(d, K, tmp_path / "pieces", env=env)
    a = json.loads((tmp_path / "pieces" / "certifier" / "answer.json").read_text())
    why = "the network certifier's records, audited as rows: egress bucket N isn't the schedule's"
    assert (a["verdict"], a["reason"]) == ("reject", why) and got["certifier"]["verdict"] == "reject"
    assert why in got["certifier"]["reason"] and not (d / CP.PRIVATE / RUN / "chain").exists()
    rec, line = _record(monkeypatch, capsys, d, tmp_path / "pieces")
    assert rec is None and (line["composed"], line["by"]) == ("reject", ["the network certifier"])


def _decide(tmp_path, warden, verdict="accepted", w=None, records=None, check=None):
    d = _serve_dir(tmp_path / "serve", warden, records)

    def chain(keep, run, windows):
        assert windows == warden["windows"]
        _chain_out(keep / run / "chain", verdict, warden["w"] if w is None else w)
        return 0
    (tmp_path / "dir").mkdir()
    a, unpacked = CP.decide(d, K, tmp_path / "dir", record=_auditors(warden), chain=chain,
                            check=check or (lambda c, p: pytest.fail("no check")), rows=lambda run: None,
                            case=lambda run, scan: {"params": {}, "commitment": "", "evidence": ""})
    assert unpacked is not None and unpacked.parent == d / CP.PRIVATE / RUN
    return a


def test_a_refused_chain_a_request_off_the_scans_link_window_and_a_rejected_period_are_rejects(tmp_path, warden):
    d = _serve_dir(tmp_path / "serve", warden)
    env = dict(os.environ, CERTIFIER_CMD=_piece_cmd(tmp_path, _chain_out(tmp_path / "chain", "refused", warden["w"]), {}))
    got = S.pieces(d, K, tmp_path / "pieces", env=env)
    assert got["certifier"]["verdict"] == "reject" and got["certifier"]["reason"].startswith("the chain refused")
    assert not (tmp_path / "pieces" / "certifier" / CP.REFUSED).exists()
    a = _decide(tmp_path / "refused", warden, "refused")
    assert a["verdict"] == "reject" and a["reason"].startswith("the chain refused: {\"n2\": \"a window refused\"}")
    assert a == CP.answer("reject", a["reason"]) and S.check_answer(a) == a
    a = _decide(tmp_path / "off", warden, w=warden["w"] + 1)
    assert a["verdict"] == "reject" and f"not the period's {warden['w'] + 1} alone" in a["reason"]
    if warden["others"]:
        a = _decide(tmp_path / "mixed", warden, records=[*warden["records"][:1], warden["others"][0]])
        assert a["verdict"] == "reject" and "spans link-windows" in a["reason"]
    rejected = CP.answer("reject", "window 0: certificates")
    assert _decide(tmp_path / "period", warden, check=lambda c, p: rejected) == rejected


LOOP = "while :; do :; done"


def _gone(pids) -> None:
    """Each of ``pids`` has exited (no process, or a zombie its reaper hasn't reaped); one that hasn't fails the test,
    killed first so it doesn't outlive it."""
    alive = [pid for pid in pids if CP.live(pid)]
    for pid in alive:
        os.kill(pid, signal.SIGKILL)
    assert not alive, f"{alive} outlived the piece's step"


def test_a_step_leaves_no_process_of_its_group_behind_however_it_ends(tmp_path, monkeypatch):
    """`run_group` ends a step's process group when its leader exits (a background process it left included), when the
    piece's time limit's SIGTERM comes while the step runs (held until the group has ended, then `NotRun`), SIGKILLing a
    process that ignores SIGTERM after `GRACE_S` (0 here), and when the SIGTERM comes while `subprocess.Popen` is still
    starting the step, before the piece holds the child, the time-limit case's orphaned busy loop."""
    pids = tmp_path / "pids"
    assert CP.run_group(["bash", "-c", f'({LOOP}) & echo "$$ $!" > "$0"; exit 4', str(pids)]) == 4
    _gone(map(int, pids.read_text().split()))
    monkeypatch.setattr(CP, "GRACE_S", 0)
    real, started = subprocess.Popen, []

    def starting(*a, **kw):
        started.append(p := real(*a, **kw))
        os.kill(os.getpid(), signal.SIGTERM)
        return p
    was = signal.signal(signal.SIGTERM, CP.ended("CERTIFIER_TIMEOUT"))
    try:
        with pytest.raises(CP.NotRun, match="SIGTERM: the piece's time limit"):
            CP.run_group(["bash", "-c", f'(trap "" TERM; {LOOP}) & echo "$$ $!" > "$0"; kill -TERM "$PPID"; {LOOP}',
                          str(pids)])
        _gone(map(int, pids.read_text().split()))
        monkeypatch.setattr(CP.subprocess, "Popen", starting)
        with pytest.raises(CP.NotRun, match="SIGTERM: the piece's time limit"):
            CP.run_group(["bash", "-c", LOOP])
    finally:
        signal.signal(signal.SIGTERM, was)
    _gone([started[0].pid])
    assert CP._held is None


@pytest.mark.parametrize("case", ["no kept run", "corrupt fetch", "no report", "no build", "no rows build",
                                  "time limit", "foreign answer", "no record", "re-signed fetch", "another session",
                                  "one copy rewritten"])
def test_a_step_that_cant_run_refuses_and_leaves_the_certifier_missing(case, tmp_path, warden, capsys, monkeypatch):
    """The front's failures (the gate refusing or busy, its tunnel or ssh) keep no run; a corrupt fetch, a chain that
    ends without a report, a missing Lean build (``network-period``'s or ``network-rows``'), SIGTERM from the piece's
    time limit and a check that answers no certifier/v0 verdict can't run; nor can a piece given no auditor's record of
    the run's deployment (no ``CERTIFIER_KEY``), or whose kept fetch isn't the record's: re-signed under another key,
    of another session, or with one copy of its deployment.json rewritten. Each leaves DIR/refused.txt and its log, no
    answer, and the certifier missing."""
    fetch = {"re-signed fetch": "resigned", "one copy rewritten": "rewritten"}.get(case, "fetch")
    d = _serve_dir(tmp_path / "serve", warden, kept=case != "no kept run", fetch=fetch)
    if case == "corrupt fetch":
        (d / CP.PRIVATE / RUN / "fetch.tar.gz").write_bytes(b"not a tarball")
    if case == "no record":
        monkeypatch.delenv(CP.CERTIFIER_KEY)
    if case == "another session":
        rec = _auditors(warden)
        dep = {**rec["deployment"], "session": "5e" * (len(rec["session"]) // 2)}
        rec |= {"session": dep["session"], "deployment": dep,
                "deployment_sha512": hashlib.sha512(json.dumps(dep).encode()).hexdigest()}
        (other := tmp_path / "certifier-key.json").write_text(json.dumps(rec))
        monkeypatch.setenv(CP.CERTIFIER_KEY, str(other))
    chain = _chain_out(tmp_path / "chain", "accepted", warden["w"])
    pids = tmp_path / "pids"
    cmd = {"no report": "exit 7", "time limit": f'({LOOP}) & echo "$$ $!" > {shlex.quote(str(pids))}; '
                                                f'kill -TERM "$PPID"; {LOOP}'}.get(case)
    lean = tmp_path / "no-such-network-period" if case == "no build" else None
    rows = tmp_path / "no-such-network-rows" if case == "no rows build" else None
    answer = {"verdict": "accept", "protocol": "other"} if case == "foreign answer" else {}
    env = dict(os.environ, CERTIFIER_CMD=_piece_cmd(tmp_path, chain, answer, cmd, lean, rows))
    S.pieces(d, K, tmp_path / "pieces", env=env)
    if case == "time limit":
        _gone(map(int, pids.read_text().split()))
    piece = tmp_path / "pieces" / "certifier"
    why = (piece / CP.REFUSED).read_text()
    assert {p.name for p in piece.iterdir()} - {"piece.json"} == {CP.REFUSED, "piece.log"}, why
    assert S.piece_row("communication", tmp_path / "pieces")["verdict"] == "missing"
    assert {"no kept run": "no warden run kept", "corrupt fetch": "fetch.tar.gz: ReadError", "no report": "the chain wrote no report (exit 7)",
            "no build": "network-period check: FileNotFoundError",
            "no rows build": "network-rows check: FileNotFoundError", "time limit": "SIGTERM: the piece's time limit",
            "foreign answer": "no verity/certifier/v0 verdict",
            "no record": "no auditor's record of the network certifier's deployment (--certifier-key, CERTIFIER_KEY)",
            "re-signed fetch": "witness/deployment.json names another key than the auditor's record's",
            "another session": "witness/deployment.json names another session than the auditor's record's",
            "one copy rewritten": "out/deployment.json names another key than the auditor's record's"}[case] in why
    if case != "no kept run":
        assert not list((d / CP.PRIVATE / RUN).glob("unpacked-*"))
    rec, line = _record(monkeypatch, capsys, d, tmp_path / "pieces")
    assert rec is not None and line["composed"] == "pending" and "the network certifier" in line["missing"]


def _piece(tmp_path: Path, warden: dict, chain: Path) -> tuple[int, Path, Path]:
    """The piece run in this process on a kept run and `chain`'s outputs, the period check a stub that accepts: its
    exit, DIR and the serve dir."""
    d = _serve_dir(tmp_path / "serve", warden)
    lean = tmp_path / "network-period"
    lean.write_text(f"#!/bin/sh\ncat <<'EOF'\n{json.dumps(CP.answer('reject', 'stub'))}\nEOF\n")
    lean.chmod(0o755)
    cmd = f'mkdir -p "$RESEARCH_RUN_DIR" && cp -r {shlex.quote(str(chain / "out"))} "$RESEARCH_RUN_DIR/"'
    piece = tmp_path / "pieces" / "certifier"
    rc = CP.main(["piece", str(d), str(K), str(piece), "--certifier-key", str(warden["key"]), "--lean", str(lean),
                  "--rows", str(_stub_rows(tmp_path)), "--chain-cmd", cmd])
    return rc, piece, d


def test_sigterm_while_the_guard_reads_releases_nothing_to_dir(tmp_path, warden, monkeypatch):
    """SIGTERM from the piece's time limit reaching it while the leak guard reads (the signal sent to this process from
    a stand-in guard) refuses the piece with nothing released: DIR holds only refused.txt, and the answer, the chain's
    report and the period's files stay in the hold, out of the record."""
    def slow_guard(d, hold, unpacked, quarantine):
        os.kill(os.getpid(), signal.SIGTERM)
        pytest.fail("SIGTERM didn't end the piece")
    monkeypatch.setattr(CP, "guard", slow_guard)
    rc, piece, d = _piece(tmp_path, warden, _chain_out(tmp_path / "chain", "accepted", warden["w"]))
    assert rc == 3 and "SIGTERM: the piece's time limit" in (piece / CP.REFUSED).read_text()
    assert [p.name for p in piece.iterdir()] == [CP.REFUSED]
    (hold,) = (d / CP.PRIVATE / "held").iterdir()
    assert (hold / "answer.json").is_file() and (hold / "chain" / "report" / "report.json").is_file()


def test_a_chain_file_holding_a_witness_value_is_moved_out_of_dir(tmp_path, warden):
    """The chain's report carrying a piece of the warden run's witness (a salt of its cells.salts, as hex) in a key
    `PUBLIC` keeps: the guard moves DIR/chain/report/report.json to the quarantine and releases the rest, the answer
    with it. The chain's job.json carrying it is never published, so it reaches neither DIR nor the quarantine."""
    with tarfile.open(warden["fetch"]) as tar:
        m = next(m for m in tar if m.name.endswith("/witness/cells.salts"))
        salt = tar.extractfile(m).read(32)[16:]
    chain = _chain_out(tmp_path / "chain", "accepted", warden["w"])
    (chain / "out" / "job.json").write_text(json.dumps({"x": salt.hex()}))
    (chain / "out" / "report" / "report.json").write_text(json.dumps({"verdict": "accepted", "stubbed": [salt.hex()]}))
    rc, piece, d = _piece(tmp_path, warden, chain)
    assert rc == 1 and (piece / "answer.json").is_file() and (piece / "chain" / "scan" / "answers.json").is_file()
    assert not (piece / "chain" / "report" / "report.json").exists() and not (piece / "chain" / "job.json").exists()
    assert (d / CP.PRIVATE / "quarantine" / "files" / "chain" / "report" / "report.json").is_file()
    assert not (d / CP.PRIVATE / "quarantine" / "files" / "chain" / "job.json").exists()
    assert list((d / CP.PRIVATE / "held").iterdir()) == []


def test_a_rehearsal_the_gates_check_failed_late_leaves_the_certifier_missing_not_rejected(tmp_path, warden, capsys,
                                                                                          monkeypatch):
    """Each request late past the network certifier's D·τ: the gate's check fails the rehearsal, and the served answers, longer
    than what the warden passed, don't bind to its certified ones. That is the endpoint's lateness, not the protocol
    rejecting a served answer, so the piece refuses before any chain and the certifier is missing, as for a busy gate;
    the same window on a rehearsal the check passed is a reject."""
    served = [dict(r, output=[*r["output"], 7]) for r in warden["records"]]
    d = _serve_dir(tmp_path / "serve", warden, served, fetch="late")
    chain = _chain_out(tmp_path / "chain", "accepted", warden["w"])
    S.pieces(d, K, tmp_path / "pieces", env=dict(os.environ, CERTIFIER_CMD=_piece_cmd(tmp_path, chain, {})))
    piece = tmp_path / "pieces" / "certifier"
    why = (piece / CP.REFUSED).read_text()
    assert why.startswith("the gate's rehearsal didn't pass, so it certified no served answer: exit 1, the rehearsal "
                          "exited 1; its check accepted False with 8 problems"), why
    assert {p.name for p in piece.iterdir()} - {"piece.json"} == {CP.REFUSED, "piece.log"}
    assert not (d / CP.PRIVATE / RUN / "chain").exists() and not list((d / CP.PRIVATE / RUN).glob("unpacked-*"))
    assert S.piece_row("communication", tmp_path / "pieces")["verdict"] == "missing"
    rec, line = _record(monkeypatch, capsys, d, tmp_path / "pieces")
    assert rec is not None and line["composed"] == "pending" and "the network certifier" in line["missing"]
    a, _ = CP.decide(_serve_dir(tmp_path / "passed", warden, served), K, tmp_path / "out", record=_auditors(warden),
                     chain=lambda *x: pytest.fail("no chain"), check=lambda *x: {}, rows=lambda run: None)
    assert a["verdict"] == "reject" and "streamed_ids aren't the certified output" in a["reason"]


def test_the_case_is_written_only_from_period_files_the_leak_guard_left(tmp_path, warden):
    """`composite.py` reads a case's commitment and evidence as hex, where the leak guard finds only a binary witness
    file's pieces: so the case is written from DIR/period after the guard has read its files raw in the hold and
    released them, and not at all when the guard moved one of them (here the evidence, holding a witness file's text)."""
    fetch = tmp_path / "fetch.tar.gz"
    fetch.write_bytes(warden["fetch"].read_bytes())
    unpacked, run = CP.unpack(fetch)
    for name, evidence in (("clean", b'{"windows":[]}'), ("leaky", (run / "witness" / "certified.jsonl").read_bytes())):
        d, hold = tmp_path / name, tmp_path / f"{name}-held"
        d.mkdir()
        (hold / "period").mkdir(parents=True)
        for f, b in (("params.json", b'{"k":1}'), ("commitment.bin", b"\1\2"), ("evidence.json", evidence)):
            (hold / "period" / f).write_bytes(b)
        g = CP.guard(d, hold, unpacked, tmp_path / f"{name}-quarantine")
        CP.write_case(d)
        if name == "clean":
            assert g == 0 and not (d / "period").exists()
            assert (d / CP.CASE).read_bytes() == b'{"commitment":"0102","evidence":"%s","params":{"k":1}}\n' % (
                evidence.hex().encode())
        else:
            assert g == 3 and not (d / CP.CASE).exists()
            assert sorted(p.name for p in (d / "period").iterdir()) == ["commitment.bin", "params.json"]
            assert (tmp_path / "leaky-quarantine" / "files" / "period" / "evidence.json").is_file()


def test_rows_check_takes_only_network_rows_verdict_and_leaves_no_case_behind(tmp_path, warden):
    """`rows_check` on the kept run, its case read off the run's records under the auditor's record: ``network-rows``'
    accept passes and its reject is `Rejected`; an answer of another protocol, a check that exits past 2, a run whose
    case can't be read and a run re-signed under another key than the record's are `NotRun`; and the case's scratch
    directory beside the run is gone however it ends."""
    record = _auditors(warden)
    fetch = tmp_path / "fetch.tar.gz"
    fetch.write_bytes(warden["fetch"].read_bytes())
    unpacked, run = CP.unpack(fetch)
    assert CP.rows_check(str(_stub_rows(tmp_path)), run, record) is None
    reject = {"protocol": CP.ROWS, "verdict": "reject", "reason": "egress bucket 2: not r cells a connection in "
              "ordinal order"}
    with pytest.raises(CP.Rejected, match=r"^the network certifier's records, audited as rows: egress bucket N: not r "
                                          r"cells a connection in ordinal order$"):
        CP.rows_check(str(_stub_rows(tmp_path, reject)), run, record)
    with pytest.raises(CP.NotRun, match="network-rows answered no verity/network-rows/v1 accept or reject"):
        CP.rows_check(str(_stub_rows(tmp_path, {"protocol": CP.PROTOCOL, "verdict": "accept"})), run, record)
    failing = tmp_path / "failing" / "network-rows"
    failing.parent.mkdir()
    failing.write_text("#!/bin/sh\necho oops >&2\nexit 3\n")
    failing.chmod(0o755)
    with pytest.raises(CP.NotRun, match="network-rows check: SystemExit: network-rows check exited 3: oops"):
        CP.rows_check(str(failing), run, record)
    (WC.witness(run) / "certified.jsonl").unlink()
    with pytest.raises(CP.NotRun, match="the network certifier's rows case: SystemExit: .*no certified.jsonl"):
        CP.rows_check(str(_stub_rows(tmp_path)), run, record)
    (resigned := tmp_path / "resigned.tar.gz").write_bytes(warden["resigned"].read_bytes())
    _, other = CP.unpack(resigned)
    with pytest.raises(CP.NotRun, match="the network certifier's rows case: SystemExit: .*witness/deployment.json "
                                        "names another key than the auditor's record's"):
        CP.rows_check(str(_stub_rows(tmp_path)), other, record)
    assert not list(unpacked.glob("rows-*"))


@pytest.fixture(scope="module")
def rows_exe() -> str:
    """``network-rows``, built where the piece builds it."""
    if shutil.which("lake") is None:
        pytest.skip("no Lean toolchain")
    return CP.lean_exe("network-rows", None)


#: the synthetic run's cells a connection a bucket, and a cell's bytes
R, B = sz.W63.params.slots, sz.W63.params.cell_bytes


def _responses(edit):
    """`published`'s ``change``: window 0's responses record with its grid (a list of rows of slots) edited in place
    by ``edit``, every other record as served."""
    def change(rec):
        if (rec.link, rec.window) != (WS.EGRESS, 0):
            return rec
        grid = [list(row) for row in rec.grid]
        edit(grid)
        return replace(rec, grid=tuple(map(tuple, grid)))
    return change


def _recut(grid) -> None:
    """Connection 1's second `DATA` cell made an `END` with its bytes."""
    _, o, payload = W.read_cell(grid[27][1], B)
    grid[27][1] = W.cell(W.END, o, payload, B)


def _reclock(grid) -> None:
    """Connection 1's lane, its one bucket past its delay (admitted at 0, D = 27), sent a bucket early, in its delay."""
    grid[26][:R], grid[27][:R] = grid[27][:R], grid[26][:R]


def _reorder(grid) -> None:
    """Bucket 2's first two connections swapped."""
    grid[2][:R], grid[2][R:2 * R] = grid[2][R:2 * R], grid[2][:R]


@pytest.mark.slow
def test_network_rows_accepts_the_network_certifiers_records_and_refuses_cells_cut_clocked_or_ordered_otherwise(
        tmp_path, rows_exe):
    """The compiled Lean audit (``network-rows``: `NetworkCheck.RowsAudit` over `NetworkCertifier.Rows.audit`) on the
    records the network certifier's own publisher writes and signs, under the auditor's record of its deployment: the
    synthetic run's accepted; the same run with a cell of window 0's egress re-cut, connection 1's cells re-clocked or
    a bucket's connections reordered, signed by its key, refused by name, and `rows_check`'s `Rejected` for the piece;
    the run re-signed under another key refused at the record, and, its deployment.json rewritten to the record's, by
    the audit at its first certificate, which isn't the record's key's; and a record changed after its certificate
    signed it refused at its head. The one test that runs the case's encoding, the binding to the signed heads and the
    audit together on real records, which the stand-ins above don't."""
    honest = published(tmp_path / "honest")
    record = WC.own_record(honest)
    (rec,) = [r for r in WS.read_records(honest, sz.W63.params)[WS.EGRESS] if r.window == 0]
    assert [W.read_cell(c, B)[:2] for c in rec.grid[27][:R]] == [(W.DATA, 1)] * (R - 1) + [(W.END, 1)]
    assert [W.read_cell(c, B)[:2] for c in rec.grid[26][:R]] == [(W.PAD, 1)] * R
    assert [W.read_cell(c, B)[1] for c in rec.grid[2] if c is not None] == [1] * R + [2] * R + [3] * R
    got = SP.lean_check(rows_exe, WC.rows_audit_case(honest, record), tmp_path / "case-honest")
    assert (got["protocol"], got["verdict"], got["reason"]) == (CP.ROWS, "accept", None), got
    assert got["profile"] == {"link": WS.LINK, "windows": sz.W63.windows} and got["public"] == {}
    assert CP.rows_check(rows_exe, honest, record) is None
    for name, edit, why in (("re-cut", _recut, "egress bucket 27 isn't the schedule's"),
                            ("re-clocked", _reclock, "egress bucket 26 isn't the schedule's"),
                            ("reordered", _reorder, "egress bucket 2: not r cells a connection in ordinal order")):
        run = published(tmp_path / name, change=_responses(edit))
        got = SP.lean_check(rows_exe, WC.rows_audit_case(run, record), tmp_path / f"case-{name}")
        assert (got["verdict"], got["reason"]) == ("reject", why), (name, got)
        with pytest.raises(CP.Rejected, match=re.escape(re.sub(r"[0-9]+", "N", why))):
            CP.rows_check(rows_exe, run, record)
    resigned = published(tmp_path / "resigned", seed=b"another network certifier")
    with pytest.raises(SystemExit, match="deployment.json names another key than the auditor's record's$"):
        WC.rows_audit_case(resigned, record)
    shutil.copyfile(honest / "deployment.json", resigned / "deployment.json")
    got = SP.lean_check(rows_exe, WC.rows_audit_case(resigned, record), tmp_path / "case-resigned")
    why = "requests window 0: the certificate isn't the key's in the session"
    assert (got["verdict"], got["reason"]) == ("reject", why), got
    with pytest.raises(CP.Rejected, match=re.escape(re.sub(r"[0-9]+", "N", why))):
        CP.rows_check(rows_exe, resigned, record)
    shutil.copyfile(tmp_path / "re-cut" / "responses.records", honest / "responses.records")
    got = SP.lean_check(rows_exe, WC.rows_audit_case(honest, record), tmp_path / "case-changed")
    assert (got["verdict"], got["reason"]) == ("reject", "responses window 0: the certified head isn't the record's")
    assert not list(tmp_path.glob("rows-*"))


def test_the_placement_names_the_kept_gate_a_local_gate_the_prover_node_and_refuses_without_the_session(tmp_path,
                                                                                                      warden):
    d = _serve_dir(tmp_path / "serve", warden)
    assert CP.placement(d, RUN)["parties"]["network_certifier"] == GATE_HOST
    (d / CP.PRIVATE / RUN / CP.SESSION).write_text(json.dumps({"run": RUN, "gate": "local"}))
    p = CP.placement(d, RUN)
    assert (p["kind"], p["links"]) == ("prover-node", []) and p[CP.PREMISE].startswith("unmet")
    assert p["parties"]["network_certifier"] == p["parties"]["prover"] == socket.gethostname()
    (d / CP.PRIVATE / RUN / CP.SESSION).unlink()
    with pytest.raises(CP.NotRun, match="keep the run with this tree's front"):
        CP.placement(d, RUN)


def _numbers(p: Path) -> set[int]:
    return {int(x) for x in re.findall(rb"(?<![0-9A-Za-z])[0-9]+(?![0-9A-Za-z])", p.read_bytes())}


def _private_chain_out(out: Path) -> Path:
    """What the chain's network certifier path leaves in out/ on a served window: each step's file carrying the window's private
    values where the chain writes them, and the report and report.md that ``chain_record.py report`` makes of them."""
    docs = {
        "custody.json": {"verdict": "accepted", "gate_runs": [{"gate_run": RUN, "strategies": {}}]},
        "certs.json": {"warden_run": RUN, "verdict": "accepted", "windows": 1, "sync_bits": 0.0,
                       "rows": [{"window": 0, "certified": True}],
                       "warden_check": {"accepted": True, "summary": {"requests": REQUESTS, "outputs": OUTPUTS}}},
        "n2/summary.json": {"warden_run": RUN, "ok": True, "link": {"unit": "n2"}, "members": {"honest": {
            "network": warden_network(), "windows": [{"window": 0, "accepted": True, "as_expected": True}]}}},
        "scan/summary.json": {"warden_run": RUN, "T": 30, "ok": True, "responses": {"0": OUTPUTS}, "rounds": {
            "honest": {"ok": True, "audit": {"accepted": True, "window_accepted": True}},
            "chain": {"ok": True, "audit": {"accepted": False, "window_accepted": False}}}},
        "scan/answers.json": {"format": "answers", "delta": "1/100", "registration": {"program": "ab" * 64},
                              "windows": [{"window": 0, "answer": {"verdict": "accept"}, "registrations": []}]},
        "job.json": {"path": CR.WARDEN, "run": "r-chain", "rc": {"n2": "0", "scan": "0"}},
    }
    for rel, doc in docs.items():
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        (out / rel).write_text(json.dumps(doc, indent=1))
    CR.main(["report", *(x for f, rel in (("custody", "custody.json"), ("certs", "certs.json"), ("n2", "n2/summary.json"),
                                         ("scan", "scan/summary.json"), ("job", "job.json"))
             for x in (f"--{f}", str(out / rel))), "--warden-run", RUN, "--out", str(out / "report")])
    return out


def test_the_piece_publishes_no_count_of_the_window_and_the_guard_moves_out_what_isnt_declared(tmp_path):
    """The chain's out/ holds the window's private values: the certificates step's view of the warden's check (the
    certified I/O's counts), the report made of it, and the scan's response lengths; it opens no endpoint records, so
    it has no endpoint step and d1's K no advice term (`chain_record.WARDEN_K`). `publish` puts none of the counts in
    the hold, only `PUBLIC`'s key paths. A file put past `publish`, in the hold (report.md copied whole) or in DIR (the
    certificates step's), though neither holds a witness piece, is one the leak guard moves to the quarantine and never
    releases: DIR declares only `PIECE_FILES`."""
    out = _private_chain_out(tmp_path / "out")
    held = {p.relative_to(out).as_posix(): _numbers(p) & PRIVATE for p in out.rglob("*") if p.is_file()}
    assert held["certs.json"] == held["report/report.json"] == PRIVATE and held["scan/summary.json"] == {OUTPUTS}
    full = json.loads((out / "report" / "report.json").read_text())
    assert full["verdict"] == "accepted" and full["d1"]["K"]["open"] == CR.WARDEN_K
    hold, d, q = tmp_path / "hold", tmp_path / "pieces" / "certifier", tmp_path / "quarantine"
    CP.publish(out, hold)
    published = {p.relative_to(hold).as_posix(): p for p in hold.rglob("*") if p.is_file()}
    assert sorted(published) == sorted(f"chain/{rel}" for rel in CP.PUBLIC)
    assert {rel: _numbers(p) & PRIVATE for rel, p in published.items()} == dict.fromkeys(published, set())
    assert KS.scan(hold)[0] == []
    report = json.loads(published["chain/report/report.json"].read_text())
    assert report["verdict"] == "accepted" and set(report["steps"]) == set(KS.STEPS)
    assert report["steps"]["scan"]["eps_at_delta"]["units"] == 60
    assert report["stubbed"] == list(CR.WARDEN_STUBBED)
    assert json.loads(published["chain/scan/answers.json"].read_text()) == json.loads(
        (out / "scan" / "answers.json").read_text())
    (hold / "chain" / "report" / "report.md").write_bytes((out / "report" / "report.md").read_bytes())
    (hold / "answer.json").write_text(json.dumps(CP.answer("accept", "")))
    (d / "chain").mkdir(parents=True)
    (d / "chain" / "certs.json").write_bytes((out / "certs.json").read_bytes())
    (d / "piece.log").write_text("leak guard\n")
    assert CP.guard(d, hold, None, q) == 3
    assert all((q / "files" / "chain" / rel).is_file() for rel in ("report/report.md", "certs.json"))
    assert sorted(p.relative_to(d).as_posix() for p in d.rglob("*") if p.is_file()) == sorted(
        ["answer.json", "piece.log", *published])
    assert not set().union(*(_numbers(p) & PRIVATE for p in d.rglob("*") if p.is_file()))
    assert not any(p.is_file() for p in hold.rglob("*"))


def _v3c(serve: Path, period: dict | None = None, committed: bool = True, circuit: bool = True) -> Path:
    """V3c's directory beside the window (`served_verdicts.sh`'s ``$D/v3c-w<K>-<class>``): its committed window (only
    its registration, which the chain step looks for), given, its period.json, and beside it the verifier's own
    window-circuit record at its class (``$D/window-circuit-w<K>-<class>.json``, `CIRCUIT`)."""
    v3c = serve / "v3c-w3-8-32-4-4-16"
    (v3c / "window").mkdir(parents=True)
    if committed:
        (v3c / "window" / "registration.json").write_text("{}")
    if period is not None:
        (v3c / "period.json").write_text(json.dumps(period))
    if circuit:
        (serve / "window-circuit-w3-8-32-4-4-16.json").write_text(json.dumps(CIRCUIT))
    return v3c


def _hook(step: str, d: Path, dir_: Path, v3c: Path, *args: str) -> int:
    """The split piece's ``step`` as `served_verdicts.sh`'s ``certifier_step`` runs it: ``bash -c CMD _ D K DIR V3C`` in
    the tree, CMD the module docstring's (``--certifier-key "${CERTIFIER_KEY:-}"``), its output DIR/<step>.log."""
    cmd = f"{shlex.quote(sys.executable)} {shlex.quote(str(HERE / 'certifier_piece.py'))} {step} \"$@\" " + \
        '--certifier-key "${CERTIFIER_KEY:-}" ' + " ".join(map(shlex.quote, args))
    dir_.mkdir(parents=True, exist_ok=True)
    with open(dir_ / f"{step}.log", "w") as log:
        return subprocess.run(["bash", "-c", cmd, "_", str(d), str(K), str(dir_), str(v3c)], cwd=ROOT, stdout=log,
                              stderr=subprocess.STDOUT, check=False).returncode


def _chained(tmp_path: Path, v3c: Path | None, verdict: str = "accepted", w: int = 0) -> list[str]:
    """``chain``'s arguments: a chain that leaves `_chain_out`'s report, its scan's summary naming V3c's window (``v3c``;
    None: no V3c window) and `PERIOD`'s registration as the one whose trees it read, and notes what it was given in
    TMP/chain-args."""
    out = _chain_out(tmp_path / "chain", verdict, w)
    (out / "out" / "scan" / "summary.json").write_text(json.dumps({"v3c": v3c and {
        "window": str(v3c / "window"), "registration": RG.digest(PERIOD["registration"])}}))
    note = shlex.quote(str(tmp_path / "chain-args"))
    return ["--rows", str(_stub_rows(tmp_path)),
            "--chain-cmd", f'printf "%s\\n" "$@" > {note}; mkdir -p "$RESEARCH_RUN_DIR" && '
                           f'cp -r {shlex.quote(str(out / "out"))} "$RESEARCH_RUN_DIR/"']


def _lean(tmp_path: Path, answer: dict) -> list[str]:
    return ["--lean", str(_stub_lean(tmp_path, answer)), "--rows", str(_stub_rows(tmp_path))]


#: V3c's period.json as `records_audit.py period` writes it (its values stand-ins of the kinds `kind_scan` declares: the
#: period check here is a stub)
PERIOD = {"answer": {"verdict": "accept"}, "free": 5, "program": {"v3c": 1}, "registration": {"program": "cd" * 64},
          "steps": 0}
#: the verifier's own window-circuit record at V3c's class, the fields `scan_period.circuit_of_record` reads
CIRCUIT = {"format": "verity/window-circuit/v1", "program": "c3" * 64}


def test_the_split_piece_chains_on_v3cs_window_and_checks_its_period_json(tmp_path, warden, period, capsys,
                                                                            monkeypatch):
    """The chain step runs the chain with its scan on V3c's window (``--class w63k4 --v3c V3C/window`` first in
    CR_SCAN_ARGS) and marks DIR/chain.json; the period step reads that chain's report again, without running it, and
    checks the rows case whose one compute audit is V3c's period.json, held to the class circuit of the verifier's own
    window-circuit record beside V3c, its answer the one `pieces` reads (with CERTIFIER_CMD ``true``, as
    `served_verdicts.sh` sets it) into the block record, its case written and both steps' logs kept."""
    accept = period.reference(period.case("accept", 1, "clock"))
    d = _serve_dir(tmp_path / "serve", warden)
    v3c, piece = _v3c(d), tmp_path / "pieces" / "certifier"
    assert _hook("chain", d, piece, v3c, *_chained(tmp_path, v3c, w=warden["w"]), "CR_SCAN_ARGS=--x 1") == 0
    assert (piece / "chain.log").read_text().splitlines()[-1] == (
        f"CERTIFIER-CHAIN accepted on V3c's window {v3c / 'window'}: the period step checks its period.json")
    assert json.loads((piece / CP.CHAINED).read_text()) == {"chain": "accepted"}
    assert f"CR_SCAN_ARGS=--class w63k4 --v3c {v3c / 'window'} --x 1" in (tmp_path / "chain-args").read_text()
    assert not (piece / "answer.json").exists()
    (v3c / "period.json").write_text(json.dumps(PERIOD))
    assert _hook("period", d, piece, v3c, *_lean(tmp_path, accept)) == 0, (piece / "period.log").read_text()
    params = json.loads((tmp_path / "seen" / "params.json").read_text())
    assert params["audits"] == [{k: PERIOD[k] for k in ("program", "steps")}]
    assert params["circuit"] == CIRCUIT["program"]
    assert json.loads(bytes.fromhex(json.loads((piece / CP.CASE).read_text())["commitment"]))["audits"] == [
        PERIOD["registration"]]
    got = S.pieces(d, K, tmp_path / "pieces", env=dict(os.environ, CERTIFIER_CMD="true"))
    assert (got["certifier"]["verdict"], got["certifier"]["rc"]) == ("accept", 0)
    assert {p.name for p in piece.iterdir()} - {"piece.json"} == {"answer.json", CP.CASE, CP.CHAINED, "chain",
                                                                  "chain.log", "period.log", "piece.log"}
    rec, line = _record(monkeypatch, capsys, d, tmp_path / "pieces")
    assert BR.check(rec) == {} and line["verdicts"]["the network certifier"] == "accept"


def test_a_chain_the_split_rejects_stands_and_its_period_step_checks_nothing(tmp_path, warden):
    d = _serve_dir(tmp_path / "serve", warden)
    v3c, piece = _v3c(d, PERIOD), tmp_path / "pieces" / "certifier"
    assert _hook("chain", d, piece, v3c, *_chained(tmp_path, v3c, "refused", warden["w"])) == 1
    a = json.loads((piece / "answer.json").read_text())
    assert a["verdict"] == "reject" and a["reason"].startswith("the chain refused") and not (piece / CP.CHAINED).exists()
    assert _hook("period", d, piece, v3c, *_lean(tmp_path, {})) == 1
    assert json.loads((piece / "answer.json").read_text()) == a and not (tmp_path / "seen").exists()


@pytest.mark.parametrize("case", ["no v3c window", "time limit", "no chain step", "a scan of no v3c", "no period.json",
                                  "no window-circuit record", "no record", "re-signed fetch"])
def test_a_split_step_that_cant_run_refuses_keeping_both_logs(case, tmp_path, warden, monkeypatch):
    """The chain step refuses without V3c's committed window, which its scan reads, when its limit (CERTIFIER_TIMEOUT)
    ends it, without the auditor's record of the run's deployment (``CERTIFIER_KEY``) and when the kept fetch was
    re-signed under another key than the record's; the period step then refuses too, naming the chain's refusal. The
    period step refuses where no chain step accepted in DIR, where the chain's scan read no V3c window, where V3c wrote
    no period.json, and where the verifier kept no window-circuit record beside V3c, whose class circuit V3c's audit is
    held to: DIR keeps its logs and refused.txt, and the certifier is missing."""
    d = _serve_dir(tmp_path / "serve", warden, fetch="resigned" if case == "re-signed fetch" else "fetch")
    v3c, piece = _v3c(d, None if case == "no period.json" else PERIOD, committed=case != "no v3c window",
                      circuit=case != "no window-circuit record"), tmp_path / "pieces" / "certifier"
    args = _chained(tmp_path, None if case == "a scan of no v3c" else v3c, w=warden["w"])
    if case == "time limit":
        args = ["--rows", str(_stub_rows(tmp_path)), "--chain-cmd", 'kill -TERM "$PPID"; while :; do :; done']
    if case == "no record":
        monkeypatch.delenv(CP.CERTIFIER_KEY)
    if case != "no chain step":
        _hook("chain", d, piece, v3c, *args)
    assert _hook("period", d, piece, v3c, *_lean(tmp_path, {})) == 3
    why = (piece / CP.REFUSED).read_text()
    assert {"no v3c window": f"V3c committed no window at {v3c / 'window'}",
            "time limit": "SIGTERM: the piece's time limit (CERTIFIER_TIMEOUT)",
            "no chain step": "no chain.json in DIR: the chain step (CERTIFIER_CHAIN_CMD) didn't run",
            "a scan of no v3c": f"the chain's scan read no V3c window, not {v3c / 'window'}",
            "no period.json": "the period's rows case: FileNotFoundError",
            "no window-circuit record": "the verifier's window-circuit record: FileNotFoundError",
            "no record": "no auditor's record of the network certifier's deployment (--certifier-key, CERTIFIER_KEY)",
            "re-signed fetch": "witness/deployment.json names another key than the auditor's record's"}[case] in why, why
    logs = {"chain.log", "period.log"} - ({"chain.log"} if case == "no chain step" else set())
    assert {p.name for p in piece.iterdir()} == {CP.REFUSED, *logs}
    said = (piece / "period.log").read_text()
    at_chain = ("no v3c window", "time limit", "no record", "re-signed fetch")
    assert ("CERTIFIER refused at the chain: " if case in at_chain else "CERTIFIER refused: ") in said
    assert S.piece_row("communication", tmp_path / "pieces")["verdict"] == "missing"
    assert not list((d / CP.PRIVATE / RUN).glob("unpacked-*"))


def test_each_steps_time_limit_names_its_own():
    for limit in ("CERTIFIER_TIMEOUT", "CERTIFIER_PERIOD_TIMEOUT"):
        with pytest.raises(CP.NotRun, match=rf"^SIGTERM: the piece's time limit \({limit}\) ended it"):
            CP.ended(limit)(signal.SIGTERM, None)


def test_n2_takes_every_link_window_of_a_short_run_unless_named():
    assert CP.sized(1, []) == ["CR_N2_DRAW="]
    assert CP.sized(8, ["CR_N2_DRAW=4"]) == ["CR_N2_DRAW=4"]
    assert CP.sized(9, ["CR_SCAN_ONLY=1"]) == ["CR_SCAN_ONLY=1"]


def _stub_gate(tmp_path: Path, monkeypatch) -> None:
    """`test_served_session.STUB` as ssh: the gate it reaches writes its request to tmp_path/request."""
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin" / "ssh").write_text(STUB)
    (tmp_path / "bin" / "ssh").chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path / 'bin'}{os.pathsep}{os.defpath}")
    monkeypatch.setenv("STUB_DIR", str(tmp_path))


def test_the_front_sends_one_group_of_the_class_and_keeps_the_fetch_in_the_serve_dirs_private(tmp_path, monkeypatch):
    _stub_gate(tmp_path, monkeypatch)
    serve, out = tmp_path / "serve", tmp_path / "out" / "certifier-front"
    assert CP.main(["front", "18190", str(serve), str(out), "--key", str(tmp_path / "key")]) == 0
    assert (tmp_path / "request").read_text() == (
        "warden-rehearse --peer vy-dc-h200 --port 18190 --minutes 7 --requests 4 --height 10 --scan --prompts "
        "--tokens 4 --scan-class w63k4 --window-per-group")
    prompts = json.loads((tmp_path / "prompts.json").read_text())
    assert [len(p) for p in prompts] == [CP.PROMPT_IDS] * CP.PROMPTS and all(t < 128000 for p in prompts for t in p)
    kept = serve / CP.PRIVATE / RUN / "fetch.tar.gz"
    assert [p.stat().st_mode & 0o777 for p in (serve / "private", serve / CP.PRIVATE, kept)] == [0o700, 0o700, 0o600]
    assert sorted(p.name for p in out.iterdir()) == ["bundle.tgz", "leak-guard.json", "session.json", "status.json"]
    assert (kept.parent / "start.json").stat().st_mode & 0o777 == 0o600
    assert json.loads((out / "session.json").read_text())["keep"] == str(kept)
    assert (kept.parent / CP.SESSION).read_text() == (out / "session.json").read_text()
    assert json.loads((out / "session.json").read_text())["gate"] == CP.SS.GATE
    assert CP.main(["front", "18190", str(serve), str(out), "--tokens", "64"]) == 2
    assert "past one link-window" in json.loads((out / "session.json").read_text())["refused"]
    assert CP.main(["front", "18190", str(serve), str(out), "--tokens", "5"]) == 2
    assert "1 to KO = 4 ids" in json.loads((out / "session.json").read_text())["refused"]
    assert CP.main(["front", "18190", str(tmp_path / "serve-w63"), str(tmp_path / "out-w63"), "--key",
                    str(tmp_path / "key"), "--scan-class", "w63"]) == 0
    assert (tmp_path / "request").read_text().endswith("--tokens 4 --scan-class w63 --window-per-group")


def test_the_front_sends_a_span_of_prompt_lengths(tmp_path, monkeypatch):
    """``--prompt-ids 1:32``: the group's prompts are `served_session.prompts`' at the span, 1 to 32 ids below the
    vocabulary, and the gate's request is unchanged. The seed is pinned."""
    _stub_gate(tmp_path, monkeypatch)
    monkeypatch.setattr(CP.secrets, "token_bytes", lambda n: b"s" * n)
    serve, out = tmp_path / "serve", tmp_path / "out"
    assert CP.main(["front", "18190", str(serve), str(out), "--key", str(tmp_path / "key"), "--prompt-ids", "1:32"]) == 0
    ps = json.loads((tmp_path / "prompts.json").read_text())
    assert ps == CP.SS.prompts(CP.PROMPTS, (1, 32), 128000, b"s" * 32) and len({len(p) for p in ps}) > 1
    assert (tmp_path / "request").read_text().endswith("--tokens 4 --scan-class w63k4 --window-per-group")
    assert json.loads((out / "session.json").read_text())["prompts"] == {"count": CP.PROMPTS, "ids": sum(map(len, ps))}


class _Ids(socketserver.StreamRequestHandler):
    """serve-openai's ``--protocol ids`` as the relay reads it: the request read whole, then `IDS`' id events at the
    scan's payload and the trailer, and the connection closed."""

    def handle(self):
        head = b""
        while not head.endswith(b"\r\n\r\n"):
            head += self.rfile.readline()
        self.rfile.read(int(re.search(rb"(?i)content-length:\s*(\d+)", head).group(1)))
        self.wfile.write(AT.WS.id_response(IDS, SCAN_PAYLOAD))


IDS = [5, 6]
SCAN_PAYLOAD = CP.SS.A.SCAN["--cell-bytes"] - AT.IT.HEADER.size


def test_an_attack_front_puts_the_relay_between_the_gate_and_the_endpoint_and_ends_it(tmp_path, monkeypatch):
    """``--attack token``: the gate's request names the relay's port, a request through it (the session's stand-in)
    reaches the endpoint on PORT and comes back with its first id one other, its line in the prover's 0700 attacks log,
    and the relay is gone once the front ends."""
    _stub_gate(tmp_path, monkeypatch)
    endpoint = socketserver.ThreadingTCPServer(("127.0.0.1", 0), _Ids)
    threading.Thread(target=endpoint.serve_forever, daemon=True).start()
    relays, seen, real, real_relay = [], [], CP.SS.session, CP.attack_relay

    def attack_relay(a, keep):
        relays.append(real_relay(a, keep))
        return relays[-1]

    def session(a, data, ps):
        request = AT.id_request(model="m", user="u", prompt=[1, 2], max_tokens=len(IDS),
                                frame_bytes=CP.SS.A.SCAN["--cell-bytes"], offset=0)
        with socket.create_connection(("127.0.0.1", a.port)) as s:
            s.sendall(request)
            got = b""
            while chunk := s.recv(1 << 16):
                got += chunk
        seen.append(got)
        return real(a, data, ps)

    monkeypatch.setattr(CP, "attack_relay", attack_relay)
    monkeypatch.setattr(CP.SS, "session", session)
    serve, out = tmp_path / "serve", tmp_path / "out"
    try:
        assert CP.main(["front", str(endpoint.server_address[1]), str(serve), str(out), "--key", str(tmp_path / "key"),
                        "--attack", "token"]) == 0
    finally:
        endpoint.shutdown()
        endpoint.server_close()
    ((p, port),) = relays
    assert p.poll() is not None
    assert f" --port {port} " in (tmp_path / "request").read_text()
    assert seen == [AT.WS.id_response([IDS[0] ^ 1, *IDS[1:]], SCAN_PAYLOAD)]
    log = serve / CP.PRIVATE / "attacks"
    assert log.stat().st_mode & 0o777 == 0o700
    (line,) = [json.loads(x) for x in (log / "token.jsonl").read_text().splitlines()]
    assert (line["connection"], line["attack"], line["events"]) == (1, "token", len(IDS))


def test_a_fronts_rehearsal_at_4000_ms_reaches_the_gate_with_its_tick_and_the_scans_schedule_there(tmp_path,
                                                                                                   monkeypatch):
    """``--gate local``, this tree's gate in this process: the run's request records the tick, the minutes counted at
    it and greedy sampling (temperature 0, top_p 1, which the network certifier's client puts on every request, as
    ``--stop-prompt``'s warm-ups are sent), and `warden_command` is the scan's schedule at it, the network certifier's
    parameters `gate_params` gives (link-windows of 32 buckets, a window of the window circuit's class 8,32,4,4,16). The
    tunnel and the rehearsal are stand-ins, the rehearsal's check accepting."""
    real, ran, check = CP.SS.A.certifier_run, [], json.dumps({"accepted": True, "problems": []})

    def command(req, work, url):
        ran.append(CP.SS.A.warden_command(req, work, url))
        return [sys.executable, "-c", "import sys; open(sys.argv[1], 'w').write(sys.argv[2])", str(work / "check.json"),
                check]

    def certifier_run(*args, **kw):
        ran.append(kw["tunnel_argv"]([], 1))
        return real(*args, **kw | {"tunnel_argv": lambda ssh, port: [sys.executable, "-c", ""]}, command=command,
                    up=lambda tunnel, port: True)

    monkeypatch.setattr(CP.SS.A, "certifier_run", certifier_run)
    monkeypatch.setattr(CP.SS, "pinned", lambda: "unpinned")
    serve, out = tmp_path / "serve", tmp_path / "out"
    assert CP.main(["front", "18190", str(serve), str(out), "--gate", "local", "--bucket-ms", "4000"]) == 0
    minutes = CP.SS.A.warden_group_minutes(1, CP.TOKENS, CP.HEIGHT, 4000) + CP.SS.MARGIN
    assert minutes > CP.SS.A.warden_group_minutes(1, CP.TOKENS, CP.HEIGHT) + CP.SS.MARGIN
    (req,) = [json.loads(p.read_text()) for p in (serve / CP.PRIVATE).rglob("request.json")]
    assert {k: req.get(k) for k in ("bucket_ms", "scan", "prompts", "window_per_group", "minutes", "temperature",
                                    "top_p")} == {
        "bucket_ms": 4000, "scan": True, "prompts": {"count": CP.PROMPTS, "ids": CP.PROMPTS * CP.PROMPT_IDS},
        "window_per_group": True, "minutes": minutes, "temperature": 0.0, "top_p": 1.0}
    relay, warden = ran
    assert relay[-5:] == ["relay", "--listen", "1", "--to", "18190"]
    at = warden.index("--temperature")
    assert warden[at:at + 4] == ["--temperature", "0.0", "--top-p", "1.0"]
    at = warden.index("--ids")
    assert warden[at:at + 1 + len(CP.SS.A.SCAN_SCHEDULE)] == ["--ids", *CP.SS.A.scan_schedule(4000, CP.V3C_SCAN_CLASS)]
    assert CP.SS.A.scan_schedule(4000) != CP.SS.A.SCAN_SCHEDULE
    params = WS._params(WS.parser().parse_args(warden[2:]))
    assert params == CP.gate_params(CP.V3C_SCAN_CLASS, 4000)
    assert (params.bucket_ms, params.window, params.lanes, params.every, params.egress_cells) == (4000, 32, 4, 16, 5)
    doc = json.loads((out / "session.json").read_text())
    assert doc["request"] == (f"warden-rehearse --peer vy-dc-h200 --port 18190 --minutes {minutes} --requests 4 "
                              "--height 10 --scan --bucket-ms 4000 --prompts --tokens 4 --scan-class w63k4 "
                              "--window-per-group")
    assert (doc["rc"], doc["verdict"]["accepted"]) == (0, True)


@pytest.mark.parametrize("tick, scan, why", [
    ("99", True, "--bucket-ms '99': 100 to 4000"),
    ("4001", True, "--bucket-ms '4001': 100 to 4000"),
    ("4000", False, "--bucket-ms: the scan's tick, with --scan"),
])
def test_a_tick_the_gate_refuses_refuses_the_front_before_the_gate_is_asked(tmp_path, monkeypatch, tick, scan, why):
    """The front's check is this tree's gate's: a tick past the gate's range, or one on a request that isn't the scan
    rehearsal (the front's always is: here its scan flags are dropped), is refused with the gate's reason."""
    _stub_gate(tmp_path, monkeypatch)
    if not scan:
        words = CP.SS.words
        monkeypatch.setattr(CP.SS, "words", lambda a: [w for w in words(a)
                                                       if w not in ("--scan", "--prompts", "--window-per-group")])
    serve, out = tmp_path / "serve", tmp_path / "out"
    assert CP.main(["front", "18190", str(serve), str(out), "--key", str(tmp_path / "key"), "--bucket-ms", tick]) == 2
    assert json.loads((out / "session.json").read_text())["refused"] == f"the gate refuses it: {why}"
    assert not (tmp_path / "request").exists()


def _serve_windows(d: Path, windows: list[list[str]]) -> Path:
    """A serve dir as serve-openai leaves it: window k sealed and passed, serving one request a user of ``windows[k]``."""
    d.mkdir(parents=True)
    reqs = [{"engine_id": f"cmpl-{k}{u}-0", "user": u, "windows": [k], "check": {"passed": True}}
            for k, us in enumerate(windows) for u in us]
    _write(d / "requests.jsonl", reqs)
    _write(d / "seals.jsonl", [{"window": k, "requests": [r["engine_id"] for r in reqs if r["windows"] == [k]]}
                               for k in range(len(windows))])
    _write(d / "windows.jsonl", [{"window": k, "passed": True, "commit": {"steps": 4}} for k in range(len(windows))])
    return d


def test_the_pick_passes_over_the_warm_ups_windows_for_the_groups(tmp_path):
    """Each warm-up is a window of one request, before the group's: the pick (the most requests, then the earliest)
    takes the group's window, or its largest part when the engine split it, and a smaller window after it (the load's)
    loses too; only a group split down to single requests ties the warm-ups, whose earlier window then wins."""
    warm, group, load = [["warmup-t-0"], ["warmup-t-1"]], [f"user-{j}" for j in range(4)], ["varied.00000"]
    for name, layout, want in (("whole", [*warm, group, load], 2), ("3+1", [*warm, group[:3], group[3:], load], 2),
                               ("2+2", [*warm, group[:2], group[2:], load], 2),
                               ("1+1+1+1", [*warm, *([u] for u in group), load], 0)):
        got = S.pick(_serve_windows(tmp_path / name, layout), 4)
        assert (got["window"], got["users"]) == (want, sorted(layout[want])), name


class _Endpoint(http.server.BaseHTTPRequestHandler):
    """serve-openai's completions as the warm-up sees them: each answered whole, finishing on a stop for the requests
    numbered in ``stops`` and on the length otherwise, and recorded in the serve dir's request_users.jsonl first
    (serve-openai records a request there once its execution phase has closed) unless ``record`` is off; each request
    noted with whether the gate had been asked by then."""

    def do_POST(self):
        s = self.server
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        s.seen.append({"path": self.path, "body": body, "headers": {k.lower() for k in self.headers},
                       "gate asked": (s.stub_dir / "request").exists()})
        if s.status != 200:
            self.send_response(s.status)
            self.end_headers()
            return
        if s.record:
            with open(s.serve / "request_users.jsonl", "a") as fh:
                fh.write(json.dumps({"engine_id": f"cmpl-{len(s.seen)}-0", "user": body["user"]}) + "\n")
        finish = "stop" if len(s.seen) in s.stops else "length"
        out = json.dumps({"id": f"cmpl-{len(s.seen)}", "choices": [{"text": "", "finish_reason": finish}]}).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *args):
        pass


@pytest.fixture
def endpoint(tmp_path, monkeypatch):
    _stub_gate(tmp_path, monkeypatch)
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Endpoint)
    srv.seen, srv.status, srv.record, srv.serve, srv.stub_dir, srv.stops = [], 200, True, tmp_path / "serve", tmp_path, ()
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv
    srv.shutdown()
    srv.server_close()


def test_the_front_warms_the_endpoint_up_a_window_at_a_time_before_it_asks_the_gate(tmp_path, endpoint):
    port, serve, out = endpoint.server_address[1], endpoint.serve, tmp_path / "out"
    assert CP.main(["front", str(port), str(serve), str(out), "--key", str(tmp_path / "key"), "--warmup", "2"]) == 0
    assert [(s["path"], s["gate asked"]) for s in endpoint.seen] == [("/v1/completions", False)] * 2
    assert not any(h.startswith("x-verity-release") for s in endpoint.seen for h in s["headers"])
    bodies = [s["body"] for s in endpoint.seen]
    assert [(len(b["prompt"]), b["max_tokens"], b["temperature"]) for b in bodies] == [(CP.PROMPT_IDS, CP.TOKENS, 0.0)] * 2
    assert CP.PROMPT_IDS <= 32 and all(0 <= t < 128000 for b in bodies for t in b["prompt"])
    users = [b["user"] for b in bodies]
    assert len(set(users)) == 2 and all(u.startswith("warmup-") for u in users)
    assert (tmp_path / "request").read_text() == (
        f"warden-rehearse --peer vy-dc-h200 --port {port} --minutes 7 --requests 4 --height 10 --scan --prompts "
        "--tokens 4 --scan-class w63k4 --window-per-group")
    warm = json.loads((out / "warmup.json").read_text())["requests"]
    assert [(w["user"], w["finish_reason"]) for w in warm] == [(u, "length") for u in users]
    assert sorted(p.name for p in out.iterdir()) == ["bundle.tgz", "leak-guard.json", "session.json", "status.json",
                                                     "warmup.json"]


def test_a_stop_prompt_is_the_first_tried_whose_answer_stops_and_goes_last_in_the_group(tmp_path, endpoint):
    """``--stop-prompt N``: after the ``--warmup`` requests, fresh prompts of HI ids go to the endpoint as warm-ups do
    (greedy, ``--tokens`` max_tokens), one at a time, until one's answer finishes on a stop, and that prompt is the
    group's last; the gate's request is unchanged. None of N stopping refuses the front before the gate is asked."""
    port, serve, out = endpoint.server_address[1], endpoint.serve, tmp_path / "out"
    endpoint.stops = {4}
    argv = ["front", str(port), str(serve), str(out), "--key", str(tmp_path / "key"), "--prompt-ids", "8:32",
            "--warmup", "1", "--stop-prompt", "5"]
    assert CP.main(argv) == 0
    bodies = [s["body"] for s in endpoint.seen]
    assert [len(b["prompt"]) for b in bodies[1:]] == [32] * 3 and not any(s["gate asked"] for s in endpoint.seen)
    assert {(b["max_tokens"], b["temperature"], b["top_p"]) for b in bodies} == {(CP.TOKENS, 0.0, 1.0)}
    *rest, last = json.loads((tmp_path / "prompts.json").read_text())
    assert last == bodies[3]["prompt"] and len(rest) == CP.PROMPTS - 1 and all(8 <= len(p) <= 32 for p in rest)
    warm = json.loads((out / "warmup.json").read_text())
    assert [w["finish_reason"] for w in warm["requests"]] == ["length"]
    assert [(w["user"], w["finish_reason"]) for w in warm["stop_prompt"]] == [
        (b["user"], f) for b, f in zip(bodies[1:], ("length", "length", "stop"))]
    assert (tmp_path / "request").read_text().endswith("--tokens 4 --scan-class w63k4 --window-per-group")
    endpoint.seen.clear()
    endpoint.stops = ()
    (tmp_path / "request").unlink()
    assert CP.main([*argv[:-4], "--stop-prompt", "2"]) == 2
    assert json.loads((out / "session.json").read_text())["refused"] == (
        "--stop-prompt 2: no answer to 2 prompts of 32 ids finished on a stop within --tokens 4")
    assert len(endpoint.seen) == 2 and not (tmp_path / "request").exists()
    assert CP.main([*argv[:-1], "-1"]) == 2
    assert json.loads((out / "session.json").read_text())["refused"] == "--stop-prompt -1: a count of requests, 0 for none"


def test_a_warm_up_that_fails_or_isnt_recorded_closed_refuses_the_front_before_the_gate(tmp_path, endpoint):
    port, serve, out = endpoint.server_address[1], endpoint.serve, tmp_path / "out"
    endpoint.status = 503
    assert CP.main(["front", str(port), str(serve), str(out), "--warmup", "1"]) == 2
    assert json.loads((out / "session.json").read_text())["refused"].startswith("--warmup: request 0: HTTPError")
    assert not (tmp_path / "request").exists() and not (out / "warmup.json").exists()
    endpoint.status, endpoint.record, clock = 200, False, _Clock()
    a = argparse.Namespace(port=port, warmup=1, prompt_ids=CP.PROMPT_IDS, vocab=128000, tokens=CP.TOKENS)
    with pytest.raises(CP.SS.Refused, match="recorded no closed execution phase of .'warmup-"):
        CP.warmup(a, serve, clock=clock, sleep=clock.sleep)
    assert clock.t > CP.WARMUP_CLOSE_S and not (tmp_path / "request").exists()


class _Clock:
    t = 0.0

    def __call__(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.t += s
