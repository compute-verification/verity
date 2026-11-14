"""certifier_piece: the front sends one group of the window circuit's class through the gate's scan rehearsal and keeps
the gate's fetch, witness and all, 0700 under the serve dir's private/certifier, publishing only the bundle less witness/
and its outcome. The piece, on a serve dir whose window was the network certifier's synthetic run's certified requests
(the run `test_warden_chain.gate_run` publishes in the gate's layout), runs as `served_verdicts.py pieces` runs it and writes
`network-period`'s answer only when the chain accepted and the window's requests are certified ones in the scan's
link-window: the period mirror's certifier/v0 accept (`period_case`), its profile with the placement (the certifier on
the kept gate's host, the checks on the developer's node), then goes through `served_verdicts.py record` into the window
report and the block record's certifier leg, which `block_record.check` holds and its schema admits. Its case.json is
the three files `network-period` read, as `composite.py` reads a piece's case: witness, kept 0600 in the serve dir's
private/pieces/certifier with no answer beside it, and at no time under the pieces' directory, which gets
certificates.json, the five certificates of the case's evidence, each with its own fields alone and each the network
certifier's signature under the run's public key. The chain's endpoint report, with each Commit window's counts, and
the scan's answers stay in the kept run. A window request the warden didn't certify is
rejected before any chain runs; a chain that refused, requests outside the scan's link-window and a period the check
rejects are rejects; a rehearsal the gate's check failed (each request late) and a step that can't run (no kept run, no
report, no build, the time limit) refuse, which leaves the certifier missing, and no process of a step's group outlives
the piece, the time limit's SIGTERM included, whenever it comes. Of the chain, DIR gets only `PUBLIC`'s key paths,
which hold none of the window's counts or advice bits; the leak guard reads the piece's DIR against the run's witness
and moves out any file DIR doesn't declare, a period file among them; the certificates reach DIR only through it, and
a step that ends without its files released keeps no case. The front's
warm-ups go straight to the endpoint, one at a time, each recorded in a closed window of its own before the gate is
asked, and the pick passes over those windows for the group's. At a tick of 4000 ms the front's rehearsal reaches this
tree's gate with the tick in its request and the minutes counted at it, and the warden runs the scan's schedule there;
a tick the gate refuses (past its range, or on a request that isn't the scan rehearsal) refuses the front before the
gate is asked."""
from __future__ import annotations

import argparse
import http.server
import importlib.util
import json
import os
import re
import shlex
import signal
import socket
import subprocess
import sys
import tarfile
import threading
from pathlib import Path

import jsonschema
import pytest

HERE = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(HERE), str(Path(__file__).resolve().parent), str(ROOT / "benchmarks" / "one_stage")]

import certifier_piece as CP  # noqa: E402
import chain_record as CR  # noqa: E402
import scan_period as SP  # noqa: E402
import served_verdicts as S  # noqa: E402
from test_served_session import CHALLENGER_HOST, LIVE_GATE, RUN, STUB  # noqa: E402
from test_warden_chain import WC, gate_run  # noqa: E402
from verity.circuits.codec import canonical_json  # noqa: E402
from verity.protocols.accounting.certifier.verifier.certificate import Direction  # noqa: E402
from verity_e2e_demo import block_record as BR  # noqa: E402

PERIOD_CASE = ROOT / "verity" / "protocols" / "accounting" / "certifier" / "verifier" / "network" / "tests" / "period_case.py"
SCHEMA = json.loads((Path(BR.__file__).parent / "schemas" / "block-record.v2.schema.json").read_text())
K = 3
#: the gate's end of a rehearsal its check passed, and of one whose every request was late past the network certifier's
#: D·τ
PASSED = {"run": RUN, "rc": 0, "ended": "the rehearsal ended", "verdict": {"accepted": True, "problems": 0}}
LATE = {"run": RUN, "rc": 1, "ended": "the rehearsal exited 1",
        "verdict": {"accepted": False, "problems": 8, "requests": 4, "ended": 0, "failed": ["fail"]}}
GATE_HOST = "gate-host"
#: the window's private values, as numbers no other field of the chain's files takes: its request, step and output
#: counts, its committed, v0 and derived advice bits, and the tie's prompt lengths
REQUESTS, STEPS, OUTPUTS, COMMITTED, V0, DERIVED = 7919, 7927, 7933, 104723, 104729, 104743
PROMPT_LENS = [7937, 7949]
PRIVATE = {REQUESTS, STEPS, OUTPUTS, COMMITTED, V0, DERIVED, *PROMPT_LENS}


@pytest.fixture(scope="module")
def period():
    spec = importlib.util.spec_from_file_location("period_case", PERIOD_CASE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["period_case"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def warden(tmp_path_factory):
    """The synthetic network certifier run, its fetch (``late``: the same run, its rehearsal failed late), the
    execution phase holding the most whole certified requests, that link-window's two row registrations (ingress,
    egress) and the run's deployment."""
    d = tmp_path_factory.mktemp("network_certifier")
    run = gate_run(d)
    io, T = WC.io(run), WC.deployment(run)[1].window
    by: dict[int, list[dict]] = {}
    for r in io["records"]:
        ws = {b // T for b in [r["release"], *(b for _, b in r["frames"])]}
        if len(ws) == 1:
            by.setdefault(ws.pop(), []).append(r)
    w = max(by, key=lambda x: (len(by[x]), -x))
    fetches = {}
    for name, end in (("late", LATE), ("fetch", PASSED)):
        (run / "end.json").write_text(json.dumps(end))
        fetches[name] = d / f"{name}.tar.gz"
        with tarfile.open(fetches[name], "w:gz") as tar:
            tar.add(run, arcname=RUN)
    cells = [json.loads(x)["registration"] for x in (run / "out" / "cells.jsonl").read_text().splitlines()]
    registrations = [r for d_ in (Direction.INGRESS, Direction.EGRESS) for r in cells
                     if (r["link"], r["direction"], r["window"]) == (WC.WS.LINK, d_.value, w)]
    return {**fetches, "w": w, "records": by[w][:4], "others": [r for x, rs in by.items() if x != w for r in rs],
            "windows": WC.deployment(run)[0]["windows"], "registrations": registrations,
            "deployment": json.loads((run / "out" / "deployment.json").read_text())}


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _files(d: Path) -> set[str]:
    return {p.relative_to(d).as_posix() for p in d.rglob("*") if p.is_file()}


def _named(d: Path, *names: str) -> list[Path]:
    """Every path under ``d`` named one of ``names``: a case, a period file or the scan's answers, where none may be."""
    return sorted(p for p in d.rglob("*") if p.name in names)


def _serve_dir(d: Path, warden: dict, records: list[dict] | None = None, kept: bool = True,
               fetch: str = "fetch") -> Path:
    """A serve dir whose window K served ``records`` (the network certifier's, by default those of its execution phase)
    and whose gate fetch (``warden[fetch]``) the front kept."""
    records = warden["records"] if records is None else records
    d.mkdir(parents=True)
    reqs = [{"engine_id": f"e{i}", "user": r["user"], "prompt_ids": r["prompt"], "streamed_ids": r["output"],
             "windows": [K], "check": {"passed": True}} for i, r in enumerate(records)]
    _write(d / "requests.jsonl", reqs)
    _write(d / "seals.jsonl", [{"window": K, "requests": [r["engine_id"] for r in reqs]}])
    _write(d / "windows.jsonl", [{"window": K, "passed": True, "commit": {"steps": 9}}])
    _write(d / "advice_trace.jsonl", [{"step": 0}])
    (d / "serve.json").write_text("{}")
    if kept:
        (d / CP.PRIVATE / RUN).mkdir(parents=True)
        (d / CP.PRIVATE / RUN / "fetch.tar.gz").write_bytes(warden[fetch].read_bytes())
        (d / CP.PRIVATE / RUN / CP.SESSION).write_text(json.dumps({"run": RUN, "gate": f"research@{GATE_HOST}"}))
    return d


def _chain_out(d: Path, verdict: str, w: int, registrations: tuple | list = ()) -> Path:
    """What the chain leaves in its run dir's out/: its report, the endpoint's and the scan's one link-window, with
    the row registrations ``registrations``."""
    out = d / "out"
    (out / "report").mkdir(parents=True)
    (out / "endpoint").mkdir(parents=True)
    (out / "endpoint" / "report.json").write_text(json.dumps({"chain": {"bits": {"commit_windows": [
        {"window": K, "requests": 3, "carried": 0, "outputs": 12, "t0_s": 0.5}]}}}))
    (out / "scan" / "joint").mkdir(parents=True)
    (out / "report" / "report.json").write_text(json.dumps({"verdict": verdict, "why": None if verdict == "accepted"
                                                            else {"n2": "a window refused"}}))
    (out / "scan" / "answers.json").write_text(json.dumps({"registration": {"program": "ab" * 64}, "windows": [
        {"window": w, "answer": {"verdict": "accept"}, "registrations": list(registrations)}]}))
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


def _piece_cmd(tmp_path: Path, chain: Path, answer: dict, cmd: str | None = None, lean: Path | None = None) -> str:
    """The piece's command, its `network-period` `_stub_lean`'s answering ``answer``."""
    lean = lean or _stub_lean(tmp_path, answer)
    cmd = cmd or f'mkdir -p "$RESEARCH_RUN_DIR" && cp -r {shlex.quote(str(chain / "out"))} "$RESEARCH_RUN_DIR/"'
    piece = ROOT / "benchmarks/network_traces/certifier_piece.py"
    return (f"{shlex.quote(sys.executable)} {shlex.quote(str(piece))} piece \"$@\" "
            f"--lean {shlex.quote(str(lean))} --chain-cmd {shlex.quote(cmd)}")


def _integrity() -> dict:
    a = {"protocol": "verity/sampled-proofs/one-stage/v1", "verdict": "accept", "reason": None, "commitment": "ab" * 64,
         "public": {"bound": 2, "drawn": [0, 3], "per_template": None, "proved": [0, 3]},
         "profile": {"levels": [{"name": "unit", "size": 4}], "drawn": {"unit": 2}, "delta": "1/100"}}
    return S.row_of(S.INTEGRITY_OWNER, a, Path("/nonexistent"))


def _record(monkeypatch, capsys, serve: Path, pieces: Path) -> tuple[dict | None, dict]:
    """`served_verdicts.py record` over the pieces `pieces` ran, sampled proofs' row an accept: the block record (None
    where none was written) and the ``RECORD`` line."""
    monkeypatch.setattr(S, "spot_row", lambda d, run=None: _integrity())
    monkeypatch.setenv("RESEARCH_RUN_ID", "r1")
    out = pieces.parent / "record"
    capsys.readouterr()
    S.main(["record", "--serve-run", str(serve), "--window", str(K), "--sampled-proofs", str(out / "sampled-proofs"),
            "--pieces", str(pieces), "--out", str(out)])
    (line,) = [x.removeprefix("RECORD ") for x in capsys.readouterr().out.splitlines() if x.startswith("RECORD ")]
    rec = out / "block-record.json"
    return (json.loads(rec.read_text()) if rec.exists() else None), json.loads(line)


def test_the_pieces_certifier_row_is_the_period_checks_accept_and_composes_into_the_block_record(tmp_path, warden,
                                                                                                period, capsys,
                                                                                                monkeypatch):
    accept = period.reference(period.case("accept", 1, "clock"))
    assert accept["verdict"] == "accept" and accept["protocol"] == CP.PROTOCOL
    d = _serve_dir(tmp_path / "serve", warden)
    chain = _chain_out(tmp_path / "chain", "accepted", warden["w"], warden["registrations"])
    env = dict(os.environ, CERTIFIER_CMD=_piece_cmd(tmp_path, chain, accept))
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
    assert _files(piece) - {"piece.json"} == {"answer.json", CP.CERTIFICATES, "chain/report/report.json", "piece.log"}
    assert not _named(tmp_path / "pieces", CP.CASE, "period", "answers.json", "commitment.bin", "evidence.json",
                      "params.json")
    seen = {f.name: f.read_bytes() for f in (tmp_path / "seen").iterdir()}
    kept = d / CP.CASES
    assert [x.stat().st_mode & 0o777 for x in (kept.parent, kept, kept / CP.CASE)] == [0o700, 0o700, 0o600]
    assert [x.name for x in kept.iterdir()] == [CP.CASE] and _named(d, CP.CASE) == [kept / CP.CASE]
    raw = (kept / CP.CASE).read_bytes()
    case = json.loads(raw)
    assert sorted(case) == ["commitment", "evidence", "params"] and raw == canonical_json(case) + b"\n"
    assert (bytes.fromhex(case["commitment"]), bytes.fromhex(case["evidence"]), canonical_json(case["params"])) == (
        seen["commitment.bin"], seen["evidence.json"], seen["params.json"])
    signed = json.loads((piece / CP.CERTIFICATES).read_text())
    (window,) = json.loads(seen["evidence.json"])["windows"]
    assert sorted(signed) == ["records", "rows"] and signed["records"] == window["certificates"]
    assert (len(signed["records"]), len(signed["rows"]), len(window["registrations"])) == (3, 2, 2)
    assert all(sorted(c) == sorted(SP.CERTIFICATE_KEYS) for c in signed["records"] + signed["rows"])
    assert signed["rows"] == [{"link": r["link"], "direction": r["direction"], "window": r["window"],
                               "key_id": r["certificate"]["key"], "digest": r["certificate"]["digest"],
                               "signature": r["certificate"]["signature"]} for r in window["registrations"]]
    key, session = WC.cs.read_public(warden["deployment"]["key"]), bytes.fromhex(warden["deployment"]["session"])
    assert all(WC.cs.certificate_from_json(c).verify(key, session) for c in signed["records"] + signed["rows"])
    assert {(c["direction"], c["window"]) for c in signed["rows"]} == {(Direction.INGRESS.value, warden["w"]),
                                                                       (Direction.EGRESS.value, warden["w"])}
    log = (piece / "piece.log").read_text()
    assert "leak guard (exit 0)" in log
    assert (f"case: the serve dir's {CP.CASES / CP.CASE}, {len(raw)} bytes; {CP.CERTIFICATES}: 3 records' and 2 rows' "
            "certificates") in log
    keep = d / CP.PRIVATE / RUN
    assert sorted(p.name for p in keep.iterdir()) == ["chain", "endpoint", "fetch.tar.gz", CP.SESSION]
    assert (keep / "chain" / "out" / "endpoint" / "report.json").is_file()
    assert [(keep / "endpoint" / f).stat().st_mode & 0o777 for f in CP.ENDPOINT] == [0o600] * 4
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
    a, _ = CP.decide(d, K, tmp_path / "out", chain=lambda *x: pytest.fail("no chain"), check=lambda *x: {})
    assert a["verdict"] == "reject" and "streamed_ids aren't the certified output" in a["reason"]


def _decide(tmp_path, warden, verdict="accepted", w=None, records=None, check=None):
    d = _serve_dir(tmp_path / "serve", warden, records)

    def chain(keep, run, ep, windows):
        assert sorted(p.name for p in ep.iterdir()) == sorted(CP.ENDPOINT) and windows == warden["windows"]
        _chain_out(keep / run / "chain", verdict, warden["w"] if w is None else w)
        return 0
    (tmp_path / "dir").mkdir()
    a, unpacked = CP.decide(d, K, tmp_path / "dir", chain=chain, check=check or (lambda c, p: pytest.fail("no check")),
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


@pytest.mark.parametrize("case", ["no kept run", "corrupt fetch", "no report", "no build", "time limit",
                                  "foreign answer"])
def test_a_step_that_cant_run_refuses_and_leaves_the_certifier_missing(case, tmp_path, warden, capsys, monkeypatch):
    """The front's failures (the gate refusing or busy, its tunnel or ssh) keep no run; a corrupt fetch, a chain that
    ends without a report, a missing Lean build, SIGTERM from the piece's time limit and a check that answers no
    certifier/v0 verdict can't run: each leaves DIR/refused.txt and its log, no answer, and the certifier missing."""
    d = _serve_dir(tmp_path / "serve", warden, kept=case != "no kept run")
    if case == "corrupt fetch":
        (d / CP.PRIVATE / RUN / "fetch.tar.gz").write_bytes(b"not a tarball")
    chain = _chain_out(tmp_path / "chain", "accepted", warden["w"])
    pids = tmp_path / "pids"
    cmd = {"no report": "exit 7", "time limit": f'({LOOP}) & echo "$$ $!" > {shlex.quote(str(pids))}; '
                                                f'kill -TERM "$PPID"; {LOOP}'}.get(case)
    lean = tmp_path / "no-such-network-period" if case == "no build" else None
    answer = {"verdict": "accept", "protocol": "other"} if case == "foreign answer" else {}
    env = dict(os.environ, CERTIFIER_CMD=_piece_cmd(tmp_path, chain, answer, cmd, lean))
    S.pieces(d, K, tmp_path / "pieces", env=env)
    if case == "time limit":
        _gone(map(int, pids.read_text().split()))
    piece = tmp_path / "pieces" / "certifier"
    why = (piece / CP.REFUSED).read_text()
    assert {p.name for p in piece.iterdir()} - {"piece.json"} == {CP.REFUSED, "piece.log"}, why
    assert S.piece_row("communication", tmp_path / "pieces")["verdict"] == "missing"
    assert {"no kept run": "no warden run kept", "corrupt fetch": "fetch.tar.gz: ReadError", "no report": "the chain wrote no report (exit 7)",
            "no build": "network-period check: FileNotFoundError", "time limit": "SIGTERM: the piece's time limit",
            "foreign answer": "no verity/certifier/v0 verdict"}[case] in why
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
    rc = CP.main(["piece", str(d), str(K), str(piece), "--lean", str(lean), "--chain-cmd", cmd])
    return rc, piece, d


def test_sigterm_while_the_guard_reads_releases_nothing_to_dir(tmp_path, warden, monkeypatch):
    """SIGTERM from the piece's time limit reaching it while the leak guard reads (the signal sent to this process from
    a stand-in guard) refuses the piece with nothing released: DIR holds only refused.txt, the answer, the chain's
    report and the certificates stay in the hold, out of the record, and the case the step had written is gone."""
    seen = []

    def slow_guard(d, hold, unpacked, quarantine):
        seen.append((_named(d.parent.parent, CP.CASE), _files(hold)))
        os.kill(os.getpid(), signal.SIGTERM)
        pytest.fail("SIGTERM didn't end the piece")
    monkeypatch.setattr(CP, "guard", slow_guard)
    rc, piece, d = _piece(tmp_path, warden, _chain_out(tmp_path / "chain", "accepted", warden["w"]))
    assert rc == 3 and "SIGTERM: the piece's time limit" in (piece / CP.REFUSED).read_text()
    assert [p.name for p in piece.iterdir()] == [CP.REFUSED]
    (hold,) = (d / CP.PRIVATE / "held").iterdir()
    assert _files(hold) == {"answer.json", CP.CERTIFICATES, "chain/report/report.json"}
    assert seen == [([d / CP.CASES / CP.CASE], _files(hold))] and not _named(tmp_path, CP.CASE)


def test_a_step_the_leak_guard_refuses_keeps_no_case_and_a_step_that_cant_start_drops_an_earlier_steps(tmp_path,
                                                                                                     warden,
                                                                                                     monkeypatch):
    """A case is its step's: the guard refusing the release (exit 2) leaves none, and a step with no kept run removes
    the one an earlier step on the serve dir left."""
    monkeypatch.setattr(CP, "guard", lambda d, hold, unpacked, quarantine: 2)
    rc, piece, d = _piece(tmp_path, warden, _chain_out(tmp_path / "chain", "accepted", warden["w"]))
    assert rc == 3 and "the leak guard failed (exit 2)" in (piece / CP.REFUSED).read_text()
    assert not _named(tmp_path, CP.CASE) and (d / CP.CASES).stat().st_mode & 0o777 == 0o700
    bare = _serve_dir(tmp_path / "bare", warden, kept=False)
    (bare / CP.CASES).mkdir(parents=True)
    (bare / CP.CASES / CP.CASE).write_text("{}")
    out = tmp_path / "bare-pieces" / "certifier"
    assert CP.main(["piece", str(bare), str(K), str(out), "--chain-cmd", "exit 7"]) == 3
    assert "no warden run kept" in (out / CP.REFUSED).read_text() and not _named(tmp_path, CP.CASE)


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
    assert rc == 1 and _files(piece) == {"answer.json", CP.CERTIFICATES}
    assert (d / CP.CASES / CP.CASE).is_file()
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
    a, _ = CP.decide(_serve_dir(tmp_path / "passed", warden, served), K, tmp_path / "out",
                     chain=lambda *x: pytest.fail("no chain"), check=lambda *x: {})
    assert a["verdict"] == "reject" and "streamed_ids aren't the certified output" in a["reason"]


def test_the_case_is_cut_from_the_hold_before_the_guard_and_only_its_certificates_are_released(tmp_path, warden):
    """A case's evidence and params are witness: `write_case` takes the hold's period files to the private case before
    the leak guard reads the hold, so the guard is never asked to release one, and what it releases of them is the
    certificates' own fields. A certificate holding a piece of the run's witness (here a salt of its cells.salts as
    its signature) is moved out as any released file is, a period file still in the hold is a file DIR doesn't
    declare, and evidence that names no certificates refuses the step with no case written."""
    fetch = tmp_path / "fetch.tar.gz"
    fetch.write_bytes(warden["fetch"].read_bytes())
    unpacked, run = CP.unpack(fetch)
    salt = (run / "witness" / "cells.salts").read_bytes()[16:48].hex()
    cert = {"digest": "ab" * 64, "direction": 0, "key_id": "cd" * 32, "link": "a-link", "signature": "ef" * 8,
            "window": 5}
    row = {"link": "a-link", "direction": 1, "window": 5, "root": "12" * 64, "wiring": "34" * 64,
           "certificate": {"key": "cd" * 32, "digest": "56" * 64, "signature": "78" * 8}}

    def held(name: str, evidence: bytes, files=("params.json", "commitment.bin", "evidence.json")) -> tuple[Path, Path]:
        hold = tmp_path / f"{name}-held"
        (hold / "period").mkdir(parents=True)
        for f, b in (("params.json", b'{"k":1}'), ("commitment.bin", b"\1\2"), ("evidence.json", evidence)):
            if f in files:
                (hold / "period" / f).write_bytes(b)
        return hold, tmp_path / name / "private" / "pieces" / "certifier" / CP.CASE

    for name, signature in (("clean", cert["signature"]), ("leaky", salt)):
        evidence = canonical_json({"audits": [{"an": "answer"}], "windows": [
            {"answer": {"verdict": "accept"}, "certificates": [cert | {"signature": signature, "head": "9a" * 64}],
             "registrations": [row]}]})
        hold, case = held(name, evidence)
        if name == "leaky":
            case.parent.mkdir(parents=True)
            case.parent.chmod(0o755)
        else:
            case.parent.parent.parent.mkdir(parents=True)
        CP.write_case(hold, case)
        assert case.read_bytes() == b'{"commitment":"0102","evidence":"%s","params":{"k":1}}\n' % (
            evidence.hex().encode())
        modes = [x.stat().st_mode & 0o777 for x in (case.parent.parent, case.parent, case)]
        assert modes[1:] == [0o700, 0o600] and (name == "leaky" or modes[0] == 0o700)
        assert _files(hold) == {CP.CERTIFICATES}
        want = {"records": [cert | {"signature": signature}],
                "rows": [{"digest": "56" * 64, "direction": 1, "key_id": "cd" * 32, "link": "a-link",
                          "signature": "78" * 8, "window": 5}]}
        assert json.loads((hold / CP.CERTIFICATES).read_text()) == want
        d = tmp_path / name / "out"
        d.mkdir()
        g = CP.guard(d, hold, unpacked, tmp_path / f"{name}-quarantine")
        if name == "clean":
            assert g == 0 and _files(d) == {CP.CERTIFICATES} and json.loads((d / CP.CERTIFICATES).read_text()) == want
        else:
            assert g == 3 and _files(d) == set()
            assert (tmp_path / "leaky-quarantine" / "files" / CP.CERTIFICATES).is_file()
    hold, case = held("undeclared", b'{"windows":[]}')
    d = tmp_path / "undeclared" / "out"
    d.mkdir(parents=True)
    assert CP.guard(d, hold, unpacked, tmp_path / "undeclared-quarantine") == 3 and _files(d) == set()
    assert _files(tmp_path / "undeclared-quarantine" / "files") == {"period/commitment.bin", "period/evidence.json",
                                                                    "period/params.json"}
    for name, evidence, files in (("no-windows", b'{"audits":[]}', ("params.json", "commitment.bin", "evidence.json")),
                                  ("no-evidence", b"", ("params.json", "commitment.bin"))):
        hold, case = held(name, evidence, files)
        if name == "no-windows":
            with pytest.raises(CP.NotRun, match="names no certificates: KeyError"):
                CP.write_case(hold, case)
        else:
            CP.write_case(hold, case)
        assert not case.exists() and not (hold / CP.CERTIFICATES).exists()
        assert _files(hold) == {f"period/{f}" for f in files}


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
    """What the chain's warden path leaves in out/ on a served window, the endpoint's report the served node's: each
    step's file carrying the window's private values where the chain writes them, and the report and report.md that
    ``chain_record.py report`` makes of them."""
    docs = {
        "custody.json": {"verdict": "accepted", "gate_runs": [{"gate_run": RUN, "strategies": {}}]},
        "certs.json": {"warden_run": RUN, "verdict": "accepted", "windows": 1, "sync_bits": 0.0,
                       "rows": [{"window": 0, "certified": True}],
                       "warden_check": {"accepted": True, "summary": {"requests": REQUESTS, "outputs": OUTPUTS}}},
        "endpoint/report.json": {
            "warden_run": RUN, "at": CR.SERVED_NODE, "verdict": "accepted",
            "io": {"accepted": True, "requests": REQUESTS, "tokens": OUTPUTS, "violations": []},
            "chain": {"bits": {"T": 1, "bound": True, "rebuilt": True, "v0_total": V0, "commit_windows": [
                {"window": K, "requests": REQUESTS, "carried": 0, "steps": STEPS, "outputs": OUTPUTS,
                 "committed_bits": COMMITTED, "v0_bits": V0, "derived_bits": DERIVED, "bound": True, "rebuilt": True}],
                "windows": [{"window": 0, "v0_bits": V0, "derived_bits": DERIVED, "in_run": True}]}}},
        "n2/summary.json": {"warden_run": RUN, "ok": True, "link": {"unit": "n2"}, "members": {"honest": {
            "network": {"accepted": True}, "windows": [{"window": 0, "accepted": True, "as_expected": True}]}}},
        "tie/summary.json": {"warden_run": RUN, "class": "warden", "ok": True, "prompt_lens": PROMPT_LENS,
                             "releases": [[0, 1]], "link": {"unit": "tie", "T": 30},
                             "tie": {"honest": [True], "forged": [False]}},
        "scan/summary.json": {"warden_run": RUN, "T": 30, "ok": True, "responses": {"0": OUTPUTS}, "rounds": {
            "honest": {"ok": True, "audit": {"accepted": True, "window_accepted": True}},
            "chain": {"ok": True, "audit": {"accepted": False, "window_accepted": False}}}},
        "scan/answers.json": {"format": "answers", "delta": "1/100", "registration": {"program": "ab" * 64},
                              "windows": [{"window": 0, "answer": {"verdict": "accept"}, "registrations": []}]},
        "job.json": {"path": CR.WARDEN, "run": "r-chain", "rc": {"n2": "0", "scan": "0", "tie": "0"}},
    }
    for rel, doc in docs.items():
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        (out / rel).write_text(json.dumps(doc, indent=1))
    CR.main(["report", *(x for f, rel in (("custody", "custody.json"), ("certs", "certs.json"), ("endpoint",
             "endpoint/report.json"), ("n2", "n2/summary.json"), ("scan", "scan/summary.json"), ("job", "job.json"),
             ("tie", "tie/summary.json")) for x in (f"--{f}", str(out / rel))), "--warden-run", RUN,
             "--out", str(out / "report")])
    return out


def test_the_piece_publishes_no_count_or_advice_bit_of_the_window_and_the_guard_moves_out_what_isnt_declared(tmp_path):
    """The chain's out/ holds the window's private values: the endpoint's report (its Commit window's request, step and
    output counts and its advice's bits), the report made of it (the certified I/O's counts, the advice's bits as K's
    term) and report.md, the certificates step's view of the warden's check and the tie's prompt lengths. `publish`
    puts none of them in the hold, only `PUBLIC`'s key paths, the endpoint's step its verdict and where it ran, and
    nothing of the scan's answers, the period check's evidence less its certificates. A file put past `publish`, in the
    hold (the endpoint's report and the scan's answers copied whole, as the piece once published them) or in DIR (the
    certificates step's), though none holds a witness piece, is one the leak guard moves to the quarantine and never
    releases: DIR declares only `PIECE_FILES`."""
    out = _private_chain_out(tmp_path / "out")
    held = {p.relative_to(out).as_posix(): _numbers(p) & PRIVATE for p in out.rglob("*") if p.is_file()}
    assert held["endpoint/report.json"] == PRIVATE - set(PROMPT_LENS) and held["tie/summary.json"] == set(PROMPT_LENS)
    assert held["report/report.json"] >= {REQUESTS, OUTPUTS, V0, DERIVED} and held["report/report.md"] >= {
        REQUESTS, OUTPUTS, DERIVED} and held["certs.json"] == {REQUESTS, OUTPUTS}
    assert json.loads((out / "report" / "report.json").read_text())["verdict"] == "accepted"
    hold, d, q = tmp_path / "hold", tmp_path / "pieces" / "certifier", tmp_path / "quarantine"
    CP.publish(out, hold)
    published = {p.relative_to(hold).as_posix(): p for p in hold.rglob("*") if p.is_file()}
    assert sorted(published) == sorted(f"chain/{rel}" for rel in CP.PUBLIC) == ["chain/report/report.json"]
    assert {rel: _numbers(p) & PRIVATE for rel, p in published.items()} == dict.fromkeys(published, set())
    report = json.loads(published["chain/report/report.json"].read_text())
    assert report["verdict"] == "accepted" and set(report["steps"]) == {
        "custody", "certificates", "endpoint", "tie", "n2", "scan", "lean"}
    assert report["steps"]["endpoint"] == {"verdict": "accepted", "in_the_clear": CR.CLEAR["warden_endpoint_served"]}
    assert report["steps"]["scan"]["eps_at_delta"]["units"] == 60
    assert report["stubbed"] == list(CR.WARDEN_SERVED_STUBBED)
    for rel in ("endpoint/report.json", "scan/answers.json"):
        (hold / "chain" / rel).parent.mkdir()
        (hold / "chain" / rel).write_bytes((out / rel).read_bytes())
    (hold / "answer.json").write_text(json.dumps(CP.answer("accept", "")))
    (d / "chain").mkdir(parents=True)
    (d / "chain" / "certs.json").write_bytes((out / "certs.json").read_bytes())
    (d / "piece.log").write_text("leak guard\n")
    assert CP.guard(d, hold, None, q) == 3
    assert _files(q / "files") == {"chain/endpoint/report.json", "chain/certs.json", "chain/scan/answers.json"}
    assert sorted(p.relative_to(d).as_posix() for p in d.rglob("*") if p.is_file()) == sorted(
        ["answer.json", "piece.log", *published])
    assert not set().union(*(_numbers(p) & PRIVATE for p in d.rglob("*") if p.is_file()))
    assert not any(p.is_file() for p in hold.rglob("*"))


def _v3c(serve: Path, period: dict | None = None, committed: bool = True) -> Path:
    """V3c's directory beside the window (`served_verdicts.sh`'s ``$D/v3c-w<K>-<class>``): its committed window (only
    its registration, which the chain step looks for) and, given, its period.json."""
    v3c = serve / "v3c-w3-8-32-4-4-16"
    (v3c / "window").mkdir(parents=True)
    if committed:
        (v3c / "window" / "registration.json").write_text("{}")
    if period is not None:
        (v3c / "period.json").write_text(json.dumps(period))
    return v3c


def _hook(step: str, d: Path, dir_: Path, v3c: Path, *args: str) -> int:
    """The split piece's ``step`` as `served_verdicts.sh`'s ``certifier_step`` runs it: ``bash -c CMD _ D K DIR V3C`` in
    the tree, its output DIR/<step>.log."""
    cmd = f"{shlex.quote(sys.executable)} {shlex.quote(str(HERE / 'certifier_piece.py'))} {step} \"$@\" " + \
        " ".join(map(shlex.quote, args))
    dir_.mkdir(parents=True, exist_ok=True)
    with open(dir_ / f"{step}.log", "w") as log:
        return subprocess.run(["bash", "-c", cmd, "_", str(d), str(K), str(dir_), str(v3c)], cwd=ROOT, stdout=log,
                              stderr=subprocess.STDOUT, check=False).returncode


def _chained(tmp_path: Path, v3c: Path | None, verdict: str = "accepted", w: int = 0) -> list[str]:
    """``chain``'s arguments: a chain that leaves `_chain_out`'s report, its scan's summary naming V3c's window (``v3c``;
    None: no V3c window), and notes what it was given in TMP/chain-args."""
    out = _chain_out(tmp_path / "chain", verdict, w)
    (out / "out" / "scan" / "summary.json").write_text(json.dumps({"v3c": v3c and {"window": str(v3c / "window")}}))
    note = shlex.quote(str(tmp_path / "chain-args"))
    return ["--chain-cmd", f'printf "%s\\n" "$@" > {note}; mkdir -p "$RESEARCH_RUN_DIR" && '
                           f'cp -r {shlex.quote(str(out / "out"))} "$RESEARCH_RUN_DIR/"']


def _lean(tmp_path: Path, answer: dict) -> list[str]:
    return ["--lean", str(_stub_lean(tmp_path, answer))]


#: V3c's period.json as `records_audit.py period` writes it (its values stand-ins: the period check here is a stub)
PERIOD = {"answer": {"verdict": "accept"}, "free": 5, "program": {"v3c": 1}, "registration": {"v3c": 2}, "steps": 0}


def test_the_split_piece_chains_on_v3cs_window_and_checks_its_period_json(tmp_path, warden, period, capsys,
                                                                            monkeypatch):
    """The chain step runs the chain with its scan on V3c's window (``--class w63k4 --v3c V3C/window`` first in
    CR_SCAN_ARGS) and marks DIR/chain.json; the period step reads that chain's report again, without running it, and
    checks the rows case whose one compute audit is V3c's period.json, its answer the one `pieces` reads (with
    CERTIFIER_CMD ``true``, as `served_verdicts.sh` sets it) into the block record, its case written in the serve
    dir's private (the chain step first removing the one an earlier run left), its certificates beside the answer and
    both steps' logs kept."""
    accept = period.reference(period.case("accept", 1, "clock"))
    d = _serve_dir(tmp_path / "serve", warden)
    v3c, piece = _v3c(d), tmp_path / "pieces" / "certifier"
    (d / CP.CASES).mkdir(parents=True)
    (d / CP.CASES / CP.CASE).write_text("an earlier run's")
    assert _hook("chain", d, piece, v3c, *_chained(tmp_path, v3c, w=warden["w"]), "CR_SCAN_ARGS=--x 1") == 0
    assert not _named(tmp_path, CP.CASE, CP.CERTIFICATES)
    assert (piece / "chain.log").read_text().splitlines()[-1] == (
        f"CERTIFIER-CHAIN accepted on V3c's window {v3c / 'window'}: the period step checks its period.json")
    assert json.loads((piece / CP.CHAINED).read_text()) == {"chain": "accepted", "v3c": str(v3c)}
    assert f"CR_SCAN_ARGS=--class w63k4 --v3c {v3c / 'window'} --x 1" in (tmp_path / "chain-args").read_text()
    assert not (piece / "answer.json").exists()
    (v3c / "period.json").write_text(json.dumps(PERIOD))
    assert _hook("period", d, piece, v3c, *_lean(tmp_path, accept)) == 0, (piece / "period.log").read_text()
    params = json.loads((tmp_path / "seen" / "params.json").read_text())
    assert params["audits"] == [{k: PERIOD[k] for k in ("free", "program", "steps")}]
    case = d / CP.CASES / CP.CASE
    assert json.loads(bytes.fromhex(json.loads(case.read_text())["commitment"]))["audits"] == [PERIOD["registration"]]
    assert _named(tmp_path, CP.CASE) == [case] and case.stat().st_mode & 0o777 == 0o600
    got = S.pieces(d, K, tmp_path / "pieces", env=dict(os.environ, CERTIFIER_CMD="true"))
    assert (got["certifier"]["verdict"], got["certifier"]["rc"]) == ("accept", 0)
    assert _files(piece) - {"piece.json"} == {"answer.json", CP.CERTIFICATES, CP.CHAINED, "chain/report/report.json",
                                              "chain.log", "period.log", "piece.log"}
    assert len(json.loads((piece / CP.CERTIFICATES).read_text())["records"]) == 3
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


def test_a_chain_accepted_on_the_period_check_goes_on_to_that_check_whose_answer_is_the_pieces(tmp_path, warden, period):
    """The chain's report says `chain_record.RESTS` where its scan's acceptance rests on the period check. The piece
    then goes on to that check and answers what it answers, a reject as an accept, so no answer is an accept the
    period check didn't give; split in two, the chain step marks DIR/chain.json with no answer, and the period step
    reads that report again and checks V3c's period.json."""
    seen = []
    rejected = CP.answer("reject", "window 0: the scan's prog is no compute audit's slot entries")
    accepted = {"protocol": CP.PROTOCOL, "verdict": "accept"}
    assert _decide(tmp_path / "reject", warden, CR.RESTS, check=lambda c, p: seen.append(p.name) or rejected) == rejected
    assert _decide(tmp_path / "accept", warden, CR.RESTS, check=lambda c, p: seen.append(p.name) or accepted) == accepted
    assert seen == ["period", "period"]
    d = _serve_dir(tmp_path / "serve", warden)
    v3c, piece = _v3c(d, PERIOD), tmp_path / "pieces" / "certifier"
    assert _hook("chain", d, piece, v3c, *_chained(tmp_path, v3c, CR.RESTS, warden["w"])) == 0
    assert (piece / CP.CHAINED).is_file() and not (piece / "answer.json").exists()
    answer = period.reference(period.case("accept", 1, "clock"))
    assert _hook("period", d, piece, v3c, *_lean(tmp_path, answer)) == 0, (piece / "period.log").read_text()
    assert json.loads((piece / "answer.json").read_text())["verdict"] == "accept" and (tmp_path / "seen").is_dir()


@pytest.mark.parametrize("word", ["refused", "scan accepted", f"scan {CR.RESTS}", f"{CR.RESTS}, N2 refused", None])
def test_any_other_word_of_the_chains_report_is_a_reject_and_the_period_check_isnt_run(tmp_path, warden, word):
    """Only the whole chain's two accepting words go on: a refusal, the scan-only path's words and anything else are the
    chain's reject, with the word and its why, before the period check (`_decide`'s fails the test when called)."""
    a = _decide(tmp_path, warden, word)
    assert a["verdict"] == "reject" and a["reason"].startswith(f"the chain {word}: ")


@pytest.mark.parametrize("case", ["no v3c window", "time limit", "no chain step", "a scan of no v3c", "no period.json"])
def test_a_split_step_that_cant_run_refuses_keeping_both_logs(case, tmp_path, warden):
    """The chain step refuses without V3c's committed window, which its scan reads, and when its limit
    (CERTIFIER_TIMEOUT) ends it; the period step then refuses too, naming the chain's refusal. The period step refuses
    where no chain step accepted in DIR, where the chain's scan read no V3c window, and where V3c wrote no period.json:
    DIR keeps its logs and refused.txt, and the certifier is missing."""
    d = _serve_dir(tmp_path / "serve", warden)
    v3c, piece = _v3c(d, None if case == "no period.json" else PERIOD, committed=case != "no v3c window"), \
        tmp_path / "pieces" / "certifier"
    args = _chained(tmp_path, None if case == "a scan of no v3c" else v3c, w=warden["w"])
    if case == "time limit":
        args = ["--chain-cmd", 'kill -TERM "$PPID"; while :; do :; done']
    if case != "no chain step":
        _hook("chain", d, piece, v3c, *args)
    assert _hook("period", d, piece, v3c, *_lean(tmp_path, {})) == 3
    why = (piece / CP.REFUSED).read_text()
    assert {"no v3c window": f"V3c committed no window at {v3c / 'window'}",
            "time limit": "SIGTERM: the piece's time limit (CERTIFIER_TIMEOUT)",
            "no chain step": "no chain.json in DIR: the chain step (CERTIFIER_CHAIN_CMD) didn't run",
            "a scan of no v3c": f"the chain's scan read no V3c window, not {v3c / 'window'}",
            "no period.json": "the period's rows case: FileNotFoundError"}[case] in why, why
    logs = {"chain.log", "period.log"} - ({"chain.log"} if case == "no chain step" else set())
    assert {p.name for p in piece.iterdir()} == {CP.REFUSED, *logs}
    said = (piece / "period.log").read_text()
    assert ("CERTIFIER refused at the chain: " if case in ("no v3c window", "time limit") else "CERTIFIER refused: ") \
        in said
    assert S.piece_row("communication", tmp_path / "pieces")["verdict"] == "missing"
    assert not list((d / CP.PRIVATE / RUN).glob("unpacked-*"))


def test_each_steps_time_limit_names_its_own():
    for limit in ("CERTIFIER_TIMEOUT", "CERTIFIER_PERIOD_TIMEOUT"):
        with pytest.raises(CP.NotRun, match=rf"^SIGTERM: the piece's time limit \({limit}\) ended it"):
            CP.ended(limit)(signal.SIGTERM, None)


def test_n2_and_the_tie_take_every_link_window_of_a_short_run_unless_named():
    assert CP.sized(1, []) == ["CR_N2_DRAW=", "CR_TIE_WINDOWS=1"]
    assert CP.sized(8, ["CR_N2_DRAW=4"]) == ["CR_TIE_WINDOWS=8", "CR_N2_DRAW=4"]
    assert CP.sized(9, ["CR_SCAN_ONLY=1"]) == ["CR_SCAN_ONLY=1"]


@pytest.fixture(autouse=True)
def challenger_host(monkeypatch):
    monkeypatch.setenv("VY_CHALLENGER_1_HOST", CHALLENGER_HOST)


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
    assert json.loads((out / "session.json").read_text())["gate"] == LIVE_GATE
    assert CP.main(["front", "18190", str(serve), str(out), "--tokens", "64"]) == 2
    assert "past one link-window" in json.loads((out / "session.json").read_text())["refused"]
    assert CP.main(["front", "18190", str(serve), str(out), "--tokens", "5"]) == 2
    assert "1 to KO = 4 ids" in json.loads((out / "session.json").read_text())["refused"]
    assert CP.main(["front", "18190", str(tmp_path / "serve-w63"), str(tmp_path / "out-w63"), "--key",
                    str(tmp_path / "key"), "--scan-class", "w63"]) == 0
    assert (tmp_path / "request").read_text().endswith("--tokens 4 --scan-class w63 --window-per-group")


def test_the_front_without_the_challenger_hosts_address_is_refused_by_the_variables_name_before_the_gate(tmp_path,
                                                                                                    monkeypatch):
    """The front's default gate is `SS.GATE`, whose host the environment holds: unset, the front is refused by the
    variable's name and sends nothing; the dry run's ``--gate local`` reads no address (the test below runs it)."""
    _stub_gate(tmp_path, monkeypatch)
    monkeypatch.delenv("VY_CHALLENGER_1_HOST")
    serve, out = tmp_path / "serve", tmp_path / "out"
    assert CP.main(["front", "18190", str(serve), str(out), "--key", str(tmp_path / "key")]) == 2
    assert json.loads((out / "session.json").read_text())["refused"].startswith("VY_CHALLENGER_1_HOST is unset: the gate is ")
    assert not (tmp_path / "request").exists()


def test_a_fronts_rehearsal_at_4000_ms_reaches_the_gate_with_its_tick_and_the_scans_schedule_there(tmp_path,
                                                                                                   monkeypatch):
    """``--gate local``, this tree's gate in this process: the run's request records the tick and the minutes counted
    at it, and the warden's command is the scan's schedule at it. The tunnel and the rehearsal are stand-ins, the
    rehearsal's check accepting."""
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
    assert {k: req.get(k) for k in ("bucket_ms", "scan", "prompts", "window_per_group", "minutes")} == {
        "bucket_ms": 4000, "scan": True, "prompts": {"count": CP.PROMPTS, "ids": CP.PROMPTS * CP.PROMPT_IDS},
        "window_per_group": True, "minutes": minutes}
    relay, warden = ran
    assert relay[-5:] == ["relay", "--listen", "1", "--to", "18190"]
    at = warden.index("--ids")
    assert warden[at:at + 1 + len(CP.SS.A.SCAN_SCHEDULE)] == ["--ids", *CP.SS.A.scan_schedule(4000, CP.V3C_SCAN_CLASS)]
    assert CP.SS.A.scan_schedule(4000) != CP.SS.A.SCAN_SCHEDULE
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
    """serve-openai's completions as the warm-up sees them: each answered whole, and recorded in the serve dir's
    request_users.jsonl first (serve-openai records a request there once its window has closed) unless ``record`` is
    off; each request noted with whether the gate had been asked by then."""

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
        out = json.dumps({"id": f"cmpl-{len(s.seen)}", "choices": [{"text": "", "finish_reason": "length"}]}).encode()
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
    srv.seen, srv.status, srv.record, srv.serve, srv.stub_dir = [], 200, True, tmp_path / "serve", tmp_path
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


def test_a_warm_up_that_fails_or_isnt_recorded_closed_refuses_the_front_before_the_gate(tmp_path, endpoint):
    port, serve, out = endpoint.server_address[1], endpoint.serve, tmp_path / "out"
    endpoint.status = 503
    assert CP.main(["front", str(port), str(serve), str(out), "--warmup", "1"]) == 2
    assert json.loads((out / "session.json").read_text())["refused"].startswith("--warmup: request 0: HTTPError")
    assert not (tmp_path / "request").exists() and not (out / "warmup.json").exists()
    endpoint.status, endpoint.record, clock = 200, False, _Clock()
    a = argparse.Namespace(port=port, warmup=1, prompt_ids=CP.PROMPT_IDS, vocab=128000, tokens=CP.TOKENS)
    with pytest.raises(CP.SS.Refused, match="recorded no closed window of .'warmup-"):
        CP.warmup(a, serve, clock=clock, sleep=clock.sleep)
    assert clock.t > CP.WARMUP_CLOSE_S and not (tmp_path / "request").exists()


class _Clock:
    t = 0.0

    def __call__(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.t += s


# ------------------------------------------------------------------------------------------------------ a relay session
# The front's relay session (`front --session`, `close`, and the piece's guard for a session nobody closed), through a
# stub gate that answers a scripted run of statuses, with every nap an injected one: the front sends the gate one
# `relay-session` request and no prompts, returns once the gate's status shows the relay listening and leaves nothing
# in OUT; `close` waits for the session's end, keeps the lane's fetch where the piece reads it and publishes what a
# rehearsal's front does; the piece's steps close a session nobody closed before they read the kept run; and a request
# this tree's gate refuses is refused before the gate is asked.

#: a made-up name of a served node, a session's peer
PEER = "some-served-node"
#: the relay's deployment, as the gate's status gives it once the relay listens and the run's bundle at its end
DEPLOYMENT = {"windows": 20, "origin_unix": 1.5, "listen": ["127.0.0.1", 40001],
              "params": {"window": 32, "bucket_ms": 100}}
#: the gate's status of a relay session before its relay listens, once it does, and at an end its check accepted
WAITING = {"run": RUN, "request": {"device": "warden", "peer": PEER, "client_port": 8012, "windows": 20}, "end": None,
           "active": True, "relay": None}
LISTENING = WAITING | {"relay": {"client": "127.0.0.1:8012", "deployment": DEPLOYMENT}}
ENDED = LISTENING | {"active": False, "end": PASSED | {"ended": "the session ended"}}

SESSION_GATE = r'''#!PYTHON
import io, json, os, sys, tarfile
from pathlib import Path
d, cmd = Path(os.environ["STUB_DIR"]), sys.argv[-1]
with open(d / "asked", "a") as f:
    f.write(cmd + "\n")
if cmd.startswith("relay-session "):
    (d / "stdin").write_bytes(sys.stdin.buffer.read())
    if (d / "busy").exists():
        print(json.dumps({"refused": "a rehearsal is running"}))
        sys.exit(3)
    print(json.dumps({"device": "warden", "run": "RUN", "unit": "vy-certifier@RUN.service"}, sort_keys=True))
elif cmd == "status RUN":
    left = json.loads((d / "statuses.json").read_text())
    (d / "statuses.json").write_text(json.dumps(left[1:] or left))
    if left[0] is None:
        sys.exit("stub gate: no answer")
    print(json.dumps(left[0], sort_keys=True))
elif cmd == "fetch RUN" and not (d / "no-fetch").exists():
    status = json.loads((d / "statuses.json").read_text())[0]
    files = {"request.json": json.dumps(status["request"]).encode(), "end.json": json.dumps(status["end"]).encode(),
             "out/deployment.json": json.dumps(DEPLOYMENT).encode(), "out/certificates.jsonl": b"",
             "witness/certified.jsonl": b"a certified line, which only the run's own lane fetches\n"}
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, data in files.items():
            info = tarfile.TarInfo("RUN/" + name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    sys.stdout.buffer.write(buf.getvalue())
else:
    sys.exit(f"stub gate: no verb {cmd!r}")
'''.replace("PYTHON", sys.executable).replace("DEPLOYMENT", repr(DEPLOYMENT)).replace("RUN", RUN)


def _statuses(tmp_path: Path, statuses: list[dict | None]) -> None:
    """What the stub gate answers ``status`` from here on, one a call and the last ever after (None: no answer)."""
    (tmp_path / "statuses.json").write_text(json.dumps(statuses))


def _session_gate(tmp_path: Path, monkeypatch, statuses: list[dict | None]) -> None:
    """`SESSION_GATE` as ssh: each request it gets is a line of tmp_path/asked."""
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin" / "ssh").write_text(SESSION_GATE)
    (tmp_path / "bin" / "ssh").chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path / 'bin'}{os.pathsep}{os.defpath}")
    monkeypatch.setenv("STUB_DIR", str(tmp_path))
    _statuses(tmp_path, statuses)


def _asked(tmp_path: Path) -> list[str]:
    return (tmp_path / "asked").read_text().splitlines() if (tmp_path / "asked").is_file() else []


def _front(tmp_path: Path, *flags: str, serve: str = "serve", gate: str = f"research@{GATE_HOST}") -> list[str]:
    """``front --session``'s words: the endpoint at 18190, SERVE_DIR tmp_path/`serve` and OUT tmp_path/out."""
    return ["front", "18190", str(tmp_path / serve), str(tmp_path / "out"), "--key", str(tmp_path / "key"),
            "--gate", gate, "--peer", PEER, "--session", *flags]


SESSION_FLAGS = ("--client-port", "8012", "--windows", "20")


def test_the_front_opens_a_relay_session_sends_no_group_and_returns_once_its_relay_listens(tmp_path, monkeypatch,
                                                                                           capsys):
    _session_gate(tmp_path, monkeypatch, [LISTENING])
    serve, out = tmp_path / "serve", tmp_path / "out"
    assert CP.main(_front(tmp_path, *SESSION_FLAGS)) == 0
    minutes = CP.SS.A.session_minutes(20, CP.HEIGHT) + CP.SS.MARGIN
    assert _asked(tmp_path) == [f"relay-session --peer {PEER} --port 18190 --client-port 8012 --windows 20 "
                                f"--minutes {minutes} --height 10 --scan-class w63k4", f"status {RUN}"]
    assert (tmp_path / "stdin").read_bytes() == b"" and list(out.iterdir()) == []
    keep = serve / CP.PRIVATE
    kept = keep / RUN / CP.RELAY
    assert json.loads(capsys.readouterr().out.splitlines()[-1]) == {"run": RUN, "client": "127.0.0.1:8012",
                                                                    "relay": str(kept)}
    doc = json.loads(kept.read_text())
    assert (doc["run"], doc["gate"], doc["client"], doc["deployment"]) == (RUN, f"research@{GATE_HOST}",
                                                                           "127.0.0.1:8012", DEPLOYMENT)
    assert doc["args"] == {"peer": PEER, "port": 18190, "client_port": 8012, "windows": 20, "minutes": minutes,
                           "height": 10, "bucket_ms": None, "scan_class": "w63k4"}
    assert sorted(p.name for p in (keep / RUN).iterdir()) == [CP.RELAY, "start.json"]
    assert [p.stat().st_mode & 0o777 for p in (serve / "private", keep, keep / RUN, kept, keep / RUN / "start.json")] \
        == [0o700, 0o700, 0o700, 0o600, 0o600]
    assert [d["run"] for d in CP.open_sessions(keep)] == [RUN]
    # a second session in the serve dir is refused while this one is open, before the gate is asked
    assert CP.main(_front(tmp_path, "--client-port", "8013", "--windows", "20")) == 2
    assert f"relay session {RUN} is open" in json.loads((out / "session.json").read_text())["refused"]
    assert len(_asked(tmp_path)) == 2
    # the scan's tick and class reach the gate, and the minutes are counted at the tick
    assert CP.main(_front(tmp_path, *SESSION_FLAGS, "--bucket-ms", "200", "--scan-class", "w63", serve="serve-200")) == 0
    at_200 = CP.SS.A.session_minutes(20, CP.HEIGHT, 32 * 200) + CP.SS.MARGIN
    assert at_200 > minutes and _asked(tmp_path)[2].endswith(
        f"--windows 20 --minutes {at_200} --height 10 --bucket-ms 200 --scan-class w63")
    # the longest session at the scan's tick: the least height that signs its execution phases, and its minutes at it
    assert CP.main(_front(tmp_path, "--client-port", "8012", "--windows", "1119", serve="serve-long")) == 0
    assert (CP.SS.A.warden_height(1119), CP.SS.A.session_minutes(1119, 13) + CP.SS.MARGIN) == (13, 66)
    assert _asked(tmp_path)[4].endswith("--windows 1119 --minutes 66 --height 13 --scan-class w63k4")
    assert CP.main(_front(tmp_path, "--client-port", "8012", "--windows", "1119", "--bucket-ms", "200",
                          serve="serve-longest")) == 0
    assert _asked(tmp_path)[6].endswith("--windows 1119 --minutes 126 --height 13 --bucket-ms 200 --scan-class w63k4")


@pytest.mark.parametrize("statuses, rc, naps, why", [
    ([WAITING, None, WAITING, LISTENING], 0, 3, None),
    ([WAITING, WAITING | {"active": False, "end": {"run": RUN, "rc": 2, "ended": "the tunnel didn't come up"}}], 1, 1,
     f"relay session {RUN} ended before its relay listened: exit 2, the tunnel didn't come up"),
    ([WAITING | {"active": False}], 1, 1, f"relay session {RUN}: its unit ended and wrote no end.json"),
    ([WAITING], 1, 5 * 60 + 1, f"relay session {RUN}: 5 minutes of asking"),
])
def test_a_sessions_front_waits_on_the_gates_status_and_says_why_one_never_listened(tmp_path, monkeypatch, statuses,
                                                                                    rc, naps, why):
    """The front asks the gate's status every `READY_POLL_S` seconds (a gate that doesn't answer is asked again) until
    the relay listens. A session that ended first, a unit twice seen ended with no end, and a session's own minutes of
    asking each end the front with exit 1 and the reason; the session stays open, so a close keeps its fetch."""
    _session_gate(tmp_path, monkeypatch, statuses)
    if naps > 3:  # a gate that never shows the relay: its answer without the stub's process each time
        monkeypatch.setattr(CP, "gate_status", lambda a, keep, run: WAITING)
    slept = []
    a = CP.parser().parse_args(_front(tmp_path, *SESSION_FLAGS, "--minutes", "5"))
    assert CP.front(a, sleep=slept.append) == rc
    assert slept == [CP.READY_POLL_S] * naps
    doc = json.loads((tmp_path / "serve" / CP.PRIVATE / RUN / CP.RELAY).read_text())
    assert [d["run"] for d in CP.open_sessions(tmp_path / "serve" / CP.PRIVATE)] == [RUN]
    if why is None:
        assert doc["client"] == "127.0.0.1:8012" and not (tmp_path / "out" / "session.json").exists()
    else:
        assert "client" not in doc and "deployment" not in doc
        assert json.loads((tmp_path / "out" / "session.json").read_text())["refused"].startswith(why)


@pytest.mark.parametrize("flags, why", [
    ((), "--session: with --client-port PORT"),
    (("--client-port", "8012"), "--session: with --client-port PORT"),
    ((*SESSION_FLAGS, "--warmup", "1"), "--session sends no request of its own"),
    (("--client-port", "18190", "--windows", "20"), "the gate refuses it: --client-port 18190: a port of the peer's"),
    (("--client-port", "7811", "--windows", "20"), "the gate refuses it: --client-port 7811: a port of the peer's"),
    (("--client-port", "8012", "--windows", "1120"), "the gate refuses it: --windows '1120': 1 to 1119"),
    (("--client-port", "8012", "--windows", "1100", "--height", "11"),
     "the gate refuses it: --height 11: the key's 2048 one-time signatures don't sign 1100 execution phases"),
    ((*SESSION_FLAGS, "--minutes", "4"),
     "the gate refuses it: --minutes 4: 20 execution phases of 3200 ms take 5 minutes"),
    ((*SESSION_FLAGS, "--bucket-ms", "50"), "the gate refuses it: --bucket-ms '50': 100 to 4000"),
])
def test_a_session_this_trees_gate_refuses_is_refused_before_the_gate_is_asked(tmp_path, monkeypatch, flags, why):
    _session_gate(tmp_path, monkeypatch, [LISTENING])
    assert CP.main(_front(tmp_path, *flags)) == 2
    assert json.loads((tmp_path / "out" / "session.json").read_text())["refused"].startswith(why)
    assert _asked(tmp_path) == [] and list((tmp_path / "serve" / CP.PRIVATE).iterdir()) == []


def test_a_session_the_gate_doesnt_start_opens_none(tmp_path, monkeypatch):
    _session_gate(tmp_path, monkeypatch, [LISTENING])
    (tmp_path / "busy").touch()
    assert CP.main(_front(tmp_path, *SESSION_FLAGS)) == 2
    refused = json.loads((tmp_path / "out" / "session.json").read_text())["refused"]
    assert refused.startswith("the gate didn't start the session (exit 3)") and "a rehearsal is running" in refused
    assert len(_asked(tmp_path)) == 1 and CP.open_sessions(tmp_path / "serve" / CP.PRIVATE) == []
    assert CP.main(["close", str(tmp_path / "serve"), str(tmp_path / "out-close")]) == 2
    assert json.loads((tmp_path / "out-close" / "session.json").read_text())["refused"].startswith(
        "0 relay sessions open")
    assert len(_asked(tmp_path)) == 1


def test_close_waits_for_the_sessions_end_keeps_its_fetch_and_publishes_what_a_rehearsals_front_does(tmp_path,
                                                                                                     monkeypatch):
    _session_gate(tmp_path, monkeypatch, [LISTENING])
    serve, out = tmp_path / "serve", tmp_path / "out-close"
    assert CP.main(_front(tmp_path, *SESSION_FLAGS)) == 0
    _statuses(tmp_path, [LISTENING, None, LISTENING, ENDED])
    slept = []
    assert CP.close(argparse.Namespace(serve_dir=serve, out=out), sleep=slept.append) == 0
    assert slept == [CP.SS.POLL_S] * 3
    assert _asked(tmp_path)[2:] == [f"status {RUN}"] * 4 + [f"fetch {RUN}"]
    kept = serve / CP.PRIVATE / RUN / "fetch.tar.gz"
    assert kept.stat().st_mode & 0o777 == 0o600 and CP.kept_run(serve) == RUN
    assert sorted(p.name for p in out.iterdir()) == ["bundle.tgz", "leak-guard.json", "session.json", "status.json"]
    assert json.loads((out / "status.json").read_text()) == {k: v for k, v in ENDED.items() if k != "relay"}
    with tarfile.open(kept) as full, tarfile.open(out / "bundle.tgz") as shown:
        assert f"{RUN}/witness/certified.jsonl" in full.getnames()
        assert sorted(shown.getnames()) == [f"{RUN}/end.json", f"{RUN}/out/certificates.jsonl",
                                            f"{RUN}/out/deployment.json", f"{RUN}/request.json"]
    doc = json.loads((out / "session.json").read_text())
    assert (doc["run"], doc["rc"], doc["ended"], doc["verdict"]["accepted"], doc["windows"], doc["gate"]) == (
        RUN, 0, "the session ended", True, 20, f"research@{GATE_HOST}")
    assert doc["request"] == _asked(tmp_path)[0] and doc["keep"] == str(kept)
    assert (kept.parent / CP.SESSION).read_text() == (out / "session.json").read_text()
    assert CP.placement(serve, RUN)["parties"] == {"network_certifier": GATE_HOST, "prover": PEER, "verifier": PEER}
    assert CP.open_sessions(serve / CP.PRIVATE) == []
    # with the session closed there is none to close, and the gate isn't asked
    asked = len(_asked(tmp_path))
    assert CP.main(["close", str(serve), str(tmp_path / "out-again")]) == 2
    assert len(_asked(tmp_path)) == asked


def test_a_session_the_gates_check_refused_is_closed_and_kept_and_the_piece_doesnt_answer_for_it(tmp_path, monkeypatch):
    """The whole-run rule across a session: an end whose check didn't accept the run (here each connection late) is
    `close`'s exit, its fetch is kept all the same, and the piece's steps refuse on it (`rehearsed`), which leaves the
    certifier missing for every window of the served run."""
    _session_gate(tmp_path, monkeypatch, [LISTENING])
    serve = tmp_path / "serve"
    assert CP.main(_front(tmp_path, *SESSION_FLAGS)) == 0
    _statuses(tmp_path, [LISTENING | {"active": False, "end": LATE}])
    assert CP.main(["close", str(serve), str(tmp_path / "out-close")]) == 1
    doc = json.loads((tmp_path / "out-close" / "session.json").read_text())
    assert (doc["rc"], doc["verdict"]["accepted"]) == (1, False) and CP.kept_run(serve) == RUN
    unpacked, run = CP.unpack(serve / CP.PRIVATE / RUN / "fetch.tar.gz")
    try:
        with pytest.raises(CP.NotRun):
            CP.rehearsed(run)
    finally:
        CP.rm(unpacked)


def test_the_pieces_steps_close_a_relay_session_nobody_closed_before_they_read_the_kept_run(tmp_path, monkeypatch,
                                                                                            capsys):
    _session_gate(tmp_path, monkeypatch, [LISTENING])
    serve, keep = tmp_path / "serve", tmp_path / "serve" / CP.PRIVATE
    assert CP.main(_front(tmp_path, *SESSION_FLAGS)) == 0
    _statuses(tmp_path, [LISTENING, ENDED])
    slept = []
    CP.close_open(serve, sleep=slept.append)
    assert slept == [CP.SS.POLL_S] and CP.kept_run(serve) == RUN and CP.open_sessions(keep) == []
    assert sorted(p.name for p in (keep / RUN).iterdir()) == ["fetch.tar.gz", CP.RELAY, CP.SESSION, "start.json"]
    assert (keep / RUN / "fetch.tar.gz").stat().st_mode & 0o777 == 0o600
    assert json.loads((keep / RUN / CP.SESSION).read_text())["gate"] == f"research@{GATE_HOST}"
    # nothing is published and no hold is left: the guard keeps, and a close is what publishes
    assert list((keep / "held").iterdir()) == [] and list((tmp_path / "out").iterdir()) == []
    assert f"relay session {RUN} closed here: no close ran" in capsys.readouterr().out
    asked = len(_asked(tmp_path))
    CP.close_open(serve)
    assert len(_asked(tmp_path)) == asked
    # a step closes it first: by the time it reads the kept run, the session's fetch is the kept one
    again, seen = tmp_path / "serve-step", []
    assert CP.main(_front(tmp_path, *SESSION_FLAGS, serve="serve-step")) == 0
    _statuses(tmp_path, [ENDED])

    def step(hold: Path):
        seen.append((CP.kept_run(again), CP.open_sessions(again / CP.PRIVATE)))
        raise CP.NotRun("the step's own")

    with pytest.raises(CP.NotRun, match="the step's own"):
        CP.settle(again, K, tmp_path / "dir", "CERTIFIER_TIMEOUT", step)
    assert seen == [(RUN, [])]
    # a session the gate's fetch fails for can't be closed: the step doesn't run, and the session stays open
    third = tmp_path / "serve-no-fetch"
    assert CP.main(_front(tmp_path, *SESSION_FLAGS, serve="serve-no-fetch")) == 0
    (tmp_path / "no-fetch").touch()
    with pytest.raises(CP.NotRun, match=f"relay session {RUN}, which no close ended, couldn't be closed here: .*fetch"):
        CP.settle(third, K, tmp_path / "dir", "CERTIFIER_TIMEOUT", step)
    assert len(seen) == 1 and [d["run"] for d in CP.open_sessions(third / CP.PRIVATE)] == [RUN]
    assert list((third / CP.PRIVATE / "held").iterdir()) == []


def test_a_local_gates_relay_session_is_this_trees_gate_its_unit_a_relay_each_way(tmp_path, monkeypatch):
    """``--gate local``: this tree's gate takes the front's request on the scratch host under the serve dir's private,
    answers its status and gives its fetch, and the unit the front starts runs `auditor.certifier_run` with a `relay`
    each way in its ssh's place. The unit here is a stand-in that leaves the files a session's does."""
    spent = subprocess.Popen([sys.executable, "-c", ""])
    spent.wait()
    serve, out, started = tmp_path / "serve", tmp_path / "out", []
    keep = serve / CP.PRIVATE

    def unit(pid: int, listens: bool):
        class Unit:
            def __init__(self, argv, **kw):
                started.append(argv)
                self.pid = pid
                if listens:
                    d = CP.SS.scratch_host(keep / "local-gate", PEER).certifier / argv[4] / CP.SS.A.WITNESS
                    d.mkdir()
                    (d / "deployment.json").write_text(json.dumps(DEPLOYMENT))
        return Unit

    monkeypatch.setattr(CP.SS, "pinned", lambda: "unpinned")
    with monkeypatch.context() as m:  # a unit that ended at once and wrote no end: the front says so
        m.setattr(CP.subprocess, "Popen", unit(spent.pid, False))
        slept = []
        a = CP.parser().parse_args(_front(tmp_path, *SESSION_FLAGS, serve="serve-spent", gate="local"))
        assert CP.front(a, sleep=slept.append) == 1 and slept == [CP.READY_POLL_S]
    assert "its unit ended and wrote no end.json" in json.loads((out / "session.json").read_text())["refused"]
    with monkeypatch.context() as m:
        m.setattr(CP.subprocess, "Popen", unit(os.getpid(), True))
        assert CP.main(_front(tmp_path, *SESSION_FLAGS, gate="local")) == 0
    (doc,) = CP.open_sessions(keep)
    run, host = doc["run"], CP.SS.scratch_host(keep / "local-gate", PEER)
    assert started[-1] == [sys.executable, str(HERE / "certifier_piece.py"), "unit", str(keep), run, PEER, "18190"]
    assert (doc["gate"], doc["unit"], doc["client"], doc["deployment"]) == ("local", os.getpid(), "127.0.0.1:8012",
                                                                            DEPLOYMENT)
    req = json.loads((host.certifier / run / "request.json").read_text())
    minutes = CP.SS.A.session_minutes(20, CP.HEIGHT) + CP.SS.MARGIN
    assert {k: req[k] for k in ("device", "peer", "port", "client_port", "windows", "minutes", "height")} == {
        "device": "warden", "peer": PEER, "port": 18190, "client_port": 8012, "windows": 20, "minutes": minutes,
        "height": 10}
    # the unit: certifier_run on that host, its tunnel a relay to the endpoint and one from the clients' port
    ran = []

    def certifier_run(host_, run_, **kw):
        ran.append((host_.root, run_, kw["pinned"], kw["tunnel_argv"]([], 1, (8012, 2))))
        return {"rc": 0}

    with monkeypatch.context() as m:
        m.setattr(CP.SS.A, "certifier_run", certifier_run)
        assert CP.main(["unit", str(keep), run, PEER, "18190"]) == 0
    assert ran == [(host.root, run, "unpinned", [sys.executable, str(HERE / "certifier_piece.py"), "relay", "--listen",
                                                 "1", "--to", "18190", "--listen", "8012", "--to", "2"])]
    # the session's end as the unit leaves it, and the close through the same gate
    d = host.certifier / run
    (d / "out").mkdir()
    (d / "out" / "deployment.json").write_text(json.dumps(DEPLOYMENT))
    (d / "end.json").write_text(json.dumps(PASSED | {"run": run, "ended": "the session ended"}))
    assert CP.main(["close", str(serve), str(tmp_path / "out-close")]) == 0
    assert CP.kept_run(serve) == run and CP.open_sessions(keep) == []
    shown = json.loads((tmp_path / "out-close" / "session.json").read_text())
    assert (shown["gate"], shown["rc"], shown["verdict"]["accepted"]) == ("local", 0, True)
    assert set(json.loads((tmp_path / "out-close" / "status.json").read_text())) == {"run", "request", "end", "active"}
    with tarfile.open(keep / run / "fetch.tar.gz") as full, tarfile.open(tmp_path / "out-close" / "bundle.tgz") as pub:
        assert f"{run}/{CP.SS.A.WITNESS}/deployment.json" in full.getnames()
        assert not any(CP.SS.A.WITNESS in n.split("/") for n in pub.getnames())


def test_the_local_tunnel_relays_each_listen_port_to_its_own(monkeypatch):
    """`relay --listen A --to B --listen C --to D`: one relay a pair, the last started once every other listens."""
    order = []

    def relay(listen: int, to: int, *, ready=lambda: None):
        order.append((listen, to, threading.current_thread() is threading.main_thread()))
        ready()

    monkeypatch.setattr(CP, "relay", relay)
    assert CP.main(["relay", "--listen", "1", "--to", "2"]) == 0
    assert CP.main(["relay", "--listen", "3", "--to", "4", "--listen", "5", "--to", "6"]) == 0
    assert order == [(1, 2, True), (3, 4, False), (5, 6, True)]
    with pytest.raises(SystemExit, match="one --to for each --listen"):
        CP.main(["relay", "--listen", "1", "--to", "2", "--listen", "3"])


def test_the_served_tables_scan_class_is_v3cs_setup_in_scan_zk():
    """`served_table.SCAN_CLASS` is `V3C_SCAN_CLASS`'s setup (`scan_zk.W63K4`) as served_verdicts.sh's
    SLOTS,PROMPT,OUTPUTS,LANES,EVERY: a slot stride's Program slots over an execution phase's strides, KI, KO, the
    lanes and EVERY."""
    import scan_zk as SZ

    spec = importlib.util.spec_from_file_location("served_table", ROOT / "demos/onsite_served/served_table.py")
    table = importlib.util.module_from_spec(spec)
    sys.modules["served_table"] = table
    spec.loader.exec_module(table)
    setup = SZ.SETUPS[CP.V3C_SCAN_CLASS]
    c = setup.cls
    assert table.SCAN_CLASS == f"{c.SLOTS * setup.params.window // c.EVERY},{c.KI},{c.KO},{c.lanes},{c.EVERY}"
