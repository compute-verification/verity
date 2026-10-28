"""certifier_piece: the front sends one group of the window circuit's class through the gate's scan rehearsal and keeps
the gate's fetch, witness and all, 0700 under the serve dir's private/certifier, publishing only the bundle less witness/
and its outcome. The piece, on a serve dir whose window was the warden's synthetic run's certified requests (the run
`test_warden_chain.gate_run` publishes in the gate's layout), runs as `served_verdicts.py pieces` runs it and writes
`network-period`'s answer only when the chain accepted and the window's requests are certified ones in the scan's
link-window: the period mirror's certifier/v0 accept (`period_case`), its profile with the placement (the certifier on
the kept gate's host, each draw's coins the challenger's host's, the checks on the prover's node), then goes through
`served_verdicts.py record` into the window report and the block record's certifier leg, which `block_record.check` holds and its schema admits, and its case.json
is the three files `network-period` read, as `composite.py` reads a piece's case; the chain's endpoint report, with
each Commit window's counts, stays in the kept run. A window request the warden didn't certify is
rejected before any chain runs; a chain that refused, requests outside the scan's link-window and a period the check
rejects are rejects; a rehearsal the gate's check failed (each request late) and a step that can't run (no kept run, no
report, no build, the time limit, no challenger, coins not the challenger's) refuse, which leaves the certifier
missing, and no process of a step's group outlives the piece, the time limit's SIGTERM included, whenever it comes.
The leak guard reads the piece's DIR, each drawing round's challenger evidence included, against the run's witness,
leaves the evidence, and the case is written only from the period files it left. The front's warm-ups go straight to
the endpoint, one at a time, each recorded in a closed window of its own before the gate is asked, and the pick passes
over those windows for the group's. At a tick of 4000 ms the front's rehearsal reaches this tree's gate with the tick
in its request and the minutes counted at it, and the warden runs the scan's schedule there; a tick the gate refuses
(past its range, or on a request that isn't the scan rehearsal) refuses the front before the gate is asked."""
from __future__ import annotations

import argparse
import http.server
import importlib.util
import json
import os
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
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(Path(__file__).resolve().parent), str(ROOT / "benchmarks" / "one_stage")]

import certifier_piece as CP  # noqa: E402
import served_verdicts as S  # noqa: E402
from test_served_session import RUN, STUB  # noqa: E402
from test_warden_chain import WC, gate_run  # noqa: E402
from verity.ml.circuits.codec import canonical_json  # noqa: E402
from verity_e2e_demo import block_record as BR  # noqa: E402

PERIOD_CASE = ROOT / "verity" / "core" / "protocols" / "network" / "tests" / "period_case.py"
SCHEMA = json.loads((Path(BR.__file__).parent / "schemas" / "block-record.v1.schema.json").read_text())
K = 3
#: the gate's end of a rehearsal its check passed, and of one whose every request was late past the warden's D·τ
PASSED = {"run": RUN, "rc": 0, "ended": "the rehearsal ended", "verdict": {"accepted": True, "problems": 0}}
LATE = {"run": RUN, "rc": 1, "ended": "the rehearsal exited 1",
        "verdict": {"accepted": False, "problems": 8, "requests": 4, "ended": 0, "failed": ["fail"]}}
GATE_HOST = "gate-host"


@pytest.fixture(scope="module")
def period():
    spec = importlib.util.spec_from_file_location("period_case", PERIOD_CASE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["period_case"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def warden(tmp_path_factory):
    """The synthetic warden run, its fetch (``late``: the same run, its rehearsal failed late), and the link-window
    holding the most whole certified requests."""
    d = tmp_path_factory.mktemp("warden")
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
    return {**fetches, "w": w, "records": by[w][:4], "others": [r for x, rs in by.items() if x != w for r in rs],
            "windows": WC.deployment(run)[0]["windows"]}


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _serve_dir(d: Path, warden: dict, records: list[dict] | None = None, kept: bool = True,
               fetch: str = "fetch") -> Path:
    """A serve dir whose window K served ``records`` (the warden's, by default those of its link-window) and whose
    gate fetch (``warden[fetch]``) the front kept."""
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


#: a drawing round's challenger evidence as the chain leaves it in out/ (`round_coins.RoundSession`), which DIR
#: publishes
EVIDENCE = ("scan/honest/challenger/session.json", "scan/honest/challenger/pins.jsonl", "scan/honest/challenger/coins.json",
            "n2/honest/challenger/coins.json", "n2/honest/verifier/period-certificates.json")


def _chain_out(d: Path, verdict: str, w: int, coins: dict | None = None) -> Path:
    """What the chain leaves in its run dir's out/: its report, its coins step `coins` (accepted on the challenger
    ``local`` by default), each drawing round's challenger evidence, the endpoint's and the scan's one link-window."""
    out = d / "out"
    (out / "report").mkdir(parents=True)
    (out / "endpoint").mkdir(parents=True)
    (out / "endpoint" / "report.json").write_text(json.dumps({"chain": {"bits": {"commit_windows": [
        {"window": K, "requests": 3, "carried": 0, "outputs": 12, "t0_s": 0.5}]}}}))
    (out / "scan" / "joint").mkdir(parents=True)
    coins = {"verdict": "accepted", "challenger": "local", "why": None} if coins is None else coins
    (out / "report" / "report.json").write_text(json.dumps({"verdict": verdict, "why": None if verdict == "accepted"
                                                            else {"n2": "a window refused"},
                                                            "steps": {"coins": coins} if coins else {}}))
    (out / "scan" / "answers.json").write_text(json.dumps({"registration": {"program": "ab" * 64}, "windows": [
        {"window": w, "answer": {"verdict": "accept"}, "registrations": []}]}))
    (out / "scan" / "joint" / "program.json").write_text("{}")
    for rel in EVIDENCE:
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        (out / rel).write_text(json.dumps({"evidence": rel}) + "\n")
    return d


def _piece_cmd(tmp_path: Path, chain: Path, answer: dict, cmd: str | None = None, lean: Path | None = None,
               coins: str = "--challenger local") -> str:
    """The piece's command, its `network-period` a stand-in that keeps the three files it reads in TMP/seen and
    answers ``answer``, its chain's coins ``coins``."""
    if lean is None:
        lean, seen = tmp_path / "network-period", shlex.quote(str(tmp_path / "seen"))
        (tmp_path / "lean-answer.json").write_text(json.dumps(answer, sort_keys=True))
        lean.write_text(f"#!/bin/sh\nmkdir -p {seen} && cp \"$2\" \"$3\" \"$4\" {seen}/\n"
                        f"cat {shlex.quote(str(tmp_path / 'lean-answer.json'))}\n")
        lean.chmod(0o755)
    cmd = cmd or f'mkdir -p "$RESEARCH_RUN_DIR" && cp -r {shlex.quote(str(chain / "out"))} "$RESEARCH_RUN_DIR/"'
    return (f"{shlex.quote(sys.executable)} {shlex.quote(str(HERE / 'certifier_piece.py'))} piece \"$@\" "
            f"--lean {shlex.quote(str(lean))} --chain-cmd {shlex.quote(cmd)} {coins}")


def _integrity() -> dict:
    a = {"protocol": "verity/sampled-proofs/one-stage/v1", "verdict": "accept", "reason": None, "commitment": "ab" * 64,
         "public": {"bound": 2, "drawn": [0, 3], "per_template": None, "proved": [0, 3]},
         "profile": {"levels": [{"name": "unit", "size": 4}], "drawn": {"unit": 2}, "delta": "1/100"}}
    return S.row_of(S.INTEGRITY_OWNER, a, Path("/nonexistent"))


def _record(monkeypatch, capsys, serve: Path, pieces: Path) -> tuple[dict | None, dict]:
    """`served_verdicts.py record` over the pieces `pieces` ran, sampled proofs' row an accept: the block record (None
    where none was written) and the ``RECORD`` line."""
    monkeypatch.setattr(S, "sampled_proofs_row", lambda d, run=None: _integrity())
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
    assert (where["kind"], where["parties"]) == ("not-placed", {"network_certifier": GATE_HOST, "challenger": node,
                                                                "prover": node, "verifier": node})
    assert where[CP.PREMISE].startswith("assumed, the auditor's devices not placed")
    assert "the draws' coins (--challenger local)" in where[CP.PREMISE]
    assert [(x["name"], x["counted"]) for x in where["links"]] == [("certified", True), ("gate", False)]
    assert json.loads((piece / "chain" / "report" / "report.json").read_text())["verdict"] == "accepted"
    assert not (piece / "chain" / "endpoint").exists()
    assert {p.name for p in piece.iterdir()} - {"piece.json"} == {"answer.json", CP.CASE, "chain", "piece.log"}
    assert all(json.loads((piece / "chain" / rel).read_text()) == {"evidence": rel} for rel in EVIDENCE)
    seen = {f.name: f.read_bytes() for f in (tmp_path / "seen").iterdir()}
    raw = (piece / CP.CASE).read_bytes()
    case = json.loads(raw)
    assert sorted(case) == ["commitment", "evidence", "params"] and raw == canonical_json(case) + b"\n"
    assert (bytes.fromhex(case["commitment"]), bytes.fromhex(case["evidence"]), canonical_json(case["params"])) == (
        seen["commitment.bin"], seen["evidence.json"], seen["params.json"])
    log = (piece / "piece.log").read_text()
    assert "leak guard (exit 0)" in log and f"case: {CP.CASE}, {len(raw)} bytes" in log
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


REFUSED_COINS = {"verdict": "refused", "challenger": f"gate:{CP.SS.GATE}",
                 "why": "scan/honest: the challenger's session: the gate is unreachable"}
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
    was = signal.signal(signal.SIGTERM, CP.ended)
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
                                  "foreign answer", "no challenger", "coins refused", "no coins step"])
def test_a_step_that_cant_run_refuses_and_leaves_the_certifier_missing(case, tmp_path, warden, capsys, monkeypatch):
    """The front's failures (the gate refusing or busy, its tunnel or ssh) keep no run; a corrupt fetch, a chain that
    ends without a report, a missing Lean build, SIGTERM from the piece's time limit and a check that answers no
    certifier/v0 verdict can't run: each leaves DIR/refused.txt and its log, no answer, and the certifier missing. So
    do the chain's coins when they aren't the challenger's: the gate named without its key record, or a report whose
    coins step isn't accepted (the gate unreachable, a session refused) or that has none, with nothing drawn on other
    coins in their place."""
    d = _serve_dir(tmp_path / "serve", warden, kept=case != "no kept run")
    if case == "corrupt fetch":
        (d / CP.PRIVATE / RUN / "fetch.tar.gz").write_bytes(b"not a tarball")
    coins = {"coins refused": REFUSED_COINS, "no coins step": {}}.get(case)
    chain = _chain_out(tmp_path / "chain", "accepted", warden["w"], coins)
    pids = tmp_path / "pids"
    cmd = {"no report": "exit 7", "time limit": f'({LOOP}) & echo "$$ $!" > {shlex.quote(str(pids))}; '
                                                f'kill -TERM "$PPID"; {LOOP}'}.get(case)
    lean = tmp_path / "no-such-network-period" if case == "no build" else None
    answer = {"verdict": "accept", "protocol": "other"} if case == "foreign answer" else {}
    flags = "" if case == "no challenger" else "--challenger local"
    env = dict(os.environ, CERTIFIER_CMD=_piece_cmd(tmp_path, chain, answer, cmd, lean, flags))
    S.pieces(d, K, tmp_path / "pieces", env=env)
    if case == "time limit":
        _gone(map(int, pids.read_text().split()))
    piece = tmp_path / "pieces" / "certifier"
    why = (piece / CP.REFUSED).read_text()
    assert {p.name for p in piece.iterdir()} - {"piece.json"} == {CP.REFUSED, "piece.log"}, why
    assert S.piece_row("communication", tmp_path / "pieces")["verdict"] == "missing"
    assert {"no kept run": "no warden run kept", "corrupt fetch": "fetch.tar.gz: ReadError", "no report": "the chain wrote no report (exit 7)",
            "no build": "network-period check: FileNotFoundError", "time limit": "SIGTERM: the piece's time limit",
            "foreign answer": "no verity/certifier/v0 verdict",
            "no challenger": "no challenger for the chain's coins: --challenger gate:… needs --challenger-key",
            "coins refused": f"the challenger's coins (gate:{CP.SS.GATE}): scan/honest: the challenger's session: the "
                             "gate is unreachable",
            "no coins step": "the challenger's coins (None): the report has no coins step"}[case] in why
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
    rc = CP.main(["piece", str(d), str(K), str(piece), "--lean", str(lean), "--chain-cmd", cmd, "--challenger", "local"])
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
    """The chain's job.json carrying a piece of the warden run's witness (a salt of its cells.salts, as hex): the guard
    moves DIR/chain/job.json to the quarantine and releases the rest, the answer with it."""
    with tarfile.open(warden["fetch"]) as tar:
        m = next(m for m in tar if m.name.endswith("/witness/cells.salts"))
        salt = tar.extractfile(m).read(32)[16:]
    chain = _chain_out(tmp_path / "chain", "accepted", warden["w"])
    (chain / "out" / "job.json").write_text(json.dumps({"x": salt.hex()}))
    rc, piece, d = _piece(tmp_path, warden, chain)
    assert rc == 1 and (piece / "answer.json").is_file() and (piece / "chain" / "report" / "report.json").is_file()
    assert not (piece / "chain" / "job.json").exists()
    assert (d / CP.PRIVATE / "quarantine" / "files" / "chain" / "job.json").is_file()
    assert list((d / CP.PRIVATE / "held").iterdir()) == []


def test_a_rehearsal_the_gates_check_failed_late_leaves_the_certifier_missing_not_rejected(tmp_path, warden, capsys,
                                                                                          monkeypatch):
    """Each request late past the warden's D·τ: the gate's check fails the rehearsal, and the served answers, longer
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


def test_the_placement_names_the_kept_gate_a_local_gate_the_prover_node_and_refuses_without_the_session(tmp_path,
                                                                                                      warden):
    """The placement names the challenger whose sessions gave the draws' coins: the gate's host for gate:…, the
    prover's node for a dry run's local, whose premise then counts the draws' coins among what ran there."""
    d = _serve_dir(tmp_path / "serve", warden)
    gated = CP.placement(d, RUN, f"gate:research@{GATE_HOST}")
    assert gated["parties"]["network_certifier"] == gated["parties"]["challenger"] == GATE_HOST
    assert f"each drawing round's coins came from the auditor's challenger on {GATE_HOST}" in gated[CP.PREMISE]
    assert "the draws' coins" not in gated[CP.PREMISE]
    assert "the draws' coins (--challenger local)" in CP.placement(d, RUN, "local")[CP.PREMISE]
    (d / CP.PRIVATE / RUN / CP.SESSION).write_text(json.dumps({"run": RUN, "gate": "local"}))
    p = CP.placement(d, RUN, "local")
    assert (p["kind"], p["links"]) == ("prover-node", []) and p[CP.PREMISE].startswith("unmet")
    assert p["parties"]["network_certifier"] == p["parties"]["prover"] == p["parties"]["challenger"] == \
        socket.gethostname()
    (d / CP.PRIVATE / RUN / CP.SESSION).unlink()
    with pytest.raises(CP.NotRun, match="keep the run with this tree's front"):
        CP.placement(d, RUN, "local")


def test_n2_and_the_tie_take_every_link_window_of_a_short_run_unless_named():
    assert CP.sized(1, []) == ["CR_N2_DRAW=", "CR_TIE_WINDOWS=1"]
    assert CP.sized(8, ["CR_N2_DRAW=4"]) == ["CR_TIE_WINDOWS=8", "CR_N2_DRAW=4"]
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
        "--tokens 4 --window-per-group")
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
    assert warden[at:at + 1 + len(CP.SS.A.SCAN_SCHEDULE)] == ["--ids", *CP.SS.A.scan_schedule(4000)]
    assert CP.SS.A.scan_schedule(4000) != CP.SS.A.SCAN_SCHEDULE
    doc = json.loads((out / "session.json").read_text())
    assert doc["request"] == (f"warden-rehearse --peer vy-dc-h200 --port 18190 --minutes {minutes} --requests 4 "
                              "--height 10 --scan --bucket-ms 4000 --prompts --tokens 4 --window-per-group")
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
        "--tokens 4 --window-per-group")
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
