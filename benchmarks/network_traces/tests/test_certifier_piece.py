"""certifier_piece: the front sends one group of the window circuit's class through the gate's scan rehearsal and keeps
the gate's fetch, witness and all, 0700 under the serve dir's private/certifier, publishing only the bundle less witness/
and its outcome. The piece, on a serve dir whose window was the warden's synthetic run's certified requests (the run
`test_warden_chain.gate_run` publishes in the gate's layout), runs as `served_verdicts.py pieces` runs it and writes
`network-period`'s answer only when the chain accepted and the window's requests are certified ones in the scan's
link-window: the period mirror's certifier/v0 accept (`period_case`) then goes through `served_verdicts.py record` into
the block record's certifier leg, which `block_record.check` holds and its schema admits, and its case.json is the three
files `network-period` read, as `composite.py` reads a piece's case. A window request the warden didn't certify is
rejected before any chain runs; a chain that refused, requests outside the scan's link-window and a period the check
rejects are rejects; a rehearsal the gate's check failed (each request late) and a step that can't run (no kept run, no
report, no build, the time limit) refuse, which leaves the certifier missing. Of the chain, DIR gets only `PUBLIC`'s key
paths, which hold none of the window's counts or advice bits, and the leak guard reads DIR against the run's witness and
moves out any file DIR doesn't declare; the case is written only from the period files it left. The front's
warm-ups go straight to the endpoint, one at a time, each recorded in a closed window of its own before the gate is
asked, and the pick passes over those windows for the group's."""
from __future__ import annotations

import argparse
import http.server
import importlib.util
import json
import os
import re
import shlex
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
import chain_record as CR  # noqa: E402
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
#: the audited window's private values, each a number no public field of `_private_chain_out` holds: its Commit
#: window's requests, steps and outputs, its advice's committed, v0 and derived bits, and its requests' prompt lengths
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


def _piece_cmd(tmp_path: Path, chain: Path, answer: dict, cmd: str | None = None, lean: Path | None = None) -> str:
    """The piece's command, its `network-period` a stand-in that keeps the three files it reads in TMP/seen and
    answers ``answer``."""
    if lean is None:
        lean, seen = tmp_path / "network-period", shlex.quote(str(tmp_path / "seen"))
        (tmp_path / "lean-answer.json").write_text(json.dumps(answer, sort_keys=True))
        lean.write_text(f"#!/bin/sh\nmkdir -p {seen} && cp \"$2\" \"$3\" \"$4\" {seen}/\n"
                        f"cat {shlex.quote(str(tmp_path / 'lean-answer.json'))}\n")
        lean.chmod(0o755)
    cmd = cmd or f'mkdir -p "$RESEARCH_RUN_DIR" && cp -r {shlex.quote(str(chain / "out"))} "$RESEARCH_RUN_DIR/"'
    return (f"{shlex.quote(sys.executable)} {shlex.quote(str(HERE / 'certifier_piece.py'))} piece \"$@\" "
            f"--lean {shlex.quote(str(lean))} --chain-cmd {shlex.quote(cmd)}")


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
    assert json.loads((piece / "answer.json").read_text()) == accept
    assert json.loads((piece / "chain" / "report" / "report.json").read_text())["verdict"] == "accepted"
    assert {p.name for p in piece.iterdir()} - {"piece.json"} == {"answer.json", CP.CASE, "chain", "piece.log"}
    seen = {f.name: f.read_bytes() for f in (tmp_path / "seen").iterdir()}
    raw = (piece / CP.CASE).read_bytes()
    case = json.loads(raw)
    assert sorted(case) == ["commitment", "evidence", "params"] and raw == canonical_json(case) + b"\n"
    assert (bytes.fromhex(case["commitment"]), bytes.fromhex(case["evidence"]), canonical_json(case["params"])) == (
        seen["commitment.bin"], seen["evidence.json"], seen["params.json"])
    log = (piece / "piece.log").read_text()
    assert "leak guard (exit 0)" in log and f"case: {CP.CASE}, {len(raw)} bytes" in log
    keep = d / CP.PRIVATE / RUN
    assert sorted(p.name for p in keep.iterdir()) == ["chain", "endpoint", "fetch.tar.gz"]
    assert [(keep / "endpoint" / f).stat().st_mode & 0o777 for f in CP.ENDPOINT] == [0o600] * 4
    row = S.piece_row("communication", tmp_path / "pieces")
    assert (row["verdict"], row["quantity"], row["owner"]) == ("accept", accept["public"], "network-accounting")
    rec, line = _record(monkeypatch, capsys, d, tmp_path / "pieces")
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
    cmd = {"no report": "exit 7", "time limit": 'kill -TERM "$PPID"; while :; do :; done'}.get(case)
    lean = tmp_path / "no-such-network-period" if case == "no build" else None
    answer = {"verdict": "accept", "protocol": "other"} if case == "foreign answer" else {}
    env = dict(os.environ, CERTIFIER_CMD=_piece_cmd(tmp_path, chain, answer, cmd, lean))
    S.pieces(d, K, tmp_path / "pieces", env=env)
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
    file's pieces: so the case is written from DIR/period after the guard has read its files raw, and not at all when
    the guard moved one of them (here the evidence, holding a witness file's text)."""
    fetch = tmp_path / "fetch.tar.gz"
    fetch.write_bytes(warden["fetch"].read_bytes())
    unpacked, run = CP.unpack(fetch)
    for name, evidence in (("clean", b'{"windows":[]}'), ("leaky", (run / "witness" / "certified.jsonl").read_bytes())):
        d = tmp_path / name
        (d / "period").mkdir(parents=True)
        for f, b in (("params.json", b'{"k":1}'), ("commitment.bin", b"\1\2"), ("evidence.json", evidence)):
            (d / "period" / f).write_bytes(b)
        g = CP.guard(d, unpacked, tmp_path / f"{name}-quarantine")
        CP.write_case(d)
        if name == "clean":
            assert g == 0 and not (d / "period").exists()
            assert (d / CP.CASE).read_bytes() == b'{"commitment":"0102","evidence":"%s","params":{"k":1}}\n' % (
                evidence.hex().encode())
        else:
            assert g == 3 and not (d / CP.CASE).exists()
            assert sorted(p.name for p in (d / "period").iterdir()) == ["commitment.bin", "params.json"]


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
    puts none of them in DIR, only `PUBLIC`'s key paths, the endpoint's step its verdict and where it ran. A file put in
    DIR past `publish` (the endpoint's report copied whole, as the piece once published it), though it holds no witness
    piece, is one the leak guard moves to the quarantine: DIR declares only `PIECE_FILES`."""
    out = _private_chain_out(tmp_path / "out")
    held = {p.relative_to(out).as_posix(): _numbers(p) & PRIVATE for p in out.rglob("*") if p.is_file()}
    assert held["endpoint/report.json"] == PRIVATE - set(PROMPT_LENS) and held["tie/summary.json"] == set(PROMPT_LENS)
    assert held["report/report.json"] >= {REQUESTS, OUTPUTS, V0, DERIVED} and held["report/report.md"] >= {
        REQUESTS, OUTPUTS, DERIVED} and held["certs.json"] == {REQUESTS, OUTPUTS}
    assert json.loads((out / "report" / "report.json").read_text())["verdict"] == "accepted"
    d = tmp_path / "pieces" / "certifier"
    CP.publish(out, d)
    published = {p.relative_to(d).as_posix(): p for p in d.rglob("*") if p.is_file()}
    assert sorted(published) == sorted(f"chain/{rel}" for rel in CP.PUBLIC)
    assert {rel: _numbers(p) & PRIVATE for rel, p in published.items()} == dict.fromkeys(published, set())
    report = json.loads(published["chain/report/report.json"].read_text())
    assert report["verdict"] == "accepted" and set(report["steps"]) == {
        "custody", "certificates", "endpoint", "tie", "n2", "scan", "lean"}
    assert report["steps"]["endpoint"] == {"verdict": "accepted", "in_the_clear": CR.CLEAR["warden_endpoint_served"]}
    assert report["steps"]["scan"]["eps_at_delta"]["units"] == 60
    assert report["stubbed"] == list(CR.WARDEN_SERVED_STUBBED)
    assert json.loads(published["chain/scan/answers.json"].read_text()) == json.loads(
        (out / "scan" / "answers.json").read_text())
    stray = d / "chain" / "endpoint" / "report.json"
    stray.parent.mkdir(parents=True)
    stray.write_bytes((out / "endpoint" / "report.json").read_bytes())
    (d / "answer.json").write_text(json.dumps(CP.answer("accept", "")))
    (d / "piece.log").write_text("leak guard\n")
    assert CP.guard(d, None, tmp_path / "quarantine") == 3
    assert (tmp_path / "quarantine" / "files" / "chain" / "endpoint" / "report.json").is_file() and not stray.exists()
    assert sorted(p.relative_to(d).as_posix() for p in d.rglob("*") if p.is_file()) == sorted(
        ["answer.json", "piece.log", *published])
    assert not set().union(*(_numbers(p) & PRIVATE for p in d.rglob("*") if p.is_file()))


def test_n2_and_the_tie_take_every_link_window_of_a_short_run_unless_named():
    assert CP.sized(1, []) == ["CR_N2_DRAW=", "CR_TIE_WINDOWS=1"]
    assert CP.sized(8, ["CR_N2_DRAW=4"]) == ["CR_TIE_WINDOWS=8", "CR_N2_DRAW=4"]
    assert CP.sized(9, ["CR_SCAN_ONLY=1"]) == ["CR_SCAN_ONLY=1"]


def test_the_front_sends_one_group_of_the_class_and_keeps_the_fetch_in_the_serve_dirs_private(tmp_path, monkeypatch):
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin" / "ssh").write_text(STUB)
    (tmp_path / "bin" / "ssh").chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path / 'bin'}{os.pathsep}{os.defpath}")
    monkeypatch.setenv("STUB_DIR", str(tmp_path))
    serve, out = tmp_path / "serve", tmp_path / "out" / "certifier-front"
    assert CP.main(["front", "18190", str(serve), str(out), "--key", str(tmp_path / "key")]) == 0
    assert (tmp_path / "request").read_text() == (
        "warden-rehearse --peer vy-dc-h200 --port 18190 --minutes 7 --requests 4 --height 10 --scan --prompts "
        "--tokens 4 --window-per-group")
    prompts = json.loads((tmp_path / "prompts.json").read_text())
    assert [len(p) for p in prompts] == [CP.PROMPT_IDS] * CP.PROMPTS and all(t < 128000 for p in prompts for t in p)
    kept = serve / CP.PRIVATE / RUN / "fetch.tar.gz"
    assert [p.stat().st_mode & 0o777 for p in (serve / "private", serve / CP.PRIVATE, kept)] == [0o700, 0o700, 0o600]
    assert sorted(p.name for p in out.iterdir()) == ["bundle.tgz", "session.json", "start.json", "status.json"]
    assert json.loads((out / "session.json").read_text())["keep"] == str(kept)
    assert CP.main(["front", "18190", str(serve), str(out), "--tokens", "64"]) == 2
    assert "past one link-window" in json.loads((out / "session.json").read_text())["refused"]


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
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin" / "ssh").write_text(STUB)
    (tmp_path / "bin" / "ssh").chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path / 'bin'}{os.pathsep}{os.defpath}")
    monkeypatch.setenv("STUB_DIR", str(tmp_path))
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
    assert sorted(p.name for p in out.iterdir()) == ["bundle.tgz", "session.json", "start.json", "status.json",
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
