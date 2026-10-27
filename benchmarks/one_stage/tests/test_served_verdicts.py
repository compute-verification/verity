"""`served_verdicts.py`'s window pick, its subprotocols' pieces and its record, on toy serve dirs and answers:
the window it audits holds the most requests and none carried across windows; a subprotocol whose piece didn't run is
undecided, ``not run``, and every row is a core answer. The window's verdict is the Lean check's on the cases it has
(a stand-in for the executable, and its reading of the executable's line and exit status): a Lean rejection is the
window's, a rejection outside it rejects, and anything short of onsite's accept with every other subprotocol accepted
leaves the window undecided, with a block record that `block_record.check` holds and its schema admits; a rejected
window has none. The wipe composes only when its answer accepts on the record's server. PoUW's development piece checks
the window's gate session and credits nothing,
and the gate is refused while the lane has a period open. `served_verdicts.sh`'s window circuits gate on every class of
CLASSES, and the class of record's count after them gates nothing. Its load, `serve_load.py --varied`, writes each
request's send, first-token and done times beside its record, on an injected clock and a stand-in endpoint."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import itertools
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import jsonschema
import pytest
from verity.core.protocols import answer as CA
from verity.core.protocols.pous import wipe_period as W

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import served_verdicts as S
from verity_e2e_demo import block_record as BR

SCHEMA = json.loads((Path(BR.__file__).parent / "schemas" / "block-record.v1.schema.json").read_text())
_SL = importlib.util.spec_from_file_location("serve_load", HERE.parents[2] / "examples" / "vllm" / "serve_load.py")
SL = importlib.util.module_from_spec(_SL)
_SL.loader.exec_module(SL)


def _write(d: Path, name: str, rows: list[dict]) -> None:
    (d / name).write_text("".join(json.dumps(r) + "\n" for r in rows))


def _serve_dir(tmp_path: Path) -> Path:
    """Windows 0-3: w0 one request, w1 three, w2 one carried on into w3, w3 that one and its own; every seal good."""
    reqs = [("a", [0]), ("b", [1]), ("c", [1]), ("d", [1]), ("e", [2, 3]), ("f", [3])]
    by = {k: [u for u, ws in reqs if k in ws] for k in range(4)}
    _write(tmp_path, "seals.jsonl", [{"window": k, "requests": by[k], "row_dir": f"w{k}"} for k in range(4)])
    _write(tmp_path, "windows.jsonl", [{"window": k, "passed": True, "commit": {"steps": 5 + k}} for k in range(4)])
    _write(tmp_path, "requests.jsonl", [{"user": u, "engine_id": u, "windows": ws, "check": {"passed": True}} for u, ws in reqs])
    return tmp_path


def test_the_window_is_the_fullest_one_with_no_request_carried(tmp_path):
    got = S.pick(_serve_dir(tmp_path), slots=4)
    assert (got["window"], got["requests"], got["users"], got["steps"]) == (1, 3, ["b", "c", "d"], [5, 11])
    assert set(got["refused"]) == {"2", "3"} and "carried" in got["refused"]["3"]


def test_a_window_past_the_slots_or_named_and_refused_is_refused(tmp_path):
    d = _serve_dir(tmp_path)
    assert S.pick(d, slots=2)["window"] == 0
    with pytest.raises(ValueError, match="carried across windows"):
        S.pick(d, slots=4, want=3)
    with pytest.raises(ValueError, match="no seal"):
        S.pick(d, slots=4, want=9)


def test_a_failed_seal_or_commit_or_check_is_refused(tmp_path):
    d = _serve_dir(tmp_path)
    _write(d, "seals.jsonl", [{"window": 0, "requests": ["a"], "error": "boom"},
                              {"window": 1, "requests": ["b", "c", "d"]}])
    _write(d, "windows.jsonl", [{"window": 0, "passed": True, "commit": {"steps": 5}},
                                {"window": 1, "passed": False, "commit": {"steps": 6}}])
    with pytest.raises(ValueError, match="seal failed.*Commit did not pass|Commit did not pass.*seal failed"):
        S.pick(d, slots=4)


def test_a_subprotocol_whose_piece_did_not_run_is_undecided_not_run(tmp_path, capsys):
    got = S.pieces(tmp_path, 1, tmp_path / "pieces", env={})
    lines = capsys.readouterr().out.splitlines()
    assert [(g["verdict"], g["reason"]) for g in got.values()] == [("undecided", S.NOT_RUN)] * 4
    assert [line.split(" (")[0] for line in lines] == ["NOT RUN PoUW", "NOT RUN the bandwidth audit", "NOT RUN PoUS",
                                                       "NOT RUN the network certifier"]
    assert all(issue in "\n".join(lines) for issue in ("VLLM-33 G1", "VLLM-33 G2", "MEM-94", "NET-83", "INF-263"))
    for key, (_name, comp, *_rest) in S.PIECES.items():
        written = (tmp_path / "pieces" / comp / "answer.json").read_text()
        assert written == CA.Answer.undecided(S.PROTOCOLS[key], "not run").canonical() + "\n"
        row = S.piece_row(key, tmp_path / "pieces")
        assert (row["verdict"], row["reason"], row["issue"]) == ("undecided", "not run", S.PIECES[key][3])


def test_a_piece_that_ran_gives_its_answer_and_one_that_wrote_none_rejects(tmp_path, capsys):
    refusal = CA.Answer.reject(S.PROTOCOLS["space"], "no").canonical()
    env = {"POUS_CMD": f"echo '{refusal}' > \"$3/answer.json\"", "CERTIFIER_CMD": "true"}
    got = S.pieces(tmp_path, 1, tmp_path / "pieces", env=env)
    assert got["pous"] == {"verdict": "reject", "reason": "no", "rc": 0}
    assert got["certifier"]["verdict"] == "reject" and "wrote no answer" in got["certifier"]["reason"]
    assert (got["pouw"]["verdict"], got["bandwidth"]["verdict"]) == ("undecided", "undecided")
    rows = {k: S.piece_row(k, tmp_path / "pieces") for k in S.PIECES}
    assert (rows["space"]["verdict"], rows["space"]["reason"]) == ("reject", "no")
    assert rows["communication"]["answer"] == CA.Answer.reject("verity/certifier/v0", "its piece wrote no answer").to_json()


def test_a_piece_whose_answer_is_not_a_core_answer_rejects(tmp_path):
    (d := tmp_path / "pous").mkdir()
    (d / "answer.json").write_text(json.dumps({"verdict": "reject", "reason": "no"}))
    row = S.piece_row("space", tmp_path)
    assert row["verdict"] == "reject" and row["reason"].startswith("ValueError: not a core answer: answer: protocol missing")


def _accepted() -> dict:
    a = {"protocol": "verity/sampled-proofs/one-stage/v1", "verdict": "accept", "reason": None, "commitment": "ab" * 64,
         "public": {"bound": 2, "drawn": [0, 3], "per_template": None, "proved": [0, 3]},
         "profile": {"levels": [{"name": "unit", "size": 4}], "drawn": {"unit": 2}, "delta": "1/100"}}
    return S.row_of(S.INTEGRITY_OWNER, a, Path("/nonexistent"))


def _rows(integrity: dict, pieces: Path) -> dict:
    return {"integrity": integrity, **{k: S.piece_row(k, pieces) for k in S.PIECES}, "wipe": S.piece_row("wipe", pieces)}


def test_the_protocols_are_onsites_components_and_the_wipe():
    assert S.COMPONENTS == ("verity/certifier/v0", "verity/pous/p2/v4/run", "verity/pouw/v1",
                            "verity/remote-bandwidth/v0", "verity/sampled-proofs/one-stage/v1")
    assert S.PROTOCOLS["wipe"] == W.PROTOCOL


def test_sampled_proofs_alone_leaves_the_window_undecided_with_a_block_record(tmp_path):
    report, rec, line = S.record(_rows(_accepted(), tmp_path), window=1, steps=(5, 11), node="vy-dc-h200", run="r1")
    assert report["format"] == "verity/all-protocols-window/v2" and report["lean"]["components"] == []
    v = report["verdict"]
    assert (v["protocol"], v["verdict"], v["commitment"]) == ("verity/onsite/v0", "undecided", None)
    assert v["reason"].split("; ") == [
        "verity/certifier/v0 undecided: not run", "verity/pous/p2/v4/run undecided: not run",
        "verity/pouw/v1 undecided: not run", "verity/remote-bandwidth/v0 undecided: not run",
        "verity/sampled-proofs/one-stage/v1 accept: no case, so the Lean check didn't compose it",
        "verity/wipe/v0 undecided: not run"]
    assert CA.of_json(v).to_json() == v
    jsonschema.Draft202012Validator(SCHEMA).validate(rec)
    assert BR.check(rec) == {} and rec["extends"] == "verity/all-protocols-window/v2"
    assert rec["composed"]["onsite"] == {k: v[k] for k in ("protocol", "verdict", "reason")}
    assert rec["legs"]["integrity"]["quantity"]["value"] == 2 and rec["server_time"] == [{"server": "vy-dc-h200", "steps": [5, 11]}]
    assert (line["verdict"], line["record_missing"]) == ("undecided", ["bandwidth", "certifier", "pous", "pouw", "wipe"])
    assert line["not_run"] == {"PoUW": "VLLM-33 G1", "the bandwidth audit": "VLLM-33 G2", "PoUS": "MEM-94",
                               "the network certifier": "NET-83", "the wipe": "MEM-96"}
    work = report["protocols"]["work"]
    assert work["answer"] == CA.Answer.undecided("verity/pouw/v1", "not run").to_json() and work["why"] == S.PIECES["work"][4]


def test_an_audit_that_wrote_nothing_rejects_the_window_with_no_block_record(tmp_path):
    row = S.sampled_proofs_row(tmp_path)
    assert row["verdict"] == "reject" and "no summary" in row["reason"]
    report, rec, line = S.record(_rows(row, tmp_path), window=1, steps=(5, 11), node="n", run=None)
    assert rec is None and (line["verdict"], line["protocol"]) == ("reject", "verity/onsite/v0")
    assert report["verdict"]["reason"].startswith("verity/sampled-proofs/one-stage/v1 reject: RuntimeError: ")


CASE = {"commitment": "0a" * 4, "evidence": "0b" * 3, "params": {"k": 1}}


def _lean(answer: dict | Exception, seen: list):
    """A stand-in for `lean_check`'s executable that records what it was given and answers ``answer``."""
    def run(ids, params, commitment, evidence):
        seen.append((list(ids), params, commitment, evidence))
        if isinstance(answer, Exception):
            raise answer
        return answer
    return run


@pytest.mark.parametrize("lean", ["reject", "undecided", "accept"])
def test_the_lean_check_runs_on_the_components_with_a_case_and_its_rejection_is_the_windows(tmp_path, lean):
    """`record` composes the subprotocols whose case it has, by their ids, under the product's binding of their answers'
    commitments: a Lean rejection is the window's verdict as Lean wrote it, a Lean undecided leads the window's reason,
    and a Lean accept over fewer than onsite's five leaves the window undecided."""
    work = S.row_of("o", {"protocol": "verity/pouw/v1", "verdict": "accept", "reason": None, "commitment": "cd" * 64,
                          "public": {"work": {"credited": 3, "credited_tiles": 1}}, "profile": {"delta": "1/100"}}, None)
    rows = {**_rows(_accepted(), tmp_path), "work": work}
    ids = ["verity/pouw/v1", "verity/sampled-proofs/one-stage/v1"]
    answer = {"reject": CA.Answer.reject(S.PRODUCT, "verity/pouw/v1 reject: no"),
              "undecided": CA.Answer.undecided(S.PRODUCT, "verity/pouw/v1 undecided: later"),
              "accept": CA.Answer.accept(S.PRODUCT, b"\1" * 64, {}, {})}[lean].to_json()
    seen: list = []
    cases = {"integrity": (CASE, "a/case.json"), "work": (CASE, "b/case.json"), "space": (None, "no case.json")}
    report, rec, line = S.record(rows, window=1, steps=(0, 4), node="n", run=None, cases=cases, lean=_lean(answer, seen))
    (got_ids, params, commitment, evidence), = seen
    assert got_ids == ids and report["lean"]["components"] == ids and report["lean"]["answer"] == answer
    assert json.loads(params) == {"components": [{"id": i, "params": {"k": 1}} for i in ids]}
    assert json.loads(evidence) == {"components": [{**CASE, "id": i} for i in ids]}
    body = '{"components":[{"commitment":"' + "cd" * 64 + '","id":"verity/pouw/v1"},{"commitment":"' + "ab" * 64 + \
        '","id":"verity/sampled-proofs/one-stage/v1"}]}'
    assert commitment == hashlib.sha512(b"verity/product/v0\0" + body.encode()).digest()
    assert report["lean"]["cases"]["verity/pous/p2/v4/run"] == "no case.json"
    if lean == "reject":
        assert report["verdict"] == answer and rec is None and line["verdict"] == "reject"
        return
    reason = report["verdict"]["reason"].split("; ")
    assert report["verdict"]["verdict"] == "undecided" and report["verdict"]["protocol"] == "verity/onsite/v0"
    assert reason[0] == ("verity/pouw/v1 undecided: later" if lean == "undecided" else "verity/certifier/v0 undecided: not run")
    assert "verity/pous/p2/v4/run undecided: not run" in reason and rec is not None


def test_a_rejection_outside_the_lean_check_rejects_the_window_whatever_lean_answers(tmp_path):
    (d := tmp_path / "pous").mkdir()
    (d / "answer.json").write_text(CA.Answer.reject(S.PROTOCOLS["space"], "a read came back late").canonical())
    rows = _rows(_accepted(), tmp_path)
    accept = CA.Answer.accept(S.PRODUCT, b"\1" * 64, {}, {}).to_json()
    report, rec, line = S.record(rows, window=1, steps=(0, 4), node="n", run=None,
                                 cases={"integrity": (CASE, "c")}, lean=_lean(accept, []))
    assert rec is None and report["verdict"] == CA.Answer.reject(
        "verity/onsite/v0", "verity/pous/p2/v4/run reject: a read came back late").to_json()


def test_a_lean_check_that_fails_leaves_the_window_undecided_and_says_why(tmp_path):
    for lean, why in ((_lean(RuntimeError("boom"), []), "boom"), (None, "no Lean executable given"),
                      (_lean(CA.Answer.undecided(S.ONSITE, "x").to_json(), []), "answered as verity/onsite/v0, not "
                                                                                "verity/product/v0")):
        report, _rec, line = S.record(_rows(_accepted(), tmp_path), window=1, steps=(0, 4), node="n", run=None,
                                      cases={"integrity": (CASE, "c")}, lean=lean)
        assert report["lean"]["answer"] is None and why in report["lean"]["error"] and line["lean"]["error"]
        assert report["verdict"]["verdict"] == "undecided" and "the Lean check: " in report["verdict"]["reason"]


def _accepting(pid: str) -> dict:
    return {"protocol": pid, "answer": CA.Answer.accept(pid, b"\2" * 64, {}, {}).to_json()}


@pytest.mark.parametrize("wipe", ["accept", "undecided", "reject"])
def test_the_window_accepts_only_on_leans_onsite_accept_with_the_wipe_accepted(wipe):
    rows = {k: _accepting(pid) for k, pid in S.PROTOCOLS.items()}
    w = {"accept": CA.Answer.accept(W.PROTOCOL, b"\2" * 64, {}, {}), "undecided": CA.Answer.undecided(W.PROTOCOL, "not run"),
         "reject": CA.Answer.reject(W.PROTOCOL, "late")}[wipe]
    rows["wipe"] = {"protocol": W.PROTOCOL, "answer": w.to_json()}
    onsite = CA.Answer.accept(S.ONSITE, b"\3" * 64, {"verity/pouw/v1": {}}, {"verity/pouw/v1": {}}).to_json()
    composed = [k for k in rows if k != "wipe"]
    got = S.window_verdict(rows, composed, onsite)
    want = {"accept": onsite, "undecided": CA.Answer.undecided(S.ONSITE, "verity/wipe/v0 undecided: not run").to_json(),
            "reject": CA.Answer.reject(S.ONSITE, "verity/wipe/v0 reject: late").to_json()}[wipe]
    assert got == want
    assert S.window_verdict(rows, composed[1:], {**onsite, "protocol": S.PRODUCT})["verdict"] == (
        "reject" if wipe == "reject" else "undecided")


def _exe(tmp_path: Path, line: str, rc: int) -> Path:
    exe = tmp_path / "product"
    exe.write_text(f"#!/bin/sh\necho \"$1 $2\" > {tmp_path}/argv\ncat <<'EOF'\n{line}\nEOF\nexit {rc}\n")
    exe.chmod(0o755)
    return exe


def test_lean_check_reads_the_executables_one_answer_line_and_its_exit_status(tmp_path):
    a = CA.Answer.reject(S.PRODUCT, "verity/pouw/v1 reject: no")
    run = S.lean_check(_exe(tmp_path, a.canonical(), 1))
    assert run(["verity/pouw/v1", "verity/wipe/v0"], b"{}", b"", b"{}") == a.to_json()
    assert (tmp_path / "argv").read_text() == "verity/pouw/v1,verity/wipe/v0 check\n"
    for line, rc, why in ((a.canonical(), 0, "exited 0 on"), (json.dumps(a.to_json()), 1, "exited 1 on"),
                          ("{}", 1, "printed no answer"), ("usage", 64, "exited 64")):
        with pytest.raises(RuntimeError, match=why):
            S.lean_check(_exe(tmp_path, line, rc))([], b"", b"", b"")
    with pytest.raises(RuntimeError, match="no Lean executable at"):
        S.lean_check(tmp_path / "none")([], b"", b"", b"")


def test_read_case_takes_only_a_case(tmp_path):
    f = tmp_path / "case.json"
    assert S.read_case(f) == (None, "no case.json")
    f.write_text(json.dumps(CASE))
    assert S.read_case(f) == (CASE, str(f))
    for bad in ({**CASE, "x": 1}, {**CASE, "commitment": "AB"}, {**CASE, "evidence": 1}, {**CASE, "params": {"x": 0.5}}):
        f.write_text(json.dumps(bad))
        got, why = S.read_case(f)
        assert got is None and why.startswith("case.json is not a case: ")


WIPE_ACCEPT = {"protocol": W.PROTOCOL, "verdict": "accept", "reason": None, "commitment": "cd" * 64,
               "public": {"certified": True, "egress_mib": 0, "eps": str(W.EPS), "foreign_bits": 4096, "k": 5500,
                          "memory_bits": 1 << 40, "r": "1/2"},
               "profile": {"delta": str(W.EPS)}}
WIPE_REJECT = {"protocol": W.PROTOCOL, "verdict": "reject", "reason": "a read came back late", "commitment": "cd" * 64,
               "public": None, "profile": None}


@pytest.mark.parametrize("case", ["accept", "elsewhere", "unserved", "refused", "reject"])
def test_the_record_composes_the_wipe_only_when_it_accepted_on_the_records_server(tmp_path, monkeypatch, capsys, case):
    """`record` takes the wipe out of ``record_missing`` only when its answer accepts and its served.json names the
    record's node: served on another server it stays missing (``elsewhere``), an accepting answer with no served.json
    names no server and stays missing (``unserved``), a wipe that wrote no answer (served_wipe.sh refused) is undecided,
    not run, and one whose answer rejects rejects the window with no block record."""
    monkeypatch.setenv("RESEARCH_RUN_ID", "r1")
    monkeypatch.setattr(S, "sampled_proofs_row", lambda _audit_dir, _run=None: _accepted())
    wipe = tmp_path / "pieces" / "wipe"
    wipe.mkdir(parents=True)
    if case == "refused":
        (wipe / "refused.txt").write_text("WIPE REFUSED: the card has a compute process\n")
    else:
        (wipe / "answer.json").write_text(json.dumps(WIPE_REJECT if case == "reject" else WIPE_ACCEPT))
        if case != "unserved":
            (wipe / "served.json").write_text(json.dumps({"server": "vy-dc-h200-2" if case == "elsewhere" else "vy-dc-h200"}))
    (tmp_path / "serve").mkdir()
    rc = S.main(["record", "--serve-run", str(_serve_dir(tmp_path / "serve")), "--window", "1",
                 "--sampled-proofs", str(tmp_path / "audit"), "--pieces", str(tmp_path / "pieces"),
                 "--out", str(tmp_path / "record"), "--node", "vy-dc-h200", "--lean", str(tmp_path / "no-lean")])
    line = json.loads(capsys.readouterr().out.strip().splitlines()[-1].removeprefix("RECORD "))
    report = json.loads((tmp_path / "record" / "verdicts.json").read_text())
    assert report["verdict"]["verdict"] == line["verdict"] and line["lean"] == {"components": [], "verdict": None, "error": None}
    if case == "reject":
        assert rc == 1 and (line["verdict"], line["block_record"]) == ("reject", None)
        assert line["reason"] == "verity/wipe/v0 reject: a read came back late"
        assert not (tmp_path / "record" / "block-record.json").exists()
        return
    rec = json.loads((tmp_path / "record" / "block-record.json").read_text())
    jsonschema.Draft202012Validator(SCHEMA).validate(rec)
    assert rc == 0 and line["check"] == {} and line["verdict"] == "undecided"
    if case == "accept":
        assert line["record_missing"] == ["bandwidth", "certifier", "pous", "pouw"]
        assert rec["legs"]["wipe"]["server"] == "vy-dc-h200" and rec["legs"]["wipe"]["quantity"]["value"] == 4096
        assert rec["legs"]["wipe"]["runs"] == ["r1"] and rec["composed"]["legs"]["wipe"] == "accept"
        return
    assert line["record_missing"] == ["bandwidth", "certifier", "pous", "pouw", "wipe"]
    if case == "elsewhere":
        assert rec["legs"]["wipe"]["server"] == "vy-dc-h200-2" and rec["composed"]["elsewhere"] == ["wipe"]
    elif case == "unserved":
        assert rec["legs"]["wipe"] is None and "the wipe" not in line["not_run"]
    else:
        assert rec["legs"]["wipe"] is None and line["not_run"]["the wipe"] == "MEM-96"
        assert report["protocols"]["wipe"]["answer"] == CA.Answer.undecided(W.PROTOCOL, "not run").to_json()


def test_a_case_without_a_built_lean_check_fails_the_record(tmp_path, monkeypatch, capsys):
    """A case the record can't take to the Lean check (no executable at ``--lean``) leaves the window undecided, saying
    why, and the record exits 1."""
    monkeypatch.setattr(S, "sampled_proofs_row", lambda _audit_dir, _run=None: _accepted())
    (tmp_path / "audit" / "window").mkdir(parents=True)
    (tmp_path / "audit" / "window" / "case.json").write_text(json.dumps(CASE))
    (tmp_path / "serve").mkdir()
    rc = S.main(["record", "--serve-run", str(_serve_dir(tmp_path / "serve")), "--window", "1",
                 "--sampled-proofs", str(tmp_path / "audit"), "--pieces", str(tmp_path / "pieces"),
                 "--out", str(tmp_path / "record"), "--node", "n", "--lean", str(tmp_path / "no-lean")])
    line = json.loads(capsys.readouterr().out.strip().splitlines()[-1].removeprefix("RECORD "))
    assert rc == 1 and line["verdict"] == "undecided" and "no Lean executable at" in line["lean"]["error"]
    assert line["lean"]["components"] == ["verity/sampled-proofs/one-stage/v1"]


SID = "s20261026T221500Z-ab12"
GATE = "ssh -i KEY -o BatchMode=yes research@gate"


def _h1_serve_dir(tmp_path: Path, coin_source: str = "challenger", tied: bool = True) -> Path:
    """A serve dir whose window 1 served H1 (its line's ``pouw_h1`` and its record) on a gate session."""
    d = tmp_path / "serve"
    (d / "h1" / "w1").mkdir(parents=True)
    _write(d, "windows.jsonl", [{"window": 1, "passed": True, "pouw_h1": {
        "window": 1, "tied": tied, "steps_served": 4, "coin_source": coin_source, "roots1": {"reason": None},
        "gate": {"gate_passed": {"rows": 80}}, "session": {"session": SID, "end": {"closed": True, "complete": True}}}}])
    (d / S.H1_CALLS.format(k=1)).write_text(json.dumps({"commitment": {"session": {"session": SID}}}))
    return d


def _gate(calls: list):
    """The gate's answers as `served_verdicts._gate_out` gets them: the session's tree for ``fetch``, the ledger and pins."""
    def out(gate: str, *words: str) -> bytes:
        calls.append((gate, words))
        if words[0] == "fetch":
            buf = io.BytesIO()
            with tarfile.open(fileobj=buf, mode="w:gz") as t:
                for name, body in ((f"{words[1]}/request.json", b"{}"), (f"{words[1]}/out/record.json", b"{}")):
                    info = tarfile.TarInfo(name)
                    info.size = len(body)
                    t.addfile(info, io.BytesIO(body))
            return buf.getvalue()
        if words[0] == "ledger":
            return b'{"kind": "reserve", "session": "%s", "body": {"by": "vllm"}}\n' % SID.encode()
        return b'{"commit": "aa", "main": "bb", "pinned_utc": "2026-10-26T21:35:00Z"}\n'
    return out


def test_a_development_pouw_piece_checks_the_windows_session_and_credits_nothing(tmp_path, monkeypatch, capsys):
    """``pouw``: the window's session fetched from the gate with the ledger and pins, held by the challenger check; it
    accepts, its answer is undecided with its document beside it (it credits nothing), the window is undecided, and the
    pieces' line says it is a development run's."""
    calls, seen = [], []
    monkeypatch.setattr(S, "_gate_out", _gate(calls))

    def check(calls_file, tree, ledger, pins, main, key, out):
        seen.append((tree, ledger.read_text(), pins.read_text(), main, key))
        assert (tree / "out" / "record.json").is_file()
        return {"ok": True, "reason": None, "manifest": None, "rounds": 16,
                "credit": {"window": ["budget", "the window is in no period"], "manifest": None}}

    d = _h1_serve_dir(tmp_path)
    doc = S.pouw(d, 1, tmp_path / "pieces" / "pouw", gate=GATE, key="ff" * 64, main=tmp_path / "main.txt", check=check)
    assert (doc["verdict"], doc["reasons"], doc["credited"]) == ("accept", [], False)
    assert doc["credit"]["window"] == ["budget", "the window is in no period"] and doc["check"]["rounds"] == 16
    assert [w for _g, w in calls] == [("fetch", SID), ("ledger",), ("pins",)] and all(g == GATE for g, _w in calls)
    (tree, ledger, pins, main, key), = seen
    assert tree == tmp_path / "pieces" / "pouw" / "session" / SID and SID in ledger and '"commit"' in pins
    assert (main, key) == (tmp_path / "main.txt", "ff" * 64)
    row = S.piece_row("work", tmp_path / "pieces")
    assert (row["verdict"], row["reason"], row["why"], row["development"]["verdict"]) == (
        "undecided", S.DEV_UNDECIDED, S.DEV_WHY, "accept")
    _report, rec, line = S.record(_rows(_accepted(), tmp_path / "pieces"), window=1, steps=(0, 4), node="n", run="r")
    assert line["verdict"] == "undecided" and "pouw" in line["record_missing"] and line["development"] == {"PoUW": "accept"}
    assert "verity/pouw/v1 undecided: " + S.DEV_UNDECIDED in line["reason"]
    assert BR.check(rec) == {}
    env = {"POUW_CMD": f"cp {tmp_path}/pieces/pouw/development.json \"$3/\""}
    got = S.pieces(d, 1, tmp_path / "again", env=env)
    out = capsys.readouterr().out
    assert "POUW (development run, credits nothing) " in out and got["pouw"]["verdict"] == "accept"
    assert (got["pouw"]["session"], got["pouw"]["coins"], got["pouw"]["credited"]) == (SID, None, False)


@pytest.mark.parametrize("case", ["coins", "untied", "in-process", "no gate"])
def test_a_development_pouw_piece_rejects_what_its_checks_refuse(tmp_path, monkeypatch, case):
    """Coins the challenger check refuses, H1 tables untied to the Commit, coins held in the serving process, or no gate
    to check the session by: the piece rejects, naming why, and so does the record."""
    monkeypatch.setattr(S, "_gate_out", _gate([]))
    d = _h1_serve_dir(tmp_path, "in-process" if case == "in-process" else "challenger", tied=case != "untied")
    refused = {"ok": False, "reason": "session s1 is not the one the ledger opened", "manifest": None, "credit": {}}
    ok = {"ok": True, "reason": None, "manifest": None, "credit": {}}
    doc = S.pouw(d, 1, tmp_path / "pieces" / "pouw", gate=None if case == "no gate" else GATE, key="ff" * 64, main=None,
                 check=lambda *a: refused if case == "coins" else ok)
    why = {"coins": "coins: session s1 is not the one the ledger opened", "untied": "ROOTS1:",
           "in-process": "coins: in-process, not the auditor host's challenger's", "no gate": "coins: ValueError: no gate"}
    assert doc["verdict"] == "reject" and doc["reasons"][0].startswith(why[case]), doc["reasons"]
    row = S.piece_row("work", tmp_path / "pieces")
    assert row["verdict"] == "reject" and row["reason"].startswith(why[case])
    _report, rec, line = S.record(_rows(_accepted(), tmp_path / "pieces"), window=1, steps=(0, 4), node="n", run=None)
    assert rec is None and (line["verdict"], line["development"]) == ("reject", {"PoUW": "reject"})
    assert line["reason"].startswith("verity/pouw/v1 reject: " + why[case])


def test_the_gate_is_refused_while_the_lane_has_a_period_open(tmp_path, monkeypatch):
    """``gate``: a reserve joins the lane's open period, so a lane with one open is refused before serving; the line
    names the gate's latest pin and whether it is on the verifier's first-parent line of main."""
    ledger = [{"kind": "period-open", "session": None, "body": {"period": "p1", "lane": "vllm"}}]
    answers = {"ledger": ledger, "pins": [{"commit": "aa" * 20, "main": "bb" * 20, "pinned_utc": "2026-10-26T21:35:00Z"}]}
    monkeypatch.setattr(S, "_gate_out", lambda gate, verb: "".join(json.dumps(e) + "\n" for e in answers[verb]).encode())
    (main := tmp_path / "main.txt").write_text("cc" * 20 + "\n" + "aa" * 20 + "\n")
    ok, line = S.gate_ready(GATE, "vllm", main)
    assert not ok and line.startswith("GATE refused: lane vllm's open period p1 (a reserve would join it)")
    assert "on the verifier's first-parent line of main" in line and "NOT" not in line
    answers["ledger"].append({"kind": "period-close", "session": None, "body": {"period": "p1"}})
    main.write_text("cc" * 20 + "\n")
    ok, line = S.gate_ready(GATE, "vllm", main)
    assert ok and line.startswith("GATE ready: lane vllm's open period None;") and "NOT on the verifier's" in line


FAKE_PROVE_PATH = '''import json, sys
a = sys.argv[1:]
assert a[0] == "window-circuit"
c = a[a.index("--class") + 1] if "--class" in a else "record"
answer = json.load(open("answers.json"))[c]
if answer != "count":
    print("prove_path.py window-circuit: refused at", c, file=sys.stderr)
    sys.exit({"refuse": 2, "expire": 124}[answer])
open(a[a.index("--out") + 1], "w").write("{}")
print(json.dumps({"bits": 772, "count": {"free": 3, "inputs": 10, "registered": 7, "units": 1}}))
'''


def _circuits(tmp_path: Path, classes: str, answers: dict) -> tuple[int, list[str], list[str]]:
    """`served_verdicts.sh circuits 0` in a toy tree whose prove_path.py counts a class, refuses it, or exits 124 as
    `timeout` does when a count runs out of time (``answers``, by class), under a `timeout` that records the limit it got:
    the exit status, the verdict lines, and the limits in order."""
    tree, bin_, venv = tmp_path / "tree", tmp_path / "bin", tmp_path / "venv" / "bin"
    for d in (tree / "benchmarks" / "prove_path", bin_, venv, tmp_path / "serve"):
        d.mkdir(parents=True)
    (tree / "benchmarks" / "prove_path" / "prove_path.py").write_text(FAKE_PROVE_PATH)
    (tree / "answers.json").write_text(json.dumps(answers))
    (tmp_path / "serve" / "weights_registration.json").write_text("{}")
    for path, body in ((venv / "python", f'exec {sys.executable} "$@"'),
                       (bin_ / "timeout", f'echo "$1" >> {tmp_path}/limits; shift; exec "$@"')):
        path.write_text(f"#!/bin/sh\n{body}\n")
        path.chmod(0o755)
    env = {"PATH": f"{bin_}:{os.defpath}", "HOME": str(tmp_path), "VENV": str(venv.parent),
           "D": str(tmp_path / "serve"), "OUT": str(tmp_path / "out"), "CLASSES": classes}
    rc = subprocess.run(["bash", str(HERE.parent / "served_verdicts.sh"), "circuits", "0"], cwd=tree, env=env,
                        check=False).returncode
    return rc, (tmp_path / "out" / "verdicts.txt").read_text().splitlines(), (tmp_path / "limits").read_text().split()


def test_the_class_of_record_is_measured_after_the_gating_classes_and_its_timeout_fails_nothing(tmp_path):
    rc, lines, limits = _circuits(tmp_path, "4,32,4", {"4,32,4": "count", "record": "expire"})
    assert rc == 0 and limits == ["30m", "2h"], (rc, lines)
    assert lines[0].startswith("WINDOW-CIRCUIT class 4,32,4: registered 7 bits, free 3 bits of 10 input bits at the "
                               "weights registration's pins; ")
    assert lines[1:] == ["WINDOW-CIRCUIT class record (measurement) timed out after WC_RECORD_TIMEOUT=2h (exit 124): no count"]


@pytest.mark.parametrize("classes, answers, refused", [
    ("4,32,4 8,64,4", {"4,32,4": "count", "8,64,4": "refuse", "record": "count"}, "WINDOW-CIRCUIT class 8,64,4 refused (exit 2): "),
    ("4,32,4", {"4,32,4": "expire", "record": "count"}, "WINDOW-CIRCUIT class 4,32,4 timed out after WC_TIMEOUT=30m"),
    ("record", {"record": "expire"}, "WINDOW-CIRCUIT class record timed out after WC_TIMEOUT=30m"),
])
def test_any_class_of_classes_that_does_not_count_fails_the_run(tmp_path, classes, answers, refused):
    """Each class of CLASSES gates, not just one of them, whatever the class of record's measurement says; CLASSES that
    names `record` gates on it under WC_TIMEOUT and doesn't measure it again."""
    rc, lines, limits = _circuits(tmp_path, classes, answers)
    assert rc != 0 and sum(line.startswith(refused) for line in lines) == 1, lines
    assert any("prove_path.py window-circuit: refused at" in line for line in lines) == (answers.get("8,64,4") == "refuse")
    measured = [line for line in lines if line.startswith("WINDOW-CIRCUIT class record (measurement)")]
    assert len(measured) == (classes != "record") and limits[-1] == ("30m" if classes == "record" else "2h")


class _Response:
    """`urlopen`'s response, read whole or line by line; reading line i sets ``now[0]`` to ``times[i]``, and reading past the
    last line to ``end``."""

    def __init__(self, lines: list[bytes], now: list[float] | None = None, times: list[float] = (), end: float = 0.0) -> None:
        self.lines, self.now, self.times, self.end = lines, now, list(times), end

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def read(self) -> bytes:
        return b"".join(self.lines)

    def __iter__(self):
        for i, line in enumerate(self.lines):
            if self.now is not None:
                self.now[0] = self.times[i]
            yield line
        if self.now is not None:
            self.now[0] = self.end


def _sse(*events: dict | bytes) -> list[bytes]:
    """serve-openai's stream of ``events``, as `urlopen`'s response yields it: each event's data line, then its blank line."""
    return [x for e in events for x in (b"data: " + (e if isinstance(e, bytes) else json.dumps(e).encode()) + b"\n", b"\n")]


def test_serve_load_times_a_request_at_its_send_its_first_output_id_and_its_end(monkeypatch):
    """A streamed request's first token is the first event carrying output ids, not the stream's opening event, and its end
    comes once the stream is read past [DONE]; its ids and finish are what they were. An unstreamed one has no first token."""
    now = [1.0]
    lines = _sse({"choices": [{"text": "", "token_ids": None}]}, {"choices": [{"token_ids": [7]}]},
                 {"choices": [{"token_ids": [8], "finish_reason": "length"}]}, b"[DONE]")
    stream = _Response(lines, now, times=[2, 2, 3, 3, 4, 4, 5, 5], end=6.0)
    monkeypatch.setattr(SL.urllib.request, "urlopen", lambda req, timeout=None: stream)
    times: dict = {}
    assert SL.post(0, {"stream": True}, times=times, clock=lambda: now[0]) == {"token_ids": [7, 8], "finish_reason": "length"}
    assert times == {"send": 1.0, "first_token": 3.0, "done": 6.0}
    whole = _Response([json.dumps({"choices": [{"token_ids": [7]}]}).encode()])
    monkeypatch.setattr(SL.urllib.request, "urlopen", lambda req, timeout=None: whole)
    times = {}
    assert SL.post(0, {}, times=times, clock=lambda: now[0]) == {"choices": [{"token_ids": [7]}]} and set(times) == {"send", "done"}


def test_serve_load_varied_writes_each_requests_timings_beside_its_record(tmp_path, monkeypatch):
    """A varied run's timings file beside its record: its release run and wall-clock start, and one entry a request in
    release order, its times on the injected clock from the run's start (send, then first token, then done) and its
    streamed ids counted; the record's lines are what they were."""
    ticks = itertools.count()

    def clock() -> float:
        return float(next(ticks))

    def urlopen(req, timeout=None):
        b = json.loads(req.data)
        if req.full_url.endswith("/tokenize"):
            return _Response([json.dumps({"tokens": [128000] + [1000 + ord(c) for c in b["prompt"]]}).encode()])
        ids = list(range(500, 500 + b["max_tokens"]))
        return _Response(_sse({"choices": [{"text": "", "token_ids": None}]},
                              *({"choices": [{"token_ids": [t], "finish_reason": "length" if t == ids[-1] else None}]} for t in ids),
                              b"[DONE]"))

    monkeypatch.setattr(SL.urllib.request, "urlopen", urlopen)
    out = tmp_path / "load.jsonl"
    a = argparse.Namespace(port=0, model="m", varied=3, rate=2.0, prompt_range="16:24", max_tokens=4, seed=1, release="run",
                           jitter_ms=0.0, out=str(out))
    assert SL.run_varied(a, clock=clock, wall=lambda: 1.7e9, sleep=lambda s: None) == 0
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert SL.timings_path(out) == tmp_path / "load.timings.json"
    doc = json.loads(SL.timings_path(out).read_text())
    assert set(doc) == {"clock", "release_run", "start_unix", "requests"} and (doc["release_run"], doc["start_unix"]) == ("run", 1.7e9)
    assert [set(r) for r in rows] == [{"user", "release", "at_s", "delay_s", "prompt", "max_tokens", "response"}] * 3
    assert [(t["user"], t["release"], t["at_s"]) for t in doc["requests"]] == [(r["user"], r["release"], r["at_s"]) for r in rows]
    assert [r["release"] for r in rows] == [0, 1, 2]
    for t, r in zip(doc["requests"], rows):
        assert set(t) == {"user", "release", "at_s", "send_s", "first_token_s", "done_s", "tokens", "ok"} and t["ok"]
        assert 0 < t["send_s"] < t["first_token_s"] < t["done_s"]
        assert r["response"] == {"token_ids": list(range(500, 500 + r["max_tokens"])), "finish_reason": "length"}
        assert t["tokens"] == r["max_tokens"]
