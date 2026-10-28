"""`served_verdicts.py`'s window pick, its subprotocols' pieces and its record, on toy serve dirs and answers:
the window it audits holds the most requests and none carried across windows; a subprotocol whose piece didn't run is
undecided, ``not run``, naming the command it lacks, and so is one that refused to run, naming why; one that ran out of
its limit, wrote no answer, or left one that isn't a core answer errs, undecided; every row is a core answer. The
window's verdict is onsite's or remote's Lean verdict through `composite.py` (a stand-in for each product's executable)
on the pieces' private cases, onsite's when they make it up, else remote's, else undecided naming what each lacks, with
a block record that `block_record.check` holds and its schema admits, composing each subprotocol accepted on its node
and carrying one about another server as elsewhere; a rejected window has none, and a rejecting row fails the record,
with none, whatever the window's verdict. Of the composite only its public record is published: no byte of a case's
evidence reaches the record, whatever the executable prints. An accepted answer's clause says what it holds beyond its
verdict. The wipe composes only when its answer accepts on the record's server. PoUW's development piece checks the
window's gate session and credits nothing, and the gate is refused while the lane has a period open. The period the
served windows join opens, through infra's own gate, on the Program serve-openai declared and the circuit files built
for it, as core's credit reads them, and is refused on files of anything else. `served_verdicts.sh`'s window circuits
gate on every class of CLASSES, and the class of record's count after them gates nothing; its endpoint gets one token
budget, T, with or without H1. Its load, `serve_load.py --varied`, writes each request's send, first-token and done
times beside its record, on an injected clock and a stand-in endpoint."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import itertools
import json
import os
import shlex
import subprocess
import sys
import tarfile
from datetime import UTC, datetime
from fractions import Fraction
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


def test_a_subprotocol_whose_piece_did_not_run_is_undecided_not_run_naming_the_command_it_lacks(tmp_path, capsys):
    got = S.pieces(tmp_path, 1, tmp_path / "pieces", env={})
    lines = capsys.readouterr().out.splitlines()
    assert [(g["verdict"], g["reason"]) for g in got.values()] == [("undecided", S.NOT_RUN)] * 4
    assert [line.split(" (")[0] for line in lines] == ["NOT RUN PoUW", "NOT RUN PoUS", "NOT RUN the bandwidth audit",
                                                       "NOT RUN the network certifier"]
    for line, (cmd, owner) in zip(lines, (("POUW_CMD", "compute-accounting"), ("POUS_CMD", "memory-accounting"),
                                          ("BANDWIDTH_CMD", "memory-accounting"), ("CERTIFIER_CMD", "network-accounting"))):
        assert f"({cmd} unset; {owner}): " in line
    for key, (_name, comp, _owner, what) in S.PIECES.items():
        written = (tmp_path / "pieces" / comp / "answer.json").read_text()
        assert written == CA.Answer.undecided(S.PROTOCOLS[key], "not run").canonical() + "\n"
        row = S.piece_row(key, tmp_path / "pieces")
        assert (row["verdict"], row["reason"], row["why"]) == ("undecided", "not run", f"{comp.upper()}_CMD unset: {what}")


def _reject(protocol: str) -> dict:
    return {"protocol": protocol, "verdict": "reject", "reason": "no", "commitment": None, "public": None, "profile": None}


def test_a_piece_that_ran_gives_its_answer_and_one_that_wrote_none_errs(tmp_path, capsys):
    refusal = CA.Answer.reject(S.PROTOCOLS["space"], "no").canonical()
    env = {"POUS_CMD": f"echo '{refusal}' > \"$3/answer.json\"", "CERTIFIER_CMD": "true",
           "BANDWIDTH_CMD": 'echo \'{"verdict": "reject", "reason": "no"}\' > "$3/answer.json"'}
    got = S.pieces(tmp_path, 1, tmp_path / "pieces", env=env, node="vy-dc-h200")
    assert got["pous"] == {"verdict": "reject", "reason": "no", "rc": 0}
    assert (got["certifier"]["verdict"], got["certifier"]["error"]) == ("undecided", True)
    assert got["certifier"]["reason"] == "its piece ran (exit 0) and wrote no answer; its log is certifier/piece.log"
    assert (got["bandwidth"]["verdict"], got["bandwidth"]["error"]) == ("undecided", True)
    assert "not a core answer" in got["bandwidth"]["reason"]
    assert got["pouw"]["verdict"] == "undecided" and "error" not in got["pouw"]
    assert json.loads((tmp_path / "pieces" / "pous" / S.RAN).read_text()) == {"server": "vy-dc-h200", "rc": 0,
                                                                              "limit": S.PIECE_TIMEOUT}
    rows = {k: S.piece_row(k, tmp_path / "pieces") for k in S.PIECES}
    assert {k: (r["verdict"], r.get("error", False)) for k, r in rows.items()} == {
        "work": ("undecided", False), "bandwidth": ("undecided", True), "space": ("reject", False),
        "communication": ("undecided", True)}
    assert rows["communication"]["answer"] == CA.Answer.undecided("verity/certifier/v0",
                                                                  got["certifier"]["reason"]).to_json()


def test_a_piece_that_refused_to_run_is_undecided_not_run_not_rejected(tmp_path, capsys, monkeypatch):
    """A piece's ``refused.txt`` (a rehearsal the gate's check didn't pass, a step that can't run) with no answer: its
    piece didn't run, so the subprotocol is undecided, "not run", naming why, and nothing rejects or errs."""
    why = "the gate's rehearsal didn't pass, so it certified no served answer"
    env = {"CERTIFIER_CMD": f"echo \"{why}\" > \"$3/refused.txt\"; exit 3"}
    got = S.pieces(tmp_path, 1, tmp_path / "pieces", env=env)
    assert got["certifier"] == {"verdict": "undecided", "reason": S.NOT_RUN, "rc": 3,
                                "why": f"its piece refused to run: {why}"}
    assert "NOT RUN the network certifier {" in capsys.readouterr().out
    row = S.piece_row("communication", tmp_path / "pieces")
    assert (row["verdict"], row["reason"], row["why"]) == ("undecided", S.NOT_RUN, f"its piece refused to run: {why}")
    assert "error" not in row and row["answer"] == CA.Answer.undecided("verity/certifier/v0", S.NOT_RUN).to_json()
    monkeypatch.setattr(S, "sampled_proofs_row", lambda _audit_dir, _run=None: _accepted())
    (tmp_path / "serve").mkdir()
    rc = S.main(["record", "--serve-run", str(_serve_dir(tmp_path / "serve")), "--window", "1",
                 "--sampled-proofs", str(tmp_path / "audit"), "--pieces", str(tmp_path / "pieces"),
                 "--out", str(tmp_path / "record"), "--node", "n"])
    line = json.loads(capsys.readouterr().out.strip().splitlines()[-1].removeprefix("RECORD "))
    assert rc == 0 and (line["verdict"], line["rejected"], line["errors"]) == ("undecided", [], [])
    assert line["not_run"]["the network certifier"] == f"its piece refused to run: {why}"
    assert "certifier" in line["record_missing"]


def test_a_piece_whose_answer_is_not_a_core_answer_errs(tmp_path):
    (d := tmp_path / "pous").mkdir()
    (d / "answer.json").write_text(json.dumps({"verdict": "reject", "reason": "no"}))
    row = S.piece_row("space", tmp_path)
    assert (row["verdict"], row["error"]) == ("undecided", True)
    assert row["reason"].startswith("its answer isn't a core answer: ValueError: not a core answer: answer: protocol missing")


def test_a_piece_runs_within_its_limit_and_one_out_of_time_or_refused_says_so(tmp_path):
    """Each piece runs under `timeout` at ``<COMPONENT>_TIMEOUT``, else ``PIECE_TIMEOUT``: one that runs out of it (exit
    124, which the fake `timeout` gives for the limit ``0``) errs naming the limit, and one that writes ``refused.txt``
    instead of an answer is undecided, not run, with the refusal as why."""
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    (bin_ / "timeout").write_text(f'#!/bin/sh\necho "$3" >> {tmp_path}/limits\n[ "$3" = 0 ] && exit 124\nshift 3\nexec "$@"\n')
    (bin_ / "timeout").chmod(0o755)
    env = {"PATH": f"{bin_}:{os.defpath}", "PIECE_TIMEOUT": "7m", "BANDWIDTH_TIMEOUT": "0", "BANDWIDTH_CMD": "true",
           "POUS_CMD": 'echo "PoUS REFUSED: the card has a compute process" > "$3/refused.txt"; exit 3'}
    got = S.pieces(tmp_path, 1, tmp_path / "pieces", env=env, node="n")
    assert (tmp_path / "limits").read_text().split() == ["7m", "0"]
    assert got["bandwidth"] == {"verdict": "undecided", "rc": 124, "error": True,
                                "reason": "its piece ran (exit 124, out of time at 0) and wrote no answer; "
                                          "its log is bandwidth/piece.log"}
    assert got["pous"] == {"verdict": "undecided", "reason": S.NOT_RUN, "rc": 3,
                           "why": "its piece refused to run: PoUS REFUSED: the card has a compute process"}


def test_a_piece_that_left_a_file_that_is_not_json_errs_about_no_server_and_the_next_piece_still_runs(tmp_path):
    env = {"BANDWIDTH_CMD": f"echo '{json.dumps(_reject('verity/remote-bandwidth/v0'))}' > \"$3/answer.json\"; "
                            "echo '{' > \"$3/served.json\"",
           "POUS_CMD": f"echo '{json.dumps(_reject('verity/pous/p2/v4/run'))}' > \"$3/answer.json\""}
    got = S.pieces(tmp_path, 1, tmp_path / "pieces", env=env, node="n")
    assert (got["bandwidth"]["verdict"], got["bandwidth"]["error"]) == ("undecided", True)
    assert got["bandwidth"]["reason"].startswith("its piece left a file that isn't JSON: JSONDecodeError")
    assert got["pous"] == {"verdict": "reject", "reason": "no", "rc": 0}
    row, at = S.piece("bandwidth", tmp_path / "pieces")
    assert (row["verdict"], row["owner"], row["error"], at) == ("undecided", "memory-accounting", True, (None, "real"))


def _accepted() -> dict:
    a = {"protocol": "verity/sampled-proofs/one-stage/v1", "verdict": "accept", "reason": None, "commitment": "ab" * 64,
         "public": {"bound": 2, "drawn": [0, 3], "per_template": None, "proved": [0, 3]},
         "profile": {"levels": [{"name": "unit", "size": 4}], "drawn": {"unit": 2}, "delta": "1/100"}}
    return S.row_of(S.INTEGRITY_OWNER, a, Path("/nonexistent"))


def _rows(integrity: dict, pieces: Path) -> dict:
    return {"integrity": integrity, **{k: S.piece_row(k, pieces) for k in S.PIECES}, "wipe": S.piece_row("wipe", pieces)}


V2, POUW, POUW_PERIOD = "verity/sampled-proofs/one-stage/v2", "verity/pouw/v1", "verity/pouw/period/v1"
IO = "verity/inference-only/v0"


def test_the_products_need_their_lean_components():
    """`NEEDS` is `Core.Onsite.components` and `Core.Remote.components`: the wipe is onsite's, and the inference-only
    policy is both's."""
    assert S.NEEDS == {"onsite": ("verity/certifier/v0", IO, "verity/pous/p2/v4/run", POUW_PERIOD,
                                  "verity/remote-bandwidth/v0", V2, W.PROTOCOL),
                       "remote": (IO, POUW, V2)}
    assert S.PROTOCOLS["wipe"] == W.PROTOCOL and set(S.LEAN) == {"onsite", "remote"}


def _piece(root: Path, name: str, pid: str, verdict: str = "undecided", case: bool = True,
           evidence: bytes = b"{}") -> Path:
    """A case directory ``root/name`` as a piece keeps it privately (0700, its files 0600): its own core answer on
    ``pid`` and, with ``case``, its case, blobs one file."""
    d = root / name
    d.mkdir(parents=True, exist_ok=True, mode=0o700)
    digest = hashlib.sha512(pid.encode()).digest()
    a = {"accept": CA.Answer.accept(pid, digest, {"n": 1}, {"delta": "1/100"}),
         "reject": CA.Answer.reject(pid, "no", digest), "undecided": CA.Answer.undecided(pid, "later", digest)}[verdict]
    (d / "answer.json").write_text(a.canonical() + "\n")
    if case:
        (d / "blob.bin").write_bytes(pid.encode())
        (d / "case.json").write_text(json.dumps({"commitment": "0a" * 4, "evidence": evidence.hex(), "params": {"k": 1},
                                                 "blobs": [str(d / "blob.bin")]}))
    for f in d.iterdir():
        f.chmod(0o600)
    return d


def _exe(tmp_path: Path, name: str, answer: CA.Answer | str, rc: int | None = None, stderr: str = "") -> Path:
    """A stand-in for a product's executable: records its arguments, prints ``answer``'s line (and ``stderr``) and exits
    with its verdict's status (or ``rc``)."""
    line = answer if isinstance(answer, str) else answer.canonical()
    rc = S.EXIT[answer.verdict] if rc is None else rc
    exe = tmp_path / "bin" / name
    exe.parent.mkdir(exist_ok=True)
    exe.write_text(f"#!{sys.executable}\nimport json, sys\nopen({str(tmp_path / (name + '.argv'))!r}, 'w')"
                   f".write(json.dumps(sys.argv[1:]))\nprint({line!r})\nprint({stderr!r}, file=sys.stderr)\n"
                   f"sys.exit({rc})\n")
    exe.chmod(0o755)
    return exe


PRIV = Path("serve") / "private"


def _record(tmp_path: Path, rows: dict, cases=(), servers=None, **lean: Path):
    return S.record(rows, window=1, steps=(5, 11), node="vy-dc-h200", run="r1",
                    cases=[(d.name, d) for d in cases], private=tmp_path / PRIV / "composite" / "w1",
                    out=tmp_path / "record", lean={p: lean.get(p, tmp_path / "no-lean") for p in S.LEAN}, servers=servers)


def test_pieces_that_make_up_neither_product_leave_the_window_undecided_naming_what_each_lacks(tmp_path):
    S.pieces(tmp_path, 1, tmp_path / "pieces", env={})
    rows = _rows(_accepted(), tmp_path / "pieces")
    report, rec, line = S.record(rows, window=1, steps=(5, 11), node="vy-dc-h200", run="r1",
                                 cases=S.case_dirs(tmp_path, 1, tmp_path / "pieces"),
                                 private=tmp_path / PRIV / "composite" / "w1", out=tmp_path / "record")
    assert report["format"] == "verity/all-protocols-window/v2" and report["lean"]["product"] is None
    v = report["verdict"]
    assert (v["protocol"], v["verdict"], v["commitment"]) == ("verity/onsite/v0", "undecided", None)
    onsite, remote = v["reason"].split("; verity/remote/v0 lacks ")
    assert onsite.startswith("verity/onsite/v0 lacks verity/certifier/v0 (undecided; certifier holds no case.json), "
                             f"{IO} (no piece answers it), ")
    assert f"{V2} (no piece answers it), {W.PROTOCOL} (no piece answers it)" in onsite
    assert remote == f"{IO} (no piece answers it), {POUW} (undecided; pouw holds no case.json), {V2} (no piece answers it)"
    assert report["lean"]["missing"]["verity/remote/v0"] == {IO: "no piece answers it",
                                                             POUW: "undecided; pouw holds no case.json",
                                                             V2: "no piece answers it"}
    assert CA.of_json(v).to_json() == v and not (tmp_path / "record" / "composite").exists()
    assert not (tmp_path / PRIV / "composite").exists()
    jsonschema.Draft202012Validator(SCHEMA).validate(rec)
    assert BR.check(rec) == {} and rec["extends"] == "verity/all-protocols-window/v2"
    assert rec["composed"]["onsite"] == {k: v[k] for k in ("protocol", "verdict", "reason")}
    assert rec["legs"]["integrity"]["quantity"]["value"] == 2 and rec["server_time"] == [{"server": "vy-dc-h200", "steps": [5, 11]}]
    assert (line["verdict"], line["record_missing"]) == ("undecided", ["bandwidth", "certifier", "pous", "pouw", "wipe"])
    assert line["lean"] == {"product": None, "verdict": None, "error": None,
                            "missing": {"verity/onsite/v0": sorted(S.NEEDS["onsite"]), "verity/remote/v0": [IO, POUW, V2]}}
    assert {name: why.split(" unset: ")[0] for name, why in line["not_run"].items()} == {
        "PoUW": "POUW_CMD", "the bandwidth audit": "BANDWIDTH_CMD", "PoUS": "POUS_CMD", "the network certifier": "CERTIFIER_CMD",
        "the wipe": "WIPE_CMD"}
    assert (line["rejected"], line["errors"], line["elsewhere"]) == ([], [], {}) and "unserved" not in line
    work = report["protocols"]["work"]
    assert work["answer"] == CA.Answer.undecided(POUW, "not run").to_json()
    assert work["why"] == f"POUW_CMD unset: {S.PIECES['work'][3]}"


def test_an_accepted_pous_answer_says_when_its_weights_were_held_their_share_and_where_its_quantity_is_not_established(
        tmp_path):
    """PoUS's answer in the shape its served piece writes: weights reloaded on the card after the window's serve, 64.9% of
    the checkpoint, at ε = 1/100 + 2^-128, which the block record establishes at neither of its δs. Its accept's clause in
    the report's ``said`` says each."""
    eps = Fraction(1, 100) + Fraction(1, 2 ** 128)
    held = "loaded from the served snapshot on the card the window served on, after its serve ended"
    (d := tmp_path / "pous").mkdir()
    (d / "answer.json").write_text(json.dumps({
        "protocol": "verity/pous/p2/v4/run", "verdict": "accept", "reason": None, "commitment": "ef" * 64,
        "public": {"certified": True, "eps": str(eps), "s_bits": 8169621180},
        "profile": {"weights": {"held": held, "share": "237568/366225", "covers": "linears"},
                    "outside": {"bytes": {"the embedding and the head": 525336576, "the norms": 135168,
                                          "the per-row scales": 1507328}}}}))
    (d / S.RAN).write_text(json.dumps({"server": "vy-dc-h200", "rc": 0, "limit": "60m"}))
    report, rec, _line = _record(tmp_path, _rows(_accepted(), tmp_path),
                                 servers={k: S.piece_server(k, tmp_path) for k in [*S.PIECES, "wipe"]})
    assert rec["legs"]["pous"]["at"] == {d: {"holds": False, "why": f"the run established it except with {eps}, more than {d}"}
                                         for d in ("1/100", "1/1000000")}
    clause = report["said"].split("; PoUS: ", 1)[1]
    assert clause.startswith(f"accept (its weights {held}, not concurrently with the block; they are 64.9% of the "
                             "checkpoint (linears); the embedding and the head, the norms, the per-row scales are outside "
                             "its claim; its quantity is not established at δ = 1/100 or 1/1000000: the run established "
                             "it except with 1/100 + 2^-128)")
    assert report["said"].startswith("sampled proofs: accept; ")


def test_a_rejecting_row_no_product_composes_leaves_the_window_undecided_and_is_named(tmp_path):
    """No rejection is combined in Python: the window waits for a product, and the row's rejection is the line's, with
    no block record."""
    row = S.sampled_proofs_row(tmp_path)
    assert row["verdict"] == "reject" and "no summary" in row["reason"]
    report, rec, line = _record(tmp_path, _rows(row, tmp_path / "pieces"))
    assert (line["verdict"], line["protocol"], line["rejected"]) == ("undecided", "verity/onsite/v0", ["sampled proofs"])
    assert rec is None and line["block_record"] is None


@pytest.mark.parametrize("verdict", ["accept", "reject", "undecided"])
def test_cases_that_make_up_remote_get_remotes_lean_verdict_through_composite(tmp_path, verdict):
    """`composite.py remote` on its three private cases (one answering for neither product rides along unused): the
    composed case goes to the composite's private directory, the window's verdict is the executable's public outcome
    (its verdict and refusal's name), and only the public record is published, with the product's stated exceptions;
    a rejection leaves no block record."""
    priv = tmp_path / PRIV
    io, pw, v2 = _piece(priv, "inference-only", IO), _piece(priv, "pouw", POUW), _piece(priv, "sampled-proofs", V2)
    pous = _piece(priv, "pous", "verity/pous/p2/v4/run", case=False)
    want = {"accept": CA.Answer.accept("verity/remote/v0", b"\1" * 64, {POUW: {}}, {POUW: {}}),
            "reject": CA.Answer.reject("verity/remote/v0", f"{POUW} reject: no", b"\1" * 64),
            "undecided": CA.Answer.undecided("verity/remote/v0", f"{V2} undecided: later", b"\1" * 64)}[verdict]
    report, rec, line = _record(tmp_path, _rows(_accepted(), tmp_path / "pieces"), [io, pw, v2, pous],
                                remote=_exe(tmp_path, "remote", want))
    name = {"accept": None, "reject": f"{POUW} reject", "undecided": f"{V2} undecided"}[verdict]
    assert report["verdict"] == {"protocol": "verity/remote/v0", "verdict": verdict, "reason": name}
    assert report["lean"]["error"] is None and report["lean"]["product"] == "verity/remote/v0"
    assert report["lean"]["cases"] == [IO, POUW, V2]
    out = tmp_path / PRIV / "composite" / "w1" / "out"
    argv = json.loads((tmp_path / "remote.argv").read_text())
    assert argv == ["check", *(str(out / f) for f in ("params.json", "commitment.bin", "evidence.json")),
                    str(io / "blob.bin"), str(pw / "blob.bin"), str(v2 / "blob.bin")]
    assert [c["id"] for c in json.loads((out / "params.json").read_text())["components"]] == [IO, POUW, V2]
    body = '{"components":[' + ",".join('{"commitment":"' + hashlib.sha512(i.encode()).hexdigest() + f'","id":"{i}"}}'
                                        for i in (IO, POUW, V2)) + "]}"
    commitment = hashlib.sha512(b"verity/remote/v0\0" + body.encode()).digest()
    assert (out / "commitment.bin").read_bytes() == commitment
    assert (out / "answer.json").read_text().strip() == want.canonical()
    assert {f.stat().st_mode & 0o777 for f in out.rglob("*") if f.is_file()} == {0o600}
    assert (out.stat().st_mode & 0o777, out.parent.stat().st_mode & 0o777) == (0o700, 0o700)
    pub = tmp_path / "record" / "composite"
    assert sorted(f.name for f in pub.iterdir()) == ["composite.json", "verdicts.txt"]
    assert report["lean"]["composite"] == json.loads((pub / "composite.json").read_text())
    assert report["lean"]["composite"]["commitment"] == commitment.hex()
    assert report["lean"]["composite"]["exceptions"] == S.C.exceptions("remote") != []
    assert [x for x in (pub / "verdicts.txt").read_text().splitlines() if x.startswith("VERDICT ")] == [
        " ".join(["VERDICT", "verity/remote/v0", verdict, *([name] if name else [])])]
    assert (line["lean"]["product"], line["lean"]["verdict"], line["rejected"]) == ("verity/remote/v0", verdict, [])
    assert (rec is None) == (verdict == "reject")


def test_cases_that_make_up_onsite_get_onsites_verdict(tmp_path):
    priv = tmp_path / PRIV
    cases = [_piece(priv, i.split("/")[1], i) for i in S.NEEDS["onsite"]]
    cases += [_piece(priv, "pouw-window", POUW)]
    want = CA.Answer.undecided("verity/onsite/v0", f"{V2} undecided: later", b"\2" * 128)
    report, _rec, line = _record(tmp_path, _rows(_accepted(), tmp_path / "pieces"), cases,
                                 onsite=_exe(tmp_path, "onsite", want),
                                 remote=_exe(tmp_path, "remote", CA.Answer.reject("verity/remote/v0", "x")))
    assert report["verdict"] == {"protocol": "verity/onsite/v0", "verdict": "undecided", "reason": f"{V2} undecided"}
    assert line["lean"]["product"] == "verity/onsite/v0"
    assert not (tmp_path / "remote.argv").exists() and line["lean"]["missing"]["verity/onsite/v0"] == []
    out = tmp_path / PRIV / "composite" / "w1" / "out"
    params = json.loads((out / "params.json").read_text())
    assert [c["id"] for c in params["components"]] == list(S.NEEDS["onsite"])
    assert report["lean"]["cases"] == sorted([*S.NEEDS["onsite"], POUW])


def test_a_lean_check_that_cannot_answer_leaves_the_window_undecided_says_why_and_publishes_nothing(tmp_path):
    """No executable, an exit status that isn't its verdict's, no answer, or a public record composite.py refuses as a
    leak (here a case's evidence holding its params digest): the window is undecided, the reason public (none of the
    executable's or composite.py's output but composite.py's own refusal), and nothing of the composite is published."""
    priv = tmp_path / PRIV
    io, pw = _piece(priv, "inference-only", IO), _piece(priv, "pouw", POUW)
    digest = hashlib.sha512(b'{"k":1}').hexdigest().encode()
    v2 = _piece(priv, "sampled-proofs", V2, evidence=b'{"x":"' + digest + b'"}')
    a = CA.Answer.reject("verity/remote/v0", "no")
    for exe, cases, why in ((tmp_path / "none", [io, pw, _piece(priv, "v2", V2)], "no remote executable at"),
                            (_exe(tmp_path, "r0", a, rc=0), [io, pw, priv / "v2"], "exited 0 on Lean's reject"),
                            (_exe(tmp_path, "r1", "usage", rc=64, stderr="SECRET"), [io, pw, priv / "v2"],
                             "exited 64: no Lean verdict"),
                            (_exe(tmp_path, "r2", a), [io, pw, v2], "exited 65 with public output []: composite.py: the "
                             f"public output holds bytes of {V2}'s evidence; nothing written to --public-out")):
        report, _rec, line = _record(tmp_path, _rows(_accepted(), tmp_path / "pieces"), cases, remote=exe)
        assert report["lean"]["composite"] is None and why in report["lean"]["error"] and line["lean"]["error"]
        assert "SECRET" not in json.dumps(report) and not (tmp_path / "record" / "composite").exists()
        assert (report["verdict"]["protocol"], report["verdict"]["verdict"]) == ("verity/remote/v0", "undecided")
        assert report["verdict"]["reason"].startswith("the Lean check: ")


def test_found_takes_a_case_only_with_its_answer_and_once(tmp_path):
    dirs = [_piece(tmp_path, "a", POUW), _piece(tmp_path, "b", POUW), _piece(tmp_path, "c", V2, case=False),
            _piece(tmp_path, "d", W.PROTOCOL, "accept"), tmp_path / "e", tmp_path / "none"]
    (tmp_path / "e").mkdir()
    (tmp_path / "e" / "answer.json").write_text('{"verdict": "accept"}')
    (tmp_path / "e" / "case.json").write_text("{}")
    got, why = S.found([(d.name, d) for d in dirs])
    assert got == {W.PROTOCOL: tmp_path / "d"}
    assert why == {POUW: "answered by more than one piece: a, b", V2: "undecided; c holds no case.json",
                   "e": "its answer.json is no core answer (Refused)"}
    assert S.found([]) == ({}, {})


def test_each_pieces_case_is_its_private_one_else_its_public_one(tmp_path):
    """PoUS's case is in the private directory its pous.json names (else private/pous/w<K>), the wipe's in
    private/wipe; a piece with no private case keeps its case in its public directory (the certifier's); ``--cases``
    adds directories, or their subdirectories, as composite.py reads them."""
    serve, pieces = tmp_path / "serve", tmp_path / "pieces"
    elsewhere = _piece(tmp_path / "kept", "pous-w1", "verity/pous/p2/v4/run")
    _piece(pieces, "pous", "verity/pous/p2/v4/run", case=False)
    (pieces / "pous" / "pous.json").write_text(json.dumps({"private": str(elsewhere)}))
    wipe = _piece(serve / "private", "wipe", W.PROTOCOL)
    _piece(pieces, "wipe", W.PROTOCOL, case=False)
    certifier = _piece(pieces, "certifier", "verity/certifier/v0")
    _piece(serve / "private", "certifier", "verity/certifier/v0", case=False)
    extra = tmp_path / "extra"
    io, v2 = _piece(extra, "inference-only", IO), _piece(extra, "v2", V2)
    got = dict(S.case_dirs(serve, 1, pieces, [extra]))
    assert got == {"pouw": pieces / "pouw", "bandwidth": pieces / "bandwidth", "pous": elsewhere, "certifier": certifier,
                   "wipe": wipe, "inference-only": io, "v2": v2}
    (pieces / "pous" / "pous.json").unlink()
    assert dict(S.case_dirs(serve, 1, pieces))["pous"] == pieces / "pous"
    default = _piece(serve / "private" / "pous", "w1", "verity/pous/p2/v4/run")
    assert dict(S.case_dirs(serve, 1, pieces))["pous"] == default
    assert S.found(S.case_dirs(serve, 1, pieces, [extra]))[0] == {
        "verity/pous/p2/v4/run": default, W.PROTOCOL: wipe, "verity/certifier/v0": certifier, IO: io,
        V2: v2}


def _windows(b: bytes) -> set[bytes]:
    return {b[i:i + 64] for i in range(len(b) - 63)}


def test_no_byte_of_a_cases_evidence_reaches_a_public_path(tmp_path, monkeypatch, capsys):
    """A served record whose cases make up remote, each case's evidence random bytes, and an executable that prints the
    evidence back in its answer's reason and on its stderr: no 64-byte run of any case's evidence, raw or hex, is in any
    file under the run's public out (the record and the pieces' public directories) or on stdout, while the composed
    case and every answer stay in the serve dir's private directory (0700, files 0600)."""
    monkeypatch.setattr(S, "sampled_proofs_row", lambda _audit_dir, _run=None: _accepted())
    serve, out = tmp_path / "serve", tmp_path / "out"
    serve.mkdir()
    _serve_dir(serve)
    evidence = {pid: os.urandom(4096) for pid in (POUW, V2)}
    pw = _piece(serve / S.PRIVATE, "pouw", POUW, "accept", evidence=evidence[POUW])
    v2 = _piece(tmp_path / "audit-private", "v2", V2, "accept", evidence=evidence[V2])
    _piece(tmp_path / "audit-private", "inference-only", IO, "accept")
    (out / "pieces" / "pouw").mkdir(parents=True)
    (out / "pieces" / "pouw" / "answer.json").write_text(CA.Answer.accept(POUW, os.urandom(64), {"n": 1},
                                                                           {"delta": "1/100"}).canonical() + "\n")
    leak = evidence[POUW].hex() + evidence[V2].hex()
    exe = _exe(tmp_path, "remote", CA.Answer.reject("verity/remote/v0", f"params: {leak}").canonical(), rc=1,
               stderr=leak)
    rc = S.main(["record", "--serve-run", str(serve), "--window", "1", "--sampled-proofs", str(tmp_path / "audit"),
                 "--pieces", str(out / "pieces"), "--out", str(out / "record"), "--node", "n", "--remote", str(exe),
                 "--onsite", str(tmp_path / "no-lean"), "--cases", str(v2.parent)])
    stdout = capsys.readouterr().out
    report = json.loads((out / "record" / "verdicts.json").read_text())
    assert report["verdict"] == {"protocol": "verity/remote/v0", "verdict": "reject", "reason": "params"}
    assert rc == 1 and report["lean"]["error"] is None and (out / "record" / "composite" / "composite.json").is_file()
    public = [f for f in out.rglob("*") if f.is_file()]
    assert {f.relative_to(out).as_posix() for f in public} >= {"record/verdicts.json", "record/composite/composite.json",
                                                               "record/composite/verdicts.txt"}
    runs = {pid: _windows(e) | _windows(e.hex().encode()) for pid, e in evidence.items()}
    for where, b in [*((f.relative_to(out).as_posix(), f.read_bytes()) for f in public), ("stdout", stdout.encode())]:
        leaked = sorted(pid for pid, r in runs.items() if not r.isdisjoint(_windows(b)))
        assert not leaked, f"{where} holds bytes of the evidence of {leaked}"
    private = serve / "private" / "composite" / "w1"
    assert leak in (private / "out" / "answer.json").read_text() and leak in (private / "out" / "check.err").read_text()
    assert evidence[POUW].hex() in (private / "out" / "evidence.json").read_text()
    assert {f.stat().st_mode & 0o777 for f in private.rglob("*") if f.is_file() and "public" not in f.parts} == {0o600}
    assert {d.stat().st_mode & 0o777 for d in (private, private / "out", private / "out" / "pieces")} == {0o700}
    assert out not in private.parents and pw.parent == serve / "private"


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
    not run, and one whose answer rejects fails the record, with no block record, while the window, which no product
    composed, stays undecided."""
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
                 "--out", str(tmp_path / "record"), "--node", "vy-dc-h200", "--onsite", str(tmp_path / "no-lean"),
                 "--remote", str(tmp_path / "no-lean")])
    line = json.loads(capsys.readouterr().out.strip().splitlines()[-1].removeprefix("RECORD "))
    report = json.loads((tmp_path / "record" / "verdicts.json").read_text())
    assert report["verdict"]["verdict"] == line["verdict"] == "undecided" and line["lean"]["product"] is None
    if case == "reject":
        assert rc == 1 and line["rejected"] == ["the wipe"] and line["block_record"] is None
        assert not (tmp_path / "record" / "block-record.json").exists()
        return
    rec = json.loads((tmp_path / "record" / "block-record.json").read_text())
    jsonschema.Draft202012Validator(SCHEMA).validate(rec)
    assert rc == 0 and line["check"] == {}
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
        assert rec["legs"]["wipe"] is None and line["errors"] == []
        assert line["not_run"]["the wipe"] == "its piece refused to run: WIPE REFUSED: the card has a compute process"
        assert report["protocols"]["wipe"]["answer"] == CA.Answer.undecided(W.PROTOCOL, "not run").to_json()


def test_pieces_without_a_built_product_fail_the_record(tmp_path, monkeypatch, capsys):
    """Cases that make up remote (PoUW's in its private directory, the inference-only policy's and sampled proofs' a
    ``--cases``) with no executable at ``--remote`` leave the window undecided, saying why, and the record exits 1."""
    monkeypatch.setattr(S, "sampled_proofs_row", lambda _audit_dir, _run=None: _accepted())
    _piece(tmp_path / PRIV, "pouw", POUW)
    _piece(tmp_path / "kept", "sampled-proofs", V2)
    _piece(tmp_path / "kept", "inference-only", IO)
    rc = S.main(["record", "--serve-run", str(_serve_dir(tmp_path / "serve")), "--window", "1",
                 "--sampled-proofs", str(tmp_path / "audit"), "--pieces", str(tmp_path / "pieces"),
                 "--out", str(tmp_path / "record"), "--node", "n", "--remote", str(tmp_path / "no-lean"),
                 "--cases", str(tmp_path / "kept")])
    line = json.loads(capsys.readouterr().out.strip().splitlines()[-1].removeprefix("RECORD "))
    assert rc == 1 and line["verdict"] == "undecided" and "no remote executable at" in line["lean"]["error"]
    assert line["lean"]["product"] == "verity/remote/v0"


#: an accepted core answer of each piece's subprotocol, its public outputs the block record's leg certificate
ACCEPTS = {
    "pouw": {"protocol": "verity/pouw/v1", "public": {"work": {"level": 3, "n": 4, "credited_tiles": 2, "credited": 8192}},
             "profile": {"delta": "1/1000000"}},
    "bandwidth": {"protocol": "verity/remote-bandwidth/v0", "public": {"eps": "1/1000000", "q_bytes": 6963200}, "profile": {}},
    "pous": {"protocol": "verity/pous/p2/v4/run", "public": {"certified": True, "eps": "1/1000000", "s_bits": 1 << 30},
             "profile": {}},
    "certifier": {"protocol": "verity/certifier/v0", "public": {"K": 15764, "eps": "1/100", "windows": 1}, "profile": {}},
    "wipe": {k: WIPE_ACCEPT[k] for k in ("protocol", "public", "profile")},
}
COMPONENTS = ("pouw", "bandwidth", "pous", "certifier", "wipe")


@pytest.mark.parametrize("case, piece", [("accept", None)] + [
    (case, piece) for case in ("not run", "reject", "elsewhere", "malformed") for piece in COMPONENTS])
def test_the_block_record_composes_each_subprotocol_accepted_on_its_node_and_names_each_that_was_not(
        tmp_path, monkeypatch, capsys, case, piece):
    """Every piece's answer accepted on the record's node (each piece's `RAN`, which a `SERVED` naming no server leaves it
    about, and the wipe's `SERVED`): the block record composes every leg, and the report's ``said`` names each
    subprotocol's accept; the window, whose pieces left no case, stays undecided. One piece with no answer is undecided,
    not run, naming why, with no leg; one that rejects fails the record with no block record; one that accepts with
    outputs that are not its leg's certificate errs, undecided, with no leg; one about another server is carried and
    named as elsewhere."""
    monkeypatch.setenv("RESEARCH_RUN_ID", "r1")
    monkeypatch.setattr(S, "sampled_proofs_row", lambda _audit_dir, _run=None: _accepted())
    pieces = tmp_path / "pieces"
    for comp in COMPONENTS:
        (d := pieces / comp).mkdir(parents=True)
        if comp == piece and case == "not run":
            continue
        a = {"verdict": "accept", "reason": None, "commitment": "ef" * 64, **ACCEPTS[comp]}
        if comp == piece and case == "reject":
            a = {**a, "verdict": "reject", "reason": "a read came back late", "public": None, "profile": None}
        if comp == piece and case == "malformed":
            a = {**a, "public": {"unrelated": 1}}
        (d / "answer.json").write_text(json.dumps(a))
        server = "vy-dc-h200-2" if comp == piece and case == "elsewhere" else "vy-dc-h200"
        if comp == "wipe" or server != "vy-dc-h200":
            (d / S.SERVED).write_text(json.dumps({"server": server}))
        elif comp == "bandwidth":
            (d / S.SERVED).write_text(json.dumps({"served": {"window": 1}, "scope": "beside the served window"}))
        if comp != "wipe":
            (d / S.RAN).write_text(json.dumps({"server": "vy-dc-h200", "rc": 0, "limit": "60m"}))
    (tmp_path / "serve").mkdir()
    rc = S.main(["record", "--serve-run", str(_serve_dir(tmp_path / "serve")), "--window", "1",
                 "--sampled-proofs", str(tmp_path / "audit"), "--pieces", str(pieces), "--out", str(tmp_path / "record"),
                 "--node", "vy-dc-h200"])
    line = json.loads(capsys.readouterr().out.strip().splitlines()[-1].removeprefix("RECORD "))
    report = json.loads((tmp_path / "record" / "verdicts.json").read_text())
    leg, runner = piece, {v[1]: k for k, v in S.PIECES.items()} | {"wipe": "wipe"}
    name = S.NAMES[runner[piece]] if piece else None
    assert (line["verdict"], line["lean"]["product"]) == ("undecided", None)
    if case == "reject":
        assert rc == 1 and (line["rejected"], line["block_record"]) == ([name], None)
        assert line["verdicts"][name] == "reject" and not (tmp_path / "record" / "block-record.json").exists()
        return
    rec = json.loads((tmp_path / "record" / "block-record.json").read_text())
    jsonschema.Draft202012Validator(SCHEMA).validate(rec)
    assert line["check"] == {} == BR.check(rec) and line["rejected"] == []
    assert rec["composed"]["onsite"] == {k: report["verdict"][k] for k in ("protocol", "verdict", "reason")}
    if case == "accept":
        assert rc == 0 and (line["record_missing"], line["not_run"], line["elsewhere"], line["errors"]) == ([], {}, {}, [])
        assert rec["composed"]["legs"] == {k: "accept" for k in BR.LEGS} and rec["composed"]["verdict"] == "accept"
        late = {"the network certifier": " (its quantity is not established at δ = 1/1000000: the run established it "
                                         "except with 1/100)"}
        assert report["said"] == "; ".join(f"{n}: accept{late.get(n, '')}" for n in S.NAMES.values())
        return
    assert rec["composed"]["verdict"] == "pending" and line["record_missing"] == [leg]
    if case == "not run":
        what = S.WIPE[3] if piece == "wipe" else S.PIECES[runner[piece]][3]
        assert rc == 0 and line["not_run"] == {name: f"{piece.upper()}_CMD unset: {what}"}
        assert rec["legs"][leg] is None and line["elsewhere"] == {}
    elif case == "malformed":
        assert rc == 1 and line["errors"] == [name] and line["verdicts"][name] == "undecided"
        row = next(r for k, r in report["protocols"].items() if S.NAMES[k] == name)
        assert row["reason"].startswith(f"its answer is not the block record's {leg} leg: KeyError")
        assert row["answer"]["verdict"] == "undecided" and rec["legs"][leg] is None
    else:
        assert rc == 0 and line["not_run"] == {} and line["elsewhere"] == {name: "vy-dc-h200-2"}
        assert rec["legs"][leg]["server"] == "vy-dc-h200-2" and rec["composed"]["elsewhere"] == [leg]


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

    def check(calls_file, tree, ledger, pins, main, key, out, inputs):
        assert inputs == {}, "a serve dir with no declared Program or period gives credit nothing more"
        seen.append((tree, ledger.read_text(), pins.read_text(), main, key))
        assert (tree / "out" / "record.json").is_file()
        return {"ok": True, "reason": None, "manifest": None, "rounds": 16,
                "credit": {"window": ["budget", "the window is in no period"], "manifest": None}}

    d = _h1_serve_dir(tmp_path)
    doc = S.pouw(d, 1, tmp_path / "pieces" / "pouw", gate=GATE, key="ff" * 64, main=tmp_path / "main.txt", check=check)
    assert (doc["verdict"], doc["reasons"], doc["credited"]) == ("accept", [], False)
    assert doc["credit"]["window"] == ["budget", "the window is in no period"] and doc["check"]["rounds"] == 16
    assert doc["SaltIsLiveCoin"] == {"met": True, "why": S.SALT_MET}
    assert doc["AcceptedLive"] == {"met": False, "why": "credit refused the window (budget: the window is in no "
                                                       "period), so it credited no proof"}
    assert [w for _g, w in calls] == [("fetch", SID), ("ledger",), ("pins",)] and all(g == GATE for g, _w in calls)
    (tree, ledger, pins, main, key), = seen
    assert tree == tmp_path / "pieces" / "pouw" / "session" / SID and SID in ledger and '"commit"' in pins
    assert (main, key) == (tmp_path / "main.txt", "ff" * 64)
    row = S.piece_row("work", tmp_path / "pieces")
    why = f"{S.DEV_WHY}; core's credit refuses the window (budget) the window is in no period"
    assert (row["verdict"], row["reason"], row["why"], row["development"]["verdict"]) == (
        "undecided", S.DEV_UNDECIDED, why, "accept")
    report, rec, line = _record(tmp_path, _rows(_accepted(), tmp_path / "pieces"))
    assert line["verdict"] == "undecided" and "pouw" in line["record_missing"] and line["development"] == {"PoUW": "accept"}
    assert line["verdicts"]["PoUW"] == S.DEV_VERDICT == "development run, credits nothing" and "PoUW" not in line["not_run"]
    jsonschema.Draft202012Validator(SCHEMA).validate(rec)
    assert BR.check(rec) == {} and rec["composed"]["onsite"] == {k: report["verdict"][k] for k in ("protocol", "verdict", "reason")}
    assert report["said"].startswith(f"sampled proofs: accept; PoUW: {why}; ")
    for name, cmd in (("the bandwidth audit", "BANDWIDTH"), ("PoUS", "POUS"), ("the network certifier", "CERTIFIER"),
                      ("the wipe", "WIPE")):
        assert f"; {name}: not run: {cmd}_CMD unset: " in report["said"]
    env = {"POUW_CMD": f"cp {tmp_path}/pieces/pouw/development.json \"$3/\""}
    got = S.pieces(d, 1, tmp_path / "again", env=env)
    out = capsys.readouterr().out
    assert "POUW (development run, credits nothing) " in out and got["pouw"]["verdict"] == "accept"
    assert (got["pouw"]["session"], got["pouw"]["coins"], got["pouw"]["credited"]) == (SID, None, False)
    assert {p: got["pouw"][p]["met"] for p in S.PREMISES} == {"AcceptedLive": False, "SaltIsLiveCoin": True}


def test_the_pouw_piece_gives_credit_the_declared_program_the_period_and_the_proofs_it_has(tmp_path, monkeypatch):
    """With the serve dir's declared Program (``h1/window-program.json.gz``) and its period (``period.json``), and the
    drawn statements and live evidence given, the challenger check gets each as its ``--check-challenger`` flag, and the
    piece's document carries them and credit's calls."""
    monkeypatch.setattr(S, "_gate_out", _gate([]))
    d = _h1_serve_dir(tmp_path)
    (d / "h1" / "window-program.json.gz").write_bytes(b"")
    (d / S.PERIOD).write_text(json.dumps({"name": "served-r1"}))
    st, lv = tmp_path / "statements.json", tmp_path / "live.json"
    got = []

    def check(*a):
        got.append(a[-1])
        return {"ok": True, "reason": None, "manifest": None, "credit": {"window": None, "calls": [3], "credit": [3]}}

    doc = S.pouw(d, 1, tmp_path / "pieces" / "pouw", gate=GATE, key="ff" * 64, main=None, check=check, statements=st,
                 live=lv)
    want = {"--descriptor": str(d / "h1" / "window-program.json.gz"), "--period": "served-r1", "--statements": str(st),
            "--live-evidence": str(lv)}
    assert got == [want] and doc["credit_inputs"] == want
    assert (doc["credit"]["window"], doc["credit"]["calls"], doc["credit"]["credit"]) == (None, [3], [3])
    assert (doc["AcceptedLive"], doc["SaltIsLiveCoin"]) == ({"met": True, "why": S.LIVE_MET},
                                                            {"met": True, "why": S.SALT_MET})


def test_a_record_that_isnt_one_manifest_under_its_salt_coin_doesnt_meet_salt_is_live_coin(tmp_path, monkeypatch):
    """Coins the challenger check accepts with a record that isn't one manifest under them: ``SaltIsLiveCoin`` isn't
    met, and names the manifest's refusal."""
    monkeypatch.setattr(S, "_gate_out", _gate([]))
    got = {"ok": True, "reason": None, "manifest": "round 2's tables are not under the salt", "credit": {}}
    doc = S.pouw(_h1_serve_dir(tmp_path), 1, tmp_path / "pieces" / "pouw", gate=GATE, key="ff" * 64, main=None,
                 check=lambda *a: got)
    assert doc["SaltIsLiveCoin"] == {"met": False, "why": "the window's record isn't one manifest under its salt coin: "
                                                          "round 2's tables are not under the salt"}
    assert doc["AcceptedLive"]["met"] is False and doc["verdict"] == "reject"


@pytest.mark.parametrize("case", ["coins", "untied", "in-process", "no gate"])
def test_a_development_pouw_piece_rejects_what_its_checks_refuse(tmp_path, monkeypatch, case):
    """Coins the challenger check refuses, H1 tables untied to the Commit, coins held in the serving process, or no gate
    to check the session by: the piece rejects, naming why, and the record names it and writes no block record."""
    monkeypatch.setattr(S, "_gate_out", _gate([]))
    d = _h1_serve_dir(tmp_path, "in-process" if case == "in-process" else "challenger", tied=case != "untied")
    refused = {"ok": False, "reason": "session s1 is not the one the ledger opened", "manifest": None, "credit": {}}
    ok = {"ok": True, "reason": None, "manifest": None, "credit": {}}
    doc = S.pouw(d, 1, tmp_path / "pieces" / "pouw", gate=None if case == "no gate" else GATE, key="ff" * 64, main=None,
                 check=lambda *a: refused if case == "coins" else ok)
    why = {"coins": "coins: session s1 is not the one the ledger opened", "untied": "ROOTS1:",
           "in-process": "coins: in-process, not the auditor host's challenger's", "no gate": "coins: ValueError: no gate"}
    assert doc["verdict"] == "reject" and doc["reasons"][0].startswith(why[case]), doc["reasons"]
    salt = {"coins": "the challenger check refused the window's coins, its salt among them: session s1",
            "untied": S.SALT_MET, "in-process": "the challenger check didn't run on the window: coins: in-process",
            "no gate": "the challenger check didn't run on the window: coins: ValueError: no gate"}
    assert doc["SaltIsLiveCoin"]["met"] is (case == "untied") and doc["SaltIsLiveCoin"]["why"].startswith(salt[case])
    assert doc["AcceptedLive"]["met"] is False
    row = S.piece_row("work", tmp_path / "pieces")
    assert row["verdict"] == "reject" and row["reason"].startswith(why[case])
    _report, rec, line = _record(tmp_path, _rows(_accepted(), tmp_path / "pieces"))
    assert rec is None and (line["verdict"], line["development"], line["rejected"]) == ("undecided", {"PoUW": "reject"},
                                                                                       ["PoUW"])


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
    assert rc == 0 and limits == ["30m", "20m"], (rc, lines)
    assert lines[0].startswith("WINDOW-CIRCUIT class 4,32,4: registered 7 bits, free 3 bits of 10 input bits at the "
                               "weights registration's pins; ")
    assert lines[1:] == ["WINDOW-CIRCUIT class record (measurement) timed out after WC_RECORD_TIMEOUT=20m (exit 124): no count"]


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
    assert len(measured) == (classes != "record") and limits[-1] == ("30m" if classes == "record" else "20m")


DIGEST = "5e" * 64


def _auditor():
    """Infra's gate, ``benchmarks/live_challenger/auditor.py``, as a module of its own."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("sv_auditor", HERE.parents[2] / "benchmarks" / "live_challenger" / "auditor.py")
    module = sys.modules.setdefault(spec.name, importlib.util.module_from_spec(spec))
    spec.loader.exec_module(module)
    return module


def _declared(tmp_path: Path, files: dict | None = None, live=(32, 2)) -> tuple[Path, Path, dict]:
    """A serve dir whose serve-openai declared a window Program of digest `DIGEST` at T 32, S 2 (serve.json and
    ``h1/window-program.meta.json``, its layout `served.declared_layout`'s over two linears at K 32 and 64), and a
    circuit-files directory for it whose manifest holds `files` (default the live file and k32's, not k64's)."""
    from verity_vllm.protocols.pouw.h1 import served as SV
    d, c = tmp_path / "serve", tmp_path / "circuits"
    (d / "h1").mkdir(parents=True)
    c.mkdir()
    layout = SV.declared_layout([(64, 32), (32, 64)], {0: "up", 1: "down"}, 32, 2)
    (d / "serve.json").write_text(json.dumps({"protocols": {"program": DIGEST, "rows": 32, "steps": 2,
                                                            "weights_root": "77" * 64}}))
    (d / S.H1_META).write_text(json.dumps({"program_digest": DIGEST, "rows": 32, "steps": 2, "layout": layout}))
    files = {"live": "11" * 64, "k32": "22" * 64} if files is None else files
    (c / S.CIRCUITS_MANIFEST).write_text(json.dumps({"program": DIGEST, "live_bind": {"T": live[0], "S": live[1]},
                                                     "files": files}))
    return d, c, layout


def test_the_period_opens_on_the_served_programs_digest_and_files_as_cores_credit_reads_them(tmp_path, monkeypatch):
    """``period open``, through infra's own gate (`auditor.gate` and `auditor.period`, as lane vllm's key): its open
    commits the Program digest serve-openai wrote, its T, the draw K, the fixed length and the circuit files, each read
    back by core's credit as the window's period commits it; a stratum with no ``k<K>`` file yet is named and left out.
    It names no weights without ``--model`` (the gate refuses a root its registry lacks). ``period close`` closes it."""
    from verity.core.protocols.pouw import credit as CR
    from verity.ml.crypto import lms
    from verity_vllm.protocols.pouw.h1 import challenger as PHC

    A = _auditor()
    (tmp_path / "var" / "sessions").mkdir(parents=True)
    (tmp_path / "peers").write_text("vy-dc-h200 research@10.0.0.2\n")
    (tmp_path / "pins.jsonl").write_text(A.pin_line("c" * 40, "d" * 40, "2026-10-27T09:00:00Z") + "\n")
    host = A.Host(root=tmp_path / "var", peers=tmp_path / "peers", pins=tmp_path / "pins.jsonl")
    signer = A.key_signer(host, lms.Params(64, 4, 5), workers=1)
    now = datetime(2026, 10, 27, 19, 0, tzinfo=UTC)
    hooks = dict(now=now, token="ab12", own=lambda d: None, start=lambda s: None, active=lambda s: False,
                 secret=bytes(32), ready=lambda s: None,
                 period=lambda args: A.period(host, args, signer=signer, pinned="c" * 40, now=now, active=lambda s: False))
    sent = []

    def gate_out(gate: str, *words: str) -> bytes:
        sent.append(words)
        rc, out = A.gate(["--ssh", shlex.join(words), "--key", "vllm"], host, **hooks)
        if rc:
            raise RuntimeError(f"the gate refused {words[0]} (exit {rc}): {out.decode()}")
        return out

    monkeypatch.setattr(S, "_gate_out", gate_out)
    d, c, layout = _declared(tmp_path)
    doc = S.period_open(d, GATE, "served-r1", c, slots=4, max_tokens=8)
    assert doc["missing"] == [64] and doc["weights"] == [] and doc["draw_k"] == PHC.draw_k(layout)
    assert "--weights-root" not in sent[0] and "--model" not in sent[0]
    ledger = (host.root / "windows.jsonl").read_text().splitlines()
    assert CR.committed_k(ledger, "served-r1") == (PHC.draw_k(layout), None)
    assert CR.committed_rows(ledger, "served-r1") == (32, None)
    assert CR.committed_files(ledger, "served-r1") == ({"live": "11" * 64, "k32": "22" * 64}, None)
    assert CR.committed_length(ledger, "served-r1") == ([8] * 4, None)
    assert json.loads(ledger[-1])["body"]["circuit"] == DIGEST
    assert json.loads((d / S.PERIOD).read_text())["open"]["body"]["period"] == "served-r1"
    argv, _ = S.period_open_argv(d, "served-r2", c, slots=4, max_tokens=8, draw_k=3, model="LLAMA32_1B_INT8")
    assert argv[argv.index("--draw-k") + 1] == "3" and argv[argv.index("--model") + 1] == "LLAMA32_1B_INT8"
    S.period_close(d, GATE)
    kinds = [json.loads(x)["kind"] for x in (host.root / "windows.jsonl").read_text().splitlines()]
    assert kinds[-1] == "period-close" and json.loads((d / S.PERIOD).read_text())["close"]["body"]["windows"] == []


@pytest.mark.parametrize("case", ["other program", "other live T,S", "no live", "no meta", "no manifest"])
def test_a_period_open_is_refused_on_files_that_are_not_the_served_programs(tmp_path, case):
    """Circuit files of another Program, a live file for another T or S, none for the live template, a serve dir with no
    declared Program, or no circuit files: refused before anything is sent, naming the build to run."""
    d, c, _ = _declared(tmp_path, files={"k32": "22" * 64} if case == "no live" else None,
                        live=(64, 2) if case == "other live T,S" else (32, 2))
    if case == "other program":
        m = json.loads((c / S.CIRCUITS_MANIFEST).read_text())
        (c / S.CIRCUITS_MANIFEST).write_text(json.dumps({**m, "program": "6f" * 64}))
    if case == "no meta":
        (d / S.H1_META).unlink()
    if case == "no manifest":
        (c / S.CIRCUITS_MANIFEST).unlink()
    why = {"other program": "not the served", "other live T,S": "not the window's 32, 2", "no live": "not the window's",
           "no meta": "has no declared window Program", "no manifest": "no circuit files at"}[case]
    with pytest.raises(ValueError, match=why):
        S.period_open_argv(d, "p", c, slots=4, max_tokens=8)


def test_the_program_is_built_off_the_server_as_serve_openai_declares_and_writes_it(tmp_path, monkeypatch):
    """``program`` declares the served model's linears (`DECLARED`) in vLLM's order and naming, ``(n, K)`` each, at T and
    S, and writes them with serve-openai's own `served.write_program`; ``--like`` takes model, T and S from a serve
    dir and refuses a digest other than the one that run declared. The declaration and its digest are stand-ins here
    (a real one takes minutes); the node job checks the real one against a recorded run's digest."""
    from verity_vllm.protocols.pouw import h1 as PH
    from verity_vllm.protocols.pouw.h1 import served as SV

    seen = []
    monkeypatch.setattr(SV, "declared", lambda shapes, modules, rows, steps: ("program", shapes, modules, rows, steps))
    monkeypatch.setattr(SV, "declared_layout", lambda shapes, modules, rows, steps: {"tile": [32, 32]})

    def write(prog, layout, out, *, rows, steps):
        seen.append((prog, layout, out, rows, steps))
        out.mkdir(parents=True, exist_ok=True)
        (out / PH.PROGRAM_META).write_text(json.dumps({"program_digest": DIGEST, "rows": rows, "steps": steps,
                                                       "bytes": 9, "layout": layout}))
        return DIGEST
    monkeypatch.setattr(SV, "write_program", write)
    meta = S.program(tmp_path / "h1", "verity-int8/Llama-3.2-1B-INT8", 512, 16)
    (_tag, shapes, modules, rows, steps), _layout, out, *_ = seen[0]
    assert meta["program_digest"] == DIGEST and out == tmp_path / "h1" and (rows, steps) == (512, 16)
    assert len(shapes) == 64 and shapes[:4] == [(3072, 2048), (2048, 2048), (16384, 2048), (2048, 8192)]
    assert modules[0] == "model.layers.0.self_attn.qkv_proj" and modules[63] == "model.layers.15.mlp.down_proj"

    run = tmp_path / "serve"
    run.mkdir()
    (run / "serve.json").write_text(json.dumps({"served_model_name": "verity-int8/Llama-3.2-1B-INT8",
                                                "protocols": {"program": DIGEST, "rows": 32, "steps": 2}}))
    assert S.main(["program", str(tmp_path / "like"), "--like", str(run)]) == 0 and seen[-1][3:] == (32, 2)
    (run / "serve.json").write_text(json.dumps({"served_model_name": "verity-int8/Llama-3.2-1B-INT8",
                                                "protocols": {"program": "6f" * 64, "rows": 32, "steps": 2}}))
    with pytest.raises(ValueError, match="DECLARED's linears aren't the ones that run served"):
        S.program(tmp_path / "like", "x", 0, 0, like=run)
    with pytest.raises(ValueError, match="no declared linears for 'other'"):
        S.program(tmp_path / "other", "other", 512, 16)



def _serve_argv(tmp_path: Path, env: dict[str, str]) -> tuple[int, list[str]]:
    """`served_verdicts.sh serve` in a toy tree whose venv python records serve-openai's argv and exits before listening,
    under a no-op `sleep`: the exit status and that argv."""
    tree, bin_, venv = tmp_path / "tree", tmp_path / "bin", tmp_path / "venv" / "bin"
    for d in (tree / "integrations" / "vllm", bin_, venv):
        d.mkdir(parents=True)
    for path, body in ((venv / "python", f"printf '%s\\n' \"$@\" > {tmp_path}/argv; exit 1"), (bin_ / "sleep", "exit 0")):
        path.write_text(f"#!/bin/sh\n{body}\n")
        path.chmod(0o755)
    env = {"PATH": f"{bin_}:{os.defpath}", "HOME": str(tmp_path), "VENV": str(venv.parent), "D": str(tmp_path / "serve"),
           "OUT": str(tmp_path / "out"), **env}
    rc = subprocess.run(["bash", str(HERE.parent / "served_verdicts.sh"), "serve"], cwd=tree, env=env, check=False,
                        capture_output=True).returncode
    return rc, (tmp_path / "argv").read_text().splitlines()


@pytest.mark.parametrize("env, budget", [
    ({"H1_GATE": ""}, "512"),
    ({"H1_GATE": GATE}, "512"),
    ({"H1_GATE": GATE, "H1_T": "8192"}, "8192"),
    ({"H1_GATE": "", "H1_T": "8192"}, "8192"),
    ({"T": "2048", "H1_T": "8192", "H1_GATE": GATE}, "2048"),
])
def test_the_engine_gets_one_token_budget_with_or_without_h1(tmp_path, env, budget):
    """serve-openai gets ``--max-num-batched-tokens`` once, T (default H1_T, else 512), whether H1 serves or not (H1_GATE
    empty turns H1 off); without it the engine would take serve-openai's own 2048 with H1 off."""
    rc, argv = _serve_argv(tmp_path, env)
    assert rc == 10 and argv[:3] == ["-m", "verity_vllm.pipeline.cli", "serve-openai"], (rc, argv)
    assert argv.count("--max-num-batched-tokens") == 1 and argv[argv.index("--max-num-batched-tokens") + 1] == budget
    assert ("--protocols" in argv) == bool(env["H1_GATE"])


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
    a = argparse.Namespace(port=0, model="m", varied=3, fixed=0, rate=2.0, prompt_range="16:24", max_tokens=4, seed=1,
                           release="run", jitter_ms=0.0, out=str(out))
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
