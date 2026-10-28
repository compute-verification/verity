"""`served_verdicts.py`'s window pick, its subprotocols' pieces and its record, on toy serve dirs and answers:
the window it audits holds the most requests and none carried across windows; a subprotocol with no piece is named missing
with its issue and given no verdict; and sampled proofs' accepted answer composes, with the others missing, to a pending
block record that `block_record.check` holds and its schema admits, while a refusal or an audit that wrote nothing
composes to reject with no block record. The wipe composes only when its answer accepts on the record's server. The
inference-only policy's row is the verdict and commitment its piece publishes of `served_cases.py`'s private answer,
refused (an error) where that answer is missing or malformed, and missing where V3c wrote no case, which keeps the
composed verdict pending; nothing public holds the answer's profile or reason or the case's α and Γ. PoUW's
development piece checks the window's gate session and credits nothing, and the gate is refused while the lane has a
period open. `served_verdicts.sh`'s window circuits gate on every class of CLASSES, and the class of record's count after
them gates nothing; its endpoint gets one token budget, T, with or without H1; its inference-only step runs
`served_cases.py` into a fresh private directory and publishes only the verdict and commitment. Its load, `serve_load.py --varied`, writes each
request's send, first-token and done times beside its record, on an injected clock and a stand-in endpoint."""
from __future__ import annotations

import argparse
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
from verity.core.protocols.pous import wipe_period as W
from verity.core.protocols.sampled_proofs.one_stage import registration as R
from verity.ml.circuits import codec

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import served_cases as SC
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


def test_a_subprotocol_with_no_piece_is_named_missing_with_its_issue(tmp_path, capsys):
    got = S.pieces(tmp_path, 1, tmp_path / "pieces", env={})
    lines = capsys.readouterr().out.splitlines()
    assert [g["verdict"] for g in got.values()] == ["missing"] * 4
    assert [line.split(" (")[0] for line in lines] == ["MISSING PoUW", "MISSING the bandwidth audit", "MISSING PoUS",
                                                       "MISSING the network certifier"]
    assert all(issue in "\n".join(lines) for issue in ("VLLM-33 G1", "VLLM-33 G2", "MEM-94", "NET-83", "INF-263"))
    assert not (tmp_path / "pieces").exists()


def test_a_piece_that_ran_gives_its_answer_and_one_that_wrote_none_errs(tmp_path, capsys):
    env = {"POUS_CMD": 'echo \'{"verdict": "reject", "reason": "no"}\' > "$3/answer.json"', "CERTIFIER_CMD": "true"}
    got = S.pieces(tmp_path, 1, tmp_path / "pieces", env=env)
    assert got["pous"] == {"verdict": "reject", "reason": "no", "rc": 0}
    assert got["certifier"]["verdict"] == "error" and "wrote no answer" in got["certifier"]["reason"]
    assert (got["pouw"]["verdict"], got["bandwidth"]["verdict"]) == ("missing", "missing")


def _accepted() -> dict:
    a = {"protocol": "verity/sampled-proofs/one-stage/v1", "verdict": "accept", "reason": None, "commitment": "ab" * 64,
         "public": {"bound": 2, "drawn": [0, 3], "per_template": None, "proved": [0, 3]},
         "profile": {"levels": [{"name": "unit", "size": 4}], "drawn": {"unit": 2}, "delta": "1/100"}}
    return S.row_of(S.INTEGRITY_OWNER, a, Path("/nonexistent"))


def _rows(integrity: dict, pieces: Path) -> dict:
    return {"integrity": integrity, **{k: S.piece_row(k, pieces) for k in S.PIECES}, "wipe": S.piece_row("wipe", pieces)}


def test_sampled_proofs_alone_composes_to_a_pending_block_record(tmp_path):
    report, rec, line = S.record(_rows(_accepted(), tmp_path), window=1, steps=(5, 11), node="vy-dc-h200", run="r1")
    jsonschema.Draft202012Validator(SCHEMA).validate(rec)
    assert BR.check(rec) == {}
    assert rec["legs"]["integrity"]["quantity"]["value"] == 2 and rec["server_time"] == [{"server": "vy-dc-h200", "steps": [5, 11]}]
    assert (line["composed"], line["record_missing"]) == ("pending", ["bandwidth", "certifier", "pous", "pouw", "wipe"])
    assert line["missing"] == {"PoUW": "VLLM-33 G1", "the bandwidth audit": "VLLM-33 G2", "PoUS": "MEM-94",
                               "the network certifier": "NET-83", "the wipe": "MEM-96"}
    assert report["protocols"]["work"] == {"verdict": "missing", "owner": "vllm-serve-arms", "issue": "VLLM-33 G1",
                                           "why": S.PIECES["work"][4]}


def test_an_audit_that_wrote_nothing_composes_to_reject_with_no_block_record(tmp_path):
    row = S.sampled_proofs_row(tmp_path)
    assert row["verdict"] == "error" and "no summary" in row["reason"]
    _report, rec, line = S.record(_rows(row, tmp_path), window=1, steps=(5, 11), node="n", run=None)
    assert rec is None and (line["composed"], line["by"]) == ("reject", ["sampled proofs"])


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
    names no server and stays missing (``unserved``), a wipe that wrote no answer (served_wipe.sh refused) stays missing
    with its issue, and one whose answer rejects makes the record reject with no block record."""
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
                 "--out", str(tmp_path / "record"), "--node", "vy-dc-h200"])
    line = json.loads(capsys.readouterr().out.strip().splitlines()[-1].removeprefix("RECORD "))
    if case == "reject":
        assert rc == 1 and (line["composed"], line["by"], line["block_record"]) == ("reject", ["the wipe"], None)
        assert not (tmp_path / "record" / "block-record.json").exists()
        return
    rec = json.loads((tmp_path / "record" / "block-record.json").read_text())
    jsonschema.Draft202012Validator(SCHEMA).validate(rec)
    assert rc == 0 and line["check"] == {} and line["composed"] == "pending"
    if case == "accept":
        assert line["record_missing"] == ["bandwidth", "certifier", "pous", "pouw"]
        assert rec["legs"]["wipe"]["server"] == "vy-dc-h200" and rec["legs"]["wipe"]["quantity"]["value"] == 4096
        assert rec["legs"]["wipe"]["runs"] == ["r1"] and rec["composed"]["legs"]["wipe"] == "accept"
        return
    assert line["record_missing"] == ["bandwidth", "certifier", "pous", "pouw", "wipe"]
    if case == "elsewhere":
        assert rec["legs"]["wipe"]["server"] == "vy-dc-h200-2" and rec["composed"]["elsewhere"] == ["wipe"]
    elif case == "unserved":
        assert rec["legs"]["wipe"] is None and "the wipe" not in line["missing"]
    else:
        assert rec["legs"]["wipe"] is None and line["missing"]["the wipe"] == "MEM-96"


INF = SC.INFERENCE_ONLY
#: what only the prover's node may hold: a block's gates and fresh-input bits (the answer's profile, and α and Γ read off
#: them), and a rejection's reason naming its block
GATES, BITS = 918273645, 564738291
PRIVATE = (str(GATES), str(BITS), "block 7", '"blocks": [[', '"alpha"', '"gamma"', SC.FROM_DIVISION)


def _case_dir(root: Path, verdict: str = "accept", answer: dict | str | None = None) -> tuple[Path, str]:
    """`served_cases.py`'s private inference-only directory, written by its own `write`: a case committing to a
    registration record, α and Γ in its params and policy.json, and the Lean answer line (an accept profiling one block,
    or a rejection naming it; ``answer`` instead where given); and the record's digest."""
    rec = {"format": R.FORMAT, "previous": None, "program": "ab" * 64}
    case = {"commitment": codec.canonical_json(rec).hex(), "evidence": codec.canonical_json({"owner": [0]}).hex(),
            "params": {"alpha": [BITS, GATES], "gamma": GATES, "log": [], "parameters": [], "program": "{}"}}
    _, digest = SC.record_of(case)
    if answer is None:
        accepted = verdict == "accept"
        answer = {"protocol": INF, "verdict": verdict, "commitment": digest, "public": {} if accepted else None,
                  "reason": None if accepted else f"block 7 holds {BITS} fresh-input bits, more than α allows",
                  "profile": {"blocks": [[GATES, BITS]], "gates": GATES} if accepted else None}
    d = root / "private" / "inference-only"
    SC.write(d, case, answer if isinstance(answer, str) else json.dumps(answer, sort_keys=True),
             {"policy.json": {"alpha": [BITS, GATES], "gamma": GATES, "read": SC.FROM_DIVISION}})
    return d, digest


def _publish(capsys, case_dir: Path, pieces: Path) -> tuple[int, dict, str]:
    """`served_verdicts.py inference-only`: its exit, the verdict.json it published, and what it printed."""
    rc = S.main(["inference-only", str(case_dir), str(pieces / "inference-only")])
    return rc, json.loads((pieces / "inference-only" / S.VERDICT).read_text()), capsys.readouterr().out


def _record(tmp_path: Path, monkeypatch, capsys, pieces: Path) -> tuple[int, dict, dict, str]:
    """`served_verdicts.py record` over sampled proofs' accepted row and ``pieces``: its exit, its RECORD line, its
    window report, and what it printed."""
    monkeypatch.setenv("RESEARCH_RUN_ID", "r1")
    monkeypatch.setattr(S, "sampled_proofs_row", lambda _audit_dir, _run=None: _accepted())
    (tmp_path / "serve").mkdir()
    rc = S.main(["record", "--serve-run", str(_serve_dir(tmp_path / "serve")), "--window", "1",
                 "--sampled-proofs", str(tmp_path / "audit"), "--pieces", str(pieces),
                 "--out", str(tmp_path / "record"), "--node", "vy-dc-h200"])
    out = capsys.readouterr().out
    line = json.loads(out.strip().splitlines()[-1].removeprefix("RECORD "))
    return rc, line, json.loads((tmp_path / "record" / "verdicts.json").read_text()), out


def _published(root: Path, *printed: str) -> str:
    """Everything public: each file under ``root`` and what was printed."""
    return "\n".join([*(p.read_text() for p in sorted(root.rglob("*")) if p.is_file()), *printed])


@pytest.mark.parametrize("verdict", ["accept", "reject"])
def test_the_inference_only_row_is_the_private_answers_verdict_and_commitment_only(tmp_path, monkeypatch, capsys,
                                                                                    verdict):
    """The piece publishes the policy's answer's verdict and the registration record's digest it commits to, and the
    record's inference-only row is that verdict and commitment: an accept leaves the record as its other rows compose
    it (pending, with PoUW and the rest missing), and a rejection makes it reject with no block record. Neither the
    published verdict, the record nor anything printed holds the answer's profile (the block's gates and fresh-input
    bits) or reason (naming the block), or the case's α and Γ."""
    case_dir, digest = _case_dir(tmp_path, verdict)
    out = tmp_path / "out"
    rc, published, printed = _publish(capsys, case_dir, out / "pieces")
    assert rc == 0 and published == {"protocol": INF, "verdict": verdict, "commitment": digest}
    assert printed == f"INFERENCE-ONLY {json.dumps(published, sort_keys=True)}\n"
    rc, line, report, said = _record(out, monkeypatch, capsys, out / "pieces")
    assert report["protocols"]["inference-only"] == {
        "protocol": INF, "owner": "network-accounting", "verdict": verdict, "commitment": digest, "quantity": None,
        "epsilon": None, "reason": None if verdict == "accept" else S.REJECTED, "runs": ["r1"]}
    assert line["verdicts"]["the inference-only policy"] == verdict and "the inference-only policy" not in line["missing"]
    if verdict == "accept":
        assert rc == 0 and line["composed"] == "pending" and line["check"] == {}
        assert "inference-only" not in json.loads((out / "record" / "block-record.json").read_text())["legs"]
    else:
        assert rc == 1 and (line["composed"], line["by"]) == ("reject", ["the inference-only policy"])
        assert not (out / "record" / "block-record.json").exists()
    public = _published(out, printed, said)
    assert [s for s in PRIVATE if s in public] == []


@pytest.mark.parametrize("case, why", [
    ("no answer", "served_cases.py wrote no answer"),
    ("not JSON", "its answer is not a core answer"),
    ("not a core answer", "its answer is not a core answer"),
    ("another protocol's", "its answer is another protocol's"),
    ("undecided", "its answer neither accepts nor rejects"),
    ("another record's", "its answer doesn't commit to its case's registration record"),
    ("no record", "its case commits to no registration record"),
])
def test_a_missing_or_malformed_private_answer_is_refused_and_the_record_rejects(tmp_path, monkeypatch, capsys, case,
                                                                                 why):
    """No answer (`served_cases.py` wrote none: its Lean refused the case or ran out of time), one that isn't JSON or
    isn't a core answer (an accept with a reason), another protocol's, an undecided one, one committing to another
    record than its case's, or a case that commits to no registration record: the piece publishes that it refused the
    answer, in fixed words, and exits 1; the row is that refusal, an error, and the record rejects with no block
    record. Nothing private is published."""
    private = tmp_path / "node"
    answers = {
        "not JSON": "not json {",
        "not a core answer": {"protocol": INF, "verdict": "accept", "commitment": None, "public": {},
                              "reason": "block 7 holds too many bits", "profile": {"blocks": [[GATES, BITS]]}},
        "another protocol's": {"protocol": SC.V2, "verdict": "reject", "commitment": "ab" * 64, "public": None,
                               "reason": "block 7", "profile": None},
        "undecided": {"protocol": INF, "verdict": "undecided", "commitment": "ab" * 64, "public": None,
                      "reason": f"block 7 holds {BITS} bits", "profile": None},
        "another record's": {"protocol": INF, "verdict": "accept", "commitment": "cd" * 64, "public": {},
                             "reason": None, "profile": {"blocks": [[GATES, BITS]], "gates": GATES}},
    }
    case_dir, _digest = _case_dir(private, answer=answers.get(case))
    if case == "no answer":
        case_dir = private / "private" / "not-written"
    if case == "no record":
        c = json.loads((case_dir / "case.json").read_text())
        (case_dir / "case.json").write_text(json.dumps({**c, "commitment": b'{"program":1,"format":2}'.hex()}))
    out = tmp_path / "out"
    rc, published, printed = _publish(capsys, case_dir, out / "pieces")
    assert rc == 1 and published == {"protocol": INF, "refused": why}
    rc, line, report, said = _record(out, monkeypatch, capsys, out / "pieces")
    assert report["protocols"]["inference-only"] == {"protocol": INF, "owner": "network-accounting", "verdict": "error",
                                                     "reason": f"refused: {why}"}
    assert rc == 1 and (line["composed"], line["by"]) == ("reject", ["the inference-only policy"])
    assert not (out / "record" / "block-record.json").exists()
    public = _published(out, printed, said)
    assert [s for s in PRIVATE if s in public] == []


@pytest.mark.parametrize("written, why", [
    (None, "its piece wrote no verdict.json"),
    ("{", "its verdict.json is not JSON"),
    ({"protocol": INF, "verdict": "accept", "commitment": "ab" * 64, "profile": {"blocks": [[GATES, BITS]]}},
     "its verdict.json is not the policy's verdict and commitment"),
    ({"protocol": INF, "verdict": "undecided", "commitment": "ab" * 64},
     "its verdict.json is not the policy's verdict and commitment"),
    ({"protocol": SC.V2, "verdict": "accept", "commitment": "ab" * 64},
     "its verdict.json is not the policy's verdict and commitment"),
    ({"protocol": INF, "verdict": "accept", "commitment": "AB" * 64},
     "its verdict.json is not the policy's verdict and commitment"),
    ({"protocol": INF, "refused": ["served_cases.py wrote no answer"]},
     "its verdict.json is not the policy's verdict and commitment"),
])
def test_a_published_verdict_not_in_its_form_is_refused_by_the_record(tmp_path, written, why):
    """The record (which the verifier reruns on the public record alone) reads only the piece's form: a piece directory
    with no verdict.json, or one that isn't JSON, carries more than the verdict and commitment, says neither accept nor
    reject, is another protocol's, or isn't a SHA-512 digest in lowercase hex, is refused."""
    (d := tmp_path / "inference-only").mkdir()
    if written is not None:
        (d / S.VERDICT).write_text(written if isinstance(written, str) else json.dumps(written))
    assert S.inference_only_row(tmp_path, "r1") == {"protocol": INF, "owner": "network-accounting", "verdict": "error",
                                                    "reason": f"refused: {why}"}


def test_with_no_case_the_row_is_missing_and_keeps_an_accepted_block_record_pending(tmp_path, monkeypatch):
    """Where V3c's audit wrote no case the piece doesn't run and the row is missing, saying why. The block record has no
    leg for the policy, so a block whose legs all accept composes pending while the row is missing, and accept once the
    policy accepts."""
    missing = S.inference_only_row(tmp_path)
    assert missing == {"verdict": "missing", "owner": "network-accounting", "why": S.INFERENCE_ONLY[3]}
    (d := tmp_path / "inference-only").mkdir()
    (d / S.VERDICT).write_text(json.dumps({"protocol": INF, "verdict": "accept", "commitment": "ab" * 64}))
    accepted = S.inference_only_row(tmp_path)
    real = BR.from_period

    def legs_accept(*a, **k):
        rec = real(*a, **k)
        rec["composed"]["verdict"] = "accept"
        return rec

    monkeypatch.setattr(BR, "from_period", legs_accept)
    composed = {}
    for name, row in (("missing", missing), ("accepted", accepted)):
        _report, rec, line = S.record({"integrity": _accepted(), "inference-only": row}, window=1, steps=(5, 11),
                                      node="n", run="r")
        composed[name] = (line["composed"], line["missing"], rec is not None)
    assert composed == {"missing": ("pending", {"the inference-only policy": S.INFERENCE_ONLY[3]}, True),
                        "accepted": ("accept", {}, True)}


FAKE_SERVED_CASES = '''import json, shutil, sys
a = sys.argv[1:]
open({seen!r}, "w").write(json.dumps(a))
shutil.copytree({prepared!r}, a[a.index("--private") + 1] + "/inference-only")
print("CASE", json.dumps({{"protocol": "verity/inference-only/v0", "verdict": "accept"}}))
'''


def _inference_only_step(tmp_path: Path, env: dict[str, str], *, case: bool = True) -> tuple[int, Path, str]:
    """`served_verdicts.sh inference-only 1` in a toy tree whose served_cases.py records its argv and copies a prepared
    private case and answer where it is told, and whose served_verdicts.py is this one: its exit, D and the digest."""
    tree, d = tmp_path / "tree", tmp_path / "serve"
    one, bin_ = tree / "benchmarks" / "one_stage", tree / ".venv" / "bin"
    for x in (one, bin_, d):
        x.mkdir(parents=True)
    prepared, digest = _case_dir(tmp_path / "prepared")
    (one / "served_cases.py").write_text(FAKE_SERVED_CASES.format(seen=str(tmp_path / "argv"), prepared=str(prepared)))
    (one / "served_verdicts.py").symlink_to(HERE.parent / "served_verdicts.py")
    (bin_ / "python").write_text(f'#!/bin/sh\nexec {sys.executable} "$@"\n')
    (bin_ / "python").chmod(0o755)
    if case:
        (d / "v3c-w1-4-32-4" / "audit" / "window").mkdir(parents=True)
        (d / "v3c-w1-4-32-4" / "audit" / "window" / "case.json").write_text("{}")
    env = {"PATH": os.defpath, "HOME": str(tmp_path), "VENV": str(tmp_path / "venv"), "D": str(d),
           "OUT": str(tmp_path / "out"), **env}
    rc = subprocess.run(["bash", str(HERE.parent / "served_verdicts.sh"), "inference-only", "1"], cwd=tree, env=env,
                        check=False, capture_output=True).returncode
    return rc, d, digest


@pytest.mark.parametrize("env, policy", [({}, ["--from-division"]),
                                         ({"ALPHA": "3/7", "GAMMA": "100"}, ["--alpha", "3/7", "--gamma", "100"])])
def test_the_runners_inference_only_step_keeps_the_case_private_and_publishes_its_verdict(tmp_path, env, policy):
    """`served_verdicts.sh`'s step: `served_cases.py` on V3c's audit of the window, α and Γ off the division unless
    ALPHA and GAMMA are both set, into a fresh 0700 directory under D/private that holds its log; OUT gets the
    verdict and commitment and nothing private."""
    rc, d, digest = _inference_only_step(tmp_path, env)
    argv = json.loads((tmp_path / "argv").read_text())
    p = Path(argv[argv.index("--private") + 1])
    assert rc == 0 and argv == ["--audit", f"{d}/v3c-w1-4-32-4/audit", "--private", str(p), *policy]
    assert p.parent == d / "private" and p.name.startswith("cases-w1.") and p.stat().st_mode & 0o777 == 0o700
    assert (p / "served_cases.log").read_text().startswith("CASE ")
    out = tmp_path / "out"
    published = json.loads((out / "pieces" / "inference-only" / S.VERDICT).read_text())
    assert published == {"protocol": INF, "verdict": "accept", "commitment": digest}
    assert [s for s in PRIVATE if s in _published(out)] == []
    assert S.inference_only_row(out / "pieces")["verdict"] == "accept"


def test_the_runners_inference_only_step_with_no_v3c_case_names_it_missing(tmp_path):
    rc, _d, _digest = _inference_only_step(tmp_path, {}, case=False)
    lines = (tmp_path / "out" / "verdicts.txt").read_text().splitlines()
    assert rc == 0 and lines == ["MISSING the inference-only policy: V3c's audit of window 1 wrote no case"]
    assert not (tmp_path / "argv").exists() and not (tmp_path / "out" / "pieces").exists()


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
    accepts, its row in the record stays missing with its document (it credits nothing), the record composes pending, and
    the pieces' line says it is a development run's."""
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
    assert (row["verdict"], row["why"], row["development"]["verdict"]) == ("missing", S.DEV_WHY, "accept")
    _report, rec, line = S.record(_rows(_accepted(), tmp_path / "pieces"), window=1, steps=(0, 4), node="n", run="r")
    assert line["composed"] == "pending" and "pouw" in line["record_missing"] and line["development"] == {"PoUW": "accept"}
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
    assert rec is None and (line["composed"], line["by"], line["development"]) == ("reject", ["PoUW"], {"PoUW": "reject"})


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
    ({}, "512"),
    ({"H1_GATE": GATE}, "512"),
    ({"H1_GATE": GATE, "H1_T": "8192"}, "8192"),
    ({"H1_T": "8192"}, "8192"),
    ({"T": "2048", "H1_T": "8192", "H1_GATE": GATE}, "2048"),
])
def test_the_engine_gets_one_token_budget_with_or_without_h1(tmp_path, env, budget):
    """serve-openai gets ``--max-num-batched-tokens`` once, T (default H1_T, else 512), whether H1 serves or not; without
    it the engine would take serve-openai's own 2048 with H1 off."""
    rc, argv = _serve_argv(tmp_path, env)
    assert rc == 10 and argv[:3] == ["-m", "verity_vllm.pipeline.cli", "serve-openai"], (rc, argv)
    assert argv.count("--max-num-batched-tokens") == 1 and argv[argv.index("--max-num-batched-tokens") + 1] == budget
    assert ("--protocols" in argv) == ("H1_GATE" in env)


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
