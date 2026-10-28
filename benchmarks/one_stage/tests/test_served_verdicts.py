"""`served_verdicts.py`'s window pick, its subprotocols' pieces and its record, on toy serve dirs and answers:
the window it audits holds the most requests and none carried across windows; a subprotocol with no piece is named missing
with the command it lacks and given no verdict, one that refused to run is missing, and one that ran out of its limit or
wrote no answer errs; and sampled proofs' accepted answer composes, with the others missing, to a pending block record that
`block_record.check` holds and its schema admits, while a refusal or an audit that wrote nothing composes to reject with no
block record. The record accepts only when every subprotocol accepted on its node, and names each one missing, rejecting,
malformed or about another server. The wipe composes only when its answer accepts on the record's server. PoUW's
development piece checks the window's gate session and credits nothing, and the gate is refused while the lane has a period
open. The period the served windows join opens, through infra's own gate, on the Program serve-openai declared and
the circuit files built for it, as core's credit reads them, and is refused on files of anything else.
`served_verdicts.sh`'s window circuits gate on every class of CLASSES, and the class of record's count after them
gates nothing; its endpoint gets one token budget, T, with or without H1. Its preflight refuses a flag the tree's certifier
front doesn't take, a front whose endpoint isn't on the warden's id form, and a front load with no front; its composite
step publishes each case's and each product's verdict lines and nothing else. Its load, `serve_load.py --varied`, writes
each request's send, first-token and done times beside its record, on an injected clock and a stand-in endpoint."""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import itertools
import json
import os
import shlex
import shutil
import subprocess
import sys
import tarfile
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path

import jsonschema
import pytest
from verity.core.protocols.pous import wipe_period as W

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import served_verdicts as S
from verity_e2e_demo import block_record as BR

SCHEMA = json.loads((Path(BR.__file__).parent / "schemas" / "block-record.v1.schema.json").read_text())
_SL = importlib.util.spec_from_file_location("serve_load", HERE.parents[2] / "demos" / "vllm" / "serve_load.py")
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


def test_the_published_window_counts_none_of_its_requests(tmp_path):
    """What the run says of its window: its engine steps and whether its requests filled the class's slots, and the
    refused windows by number, with no count, list or user of any window's requests."""
    got = S.pick(_serve_dir(tmp_path), slots=4)
    assert S.public_pick(got, 4) == {"window": 1, "steps": [5, 11], "engine_steps": 6, "slots": 4, "every_slot": False,
                                     "refused": [2, 3]}
    assert S.public_pick(got, 3)["every_slot"] is True
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        assert S.main(["window", str(tmp_path), "--slots", "4"]) == 0
    assert json.loads(out.getvalue()) == S.public_pick(got, 4)


REGISTRATION = {"schema": "verity-vllm/advice-registration/v1", "row_bits": 256, "rows": 2, "T": 600,
                "binding": {"compiler": {"name": "c", "digest": "d"}, "advice": {"root": "ab" * 64, "bits": 392}},
                "compiler": {"name": "c", "digest": "d", "cap_bits": 8388608}, "bits_by_window": {"0": 392, "1": 7},
                "registered_roots": {"weights/w@16": "cd" * 64}, "requests": 3, "refused": {"q0": "2 of its 3 tokens"}}


def test_the_published_registration_pads_the_advices_length_to_the_cap(tmp_path, capsys):
    """The advice registration's public form states no length of the advice but its compiler's cap, counts and refuses
    none of its requests, and keeps its roots, compiler and binding's root."""
    got = S.public_registration(REGISTRATION)
    assert got["binding"]["advice"] == {"root": "ab" * 64, "bits": 8388608} and got["rows"] == 32768
    assert got["bits_by_window"] == {"0": 8388608, "1": 8388608} and got["padded"] == list(S.PADDED)
    assert not {"requests", "compiled", "refused"} & set(got) and "392" not in json.dumps(got)
    assert {k: got[k] for k in ("registered_roots", "compiler", "T", "schema")} == {
        k: REGISTRATION[k] for k in ("registered_roots", "compiler", "T", "schema")}
    (tmp_path / "r.json").write_text(json.dumps(REGISTRATION))
    assert S.main(["public-registration", str(tmp_path / "r.json")]) == 0
    assert json.loads(capsys.readouterr().out) == got


def test_the_sampled_proofs_line_names_the_partition_and_its_draw_and_no_advice_length(tmp_path, capsys):
    row = {"ok": True, "accepted": True, "draw": [0, 2], "population": 3,
           "leak": {"advice_bits": 832, "bound": "vacuous", "other_inputs": 786432, "registered": 0}}
    part = {"templates": ["SiluMul_v2{I=8192}"], "law": "subset:2",
            "scope": {"op_path_prefix": "model.layers.0.mlp.", "steps": [0, 1]}}
    line = S.sampled_line(row, part)
    head, _, rec = line.partition(": RECORDS ")
    assert head == "over SiluMul_v2{I=8192} at model.layers.0.mlp. steps [0, 1] (law subset:2), drawn members [0, 2] of 3"
    assert json.loads(rec)["leak"] == {"bound": "vacuous", "other_inputs": 786432, "registered": 0} and "832" not in line
    (tmp_path / "log").write_text("+ set -x\nRECORDS " + json.dumps(row) + "\n")
    (tmp_path / "p.json").write_text(json.dumps(part))
    assert S.main(["sampled-line", str(tmp_path / "log"), "--partition", str(tmp_path / "p.json")]) == 0
    assert capsys.readouterr().out == line + "\n"
    (tmp_path / "log").write_text("refused\n")
    assert S.main(["sampled-line", str(tmp_path / "log")]) == 1


def test_one_piece_runs_alone_or_all_but_it(tmp_path, capsys):
    """``only`` runs the named pieces alone and ``skip`` all but them, so the certifier's can run beside the others."""
    assert list(S.pieces(tmp_path, 1, tmp_path / "pieces", env={}, only=["certifier"])) == ["certifier"]
    assert "the network certifier" in capsys.readouterr().out
    assert list(S.pieces(tmp_path, 1, tmp_path / "pieces", env={}, skip=["certifier"])) == ["pouw", "bandwidth", "pous"]
    assert "the network certifier" not in capsys.readouterr().out


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


def test_a_subprotocol_with_no_piece_is_named_missing_with_the_command_it_lacks(tmp_path, capsys):
    got = S.pieces(tmp_path, 1, tmp_path / "pieces", env={})
    lines = capsys.readouterr().out.splitlines()
    assert [g["verdict"] for g in got.values()] == ["missing"] * 4
    assert [line.split(" (")[0] for line in lines] == ["MISSING PoUW", "MISSING the bandwidth audit", "MISSING PoUS",
                                                       "MISSING the network certifier"]
    for line, (cmd, owner) in zip(lines, (("POUW_CMD", "compute-accounting"), ("BANDWIDTH_CMD", "memory-accounting"),
                                          ("POUS_CMD", "memory-accounting"), ("CERTIFIER_CMD", "network-accounting"))):
        assert f"({cmd} unset; {owner}): " in line
    assert not (tmp_path / "pieces").exists()


def _reject(protocol: str) -> dict:
    return {"protocol": protocol, "verdict": "reject", "reason": "no", "commitment": None, "public": None, "profile": None}


def test_a_piece_that_ran_gives_its_answer_and_one_that_wrote_none_errs(tmp_path, capsys):
    env = {"POUS_CMD": f"echo '{json.dumps(_reject('verity/pous/p2/v4/run'))}' > \"$3/answer.json\"",
           "CERTIFIER_CMD": "true", "BANDWIDTH_CMD": 'echo \'{"verdict": "reject", "reason": "no"}\' > "$3/answer.json"'}
    got = S.pieces(tmp_path, 1, tmp_path / "pieces", env=env, node="vy-dc-h200")
    assert got["pous"] == {"verdict": "reject", "reason": "no", "rc": 0}
    assert got["certifier"]["verdict"] == "error" and "wrote no answer" in got["certifier"]["reason"]
    assert got["bandwidth"]["verdict"] == "error" and "not a core answer" in got["bandwidth"]["reason"]
    assert got["pouw"]["verdict"] == "missing"
    assert json.loads((tmp_path / "pieces" / "pous" / S.RAN).read_text()) == {"server": "vy-dc-h200", "rc": 0,
                                                                              "limit": S.PIECE_TIMEOUT}
    assert {k: S.piece_row(k, tmp_path / "pieces")["verdict"] for k in S.PIECES} == {
        "work": "missing", "bandwidth": "error", "space": "reject", "communication": "error"}


def test_a_piece_runs_within_its_limit_and_one_out_of_time_or_refused_says_so(tmp_path):
    """Each piece runs under `timeout` at ``<COMPONENT>_TIMEOUT``, else ``PIECE_TIMEOUT``: one that runs out of it (exit
    124, which the fake `timeout` gives for the limit ``0``) errs naming the limit, and one that writes ``refused.txt``
    instead of an answer is missing with the refusal as why."""
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    (bin_ / "timeout").write_text(f'#!/bin/sh\necho "$3" >> {tmp_path}/limits\n[ "$3" = 0 ] && exit 124\nshift 3\nexec "$@"\n')
    (bin_ / "timeout").chmod(0o755)
    env = {"PATH": f"{bin_}:{os.defpath}", "PIECE_TIMEOUT": "7m", "BANDWIDTH_TIMEOUT": "0", "BANDWIDTH_CMD": "true",
           "POUS_CMD": 'echo "PoUS REFUSED: the card has a compute process" > "$3/refused.txt"; exit 3'}
    got = S.pieces(tmp_path, 1, tmp_path / "pieces", env=env, node="n")
    assert (tmp_path / "limits").read_text().split() == ["0", "7m"]
    assert got["bandwidth"] == {"verdict": "error", "rc": 124,
                                "reason": "its piece ran (exit 124, out of time at 0) and wrote no answer; "
                                          "its log is bandwidth/piece.log"}
    assert got["pous"] == {"verdict": "missing", "rc": 3,
                           "why": "its piece refused to run: PoUS REFUSED: the card has a compute process"}


def test_a_piece_that_left_a_file_that_is_not_json_errs_about_no_server_and_the_next_piece_still_runs(tmp_path):
    env = {"BANDWIDTH_CMD": f"echo '{json.dumps(_reject('verity/remote-bandwidth/v0'))}' > \"$3/answer.json\"; "
                            "echo '{' > \"$3/served.json\"",
           "POUS_CMD": f"echo '{json.dumps(_reject('verity/pous/p2/v4/run'))}' > \"$3/answer.json\""}
    got = S.pieces(tmp_path, 1, tmp_path / "pieces", env=env, node="n")
    assert got["bandwidth"]["verdict"] == "error"
    assert got["bandwidth"]["reason"].startswith("its piece left a file that isn't JSON: JSONDecodeError")
    assert got["pous"] == {"verdict": "reject", "reason": "no", "rc": 0}
    row, at = S.piece("bandwidth", tmp_path / "pieces")
    assert (row["verdict"], row["owner"], at) == ("error", "memory-accounting", (None, "real"))


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
    assert {name: why.split(" unset: ")[0] for name, why in line["missing"].items()} == {
        "PoUW": "POUW_CMD", "the bandwidth audit": "BANDWIDTH_CMD", "PoUS": "POUS_CMD", "the network certifier": "CERTIFIER_CMD",
        "the wipe": "WIPE_CMD"}
    assert report["protocols"]["work"] == {"verdict": "missing", "owner": "compute-accounting",
                                           "why": f"POUW_CMD unset: {S.PIECES['work'][3]}"}
    assert line["elsewhere"] == {} and "unserved" not in line


def test_an_accepted_pous_answer_says_when_its_weights_were_held_their_share_and_where_its_quantity_is_not_established(
        tmp_path):
    """PoUS's answer in the shape its served piece writes: weights reloaded on the card after the window's serve, 64.9% of
    the checkpoint, at ε = 1/100 + 2^-128, which the block record establishes at neither of its δs. Its accept's clause in
    the composed reason says each, and the ``RECORD`` line carries that reason."""
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
    rows = _rows(_accepted(), tmp_path)
    report, rec, line = S.record(rows, window=1, steps=(5, 11), node="vy-dc-h200", run="r1",
                                 servers={k: S.piece_server(k, tmp_path) for k in [*S.PIECES, "wipe"]})
    assert rec["legs"]["pous"]["at"] == {d: {"holds": False, "why": f"the run established it except with {eps}, more than {d}"}
                                         for d in ("1/100", "1/1000000")}
    clause = line["reason"].split("; PoUS: ", 1)[1]
    assert clause.startswith(f"accept (its weights {held}, not concurrently with the block; they are 64.9% of the "
                             "checkpoint (linears); the embedding and the head, the norms, the per-row scales are outside "
                             "its claim; its quantity is not established at δ = 1/100 or 1/1000000: the run established "
                             "it except with 1/100 + 2^-128)")
    assert line["reason"] == report["composed"]["reason"] == rec["composed"]["onsite"]["reason"]
    assert line["reason"].startswith("sampled proofs: accept; ")


def test_an_accepted_certifier_answer_says_what_its_trusted_host_premise_rests_on(tmp_path):
    """The network certifier's answer as its piece writes it when the warden ran on the gate's host and its checks, draws
    and coins on the served node: its accept's clause says the premise is assumed and why, so the ``RECORD`` line's
    accept isn't read as the auditor's devices having placed it."""
    premise = ("assumed, the auditor's devices not placed: the network certifier ran on 204.12.178.84, not the prover's "
               "vy-dc-h200, but the chain's checks and draws, the period check and their proofs' coins ran on vy-dc-h200")
    (d := tmp_path / "certifier").mkdir()
    (d / "answer.json").write_text(json.dumps({
        "protocol": "verity/certifier/v0", "verdict": "accept", "reason": None, "commitment": "cd" * 64,
        "public": {"K": 15764, "eps": "1/100", "windows": 1},
        "profile": {"placement": {"kind": "not-placed", "trusted-host": premise}}}))
    (d / S.RAN).write_text(json.dumps({"server": "vy-dc-h200", "rc": 0, "limit": "60m"}))
    _report, _rec, line = S.record(_rows(_accepted(), tmp_path), window=1, steps=(5, 11), node="vy-dc-h200", run="r1",
                                   servers={k: S.piece_server(k, tmp_path) for k in [*S.PIECES, "wipe"]})
    assert line["reason"].split("; the network certifier: ", 1)[1].startswith(f"accept (trusted-host {premise}; ")


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
        assert rec["legs"]["wipe"] is None
        assert line["missing"]["the wipe"] == "its piece refused to run: WIPE REFUSED: the card has a compute process"


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
    (case, piece) for case in ("missing", "reject", "elsewhere", "malformed") for piece in COMPONENTS])
def test_the_record_accepts_only_when_every_subprotocol_accepts_on_its_node_and_names_each_that_does_not(
        tmp_path, monkeypatch, capsys, case, piece):
    """Every piece's answer accepted on the record's node (each piece's `RAN`, which a `SERVED` naming no server leaves it
    about, and the wipe's `SERVED`): the composed verdict is ``accept``, with nothing missing, and its reason names each
    subprotocol's accept. One piece with no answer leaves it ``pending`` and names that piece missing with
    why; one that rejects, or accepts with outputs that are not its leg's certificate, makes it ``reject`` naming it, with
    no block record; one about another server leaves it ``pending``, carried and named as elsewhere."""
    monkeypatch.setenv("RESEARCH_RUN_ID", "r1")
    monkeypatch.setattr(S, "sampled_proofs_row", lambda _audit_dir, _run=None: _accepted())
    pieces = tmp_path / "pieces"
    for comp in COMPONENTS:
        (d := pieces / comp).mkdir(parents=True)
        if comp == piece and case == "missing":
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
    leg, runner = piece, {v[1]: k for k, v in S.PIECES.items()} | {"wipe": "wipe"}
    name = S.NAMES[runner[piece]] if piece else None
    if case in ("reject", "malformed"):
        assert rc == 1 and (line["composed"], line["by"], line["block_record"]) == ("reject", [name], None)
        assert line["verdicts"][name] == ("reject" if case == "reject" else "error")
        assert not (tmp_path / "record" / "block-record.json").exists()
        report = json.loads((tmp_path / "record" / "verdicts.json").read_text())
        row = next(r for k, r in report["protocols"].items() if S.NAMES[k] == name)
        assert case == "reject" or row["reason"].startswith(f"its answer is not the block record's {leg} leg: KeyError")
        return
    rec = json.loads((tmp_path / "record" / "block-record.json").read_text())
    jsonschema.Draft202012Validator(SCHEMA).validate(rec)
    assert rc == 0 and line["check"] == {} == BR.check(rec)
    if case == "accept":
        assert (line["composed"], line["record_missing"], line["missing"], line["elsewhere"]) == ("accept", [], {}, {})
        assert rec["composed"]["legs"] == {k: "accept" for k in BR.LEGS}
        late = {"the network certifier": " (its quantity is not established at δ = 1/1000000: the run established it "
                                         "except with 1/100)"}
        assert rec["composed"]["onsite"]["reason"] == line["reason"] == "; ".join(
            f"{n}: accept{late.get(n, '')}" for n in S.NAMES.values())
        return
    assert line["composed"] == "pending" and line["record_missing"] == [leg]
    if case == "missing":
        what = S.WIPE[3] if piece == "wipe" else S.PIECES[runner[piece]][3]
        assert line["missing"] == {name: f"{piece.upper()}_CMD unset: {what}"}
        assert rec["legs"][leg] is None and line["elsewhere"] == {}
    else:
        assert line["missing"] == {} and line["elsewhere"] == {name: "vy-dc-h200-2"}
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
    accepts, its row in the record stays missing with its document (it credits nothing), the record composes pending, and
    the pieces' line says it is a development run's."""
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
    assert (row["verdict"], row["why"], row["development"]["verdict"]) == ("missing", why, "accept")
    report, rec, line = S.record(_rows(_accepted(), tmp_path / "pieces"), window=1, steps=(0, 4), node="n", run="r")
    assert line["composed"] == "pending" and "pouw" in line["record_missing"] and line["development"] == {"PoUW": "accept"}
    assert line["verdicts"]["PoUW"] == S.DEV_VERDICT == "development run, credits nothing" and line["missing"]["PoUW"] == why
    jsonschema.Draft202012Validator(SCHEMA).validate(rec)
    assert BR.check(rec) == {} and rec["composed"]["onsite"] == report["composed"]
    reason = rec["composed"]["onsite"]["reason"]
    assert reason.startswith(f"sampled proofs: accept; PoUW: {why}; ")
    for name, cmd in (("the bandwidth audit", "BANDWIDTH"), ("PoUS", "POUS"), ("the network certifier", "CERTIFIER"),
                      ("the wipe", "WIPE")):
        assert f"; {name}: missing: {cmd}_CMD unset: " in reason
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
    salt = {"coins": "the challenger check refused the window's coins, its salt among them: session s1",
            "untied": S.SALT_MET, "in-process": "the challenger check didn't run on the window: coins: in-process",
            "no gate": "the challenger check didn't run on the window: coins: ValueError: no gate"}
    assert doc["SaltIsLiveCoin"]["met"] is (case == "untied") and doc["SaltIsLiveCoin"]["why"].startswith(salt[case])
    assert doc["AcceptedLive"]["met"] is False
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
    assert rc == 0 and limits == ["30m", "45m"], (rc, lines)
    assert lines[0].startswith("WINDOW-CIRCUIT class 4,32,4: registered 7 bits, free 3 bits of 10 input bits at the "
                               "weights registration's pins; ")
    assert lines[1:] == ["WINDOW-CIRCUIT class record (measurement) timed out after WC_RECORD_TIMEOUT=45m (exit 124): no count"]
    published = json.loads((tmp_path / "out" / "window-circuit-4-32-4.json").read_text())
    kept = json.loads((tmp_path / "serve" / "window-circuit-4-32-4.count.json").read_text())
    assert "bits" not in published and '"bits"' not in lines[0] and kept["bits"] == 772 and published["count"] == kept["count"]


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
    assert len(measured) == (classes != "record") and limits[-1] == ("30m" if classes == "record" else "45m")


def _script(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"#!/bin/sh\n{body}\n")
    path.chmod(0o755)


FRONT = '"$PY" benchmarks/network_traces/certifier_piece.py front "$1" "$2" "$3" --bucket-ms 4000 --warmup 0'


IDS = "--protocol ids --id-payload 503"


@pytest.mark.parametrize("load, front, takes, serve_args, line", [
    ("varied", FRONT, "--gate --warmup --bucket-ms", IDS, None),
    ("front", FRONT, "--gate --warmup --bucket-ms", IDS, None),
    ("varied", "", "", "", None),
    ("varied", FRONT, "--gate --warmup --bucket-ms-x", IDS, "PREFLIGHT: CERTIFIER_FRONT passes --bucket-ms, which this "
                                                             "tree's certifier front doesn't take"),
    ("front", "", "", "", "PREFLIGHT: LOAD=front sends the front's group alone, and CERTIFIER_FRONT is unset"),
    ("front", FRONT, "--gate --warmup --bucket-ms", "--seal-hold-gb 8", "PREFLIGHT: the certifier front's warden sends "
     "id-form requests, which serve-openai answers only under SERVE_ARGS='--protocol ids --id-payload 503'"),
])
def test_the_preflight_refuses_a_front_flag_the_trees_front_does_not_take_and_a_front_load_with_no_front(
        tmp_path, load, front, takes, serve_args, line):
    tree, venv = tmp_path / "tree", tmp_path / "venv" / "bin"
    (tree / "benchmarks" / "network_traces").mkdir(parents=True)
    (tree / "benchmarks" / "network_traces" / "certifier_piece.py").write_text(
        f"import sys\nassert sys.argv[1:] == ['front', '--help']\nprint('usage: certifier_piece.py front PORT [{takes}]')\n")
    _script(venv / "python", f'exec {sys.executable} "$@"')
    env = {"PATH": os.defpath, "HOME": str(tmp_path), "VENV": str(venv.parent), "OUT": str(tmp_path / "out"),
           "LOAD": load, "CERTIFIER_FRONT": front, "SERVE_ARGS": serve_args}
    rc = subprocess.run(["bash", str(HERE.parent / "served_verdicts.sh"), "front-flags"], cwd=tree, env=env,
                        check=False).returncode
    said = (tmp_path / "out" / "verdicts.txt").read_text().splitlines() if (tmp_path / "out" / "verdicts.txt").exists() else []
    assert (rc, said) == ((3, [line]) if line else (0, []))


FAKE_COMPOSITE = '''import json, sys
from pathlib import Path
COMPONENTS = {"onsite": ("verity/pous/p2/v4/run", "verity/wipe/v0"),
              "remote": ("verity/pouw/v1", "verity/sampled-proofs/one-stage/v2")}
if __name__ == "__main__":
    a = sys.argv[1:]
    Path(a[a.index("--private-out") + 1]).mkdir(parents=True)
    Path(a[a.index("--public-out") + 1]).mkdir(parents=True)
    Path(a[a.index("--public-out") + 1], "composite.json").write_text(json.dumps({"product": a[0]}))
    cases = [a[i + 1] for i, x in enumerate(a) if x == "--cases"]
    Path(a[a.index("--private-out") + 1], "argv.json").write_text(json.dumps(cases))
    print("PIECE verity/pous/p2/v4/run accept 0123456789abcdef0123")
    print("NOT_ESTABLISHED missing verity/wipe/v0")
    print(f"VERDICT verity/{a[0]}/v0 reject components")
    sys.exit(1)
'''


def test_the_composite_step_publishes_only_each_cases_verdict_and_each_products_lean_verdict(tmp_path):
    """served_composite.sh with a stand-in lake, served_case.sh, served_cases.py and composite.py: PoUS's case is written
    under the serve dir's private dir with both Lean answers built and passed, and served_cases.py's under it, given no α
    or Γ while ALPHA and GAMMA are unset; each product gets the cases of its own subprotocols; OUT/verdicts.txt gets each
    CASE line's protocol and verdict, each case not written by its protocol, and each product's NOT_ESTABLISHED and
    VERDICT lines, never a PIECE line, a reason, a log or a path; each product's public output is OUT/composite/PRODUCT;
    and without composite.py in the tree it says so and fails, since no Lean answered."""
    tree, bin_, d, out = tmp_path / "tree", tmp_path / "bin", tmp_path / "serve", tmp_path / "out"
    out.mkdir()
    _script(bin_ / "lake", 'mkdir -p .lake/build/bin && touch ".lake/build/bin/$2"')
    _script(bin_ / "systemd-run", "exit 1")
    _script(tree / "benchmarks" / "pous" / "served_case.sh", f'''set -e
[ -x "$POUS_EXE" ] && [ -x "$WIPE_EXE" ] && [ "$1" = {out}/pieces ] && [ "$MEM_GB" = 170 ]
for w in pous/w0 other; do mkdir -p {d}/private/$w; echo '{{"evidence": "beef"}}' > {d}/private/$w/case.json; done
echo '{{"protocol": "verity/pous/p2/v4/run", "verdict": "accept"}}' > {d}/private/pous/w0/answer.json
echo '{{"protocol": "verity/certifier/v9", "verdict": "accept"}}' > {d}/private/other/answer.json
echo "CASE: evidence beef"
echo 'CASE {{"piece": "{d}/private/pous/w0", "protocol": "verity/pous/p2/v4/run", "verdict": "accept"}}' ''')
    for pkg in ("protocols/pous/check", "devices/memory_challenger/lean", "protocols/remote", "protocols/onsite"):
        (tree / "verity" / "core" / pkg).mkdir(parents=True)
    (tree / "benchmarks" / "one_stage").mkdir(parents=True)
    (tree / "benchmarks" / "one_stage" / "composite.py").write_text(FAKE_COMPOSITE)
    v2 = d / "private" / "sampled-proofs-v2"
    (tree / "benchmarks" / "one_stage" / "served_cases.py").write_text(f'''import json, sys
from pathlib import Path
assert sys.argv[1:] == ["--serve-dir", "{d}", "--private", "{d}/private", "--from-division"], sys.argv
Path("{v2}").mkdir(parents=True)
Path("{v2}", "case.json").write_text(json.dumps({{"evidence": "beef"}}))
Path("{v2}", "answer.json").write_text(json.dumps({{"protocol": "verity/sampled-proofs/one-stage/v2", "verdict": "accept"}}))
print("CASE " + json.dumps({{"protocol": "verity/sampled-proofs/one-stage/v2", "verdict": "accept", "dir": "{v2}"}}))
print("CASE " + json.dumps({{"protocol": "verity/registration/v0", "verdict": "accept", "dir": "{d}/private/reg", "alpha_gamma": "235330880/1544757928"}}))
print("NOT_WRITTEN " + json.dumps({{"protocol": "verity/inference-only/v0", "reason": "no alpha 0123456789abcdef"}}))
sys.exit(1)
''')
    env = {"PATH": f"{bin_}:{os.defpath}", "HOME": str(tmp_path), "PY": sys.executable, "TPY": sys.executable}
    run = lambda: subprocess.run(["bash", str(HERE.parent / "served_composite.sh"), str(d), str(out)], cwd=tree, env=env,  # noqa: E731
                                 check=False, capture_output=True, text=True)
    r = run()
    assert r.returncode == 0, r.stderr
    assert (out / "verdicts.txt").read_text().splitlines() == [
        "COMPOSITE CASE verity/pous/p2/v4/run accept", "COMPOSITE CASES: served_case.sh exit 0",
        "COMPOSITE CASE verity/sampled-proofs/one-stage/v2 accept",
        "COMPOSITE CASE verity/registration/v0 accept",
        "COMPOSITE NOT_WRITTEN verity/inference-only/v0",
        "COMPOSITE CASES: served_cases.py exit 1, its log on the node",
        "COMPOSITE remote: NOT_ESTABLISHED missing verity/wipe/v0", "COMPOSITE remote: VERDICT verity/remote/v0 reject components",
        "COMPOSITE onsite: NOT_ESTABLISHED missing verity/wipe/v0", "COMPOSITE onsite: VERDICT verity/onsite/v0 reject components"]
    p = d / "private" / "composite"
    assert json.loads((p / "remote" / "private" / "argv.json").read_text()) == [str(v2)]
    assert json.loads((p / "onsite" / "private" / "argv.json").read_text()) == [str(d / "private" / "pous" / "w0")]
    assert oct((d / "private").stat().st_mode & 0o777) == "0o700"
    assert sorted(out.iterdir()) == [out / "composite", out / "verdicts.txt"]
    assert sorted(f.relative_to(out).as_posix() for f in (out / "composite").rglob("*") if f.is_file()) == [
        "composite/onsite/composite.json", "composite/remote/composite.json"]
    (tree / "benchmarks" / "one_stage" / "composite.py").unlink()
    (out / "verdicts.txt").unlink()
    shutil.rmtree(out / "composite")
    assert run().returncode == 1
    assert (out / "verdicts.txt").read_text() == "COMPOSITE: not run, this tree has no benchmarks/one_stage/composite.py\n"


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
