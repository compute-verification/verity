"""`served_sources.py`'s record on a toy serve dir, replay and gate: PoUW's row is the replay's verdict under the
conditions of record read from the gate's live answers, and no answer file the served node wrote changes it; a copy the
served node handed off that differs from the gate's refuses the row, naming both sides; a later open on the run's
commitment that only the gate's ledger holds is seen; the gate is asked its three read verbs alone, each once; a replay
not held to the inputs here, or whose verifier rejects, is not an accept; and the record says where it was composed, which
part was verified again off the served node and which was not, takes no composite accept, and leaves nothing that names
the window or hashes the coin as drawn under what it publishes. The public record is each row's judgment: exact totals
leave only under the switch, a reason leaves as its kind, Pearl+'s accept beside an audit that did not accept is not a
bare accept, and who controls the host is one of two words, required."""
from __future__ import annotations

import hashlib
import io
import json
import stat
import sys
import tarfile
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import served_sources as SS  # noqa: E402
import test_served_verdicts as T  # noqa: E402

S = T.S
C = "c1" * 64
NODE, HOST = "vy-dc-h200", "vy-cpu"
UNTIED = "weights untied (W v1)"
PIN = hashlib.sha512(T.PUBLIC).hexdigest()
COMMIT = "ab" * 20
LEDGER = [
    {"kind": "period-open", "body": {"period": "p-pc", "lane": "vllm", "cap": 1,
                                     "circuit": hashlib.sha512(T.PEARL_RECORD).hexdigest()}},
    {"kind": "reserve", "session": T.SID, "body": {"by": "vllm"}},
    {"kind": "period-close", "session": None, "body": {"period": "p-pc", "lane": "vllm"}},
    {"kind": "reserve", "session": T.DSID, "body": {"by": "vllm", "families": [["pouw/draw", 1]], "withheld": True}},
    {"kind": "open", "session": T.DSID, "body": {"window": {"root": C}, "withheld": True}},
]
SECOND_OPEN = {"kind": "open", "session": "s-other", "body": {"window": {"root": C}, "withheld": True}}
KEY = json.dumps({"H": 5, "W": 4, "public": T.PUBLIC.hex()}).encode()
#: the files of its piece the served node wrote its answers in, which no row here reads
NODE_ANSWERS = (S.DEVELOPMENT, S.PEARLPLUS_ANSWER, S.PEARLPLUS_LEAN, S.RAN, S.COIN_PUT)


def _jsonl(entries: list[dict]) -> bytes:
    return "".join(json.dumps(e) + "\n" for e in entries).encode()


def _tree(sid: str, files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as t:
        for name, body in files.items():
            info = tarfile.TarInfo(f"{sid}/{name}")
            info.size = len(body)
            t.addfile(info, io.BytesIO(body))
    return buf.getvalue()


class Gate:
    """A stand-in gate: its ledger, pins and each session's tree, and every word it was asked."""

    def __init__(self, ledger: list[dict] | None = None) -> None:
        self.ledger, self.calls = _jsonl(ledger or LEDGER), []

    def __call__(self, *words: str) -> bytes:
        self.calls.append(words)
        if words[0] == "fetch":
            return _tree(words[1], {"out/key.json": KEY, "out/record.json": b"{}"})
        return self.ledger if words[0] == "ledger" else (T.PINS_LINE + "\n").encode()


def _unpack(reads, out: Path, sid: str, tgz: str) -> None:
    (out / tgz).write_bytes(reads("fetch", sid))
    with tarfile.open(out / tgz) as t:
        t.extractall(out / "session", filter="data")
    for name in ("ledger", "pins"):
        (out / f"{name}.jsonl").write_bytes(reads(name))


@pytest.fixture
def world(tmp_path, monkeypatch):
    """A serve dir's copy off the served node (the hand-off, the case, W, the audit of record, which rejected), a replay
    of its case that both verifiers accepted, and the checks of the live sessions standing in: each reads the gate as
    the real one does and refuses nothing. The served node's own answer files all say reject."""
    def coins(served, out, gate, key, main, fetch=None):
        sid = served["coins"]["session"]["session"]
        _unpack(fetch, out, sid, "session.tgz")
        return {"session": sid, "closed": None, "passes": len(served["passes"]), "reason": None}

    def drawn(record, out, gate, key, main, fetch=None):
        sid = json.loads(record.read_text())["draw"]["reserved"]["session"]
        _unpack(fetch, out, sid, f"draw-{sid}.tgz")
        return {"from": "challenger", "session": sid, "reason": None}
    monkeypatch.setattr(S, "pearlplus_coins", coins)
    monkeypatch.setattr(S, "pearlplus_draw", drawn)
    monkeypatch.setattr(S.socket, "gethostname", lambda: HOST)

    d = tmp_path / "serve"
    handed, piece = d / "private" / "compose", d / "private" / "compose" / "pieces" / "pouw"
    for sid in (T.SID, T.DSID):
        (piece / "session" / sid / "out").mkdir(parents=True)
        (piece / "session" / sid / "out" / "key.json").write_bytes(KEY)
    (handed / "compose.env").write_text(f"K=3\nPRODUCT=remote\nALPHA=''\nSERVED_NODE={NODE}\nSERVED_DIR=/srv/serve\n"
                                        "SAMPLED_AUDIT=v3c-w3-4-32-4/audit\nSTEPS=8,12\n")
    coin = json.dumps({"schema": S.COIN_KIND, "commitment": C, "source": T.DRAW_COIN,
                       "passes": [f"None:/srv/serve/pearl/w3/pass-0000{i}/manifest.json sha256:aa" for i in range(2)],
                       "draw": T.DRAW}, indent=1) + "\n"
    (piece / S.COIN_RECORD).write_text(coin)
    (piece / S.COIN_PUT).write_text(json.dumps({"art": "art:bb", "preserved": False,
                                                "sha512": hashlib.sha512(coin.encode()).hexdigest()}))
    (piece / "ledger.jsonl").write_bytes(_jsonl(LEDGER))
    for name in (S.DEVELOPMENT, S.PEARLPLUS_ANSWER, S.PEARLPLUS_LEAN):
        (piece / name).write_text(json.dumps({"verdict": "reject", "reason": "the served node's own"}))
    (piece / S.RAN).write_text(json.dumps({"server": NODE, "rc": 1}))
    (handed / "inference-only.json").write_text(json.dumps({"format": "verity-bench/inference-only-verdict/v4",
                                                            "verdict": "accept", "complies": True, "reason": None}))
    case = d / "private" / "pearlplus" / "case"
    case.mkdir(parents=True)
    (case / SS.PARTS).write_text(json.dumps({"manifests": [
        {"pass_index": i, "matmuls": [{}, {}], "session": {"session": T.SID, "seq": 2}} for i in range(2)]}))
    window = d / "pearlplus" / "w3"
    (window / "tie").mkdir(parents=True)
    (window / "registration.json").write_text('{"toy": "W"}')
    (window / "tie" / S.CIRCUIT_RECORD).write_bytes(T.PEARL_RECORD)
    audit = d / "v3c-w3-4-32-4" / "audit"
    (audit / "window").mkdir(parents=True)
    (audit / S.WINDOW_REGISTRATION).write_text(json.dumps({"format": T.REG.FORMAT_V2, "previous": T.PEARL_W,
                                                           "registered": []}))
    (audit / S.AUDIT_SUMMARY).write_text(json.dumps({"tie": {"record": T.PEARL_W}, "audit": {"why": {
        "lean-verify/m0412": "session files: /srv/serve/v3c-w3-4-32-4/audit/window/session-m0412: no such file"}}}))
    (audit / "window" / "audit.json").write_text(json.dumps({"accepted": False, "checks": {
        "coins": {"ok": False}, "proved": {"ok": False}, "ledger": {"ok": True}}}))

    replay = tmp_path / "replay"
    replay.mkdir()
    tie = {"record": T.PEARL_W, "tables": 6, "weights": UNTIED}
    answer = {"protocol": S.PEARLPLUS_PROTOCOL, "verdict": "accept", "reason": None, "commitment": C,
              "public": {"work": {**T.PEARL_LEAN_WORK, "credited": 5300}, "unit": "MAC", "tie": tie},
              "profile": {"delta": "1/100", "draws": 459, "source": T.DRAW_COIN}}
    lean = {"verdict": "accept", "reason": None, "refused": None, "commitment": C, "source": T.DRAW_COIN, "drawn": 459,
            "verdicts": [], "work": T.PEARL_LEAN_WORK, "tie": tie}
    (replay / S.PEARLPLUS_ANSWER).write_text(json.dumps(answer))
    (replay / S.PEARLPLUS_LEAN).write_text(json.dumps(lean))
    rec = {"schema": SS.REPLAY_SCHEMA, "verifier_commit": COMMIT, "coin_sha512": SS.sha512(piece / S.COIN_RECORD),
           "w_sha512": SS.sha512(window / "registration.json"), "parts_sha512": SS.sha512(case / SS.PARTS),
           "coin_unchanged": True, "exe_sha256": "ee" * 32,
           "python": {"exit": 0, "verdict": "accept", "reason": None, "tie": tie, "draw": "challenger", "line": "ACCEPT"},
           "lean": {"exit": 0, "verdict": "accept", "reason": None, "tie": tie, "differ": []},
           "opened": {"opened": 5, "case_files": 4, "opened_case_files": 4, "opened_outside_the_case": [],
                      "case_files_not_opened": [], "opened_other": ["coins.json"]}}
    (replay / SS.REPLAY).write_text(json.dumps(rec))
    (tmp_path / "published-coin.json").write_text(S.public_doc(coin))
    (tmp_path / "main.txt").write_text("aa\nbb\n")
    return d, replay


def _row(world, tmp_path: Path, gate: Gate, name: str = "private", **over) -> dict:
    d, replay = world
    piece = d / "private" / "compose" / "pieces" / "pouw"
    args = {"replay": replay, "coin": piece / S.COIN_RECORD, "w": d / "pearlplus" / "w3" / "registration.json",
            "case": d / "private" / "pearlplus" / "case", "audit": d / "v3c-w3-4-32-4" / "audit",
            "circuit_record": d / "pearlplus" / "w3" / "tie" / S.CIRCUIT_RECORD, "node": piece,
            "private": tmp_path / name, "reads": SS.GateReads(gate, now=lambda: "T0"), "key": PIN, "lane": "vllm",
            "main": tmp_path / "main.txt", "published_coin": tmp_path / "published-coin.json", "commit": COMMIT,
            "runs": ["r-replay"]}
    return SS.pearlplus_row(**{**args, **over})


def _edit(path: Path, **fields) -> None:
    path.write_text(json.dumps({**json.loads(path.read_text()), **fields}))


def test_pouws_row_is_the_replays_verdict_under_the_gates_live_reads(world, tmp_path):
    """The replay's accept, with the Lean's work, under every condition of record read from the gate: the row says where
    it is from, names the weights untied with what that costs, and holds each of the gate's reads with when it was read
    and the served node's copy held to it. The served node's own answer files, which all say reject, are not read."""
    gate = Gate()
    row = _row(world, tmp_path, gate)
    assert (row["verdict"], row["protocol"], row["commitment"], row["reason"]) == ("accept", S.PEARLPLUS_PROTOCOL, C, None)
    assert row["answer_from"] == SS.REPLAYED_FROM + SS.GATE_HELD and row["said"] == f"{SS.REPLAYED_SAID}, {UNTIED}"
    assert row["quantity"]["work"] == T.PEARL_LEAN_WORK and row["quantity"]["tie"]["record"] == T.PEARL_W
    assert row["weights"] == {"tie": UNTIED, "cost": SS.WEIGHTS_COST} and "same weights" in SS.WEIGHTS_COST
    profile = row["answer"]["profile"]
    assert (profile["session"], profile["draw_session"], profile["period"], profile["key_sha512"]) == (T.SID, T.DSID,
                                                                                                     "p-pc", PIN)
    assert row["runs"] == ["r-replay"] and row["replay"]["verifier_commit"] == COMMIT
    assert row["replay"]["opened"] == {"opened": 5, "case_files": 4, "opened_case_files": 4,
                                       "opened_outside_the_case": 0, "case_files_not_opened": 0, "opened_other": 1}
    reads = row["gate"]
    assert sorted(reads) == sorted([f"fetch {T.DSID}", f"fetch {T.SID}", "ledger", "pins"])
    assert all((r["read"], r["read_utc"]) == (SS.READ_LIVE, "T0") for r in reads.values())
    ledger = _jsonl(LEDGER)
    assert (reads["ledger"]["bytes"], reads["ledger"]["sha512"], reads["ledger"]["lines"]) == (
        len(ledger), hashlib.sha512(ledger).hexdigest(), 5)
    assert reads["ledger"]["node_copy"] == {"lines": 5, "sha512": hashlib.sha512(ledger).hexdigest(),
                                            "held": "line for line the start of the gate's", "gate_lines_after": 0}
    assert reads["pins"]["node_copy"] == "none: the served node's hand-off holds no copy of the host's pins log"
    assert reads[f"fetch {T.SID}"]["node_copy"] == {"files": {"out/key.json": hashlib.sha512(KEY).hexdigest()},
                                                    "held": "each byte for byte the gate's"}
    assert sorted(gate.calls) == sorted([("ledger",), ("pins",), ("fetch", T.SID), ("fetch", T.DSID)])
    private = tmp_path / "private"
    assert stat.S_IMODE((private / "pouw").stat().st_mode) == 0o700
    assert (private / "pouw" / "ledger.jsonl").read_bytes() == ledger


@pytest.mark.parametrize("name", NODE_ANSWERS)
def test_editing_an_answer_file_the_served_node_wrote_leaves_pouws_row_as_it_was(world, tmp_path, name):
    """No answer file of the served node's piece is a source of PoUW's row: with one edited (to an accept of other work,
    or to something that isn't JSON) or gone, the row is the same, byte for byte."""
    d, _replay = world
    f = d / "private" / "compose" / "pieces" / "pouw" / name
    before = json.dumps(_row(world, tmp_path, Gate(), "a"), sort_keys=True)
    _edit(f, verdict="accept", preserved=True, public={"work": {"credited": 10**15}}, rc=0)
    assert json.dumps(_row(world, tmp_path, Gate(), "b"), sort_keys=True) == before
    f.write_text("{")
    assert json.dumps(_row(world, tmp_path, Gate(), "c"), sort_keys=True) == before
    f.unlink()
    assert json.dumps(_row(world, tmp_path, Gate(), "d"), sort_keys=True) == before


@pytest.mark.parametrize("case, said", [
    ("ledger line", ("the served node's copy of the auditor's ledger is not the start of the one the gate reads now: "
                     "line 2 is sha512 ", " in the node's copy and ", " at the gate; the served node's copy (5 lines, "
                     "sha512 ", " and the gate's, read T0 (5 lines, sha512 ")),
    ("ledger longer", ("the served node's copy of the auditor's ledger is longer than the one the gate reads now: the "
                       "served node's copy (6 lines, sha512 ", " and the gate's, read T0 (5 lines, sha512 ")),
    ("key", (f"the served node's copy of session {T.DSID}'s out/key.json is not the gate's: sha512 ",
             " in the node's copy and sha512 ", " in the session's tree the gate gives, read T0")),
    ("extra file", (f"the served node's copy of session {T.SID}'s out/other.json is not the gate's: sha512 ",
                    " in the node's copy and no such file in the session's tree the gate gives, read T0")),
])
def test_a_copy_the_served_node_handed_off_that_differs_from_the_gates_refuses_the_row(world, tmp_path, case, said):
    """A difference between the gate's live answer and the served node's copy of it is the row's refusal, naming both
    sides, not a warning: the row is an error and credits nothing."""
    d, _replay = world
    piece = d / "private" / "compose" / "pieces" / "pouw"
    if case == "ledger line":
        (piece / "ledger.jsonl").write_bytes(_jsonl([LEDGER[0], {**LEDGER[1], "body": {"by": "kernels"}}, *LEDGER[2:]]))
    elif case == "ledger longer":
        (piece / "ledger.jsonl").write_bytes(_jsonl([*LEDGER, SECOND_OPEN]))
    elif case == "key":
        (piece / "session" / T.DSID / "out" / "key.json").write_bytes(KEY + b"\n")
    else:
        (piece / "session" / T.SID / "out" / "other.json").write_text("{}")
    row = _row(world, tmp_path, Gate())
    assert (row["verdict"], row.get("quantity"), row["answer_from"]) == ("error", None, SS.REPLAYED_FROM)
    at = 0
    for part in said:
        assert part in row["reason"][at:], row["reason"]
        at = row["reason"].index(part, at) + len(part)


def test_a_later_open_on_the_runs_commitment_that_only_the_gates_ledger_holds_is_seen(world, tmp_path):
    """The served node's copy of the ledger ends before a second open on the run's commitment: held to the gate's it is
    still its start, and read alone it would show the draw taken once. The conditions of record read the gate's, so the
    row is not an accept, and says which sessions the ledger opens on the commitment."""
    row = _row(world, tmp_path, Gate([*LEDGER, SECOND_OPEN]))
    assert row["verdict"] == "missing" and "quantity" not in row
    assert row["why"] == (f"not PoUW's verdict of record: the ledger opens ['{T.DSID}', 's-other'] on the run's "
                          f"commitment, not the draw session {T.DSID} alone")
    assert row["gate"]["ledger"]["node_copy"]["gate_lines_after"] == 1 and row["gate"]["ledger"]["lines"] == 6
    assert _row(world, tmp_path, Gate(), "alone")["verdict"] == "accept"


@pytest.mark.parametrize("words", [("open", T.SID), ("reserve",), ("draw", T.DSID), ("fetch",), ("fetch", "a b; c"),
                                   ("fetch", T.SID, "x"), ("ledger", "--since", "0"), ("close", T.SID), ()])
def test_the_gate_is_asked_its_three_read_verbs_alone_each_once(words):
    gate = Gate()
    reads = SS.GateReads(gate, now=lambda: "T0")
    with pytest.raises(ValueError, match="the gate is asked to read alone"):
        reads(*words)
    assert gate.calls == []
    assert [reads("ledger"), reads("ledger"), reads("pins"), reads("fetch", T.SID), reads("fetch", T.SID)][0] == gate.ledger
    assert gate.calls == [("ledger",), ("pins",), ("fetch", T.SID)]
    assert sorted(reads.said()) == [f"fetch {T.SID}", "ledger", "pins"]


@pytest.mark.parametrize("case, verdict, said", [
    ("other coin", "error", "the replay was not run on the coin as drawn here: its coin_sha512 is another file's"),
    ("other W", "error", "the replay was not run on the window record W here: its w_sha512 is another file's"),
    ("other case", "error", "the replay was not run on the case's parts here: its parts_sha512 is another file's"),
    ("opened outside", "error", "the replay's Python verifier opened files outside the case"),
    ("opens unrecorded", "error", "the replay's Python verifier opened files outside the case, or its opens weren't"),
    ("Lean differs", "error", "the replay's verifiers accept but didn't end clean: python exit 0, lean exit 2, "
                              "differing on ['work']"),
    ("other tree", "error", "the replay's verifiers are tree cdcdcdcdcdcd's, not abababababab's"),
    ("no replay", "error", "no replay record (replay.json: FileNotFoundError"),
    ("Python rejects", "reject", "tile 7: y is not the product"),
    ("Lean rejects", "reject", "the Lean PouwCheck.PearlPlus verdict is reject: tile 7"),
    ("other published coin", "missing", "not PoUW's verdict of record: the verdict is not held to the coin published at "
                                        "its draw: the coin as drawn, in its public form (sha512 "),
    ("no published coin", "missing", "no published coin record to hold the coin as drawn to (--published-coin)"),
    ("Lean ties the weights", "missing", "the Lean verdict's tie names its weights None, the Python's 'weights untied"),
    ("no key pin", "missing", "not PoUW's verdict of record: the compose holds the coins to no key pin"),
    ("two salt sessions", "error", "the case's passes name 2 salt sessions, not one"),
])
def test_a_replay_not_held_to_the_inputs_here_or_that_rejects_is_no_accept(world, tmp_path, case, verdict, said):
    """Each refusal is the row's own, with why: a replay of other inputs, one whose verifier read outside the case, one
    by another tree or whose verifiers disagree is an error; a verifier's reject is a reject; a condition of record that
    fails leaves the row missing. A reject asks the gate nothing."""
    d, replay = world
    rec, over = replay / SS.REPLAY, {}
    doc = json.loads(rec.read_text())
    if case == "other coin":
        _edit(d / "private" / "compose" / "pieces" / "pouw" / S.COIN_RECORD, source="77" * 64)
    elif case == "other W":
        (d / "pearlplus" / "w3" / "registration.json").write_text('{"toy": "W2"}')
    elif case == "other case":
        _edit(d / "private" / "pearlplus" / "case" / SS.PARTS, more=1)
    elif case == "opened outside":
        _edit(rec, opened={**doc["opened"], "opened_outside_the_case": ["pass-00000/b.bin"]})
    elif case == "opens unrecorded":
        _edit(rec, opened=None)
    elif case == "Lean differs":
        _edit(rec, lean={**doc["lean"], "exit": 2, "differ": ["work"]})
    elif case == "other tree":
        _edit(rec, verifier_commit="cd" * 20)
    elif case == "no replay":
        rec.unlink()
    elif case == "Python rejects":
        _edit(replay / S.PEARLPLUS_ANSWER, verdict="reject", reason="tile 7: y is not the product", public=None,
              profile=None)
        _edit(rec, python={**doc["python"], "exit": 1, "verdict": "reject"}, lean={**doc["lean"], "exit": 2, "verdict": None})
    elif case == "Lean rejects":
        _edit(replay / S.PEARLPLUS_LEAN, verdict="reject", reason="tile 7")
        _edit(rec, lean={**doc["lean"], "exit": 1, "verdict": "reject"})
    elif case == "other published coin":
        (tmp_path / "published-coin.json").write_text("{}\n")
    elif case == "no published coin":
        over["published_coin"] = None
    elif case == "Lean ties the weights":
        _edit(rec, lean={**doc["lean"], "tie": {"record": T.PEARL_W}})
    elif case == "no key pin":
        over["key"] = None
    else:
        _edit(d / "private" / "pearlplus" / "case" / SS.PARTS, manifests=[
            {"matmuls": [{}], "session": {"session": s}} for s in (T.SID, T.DSID)])
        _edit(rec, parts_sha512=SS.sha512(d / "private" / "pearlplus" / "case" / SS.PARTS))
    gate = Gate()
    row = _row(world, tmp_path, gate, **over)
    assert row["verdict"] == verdict and said in (row.get("reason") or row.get("why")), row
    assert row["answer_from"] == SS.REPLAYED_FROM and row.get("quantity") is None
    assert gate.calls == [] or verdict == "missing"


def _record(world, tmp_path: Path, monkeypatch, gate: Gate, capsys, *more: str) -> tuple[int, Path, Path, str]:
    d, replay = world
    monkeypatch.setattr(S, "_gate_out", lambda g, *words: gate(*words) if g == T.GATE else b"")
    (tmp_path / "causes.json").write_text(json.dumps(["the coin link's frame limit", "seven members out of memory"]))
    out, private = tmp_path / "out", tmp_path / "kept"
    rc = SS.main(["record", "--serve-dir", str(d), "--replay", str(replay), "--private", str(private), "--out", str(out),
                  "--gate", T.GATE, "--key-sha512", PIN, "--main", str(tmp_path / "main.txt"), "--controlled-by",
                  "developer", "--published-coin", str(tmp_path / "published-coin.json"),
                  "--coin-put", str(d / "private" / "compose" / "pieces" / "pouw" / S.COIN_PUT),
                  "--sampled-causes", str(tmp_path / "causes.json"), "--verifier-commit", COMMIT, "--run", "r-compose",
                  "--replay-run", "r-replay", "--audit-run", "r-audit", "--tree", "ee" * 20, *more])
    return rc, out, private, capsys.readouterr().out


MEMBER = "lean-verify/m0412: session files: /srv/serve/v3c-w3-4-32-4/audit/window/session-m0412: no such file"
PARTS_SAID = ("Pearl+ re-verified off the served node (Python and Lean): accept, tie unaudited; the inference-only verdict "
              "is the served node's own answer, not recomputed, because the block count stays on that node; the "
              "sampled-proofs audit of record rejected.")


def test_the_record_says_where_it_was_composed_and_which_part_was_verified_again(world, tmp_path, monkeypatch, capsys):
    """The record over the replayed Pearl+, the rejected audit of record and the served node's inference-only answer:
    composed off the served node on a host it names as the developer's side and not the auditor's; rejected, because
    the audit of record rejected, with the lane's causes beside the audit's own reason; Pearl+'s accept, whose tie reads
    that audit, is `ACCEPT_UNAUDITED` and counts as no accept; its sentence says Pearl+ alone was verified again; it is
    no composite and writes no block record. The full record, with the audit's own words, each hash and the coin as
    drawn, stays in the private directory."""
    d, _replay = world
    rc, out, private, said = _record(world, tmp_path, monkeypatch, Gate(), capsys)
    assert rc == 0, said
    kept = json.loads((private / "record" / "verdicts.json").read_text())
    where = kept["composed_on"]
    assert (where["host"], where["served_node"], where["where"], where["controlled_by"], where["from"], where["tree"]) == (
        HOST, False, "off the served node, on a CPU node the lane runs; not the auditor's host",
        "the developer's side, which also ran the served node; no auditor controls this host", SS.SOURCES_FROM, "ee" * 20)
    composed = kept["composed"]
    assert composed["verdict"] == "reject" and composed["composite"] == SS.NOT_COMPOSITE
    assert composed["parts"] == PARTS_SAID and composed["reverified_off_served_node"] == ["PoUW"]
    assert composed["reason"] == (f"sampled proofs: reject: audit record: coins, proved; {MEMBER}; PoUW: accept, tie "
                                  f"unaudited, {SS.REPLAYED_SAID}, {UNTIED}")
    assert sorted(kept["protocols"]) == ["integrity", "work"]
    audit, work = kept["protocols"]["integrity"], kept["protocols"]["work"]
    assert (audit["verdict"], audit["answer_from"], audit["runs"]) == ("reject", SS.AUDIT_FROM, ["r-audit"])
    assert audit["reason"] == f"audit record: coins, proved; {MEMBER}"
    assert audit["files"] == {name: SS.sha512(d / "v3c-w3-4-32-4" / "audit" / name)
                              for name in ("summary.json", "window/audit.json")}
    assert audit["causes"] == {"by": SS.CAUSES_BY, "of": ["the coin link's frame limit", "seven members out of memory"]}
    assert (work["verdict"], work["runs"], work["tie_rests_on"]) == (SS.ACCEPT_UNAUDITED, ["r-compose", "r-replay"],
                                                                     SS.TIE_UNAUDITED)
    assert work["quantity"]["work"] == T.PEARL_LEAN_WORK and work["gate"]["ledger"]["sha512"]
    assert work["coin"] == {"published_sha512": SS.sha512(tmp_path / "published-coin.json"), "as_drawn": SS.COIN_HELD,
                            "reason": None}
    assert kept["inference_only"] == {
        "source": "served-node", "answer_from": SS.INFERENCE_FROM["served-node"], "format":
        "verity-bench/inference-only-verdict/v4", "verdict": "accept", "complies": True, "reason": None, "failure": None,
        "published_sha512": SS.sha512(d / "private" / "compose" / "inference-only.json")}
    assert kept["notes"] == ["the coin record's own artifact was never preserved in the evidence store (coin-put.json: "
                             "preserved: false); the store holds the redacted coin record in the run's published record"]
    assert (kept["run"], kept["node"], kept["servers"]["work"]) == ("r-compose", NODE, NODE)
    assert kept["window"] == 3 and stat.S_IMODE(private.stat().st_mode) == 0o700
    assert "/w3/" in (private / "pouw" / S.COIN_RECORD).read_text()
    assert SS.sha512(d / "private" / "compose" / "pieces" / "pouw" / S.COIN_RECORD) not in json.dumps(kept)
    assert not (out / "record" / "block-record.json").exists()


def test_the_public_record_is_each_rows_judgment(world, tmp_path, monkeypatch, capsys):
    """What leaves by default: the record, its line and the redacted coin record, and not the two Pearl+ answers. Each
    row is its verdict with its ε and δ, where it is from and what it rests on: no exact total, no hash of a private
    file, the gate's reads as when each was read and how the node's copy was held. The rejected audit's reason is its
    kind: no member's index and no session path, in the row, the composed reason or the line. Nothing names the window
    or hashes the coin as drawn."""
    d, _replay = world
    rc, out, private, said = _record(world, tmp_path, monkeypatch, Gate(), capsys)
    assert rc == 0, said
    assert sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()) == [
        "pieces/pouw/coin.json", "record/verdicts.json", "verdicts.txt"]
    rep = json.loads((out / "record" / "verdicts.json").read_text())
    kept = json.loads((private / "record" / "verdicts.json").read_text())
    audit, work = rep["protocols"]["integrity"], rep["protocols"]["work"]
    assert audit == {"protocol": kept["protocols"]["integrity"]["protocol"], "owner": kept["protocols"]["integrity"]["owner"],
                     "verdict": "reject",
                     "commitment": kept["protocols"]["integrity"]["commitment"], "delta": None, "epsilon": None,
                     "runs": ["r-audit"], "answer_from": SS.AUDIT_FROM, "reason": "audit record: coins, proved",
                     "detail": SS.DETAIL_HELD,
                     "causes": {"by": SS.CAUSES_BY, "of": ["the coin link's frame limit", "seven members out of memory"]}}
    assert sorted(work) == sorted(["protocol", "owner", "verdict", "commitment", "delta", "epsilon", "runs", "answer_from",
                                   "said", "weights", "tie_rests_on", "coin", "reason", "totals", "gate", "replay"])
    assert (work["verdict"], work["epsilon"], work["delta"], work["reason"], work["totals"]) == (
        SS.ACCEPT_UNAUDITED, "1/100", "1/100", None, SS.TOTALS_HELD)
    assert work["gate"]["ledger"] == {"read": SS.READ_LIVE, "read_utc": work["gate"]["ledger"]["read_utc"],
                                      "node_copy": "line for line the start of the gate's"}
    assert work["gate"][f"fetch {T.SID}"]["node_copy"] == "each byte for byte the gate's"
    assert work["replay"] == {"schema": SS.REPLAY_SCHEMA, "verifier_commit": COMMIT, "python": {"exit": 0, "verdict": "accept"},
                              "lean": {"exit": 0, "verdict": "accept", "differ": []}}
    assert rep["composed"]["reason"] == (f"sampled proofs: reject: audit record: coins, proved; PoUW: accept, tie "
                                         f"unaudited, {SS.REPLAYED_SAID}, {UNTIED}")
    assert rep["composed"]["parts"] == PARTS_SAID and rep["composed"]["verdict"] == "reject" and "window" not in rep
    line = json.loads(said.splitlines()[0].removeprefix("RECORD "))
    assert (line["composed"], line["parts"], line["verdicts"]["PoUW"], line["verdicts"]["sampled proofs"]) == (
        "reject", PARTS_SAID, SS.ACCEPT_UNAUDITED, "reject")
    assert line["reason"] == rep["composed"]["reason"] and "block_record" not in line
    assert said.splitlines()[1].startswith("RECORD-FILES 3 under the published directory: ")
    drawn = SS.sha512(d / "private" / "compose" / "pieces" / "pouw" / S.COIN_RECORD)
    private_hashes = [drawn, kept["protocols"]["work"]["gate"]["ledger"]["sha512"], kept["protocols"]["work"]["replay"]["w_sha512"],
                      *kept["protocols"]["integrity"]["files"].values()]
    for p in out.rglob("*"):
        if p.is_file():
            text = p.read_text()
            assert "w3" not in text and '"window": 3' not in text and "m0412" not in text and "session-m" not in text, p
            assert '"credited"' not in text and "5300" not in text and "5400" not in text and '"macs"' not in text, p
            assert not any(h[:16] in text for h in private_hashes), p
    assert (out / "pieces" / "pouw" / "coin.json").read_bytes() == (tmp_path / "published-coin.json").read_bytes()


def test_exact_totals_leave_only_under_the_switch(world, tmp_path, monkeypatch, capsys):
    rc, out, _private, said = _record(world, tmp_path, monkeypatch, Gate(), capsys, "--exact-totals", "public")
    assert rc == 0, said
    assert sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()) == [
        "pieces/pouw/coin.json", "pieces/pouw/pearlplus-answer.json", "pieces/pouw/pearlplus-lean.json",
        "record/verdicts.json", "verdicts.txt"]
    work = json.loads((out / "record" / "verdicts.json").read_text())["protocols"]["work"]
    assert work["quantity"]["work"] == T.PEARL_LEAN_WORK and work["answer"]["verdict"] == "accept" and "totals" not in work
    assert work["verdict"] == SS.ACCEPT_UNAUDITED
    with pytest.raises(ValueError, match="exact totals are private or public"):
        SS.public_row({"verdict": "accept"}, "some")


@pytest.mark.parametrize("reason, kind", [
    (f"audit record: coins, proved (drawn units not proved: T 4-9); {MEMBER}; and 4 more", "audit record: coins, proved"),
    ("before the draw: ledger, registration", "before the draw: ledger, registration"),
    ('refused: {"window": "/srv/serve/w3"}', "refused"),
    ("RuntimeError: the records audit left no summary under /srv/serve/w3/audit: [Errno 2]", "RuntimeError"),
    ("the served node's copy of the auditor's ledger is not the start of the one the gate reads now: line 2 is sha512 "
     "0123456789abcdef... in the node's copy", "the served node's copy of the auditor's ledger is not the start of the "
     "one the gate reads now"),
    ("not PoUW's verdict of record: the verdict is not held to the coin published at its draw: the coin as drawn, in its "
     "public form (sha512 0123...)", "not PoUW's verdict of record: the verdict is not held to the coin published at its "
     "draw: the coin as drawn, in its public form"),
    ("no replay record (replay.json: FileNotFoundError: /srv/replay.json)", "no replay record"),
    ("tile 7: y is not the product", SS.NO_KIND),
    ("pass 153 matmul 12", SS.NO_KIND),
    (None, SS.NO_KIND),
])
def test_a_reason_leaves_as_its_kind(reason, kind):
    """The public form of a reason is its leading plain clauses: nothing with a digit, a path, a quote or a bracket."""
    assert SS.kind_of(reason) == kind
    row = SS.public_row({"verdict": "reject", "reason": reason, "quantity": None, "files": {"summary.json": "ab" * 64}})
    assert row == ({"verdict": "reject", "reason": None} if reason is None else
                   {"verdict": "reject", "reason": kind, "detail": SS.DETAIL_HELD})


def test_a_hash_of_a_file_published_redacted_is_refused_before_anything_leaves(world, tmp_path):
    """`write` refuses a public file holding the SHA-512 or SHA-256 of a file whose redacted form is published (the coin
    as drawn; the private record itself; a gate's answer that names a window), whole or by its first 16 hex digits,
    and writes nothing under what leaves; a file that is its own public form is not withheld."""
    d, _replay = world
    coin = d / "private" / "compose" / "pieces" / "pouw" / S.COIN_RECORD
    work = _row(world, tmp_path, Gate(), withheld=(held := SS.Withheld()))
    assert held.found(SS.sha512(coin)) == ["the coin as drawn"]
    assert held.found(hashlib.sha256(coin.read_bytes()).hexdigest()[:16] + "...") == ["the coin as drawn"]
    assert held.found(hashlib.sha512(_jsonl(LEDGER)).hexdigest()) == [] and held.found(hashlib.sha512(KEY).hexdigest()) == []
    held.add("the gate's answer to ledger", b'{"window": 3, "kind": "open"}\n')
    assert held.found(hashlib.sha512(b'{"window": 3, "kind": "open"}\n').hexdigest()[:16]) == ["the gate's answer to ledger"]
    policy = SS.inference_only(d / "private" / "compose" / "inference-only.json")
    rows = {"integrity": SS.integrity_row(d / "v3c-w3-4-32-4" / "audit"), "work": work}
    for n, leak in enumerate((SS.sha512(coin), SS.sha512(coin)[:16] + "...")):
        report, line = SS.compose(rows, policy, window=3, steps=(8, 12), node=NODE, controlled_by="developer",
                                  product="remote", notes=[f"the coin as drawn is sha512 {leak}"])
        with pytest.raises(RuntimeError, match="would leave holding a hash of the coin as drawn"):
            SS.write(tmp_path / f"out{n}", tmp_path / f"kept{n}", report, line, {S.COIN_RECORD: coin}, withheld=held)
        assert not (tmp_path / f"out{n}").exists()
    report, line = SS.compose(rows, policy, window=3, steps=(8, 12), node=NODE, controlled_by="developer", product="remote")
    assert SS.write(tmp_path / "out-ok", tmp_path / "kept-ok", report, line, {S.COIN_RECORD: coin}, withheld=held) == [
        "pieces/pouw/coin.json", "record/verdicts.json", "verdicts.txt"]
    full = (tmp_path / "kept-ok" / "record" / "verdicts.json").read_text()
    assert '"window": 3' in full
    cites = {**line, "notes": [f"the full record is sha512 {hashlib.sha512(full.encode()).hexdigest()}"]}
    with pytest.raises(RuntimeError, match="verdicts.txt would leave holding a hash of the private record"):
        SS.write(tmp_path / "out-rec", tmp_path / "kept-ok", report, cites, {})
    assert not (tmp_path / "out-rec").exists()


def test_who_controls_the_host_is_one_of_two_words_and_required(world, tmp_path, monkeypatch, capsys):
    d, replay = world
    base = ["record", "--serve-dir", str(d), "--replay", str(replay), "--private", str(tmp_path / "kept"), "--out",
            str(tmp_path / "out"), "--gate", T.GATE, "--key-sha512", PIN, "--main", str(tmp_path / "main.txt")]
    for more in ([], ["--controlled-by", "the lane"], ["--controlled-by", "developer", "--exact-totals", "some"]):
        with pytest.raises(SystemExit):
            SS.main([*base, *more])
        assert not (tmp_path / "out").exists() and not (tmp_path / "kept").exists()
    capsys.readouterr()
    work = _row(world, tmp_path, Gate())
    policy = SS.inference_only(d / "private" / "compose" / "inference-only.json")
    with pytest.raises(ValueError, match="who controls the composing host is one of auditor, developer"):
        SS.compose({"integrity": T._accepted(), "work": work}, policy, window=3, steps=(8, 12), node=NODE,
                   controlled_by="the lane", product="remote")
    report, _line = SS.compose({"integrity": T._accepted(), "work": work}, policy, window=3, steps=(8, 12), node=NODE,
                               controlled_by="auditor", product="remote")
    assert (report["composed_on"]["where"], report["composed_on"]["controlled_by"]) == (
        "off the served node, on an auditor's host",
        "an auditor's host: the developer's side controls neither it nor what runs on it")
    assert report["composed"]["verdict"] == "pending"


@pytest.mark.parametrize("case", ["served node", "private under out", "private exists"])
def test_the_record_is_refused_on_the_served_node_or_where_what_stays_would_leave(world, tmp_path, monkeypatch, capsys,
                                                                                    case):
    if case == "served node":
        monkeypatch.setattr(S.socket, "gethostname", lambda: NODE)
        more, why = (), f"a record composed off the served node is not composed on it: this host is {NODE}"
    elif case == "private under out":
        more, why = ("--private", str(tmp_path / "out" / "kept")), "--private and --out must be apart"
    else:
        (tmp_path / "kept").mkdir()
        more, why = (), "FileExistsError"
    rc, out, _private, said = _record(world, tmp_path, monkeypatch, Gate(), capsys, *more)
    assert rc == 2 and said.startswith("RECORD refused: ") and why in said
    assert not out.exists()


def test_an_accept_of_every_row_composes_pending_and_a_part_not_accepted_is_said(world, tmp_path):
    """Sampled proofs and PoUW both accepting is still no composite accept here, since the inference-only verdict is not
    verified again off the served node; and the sentence says of each part what it is: Pearl+ not accepted, the served
    node's inference-only verdict when it isn't an accept, an audit that gave no verdict."""
    work = _row(world, tmp_path, Gate())
    policy = SS.inference_only(world[0] / "private" / "compose" / "inference-only.json")
    report, line = SS.compose({"integrity": T._accepted(), "work": work}, policy, window=3, steps=(8, 12), node=NODE,
                              controlled_by="developer", product="remote")
    assert (report["composed"]["verdict"], line["composed"]) == ("pending", "pending")
    assert report["protocols"]["work"]["verdict"] == "accept" and line["verdicts"]["PoUW"] == "accept"
    assert report["composed"]["reason"].startswith(f"{SS.NOT_ALL_HERE}; sampled proofs: accept")
    assert report["composed"]["parts"].startswith("Pearl+ re-verified off the served node (Python and Lean); the ")
    assert report["composed"]["parts"].endswith("the sampled-proofs audit of record accepted.")
    assert "tie_rests_on" not in report["protocols"]["work"] and "steps" not in line and "check" not in line
    missing = {"verdict": "missing", "owner": "compute-accounting", "why": "no", "answer_from": SS.REPLAYED_FROM}
    said, again = SS.parts_said({"integrity": {"verdict": "error"}, "work": missing}, {**policy, "verdict": "reject"})
    assert again == [] and said == (
        "Pearl+ not accepted off the served node (missing: no); the inference-only verdict (reject) is the served node's "
        "own answer, not recomputed, because the block count stays on that node; the sampled-proofs audit of record gave "
        "no verdict (error).")
    gone = SS.inference_only(tmp_path / "none.json")
    assert (gone["verdict"], gone["source"]) == ("error", "served-node") and "FileNotFoundError" in gone["reason"]
    with pytest.raises(KeyError):
        SS.inference_only(tmp_path / "none.json", "recomputed")


def test_the_coins_note_is_the_published_put_and_never_pouws_row(tmp_path):
    put = tmp_path / S.COIN_PUT
    assert "is not known here (coin-put.json: FileNotFoundError)" in SS.coin_note(put)
    assert "is not known here (no coin-put.json given)" in SS.coin_note(None)
    put.write_text(json.dumps({"preserved": True, "art": "art:bb"}))
    assert SS.coin_note(put) == ("the coin record's own artifact is preserved in the evidence store (coin-put.json: "
                                 "preserved: true, art:bb)")
