"""v2_case.py: sampled proofs v2's case of a served window from its kept files. The rebuild reads what a served run's
``m0/compose.log`` (``fixtures/served-window-m0``, a served H200 run's) records, the case names the auditor's challenger
by its key record and carries the window's kept release of the draw's coins, a release naming another key is refused,
and the case is written private as `composite.py` reads it, or, when a blob has no file or the release is refused, only
``missing.json``, which `composite.py` leaves out."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import stat
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import a5
import composite as C
import v2_case as V

from verity.core.protocols.sampled_proofs import challenger as Ch

LOG = Path(__file__).resolve().parent / "fixtures" / "served-window-m0" / "compose.log"
_spec = importlib.util.spec_from_file_location("one_stage_replay_v2_case", Path(V.__file__).parent / "replay.py")
REPLAY = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = REPLAY
_spec.loader.exec_module(REPLAY)


@pytest.fixture
def RP(monkeypatch):
    """This suite's `replay`, which a5 imports by that name, under that name for one test only (`network_traces` has a
    `replay` of its own)."""
    monkeypatch.setitem(sys.modules, "replay", REPLAY)
    return REPLAY


def test_the_compose_log_names_the_rebuild():
    c = V.compose_command(LOG.read_text())
    assert c["subcircuit"] == "silu-mul-v2/i8192/bf16" and c["set"] == "silumul-v2-i8192"
    assert c["pin"].startswith("13d14493cb78f7ad") and len(c["pin"]) == 128
    assert c["partition"].startswith("e94be647") and c["program"].startswith("1831c097")


def _v2(tmp_path: Path, blobs: list[Path]) -> dict:
    return {"blobs": [str(b) for b in blobs], "commitment": b"registration".hex(), "evidence": b'{"proofs":{}}'.hex(),
            "params": {"proofs": {"circuit": "00" * 64, "registered": None}}}


def test_a_case_with_its_blobs_is_written_private(tmp_path):
    blob = tmp_path / "circuit.txt"
    blob.write_text("gates")
    out = tmp_path / "v2"
    assert V.write(out, _v2(tmp_path, [blob]), "kept") == []
    assert stat.S_IMODE(out.stat().st_mode) == 0o700
    assert {f.name: stat.S_IMODE(f.stat().st_mode) for f in out.iterdir()} == dict.fromkeys(V.CASE_FILES[:4], 0o600)
    assert (out / "commitment.bin").read_bytes() == b"registration"
    assert json.loads((out / "case.json").read_text())["blobs"] == [str(blob)]


def kept_window(root: Path, RP) -> Path:
    """A served window's kept files as `prove_path/run.sh` keeps them: one member, m0, its one session."""
    w = root / "window"
    (w / "m0").mkdir(parents=True)
    (w / "m0" / "compose.log").write_text(LOG.read_text())
    (w / "m0" / "circuit.txt").write_text("gates")
    (w / "m0" / "pub-4.bin").write_bytes(b"public")
    s = w / "session-m0" / "sessions" / "a"
    s.mkdir(parents=True)
    (s / "session.json").write_bytes(b"{}")
    (s / "x.bin").write_bytes(b"rep")
    name = hashlib.sha512(b"{}").hexdigest()
    evidence = {"draw": {"units": [1]}, "sessions": [name], "verdicts": {}}
    (w / "case.json").write_text(json.dumps({"commitment": "ab", "evidence": RP.DT.canonical(evidence).encode().hex(),
                                             "params": {"delta": "1/100"}}))
    (w / "partition.json").write_text(json.dumps({"program": "p", "rule": "subset:1"}))
    return w


COINS = {"ledger": ["{}"], "main": ["cc" * 20], "pins": ["{}"],
         "segment": {"label": "draw", "start": 1, "after": "dd" * 64, "openings": [{"coin": "ee" * 64}]},
         "session": {"close": {}, "hellos": [], "log": [], "record": {}}}


def key_record(tmp_path: Path, public: bytes) -> Path:
    """The auditor's challenger-key/v1 record of the key whose public key is `public`, as a file."""
    f = tmp_path / f"challenger-key-{public[:1].hex()}.json"
    f.write_text(json.dumps({"schema": Ch.KEY_RECORD_SCHEMA, "W": 4, "H": 5, "pubkey": public.hex(),
                             "pubkey_sha512": hashlib.sha512(public).hexdigest()}))
    return f


def V2CASE(out: str) -> dict:
    return json.loads(next(x for x in out.splitlines() if x.startswith("V2CASE ")).split(" ", 1)[1])


def test_a_kept_release_names_the_auditors_challenger_and_carries_the_draws_coins(tmp_path, RP, capsys):
    """A window kept with the challenger's release of its draw's coins (`a5.RELEASE`) gets, under the auditor's key
    record, a case naming that challenger and the release's window root, its evidence carrying the coins; one kept
    without, or given no record, names none, which v2 refuses (NO_CHALLENGER), and its V2CASE line says so."""
    w = kept_window(tmp_path, RP)
    got, how, n, why = V.case(w, tmp_path / "scratch")
    assert (how, n, why) == ("kept", 1, "none: the window kept no release")
    assert "challenger" not in got["params"] and "window_root" not in got["params"]
    assert "coins" not in json.loads(bytes.fromhex(got["evidence"]))
    record = key_record(tmp_path, b"\xaa" * Ch.PUBLIC_BYTES)
    key = Ch.key_of_record(json.loads(record.read_text()))
    drawn = a5.drawn_by({**COINS, "key": key.to_json()}, "ab" * 64)
    (w / a5.RELEASE).write_text(json.dumps(drawn))
    got, _, _, why = V.case(w, tmp_path / "scratch")
    assert why.startswith("none: no --challenger-key") and "challenger" not in got["params"]
    got, _, _, why = V.case(w, tmp_path / "scratch", key)
    assert why == "the auditor's record's key"
    assert (got["params"]["challenger"], got["params"]["window_root"]) == ({"key": key.to_json(), "label": "draw"},
                                                                         "ab" * 64)
    assert json.loads(bytes.fromhex(got["evidence"]))["coins"] == drawn["evidence"]["coins"]
    capsys.readouterr()
    assert V.main(["--window", str(w), "--out", str(tmp_path / "v2"), "--scratch", str(tmp_path / "scratch"),
                   "--challenger-key", str(record)]) == 0
    assert V2CASE(capsys.readouterr().out)["challenger"] == "the auditor's record's key"
    assert json.loads((tmp_path / "v2" / "params.json").read_text())["challenger"] == drawn["params"]["challenger"]


def test_a_release_naming_another_key_than_the_auditors_record_is_refused(tmp_path, RP, capsys):
    """The key is the auditor's record's, never the release's, which the served node writes: a release naming another
    key (the prover's own, under which it could sign its own coins) is refused, and the window's case directory keeps
    only ``missing.json``, so `composite.py` leaves v2 out and the product's Lean names it missing."""
    w = kept_window(tmp_path, RP)
    mine, theirs = b"\xaa" * Ch.PUBLIC_BYTES, b"\xbb" * Ch.PUBLIC_BYTES
    record = key_record(tmp_path, mine)
    prover = {"W": 4, "H": 5, "public": theirs.hex()}
    (w / a5.RELEASE).write_text(json.dumps(a5.drawn_by({**COINS, "key": prover}, "ab" * 64)))
    with pytest.raises(V.Refused, match="names another challenger key than the auditor's record's"):
        V.case(w, tmp_path / "scratch", Ch.key_of_record(json.loads(record.read_text())))
    out = tmp_path / "v2"
    assert V.write(out, _v2(tmp_path, [record]), "kept") == []
    capsys.readouterr()
    assert V.main(["--window", str(w), "--out", str(out), "--scratch", str(tmp_path / "scratch"),
                   "--challenger-key", str(record)]) == 2
    assert "another challenger key" in V2CASE(capsys.readouterr().out)["refused"]
    assert [f.name for f in out.iterdir()] == ["missing.json"]
    assert json.loads((out / "missing.json").read_text())["why"].startswith("refused: ")
    assert C.case_dirs([out]) == []
    with pytest.raises(SystemExit):
        V.main(["--window", str(w), "--out", str(out), "--scratch", str(tmp_path / "scratch"),
                "--challenger-key", str(tmp_path / "no-record.json")])


def test_a_blob_with_no_file_leaves_only_missing(tmp_path):
    out, kept, gone = tmp_path / "v2", tmp_path / "pub-3.bin", tmp_path / "m0" / "circuit.txt"
    kept.write_bytes(b"public")
    assert V.write(out, _v2(tmp_path, [kept]), "kept") == []
    assert V.write(out, _v2(tmp_path, [gone]), "rebuild failed (exit 1)") == [str(gone)]
    assert [f.name for f in out.iterdir()] == ["missing.json"]
    why = json.loads((out / "missing.json").read_text())
    assert why["protocol"] == V.ID and str(gone) in why["why"] and "rebuild failed" in why["why"]
    assert C.case_dirs([out]) == []
