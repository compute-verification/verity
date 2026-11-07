"""v2_case.py: sampled proofs v2's case of a served window from its kept files. The rebuild reads what a served run's
``m0/compose.log`` (``fixtures/served-window-m0``, a served H200 run's) records, the case names the auditor's challenger
by its key record and carries the window's kept release of the draw's coins, a release naming another key is refused,
the public file is named by a5's pin of it at staging, before the draw, a public file changed after the draw and a
window with no pin refused, the case reads the drawn member's files whichever member it is, and the case is written private as `composite.py` reads it, or, when a blob has no file or
the case is refused, only ``missing.json``, which `composite.py` leaves out."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import stat
import sys
import types
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


PUBLIC = b"header\npublic"


def public_of(k: int) -> bytes:
    """Member `k`'s public file: `PUBLIC` for m0."""
    return PUBLIC + (f" m{k}".encode() if k else b"")


def kept_window(root: Path, RP, k: int = 0) -> Path:
    """A served window's kept files as `prove_path/run.sh` keeps them: members m0 to m<k> of one template, its circuit
    composed once at m0 (``compose.log``, `a5.audit`) and linked at the drawn member m<k>, each member's public file
    and a5's pin of them at staging (`a5.public_pin`), the members in ``summary.json``, and m<k>'s one session."""
    w = root / "window"
    (w / "m0").mkdir(parents=True)
    (w / "m0" / "compose.log").write_text(LOG.read_text())
    (w / "m0" / "circuit.txt").write_text("gates")
    staged, publics = [], {}
    for i in range(k + 1):
        pub = w / f"m{i}" / "pub-4.bin"
        pub.parent.mkdir(exist_ok=True)
        pub.write_bytes(public_of(i))
        staged.append({"k": i, "m": types.SimpleNamespace(key=f"m{i}-key", tables=(("t", 4),)),
                       "st": {"public": hashlib.sha512(public_of(i)).hexdigest()}})
        publics[f"m{i}-key"] = a5.body(pub)
    if k:
        os.link(w / "m0" / "circuit.txt", w / f"m{k}" / "circuit.txt")
    (w / a5.PIN).write_text(json.dumps(a5.public_pin(staged, publics)))
    (w / "summary.json").write_text(json.dumps({"members": [{"k": i, "template": "A", "n": 4, "base": 4 * i}
                                                            for i in range(k + 1)]}))
    s = w / f"session-m{k}" / "sessions" / "a"
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


@pytest.mark.parametrize("how, why", [
    ("changed after the draw", f"the kept public file isn't, by SHA-512, the one {a5.PIN} pinned for m0"),
    ("no pin", f"the window kept no {a5.PIN} naming m0"),
])
def test_a_public_file_not_the_one_pinned_at_staging_is_refused(tmp_path, RP, capsys, how, why):
    """θ names the public file by a5's pin of it at staging, before the draw (`a5.PIN`), never by the kept file's
    SHA-512 read after it, which the prover could choose: a public file changed after the draw, or a window that kept no
    pin, is refused, and the case directory keeps only ``missing.json``."""
    w, out = kept_window(tmp_path, RP), tmp_path / "v2"
    argv = ["--window", str(w), "--out", str(out), "--scratch", str(tmp_path / "scratch")]
    assert V.main(argv) == 0
    assert json.loads((out / "params.json").read_text())["proofs"]["public"] == hashlib.sha512(PUBLIC).hexdigest()
    if how == "no pin":
        (w / a5.PIN).unlink()
    else:
        (w / "m0" / "pub-4.bin").write_bytes(b"header\nanother public")
    with pytest.raises(V.Refused, match=why):
        V.case(w, tmp_path / "scratch")
    capsys.readouterr()
    assert V.main(argv) == 2
    assert why in V2CASE(capsys.readouterr().out)["refused"]
    assert [f.name for f in out.iterdir()] == ["missing.json"] and C.case_dirs([out]) == []


def test_the_case_reads_the_drawn_members_files_whichever_member_it_is(tmp_path, RP):
    """A kept window whose one drawn member is m2: the case reads m2's session, public file, held to m2's pin, and
    circuit file, or its template's first member's where m2 kept no link, and the compose.log of that first member,
    where a5 composed the template's circuit once (`summary.json`'s members). A window that kept sessions of two
    members or of none, no public file of the drawn member, or no compose.log a member of the drawn one's template
    kept, is refused."""
    w, scratch = kept_window(tmp_path, RP, k=2), tmp_path / "scratch"
    got, how, n, _ = V.case(w, scratch)
    assert (how, n, got["params"]["proofs"]["circuit"]) == ("kept", 1, V.compose_command(LOG.read_text())["pin"])
    assert got["params"]["proofs"]["public"] == hashlib.sha512(public_of(2)).hexdigest()
    assert got["params"]["proofs"]["members"] == [["A/c0", 4], ["A/c1", 4], ["A/c2", 4]]
    assert got["blobs"][:2] == [str(w / "m2" / "circuit.txt"), str(w / "m2" / "pub-4.bin")]
    (w / "m2" / "circuit.txt").unlink()
    assert V.case(w, scratch)[0]["blobs"][0] == str(w / "m0" / "circuit.txt")
    (w / "m2" / "pub-4.bin").write_bytes(public_of(1))
    with pytest.raises(V.Refused, match=f"isn't, by SHA-512, the one {a5.PIN} pinned for m2"):
        V.case(w, scratch)
    (w / "m2" / "pub-4.bin").write_bytes(public_of(2))
    (w / "m2" / "pub-4.bin").rename(w / "m2" / "kept.bin")
    with pytest.raises(V.Refused, match="the drawn member m2 kept no public file"):
        V.case(w, scratch)
    (w / "m2" / "kept.bin").rename(w / "m2" / "pub-4.bin")
    summary = (w / "summary.json").read_text()
    (w / "summary.json").unlink()
    with pytest.raises(V.Refused, match="m2 kept no compose.log"):
        V.case(w, scratch)
    (w / "summary.json").write_text(summary)
    (w / "session-m1" / "sessions" / "b").mkdir(parents=True)
    (w / "session-m1" / "sessions" / "b" / "session.json").write_bytes(b"{}")
    with pytest.raises(V.Refused, match="sessions of 2 members"):
        V.case(w, scratch)
    for d in ("session-m1", "session-m2"):
        shutil.rmtree(w / d)
    with pytest.raises(V.Refused, match="sessions of 0 members"):
        V.case(w, scratch)


def test_the_layout_is_the_summarys_members_keyed_as_a5_named_them(tmp_path):
    """`layout_of`: the window's members in unit order, each its key as a5 recorded it, or, in a summary from before
    a5 recorded keys, its template's chunk ``<template>/c<j>`` counted from 0 per template; members that don't tile the
    window from unit 0, or no summary, are refused."""
    def summary(members: list[dict]) -> Path:
        (tmp_path / "summary.json").write_text(json.dumps({"members": members}))
        return tmp_path

    old = [{"k": 2, "template": "B", "n": 1, "base": 5}, {"k": 0, "template": "A", "n": 2, "base": 0},
           {"k": 1, "template": "A", "n": 3, "base": 2}]
    assert V.layout_of(summary(old)) == [["A/c0", 2], ["A/c1", 3], ["B/c0", 1]]
    assert V.layout_of(summary([m | {"key": f"K{m['k']}"} for m in old])) == [["K0", 2], ["K1", 3], ["K2", 1]]
    with pytest.raises(V.Refused, match="m2 starts at unit 6, not 5"):
        V.layout_of(summary([old[0] | {"base": 6}, *old[1:]]))
    (tmp_path / "summary.json").unlink()
    with pytest.raises(V.Refused, match="names no members"):
        V.layout_of(tmp_path)


def test_the_onsite_record_reads_a5s_pin_against_the_registration_the_coins_committed_to(tmp_path, RP):
    """The pin a5 writes (`a5.public_pin`) is the one the onsite record reads (`onsite_record.pinned`): its leaf layer
    is the window's over the public bodies (`rows.leaf_layer`), so the registration that holds it (R4) holds the pin,
    and the kept public file is its member's; a registration holding another leaf layer doesn't."""
    import onsite_record as OR

    from verity.ml.circuits.codec import canonical_json
    from verity.ml.windows import rows

    w = kept_window(tmp_path, RP)
    pub, pin = w / "m0" / "pub-4.bin", w / a5.PIN
    layer = rows.leaf_layer({"m0-key": a5.body(pub)})
    assert json.loads(pin.read_text())["leaf_layer"] == layer

    def registration(leaf_layer: str) -> str:
        return canonical_json({"format": "verity/registration/v1", "leaf_layer": leaf_layer}).hex()

    assert OR.pinned(pin, registration(layer), {registration(layer)}, pub) is None
    other = registration("00" * 64)
    assert "leaf layer" in OR.pinned(pin, other, {other}, pub)


def test_a_drawn_member_a5_completed_is_named_by_its_completed_file_and_held_to_the_pin(tmp_path, RP):
    """a5 rewrites its drawn member's public file after the draw under the completed header (`a5.complete_drawn`; a
    member with tables keeps its body) and pins it once the tie holds (`a5.pin_completed`). Before that pin a kept
    completed file is refused; with it the case names the completed file, the onsite record holds it to the pin and to
    the member's staged body (`onsite_record.pinned`), the staged bodies still the registration's leaf layer, and the
    staged file, or a file neither entry pins, is refused."""
    import onsite_record as OR

    from verity.ml.circuits.codec import canonical_json

    w, scratch = kept_window(tmp_path, RP), tmp_path / "scratch"
    pub, pin = w / "m0" / "pub-4.bin", w / a5.PIN
    staged = pub.read_bytes()
    completed = b"completed header\n" + a5.body(pub)
    pub.write_bytes(completed)
    with pytest.raises(V.Refused, match="pinned for m0"):
        V.case(w, scratch)
    a5.pin_completed(w, [({"m": types.SimpleNamespace(key="m0-key", n=4), "st": {"dir": w / "m0"}}, [])])
    assert V.case(w, scratch)[0]["params"]["proofs"]["public"] == hashlib.sha512(completed).hexdigest()
    reg = canonical_json({"format": "verity/registration/v1", "leaf_layer": json.loads(pin.read_text())["leaf_layer"]})
    assert OR.pinned(pin, reg.hex(), {reg.hex()}, pub) is None
    pub.write_bytes(staged)
    with pytest.raises(V.Refused, match="pinned for m0"):
        V.case(w, scratch)
    pub.write_bytes(b"completed header\nanother body")
    with pytest.raises(V.Refused, match="pinned for m0"):
        V.case(w, scratch)
    assert "isn't, by SHA-512, one --public-pin pinned" in OR.pinned(pin, reg.hex(), {reg.hex()}, pub)


def test_a_blob_with_no_file_leaves_only_missing(tmp_path):
    out, kept, gone = tmp_path / "v2", tmp_path / "pub-3.bin", tmp_path / "m0" / "circuit.txt"
    kept.write_bytes(b"public")
    assert V.write(out, _v2(tmp_path, [kept]), "kept") == []
    assert V.write(out, _v2(tmp_path, [gone]), "rebuild failed (exit 1)") == [str(gone)]
    assert [f.name for f in out.iterdir()] == ["missing.json"]
    why = json.loads((out / "missing.json").read_text())
    assert why["protocol"] == V.ID and str(gone) in why["why"] and "rebuild failed" in why["why"]
    assert C.case_dirs([out]) == []
