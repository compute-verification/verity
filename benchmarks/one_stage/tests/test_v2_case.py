"""v2_case.py: sampled proofs v2's case of a served window from its kept files, in a5's form: one statement for a
window of one member, per member for several. The rebuild reads what a served run's ``m0/compose.log``
(``fixtures/served-window-m0``, a served H200 run's) records, the case names the auditor's challenger by its key record
and carries the window's kept release of the draw's coins, a release naming another key is refused, each statement's
public file is the one a5 pinned for its member, a public file changed after the draw and a window with no pin refused,
and the case is written private as `composite.py` reads it, or, when a blob has no file or the case is refused, only
``missing.json``, which `composite.py` leaves out."""

from __future__ import annotations

import hashlib
import importlib.util
import json
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
    """This suite's `replay`, which a5's `case_v2` imports by that name, under that name for one test only (as
    `test_a5_case_v2.py`'s; `network_traces` has a `replay` of its own)."""
    monkeypatch.setitem(sys.modules, "replay", REPLAY)
    return REPLAY


def test_the_compose_log_names_the_rebuild():
    c = V.compose_command(LOG.read_text())
    assert c["subcircuit"] == "silu-mul-v2/i8192/bf16" and c["set"] == "silumul-v2-i8192"
    assert c["pin"].startswith("13d14493cb78f7ad") and len(c["pin"]) == 128
    assert c["partition"].startswith("e94be647") and c["program"].startswith("1831c097")


def test_a_template_composed_from_no_input_set_is_not_rebuilt(tmp_path):
    c = V.compose_command('compose_circuit.py --template t --out x\n{"pin": "' + "ab" * 64 + '"}\n')
    assert (c["set"], c["subcircuit"], c["pin"]) == (None, None, "ab" * 64)
    assert V.rebuild(c, tmp_path) == (None, "not rebuilt: its compose.log names no input set")


def _window(w: Path, RP, members: list[dict], drawn: list[int], pins: dict | None,
            per_block: dict | None = None) -> list[str]:
    """A kept window of ``members`` (a5's ``summary.json``'s, with ``pins`` its ``circuits`` and ``per_block`` their
    units a proof block, where it names one; ``pins`` None: no summary), each template's circuit file in its first
    member's directory, its META at two units a proof block, each member's public file as a5 pinned it (`a5.PIN`), a
    session per ``drawn`` member; its sessions' names."""
    for m in members:
        d = w / f"m{m['k']}"
        d.mkdir(parents=True)
        (d / f"pub-{m['n']}.bin").write_bytes(d.name.encode())
    (w / a5.PIN).write_text(json.dumps({"members": {f"m{m['k']}-key": {
        "k": m["k"], "n": m["n"], "file": hashlib.sha512(f"m{m['k']}".encode()).hexdigest()} for m in members}}))
    for t in dict.fromkeys(m["template"] for m in members):
        (w / f"m{next(m['k'] for m in members if m['template'] == t)}" / "circuit.txt").write_text(
            f'FLOCK\nMETA {{"vus_per_block": 2}}\n{t}\'s gates')
    names = [f"s{k}" for k in drawn]
    for k in drawn:
        s = w / f"session-m{k}" / "sessions" / "a"
        s.mkdir(parents=True)
        (s / "session.json").write_text(json.dumps({"k": k}))
    evidence = {"sessions": names, "verdicts": {}}
    (w / "case.json").write_text(json.dumps({"commitment": "ab", "evidence": RP.DT.canonical(evidence).encode().hex(),
                                             "params": {}}))
    (w / "partition.json").write_text('{"rule": "r"}')
    if pins is not None:
        (w / "summary.json").write_text(json.dumps({"members": members, "circuits": {
            t: {"pin": p, **({"vus_per_block": per_block[t]} if t in (per_block or {}) else {})} for t, p in pins.items()}}))
    return names


def test_a_window_of_several_members_is_the_per_member_case(tmp_path, RP):
    """A window whose summary lists several members gets a5's per-member case (`a5.v2_proof`): every member, the
    undrawn too, with its units, its template's pin, its template's units a proof block (the summary's, else its
    circuit file's META) and its kept registered file; a statement a drawn member on its template's circuit file, kept
    in the template's first member's directory, under the evidence's sessions in order."""
    w = tmp_path / "window"
    members = [{"k": 0, "template": "t", "n": 2, "base": 0}, {"k": 1, "template": "t", "n": 3, "base": 2},
               {"k": 2, "template": "u", "n": 1, "base": 5}]
    pins = {"t": "aa" * 64, "u": "bb" * 64}
    names = _window(w, RP, members, [1, 2], pins, per_block={"t": 4})
    (w / "m2" / "registered.json").write_text('{"ports": {}}')
    v2, how, n, _ = V.case(w, tmp_path / "scratch")
    assert (how, n) == ("m0: kept; m2: kept", 2)
    assert v2["params"] == {"partition": {"rule": "r"}, "proofs": {"members": {
        "m0": {"circuit": pins["t"], "registered": None, "units": [[0, 2]], "vus_per_block": 4},
        "m1": {"circuit": pins["t"], "registered": None, "units": [[2, 5]], "vus_per_block": 4},
        "m2": {"circuit": pins["u"], "registered": {"ports": {}}, "units": [[5, 6]], "vus_per_block": 2}}}}
    e = json.loads(bytes.fromhex(v2["evidence"]))
    assert [(st["member"], [x["name"] for x in st["sessions"]]) for st in e["proofs"]["statements"]] == [
        ("m1", names[:1]), ("m2", names[1:])]
    assert v2["blobs"] == [str(w / f) for f in ("m0/circuit.txt", "m1/pub-3.bin", "session-m1/sessions/a/session.json",
                                                "m2/circuit.txt", "m2/pub-1.bin", "session-m2/sessions/a/session.json")]


@pytest.mark.parametrize("summary", [True, False])
def test_a_window_of_one_member_is_the_one_statement_case(tmp_path, RP, summary):
    """A window of one member, listed by its summary or, with none, ``m0`` alone with its pin its ``compose.log``'s, gets
    the one-statement form."""
    w, pin = tmp_path / "window", V.compose_command(LOG.read_text())["pin"]
    names = _window(w, RP, [{"k": 0, "template": "t", "n": 4, "base": 0}], [0], {"t": pin} if summary else None)
    if not summary:
        shutil.copy(LOG, w / "m0" / "compose.log")
    v2, how, n, _ = V.case(w, tmp_path / "scratch")
    assert (how, n) == ("kept", 1)
    assert v2["params"]["proofs"] == {"circuit": pin, "registered": None}
    e = json.loads(bytes.fromhex(v2["evidence"]))
    assert e["proofs"]["public"] == V.PF.file_sha512(w / "m0" / "pub-4.bin")
    assert [x["name"] for x in e["proofs"]["sessions"]] == names


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
    composed once at m0 (``compose.log``, `a5.audit`), its META at two units a proof block, each member's public file
    and a5's pin of them at staging (`a5.public_pin`), the members in ``summary.json``, and m<k>'s one session."""
    w = root / "window"
    (w / "m0").mkdir(parents=True)
    (w / "m0" / "compose.log").write_text(LOG.read_text())
    (w / "m0" / "circuit.txt").write_text('FLOCK\nMETA {"vus_per_block": 2}\ngates')
    staged, publics = [], {}
    for i in range(k + 1):
        pub = w / f"m{i}" / "pub-4.bin"
        pub.parent.mkdir(exist_ok=True)
        pub.write_bytes(public_of(i))
        staged.append({"k": i, "m": types.SimpleNamespace(key=f"m{i}-key", n=4, tables=(("t", 4),)),
                       "st": {"public": hashlib.sha512(public_of(i)).hexdigest()}})
        publics[f"m{i}-key"] = a5.body(pub)
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
    """A statement's public file is the one a5 pinned at staging, before the draw (`a5.PIN`), never a kept file read
    after it, which the prover could choose: a public file changed after the draw, or a window that kept no pin, is
    refused, and the case directory keeps only ``missing.json``."""
    w, out = kept_window(tmp_path, RP), tmp_path / "v2"
    argv = ["--window", str(w), "--out", str(out), "--scratch", str(tmp_path / "scratch")]
    assert V.main(argv) == 0
    assert json.loads((out / "evidence.json").read_text())["proofs"]["public"] == hashlib.sha512(PUBLIC).hexdigest()
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


def test_a_drawn_members_statement_reads_its_own_files_held_to_its_own_pin(tmp_path, RP):
    """A kept window of three members of one template whose one drawn member is m2: the case is the per-member one,
    every member with its template's pin (its first member's ``compose.log``'s, where a5 composed the template's
    circuit once) and its units a proof block, and one statement, m2's, on m2's session and public file, held to m2's
    pin, and the template's circuit file. A public file of m2 that is another member's, or none kept, is refused."""
    w, scratch = kept_window(tmp_path, RP, k=2), tmp_path / "scratch"
    got, how, n, _ = V.case(w, scratch)
    pin = V.compose_command(LOG.read_text())["pin"]
    assert (how, n) == ("m0: kept", 1)
    assert got["params"]["proofs"] == {"members": {f"m{i}": {"circuit": pin, "registered": None,
                                                              "units": [[4 * i, 4 * i + 4]], "vus_per_block": 2}
                                                   for i in range(3)}}
    [st] = json.loads(bytes.fromhex(got["evidence"]))["proofs"]["statements"]
    assert (st["member"], st["public"]) == ("m2", hashlib.sha512(public_of(2)).hexdigest())
    assert got["blobs"][:2] == [str(w / "m0" / "circuit.txt"), str(w / "m2" / "pub-4.bin")]
    (w / "m2" / "pub-4.bin").write_bytes(public_of(1))
    with pytest.raises(V.Refused, match=f"isn't, by SHA-512, the one {a5.PIN} pinned for m2"):
        V.case(w, scratch)
    (w / "m2" / "pub-4.bin").unlink()
    with pytest.raises(V.Refused, match="the drawn member m2 kept no public file"):
        V.case(w, scratch)


def test_a_drawn_member_a5_completed_is_named_by_its_completed_file_and_held_to_the_pin(tmp_path, RP):
    """a5 rewrites its drawn member's public file after the draw under the completed header (`a5.complete_drawn`; a
    member with tables keeps its body) and pins it once the tie holds (`a5.pin_completed`). Before that pin a kept
    completed file is refused; with it the statement's public file is the completed one, and the staged file, or a file
    neither entry pins, is refused."""
    w, scratch = kept_window(tmp_path, RP), tmp_path / "scratch"
    pub = w / "m0" / "pub-4.bin"
    staged = pub.read_bytes()
    completed = b"completed header\n" + a5.body(pub)
    pub.write_bytes(completed)
    with pytest.raises(V.Refused, match="pinned for m0"):
        V.case(w, scratch)
    a5.pin_completed(w, [({"m": types.SimpleNamespace(key="m0-key", n=4), "st": {"dir": w / "m0"}}, [])])
    evidence = json.loads(bytes.fromhex(V.case(w, scratch)[0]["evidence"]))
    assert evidence["proofs"]["public"] == hashlib.sha512(completed).hexdigest()
    pub.write_bytes(staged)
    with pytest.raises(V.Refused, match="pinned for m0"):
        V.case(w, scratch)
    pub.write_bytes(b"completed header\nanother body")
    with pytest.raises(V.Refused, match="pinned for m0"):
        V.case(w, scratch)


def test_a_blob_with_no_file_leaves_only_missing(tmp_path):
    out, kept, gone = tmp_path / "v2", tmp_path / "pub-3.bin", tmp_path / "m0" / "circuit.txt"
    kept.write_bytes(b"public")
    assert V.write(out, _v2(tmp_path, [kept]), "kept") == []
    assert V.write(out, _v2(tmp_path, [gone]), "rebuild failed (exit 1)") == [str(gone)]
    assert [f.name for f in out.iterdir()] == ["missing.json"]
    why = json.loads((out / "missing.json").read_text())
    assert why["protocol"] == V.ID and str(gone) in why["why"] and "rebuild failed" in why["why"]
    assert C.case_dirs([out]) == []
