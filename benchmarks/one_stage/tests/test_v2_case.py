"""v2_case.py: sampled proofs v2's case of a served window from its kept files, in a5's form: one member's statement,
and no case for a window of several. The rebuild reads what a served run's ``m0/compose.log``
(``fixtures/served-window-m0``, a served H200 run's) records, and the case is written private as `composite.py` reads
it, or, with no case or a blob with no file, only ``missing.json``, which `composite.py` leaves out. `case` runs on a
small kept window, so it is held to a5's `case_v2` as that is: its public digest is the one the audit staged where the
window keeps it, else the kept file's own, which it says, with whether the audit staged the file before the sample. A
kept public file the audit did not stage leaves only ``missing.json`` too."""

from __future__ import annotations

import hashlib
import json
import shutil
import stat
import sys
from pathlib import Path

import pytest
from test_a5_case_v2 import RP  # noqa: F401 -- the fixture: this suite's `replay` under the name a5's `case_v2` imports

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import composite as C
import v2_case as V

LOG = Path(__file__).resolve().parent / "fixtures" / "served-window-m0" / "compose.log"


def test_the_compose_log_names_the_rebuild():
    c = V.compose_command(LOG.read_text())
    assert c["subcircuit"] == "silu-mul-v2/i8192/bf16" and c["set"] == "silumul-v2-i8192"
    assert c["pin"].startswith("13d14493cb78f7ad") and len(c["pin"]) == 128
    assert c["partition"].startswith("e94be647") and c["program"].startswith("1831c097")


def test_a_template_composed_from_no_input_set_is_not_rebuilt(tmp_path):
    c = V.compose_command('compose_circuit.py --template t --out x\n{"pin": "' + "ab" * 64 + '"}\n')
    assert (c["set"], c["subcircuit"], c["pin"]) == (None, None, "ab" * 64)
    assert V.rebuild(c, tmp_path) == (None, "not rebuilt: its compose.log names no input set")


def _members_window(w: Path, RP, members: list[dict], drawn: list[int], pins: dict | None,
                    meta: dict | None = None) -> list[str]:
    """A kept window of ``members`` (a5's ``summary.json``'s, with ``pins`` its ``circuits``; ``pins`` None: no
    summary), each template's circuit file in its first member's directory with ``meta`` its META, a session per
    ``drawn`` member; its sessions' names."""
    for m in members:
        d = w / f"m{m['k']}"
        d.mkdir(parents=True)
        (d / f"pub-{m['n']}.bin").write_bytes(d.name.encode())
    for t in dict.fromkeys(m["template"] for m in members):
        (w / f"m{next(m['k'] for m in members if m['template'] == t)}" / "circuit.txt").write_text(
            f"FLOCK\nMETA {json.dumps(meta or {})}\n{t}'s gates")
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
        (w / "summary.json").write_text(json.dumps({"members": members,
                                                    "circuits": {t: {"pin": p} for t, p in pins.items()}}))
    return names


def test_a_window_of_several_members_has_no_v2_case_and_leaves_only_missing(tmp_path, RP):
    """v2 reads one member's statement and refuses a windowed registration, so a window whose summary lists several
    members gets no case, whichever of them were sampled: only ``missing.json`` saying so, which `composite.py` leaves
    out."""
    w, out = tmp_path / "window", tmp_path / "v2"
    members = [{"k": 0, "template": "t", "n": 2, "base": 0}, {"k": 1, "template": "t", "n": 3, "base": 2},
               {"k": 2, "template": "u", "n": 1, "base": 5}]
    _members_window(w, RP, members, [1], {"t": "aa" * 64, "u": "bb" * 64})
    v2, why, n = V.case(w, tmp_path / "scratch")
    assert (v2, n) == (None, 0) and "this window has 3 members" in why
    assert V.write(out, v2, why) == []
    assert [f.name for f in out.iterdir()] == ["missing.json"]
    assert json.loads((out / "missing.json").read_text()) == {"protocol": V.ID, "why": why}
    assert C.case_dirs([out]) == []
    assert V.main(["--window", str(w), "--out", str(out), "--scratch", str(tmp_path / "scratch")]) == 2


@pytest.mark.parametrize("summary", [True, False])
def test_a_window_of_one_member_is_the_one_statement_case(tmp_path, RP, summary):
    """A window of one member, listed by its summary or, with none, ``m0`` alone with its pin its ``compose.log``'s, gets
    v2's one statement: ``proofs`` exactly the pin, the kept public file's digest and the registered file, which v2's
    Lean reads, and the sessions under the evidence's names."""
    w, pin = tmp_path / "window", V.compose_command(LOG.read_text())["pin"]
    names = _members_window(w, RP, [{"k": 0, "template": "t", "n": 4, "base": 0}], [0], {"t": pin} if summary else None)
    if not summary:
        shutil.copy(LOG, w / "m0" / "compose.log")
    v2, how, n = V.case(w, tmp_path / "scratch")
    assert n == 1 and how == ("circuit file kept; public digest the kept file's own, taken after the audit: the window "
                              "keeps none from staging")
    assert v2["params"] == {"partition": {"rule": "r"}, "proofs": {
        "circuit": pin, "public": V.PF.file_sha512(w / "m0" / "pub-4.bin"), "registered": None}}
    e = json.loads(bytes.fromhex(v2["evidence"]))
    assert list(e["proofs"]) == ["sessions"] and [x["name"] for x in e["proofs"]["sessions"]] == names
    assert v2["blobs"] == [str(w / f) for f in ("m0/circuit.txt", "m0/pub-4.bin", "session-m0/sessions/a/session.json")]


def test_the_one_statement_names_the_members_registered_file_and_the_tables_its_circuit_looks_up(tmp_path, RP):
    """A member reading registered values has the verifier's kept ``registered.json`` as the case's ``registered``, and
    the blobs end with the lookup tables its circuit's META names, no other file of ``tables/``."""
    w, table = tmp_path / "window", next(iter(V.PF.MUFU_SHA512))
    _members_window(w, RP, [{"k": 0, "template": "t", "n": 4, "base": 0}], [0], {"t": "aa" * 64}, meta={"lookups": {table: 1}})
    (w / "m0" / "registered.json").write_text('{"ports": {}}')
    (w / "m0" / "tables").mkdir()
    for name in (table, "other"):
        (w / "m0" / "tables" / f"{name}.u32").write_bytes(b"\0" * 4)
    v2, _, _ = V.case(w, tmp_path / "scratch")
    assert v2["params"]["proofs"]["registered"] == {"ports": {}}
    assert v2["blobs"][-1] == str(w / "m0" / "tables" / f"{table}.u32") and len(v2["blobs"]) == 4


def _v2(tmp_path: Path, blobs: list[Path]) -> dict:
    return {"blobs": [str(b) for b in blobs], "commitment": b"registration".hex(), "evidence": b'{"proofs":{}}'.hex(),
            "params": {"proofs": {"circuit": "00" * 64, "public": "11" * 64, "registered": None}}}


def test_a_case_with_its_blobs_is_written_private(tmp_path):
    blob = tmp_path / "circuit.txt"
    blob.write_text("gates")
    out = tmp_path / "v2"
    assert V.write(out, _v2(tmp_path, [blob]), "kept") == []
    assert stat.S_IMODE(out.stat().st_mode) == 0o700
    assert {f.name: stat.S_IMODE(f.stat().st_mode) for f in out.iterdir()} == dict.fromkeys(V.CASE_FILES[:4], 0o600)
    assert (out / "commitment.bin").read_bytes() == b"registration"
    assert json.loads((out / "case.json").read_text()) == {**_v2(tmp_path, [blob]), "files": "kept"}


def test_a_blob_with_no_file_leaves_only_missing(tmp_path):
    out, kept, gone = tmp_path / "v2", tmp_path / "pub-3.bin", tmp_path / "m0" / "circuit.txt"
    kept.write_bytes(b"public")
    assert V.write(out, _v2(tmp_path, [kept]), "kept") == []
    assert V.write(out, _v2(tmp_path, [gone]), "rebuild failed (exit 1)") == [str(gone)]
    assert [f.name for f in out.iterdir()] == ["missing.json"]
    why = json.loads((out / "missing.json").read_text())
    assert why["protocol"] == V.ID and str(gone) in why["why"] and "rebuild failed" in why["why"]
    assert C.case_dirs([out]) == []


PUBLIC = hashlib.sha512(b"public").hexdigest()


def _window(tmp_path: Path) -> Path:
    """A kept window as `case` reads it: the served run's ``compose.log`` beside a circuit file and a public file, the
    verifier's partition, one session, and the v1 case whose evidence names it."""
    w = tmp_path / "window"
    s = w / "session-m0" / "sessions" / "a"
    s.mkdir(parents=True)
    (s / "session.json").write_text("{}")
    (s / "rep-0.bin").write_bytes(b"rep")
    (w / "m0").mkdir()
    (w / "m0" / "compose.log").write_text(LOG.read_text())
    (w / "m0" / "circuit.txt").write_text("gates")
    (w / "m0" / "pub-4.bin").write_bytes(b"public")
    (w / "partition.json").write_text('{"rule": "r"}')
    evidence = {"sessions": ["s0"], "verdicts": {"m0": {"accepted": True}}}
    (w / "case.json").write_text(json.dumps({"commitment": "ab", "params": {"delta": "1/100"},
                                             "evidence": json.dumps(evidence).encode().hex()}))
    return w


def test_a_window_keeping_no_staged_digest_gets_the_kept_public_files_own_which_the_case_says(tmp_path, RP):
    """No ``case-v2.json``, or one of an audit that named no public file in its params: the digest is taken after the
    audit, so the case shows only that the kept files verify, and `how` says which digest it is."""
    w = _window(tmp_path)
    v2, how, n = V.case(w, tmp_path / "scratch")
    pin = V.compose_command(LOG.read_text())["pin"]
    assert v2["params"] == {"delta": "1/100", "partition": {"rule": "r"},
                            "proofs": {"circuit": pin, "public": PUBLIC, "registered": None}}
    assert n == 1 and how == ("circuit file kept; public digest the kept file's own, taken after the audit: the window "
                              "keeps none from staging")
    s = w / "session-m0" / "sessions" / "a"
    assert v2["blobs"] == [str(f) for f in (w / "m0" / "circuit.txt", w / "m0" / "pub-4.bin", s / "session.json",
                                            s / "rep-0.bin")]
    e = json.loads(bytes.fromhex(v2["evidence"]))
    assert "verdicts" not in e and [x["name"] for x in e["proofs"]["sessions"]] == e["sessions"] == ["s0"]
    (w / "case-v2.json").write_text(json.dumps({"params": {"proofs": {"circuit": pin, "registered": None}}}))
    assert V.case(w, tmp_path / "scratch") == (v2, how, n)


STAGED = {"params": {"proofs": {"circuit": "cc" * 64, "public": PUBLIC, "registered": None}}}


def test_the_case_names_the_public_digest_the_audit_staged_and_says_whether_before_the_sample(tmp_path, RP, capsys):
    """The window's ``summary.json`` says whether the audit's caller handed it the sample (``handed``), which a5 stages
    the public file after. A window keeping no summary cannot tell. The written ``case.json`` carries the label under
    ``files``, beside the case and no part of it."""
    w, out, scratch = _window(tmp_path), tmp_path / "v2", tmp_path / "scratch"
    (w / "case-v2.json").write_text(json.dumps(STAGED))
    label = "circuit file kept; public digest case-v2.json's, taken when the audit staged the file"
    v2, how, _ = V.case(w, scratch)
    assert v2["params"]["proofs"]["public"] == PUBLIC
    assert how == label + "; cannot tell whether before the sample: the window keeps no summary.json"
    (w / "summary.json").write_text(json.dumps({"handed": {"sessions": ["s0"]}}))
    assert V.case(w, scratch) == (v2, label + ", after the sample its caller handed it", 1)
    (w / "summary.json").write_text(json.dumps({"audit": {"accepted": True}}))
    how = label + ", before the sample"
    assert V.case(w, scratch) == (v2, how, 1)
    assert V.main(["--window", str(w), "--out", str(out), "--scratch", str(scratch)]) == 0
    assert json.loads((out / "params.json").read_text())["proofs"]["public"] == PUBLIC
    assert json.loads(capsys.readouterr().out.removeprefix("V2CASE "))["files"] == how
    assert json.loads((out / "case.json").read_text()) == {**v2, "files": how}
    (out / "answer.json").write_text(json.dumps({"protocol": V.ID, "verdict": "accept"}))
    piece = C.piece(out)
    assert "files" not in piece and (piece["params"], piece["blobs"]) == (v2["params"], v2["blobs"])


def test_a_kept_public_file_the_audit_did_not_stage_is_refused_and_leaves_only_missing(tmp_path, RP, capsys):
    """The refusal ends as the script's others do: a case already in ``--out`` is cleared, ``missing.json`` and the
    ``V2CASE`` line carry the refusal, which names both files, and the exit is 2, so `composite.py` leaves the protocol
    out."""
    w, out = _window(tmp_path), tmp_path / "v2"
    kept, argv = w / "case-v2.json", ["--window", str(w), "--out", str(out), "--scratch", str(tmp_path / "scratch")]
    kept.write_text(json.dumps(STAGED))
    assert V.main(argv) == 0 and (out / "case.json").is_file()
    capsys.readouterr()
    (w / "m0" / "pub-4.bin").write_bytes(b"public, changed since the audit staged it")
    with pytest.raises(V.Refused, match="not the public file the audit staged") as x:
        V.case(w, tmp_path / "scratch")
    why = str(x.value)
    assert str(w / "m0" / "pub-4.bin") in why and str(kept) in why and PUBLIC[:16] in why
    assert V.main(argv) == 2
    assert json.loads(capsys.readouterr().out.removeprefix("V2CASE ")) == {"refused": why}
    assert [f.name for f in out.iterdir()] == ["missing.json"]
    assert json.loads((out / "missing.json").read_text()) == {"protocol": V.ID, "why": why}
    assert C.case_dirs([out]) == []
