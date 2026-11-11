"""v2_case.py: sampled proofs v2's case of a served window from its kept files, in a5's form: one member's statement,
its public file named by the kept file's digest, and no case for a window of several. The rebuild reads what a served
run's ``m0/compose.log`` (``fixtures/served-window-m0``, a served H200 run's) records, and the case is written private
as `composite.py` reads it, or, with no case or a blob with no file, only ``missing.json``, which `composite.py`
leaves out."""

from __future__ import annotations

import importlib.util
import json
import shutil
import stat
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import composite as C
import v2_case as V

LOG = Path(__file__).resolve().parent / "fixtures" / "served-window-m0" / "compose.log"
_spec = importlib.util.spec_from_file_location("one_stage_replay", Path(__file__).resolve().parents[1] / "replay.py")
REPLAY = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = REPLAY
_spec.loader.exec_module(REPLAY)


@pytest.fixture
def RP(monkeypatch):
    """This suite's `replay`, which a5's `case_v2` imports by that name, under that name for one test only (as
    `test_a5_case_v2.py`'s)."""
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
    _window(w, RP, members, [1], {"t": "aa" * 64, "u": "bb" * 64})
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
    names = _window(w, RP, [{"k": 0, "template": "t", "n": 4, "base": 0}], [0], {"t": pin} if summary else None)
    if not summary:
        shutil.copy(LOG, w / "m0" / "compose.log")
    v2, how, n = V.case(w, tmp_path / "scratch")
    assert (how, n) == ("kept", 1)
    assert v2["params"] == {"partition": {"rule": "r"}, "proofs": {
        "circuit": pin, "public": V.PF.file_sha512(w / "m0" / "pub-4.bin"), "registered": None}}
    e = json.loads(bytes.fromhex(v2["evidence"]))
    assert list(e["proofs"]) == ["sessions"] and [x["name"] for x in e["proofs"]["sessions"]] == names
    assert v2["blobs"] == [str(w / f) for f in ("m0/circuit.txt", "m0/pub-4.bin", "session-m0/sessions/a/session.json")]


def test_the_one_statement_names_the_members_registered_file_and_the_tables_its_circuit_looks_up(tmp_path, RP):
    """A member reading registered values has the verifier's kept ``registered.json`` as the case's ``registered``, and
    the blobs end with the lookup tables its circuit's META names, no other file of ``tables/``."""
    w, table = tmp_path / "window", next(iter(V.PF.MUFU_SHA512))
    _window(w, RP, [{"k": 0, "template": "t", "n": 4, "base": 0}], [0], {"t": "aa" * 64}, meta={"lookups": {table: 1}})
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
    assert json.loads((out / "case.json").read_text())["blobs"] == [str(blob)]


def test_a_blob_with_no_file_leaves_only_missing(tmp_path):
    out, kept, gone = tmp_path / "v2", tmp_path / "pub-3.bin", tmp_path / "m0" / "circuit.txt"
    kept.write_bytes(b"public")
    assert V.write(out, _v2(tmp_path, [kept]), "kept") == []
    assert V.write(out, _v2(tmp_path, [gone]), "rebuild failed (exit 1)") == [str(gone)]
    assert [f.name for f in out.iterdir()] == ["missing.json"]
    why = json.loads((out / "missing.json").read_text())
    assert why["protocol"] == V.ID and str(gone) in why["why"] and "rebuild failed" in why["why"]
    assert C.case_dirs([out]) == []
