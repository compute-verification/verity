"""v2_case.py: sampled proofs v2's case of a served window from its kept files, in a5's form: one statement for a
window of one member, per member for several. The rebuild reads what a served run's ``m0/compose.log``
(``fixtures/served-window-m0``, a served H200 run's) records, and the case is written private as `composite.py` reads
it, or, when a blob has no file, only ``missing.json``, which `composite.py` leaves out."""

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
            per_block: dict | None = None) -> list[str]:
    """A kept window of ``members`` (a5's ``summary.json``'s, with ``pins`` its ``circuits`` and ``per_block`` their
    units a proof block, where it names one; ``pins`` None: no summary), each template's circuit file in its first
    member's directory, its META at two units a proof block, a session per ``drawn`` member; its sessions' names."""
    for m in members:
        d = w / f"m{m['k']}"
        d.mkdir(parents=True)
        (d / f"pub-{m['n']}.bin").write_bytes(d.name.encode())
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
    unsampled too, with its units, its template's pin, its template's units a proof block (the summary's, else its
    circuit file's META) and its kept registered file; a statement a sampled member on its template's circuit file, kept
    in the template's first member's directory, under the evidence's sessions in order."""
    w = tmp_path / "window"
    members = [{"k": 0, "template": "t", "n": 2, "base": 0}, {"k": 1, "template": "t", "n": 3, "base": 2},
               {"k": 2, "template": "u", "n": 1, "base": 5}]
    pins = {"t": "aa" * 64, "u": "bb" * 64}
    names = _window(w, RP, members, [1, 2], pins, per_block={"t": 4})
    (w / "m2" / "registered.json").write_text('{"ports": {}}')
    v2, how, n = V.case(w, tmp_path / "scratch")
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
    v2, how, n = V.case(w, tmp_path / "scratch")
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


def test_a_blob_with_no_file_leaves_only_missing(tmp_path):
    out, kept, gone = tmp_path / "v2", tmp_path / "pub-3.bin", tmp_path / "m0" / "circuit.txt"
    kept.write_bytes(b"public")
    assert V.write(out, _v2(tmp_path, [kept]), "kept") == []
    assert V.write(out, _v2(tmp_path, [gone]), "rebuild failed (exit 1)") == [str(gone)]
    assert [f.name for f in out.iterdir()] == ["missing.json"]
    why = json.loads((out / "missing.json").read_text())
    assert why["protocol"] == V.ID and str(gone) in why["why"] and "rebuild failed" in why["why"]
    assert C.case_dirs([out]) == []
