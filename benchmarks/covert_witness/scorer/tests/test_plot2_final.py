"""plot2_final.py's puts: a note read from a file with the stored tables' art id in it, a dry form that stores nothing
and PUTs nothing, and the fit's console body sent as a build wrote it."""
import hashlib
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import plot2_final as P2
from te_core import algorithm_policy as ap
from te_core import console, frontier_fit

ART = "art:" + "1" * 64


def panel(pid: str, note: str) -> dict:
    return {"id": pid, "title": "t", "kind": "table", "updated_at": "2026-01-01T00:00:00Z", "source": "s",
            "note": note, "columns": ["a"], "rows": [[1], [2]]}


@pytest.fixture
def route(tmp_path, monkeypatch):
    """A built overlay directory and a route that records what it is sent: `cut` gives one body whatever the rows,
    with the note it is handed, else the builder's own, naming the art."""
    (tmp_path / "tables").mkdir()
    (tmp_path / "tables" / f"{ap.NAME}.json").write_text(json.dumps(panel(ap.PANEL, "")))
    (tmp_path / "tables" / "final.json").write_text(json.dumps({"cells": []}))
    seen = {"stored": [], "put": [], "served": {}}
    monkeypatch.setattr(P2, "cut", lambda p, cells, art=None, text=None: panel(
        ap.PANEL, f"the builder's note. Tables {art}." if text is None else text))
    monkeypatch.setattr(ap, "store", lambda path, meta: seen["stored"].append(path) or ART)
    monkeypatch.setattr(console, "served", lambda *a: seen["served"])
    monkeypatch.setattr(console, "refused", lambda q: [])
    monkeypatch.setattr(console, "refusal", lambda q: None)

    def put(q):
        seen["put"].append(q)
        seen["served"] = {q["id"]: q}
        return "200"
    monkeypatch.setattr(console, "put", put)
    return tmp_path, seen


def test_a_put_takes_its_note_from_a_file_and_names_the_stored_tables_in_it(route):
    d, seen = route
    note = d / "note.txt"
    note.write_text("Corrected: what moved.\nTables {art}.\n")
    assert P2.main(["put", str(d), "--note", str(note)]) == 0
    assert seen["stored"] == [d / "tables"]
    assert [q["note"] for q in seen["put"]] == [f"Corrected: what moved. Tables {ART[:16]}."]
    assert json.loads((d / "console" / f"{ap.NAME}.json").read_text()) == seen["put"][0]


def test_a_note_that_cites_no_tables_or_is_over_the_routes_limit_is_refused_before_anything_is_stored(route):
    d, seen = route
    note = d / "note.txt"
    note.write_text("Corrected: what moved.")
    with pytest.raises(SystemExit, match="cites no stored tables"):
        P2.main(["put", str(d), "--note", str(note)])
    note.write_text("x" * (console.NOTE_MAX - len(P2.cited(ART)) + 1) + "{art}")
    with pytest.raises(SystemExit, match=f"note {console.NOTE_MAX + 1} characters of {console.NOTE_MAX}"):
        P2.main(["put", str(d), "--note", str(note)])
    assert seen["stored"] == [] and seen["put"] == []


def test_a_dry_put_stores_nothing_puts_nothing_and_says_what_it_would_send(route, monkeypatch, capsys):
    d, seen = route
    note = d / "note.txt"
    note.write_text("Corrected: what moved. Tables {art}.")
    monkeypatch.setattr(console, "refusal", lambda q: "9 files of te_core differ here")
    assert P2.main(["put", str(d), "--note", str(note), "--dry"]) == 0
    assert seen["stored"] == [] and seen["put"] == [] and not (d / "console" / f"{ap.NAME}.json").exists()
    body = console.body(panel(ap.PANEL, f"Corrected: what moved. Tables {P2.cited(P2.NO_ART)}."))
    assert len(P2.NO_ART) == len(ART) and P2.cited(ART) == "art:" + "1" * 12
    assert capsys.readouterr().out.strip() == (
        f"dry: {ap.PANEL}: sha256 {hashlib.sha256(body).hexdigest()}, {len(body)} bytes, 2 rows, note "
        f"{len('Corrected: what moved. Tables .') + 16} characters; console.put would refuse: 9 files of te_core "
        f"differ here; nothing stored, nothing PUT")


def test_the_fits_body_is_put_as_a_build_wrote_it_and_read_back(route, capsys):
    d, seen = route
    q = panel(frontier_fit.PANEL, "the fit's note")
    body = d / "frontier-fit.json"
    body.write_bytes(console.body(q))
    assert P2.main(["put-fit", str(body), "--dry"]) == 0
    assert seen["put"] == []
    assert f"sha256 {hashlib.sha256(body.read_bytes()).hexdigest()}, {len(body.read_bytes())} bytes, 2 rows" in \
        capsys.readouterr().out
    assert P2.main(["put-fit", str(body)]) == 0
    assert seen["put"] == [q] and seen["stored"] == []
    assert f"GET {frontier_fit.PANEL}: the body, 2 rows" in capsys.readouterr().out
    (d / "other.json").write_bytes(console.body(panel(ap.PANEL, "n")))
    with pytest.raises(SystemExit, match=f"not {frontier_fit.PANEL}'s"):
        P2.main(["put-fit", str(d / "other.json")])


def test_the_fits_put_takes_its_note_from_a_file_and_refuses_one_over_the_routes_limit(route, capsys):
    d, seen = route
    q = panel(frontier_fit.PANEL, "the build's note")
    body = d / "frontier-fit.json"
    body.write_bytes(console.body(q))
    note = d / "note.txt"
    note.write_text("The fit's note,\nfrom a file.\n")
    sent = q | {"note": "The fit's note, from a file."}
    assert P2.main(["put-fit", str(body), "--note", str(note), "--dry"]) == 0 and seen["put"] == []
    raw = console.body(sent)
    assert f"sha256 {hashlib.sha256(raw).hexdigest()}, {len(raw)} bytes, 2 rows, note 28 characters" in \
        capsys.readouterr().out
    assert P2.main(["put-fit", str(body), "--note", str(note)]) == 0 and seen["put"] == [sent]
    note.write_text("x" * (console.NOTE_MAX + 1))
    with pytest.raises(SystemExit, match=f"note {console.NOTE_MAX + 1} characters of {console.NOTE_MAX}"):
        P2.main(["put-fit", str(body), "--note", str(note)])
    assert seen["put"] == [sent]


def test_the_fits_read_back_takes_the_routes_integer_for_a_whole_float_as_that_number(route, monkeypatch, capsys):
    """The route writes a float with no fractional part as an integer (6.52837e+20 comes back 652837000000000000000),
    which Python's != holds unequal to the float; the read-back compares as numbers, and still catches a changed
    cell."""
    d, seen = route
    q = panel(frontier_fit.PANEL, "the fit's note") | {"rows": [[6.52837e+20], [2.5]]}
    body = d / "frontier-fit.json"
    body.write_bytes(console.body(q))
    served = {frontier_fit.PANEL: q | {"rows": [[652837000000000000000], [2.5]]}}
    monkeypatch.setattr(console, "served", lambda *a: served)
    monkeypatch.setattr(console, "put", lambda q: "200")
    assert served[frontier_fit.PANEL] != q
    assert P2.main(["put-fit", str(body)]) == 0
    assert f"GET {frontier_fit.PANEL}: the body, 2 rows" in capsys.readouterr().out
    served[frontier_fit.PANEL] = q | {"rows": [[652838000000000000000], [2.5]]}
    assert P2.main(["put-fit", str(body)]) == 1
    assert "the route serves another panel than the body (rows differ)" in capsys.readouterr().out
