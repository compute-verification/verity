"""`released_keys`: every JSON file a chain record releases holds only keys its declared list names, each with its
kind; a file is read strictly, by key and never by bytes; and a JSON file whose list isn't written yet is counted as
not yet checked and one under no list fails. The leak guard's pass by key (``--keys``) is tested with the guard
(`benchmarks/network_traces/tests/test_leak_guard.py`)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from verity.protocols.resource_accounting.network.prover import released_keys as RK

ROOT = Path(__file__).resolve().parents[6]

LIST = {"name": "a step's summary", "writer": "step.py", "files": ["step/*.json"], "form": "json",
        "keys": {"schema": "structure", "verdict/accepted": "output", "rounds/*/root": "commitment",
                 "rounds/*/sample/*": "challenge", "units/*/root": "commitment", "units/total": "structure"},
        "maps": {"units/*": "u[0-9]+"}}
GOOD = {"schema": "s/v1", "verdict": {"accepted": True}, "rounds": [{"root": "00", "sample": [1, 2]}, {"root": "01"}],
        "units": {"total": 2, "u0": {"root": "02"}, "u17": {"root": "03"}}}


def _declared(*lists: dict, not_yet: tuple[str, ...] = ()) -> dict:
    return {"schema": RK.SCHEMA, "lists": list(lists), "not_yet": list(not_yet)}


def _undeclared(doc, listing: dict = LIST) -> dict:
    return RK.Listing(listing, "list 0").undeclared(json.dumps(doc).encode())


def test_the_declared_lists_load_and_each_names_its_writer_in_the_tree():
    """released_keys.json loads, so each of its keys has one of the four kinds; each list's writer is a file of the
    tree; and no path waits under ``not_yet`` that a list already rules."""
    declared = RK.load()
    assert declared.lists
    for x in declared.lists:
        assert (ROOT / x.writer.split(" ")[0]).is_file(), f"the list {x.name!r} names a writer the tree doesn't have"
    waiting = set(json.loads(RK.DECLARED.read_text())["not_yet"])
    assert not waiting & {g for x in declared.lists for g in x.globs}


def test_a_file_is_read_by_key_and_strictly():
    """A document of declared keys alone passes, whatever its values and however long; a key the list doesn't name,
    under an object, a list's item or a map's id, is undeclared by its path (an index and an id as ``*``), counted;
    so are a map's key that is no id of its form, an object in place of a list, a list in place of a map or of a
    value, a value in place of deeper keys, and a document that doesn't parse. An empty list or object and null hold
    no value and pass."""
    assert _undeclared(GOOD) == {}
    assert _undeclared({**GOOD, "schema": "x" * 4096, "rounds": [{"root": "0" * 128}] * 50}) == {}
    assert _undeclared({"schema": None, "verdict": {}, "rounds": [], "units": {"u0": None}}) == {}
    assert _undeclared({**GOOD, "note": 1}) == {"note": 1}
    assert _undeclared({**GOOD, "verdict": {"accepted": True, "problems": 0}}) == {"verdict/problems": 1}
    assert _undeclared({**GOOD, "rounds": [{"salt": "a"}, {"root": "0", "salt": "b"}]}) == {"rounds/*/salt": 2}
    assert _undeclared({**GOOD, "units": {"u0": {"root": "0", "rows": 3}}}) == {"units/*/rows": 1}
    assert _undeclared({**GOOD, "units": {"u0": {"root": "0"}, "unit-0": {"root": "1"}, "x": 1}}) == {
        "units/*" + RK.NO_ID: 2}
    assert _undeclared({**GOOD, "rounds": {"0": {"root": "0"}}}) == {"rounds/0": 1}
    assert _undeclared({**GOOD, "units": [{"root": "0"}]}) == {"units/*": 1}
    assert _undeclared({**GOOD, "schema": ["s"]}) == {"schema/*": 1}
    assert _undeclared({**GOOD, "schema": {"v": 1}}) == {"schema/v": 1}
    assert _undeclared({**GOOD, "verdict": "accepted"}) == {"verdict": 1}
    assert _undeclared({**GOOD, "rounds": [{"root": "0", "sample": 7}]}) == {"rounds/*/sample": 1}
    assert _undeclared([GOOD]) == {"*": 1} and _undeclared(3) == {"(the document)": 1}
    listing = RK.Listing(LIST, "list 0")
    assert listing.undeclared(b'{"schema": ') == {RK.NOT_JSON: 1} == listing.undeclared(b"\xff\xfe")
    lines = RK.Listing({**LIST, "form": "jsonl"}, "list 0")
    two = (json.dumps(GOOD) + "\n\n" + json.dumps({**GOOD, "note": 1}) + "\n").encode()
    assert lines.undeclared(two) == {"note": 1} and listing.undeclared(two) == {RK.NOT_JSON: 1}
    for key in ("note", "units/*" + RK.NO_ID, RK.NOT_JSON):
        assert LIST["writer"] in listing.fix(key)


@pytest.mark.parametrize("change, why", [
    ({"keys": {"schema": "public"}}, "none of structure, output, challenge, commitment"),
    ({"keys": {"schema": None}}, "none of structure"),
    ({"keys": {}}, "each with its kind"),
    ({"keys": {"a": "structure", "a/b": "output"}}, "under a key declared as a value"),
    ({"keys": {"a/b": "output", "a": "structure"}}, "declared as a value and holds deeper keys"),
    ({"keys": {"a//b": "output"}}, "no key path"),
    ({"keys": {"rows/*/n": "structure", "rows/total": "structure"}, "maps": {}}, "named keys and a list's index"),
    ({"maps": {"rounds": "[0-9]+"}}, "no path ending in"),
    ({"maps": {"steps/*": "[0-9]+"}}, "no path ending in"),
    ({"maps": {"units/*": "u[0-9"}}, "a regular expression"),
    ({"form": "yaml"}, "its form is one of"),
    ({"files": ["step/prove.log"]}, "only those have keys"),
    ({"files": ["step/**.json"]}, "is a whole segment"),
    ({"files": []}, "record path globs"),
    ({"writer": ""}, "its writer"),
    ({"why": "its value is public whatever it holds"}, "at most maps beside them"),
    ({"excuse": ["note"]}, "at most maps beside them"),
])
def test_a_malformed_list_is_refused(change, why):
    """A list is refused for a key of no kind, a key path the others contradict, names beside a list's index, a map
    that is no wildcard on its keys' way or has no form for its ids, a form or a file that has no keys to read, and
    any field beside its own: no entry excuses a key."""
    with pytest.raises(RK.Malformed, match=why):
        RK.Declared(_declared({**LIST, **change}))


def test_malformed_declared_data_is_refused(tmp_path):
    """The declared data is refused when it is not of its schema, holds anything beside its lists and what waits, or
    names one list or one glob twice; and a record path two lists rule is refused when it is met."""
    for doc, why in (({**_declared(LIST), "schema": "v0"}, "is not"), ({**_declared(LIST), "allow": []}, "is not"),
                     (_declared(LIST, LIST), "the name"), (_declared(LIST, {**LIST, "name": "another"}), "the glob"),
                     ({**_declared(LIST), "not_yet": "scan/**"}, "not_yet is a list")):
        with pytest.raises(RK.Malformed, match=why):
            RK.Declared(doc)
    two = RK.Declared(_declared(LIST, {**LIST, "name": "another", "files": ["*/summary.json"]}))
    assert two.listing("step/other.json").name == LIST["name"] and two.listing("scan/summary.json").name == "another"
    with pytest.raises(RK.Malformed, match="under two lists"):
        two.listing("step/summary.json")
    (tmp_path / "bad.json").write_text("{")
    with pytest.raises(RK.Malformed):
        RK.load(tmp_path / "bad.json")
    with pytest.raises(RK.Malformed):
        RK.load(tmp_path / "absent.json")


def test_a_glob_stays_inside_its_segments():
    """A list's ``*`` and ``?`` match inside one path segment and ``**`` any number of them, so a list rules the files
    it names and none deeper."""
    for pattern, yes, no in (("gates/*/*/out/a.jsonl", ["gates/c/r/out/a.jsonl"], ["gates/c/out/a.jsonl",
                                                                                   "gates/c/r/s/out/a.jsonl"]),
                             ("n2/**", ["n2/a.json", "n2/v/w/a.json"], ["n2", "n20/a.json"]),
                             ("scan/**/w???-outcome.json", ["scan/w000-outcome.json", "scan/v/w001-outcome.json"],
                              ["scan/v/w01-outcome.json", "scan/v/w000-outcome.json.gz"]),
                             ("job.json", ["job.json"], ["n2/job.json", "job_json"])):
        g = RK.glob(pattern)
        assert all(g.fullmatch(x) for x in yes) and not any(g.fullmatch(x) for x in no), pattern


def _put(root: Path, files: dict[str, bytes | dict]) -> None:
    for name, b in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_bytes(b if isinstance(b, bytes) else json.dumps(b).encode())


def test_a_pass_counts_what_it_checked_what_waits_and_what_has_no_keys(tmp_path, capsys):
    """Over a record's out/, a JSON file under a list is checked, one whose list isn't written yet is counted as not
    yet checked under its step, a file with no keys under its extension, and a JSON file under no list and no
    ``not_yet`` glob fails as a key does, with what to change."""
    (tmp_path / "declared.json").write_text(json.dumps(_declared(LIST, not_yet=("scan/**", "job.json"))))
    out = tmp_path / "out"
    _put(out, {"step/a.json": GOOD, "step/b.json": GOOD, "scan/summary.json": {"anything": 1}, "job.json": {"x": 1},
               "scan/v/w000-outcome.json": {"y": 2}, "scan/prove.log": b"free text", "scan/pub-0.bin": b"\0\1",
               "n2/circuit.rep0.bin": b"\2", "host": b"text"})
    argv = [str(out), "--declared", str(tmp_path / "declared.json")]
    assert RK.main(argv) == 0
    assert capsys.readouterr().out == (
        "released keys: 2 files checked by key, 0 with a key not declared public; 3 not yet checked ((top level) 1, "
        "scan 2); 4 with no keys ((none) 1, .bin 2, .log 1)\n")
    _put(out, {"n2/outcome.json": {"verdict": "accept"}, "step/b.json": {**GOOD, "note": [1, 2]}})
    assert RK.main(argv) == 3
    said = capsys.readouterr().out
    assert "2 files checked by key, 2 with a key not declared public" in said
    assert "  n2/outcome.json: no list of released_keys.json names this file" in said
    assert "  step/b.json: key note is not declared public: give it its kind" in said
    assert RK.main([str(out / "absent"), "--declared", str(tmp_path / "declared.json")]) == 2
    (tmp_path / "declared.json").write_text(json.dumps(_declared({**LIST, "keys": {"schema": "secret"}})))
    assert RK.main(argv) == 2 and "refused" in capsys.readouterr().err
