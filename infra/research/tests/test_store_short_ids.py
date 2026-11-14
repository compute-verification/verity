"""Abbreviated artifact ids on the `research data` command line (`art:cc59294a`, the form notes and reports print): resolved against
the local catalog wherever an id is taken, never inside free text, and refused by name when missing or ambiguous."""
from __future__ import annotations

import json

import pytest

from research.store.cli import expand_short_ids, main
from research.store.index import Index
from research.store.local import LocalStore


@pytest.fixture
def st(tmp_path, monkeypatch) -> LocalStore:
    monkeypatch.setenv("RESEARCH_STORE", str(tmp_path / "store"))
    return LocalStore()


def _put(st: LocalStore, n: int, **refs) -> str:
    return st.put_artifact("proof/v1", {"n": n}, refs)


def test_short_ids_resolve_in_positionals_refs_and_member_lists(st, capsys):
    a, b = _put(st, 1), _put(st, 2)
    sa, sb = a[:12], b[:14]

    assert main(["show", sa]) == 0
    assert a in capsys.readouterr().out

    assert main(["label", sa, "note", "hello", "--by", "tester"]) == 0
    assert [lab.value for lab in st.labels(a)] == ["hello"]

    assert main(["put", "--kind", "proof/v1", "--meta", '{"n": 3}', "--ref", f"inputs={sa},{sb}", "--ref", f"first={sa}"]) == 0
    c = capsys.readouterr().out.strip().splitlines()[-1]
    assert st.get_manifest(c).refs == {"inputs": [a, b], "first": a}

    assert main(["snapshot", "--name", "s", "--members", f"{sa},{sb}", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["members"] == sorted([a, b])

    assert main(["select", "--where", f"meta.n=1", "--ids"]) == 0      # an unrelated `=` token passes through untouched
    assert capsys.readouterr().out.split() == [a]


def test_free_text_json_and_sql_are_never_rewritten(st):
    a = _put(st, 1)
    short = a[:12]
    meta = json.dumps({"see": short})
    assert expand_short_ids("put", ["--kind", "k", "--meta", meta, "--note", short, f"--meta={meta}"]) == \
        ["--kind", "k", "--meta", meta, "--note", short, f"--meta={meta}"]
    assert expand_short_ids("sql", [f"SELECT '{short}'", short]) == [f"SELECT '{short}'", short]
    assert expand_short_ids("show", [short]) == [a]
    assert expand_short_ids("show", [a[:11]]) == [a[:11]]             # 7 hex digits: not an abbreviation, passed through as typed


def test_seven_hex_digits_is_not_an_abbreviation(st, capsys):
    a = _put(st, 1)
    assert main(["show", a[:11]]) == 2
    assert ">= 8 hex digits" in capsys.readouterr().err


def test_unknown_and_ambiguous_prefixes_are_refused_by_name(st, monkeypatch, capsys):
    _put(st, 1)
    assert main(["show", "art:00000000"]) == 1
    err = capsys.readouterr().err
    assert "art:00000000 matches no artifact" in err and "research data refresh" in err

    monkeypatch.setattr(Index, "resolve_art_prefix", lambda self, prefix, limit=2: [prefix + "0" * 8, prefix + "1" * 8][:limit])
    assert main(["where", "art:abcdef12"]) == 1
    assert "art:abcdef12 is ambiguous" in capsys.readouterr().err


def test_prefix_scan_is_exact_at_the_boundaries(st):
    ids_ = sorted(_put(st, n) for n in range(40))
    for full in ids_:
        assert st.index.resolve_art_prefix(full[:12], limit=50) == [full]
    assert st.index.resolve_art_prefix("art:", limit=100) == ids_
