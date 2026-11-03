"""The exporter's own logic on toy inputs (the board's parsing and ids, how an entry's ids resolve, the bases, the method
keys, the schema's refusals, the publish command) and the committed curation against the schema it is checked by."""
from __future__ import annotations

import hashlib
import json
import re
import sys
from datetime import UTC, datetime, timedelta

import jsonschema
import pytest

from dataset import export as X

ART = "9d9502d94427b9179955e85fc7f850e0940cfdf6282081a0745d78250bf9f6c9"
TWIN = "9d9502ff00000000000000000000000000000000000000000000000000000000"
D6B1, R217F, R3A1F = "r20261102-133933-d6b1", "r20261102-133520-217f", "r20261102-120000-3a1f"

BOARD = """# Board

The preamble.

## Entries

### 2026-11-02

13:39Z hybrid [FINDING] r20261102-133933-d6b1@200 ends at 1.2 (art:9d9502d94427)
13:39Z hybrid [FINDING] r20261102-133933-d6b1@200 ends at 1.2 (art:9d9502d94427)
14:05Z scorer [DIGEST] hybrid-000146..148 and -133520-217f; see d6b1
  a continuation line
23:59Z scorer untagged, about 9d9502d944

### 2026-11-03

00:10Z hybrid [NEGATIVE] r20261103-000000-ffff is unknown, art:9d9502 is ambiguous, art:0123456789ab unknown; see 3a1f

## Notes

09:00Z nobody [FINDING] after the entries, not one of them
"""


def index() -> X.Index:
    ledger = {f"hybrid-00014{i}": {"run_id": R3A1F, "checkpoint_step": 100 + 10 * i} for i in (6, 7, 8)}
    ledger["hybrid-000150"] = {"run_id": D6B1, "checkpoint_step": 200}
    return X.Index({D6B1: {"hybrid"}, R217F: {"hybrid"}, R3A1F: {"hybrid"}}, {ART: R217F, TWIN: None}, ledger)


def test_board_entries_are_dated_collapsed_and_continued():
    rows = X.board_entries(BOARD)
    assert [(r["at"], r["agent"], r["tag"], r["copies"]) for r in rows] == [
        (datetime(2026, 11, 2, 13, 39, tzinfo=UTC), "hybrid", "FINDING", 2),
        (datetime(2026, 11, 2, 14, 5, tzinfo=UTC), "scorer", "DIGEST", 1),
        (datetime(2026, 11, 2, 23, 59, tzinfo=UTC), "scorer", None, 1),
        (datetime(2026, 11, 3, 0, 10, tzinfo=UTC), "hybrid", "NEGATIVE", 1),
    ]
    digest = rows[1]
    assert digest["text"] == "hybrid-000146..148 and -133520-217f; see d6b1\n  a continuation line"
    assert digest["id"] == "20261102T1405Z-scorer-" + hashlib.sha256(f"DIGEST {digest['text']}".encode()).hexdigest()[:6]
    assert len({r["id"] for r in rows}) == len(rows)


@pytest.mark.parametrize("text", [
    "# Board\n\n13:39Z hybrid [FINDING] no entries section\n",
    "## Entries\n\n13:39Z hybrid [FINDING] no day heading\n",
    "## Entries\n\n### 2026-11-02\n\n## Notes\n",
])
def test_a_board_it_cannot_date_is_refused_with_the_fix(text):
    with pytest.raises(SystemExit, match=re.escape(X.FIX_BOARD)):
        X.board_entries(text)


def test_links_resolve_each_kind_of_id():
    ix = index()
    finding, digest, untagged, negative = (X.links(e, ix) for e in X.board_entries(BOARD))
    assert finding == {"runs": [R217F, D6B1], "checkpoints": ["hybrid-000150"], "arts": [f"art:{ART}"],
                       "unresolved": []}
    # a ledger range and a short run id resolve; a bare suffix names only its author's own run
    assert digest == {"runs": [R3A1F, R217F], "checkpoints": ["hybrid-000146", "hybrid-000147", "hybrid-000148"],
                      "arts": [], "unresolved": []}
    assert untagged == {"runs": [R217F], "checkpoints": [], "arts": [f"art:{ART}"], "unresolved": []}
    assert negative == {"runs": [R3A1F], "checkpoints": [], "arts": [],
                        "unresolved": ["art:0123456789ab", "art:9d9502", "r20261103-000000-ffff"]}


def test_bases_and_the_worst_of_several():
    assert X.basis_of(None, None) == X.basis_of("measured: x", None) == "unmeasured"
    assert X.basis_of("measured: the harness's count", 1.0) == X.basis_of("counted: ops", 1.0) == "measured"
    assert X.basis_of(next(iter(X.MEASURED)), 1.0) == "measured"
    assert X.basis_of("upper bound (witness): 8 |E|", 8.0) == "upper-bound"
    assert X.basis_of("stub: one token id", 16) == X.basis_of(None, 3) == "estimated"
    assert X.worst("measured", "upper-bound") == "upper-bound"
    assert X.worst("measured", "estimated", "upper-bound") == "estimated"
    assert X.worst("estimated", "unmeasured") == "unmeasured"


def test_method_keys_take_the_launchers_tags():
    assert X.tag("QLoRA+Gains") == "qloragains"
    assert X.tag(None) is None
    assert X.tag("x" * 70) == "x" * 64
    assert X.method_key("Sparse", "qlora+gains") == "sparse/qloragains"


def test_non_finite_values_are_written_null_and_counted():
    count = [0]
    assert X.finite({"a": (1.0, float("inf")), "b": float("nan"), "c": "x", "d": 2}, count) == {
        "a": [1.0, None], "b": None, "c": "x", "d": 2}
    assert count == [2]


def test_the_second_night_starts_at_its_boundary():
    assert X.night_of(None) is None
    assert X.night_of(X.NIGHT2) == 2
    assert X.night_of(X.NIGHT2 - timedelta(seconds=1)) == 1


def test_validate_names_the_table_row_and_field():
    schema = json.loads(X.SCHEMA.read_text())
    cur = json.loads(X.CURATION.read_text())
    row = {k: cur["lessons"][0].get(k) for k in ("id", "kind", "lesson", "fix", "status", "rank", "evidence")}
    X.validate({"lessons": [row]}, schema)
    with pytest.raises(SystemExit) as e:
        X.validate({"lessons": [row | {"kind": "anecdote"}]}, schema)
    assert f"lessons row 0 ({row['id']}) at `kind`" in str(e.value)


def test_the_schema_has_one_definition_per_table():
    schema = json.loads(X.SCHEMA.read_text())
    assert schema["required"] == ["tables", "manifest"]
    assert schema["properties"]["tables"]["required"] == list(X.TABLES)
    assert all(t in schema["$defs"] for t in X.TABLES)


#: what curation evidence may cite: a board line, a run, a ledger id or an art prefix of 12 hex digits or more
EVIDENCE = re.compile(r"\d{8}T\d{4}Z-\S+-[0-9a-f]{6}|r\d{8}-\d{6}-[0-9a-f]{4}|[a-z][a-z0-9_-]*-\d{6}|art:[0-9a-f]{12,64}")


def test_the_curation_is_well_formed():
    defs = json.loads(X.SCHEMA.read_text())["$defs"]
    cur = json.loads(X.CURATION.read_text())
    assert set(cur) == {"methods", "directions", "lessons"}
    idea = jsonschema.Draft202012Validator(defs["idea"] | {"$defs": defs})
    lesson = jsonschema.Draft202012Validator(defs["lessons"] | {"$defs": defs})
    cited = []
    for name, d in cur["directions"].items():
        assert set(d) <= {"what", "ideas", "learned", "open", "evidence"} and d["what"], name
        for i in d["ideas"]:
            idea.validate(i)
            cited += i["evidence"]
        cited += d.get("evidence", [])
    for v in cur["lessons"]:
        lesson.validate(v)
        cited += v["evidence"]
    assert len({v["id"] for v in cur["lessons"]}) == len(cur["lessons"])
    assert sorted(v["rank"] for v in cur["lessons"] if v["rank"] is not None) == [1, 2, 3, 4, 5]
    for key, m in cur["methods"].items():
        assert key == X.method_key(*key.split("/", 1)), key
        assert set(m) <= {"label", "description", "learned", "open"} and m.get("label", "x"), key
        assert len(m.get("description") or "") <= 200, key
    assert not [e for e in cited if not EVIDENCE.fullmatch(e)]


def test_publish_puts_the_tree_and_returns_its_art_id(tmp_path):
    argv = tmp_path / "argv.json"
    fake = tmp_path / "research"
    fake.write_text(f"#!{sys.executable}\nimport json, sys\nopen({str(argv)!r}, 'w').write(json.dumps(sys.argv[1:]))\n"
                    "print(json.dumps({'id': 'art:00ff'}))\n")
    fake.chmod(0o755)
    manifest = {"te_core": "4.1", "checkout": "c", "pin": ["p"], "stamp": "s", "counts": {"checkpoints": 1},
                "sources": {"pretraining": {"art": "art:8c8d"}}}
    assert X.publish(tmp_path / "out", str(fake), manifest) == "art:00ff"
    got = json.loads(argv.read_text())
    assert got[:7] == ["data", "put", "--kind", "evidence/v1", "--tree", str(tmp_path / "out"), "--preserve"]
    assert got[-2:] == ["--ref", "pretraining=art:8c8d"]
