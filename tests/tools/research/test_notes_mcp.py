"""The notes index (front matter, stamps, citations, catch-up), its joins to the evidence catalog or a compact copy of it, the
read-only SQL, and `research notes mcp` speaking MCP on stdio."""
from __future__ import annotations

import io
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from research import notes
from research import notes_index as NI
from research.notes_mcp import PROTOCOL_VERSIONS, Server, serve
from research.store.cli import main as data_main
from research.store.index import SCHEMA_VERSION
from research.store.local import LocalStore

from test_store import _attempt

RUN, WAIVED, LOST = "r20260928-060054-e5c1", "r20260928-042319-9aac", "r20260928-000000-0000"
HANDOFF, ANSWER = "lanes/x/20260928T0600Z-handoff-from-y.md", "lanes/x/2026-09/20260920T0100Z-answer-z.md"
NOTE = "note:x/20260928T0600Z-handoff-from-y"
TS = "2026-09-28T06:10:00Z"


def _world(tmp_path: Path) -> tuple[Path, Path, str]:
    """A notes root, and the compact copy of a store's catalog: an artifact made by RUN, labels on RUN, on WAIVED (its
    custody waived, no attempt) and on the artifact (`abandoned`, ref NOTE).  (root, compact catalog, the artifact's id)."""
    store = LocalStore()
    art = store.put_artifact("bench-result/v1", {"n": 1}, {})
    store.put_attempt(_attempt(run_id=RUN, outputs={"result": art}))
    store.add_label(RUN, "finding", "the capture matches", by="x", ts=TS)
    store.add_label(WAIVED, "custody", "waived", by="coordinator", ts=TS)
    store.add_label(art, "abandoned", "the path fails on sm_90", by="x", ref=NOTE, ts=TS)
    root = tmp_path / "notes"
    files = {
        HANDOFF: f"---\nlane: x\nkind: handoff\nfrom: y\nto: x\ncreated: 2026-09-28T06:00Z\n---\n# Probe results for sm_90\n"
                 f"The capture {art[:12]} came from {RUN}; {WAIVED} passed but its record was lost.\n",
        ANSWER: f"---\nkind: answer\n---\nArchived answer about wgmma, citing {LOST}.\n",
        "kb/doc.md": "---\nid: kb-doc\ncreated: 2026-09-27 23:30 -07:00\n---\nNo heading here, so the first line is the title\n",
        "lanes/x/.draft.md": "a dotfile", ".hidden/skip.md": "a dot-directory", "lanes/x/notes.txt": "not markdown",
    }
    for rel, text in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)
    compact = tmp_path / "compact.sqlite"
    assert data_main(["catalog", "--compact", str(compact)]) == 0
    return root, compact, art


def _index(tmp_path: Path, catalog: Path | None) -> NI.NotesIndex:
    idx = NI.NotesIndex(tmp_path / "notes", tmp_path / "index.sqlite", catalog)
    idx.refresh()
    return idx


def test_compact_catalog_keeps_the_query_tables_without_manifests_or_custody(tmp_path, capsys):
    _world(tmp_path)
    capsys.readouterr()
    full = LocalStore().index.path
    sqlite3.connect(full).execute("INSERT INTO replicas VALUES ('art:x', 'remote', 'now', '{}')").connection.commit()
    out = tmp_path / "again.sqlite"
    assert data_main(["catalog", "--compact", str(out)]) == 0
    assert capsys.readouterr().out.startswith(f"{out}: ")
    src, dst = sqlite3.connect(full), sqlite3.connect(out)
    tables = {t for (t,) in src.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {t for (t,) in dst.execute("SELECT name FROM sqlite_master WHERE type='table'")} == tables - {"replicas", "replica_checks"}
    for t in tables - {"replicas", "replica_checks"}:
        assert dst.execute(f"SELECT count(*) FROM {t}").fetchone() == src.execute(f"SELECT count(*) FROM {t}").fetchone()
    assert dst.execute("SELECT count(*) FROM attempts WHERE manifest_json IS NOT NULL").fetchone() == (0,)
    assert dst.execute("SELECT count(*) FROM artifacts WHERE manifest_json IS NOT NULL").fetchone() == (0,)
    assert src.execute("SELECT count(*) FROM artifacts WHERE manifest_json IS NOT NULL").fetchone() == (1,)
    assert dst.execute("PRAGMA user_version").fetchone() == (SCHEMA_VERSION,)
    assert {i for (i,) in dst.execute("SELECT name FROM sqlite_master WHERE type='index' AND name='labels_target'")} == {"labels_target"}
    assert sorted(p.name for p in tmp_path.iterdir() if p.name.startswith(".")) == []


def test_the_index_reads_front_matter_stamps_and_citations_and_catches_up(tmp_path):
    root, catalog, art = _world(tmp_path)
    idx = NI.NotesIndex(root, tmp_path / "index.sqlite", catalog)
    assert idx.refresh() == {"notes": 3, "read": 3, "dropped": 0}
    rows = idx.db.execute("SELECT path, lane, id, kind, time, sender, recipient, title FROM notes ORDER BY path").fetchall()
    assert rows == [
        ("kb/doc.md", None, "kb-doc", None, "2026-09-28T06:30Z", None, None, "No heading here, so the first line is the title"),
        (ANSWER, "x", "20260920T0100Z-answer-z", "answer", "2026-09-20T01:00Z", None, None, f"Archived answer about wgmma, citing {LOST}."),
        (HANDOFF, "x", "20260928T0600Z-handoff-from-y", "handoff", "2026-09-28T06:00Z", "y", "x", "Probe results for sm_90")]
    assert idx.db.execute("SELECT path, target, kind FROM cites ORDER BY path, target").fetchall() == [
        (ANSWER, LOST, "run"), (HANDOFF, art[:12], "art"), (HANDOFF, WAIVED, "run"), (HANDOFF, RUN, "run")]
    assert idx.refresh() == {"notes": 3, "read": 0, "dropped": 0}
    with (root / HANDOFF).open("a") as f:
        f.write("A new word: zyzzyva.\n")
    (root / "kb/doc.md").unlink()
    (root / "lanes/y").mkdir()
    (root / "lanes/y/20260928T0700Z-reply-to-x.md").write_text("zyzzyva too\n")
    assert idx.refresh() == {"notes": 3, "read": 2, "dropped": 1}
    assert {line.split("  ")[1] for line in idx.search("zyzzyva").splitlines() if not line.startswith(" ")} == {
        "lanes/y/20260928T0700Z-reply-to-x.md", HANDOFF}
    assert idx.search("heading") == "no notes match"
    assert idx.db.execute("SELECT count(*) FROM notes_fts").fetchone() == (3,)


def test_search_takes_plain_text_filters_and_lists_the_newest(tmp_path):
    root, catalog, _ = _world(tmp_path)
    idx = _index(tmp_path, catalog)
    hit = idx.search(WAIVED)
    assert hit.splitlines()[0] == f"2026-09-28T06:00Z  {HANDOFF}  [handoff]  Probe results for sm_90  cites 3, 3 in the catalog"
    assert hit.splitlines()[1].startswith("    ") and "[r20260928" in hit.splitlines()[1]
    assert idx.search("sm_90").splitlines()[0].split("  ")[1] == HANDOFF
    assert [ln.split("  ")[1] for ln in idx.search(lane="x").splitlines()] == [HANDOFF, ANSWER]
    assert idx.search(kind="answer") == f"2026-09-20T01:00Z  {ANSWER}  [answer]  Archived answer about wgmma, citing {LOST}.  cites 1, 0 in the catalog"
    assert [ln.split("  ")[1] for ln in idx.search(since="2026-09-28").splitlines()] == ["kb/doc.md", HANDOFF]
    assert [ln.split("  ")[1] for ln in idx.search(limit=1).splitlines()] == ["kb/doc.md"]
    assert idx.search("wgmma", lane="y") == "no notes match"


def test_evidence_joins_notes_to_the_catalog_and_says_when_there_is_none(tmp_path, monkeypatch):
    root, catalog, art = _world(tmp_path)
    idx = _index(tmp_path, catalog)
    run = idx.evidence(RUN).splitlines()
    assert run[0].startswith(f"{RUN}: bench_vu exited rc=0 validation=passed start=2026-09-22T10:59:05Z wall_s=2 host=vy-g5 ")
    assert run[1:] == [f"  output result: {art[:12]} (bench-result/v1)", "labels (1):", f"  finding = the capture matches  (x, {TS})",
                       "notes citing it (1):", f"  2026-09-28T06:00Z  {HANDOFF}  Probe results for sm_90"]
    assert idx.evidence(WAIVED).splitlines()[:3] == [f"{WAIVED}: no attempt in the catalog", "labels (1):", f"  custody = waived  (coordinator, {TS})"]
    assert idx.evidence(LOST).splitlines() == [f"{LOST}: no attempt in the catalog", "labels: none", "notes citing it (1):",
                                               f"  2026-09-20T01:00Z  {ANSWER}  Archived answer about wgmma, citing {LOST}."]
    a = idx.evidence(art[:12]).splitlines()
    assert a[0].startswith(f"{art}: bench-result/v1, ") and a[1:4] == [f"  made by {RUN}", "labels (1):",
                                                                       f"  abandoned = the path fails on sm_90  (x, {TS}, ref {NOTE})"]
    assert a[-2:] == ["notes citing it (1):", f"  2026-09-28T06:00Z  {HANDOFF}  Probe results for sm_90"]
    assert idx.evidence(NOTE).splitlines() == [f"{NOTE}: {HANDOFF}", "labels whose ref names 20260928T0600Z-handoff-from-y (1):",
                                               f"  {art}  abandoned = the path fails on sm_90  (x, {TS})"]
    assert idx.evidence("art:00000000").splitlines()[:2] == ["art:00000000: not in the catalog", "notes citing it (0):"]
    with pytest.raises(ValueError, match="an id is"):
        idx.evidence("r2026")
    text = idx.read(NOTE)
    assert text.startswith(f"# {HANDOFF}\n---\nlane: x\n") and text.splitlines()[-4:] == [
        "## Cited evidence (3)", f"- {art[:12]}: bench-result/v1, {a[0].split(', ')[1]}",
        f"- {WAIVED}: no attempt in the catalog; labels custody=waived",
        f"- {RUN}: bench_vu exited rc=0 2026-09-22T10:59 on vy-g5; labels finding=the capture matches"]
    with pytest.raises(LookupError, match="no note 'missing'"):
        idx.read("missing")
    (root / "kb/other.md").write_text("---\nid: kb-doc\ncreated: 2026-09-01T00:00Z\n---\nthe same id\n")
    idx.refresh()
    assert idx.read("kb-doc").splitlines() == ["2 notes match 'kb-doc'; read one by path:", "kb/doc.md", "kb/other.md"]

    monkeypatch.setenv("RESEARCH_STORE", str(tmp_path / "no-store"))
    bare = NI.NotesIndex(root, tmp_path / "bare.sqlite")
    bare.refresh()
    assert bare.catalog is None and NI.NotesIndex(root, tmp_path / "bare.sqlite", catalog).catalog == catalog
    assert bare.evidence(RUN).splitlines() == [
        "no catalog attached: `research data refresh` builds the local store's, or pass --catalog with a compact copy",
        "notes citing it (1):", f"  2026-09-28T06:00Z  {HANDOFF}  Probe results for sm_90"]
    assert bare.search(WAIVED).splitlines()[0].endswith("Probe results for sm_90  cites 3")
    assert bare.read(NOTE).splitlines()[-1] == f"- {RUN}: no catalog attached"


def test_the_default_catalog_is_the_local_stores(tmp_path):
    root, _, _ = _world(tmp_path)
    assert NI.NotesIndex(root, tmp_path / "index.sqlite").catalog == LocalStore().index.path


def test_a_citation_finds_its_note_in_every_form_notes_cite_by_and_after_a_move(tmp_path):
    """Notes cite each other as note:<name>, note:<lane>/<name>, note:<campaign>/<lane>/<name> (the notes README's form) or by a
    path, which `research notes archive` may since have moved into a month folder.  A handoff sent to several lanes has the same
    name in each: the lane in the citation picks one, and a lane holding neither lists both."""
    root, _, art = _world(tmp_path)
    twin = "lanes/y/20260920T0100Z-answer-z.md"
    (root / twin).parent.mkdir(parents=True, exist_ok=True)
    (root / twin).write_text("---\nkind: answer\n---\nThe same answer, sent to y too.\n")
    LocalStore().add_label(WAIVED, "abandoned", "gave up on the waiver", by="y", ref="note:r20-proof/x/20260928T0600Z-handoff-from-y",
                           ts="2026-09-28T06:20:00Z")
    idx = _index(tmp_path, LocalStore().index.path)
    for ref in ("note:x/20260920T0100Z-answer-z", "note:r20-proof/x/20260920T0100Z-answer-z", "lanes/x/20260920T0100Z-answer-z.md",
                "x/2026-09/20260920T0100Z-answer-z.md", ANSWER):
        assert idx.resolve(ref) == [ANSWER], ref
    assert idx.resolve("note:y/20260920T0100Z-answer-z") == [twin]
    assert idx.resolve("note:20260920T0100Z-answer-z") == idx.resolve("note:z/20260920T0100Z-answer-z") == [ANSWER, twin]
    assert idx.read("note:20260920T0100Z-answer-z").splitlines() == [
        "2 notes match 'note:20260920T0100Z-answer-z'; read one by path:", ANSWER, twin]
    assert idx.resolve("note:kb/doc") == idx.resolve("kb-doc") == ["kb/doc.md"] and idx.resolve("note:x/missing") == []
    assert idx.evidence(NOTE).splitlines() == [
        f"{NOTE}: {HANDOFF}", "labels whose ref names 20260928T0600Z-handoff-from-y (2):",
        f"  {art}  abandoned = the path fails on sm_90  (x, {TS})",
        f"  {WAIVED}  abandoned = gave up on the waiver  (y, 2026-09-28T06:20:00Z, ref note:r20-proof/x/20260928T0600Z-handoff-from-y)"]


def test_sql_reads_both_databases_and_writes_neither(tmp_path, monkeypatch):
    _, catalog, _ = _world(tmp_path)
    idx = _index(tmp_path, catalog)
    assert idx.sql("SELECT n.path, a.tool FROM cites c JOIN notes n ON n.path = c.path JOIN ev.attempts a ON a.id = c.target") == \
        f"path\ttool\n{HANDOFF}\tbench_vu"
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        idx.sql("DELETE FROM notes")
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        idx.sql("DELETE FROM ev.labels")
    with pytest.raises(sqlite3.DatabaseError, match="not authorized"):
        idx.sql(f"ATTACH DATABASE '{tmp_path / 'other.sqlite'}' AS other")
    assert not (tmp_path / "other.sqlite").exists()
    many = idx.sql("WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c WHERE x < 250) SELECT x FROM c").splitlines()
    assert (len(many), many[0], many[-2], many[-1]) == (202, "x", "200", "(the first 200 rows)")
    monkeypatch.setattr(NI, "SQL_BUDGET", 5)
    with pytest.raises(ValueError, match="more than 5,000 SQLite steps"):
        idx.sql("WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) SELECT count(*) FROM c")
    assert idx.db.execute("SELECT count(*) FROM notes").fetchone() == (3,)


def test_a_refresh_leaves_the_catalog_free_for_the_stores_writers(tmp_path, monkeypatch):
    _, catalog, _ = _world(tmp_path)
    idx = _index(tmp_path, catalog)
    walk = NI._files

    def walk_while_the_store_writes(root):
        other = sqlite3.connect(catalog, isolation_level=None)
        other.execute("PRAGMA busy_timeout = 0")
        other.execute("BEGIN EXCLUSIVE")
        other.execute("ROLLBACK")
        other.close()
        return walk(root)

    monkeypatch.setattr(NI, "_files", walk_while_the_store_writes)
    assert idx.refresh()["notes"] == 3


def _rpc(*msgs) -> bytes:
    return b"".join((m if isinstance(m, bytes) else json.dumps(m).encode()) + b"\n" for m in msgs)


def test_the_server_speaks_mcp_over_stdio(tmp_path):
    _, catalog, _ = _world(tmp_path)
    out = io.BytesIO()
    msgs = _rpc({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2024-11-05", "capabilities": {}}},
                {"jsonrpc": "2.0", "method": "notifications/initialized"},
                {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
                {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "search_notes", "arguments": {"query": "wgmma"}}},
                {"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {"name": "evidence", "arguments": {"id": "nonsense"}}},
                {"jsonrpc": "2.0", "id": 5, "method": "tools/call", "params": {"name": "search_notes", "arguments": {"color": "red"}}},
                {"jsonrpc": "2.0", "id": 6, "method": "tools/call", "params": {"name": "rm", "arguments": {}}},
                {"jsonrpc": "2.0", "id": 7, "method": "resources/list"},
                b"{not json",
                [{"jsonrpc": "2.0", "id": 8, "method": "ping"}, {"jsonrpc": "2.0", "method": "notifications/cancelled"}],
                {"jsonrpc": "2.0", "id": 99, "result": {}},
                {"jsonrpc": "2.0", "id": 9, "method": "initialize", "params": {"protocolVersion": "1999-01-01"}})
    assert serve(Server(_index(tmp_path, catalog)), io.BytesIO(msgs), out) == 0
    got = [json.loads(line) for line in out.getvalue().splitlines()]
    assert [g["id"] if isinstance(g, dict) else [x["id"] for x in g] for g in got] == [1, 2, 3, 4, 5, 6, 7, None, [8], 9]
    init = got[0]["result"]
    assert (init["protocolVersion"], init["capabilities"], init["serverInfo"]["name"]) == ("2024-11-05", {"tools": {"listChanged": False}},
                                                                                         "research-notes")
    assert [t["name"] for t in got[1]["result"]["tools"]] == ["search_notes", "read_note", "evidence", "sql"]
    assert all(t["inputSchema"]["type"] == "object" for t in got[1]["result"]["tools"])
    assert got[2]["result"]["isError"] is False and got[2]["result"]["content"][0]["text"].splitlines()[0].split("  ")[1] == ANSWER
    assert got[3]["result"]["isError"] is True and got[3]["result"]["content"][0]["text"].startswith("ValueError: 'nonsense': an id is")
    assert got[4]["result"]["isError"] is True and got[4]["result"]["content"][0]["text"].startswith("TypeError:")
    assert got[5]["error"] == {"code": -32602, "message": "Unknown tool: rm"}
    assert got[6]["error"] == {"code": -32601, "message": "Method not found: resources/list"}
    assert got[7]["error"] == {"code": -32700, "message": "Parse error"}
    assert got[8] == [{"jsonrpc": "2.0", "id": 8, "result": {}}]
    assert got[9]["result"]["protocolVersion"] == PROTOCOL_VERSIONS[0]


def test_research_notes_mcp_and_index_run_from_the_command_line(tmp_path, capsys):
    root, catalog, _ = _world(tmp_path)
    args = ["--root", str(root), "--index", str(tmp_path / "index.sqlite"), "--catalog", str(catalog)]
    p = subprocess.run([sys.executable, "-m", "research", "notes", "mcp", *args], capture_output=True,
                       input=_rpc({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
                                  {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                                   "params": {"name": "evidence", "arguments": {"id": WAIVED}}}))
    assert p.returncode == 0 and p.stderr.decode().startswith(f"research notes mcp: 3 notes under {root}; catalog {catalog}")
    got = [json.loads(line) for line in p.stdout.splitlines()]
    assert got[1]["result"]["content"][0]["text"].splitlines()[0] == f"{WAIVED}: no attempt in the catalog"
    capsys.readouterr()
    assert notes.main(["index", *args]) == 0
    assert capsys.readouterr().out == f"research notes index: 3 notes (0 read, 0 dropped) in {tmp_path / 'index.sqlite'}; catalog {catalog}\n"
    assert notes.main(["index", *args, "--evidence", LOST]) == 0
    assert capsys.readouterr().out.startswith(f"{LOST}: no attempt in the catalog\n")
    assert notes.main(["index", *args, "--evidence", "bogus"]) == 2
    assert notes.main(["mcp", "--root", str(tmp_path / "none")]) == 2
